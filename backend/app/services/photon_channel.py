"""Photon 2.0 (Beta) iMessage channel.

A background thread polls Photon for inbound texts, feeds each one through the
same `store.record_reply` path the "Simulate family reply" box uses, and sends a
short acknowledgement. It also delivers outbound messages for the store
(`send_text`), into conversations a family has already opened.
API contract: https://photon.codes/docs/beta/get-started/imessage

Guardrails (PRD 3 section 3):
- Inbound text is data only. It goes to the classifier and is never executed.
- Safety-critical replies get no automated answer; the counselor takes over.
- STOP is honored immediately: the sender is opted out and nothing is sent.
- Every outbound message passes `outreach.blocked_reason` first (household
  safety suppression, opt-out, no links).
- A shared line cannot send first: a family is reachable only after it has
  texted the agent line. Until then outbound messages stay queued.
- Every inbound and outbound message is appended to `audit`.

The channel is off unless the PHOTON_* variables are set, so tests never touch it.
"""
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from ..models import AbsenceStatus, FamilyReply, Student
from ..store import store
from . import outreach

log = logging.getLogger("uvicorn.error")

REQUIRED = ("PHOTON_API_KEY", "PHOTON_PROJECT_ID", "PHOTON_AGENT_HANDLE", "PHOTON_TEST_SENDER")
POLL_SECONDS = 2.0
CONTACT_LOOKBACK_DAYS = 3  # on startup, relearn which families have already texted the agent line
STOP_WORDS = {"stop", "stopall", "unsubscribe", "cancel", "end", "quit", "alto", "parar"}

ACK = {
    "en": "Thank you for letting us know. The school office has your message and will follow up.",
    "es": "Gracias por avisarnos. La oficina de la escuela recibió su mensaje y le dará seguimiento.",
}
NOT_LINKED = "This number is not linked to a student record. Please call the school office."

audit: List[dict] = []
opted_out = outreach.opted_out
contacts: Dict[str, dict] = {}  # phone digits -> {"space_id", "agent_id"} of the conversation that family opened
_seen: set = set()
_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_client = None  # the live Photon client, owned by the polling thread
_project_id = ""

_digits = outreach.digits


def _links() -> Dict[str, str]:
    """Phone (last 10 digits) -> student_id. The test sender plays the demo student's guardian."""
    links = {_digits(s.guardian_phone): s.id for s in store.students.values()}
    sender = os.environ.get("PHOTON_TEST_SENDER")
    if sender:
        links[_digits(sender)] = outreach.demo_student_id()
    return links


def _record(direction: str, handle: str, text: str, **extra) -> None:
    audit.append({"direction": direction, "channel": "imessage", "handle": handle, "text": text,
                  "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra})


def handle_inbound(sender: str, text: str, received_at: str, message_id: str = "") -> dict:
    """Process one inbound text. Returns {"route", "absence_id", "reply"}; reply None means send nothing."""
    result = {"route": "ignored", "absence_id": None, "reply": None}
    key = _digits(sender)

    if text.strip().lower() in STOP_WORDS:
        opted_out.add(key)
        result["route"] = "opted_out"
    elif key in opted_out:
        result["route"] = "opted_out"
    else:
        student = store.students.get(_links().get(key, ""))
        if student is None:
            result.update(route="unlinked", reply=NOT_LINKED)
        else:
            mine = sorted((a for a in store.absences.values() if a.student_id == student.id), key=lambda a: a.date)
            waiting = [a for a in mine if a.status == AbsenceStatus.AWAITING_REPLY]
            target = (waiting or mine or [None])[-1]
            if target is None:
                result.update(route="no_absence", reply=ACK.get(student.preferred_language, ACK["en"]))
            else:
                route = store.record_reply(target.id, FamilyReply(
                    text=text, channel="text", language=student.preferred_language, received_at=received_at))
                # The acknowledgement obeys the same rules as any outbound message. In particular,
                # an open safety escalation in the household means the counselor answers, never the bot.
                reply = ACK.get(student.preferred_language, ACK["en"])
                if outreach.blocked_reason(sender, reply, store.household_suppressed(student.id)):
                    reply = None
                result.update(route=route, absence_id=target.id, reply=reply)

    _record("inbound", sender, text, message_id=message_id, route=result["route"], absence_id=result["absence_id"])
    return result


# ---------- Photon polling loop ----------
def send_text(student: Student, phone: str, text: str, key: str) -> dict:
    """Deliver one outbound message for the store. The store has already applied the outreach rules."""
    if _client is None or not (_thread and _thread.is_alive()):
        return {"status": "queued", "reason": "Photon channel is off", "channel": None}
    contact = contacts.get(_digits(phone))
    if contact is None:
        return {"status": "queued", "channel": None,
                "reason": "family has not texted the agent line yet (a shared line cannot send first)"}

    from photon_api.generated.models import SendMessageRequest, SendTextContent
    from photon_api.rpc_generated import SendMessageHeader, SendMessageInput, SendMessagePath
    try:
        _client.projects.send_message(SendMessageInput(
            path=SendMessagePath(id=_project_id),
            headers=SendMessageHeader.model_validate({"idempotency-key": key}),
            body=SendMessageRequest(
                spaceId=contact["space_id"], senderId=contact["agent_id"],
                content=SendTextContent(type="text", format="plain", text=text)),
        ))
    except Exception as error:
        detail = f"{type(error).__name__} {getattr(error, 'status', '')}".strip()
        log.warning("photon send failed: %s", detail)
        return {"status": "failed", "reason": f"Photon rejected the send ({detail})", "channel": "imessage"}
    _record("outbound", phone, text, key=key)
    # Accepted for asynchronous dispatch; Photon does not confirm delivery here.
    return {"status": "sent", "reason": None, "channel": "imessage"}


def _poll(photon, cfg: dict, created_since: str, handle: bool = True) -> str:
    """Read new inbound messages. With handle=False, only relearn contacts (startup catch-up)."""
    from photon_api.generated.models import SendMessageRequest, SendTextContent, TextContent, UserSnapshot
    from photon_api.rpc_generated import ListMessagesInput, SendMessageHeader, SendMessageInput, SendMessagePath

    query = {"createdSince": created_since, "direction": "received", "order": "asc", "pageSize": 100,
             "platform": "imessage"}
    newest = created_since
    while True:
        page = photon.projects.list_messages(
            ListMessagesInput.model_validate({"path": {"id": cfg["PHOTON_PROJECT_ID"]}, "query": query}))
        for item in page.items:
            m = item.root.root
            if m.id in _seen:
                continue
            _seen.add(m.id)
            newest = max(newest, m.createdAt)
            if m.platform != "imessage" or m.status != "received" or not isinstance(m.receivedBy, UserSnapshot):
                continue
            sender, agent, space = m.sentBy.root, m.receivedBy.root, m.space.root
            if sender.kind != "human" or agent.kind != "agent" or sender.id == agent.id or space.kind != "direct":
                continue
            if agent.handle != cfg["PHOTON_AGENT_HANDLE"]:
                continue
            contacts[_digits(sender.handle)] = {"space_id": space.id, "agent_id": agent.id}
            if not handle:
                continue
            if not (isinstance(m.content, TextContent) and m.content.format == "plain"):
                continue  # Beta iMessage carries text and images only; non-text is skipped for now

            result = handle_inbound(sender.handle, m.content.text.strip(), m.createdAt, m.id)
            log.info("photon inbound %s -> %s (%s)", m.id, result["route"], result["absence_id"])
            if result["reply"]:
                photon.projects.send_message(SendMessageInput(
                    path=SendMessagePath(id=cfg["PHOTON_PROJECT_ID"]),
                    headers=SendMessageHeader.model_validate({"idempotency-key": f"rollcall-ack:{m.id}"}),
                    body=SendMessageRequest(
                        spaceId=space.id, senderId=agent.id, parentMessageId=m.id,
                        content=SendTextContent(type="text", format="plain", text=result["reply"])),
                ))
                _record("outbound", sender.handle, result["reply"], in_reply_to=m.id)
        token = page.nextPageToken
        if not isinstance(token, str):
            return newest
        query["pageToken"] = token


def _loop(cfg: dict) -> None:
    global _client, _project_id
    from photon_api import Photon

    def stamp(moment: datetime) -> str:
        return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    now = datetime.now(timezone.utc)
    # Start slightly in the past; _seen prevents double handling within a run.
    since = stamp(now - timedelta(seconds=5))
    with Photon(headers={"Authorization": f"Bearer {cfg['PHOTON_API_KEY']}"}) as photon:
        try:  # older messages are never re-classified, only used to learn who can be messaged
            _poll(photon, cfg, stamp(now - timedelta(days=CONTACT_LOOKBACK_DAYS)), handle=False)
        except Exception as error:
            log.warning("photon contact catch-up failed: %s", type(error).__name__)
        _client, _project_id = photon, cfg["PHOTON_PROJECT_ID"]
        log.info("photon channel listening on %s (%d reachable families)", cfg["PHOTON_AGENT_HANDLE"], len(contacts))
        while not _stop.is_set():
            try:
                since = _poll(photon, cfg, since)
            except Exception as error:  # keep the loop alive through API or network errors
                log.warning("photon poll failed: %s", type(error).__name__)
            _stop.wait(POLL_SECONDS)
    _client = None


def enabled() -> bool:
    return all(os.environ.get(name) for name in REQUIRED)


def start() -> bool:
    global _thread
    if not enabled() or (_thread and _thread.is_alive()):
        return False
    _stop.clear()
    _thread = threading.Thread(target=_loop, args=({n: os.environ[n] for n in REQUIRED},),
                               name="photon-channel", daemon=True)
    _thread.start()
    store.sender = send_text
    return True


def stop() -> None:
    _stop.set()


def status() -> dict:
    return {"enabled": enabled(), "listening": bool(_thread and _thread.is_alive()),
            "agent_handle": os.environ.get("PHOTON_AGENT_HANDLE"),
            "demo_student_id": outreach.demo_student_id(), "reachable_families": len(contacts),
            "opted_out": len(opted_out), "messages": audit[-50:]}
