"""Photon 2.0 (Beta) iMessage channel.

A background thread polls Photon for inbound texts, feeds each one through the
same `store.record_reply` path the "Simulate family reply" box uses, and sends a
short acknowledgement. It also delivers outbound messages for the store
(`send_text`), into conversations a family has already opened.
API contract: https://photon.codes/docs/beta/get-started/imessage

Guardrails (PRD 3 section 3):
- Inbound text is data only. It goes to the classifier and is never executed.
- Safety-critical replies get no automated answer; the counselor takes over.
- Opt-outs are honored immediately, as a keyword (STOP) or in plain language
  ("no más mensajes"): nothing more is sent to that number. Texts from an
  opted-out family are still classified, so a safety disclosure is never dropped.
- Every outbound message passes `outreach.blocked_reason` first (household
  safety suppression, opt-out, no links).
- A shared line cannot send first: a family is reachable only after it has
  texted the agent line. Until then outbound messages stay queued.
- Every inbound and outbound message is written to the store's message log,
  with the time from arrival to label (`latency_ms`).

The channel is off unless the PHOTON_* variables are set, so tests never touch it.
"""
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from ..models import AbsenceStatus, Barrier, FamilyReply, Student
from ..store import elapsed_ms, iso, store, utc_now
from . import classifier, demo, outreach

log = logging.getLogger("uvicorn.error")

REQUIRED = ("PHOTON_API_KEY", "PHOTON_PROJECT_ID", "PHOTON_AGENT_HANDLE", "PHOTON_TEST_SENDER")
POLL_SECONDS = 2.0
CONTACT_LOOKBACK_DAYS = 3  # on startup, relearn which families have already texted the agent line

ACK = {
    "en": "Thank you for letting us know. The school office has your message and will follow up.",
    "es": "Gracias por avisarnos. La oficina de la escuela recibió su mensaje y le dará seguimiento.",
}
NOT_LINKED = "This number is not linked to a student record. Please call the school office."

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


def _parse(received_at: str) -> datetime:
    moment = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def handle_inbound(sender: str, text: str, received_at: str, message_id: str = "",
                   channel: str = "imessage") -> dict:
    """Process one inbound text and log it.

    Returns {"route", "absence_id", "student_id", "reply"}; reply None means send nothing.
    Every text from a known family is classified, including after an opt-out, so a reason or
    a safety disclosure is never dropped. Opting out only stops what we send.
    """
    result = {"route": "ignored", "absence_id": None, "student_id": None, "reply": None}
    staff_reply = store.staff_command(sender, text)
    if staff_reply is not None:  # "ACK" from a staff phone: never classified as a family reply
        store.log_message(direction="inbound", channel=channel, raw_text_or_transcript=text,
                          timestamp=iso(_parse(received_at)), route="staff", phone=sender,
                          external_id=message_id or None)
        return {**result, "route": "staff", "reply": staff_reply}
    key = _digits(sender)
    received = _parse(received_at)
    logged = False
    note = None

    student = store.students.get(_links().get(key, ""))
    if student is not None:
        result["student_id"] = student.id

    if outreach.is_opt_in(text):
        opted_out.discard(key)
        result["route"], note = "opted_in", "family opted back in to messages"
    else:
        if outreach.is_opt_out(text):
            opted_out.add(key)
            note = "family opted out of messages"
        # "STOP" or "no más mensajes" on its own says nothing about an absence, so it is not attached
        # to one (it would overwrite a real reason with "unknown"). A longer message that also gives
        # a reason, e.g. "no more texts, she has a fever", is still classified below.
        only_a_request = note is not None and classifier.classify(
            text, store.names_for(student.id) if student else ()).barrier == Barrier.UNKNOWN
        if only_a_request:
            result["route"] = "opted_out"
        elif student is None:
            result.update(route="unlinked", reply=NOT_LINKED)
        else:
            mine = sorted((a for a in store.absences.values() if a.student_id == student.id), key=lambda a: a.date)
            waiting = [a for a in mine if a.status == AbsenceStatus.AWAITING_REPLY]
            target = (waiting or mine or [None])[-1]
            if target is None:
                result.update(route="no_absence", reply=ACK.get(student.preferred_language, ACK["en"]))
            else:
                route = store.record_reply(
                    target.id,
                    FamilyReply(text=text, channel="text", language=student.preferred_language,
                                received_at=received_at),
                    channel=channel, received=received, external_id=message_id or None, phone=sender,
                    reason=note)
                logged = True
                # Answer in the language the family wrote in when we have a template for it.
                label = store.absences[target.id].classification
                lang = label.language if label and label.language in ACK else student.preferred_language
                # The acknowledgement obeys the same rules as any outbound message. In particular,
                # an open safety escalation in the household means the counselor answers, never the bot.
                reply = ACK.get(lang, ACK["en"])
                if outreach.blocked_reason(sender, reply, store.household_suppressed(student.id)):
                    reply = None
                result.update(route=route, absence_id=target.id, reply=reply)

    if key in opted_out:
        result["reply"] = None  # an opted-out number gets nothing back, not even a confirmation

    if not logged:  # texts that are not attached to an absence are still on the record
        store.log_message(direction="inbound", channel=channel, student_id=result["student_id"],
                          raw_text_or_transcript=text, timestamp=iso(received), route=result["route"],
                          phone=sender, external_id=message_id or None, reason=note)
    return result


# ---------- Photon polling loop ----------
def send_text(student: Student, phone: str, text: str, key: str) -> dict:
    """Deliver one outbound message for the store. The store has already applied the outreach rules."""
    if _client is None or not (_thread and _thread.is_alive()):
        return {"status": "queued", "reason": "Photon channel is off", "channel": None}
    contact = contacts.get(_digits(phone))
    if contact is None:
        return {"status": "queued", "channel": None,
                "reason": "this number has not texted the agent line yet (a shared line cannot send first)"}

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
            # Any agent line in this project counts: on a shared line, Photon assigns each sender
            # their own agent number, so a teammate's or staff member's phone texts a different line
            # than PHOTON_AGENT_HANDLE. Replies and pages go back out on the line that person texted.
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
                student = store.students.get(result["student_id"] or "")
                store.log_message(
                    direction="outbound", kind="ack", status="sent", channel="imessage",
                    absence_id=result["absence_id"], student_id=result["student_id"],
                    language=student.preferred_language if student else "en",
                    raw_text_or_transcript=result["reply"], timestamp=iso(utc_now()),
                    latency_ms=elapsed_ms(_parse(m.createdAt)),  # family's text arrived -> our reply accepted
                    phone=sender.handle)
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
                store.flush_due()  # release messages held overnight for quiet hours
                store.check_paging()  # page the principal about cases unacknowledged for 15 minutes
            except Exception as error:  # keep the loop alive through API or network errors
                log.warning("photon poll failed: %s", type(error).__name__)
            _stop.wait(POLL_SECONDS)
    _client = None


def enabled() -> bool:
    """Photon runs when its settings are present, never in demo mode (the replay channel stands in)."""
    return all(os.environ.get(name) for name in REQUIRED) and not demo.enabled()


def start() -> bool:
    global _thread
    if not enabled() or (_thread and _thread.is_alive()):
        return False
    _stop.clear()
    _thread = threading.Thread(target=_loop, args=({n: os.environ[n] for n in REQUIRED},),
                               name="photon-channel", daemon=True)
    _thread.start()
    store.sender = send_text
    store.staff_sender = lambda phone, text, key: send_text(None, phone, text, key)
    return True


def stop() -> None:
    _stop.set()


def status() -> dict:
    return {"enabled": enabled(), "listening": bool(_thread and _thread.is_alive()),
            "agent_handle": os.environ.get("PHOTON_AGENT_HANDLE"),
            "demo_student_id": outreach.demo_student_id(), "reachable_families": len(contacts),
            "opted_out": len(opted_out), "latency": store.latency_summary("imessage"),
            "messages": [m for m in store.messages if m.channel == "imessage"][-50:]}
