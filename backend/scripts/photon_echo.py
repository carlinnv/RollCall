"""Photon 2.0 (Beta) round-trip smoke test.

Waits up to two minutes for one plain-text iMessage from PHOTON_TEST_SENDER to the
assigned agent line, echoes it back once, and exits. Adapted from
https://photon.codes/docs/beta/get-started/imessage

A shared line cannot send the first message: text the agent number from your
phone after this prints "Ready".

Run from backend/:  .venv/bin/python scripts/photon_echo.py
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from photon_api import ApiError, Photon
from photon_api.generated.models import (
    SendMessageRequest,
    SendTextContent,
    TextContent,
    UserSnapshot,
)
from photon_api.rpc_generated import (
    ListMessagesInput,
    SendMessageHeader,
    SendMessageInput,
    SendMessagePath,
)

REQUIRED = ("PHOTON_API_KEY", "PHOTON_PROJECT_ID", "PHOTON_AGENT_HANDLE", "PHOTON_TEST_SENDER")


def configuration():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
    if not os.environ["PHOTON_PROJECT_ID"].startswith("pho_prj_"):
        raise RuntimeError(
            "PHOTON_PROJECT_ID must start with pho_prj_. Copy it from the Beta dashboard "
            "(app.beta.photon.codes > Settings); Stable/Spectrum credentials do not work here."
        )
    return {name: os.environ[name] for name in REQUIRED}


def incoming_text(item, c, started_at):
    """Return the text if this is a fresh plain-text DM from the test sender, else None."""
    m = item.root.root
    if m.platform != "imessage" or m.status != "received":
        return None
    if datetime.fromisoformat(m.createdAt) < started_at:
        return None
    if not isinstance(m.receivedBy, UserSnapshot):
        return None
    sender, agent, space = m.sentBy.root, m.receivedBy.root, m.space.root
    if sender.kind != "human" or sender.handle != c["PHOTON_TEST_SENDER"]:
        return None
    if agent.kind != "agent" or agent.handle != c["PHOTON_AGENT_HANDLE"] or sender.id == agent.id:
        return None
    if space.kind != "direct":
        return None
    content = m.content
    if isinstance(content, TextContent) and content.format == "plain":
        return content.text.strip()
    return None


def run(c, window_s=120.0, poll_s=2.0):
    started_at = datetime.now(timezone.utc)
    deadline = time.monotonic() + window_s
    created_since = (started_at - timedelta(seconds=5)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    with Photon(headers={"Authorization": f"Bearer {c['PHOTON_API_KEY']}"}) as photon:
        print(f"Ready. Text anything to {c['PHOTON_AGENT_HANDLE']} from {c['PHOTON_TEST_SENDER']}.")
        while True:
            query = {
                "createdSince": created_since,
                "direction": "received",
                "order": "asc",
                "pageSize": 100,
                "platform": "imessage",
            }
            while True:
                page = photon.projects.list_messages(
                    ListMessagesInput.model_validate({"path": {"id": c["PHOTON_PROJECT_ID"]}, "query": query})
                )
                for item in page.items:
                    text = incoming_text(item, c, started_at)
                    if text is None:
                        continue
                    m = item.root.root
                    print(f"Received: {text!r}")
                    photon.projects.send_message(
                        SendMessageInput(
                            path=SendMessagePath(id=c["PHOTON_PROJECT_ID"]),
                            headers=SendMessageHeader.model_validate({"idempotency-key": f"rollcall-echo:{m.id}"}),
                            body=SendMessageRequest(
                                spaceId=m.space.root.id,
                                senderId=m.receivedBy.root.id,
                                parentMessageId=m.id,
                                content=SendTextContent(type="text", format="plain", text=f"RollCall echo: {text}"),
                            ),
                        )
                    )
                    print("Reply accepted by Photon (delivery is asynchronous; check your phone).")
                    return
                token = page.nextPageToken
                if not isinstance(token, str):
                    break
                query["pageToken"] = token
            if time.monotonic() + poll_s > deadline:
                raise RuntimeError("No message from the test sender arrived within two minutes.")
            time.sleep(poll_s)


if __name__ == "__main__":
    try:
        run(configuration())
    except ApiError as error:
        print(f"Photon API error {error.status} (request {error.request_id or 'unknown'}).", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
