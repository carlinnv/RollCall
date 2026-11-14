"""Photon 2.0 (Beta) outbound check: send one message to your own phone.

Finds the conversation PHOTON_TEST_SENDER opened with the agent line in the last
three days and sends one text into it. A shared line cannot send first, so text
the agent number from your phone before running this.

Run from backend/:  .venv/bin/python scripts/photon_send.py ["message text"]
"""
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402
from photon_api import ApiError, Photon  # noqa: E402
from photon_api.generated.models import SendMessageRequest, SendTextContent, UserSnapshot  # noqa: E402
from photon_api.rpc_generated import (  # noqa: E402
    ListMessagesInput,
    SendMessageHeader,
    SendMessageInput,
    SendMessagePath,
)

from app.services import outreach  # noqa: E402

REQUIRED = ("PHOTON_API_KEY", "PHOTON_PROJECT_ID", "PHOTON_AGENT_HANDLE", "PHOTON_TEST_SENDER")
DEFAULT_TEXT = "RollCall outbound test: this message was sent by the school attendance agent."


def find_conversation(photon, c):
    """Return (space_id, agent_id) of the newest DM from the test sender to the agent line, or None."""
    since = datetime.now(timezone.utc) - timedelta(days=3)
    query = {"createdSince": since.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
             "direction": "received", "order": "asc", "pageSize": 100, "platform": "imessage"}
    found = None
    while True:
        page = photon.projects.list_messages(
            ListMessagesInput.model_validate({"path": {"id": c["PHOTON_PROJECT_ID"]}, "query": query}))
        for item in page.items:
            m = item.root.root
            if m.status != "received" or not isinstance(m.receivedBy, UserSnapshot):
                continue
            sender, agent, space = m.sentBy.root, m.receivedBy.root, m.space.root
            if (sender.kind == "human" and agent.kind == "agent" and space.kind == "direct"
                    and agent.handle == c["PHOTON_AGENT_HANDLE"]
                    and outreach.digits(sender.handle) == outreach.digits(c["PHOTON_TEST_SENDER"])):
                found = (space.id, agent.id)
        token = page.nextPageToken
        if not isinstance(token, str):
            return found
        query["pageToken"] = token


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
    c = {name: os.environ[name] for name in REQUIRED}
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT

    reason = outreach.blocked_reason(c["PHOTON_TEST_SENDER"], text, household_suppressed=False)
    if reason:
        raise RuntimeError(f"Not sent: {reason}.")

    with Photon(headers={"Authorization": f"Bearer {c['PHOTON_API_KEY']}"}) as photon:
        conversation = find_conversation(photon, c)
        if conversation is None:
            raise RuntimeError("No message from your phone in the last 3 days. Text the agent number first, then rerun.")
        space_id, agent_id = conversation
        photon.projects.send_message(SendMessageInput(
            path=SendMessagePath(id=c["PHOTON_PROJECT_ID"]),
            headers=SendMessageHeader.model_validate({"idempotency-key": f"rollcall-sendtest:{uuid.uuid4()}"}),
            body=SendMessageRequest(spaceId=space_id, senderId=agent_id,
                                    content=SendTextContent(type="text", format="plain", text=text)),
        ))
    print("Accepted by Photon (delivery is asynchronous; check your phone).")


if __name__ == "__main__":
    try:
        main()
    except ApiError as error:
        print(f"Photon API error {error.status} (request {error.request_id or 'unknown'}).", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
