"""The "Send check-ins" button: one supportive first text per absence still awaiting a reply."""
from fastapi.testclient import TestClient

from app.main import app
from app.models import AbsenceStatus
from app.store import Store


def _sent(student, phone, text, key):
    return {"status": "sent", "reason": None, "channel": "imessage"}


def test_texts_each_awaiting_family_once():
    s = Store()
    s.sender = _sent
    awaiting = [a.id for a in s.absences.values() if a.status == AbsenceStatus.AWAITING_REPLY and a.reply is None
                and not s.household_suppressed(a.student_id)]
    assert awaiting and {a.id for a in s.check_in_targets()} == set(awaiting)

    first = s.send_check_ins()
    assert first["attempted"] == first["sent"] == len(awaiting)
    # Pressing again sends nothing: every family already has its check-in.
    assert s.check_in_targets() == []
    assert s.send_check_ins()["attempted"] == 0


def test_undeliverable_check_ins_are_tried_again():
    s = Store()  # no channel: messages queue with "messaging channel is off"
    n = len(s.check_in_targets())
    first = s.send_check_ins()
    assert first["queued"] == n and first["reasons"] == {"messaging channel is off": n}
    assert len(s.check_in_targets()) == n  # not delivered, so still waiting for a check-in

    s.sender = _sent
    assert s.send_check_ins()["sent"] == n


def test_household_with_open_safety_case_is_skipped():
    s = Store()
    held = [a for a in s.absences.values() if a.status == AbsenceStatus.AWAITING_REPLY and s.household_suppressed(a.student_id)]
    assert held, "the demo data has a household with an open safety case"
    assert not {a.id for a in held} & {a.id for a in s.check_in_targets()}
    assert s.send_check_ins()["blocked"] == 0


def test_families_who_replied_are_skipped():
    s = Store()
    replied = [a.id for a in s.absences.values() if a.reply is not None]
    assert replied and not set(replied) & {a.id for a in s.check_in_targets()}


def test_api_counts_then_sends():
    client = TestClient(app)
    client.post("/api/reset")
    pending = client.get("/api/check-ins").json()["to_send"]
    result = client.post("/api/check-ins").json()
    assert result["attempted"] == pending
    client.post("/api/reset")


def test_turning_quiet_hours_off_releases_held_check_ins(monkeypatch):
    from datetime import datetime

    from app.services import outreach

    s = Store()
    s.sender = _sent
    night = datetime(2026, 11, 14, 22, 0, tzinfo=outreach.school_now().tzinfo)
    monkeypatch.setenv("QUIET_HOURS", "on")
    monkeypatch.setattr(outreach, "school_now", lambda: night)
    held = s.send_check_ins()
    assert held["queued"] == held["attempted"] > 0  # all held until 8am
    assert s.flush_due() == 0  # still night

    monkeypatch.setenv("QUIET_HOURS", "off")
    assert s.flush_due() == held["attempted"]
    assert all(m.status == "sent" for m in s.messages if m.kind == "check_in")
