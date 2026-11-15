"""Saving the store to TigerData: the snapshot/restore round trip and the change diff, without a database."""
from app.models import Barrier, FamilyReply
from app.services import outreach, persist
from app.store import Store


class FakeConn:
    """Records what persist._write sends, standing in for a psycopg connection."""

    def __init__(self):
        self.calls = []

    def transaction(self):
        return self

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, sql, rows):
        self.calls.append((sql.split()[0], list(rows)))


def _changed_store() -> Store:
    s = Store()
    # A reply, a correction, a plan, a setting and an opt-out: state beyond the seed.
    aid = next(a.id for a in s.absences.values() if a.reply is None)
    s.record_reply(aid, FamilyReply(channel="text", text="The bus never came this morning", language="en", received_at="2026-11-12T08:30:00"))
    s.correct(aid, Barrier.TRANSPORTATION, "clerk")
    s.draft_plan(absence_id=aid)
    s.update_settings(s.settings.model_copy(update={"daily_rate": 80}))
    s.complete_learning_plan(next(iter(s.students)), supervised_by="Mr. Chen")
    return s


def test_restore_reproduces_the_snapshot():
    before = _changed_store()
    outreach.opted_out.add("5551234567")
    try:
        rows = persist.snapshot(before)
        after = Store()
        outreach.opted_out.clear()
        persist.restore(after, rows)
        assert persist.snapshot(after) == rows
        assert "5551234567" in outreach.opted_out
        assert after.settings.daily_rate == 80
        assert len(after.messages) == len(before.messages)
        assert list(after.absences) == list(before.absences)  # order survives
    finally:
        outreach.opted_out.discard("5551234567")


def test_write_sends_only_changes():
    s = Store()
    first = persist.snapshot(s)
    conn = FakeConn()
    assert persist._write(conn, first, first) == (0, 0, False)
    assert conn.calls == []

    aid = next(a.id for a in s.absences.values() if a.reply is None)
    s.record_reply(aid, FamilyReply(channel="text", text="She has the flu", language="en", received_at="2026-11-12T08:30:00"))
    second = persist.snapshot(s)
    up, down, absences_changed = persist._write(conn, second, first)
    assert absences_changed and down == 0
    upserted = {(c, i) for c, i, _, _ in conn.calls[0][1]}
    assert ("absences", aid) in upserted
    assert all(c in {"absences", "messages", "state"} for c, _ in upserted)


def test_write_deletes_removed_records():
    s = Store()
    full = persist.snapshot(s)
    gone = next(k for k in full if k[0] == "absences")
    smaller = {k: v for k, v in full.items() if k != gone}
    conn = FakeConn()
    assert persist._write(conn, smaller, full)[1] == 1
    assert conn.calls[-1] == ("DELETE", [gone])


def test_off_without_a_database_url():
    assert not persist.enabled()
    assert not persist.save(Store())
