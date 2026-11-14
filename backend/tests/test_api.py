"""Acceptance tests mapped to PRD 1 features."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.data.seed import PACING_GUIDE
from app.main import app
from app.services import outreach, photon_channel
from app.services.classifier import classify
from app.store import _no_channel, store

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    store.reset()
    photon_channel.contacts.clear()
    outreach.opted_out.clear()
    monkeypatch.setattr(store, "sender", _no_channel)


# F1: every absence has a status and inspectable source message
def test_every_absence_has_status_and_source():
    rows = client.get("/api/absences").json()
    assert len(rows) > 30
    for row in rows:
        assert row["status"] in {"reason captured", "awaiting reply", "unknown"}
        if row["status"] == "reason captured":
            assert row["reply"]["text"]
            assert row["classification"]["evidence"]


def test_queue_puts_safety_first():
    rows = client.get("/api/absences").json()
    assert rows[0]["classification"]["barrier"] == "safety or bullying"


@pytest.mark.parametrize("text,barrier", [
    ("El autobús de la ruta 7 no pasó", "transportation"),
    ("She has a fever", "illness"),
    ("We got evicted", "housing or food insecurity"),
    ("Se quedó a cuidar a su hermanito", "caregiving or family obligation"),
    ("He hates school", "disengagement or academic struggle"),
    ("Dentist appointment", "appointment or travel"),
    ("Kids are bullying her on the bus", "safety or bullying"),  # safety beats transportation
    ("He was just out", "unknown"),
])
def test_classifier(text, barrier):
    assert classify(text).barrier.value == barrier


def test_safety_reply_escalates_without_plan():
    aid = "A031"  # awaiting reply
    r = client.post(f"/api/absences/{aid}/reply", json={"text": "Tengo miedo, le están haciendo acoso."}).json()
    assert r["route"] == "escalated"
    assert any(e["absence_id"] == aid for e in client.get("/api/escalations").json())
    assert client.post("/api/plans/draft", json={"absence_id": aid}).status_code == 400


def test_csv_import():
    csv = "student_id,date,sis_code\nS040,2026-11-13,unverified\nS999,2026-11-13,x\nS040,bad,x\n"
    r = client.post("/api/absences/import", json={"csv": csv}).json()
    assert r["added"] == 1
    assert len(r["errors"]) == 2


# Clusters: one bus-route cluster and one illness cluster in the demo data
def test_demo_clusters():
    clusters = client.get("/api/clusters").json()
    kinds = {(c["barrier"], c["link_type"], c["link_value"]) for c in clusters}
    assert ("transportation", "bus_route", "Route 7") in kinds
    assert ("illness", "classroom", "Room 104") in kinds
    bus = next(c for c in clusters if c["link_value"] == "Route 7")
    assert bus["ablation"]["text_only"] > bus["ablation"]["combined"]  # text alone over-groups
    assert bus["dollars_at_risk_week"] > 0


# F4: approve and send from a cluster
def test_cluster_plan_approve_and_send():
    bus = next(c for c in client.get("/api/clusters").json() if c["link_value"] == "Route 7")
    plan = client.post("/api/plans/draft", json={"cluster_id": bus["id"]}).json()
    assert plan["status"] == "draft" and plan["message_language"] == "es"
    sent = client.post(f"/api/plans/{plan['id']}/approve", json={"approved_by": "clerk"}).json()
    assert sent["status"] == "sent"
    outbox = client.get("/api/outbox").json()
    assert len(outbox) == len(bus["student_ids"])
    assert all("{guardian}" not in m["text"] for m in outbox)


# F5: plan only uses pacing-guide topics and scales with missed days
def test_learning_plan_grounded_and_scaled():
    plan = client.get("/api/learning-plans/S010").json()  # illness, 2 days
    topics = {t for _, _, t, _ in PACING_GUIDE[2]}
    assert plan["missed_days"] == 2
    assert all(s["topic"] in topics for s in plan["steps"])
    one_day = client.get("/api/learning-plans/S011").json()
    assert one_day["missed_days"] == 1 and one_day["total_minutes"] < plan["total_minutes"]
    assert plan["steps"][0]["minutes"] < 30  # lighter after illness


# F6: changing the daily rate recalculates every dollar figure
def test_rate_change_recalculates():
    before = client.get("/api/revenue").json()
    after = client.put("/api/revenue/settings", json={"state": "CA", "daily_rate": 130, "recovery_enabled": True}).json()
    assert after["dollars_at_risk_week"] == pytest.approx(before["dollars_at_risk_week"] * 2)
    assert after["dollars_recovered_to_date"] == pytest.approx(before["dollars_recovered_to_date"] * 2)
    bus = next(c for c in client.get("/api/clusters").json() if c["link_value"] == "Route 7")
    assert bus["dollars_at_risk_week"] == 130 * len(bus["absence_ids"])


def test_completing_learning_plan_recovers_dollars():
    before = client.get("/api/revenue").json()["days_recovered_to_date"]
    client.post("/api/learning-plans/S010/complete")
    assert client.get("/api/revenue").json()["days_recovered_to_date"] == before + 2


# F2: inbound iMessage runs through the same classify path as a simulated reply
@pytest.fixture
def parent(monkeypatch):
    monkeypatch.setenv("PHOTON_TEST_SENDER", "+15550001111")
    monkeypatch.setenv("PHOTON_DEMO_STUDENT_ID", "S002")
    return "+15550001111"


def test_inbound_text_classifies_awaiting_absence(parent):
    out = photon_channel.handle_inbound(parent, "El autobus no llego a la parada.", "2026-11-14T09:00:00Z", "m1")
    assert out["absence_id"] == "A004" and out["route"] in {"cluster", "case"}
    assert out["reply"] == photon_channel.ACK["es"] and "http" not in out["reply"]
    absence = client.get("/api/absences/A004").json()
    assert absence["status"] == "reason captured"
    assert absence["classification"]["barrier"] == "transportation"
    logged = client.get("/api/channel").json()["messages"][-1]
    assert logged["route"] == out["route"] and logged["external_id"] == "m1"


def test_inbound_safety_message_gets_no_automated_reply(parent):
    out = photon_channel.handle_inbound(parent, "She is scared, kids are bullying her on the bus.", "2026-11-14T09:00:00Z")
    assert out["route"] == "escalated" and out["reply"] is None
    assert any(e["absence_id"] == "A004" for e in client.get("/api/escalations").json())


def test_stop_opts_out_and_silences_later_messages(parent):
    first = photon_channel.handle_inbound(parent, " STOP ", "2026-11-14T09:00:00Z")
    assert (first["route"], first["absence_id"], first["reply"]) == ("opted_out", None, None)
    later = photon_channel.handle_inbound(parent, "The bus never came.", "2026-11-14T09:05:00Z")
    assert later["route"] == "opted_out" and later["reply"] is None
    assert client.get("/api/absences/A004").json()["status"] == "awaiting reply"


def test_unlinked_number_is_not_classified():
    out = photon_channel.handle_inbound("+19995550000", "The bus never came.", "2026-11-14T09:00:00Z")
    assert out["route"] == "unlinked" and out["absence_id"] is None


# F2 outbound: rules are checked with a stand-in for the Photon delivery function
@pytest.fixture
def delivered(monkeypatch):
    """Replace delivery with a recorder, as if every family were reachable over iMessage."""
    calls = []

    def fake(student, phone, text, key):
        calls.append({"student_id": student.id, "phone": phone, "text": text, "key": key})
        return {"status": "sent", "reason": None, "channel": "imessage"}

    monkeypatch.setattr(store, "sender", fake)
    return calls


def test_check_in_wording_follows_outreach_rules():
    for absence_id, lang_word in (("A016", "attendance clerk"), ("A004", "asistencia")):
        text = client.get(f"/api/absences/{absence_id}").json()["check_in_message"]
        assert lang_word in text  # says who reads replies, in the family's language
        assert "absence(s)" not in text and "ausencia" not in text  # no absence count
        assert "voice" not in text and "voz" not in text
        assert not outreach.has_link(text)


def test_check_in_is_delivered_once_rules_pass(delivered):
    out = client.post("/api/absences/A004/check-in").json()
    assert out["status"] == "sent" and out["kind"] == "check_in" and out["channel"] == "imessage"
    assert [c["student_id"] for c in delivered] == ["S002"]
    assert delivered[0]["key"] == "rollcall-checkin:A004"  # stable key, so a retry cannot double-send
    assert client.get("/api/outbox").json()[-1]["status"] == "sent"


def test_check_in_waits_in_outbox_when_channel_is_off():
    out = client.post("/api/absences/A004/check-in").json()
    assert out["status"] == "queued" and "off" in out["reason"]


def test_check_in_skipped_when_reason_already_on_file(delivered):
    client.post("/api/absences/A004/reply", json={"text": "El autobus no llego.", "language": "es"})
    out = client.post("/api/absences/A004/check-in").json()
    assert out["status"] == "blocked" and not delivered


def test_open_safety_escalation_suppresses_all_outbound_to_household(parent, delivered):
    # S030's household has a seeded open escalation: nothing automated may go to it.
    assert store.household_suppressed("S030")
    out = client.post("/api/absences/A021/check-in").json()
    assert out["status"] == "blocked" and "escalation" in out["reason"] and not delivered

    # After a safety message, a later ordinary text from the same family gets no automatic reply either.
    photon_channel.handle_inbound(parent, "She is scared, kids are bullying her.", "2026-11-14T09:00:00Z")
    later = photon_channel.handle_inbound(parent, "Also the bus was late.", "2026-11-14T09:05:00Z")
    assert later["reply"] is None
    assert client.post("/api/absences/A004/check-in").json()["status"] == "blocked"


def test_opted_out_family_is_not_messaged(parent, delivered):
    photon_channel.handle_inbound(parent, "STOP", "2026-11-14T09:00:00Z")
    out = client.post("/api/absences/A004/check-in").json()
    assert out["status"] == "blocked" and "opted out" in out["reason"] and not delivered


def test_plan_with_a_link_is_blocked(delivered):
    bus = next(c for c in client.get("/api/clusters").json() if c["link_value"] == "Route 7")
    plan = client.post("/api/plans/draft", json={"cluster_id": bus["id"]}).json()
    client.patch(f"/api/plans/{plan['id']}", json={"message": "Hola {guardian}, vea https://example.com/ruta7"})
    client.post(f"/api/plans/{plan['id']}/approve", json={"approved_by": "clerk"})
    outbox = client.get("/api/outbox").json()
    linked = [m for m in outbox if outreach.has_link(m["text"])]
    assert linked and all(m["status"] == "blocked" and "link" in m["reason"] for m in linked)
    # Families in another language get the standard template instead of the edited text, so those still go.
    assert not any(outreach.has_link(c["text"]) for c in delivered)


def test_approved_plan_is_delivered_to_each_family(delivered):
    bus = next(c for c in client.get("/api/clusters").json() if c["link_value"] == "Route 7")
    plan = client.post("/api/plans/draft", json={"cluster_id": bus["id"]}).json()
    client.post(f"/api/plans/{plan['id']}/approve", json={"approved_by": "clerk"})
    assert sorted(c["student_id"] for c in delivered) == sorted(bus["student_ids"])
    assert all(m["status"] == "sent" and m["plan_id"] == plan["id"] for m in client.get("/api/outbox").json())


def test_photon_send_needs_the_family_to_text_first():
    student = store.students["S002"]
    out = photon_channel.send_text(student, student.guardian_phone, "hello", "k1")
    assert out["status"] == "queued"


# F2 message log: every text is kept, with measured latency
def test_every_reply_is_kept_not_overwritten():
    client.post("/api/absences/A004/reply", json={"text": "Esta enferma, tiene fiebre.", "language": "es"})
    client.post("/api/absences/A004/reply", json={"text": "Y el autobus no llego.", "language": "es"})
    thread = client.get("/api/absences/A004").json()["messages"]
    assert [m["raw_text_or_transcript"] for m in thread] == ["Esta enferma, tiene fiebre.", "Y el autobus no llego."]
    assert [m["classification"]["barrier"] for m in thread] == ["illness", "transportation"]
    assert thread[0]["message_id"] != thread[1]["message_id"]


def test_reply_latency_is_measured_with_real_timestamps():
    client.post("/api/absences/A004/reply", json={"text": "El autobus no llego.", "language": "es"})
    message = client.get("/api/messages", params={"absence_id": "A004"}).json()[-1]
    assert message["direction"] == "inbound" and message["channel"] == "simulated"
    assert message["timestamp"].startswith(str(datetime.now(timezone.utc).year))  # real clock, not the demo date
    assert 0 <= message["latency_ms"] <= 10_000
    summary = client.get("/api/messages/latency").json()
    assert summary == {"target_ms": 10_000, "measured": 1, "within_target": 1,
                       "median_ms": message["latency_ms"], "max_ms": message["latency_ms"]}


def test_seed_history_is_logged_without_latency():
    seeded = client.get("/api/messages").json()
    assert seeded and all(m["channel"] == "seed" and m["latency_ms"] is None for m in seeded)
    assert client.get("/api/messages/latency").json()["measured"] == 0


def test_imessage_latency_counts_from_when_the_family_sent_it(parent):
    sent = (datetime.now(timezone.utc) - timedelta(seconds=3)).isoformat()
    photon_channel.handle_inbound(parent, "El autobus no llego.", sent, "m9")
    message = client.get("/api/messages", params={"absence_id": "A004"}).json()[-1]
    assert message["channel"] == "imessage" and message["phone"] == parent
    assert 3_000 <= message["latency_ms"] < 10_000  # includes the wait before we polled, not just classification
    assert client.get("/api/messages/latency", params={"channel": "imessage"}).json()["measured"] == 1


def test_outbound_messages_are_logged_with_their_outcome(delivered):
    client.post("/api/absences/A004/check-in")
    client.post("/api/absences/A021/check-in")  # household with an open escalation
    out = client.get("/api/messages", params={"direction": "outbound"}).json()
    assert [(m["kind"], m["status"], m["absence_id"]) for m in out] == [
        ("check_in", "sent", "A004"), ("check_in", "blocked", "A021")]
    assert out[0]["latency_ms"] is not None and out[1]["latency_ms"] is None and out[1]["reason"]


def test_texts_that_match_no_absence_are_still_logged():
    photon_channel.handle_inbound("+19995550000", "hello?", "2026-11-14T09:00:00Z", "m7")
    message = client.get("/api/messages").json()[-1]
    assert message["route"] == "unlinked" and message["absence_id"] is None and message["classification"] is None
