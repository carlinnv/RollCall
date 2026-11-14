"""Acceptance tests mapped to PRD 1 features."""
import pytest
from fastapi.testclient import TestClient

from app.data.seed import PACING_GUIDE
from app.main import app
from app.services import photon_channel
from app.services.classifier import classify
from app.store import store

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh():
    store.reset()
    photon_channel.audit.clear()
    photon_channel.opted_out.clear()


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
    assert client.get("/api/channel").json()["messages"][-1]["route"] == out["route"]


def test_inbound_safety_message_gets_no_automated_reply(parent):
    out = photon_channel.handle_inbound(parent, "She is scared, kids are bullying her on the bus.", "2026-11-14T09:00:00Z")
    assert out["route"] == "escalated" and out["reply"] is None
    assert any(e["absence_id"] == "A004" for e in client.get("/api/escalations").json())


def test_stop_opts_out_and_silences_later_messages(parent):
    assert photon_channel.handle_inbound(parent, " STOP ", "2026-11-14T09:00:00Z") == {
        "route": "opted_out", "absence_id": None, "reply": None}
    later = photon_channel.handle_inbound(parent, "The bus never came.", "2026-11-14T09:05:00Z")
    assert later["route"] == "opted_out" and later["reply"] is None
    assert client.get("/api/absences/A004").json()["status"] == "awaiting reply"


def test_unlinked_number_is_not_classified():
    out = photon_channel.handle_inbound("+19995550000", "The bus never came.", "2026-11-14T09:00:00Z")
    assert out["route"] == "unlinked" and out["absence_id"] is None
