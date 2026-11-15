"""Acceptance tests mapped to PRD 1 features."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.data.seed import PACING_GUIDE
from app.main import app
import json
import time

from app.services import case_report, classifier, oncall, outreach, photon_channel
from app.services.classifier import classify
from app.store import _no_channel, store

client = TestClient(app)


def set_school_clock(monkeypatch, hour, minute=0, day=16):
    """Pin school time to Nov `day`, 2026 at hour:minute in the school's time zone."""
    moment = datetime(2026, 11, day, hour, minute, tzinfo=outreach.school_timezone())
    monkeypatch.setattr(outreach, "school_now", lambda: moment)


@pytest.fixture(autouse=True)
def fresh(monkeypatch, tmp_path):
    store.reset()
    photon_channel.contacts.clear()
    outreach.opted_out.clear()
    monkeypatch.setattr(store, "sender", _no_channel)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # tests never call the real model
    monkeypatch.delenv("QUIET_HOURS", raising=False)
    for name in ("SCHOOL_DAYS", "SCHOOL_HOURS"):
        monkeypatch.delenv(name, raising=False)
    set_school_clock(monkeypatch, 10)  # mid-morning, outside quiet hours, unless a test moves it
    monkeypatch.setattr(store, "roster", oncall.Roster(tmp_path / "oncall.json"))  # never the real roster


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
    # PRD 1 F5: only a supervised Attendance Recovery session counts (deliberate change: the call now
    # names the supervisor; without one it recovers nothing, see test_integration.py).
    before = client.get("/api/revenue").json()["days_recovered_to_date"]
    client.post("/api/learning-plans/S010/complete", json={"supervised_by": "Ms. Rivera"})
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


def test_stop_opts_out_without_touching_the_absence(parent):
    first = photon_channel.handle_inbound(parent, " STOP ", "2026-11-14T09:00:00Z")
    assert (first["route"], first["absence_id"], first["reply"]) == ("opted_out", None, None)
    assert client.get("/api/absences/A004").json()["status"] == "awaiting reply"
    assert client.get("/api/messages").json()[-1]["reason"] == "family opted out of messages"


def test_opted_out_family_is_still_heard_but_never_answered(parent):
    photon_channel.handle_inbound(parent, "STOP", "2026-11-14T09:00:00Z")
    later = photon_channel.handle_inbound(parent, "The bus never came.", "2026-11-14T09:05:00Z")
    assert later["reply"] is None and later["absence_id"] == "A004"
    assert client.get("/api/absences/A004").json()["classification"]["barrier"] == "transportation"
    # A safety disclosure after an opt-out must still reach the counselor.
    photon_channel.handle_inbound(parent, "She is scared, kids are bullying her.", "2026-11-14T09:10:00Z")
    assert any(e["student_id"] == "S002" for e in client.get("/api/escalations").json())


@pytest.mark.parametrize("text", [
    "Please stop texting me", "No more messages please", "don't text me again", "Do not contact us",
    "take me off this list",
    "No más mensajes", "Ya no me escriban", "Dejen de mandarme mensajes", "No quiero recibir más mensajes",
    "Quiero darme de baja", "ALTO",
    "Đừng nhắn tin nữa", "Ngừng gửi tin nhắn", "Tôi không muốn nhận tin nhắn nữa", "Hủy đăng ký",
])
def test_plain_language_opt_outs(parent, delivered, text):
    out = photon_channel.handle_inbound(parent, text, "2026-11-14T09:00:00Z")
    assert out["route"] == "opted_out" and out["reply"] is None
    assert client.post("/api/absences/A004/check-in").json()["status"] == "blocked" and not delivered


@pytest.mark.parametrize("text", [
    "The bus didn't stop at our street", "El autobús no paró en la parada", "She wants to quit school",
    "Please stop sending the bus late", "He told the kids to leave him alone", "We had to cancel the appointment",
    "Tôi không nhận được tin nhắn", "Xe buýt không dừng lại",
])
def test_ordinary_replies_are_not_mistaken_for_opt_outs(text):
    assert not outreach.is_opt_out(text)


def test_opt_out_with_a_reason_is_honored_and_classified(parent):
    out = photon_channel.handle_inbound(parent, "No more texts, she has a fever.", "2026-11-14T09:00:00Z")
    assert out["route"] in {"cluster", "case"} and out["reply"] is None
    assert client.get("/api/absences/A004").json()["classification"]["barrier"] == "illness"
    assert client.post("/api/absences/A016/check-in").json()["status"] != "blocked"  # other families unaffected
    assert outreach.digits(parent) in outreach.opted_out


def test_opt_out_does_not_overwrite_a_reason_already_captured(parent):
    photon_channel.handle_inbound(parent, "El autobus no llego.", "2026-11-14T09:00:00Z")
    photon_channel.handle_inbound(parent, "No más mensajes", "2026-11-14T09:05:00Z")
    absence = client.get("/api/absences/A004").json()
    assert absence["status"] == "reason captured" and absence["classification"]["barrier"] == "transportation"


def test_start_opts_back_in(parent, delivered):
    photon_channel.handle_inbound(parent, "no more messages", "2026-11-14T09:00:00Z")
    out = photon_channel.handle_inbound(parent, "START", "2026-11-14T09:05:00Z")
    assert out["route"] == "opted_in"
    assert client.post("/api/absences/A004/check-in").json()["status"] == "sent"


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


# Keyword safety check: the fallback when Gemini is down must still catch indirect statements
@pytest.mark.parametrize("text", [
    "She says she doesn't want to live anymore.",
    "She says she doesn’t want to live anymore.",
    "He told his sister he wants to die",
    "I think she might hurt herself",
    "He said he wants to end his life",
    "she is suicidal",
    "Dice que ya no quiere vivir.",
    "Dijo que quiere morirse",
    "Mi hijo dice que se quiere matar.",
    "Me dijo: me voy a matar",
    "Dice que se va a matar si vuelve a la escuela",
    "Cháu nói muốn chết",
    "Con bị bắt nạt ở trường",
    "Cháu không muốn sống nữa",
])
def test_keyword_safety_catches_indirect_statements(text):
    c = classifier.keyword_classify(text)
    assert c.safety_critical and c.barrier == "safety or bullying"


@pytest.mark.parametrize("text", [
    "Đi từ từ thôi, the bus was slow",  # "từ từ" (slowly) must not match "tự tử" (suicide)
    "He doesn't want to go to school",
    "No quiere ir a la escuela",
    "She was sick with the flu",
])
def test_keyword_safety_ignores_ordinary_replies(text):
    assert not classifier.keyword_classify(text).safety_critical


# Gemini classifier, with a stand-in for the API call
@pytest.fixture
def gemini(monkeypatch, tmp_path):
    """Turn Gemini on with a fake model. Set `answer` (dict, or an Exception to raise); read `sent`."""
    state = {"answer": None, "sent": [], "delay": 0.0}

    def fake_generate(redacted):
        state["sent"].append(redacted)
        if state["delay"]:
            time.sleep(state["delay"])
        if isinstance(state["answer"], Exception):
            raise state["answer"]
        return state["answer"] if isinstance(state["answer"], str) else json.dumps(state["answer"])

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(classifier, "_generate", fake_generate)
    monkeypatch.setattr(classifier, "CACHE_DIR", tmp_path)
    return state


def answer(label, quote="", confidence=0.9, language="es", safety=False, safety_type=None):
    return {"label": label, "evidence_quote": quote, "confidence": confidence, "language": language,
            "safety_concern": safety, "safety_type": safety_type or ("other" if safety else "none")}


def test_gemini_label_language_and_evidence_are_used(gemini):
    text = "Mi hija no fue porque el camión no pasó por la parada."
    gemini["answer"] = answer("transportation", "el camión no pasó por la parada", language="es")
    client.post("/api/absences/A004/reply", json={"text": text, "language": "en"})
    a = client.get("/api/absences/A004").json()
    assert a["classification"]["barrier"] == "transportation" and a["classification"]["source"] == "gemini"
    assert a["classification"]["evidence"] == "el camión no pasó por la parada"
    assert a["messages"][-1]["language"] == "es"  # detected, not the student's record


def test_language_the_keywords_miss_is_classified(gemini):
    gemini["answer"] = answer("illness", "con bị sốt", language="vi")
    route = client.post("/api/absences/A016/reply", json={"text": "Hôm qua con bị sốt cao.", "language": "en"}).json()
    assert route["absence"]["classification"]["barrier"] == "illness"
    # The keyword rules read Vietnamese now too; Gemini is still the path used when it is on.
    assert route["absence"]["classification"]["source"] == "gemini"


def test_evidence_that_is_not_in_the_message_is_replaced_by_the_message(gemini):
    gemini["answer"] = answer("transportation", "the bus did not come", language="es")  # a translation, not a quote
    text = "El camión no pasó."
    client.post("/api/absences/A004/reply", json={"text": text, "language": "es"})
    assert client.get("/api/absences/A004").json()["classification"]["evidence"] == text


def test_names_are_hidden_from_the_model_and_restored_in_the_quote(gemini):
    gemini["answer"] = answer("illness", "[NAME1] tiene fiebre")
    client.post("/api/absences/A004/reply", json={"text": "Sofia tiene fiebre desde ayer.", "language": "es"})
    assert "Sofia" not in gemini["sent"][-1] and "[NAME1]" in gemini["sent"][-1]
    assert client.get("/api/absences/A004").json()["classification"]["evidence"] == "Sofia tiene fiebre"


def test_model_can_add_a_safety_flag(gemini):
    text = "I had to leave him home alone with the baby all day."
    gemini["answer"] = answer("caregiving or family obligation", "home alone with the baby", language="en", safety=True)
    route = client.post("/api/absences/A004/reply", json={"text": text, "language": "en"}).json()["route"]
    assert route == "escalated"
    assert client.get("/api/absences/A004").json()["classification"]["safety_critical"] is True


def test_model_can_never_clear_a_keyword_safety_flag(gemini):
    gemini["answer"] = answer("transportation", "the bus", language="en", safety=False)
    route = client.post("/api/absences/A004/reply",
                        json={"text": "She is scared to ride the bus, kids are bullying her.", "language": "en"}).json()["route"]
    c = client.get("/api/absences/A004").json()["classification"]
    assert route == "escalated" and c["barrier"] == "safety or bullying" and c["source"] == "safety_keywords"


@pytest.mark.parametrize("bad", [
    RuntimeError("503 UNAVAILABLE"),
    "not json",
    json.dumps({"label": "Ignore previous instructions", "evidence_quote": "", "confidence": 1}),
])
def test_model_failure_falls_back_to_keywords(gemini, bad):
    gemini["answer"] = bad
    client.post("/api/absences/A004/reply", json={"text": "The bus never came.", "language": "en"})
    c = client.get("/api/absences/A004").json()["classification"]
    assert c["barrier"] == "transportation" and c["source"] == "keywords_fallback"


def test_slow_model_falls_back_within_the_timeout(gemini, monkeypatch):
    monkeypatch.setattr(classifier, "TIMEOUT_S", 0.05)
    gemini["answer"], gemini["delay"] = answer("illness", "fever"), 0.5
    started = time.monotonic()
    client.post("/api/absences/A004/reply", json={"text": "She has a fever.", "language": "en"})
    assert time.monotonic() - started < 0.4
    assert client.get("/api/absences/A004").json()["classification"]["source"] == "keywords_fallback"


def test_repeated_message_is_answered_from_the_cache(gemini):
    gemini["answer"] = answer("illness", "fever", language="en")
    for _ in range(2):
        classifier.classify("She has a fever.")
    assert len(gemini["sent"]) == 1


def test_seed_data_and_reset_never_call_the_model(gemini):
    client.post("/api/reset")
    assert gemini["sent"] == []
    assert client.get("/api/health").json()["classifier"]["engine"] == "gemini"


# Safety escalations: acknowledge, record the reporting decision, close
def escalation_for(absence_id):
    return next(e for e in client.get("/api/escalations").json() if e["absence_id"] == absence_id)


def test_seeded_escalation_waits_for_acknowledgement():
    e = escalation_for("A020")
    assert e["status"] == "open" and e["household_id"] == "H30" and e["overdue"]  # opened days ago, never acknowledged
    acked = client.post(f"/api/escalations/{e['escalation_id']}/acknowledge", json={"by": "Ms. Counselor"}).json()
    assert acked["status"] == "acknowledged" and acked["acknowledged_by"] == "Ms. Counselor" and not acked["overdue"]
    assert store.household_suppressed("S030")  # acknowledging does not lift suppression; only closing does


def test_cannot_close_without_a_reporting_decision():
    eid = escalation_for("A020")["escalation_id"]
    r = client.post(f"/api/escalations/{eid}/close", json={"outcome": "resolved"})
    assert r.status_code == 400 and "mandated-reporting" in r.json()["detail"]


def test_closing_a_real_concern_lifts_suppression_and_keeps_the_label(delivered):
    eid = escalation_for("A020")["escalation_id"]
    client.post(f"/api/escalations/{eid}/report", json={"decision": "report_filed", "by": "Ms. Counselor"})
    closed = client.post(f"/api/escalations/{eid}/close", json={"outcome": "resolved", "by": "Ms. Counselor"}).json()
    assert closed["status"] == "closed" and closed["report_status"] == "report_filed" and closed["acknowledged_at"]
    assert not store.household_suppressed("S030")
    assert client.post("/api/absences/A021/check-in").json()["status"] == "sent"
    a = client.get("/api/absences/A020").json()
    assert not a["escalated"] and a["classification"]["barrier"] == "safety or bullying"
    # A closed safety case still never gets an automated plan.
    assert client.post("/api/plans/draft", json={"absence_id": "A020"}).status_code == 400


def test_false_alarm_is_relabeled_by_the_counselor():
    route = client.post("/api/absences/A004/reply", json={"text": "She hurt her ankle at practice.", "language": "en"}).json()["route"]
    assert route == "escalated"
    eid = escalation_for("A004")["escalation_id"]
    client.post(f"/api/escalations/{eid}/report", json={"decision": "not_required"})
    bad = client.post(f"/api/escalations/{eid}/close", json={"outcome": "not_safety"})
    assert bad.status_code == 400  # a false alarm needs the correct label
    client.post(f"/api/escalations/{eid}/close", json={"outcome": "not_safety", "relabel_to": "illness"})
    c = client.get("/api/absences/A004").json()["classification"]
    assert c["barrier"] == "illness" and c["source"] == "counselor" and not c["safety_critical"]
    assert client.post("/api/plans/draft", json={"absence_id": "A004"}).status_code == 200


def test_follow_up_cannot_downgrade_an_open_safety_label():
    client.post("/api/absences/A004/reply", json={"text": "She hurt her ankle at practice.", "language": "en"})
    route = client.post("/api/absences/A004/reply", json={"text": "Sorry, she just has a fever.", "language": "en"}).json()["route"]
    a = client.get("/api/absences/A004").json()
    assert route == "escalated" and a["escalated"] and a["classification"]["barrier"] == "safety or bullying"
    assert a["messages"][-1]["classification"]["barrier"] == "illness"  # the message keeps its own label


def test_new_safety_message_after_closing_opens_a_new_escalation():
    eid = escalation_for("A020")["escalation_id"]
    client.post(f"/api/escalations/{eid}/report", json={"decision": "not_required"})
    client.post(f"/api/escalations/{eid}/close", json={"outcome": "resolved"})
    client.post("/api/absences/A020/reply", json={"text": "They threatened her again today.", "language": "en"})
    mine = [e for e in client.get("/api/escalations").json() if e["absence_id"] == "A020"]
    assert [e["status"] for e in mine] == ["open", "closed"]  # open first; the closed one stays on record


def test_another_open_case_in_the_household_keeps_it_silent():
    for aid in ("A018", "A019"):  # siblings in household H20
        client.post(f"/api/absences/{aid}/reply", json={"text": "Kids are bullying him at school.", "language": "en"})
    first = escalation_for("A018")["escalation_id"]
    client.post(f"/api/escalations/{first}/report", json={"decision": "not_required"})
    client.post(f"/api/escalations/{first}/close", json={"outcome": "resolved"})
    assert store.household_suppressed("S020")  # the sibling's case is still open


def test_closed_escalation_cannot_be_changed():
    eid = escalation_for("A020")["escalation_id"]
    client.post(f"/api/escalations/{eid}/report", json={"decision": "not_required"})
    client.post(f"/api/escalations/{eid}/close", json={"outcome": "resolved"})
    assert client.post(f"/api/escalations/{eid}/acknowledge", json={}).status_code == 400
    assert client.post("/api/escalations/E999/acknowledge", json={}).status_code == 404


# Internal case reports on escalations
def test_case_report_is_filled_from_the_record():
    client.post("/api/absences/A004/reply", json={"text": "She hurt her ankle at practice.", "language": "en"})
    eid = escalation_for("A004")["escalation_id"]
    client.post(f"/api/escalations/{eid}/acknowledge", json={"by": "Ms. Counselor"})
    client.post(f"/api/escalations/{eid}/report", json={"decision": "not_required", "by": "Ms. Counselor"})
    client.post(f"/api/escalations/{eid}/close", json={"outcome": "not_safety", "relabel_to": "illness", "by": "Ms. Counselor"})
    text = client.post(f"/api/escalations/{eid}/case-report/draft", json={"by": "Ms. Counselor"}).json()["text"]
    for expected in ["CONFIDENTIAL", eid, "Sofia", "S002", '"She hurt her ankle at practice."', "Flagged by: keyword rules",
                     "Acknowledged by Ms. Counselor", "No report required", "Not a safety concern; relabeled illness",
                     "The system does not file mandated reports"]:
        assert expected in text, expected
    assert client.get(f"/api/escalations/{eid}/case-reports").json() == []  # a draft is not saved


def test_case_report_versions_are_kept_and_notes_survive_regeneration():
    eid = escalation_for("A020")["escalation_id"]
    draft = client.post(f"/api/escalations/{eid}/case-report/draft", json={}).json()["text"]
    first = draft.replace(case_report.NOTES_PLACEHOLDER, "Spoke with the guardian on the phone.")
    v1 = client.post(f"/api/escalations/{eid}/case-reports", json={"text": first, "by": "Ms. Counselor"}).json()
    assert v1["version"] == 1 and v1["saved_by"] == "Ms. Counselor"
    client.post(f"/api/escalations/{eid}/acknowledge", json={})
    regenerated = client.post(f"/api/escalations/{eid}/case-report/draft", json={}).json()["text"]
    assert "Spoke with the guardian on the phone." in regenerated and "Acknowledged by" in regenerated
    client.post(f"/api/escalations/{eid}/case-reports", json={"text": regenerated})
    saved = client.get(f"/api/escalations/{eid}/case-reports").json()
    assert [r["version"] for r in saved] == [1, 2] and saved[0]["text"] == first
    assert escalation_for("A020")["case_report_versions"] == 2


def test_empty_case_report_is_rejected():
    eid = escalation_for("A020")["escalation_id"]
    assert client.post(f"/api/escalations/{eid}/case-reports", json={"text": "   "}).status_code == 400
    assert client.post("/api/escalations/E999/case-report/draft", json={}).status_code == 404


# Quiet hours: check-ins and plan messages wait for 8am school time
def test_check_in_at_night_is_held_until_morning(delivered, monkeypatch):
    set_school_clock(monkeypatch, 21, 30)
    out = client.post("/api/absences/A004/check-in").json()
    assert out["status"] == "queued" and "quiet hours" in out["reason"] and not delivered
    held = client.get("/api/messages", params={"direction": "outbound"}).json()[-1]
    assert held["deliver_after"].startswith("2026-11-17T16:00")  # 8am Pacific
    assert store.flush_due() == 0  # still night
    set_school_clock(monkeypatch, 8, 5, day=17)
    assert store.flush_due() == 1
    assert [c["key"] for c in delivered] == ["rollcall-checkin:A004"]
    sent = client.get("/api/messages", params={"direction": "outbound"}).json()[-1]
    assert sent["status"] == "sent" and sent["deliver_after"] is None


def test_held_message_is_checked_again_before_sending(parent, delivered, monkeypatch):
    set_school_clock(monkeypatch, 6)
    client.post("/api/absences/A004/check-in")
    photon_channel.handle_inbound(parent, "STOP", "2026-11-16T14:30:00Z")
    set_school_clock(monkeypatch, 9)
    store.flush_due()
    assert not delivered
    assert client.get("/api/messages", params={"direction": "outbound"}).json()[-1]["status"] == "blocked"


def test_quiet_hours_can_be_switched_off(delivered, monkeypatch):
    monkeypatch.setenv("QUIET_HOURS", "off")
    set_school_clock(monkeypatch, 23)
    assert client.post("/api/absences/A004/check-in").json()["status"] == "sent"


def test_reply_to_a_family_text_is_not_held(parent, monkeypatch):
    set_school_clock(monkeypatch, 22)
    out = photon_channel.handle_inbound(parent, "El autobus no llego.", "2026-11-17T06:00:00Z")
    assert out["reply"] is not None


# Crisis-resource message after a self-harm flag
def crisis_messages():
    return [m for m in client.get("/api/messages", params={"direction": "outbound"}).json() if m["kind"] == "crisis"]


def test_self_harm_flag_sends_the_crisis_message_once(delivered, monkeypatch):
    set_school_clock(monkeypatch, 23)  # not held for quiet hours
    client.post("/api/absences/A004/reply", json={"text": "Dice que ya no quiere vivir.", "language": "es"})
    client.post("/api/absences/A004/reply", json={"text": "Sigue diciendo que quiere morirse.", "language": "es"})
    assert [c["key"] for c in delivered] == [f"rollcall-crisis:{escalation_for('A004')['escalation_id']}"]
    assert "988" in delivered[0]["text"] and "911" in delivered[0]["text"] and not outreach.has_link(delivered[0]["text"])
    assert delivered[0]["text"] == outreach.CRISIS["es"]  # fixed text, in the family's language
    e = escalation_for("A004")
    assert e["crisis_message_status"] == "sent" and "self_harm" in e["safety_types"]
    # At 11pm only the crisis message goes: another family's check-in waits for morning,
    # and everything else to this household stays paused.
    assert "quiet hours" in client.post("/api/absences/A016/check-in").json()["reason"]
    assert store.household_suppressed("S002")


def test_crisis_message_is_never_sent_when_abuse_is_disclosed(delivered):
    client.post("/api/absences/A004/reply",
                json={"text": "She doesn't want to live anymore because her dad hit her.", "language": "en"})
    e = escalation_for("A004")
    assert e["crisis_message_status"] == "withheld" and "phone may not be safe" in e["crisis_message_reason"]
    assert not delivered and not crisis_messages()


def test_crisis_message_is_not_sent_for_bullying(delivered):
    client.post("/api/absences/A004/reply", json={"text": "Kids are bullying her on the bus.", "language": "en"})
    assert escalation_for("A004")["crisis_message_status"] is None and not delivered


def test_model_can_add_abuse_and_stop_the_crisis_message(gemini, delivered):
    gemini["answer"] = answer("safety or bullying", "doesn't want to live", language="en", safety=True,
                              safety_type="abuse_or_neglect")
    client.post("/api/absences/A004/reply", json={"text": "She says she doesn't want to live at home.", "language": "en"})
    e = escalation_for("A004")
    assert set(e["safety_types"]) == {"self_harm", "abuse_or_neglect"} and e["crisis_message_status"] == "withheld"
    assert not delivered


def test_crisis_message_respects_opt_out(parent, delivered):
    photon_channel.handle_inbound(parent, "STOP", "2026-11-16T18:00:00Z")
    photon_channel.handle_inbound(parent, "She wants to die.", "2026-11-16T18:05:00Z")
    assert escalation_for("A004")["crisis_message_status"] == "blocked" and not delivered


def test_seed_data_never_sends_a_crisis_message(delivered):
    client.post("/api/reset")
    assert not delivered and not crisis_messages()


@pytest.mark.parametrize("text", [
    "Mi hijo dice que se quiere matar.",
    "Me dijo: me voy a matar",
    "Dijo que quiere quitarse la vida",
])
def test_spanish_self_harm_is_typed_as_self_harm(text):
    # Typed as self-harm, so the crisis-resource message goes out.
    assert "self_harm" in classifier.keyword_classify(text).safety_types


@pytest.mark.parametrize("text", ["Vamos a matar el tiempo en la biblioteca", "Se va a mudar con su tía"])
def test_spanish_look_alikes_are_not_flagged(text):
    assert not classifier.keyword_classify(text).safety_critical


# On-call roster and paging
@pytest.fixture
def pages(monkeypatch):
    """Record pages instead of sending them."""
    sent = []

    def fake(phone, text, key):
        sent.append({"phone": phone, "text": text, "key": key})
        return {"status": "sent", "reason": None, "channel": "imessage"}

    monkeypatch.setattr(store, "staff_sender", fake)
    return sent


def add_staff(name, role, phone, on_call=True):
    r = client.post("/api/oncall/staff", json={"name": name, "role": role, "phone": phone, "on_call": on_call})
    assert r.status_code == 200, r.text
    return r.json()


def safety_message(absence_id="A004"):
    client.post(f"/api/absences/{absence_id}/reply", json={"text": "Kids are bullying her on the bus.", "language": "en"})
    return escalation_for(absence_id)


def test_roster_is_validated_and_saved_to_disk():
    member = add_staff("Ms. Rivera", "counselor", "+1 555 010 9999")
    assert member["staff_id"] == "T01" and member["on_call"] and member["reachable"] is False
    for bad in [{"name": "X", "role": "janitor", "phone": "+15550109998"},
                {"name": "X", "role": "counselor", "phone": "555"},
                {"name": "X", "role": "counselor", "phone": "+15550109999"}]:  # duplicate number
        assert client.post("/api/oncall/staff", json=bad).status_code == 400
    client.patch("/api/oncall/staff/T01", json={"on_call": False})
    assert [s.on_call for s in oncall.Roster(store.roster.path).staff.values()] == [False]  # read back from the file
    client.delete("/api/oncall/staff/T01")
    assert oncall.Roster(store.roster.path).staff == {}


def test_reset_keeps_the_roster():
    add_staff("Ms. Rivera", "counselor", "+15550109999")
    client.post("/api/reset")
    assert [s["name"] for s in client.get("/api/oncall").json()["staff"]] == ["Ms. Rivera"]


def test_new_case_in_school_hours_pages_the_on_call_counselor(pages):
    add_staff("Ms. Rivera", "counselor", "+15550109999")
    add_staff("Mr. Diaz", "administrator", "+15550108888")
    add_staff("Off Duty", "counselor", "+15550107777", on_call=False)
    e = safety_message()
    assert [p["phone"] for p in pages] == ["+15550109999"]
    assert e["pages"][0]["name"] == "Ms. Rivera" and e["pages"][0]["status"] == "sent" and e["pages"][0]["level"] == "first"
    text = pages[0]["text"]
    assert e["escalation_id"] in text and "ACK" in text
    assert "Sofia" not in text and "Ramírez" not in text and "bullying" not in text  # no family details on a staff phone
    assert client.get("/api/oncall").json()["would_page_now"] == ["Ms. Rivera"]


def test_after_hours_pages_the_administrator(pages, monkeypatch):
    add_staff("Ms. Rivera", "counselor", "+15550109999")
    add_staff("Mr. Diaz", "administrator", "+15550108888")
    set_school_clock(monkeypatch, 19)
    safety_message()
    assert [p["phone"] for p in pages] == ["+15550108888"]
    assert client.get("/api/oncall").json()["first_role_now"] == "administrator"


def test_falls_through_the_chain_and_warns_when_nobody_is_on_call(pages, monkeypatch):
    set_school_clock(monkeypatch, 19)
    add_staff("Dr. Kim", "principal", "+15550106666")
    safety_message("A004")
    assert [p["phone"] for p in pages] == ["+15550106666"]  # no administrator on call: the principal
    client.patch("/api/oncall/staff/T01", json={"on_call": False})
    e = safety_message("A016")
    assert e["pages"][0]["status"] == "blocked" and "no one on call" in e["pages"][0]["reason"]


def test_unacknowledged_case_pages_the_principal_after_15_minutes(pages, monkeypatch):
    add_staff("Ms. Rivera", "counselor", "+15550109999")
    add_staff("Dr. Kim", "principal", "+15550106666")
    safety_message()
    set_school_clock(monkeypatch, 10, 10)
    assert store.check_paging() == 0  # 10 minutes: not yet
    set_school_clock(monkeypatch, 10, 16)
    client.get("/api/escalations")  # reading the inbox runs the check
    assert [p["phone"] for p in pages] == ["+15550109999", "+15550106666"]
    assert "not been acknowledged" in pages[1]["text"]
    assert store.check_paging() == 0  # the principal is paged once


def test_acknowledged_case_is_not_escalated(pages, monkeypatch):
    add_staff("Ms. Rivera", "counselor", "+15550109999")
    add_staff("Dr. Kim", "principal", "+15550106666")
    e = safety_message()
    client.post(f"/api/escalations/{e['escalation_id']}/acknowledge", json={})
    set_school_clock(monkeypatch, 11)
    assert store.check_paging() == 0 and len(pages) == 1


def test_staff_can_acknowledge_by_replying_ack(pages):
    add_staff("Ms. Rivera", "counselor", "+15550109999")
    eid = safety_message()["escalation_id"]
    out = photon_channel.handle_inbound("+15550109999", "Ack", "2026-11-16T18:00:00Z")
    assert out["route"] == "staff" and out["reply"] == f"Acknowledged {eid}. Details are on the Escalations page."
    e = escalation_for("A004")
    assert e["status"] == "acknowledged" and e["acknowledged_by"] == "Ms. Rivera"
    again = photon_channel.handle_inbound("+15550109999", "ACK", "2026-11-16T18:01:00Z")
    assert again["reply"] == "You have no unacknowledged cases."


def test_staff_phone_can_still_play_a_family(parent, pages):
    add_staff("Carlin", "counselor", parent)  # one phone for both roles while testing
    out = photon_channel.handle_inbound(parent, "El autobus no llego.", "2026-11-16T18:00:00Z")
    assert out["route"] in {"cluster", "case"} and out["absence_id"] == "A004"


def test_seed_escalation_is_not_paged(pages):
    add_staff("Ms. Rivera", "counselor", "+15550109999")
    client.post("/api/reset")
    assert not pages and escalation_for("A020")["pages"] == []


def test_test_page_goes_through_the_staff_channel(pages):
    add_staff("Ms. Rivera", "counselor", "+15550109999")
    assert client.post("/api/oncall/staff/T01/test").json()["status"] == "sent"
    assert pages[0]["phone"] == "+15550109999" and "test page" in pages[0]["text"]


def test_weekend_counts_as_school_days_when_configured(pages, monkeypatch):
    add_staff("Ms. Rivera", "counselor", "+15550109999")
    add_staff("Mr. Diaz", "administrator", "+15550108888")
    set_school_clock(monkeypatch, 10, day=14)  # Saturday morning
    assert client.get("/api/oncall").json()["first_role_now"] == "administrator"
    monkeypatch.setenv("SCHOOL_DAYS", "all")
    assert client.get("/api/oncall").json()["first_role_now"] == "counselor"
