"""Integration tests for the PRD 2 live pipeline: clusters (app.ml.live), F8 corrections, F5 claim check
and the Attendance Recovery rule. No network: the Gemini paths use injected fake clients."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.data.seed import PACING_GUIDE, WEEK_END, WEEK_START
from app.main import app
from app.ml import detectors, live
from app.models import Barrier
from app.services import claim_check, corrections, learning_plan
from app.store import TODAY, store

client = TestClient(app)

PRD2_FIELDS = ("id", "layers_involved", "zone", "member_ids", "likely_cause", "cause_provisional", "observed",
               "expected", "p_value", "start_date", "detected_on", "weekly_revenue_at_risk",
               "evidence_message_ids", "method", "method_note", "dataset")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    for var in ("DATASET", "CLUSTER_METHOD", "DETECTOR", "CLASSIFIER", "CLAIM_CHECK_NLI", "CORRECTION_SIM_THRESHOLD"):
        monkeypatch.delenv(var, raising=False)
    store.reset()


def _clusters():
    return client.get("/api/clusters").json()


def _find(clusters, link_value, cause):
    return next(c for c in clusters if c["link_value"] == link_value and c["likely_cause"] == cause)


# ---------------------------------------------------------------------------------------------
# clusters


def test_clusters_have_prd2_fields_and_acceptance_clusters():
    clusters = _clusters()
    replies = {f"M-{a.id}": a for a in store.absences.values() if a.reply}
    for c in clusters:
        for f in PRD2_FIELDS:
            assert f in c, f
        assert c["method"] == "rule_based" or c["method"].startswith("scan:")
        assert c["method_note"]
        assert c["dataset"] == "demo"
        assert c["member_ids"] == c["student_ids"]
        assert c["likely_cause"] == c["barrier"] and c["likely_cause"] not in ("safety or bullying", "unknown")
        assert 0 <= c["p_value"] <= 1 and c["observed"] > c["expected"]
        assert c["start_date"] <= c["detected_on"]
        assert c["weekly_revenue_at_risk"] == c["dollars_at_risk_week"]
        assert c["evidence_message_ids"], "every cluster cites family messages"
        for mid in c["evidence_message_ids"]:  # each cited message is a real reply of a member
            assert mid in replies and replies[mid].student_id in c["member_ids"]
    # PRD 2 section 3.2 acceptance: a route cluster and an illness cluster, each with a p-value
    bus = _find(clusters, "Route 7", "transportation")
    ill = _find(clusters, "Room 104", "illness")
    assert bus["link_type"] == "bus_route" and bus["zone"] == "route:Route 7" and "route" in bus["layers_involved"]
    assert ill["link_type"] == "classroom" and ill["zone"] == "section:Room 104"
    assert bus["p_value"] < 0.01 and ill["p_value"] < 0.01
    assert set(bus["member_ids"]) == {"S001", "S002", "S003", "S004", "S005"}
    assert "S035" not in bus["member_ids"]  # car trouble off Route 7: text alone would over-group
    assert bus["detected_on"] == "2026-11-10" and ill["detected_on"] == "2026-11-12"


def test_cluster_ids_are_deterministic():
    first = [c["id"] for c in _clusters()]
    store.reset()
    assert [c["id"] for c in _clusters()] == first
    assert len(set(first)) == len(first)


def test_rule_based_p_value_is_exact_hypergeometric():
    bus = _find(_clusters(), "Route 7", "transportation")
    assert bus["method"] == "rule_based" or bus["method"].startswith("scan:")
    if bus["method"] == "rule_based":
        # 6 students with transportation on Nov 10-11 (5 on Route 7 + S035), Route 7 = 7 of 28 students
        assert bus["p_value"] == pytest.approx(live.hypergeom_p(5, 28, 7, 6))
        assert bus["expected"] == pytest.approx(6 * 7 / 28, abs=1e-3)


def test_cache_bumps_on_reply_and_cluster_grows():
    v0 = client.get("/api/clusters/meta").json()["version"]
    assert "S006" not in _find(_clusters(), "Route 7", "transportation")["member_ids"]
    r = client.post("/api/absences/A009/reply", json={"text": "No pasó el autobús otra vez.", "language": "es"}).json()
    assert r["route"] == "cluster"  # S006 rides Route 7
    assert client.get("/api/clusters/meta").json()["version"] > v0
    bus = _find(_clusters(), "Route 7", "transportation")
    assert "S006" in bus["member_ids"] and "M-A009" in bus["evidence_message_ids"]


def test_no_cluster_cites_a_safety_message():
    client.post("/api/absences/A004/reply", json={"text": "Kids are bullying him at the bus stop.", "language": "en"})
    safety_msgs = {f"M-{a.id}" for a in store.absences.values() if a.classification and a.classification.safety_critical}
    assert "M-A004" in safety_msgs
    for c in _clusters():
        assert not safety_msgs & set(c["evidence_message_ids"])
        assert not {a for a in c["absence_ids"] if f"M-{a}" in safety_msgs}


def test_method_switches(monkeypatch):
    monkeypatch.setenv("CLUSTER_METHOD", "rules")
    store.touch()
    assert {c["method"] for c in _clusters()} == {"rule_based"}
    monkeypatch.setenv("CLUSTER_METHOD", "scan")
    store.touch()
    for c in _clusters():
        assert c["method"] == f"scan:{live.detector_name()}+synthetic_baseline"
        assert "synthetic quiet baseline" in c["method_note"]
    meta = client.get("/api/clusters/meta").json()
    assert meta["method"] == "scan" and meta["synthetic_baseline"]["days"] == live.SYNTHETIC_BASELINE_DAYS
    assert meta["contact_graph"] is False and set(meta["zones"]) == {"section", "route", "grade"}


def test_detector_selection(monkeypatch):
    monkeypatch.setenv("DETECTOR", "no_such_detector")
    expected = "two_stage" if hasattr(detectors, "two_stage") else (
        "graph_labels" if hasattr(detectors, "graph_labels") else "graph_only")
    assert live.detector_name() == expected
    monkeypatch.setenv("DETECTOR", "combined")
    assert live.detector_name() == "combined"
    monkeypatch.delenv("DETECTOR")
    monkeypatch.delattr(detectors, "two_stage", raising=False)
    monkeypatch.delattr(detectors, "graph_labels", raising=False)
    assert live.detector_name() == "graph_only"


@pytest.mark.parametrize("detector", ["combined", "graph_only", "content_only"])
def test_scan_runs_with_each_detector(detector):
    found, meta = live.detect_demo(store.students, list(store.absences.values()), store.settings,
                                   WEEK_START, WEEK_END, how="scan", detector=detector)
    assert meta["detector"] == detector
    for c in found:
        assert c.method == f"scan:{detector}+synthetic_baseline" and c.p_value is not None


def test_demo_run_adapter():
    run, cal = live.demo_run(store.students, store.absences.values(), WEEK_START, WEEK_END)
    assert run.n_days == live.SYNTHETIC_BASELINE_DAYS + 5 and cal.days[live.SYNTHETIC_BASELINE_DAYS] == WEEK_START
    real = [a for a in run.absences if not a.id.startswith(live.SYNTHETIC_PREFIX)]
    assert {a.id for a in real} == set(store.absences)
    assert all(a.day >= live.SYNTHETIC_BASELINE_DAYS for a in real)
    assert all(a.day < live.SYNTHETIC_BASELINE_DAYS for a in run.absences if a.id.startswith(live.SYNTHETIC_PREFIX))
    labels = {r.absence_id: r.observed_label for r in run.replies}
    assert labels["A020"] == "unknown"  # Olivia's bullying reply is hidden from the detectors
    assert labels["A001"] == "transportation"
    assert run.school.daily == [] and run.school.section["S010"] == "Room 104"
    # deterministic synthetic baseline
    run2, _ = live.demo_run(store.students, store.absences.values(), WEEK_START, WEEK_END)
    assert [(a.id, a.student, a.day) for a in run2.absences] == [(a.id, a.student, a.day) for a in run.absences]


def test_sim_dataset(monkeypatch):
    if not (Path(live.__file__).resolve().parents[2] / "data_raw" / "primary2009").exists():
        pytest.skip("Lyon graph not downloaded")
    monkeypatch.setenv("DATASET", "sim")
    store.touch()
    clusters = _clusters()
    assert clusters
    for c in clusters:
        assert c["dataset"] == "sim" and c["method"] == f"scan:{live.detector_name()}"
        assert all(s.startswith("N") for s in c["member_ids"])
    assert client.get("/api/clusters/meta").json()["students"] > 200
    r = client.post("/api/plans/draft", json={"cluster_id": clusters[0]["id"]})
    assert r.status_code == 400


# ---------------------------------------------------------------------------------------------
# eval endpoints


def test_eval_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "EVAL_DIR", tmp_path)
    assert client.get("/api/eval/ablation").status_code == 404
    assert client.get("/api/eval/retrieval").status_code == 404
    (tmp_path / "ablation.json").write_text(json.dumps({"meta": {"runs": 3}}))
    (tmp_path / "retrieval.json").write_text(json.dumps({"recall_at_5": 0.9}))
    assert client.get("/api/eval/ablation").json() == {"meta": {"runs": 3}}
    assert client.get("/api/eval/retrieval").json() == {"recall_at_5": 0.9}


# ---------------------------------------------------------------------------------------------
# F8: learning from corrections

WRONG = "La abuela está en el hospital y tuvimos que quedarnos con ella."
SIMILAR = "La abuela sigue en el hospital, nos quedamos con ella."


def test_correction_then_similar_reply_is_classified_correctly():
    # before: the keyword classifier reads "hospital" as illness
    r = client.post("/api/absences/A035/reply", json={"text": WRONG, "language": "es"}).json()
    assert r["absence"]["classification"]["barrier"] == "illness"
    assert corrections.classify(SIMILAR, "es", store.corrections)[0].barrier == Barrier.ILLNESS

    c = client.post("/api/absences/A035/correct",
                    json={"label": "caregiving or family obligation", "corrected_by": "Maria (clerk)"})
    assert c.status_code == 200
    body = c.json()
    assert body["absence"]["classification"]["barrier"] == "caregiving or family obligation"
    assert body["absence"]["classification"]["source"] == "staff_correction"
    assert body["correction"]["old_label"] == "illness" and body["correction"]["text"] == WRONG
    listed = client.get("/api/corrections").json()
    assert len(listed) == 1 and listed[0]["absence_id"] == "A035" and listed[0]["school_id"] == "demo"

    # after: the next similar reply comes back right (keyword path + nearest-correction override)
    r = client.post("/api/absences/A036/reply", json={"text": SIMILAR, "language": "es"}).json()
    cls = r["absence"]["classification"]
    assert cls["barrier"] == "caregiving or family obligation"
    assert cls["source"] == "correction_override" and cls["correction_ref"] == "A035"
    assert cls["correction_similarity"] >= corrections.threshold()
    assert cls["confidence"] < 0.8  # below the auto-accept tau: the clerk still confirms it

    # a dissimilar reply is not affected
    r = client.post("/api/absences/A037/reply", json={"text": "Tiene fiebre y vómito.", "language": "es"}).json()
    assert r["absence"]["classification"]["barrier"] == "illness"
    assert r["absence"]["classification"]["source"] == "keyword"


def test_correction_never_removes_keyword_safety():
    client.post("/api/absences/A035/reply", json={"text": "The bus never came.", "language": "en"})
    client.post("/api/absences/A035/correct", json={"label": "caregiving or family obligation", "corrected_by": "clerk"})
    r = client.post("/api/absences/A036/reply",
                    json={"text": "The bus never came and kids keep bullying him.", "language": "en"}).json()
    assert r["route"] == "escalated"
    assert r["absence"]["classification"]["barrier"] == "safety or bullying"


def test_only_counselor_downgrades_safety_and_upgrade_escalates():
    client.post("/api/absences/A035/reply", json={"text": "She is scared to go back.", "language": "en"})
    assert any(e["absence_id"] == "A035" for e in client.get("/api/escalations").json())
    r = client.post("/api/absences/A035/correct", json={"label": "illness", "corrected_by": "clerk"})
    assert r.status_code == 403
    r = client.post("/api/absences/A035/correct", json={"label": "illness", "corrected_by": "Ms. Ruiz (counselor)"})
    assert r.status_code == 200 and r.json()["escalated"] is False
    esc = next(e for e in client.get("/api/escalations").json() if e["absence_id"] == "A035")
    assert esc["status"].startswith("closed")

    # a clerk may ADD a safety label: it opens a counselor case and pauses the household
    r = client.post("/api/absences/A030/correct", json={"label": "safety or bullying", "corrected_by": "clerk"})
    assert r.status_code == 200 and r.json()["escalated"] is True
    sid = store.absences["A030"].student_id
    assert store.is_suppressed(sid)


def test_correction_requires_known_label_and_author():
    assert client.post("/api/absences/A035/correct", json={"label": "aliens", "corrected_by": "clerk"}).status_code == 422
    assert client.post("/api/absences/A035/correct", json={"label": "illness", "corrected_by": " "}).status_code == 400
    assert client.post("/api/absences/A999/correct", json={"label": "illness", "corrected_by": "clerk"}).status_code == 404


def test_similarity_ranking():
    cs = corrections.CorrectionStore()
    cs.add(corrections.make_correction("X1", Barrier.ILLNESS, Barrier.CAREGIVING, "clerk", WRONG, "es"))
    cs.add(corrections.make_correction("X2", Barrier.UNKNOWN, Barrier.TRANSPORTATION, "clerk", "Our car broke down on the highway.", "en"))
    cs.add(corrections.make_correction("X3", Barrier.UNKNOWN, Barrier.APPOINTMENT, "clerk", "Court date downtown.", "en", school_id="other"))
    top = cs.similar(SIMILAR, "demo", k=3)
    assert top[0].correction.absence_id == "X1"
    assert all(m.correction.school_id == "demo" for m in top)  # per school
    assert cs.similar("Court date downtown.", "demo") == []


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, label):
        self.label, self.prompts = label, []

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        return _FakeResponse(json.dumps({"label": self.label, "confidence": 0.9, "evidence_quote": "abuela"}))


class _FakeClient:
    def __init__(self, label):
        self.models = _FakeModels(label)


def test_gemini_path_gets_few_shot_examples_without_network():
    cs = corrections.CorrectionStore()
    cs.add(corrections.make_correction("X1", Barrier.ILLNESS, Barrier.CAREGIVING, "clerk",
                                       "Lucía: la abuela está en el hospital y tuvimos que quedarnos con ella.", "es"))
    fake = _FakeClient("caregiving or family obligation")
    model = corrections.gemini_classifier(client=fake, n_samples=1, use_cache=False)
    result, info = corrections.classify(SIMILAR, "es", cs, names=["Lucía"], model=model)
    assert result.barrier == Barrier.CAREGIVING and result.source == "gemini"
    assert info["path"] == "gemini" and info["examples"][0]["absence_id"] == "X1"
    prompt = fake.models.prompts[0]
    assert "label: caregiving or family obligation" in prompt and "<example>" in prompt
    assert SIMILAR in prompt
    assert "Lucía" not in prompt  # names in examples are redacted too


def test_gemini_failure_falls_back_to_override():
    class Boom:
        class models:
            @staticmethod
            def generate_content(**_):
                raise RuntimeError("429")

    cs = corrections.CorrectionStore()
    cs.add(corrections.make_correction("X1", Barrier.ILLNESS, Barrier.CAREGIVING, "clerk", WRONG, "es"))
    model = corrections.gemini_classifier(client=Boom(), n_samples=1, use_cache=False)
    result, info = corrections.classify(SIMILAR, "es", cs, model=model)
    assert result.barrier == Barrier.CAREGIVING and result.source == "correction_override"


# ---------------------------------------------------------------------------------------------
# F5: claim verification


def test_every_seed_plan_is_verified_and_grounded():
    for sid, s in store.students.items():
        plan = client.get(f"/api/learning-plans/{sid}").json()
        topics = {t for _, _, t, _ in PACING_GUIDE[s.grade]}
        assert plan["verification_status"] == "verified", (sid, plan["flags"])
        assert plan["verification_method"] == "deterministic"
        assert all(st["topic"] in topics for st in plan["steps"])
        assert plan["eligible"] == (plan["missed_days"] >= 2)
        assert "video" not in json.dumps(plan).lower()  # no self-paced video plans for K-5


def test_eligibility_and_home_language():
    two = client.get("/api/learning-plans/S001").json()  # Mateo, es, 2 missed days
    assert two["eligible"] and two["home_language"] == "es" and "Spanish" in two["take_home"]
    assert two["reteach_suggestion"] and two["attendance_recovery_offer"]
    one = client.get("/api/learning-plans/S011").json()
    assert not one["eligible"] and one["eligibility_note"] and not one["attendance_recovery_offer"]


def _plan(sid="S010"):
    return store.learning_plan(sid)


def _missed(sid="S010"):
    return [a.date for a in store.absences.values() if a.student_id == sid]


def test_claim_check_flags_unsupported_steps():
    plan = _plan()
    bad = plan.model_copy(deep=True)
    bad.steps[0].topic = "Intro to calculus"
    bad.steps[1].source = "Made-up source"
    status, flags, _ = claim_check.verify(bad, 2, _missed())
    assert status == "flagged"
    assert {"topic_not_in_guide", "source_mismatch"} <= {f.rule for f in flags}

    swapped = plan.model_copy(deep=True)
    swapped.steps[0].topic = PACING_GUIDE[2][0][2]  # a real topic, but not the one taught that day
    assert "topic_date_mismatch" in {f.rule for f in claim_check.verify(swapped, 2, _missed())[1]}

    wrong_day = plan.model_copy(deep=True)
    assert "date_not_missed" in {f.rule for f in claim_check.verify(wrong_day, 2, _missed()[:1])[1]}

    totals = plan.model_copy(update={"total_minutes": plan.total_minutes + 5, "missed_days": 5})
    assert {"total_mismatch", "count_mismatch"} <= {f.rule for f in claim_check.verify(totals, 2, _missed())[1]}


def test_claim_check_flags_unsupported_free_text():
    plan = _plan()
    real_topic = plan.steps[0].topic
    text = (f"Review '{real_topic}' and 'Long division' on Nov 20 in a 45 minutes session (lesson 2 of the "
            f"Grade 4 Math guide), then a self-paced video at home.")
    bad = plan.model_copy(update={"load_note": text})
    status, flags, _ = claim_check.verify(bad, 2, _missed())
    rules = {f.rule for f in flags}
    assert status == "flagged"
    assert {"unsupported_topic", "unsupported_date", "unsupported_minutes", "unsupported_lesson",
            "unsupported_course", "no_video_k5"} <= rules
    assert all(f.where == "load_note" for f in flags)
    assert not any(f.claim == real_topic for f in flags)  # the real topic is supported

    # supported claims pass: a missed date, the step minutes, the step's lesson, the course
    s0 = plan.steps[0]
    ok = (f"Missed {s0.missed_date.strftime('%b')} {s0.missed_date.day}: {s0.minutes} min on '{s0.topic}' "
          f"({s0.source.split(', ')[-1]}) in {plan.course}. The student's packet doesn't need a printer.")
    assert claim_check.verify(plan.model_copy(update={"load_note": ok}), 2, _missed())[0] == "verified"


def test_claim_check_optional_nli_with_fake_client():
    class Models:
        def generate_content(self, model, contents, config):
            assert "<statements>" in contents and "Pacing guide" in contents
            return _FakeResponse(json.dumps({"results": [{"i": 0, "supported": False, "reason": "no source"},
                                                         {"i": 1, "supported": True}]}))

    class Client:
        models = Models()

    status, flags, method = claim_check.verify(_plan(), 2, _missed(), nli=True, client=Client())
    assert method == "deterministic+gemini_nli" and status == "flagged"
    assert [f.rule for f in flags] == ["nli"] and flags[0].where == "load_note"


def test_plan_build_runs_claim_check():
    plan = learning_plan.build("S010", 2, _missed(), Barrier.ILLNESS, TODAY)
    assert plan.verification_status == "verified" and plan.flags == []


# ---------------------------------------------------------------------------------------------
# Attendance Recovery rule (PRD 1 F5)


def _recovered():
    return client.get("/api/revenue").json()["days_recovered_to_date"]


def test_only_supervised_recovery_for_eligible_plans_counts():
    base = _recovered()
    one_day = client.post("/api/learning-plans/S011/complete", json={"supervised_by": "Ms. Rivera"}).json()
    assert one_day["counted"] is False and one_day["days_recovered"] == 0 and "eligible" in one_day["reason"]
    take_home = client.post("/api/learning-plans/S010/complete",
                            json={"session_type": "take_home", "supervised_by": "Ms. Rivera"}).json()
    assert take_home["counted"] is False
    unsupervised = client.post("/api/learning-plans/S010/complete").json()
    assert unsupervised["counted"] is False and "supervised" in unsupervised["reason"]
    assert _recovered() == base

    ok = client.post("/api/learning-plans/S010/complete", json={"supervised_by": "Ms. Rivera"}).json()
    assert ok["counted"] is True and ok["days_recovered"] == 2
    assert _recovered() == base + 2
    client.post("/api/learning-plans/S010/complete", json={"supervised_by": "Ms. Rivera"})
    assert _recovered() == base + 2  # idempotent per student
    log = client.get("/api/recovery-log").json()
    assert [e["counted"] for e in log] == [False, False, False, True, True]
