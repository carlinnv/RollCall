"""DEMO_MODE: the scripted demo runs from recorded model answers, on the replay channel, with no network."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import classifier, demo, oncall, outreach, photon_channel
from app.store import store

client = TestClient(app)


@pytest.fixture
def demo_mode(monkeypatch, tmp_path):
    """Demo mode on, with no Gemini key and the network path made to fail loudly."""
    for name in ("DEMO_MODE", "DEMO_STORY", "QUIET_HOURS", "PHOTON_TEST_SENDER", "PHOTON_DEMO_STUDENT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    demo.apply_settings()  # DEMO_STORY=live, QUIET_HOURS=off (monkeypatch restores both)

    def no_network(_redacted):
        raise AssertionError("demo mode called the model")

    monkeypatch.setattr(classifier, "_ask_gemini", no_network)
    monkeypatch.setattr(store, "sender", demo.replay_sender)
    monkeypatch.setattr(store, "staff_sender", demo.replay_staff_sender)
    monkeypatch.setattr(store, "roster", oncall.Roster(tmp_path / "oncall.json"))
    photon_channel.contacts.clear()
    outreach.opted_out.clear()
    store.reset()
    yield
    monkeypatch.undo()
    store.reset()


def route7_cluster():
    return [c for c in store.clusters() if c.link_value == "Route 7"]


def test_script_runs_in_order_from_recordings(demo_mode):
    assert classifier.status()["recorded"] is True
    script = demo.script()["steps"]
    assert not route7_cluster()  # the live story: no cluster at 9am

    for step in script:
        out = client.post("/api/demo/next").json()
        assert out["id"] == step["id"] and out["done"]
        if step["kind"] == "view":  # only opens a page
            assert out["outcome"] is None
            continue
        expected = step.get("expect", {}).get("barrier") or step.get("label")
        assert out["outcome"]["barrier"] == expected
        if step["id"] == "route7":
            assert out["outcome"]["source"] == "gemini"
            assert route7_cluster(), "the Route 7 cluster lights up on the first reply"

    state = client.get("/api/demo").json()
    assert state["next"] is None and all(s["done"] for s in state["steps"])
    assert client.post("/api/demo/next").status_code == 409  # finished


def play_through(step_id):
    """Play steps up to and including `step_id`; returns that step's state."""
    while True:
        out = client.post("/api/demo/next").json()
        if out["id"] == step_id:
            return out


def test_correction_beat_cites_the_correction(demo_mode):
    play_through("custody2")
    steps = {s["id"]: s for s in client.get("/api/demo").json()["steps"]}
    first, second = steps["custody"]["outcome"]["absence_id"], steps["custody2"]["outcome"]["absence_id"]
    label = store.absences[second].classification
    assert label.barrier.value == "caregiving or family obligation" and label.correction_ref == first


def test_correcting_by_hand_counts_as_the_step(demo_mode):
    play_through("custody")
    absence_id = next(m.absence_id for m in store.messages if m.external_id == "demo-custody")
    r = client.post(f"/api/absences/{absence_id}/correct",
                    json={"label": "caregiving or family obligation", "corrected_by": "clerk"})
    assert r.status_code == 200
    assert next(s for s in client.get("/api/demo").json()["steps"] if s["id"] == "fix")["done"]
    assert client.post("/api/demo/next").json()["id"] == "custody2"  # skips the correction step


def test_replies_use_the_replay_channel(demo_mode):
    client.post("/api/demo/next")
    logged = [m for m in store.messages if m.channel == "replay"]
    assert [m.direction for m in logged] == ["inbound", "outbound"]  # the family's text, then the automatic reply
    assert logged[1].kind == "ack" and logged[1].status == "sent"


def test_safety_step_escalates_and_sends_no_reply(demo_mode):
    out = play_through("safety")
    assert out["outcome"]["escalated"]
    absence_id = out["outcome"]["absence_id"]
    assert not [m for m in store.messages if m.absence_id == absence_id and m.kind == "ack"]


def test_off_script_text_gets_keyword_label_without_network(demo_mode):
    label = classifier.classify("The bus did not come this morning.")
    assert label.barrier.value == "transportation" and label.source == "keywords"


def test_reset_rewinds_the_script(demo_mode):
    play_through("route7-cluster")  # a reply step and a view step
    client.post("/api/reset")
    assert client.get("/api/demo").json()["next"] == 0


def test_photon_stays_off_in_demo_mode(demo_mode, monkeypatch):
    for name in photon_channel.REQUIRED:
        monkeypatch.setenv(name, "x")
    assert photon_channel.enabled() is False


def test_next_refuses_outside_demo_mode():
    assert client.post("/api/demo/next").status_code == 409
    assert client.get("/api/demo").json()["enabled"] is False
