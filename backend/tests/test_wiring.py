"""PRD 1 <-> PRD 2 wiring: live demo story, voice endpoint, assistant endpoint, eval files.
No network: the voice pipeline is replaced with a fake and the assistant runs without a key."""
import base64

import pytest
from fastapi.testclient import TestClient

from app import main
from app.models import Barrier, Classification, FamilyReply
from app.services import voice
from app.store import Store


@pytest.fixture
def client():
    main.store.reset()
    yield TestClient(main.app)
    main.store.reset()


def test_live_story_cluster_appears_after_the_demo_reply(monkeypatch):
    monkeypatch.setenv("DEMO_STORY", "live")
    s = Store()
    assert not [c for c in s.clusters() if c.link_value == "Route 7"]
    route = s.record_reply("A004", FamilyReply(text="Otra vez no pasó el autobús de la ruta 7.", channel="text",
                                               language="es", received_at="2026-11-14T09:00:00"))
    assert route == "cluster"
    r7 = [c for c in s.clusters() if c.link_value == "Route 7"]
    assert r7 and r7[0].barrier == Barrier.TRANSPORTATION and len(r7[0].student_ids) >= 3


def test_seed_reset_never_calls_a_model(monkeypatch):
    monkeypatch.setenv("CLASSIFIER", "gemini")
    from app.services import corrections

    def boom(*a, **k):
        raise AssertionError("seed load must not call the model")

    monkeypatch.setattr(corrections, "gemini_classifier", boom)
    monkeypatch.setattr(corrections, "use_gemini", lambda: True)
    Store()  # would raise if any seed reply went to the model


def _fake_voice(route="auto", transcript="El autobús no pasó.", barrier=Barrier.TRANSPORTATION, visibility="staff"):
    def fake(audio, mime, language_hint=None, names=None):
        meta = voice.VoiceMeta(route=route, needs_review=route != "auto")
        meta.transcript = transcript
        meta.transcript_source = "gemini_audio"
        meta.language = "es"
        meta.transcript_visibility = visibility
        return Classification(barrier=barrier, confidence=0.9, evidence=transcript or ""), meta
    return fake


def test_voice_endpoint_records_transcript_reply(client, monkeypatch):
    monkeypatch.setattr(voice, "process_voice_note", _fake_voice())
    r = client.post("/api/absences/A004/voice", json={"audio_b64": base64.b64encode(b"fake").decode(), "mime_type": "audio/mpeg"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["absence"]["reply"]["channel"] == "voice"
    assert body["absence"]["classification"]["barrier"] == "transportation"


def test_voice_keyword_safety_overrides_model_label(client, monkeypatch):
    monkeypatch.setattr(voice, "process_voice_note",
                        _fake_voice(transcript="Le pegan en el recreo y tiene miedo.", barrier=Barrier.ILLNESS))
    r = client.post("/api/absences/A004/voice", json={"audio_b64": base64.b64encode(b"x").decode()})
    body = r.json()
    assert body["route"] == "escalated"
    assert body["absence"]["classification"]["safety_critical"] is True


def test_voice_safety_transcript_hidden_from_clerk(client, monkeypatch):
    monkeypatch.setattr(voice, "process_voice_note", _fake_voice(
        route="counselor", transcript="Le pegan en el recreo.", barrier=Barrier.SAFETY, visibility="counselor_only"))
    body = client.post("/api/absences/A004/voice", json={"audio_b64": base64.b64encode(b"x").decode()}).json()
    assert body["voice"]["transcript"] is None


def test_voice_rejects_bad_audio(client):
    assert client.post("/api/absences/A004/voice", json={"audio_b64": "not base64!!"}).status_code == 400


def test_assistant_needs_a_key(client, monkeypatch):
    from app.services import gemini_classifier
    monkeypatch.setattr(gemini_classifier, "has_api_key", lambda: False)
    assert client.post("/api/assistant/ask", json={"question": "Why are students absent?"}).status_code == 503
    assert client.post("/api/assistant/ask", json={"question": "  "}).status_code == 400


def test_eval_files_served(client):
    for name in ("ablation", "retrieval", "scaling"):
        assert client.get(f"/api/eval/{name}").status_code == 200
