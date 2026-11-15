"""PRD 1 <-> PRD 2 wiring: live demo story, assistant endpoint, eval files.
No network: the assistant runs without a key."""
import pytest
from fastapi.testclient import TestClient

from app import main
from app.models import Barrier, FamilyReply
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
    from app.services import classifier

    def boom(*a, **k):
        raise AssertionError("seed load must not call the model")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(classifier, "_generate", boom)
    Store()  # would raise (or log a fallback) if any seed reply went to the model
    assert all(a.classification.source != "keywords_fallback" for a in Store().absences.values() if a.classification)

def test_assistant_needs_a_key(client, monkeypatch):
    from app.services import gemini_classifier
    monkeypatch.setattr(gemini_classifier, "has_api_key", lambda: False)
    assert client.post("/api/assistant/ask", json={"question": "Why are students absent?"}).status_code == 503
    assert client.post("/api/assistant/ask", json={"question": "  "}).status_code == 400


def test_eval_files_served(client):
    for name in ("ablation", "retrieval", "scaling"):
        assert client.get(f"/api/eval/{name}").status_code == 200
