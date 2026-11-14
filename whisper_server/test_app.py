"""Tests for the Whisper server with a fake model (no model download).

    WHISPER_TOKEN is not needed in the environment; each test builds its own Settings.
    Run: pip install -r requirements.txt pytest && pytest -q test_app.py
"""
import io
import logging
import math
import struct
import wave
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as server

TOKEN = "test-token-0123456789abcdef"
SECRET_WORDS = "my son has a fever"


def make_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
                          for i in range(int(seconds * rate)))
        w.writeframes(frames)
    return buf.getvalue()


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, **kwargs):
        self.calls.append({"samples": len(audio), "language": language, **kwargs})
        segments = [SimpleNamespace(text=" My son has a fever", start=0.0, end=0.6, avg_logprob=-0.2, no_speech_prob=0.01,
                                    compression_ratio=1.1),
                    SimpleNamespace(text=" today.", start=0.6, end=1.0, avg_logprob=-0.4, no_speech_prob=0.02,
                                    compression_ratio=1.3)]
        info = SimpleNamespace(language=language or "en", language_probability=0.97)
        return iter(segments), info


def settings(**overrides):
    s = server.Settings()
    s.token = TOKEN
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


@pytest.fixture
def fake():
    return FakeModel()


@pytest.fixture
def client(fake):
    with TestClient(server.create_app(model=fake, settings=settings())) as c:
        yield c


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def post(client, audio, headers=None, language=None, mime="audio/wav"):
    data = {"language": language} if language else None
    return client.post("/transcribe", files={"file": ("note.wav", audio, mime)}, data=data,
                       headers=auth() if headers is None else headers)


def test_refuses_to_start_without_token(monkeypatch):
    monkeypatch.delenv("WHISPER_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        server.create_app(model=FakeModel())
    with pytest.raises(RuntimeError):
        server.create_app(model=FakeModel(), settings=settings(token="short"))


def test_health_needs_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "small", "model_loaded": True}


@pytest.mark.parametrize("headers", [{}, auth("wrong-token-0123456789"), {"Authorization": TOKEN}])
def test_rejects_missing_or_bad_token(client, fake, headers):
    r = post(client, make_wav(), headers=headers)
    assert r.status_code == 401
    assert fake.calls == []


def test_transcribe_ok(client, fake):
    r = post(client, make_wav(1.0), language="es-MX")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "My son has a fever today."
    assert body["language"] == "es"
    assert body["language_probability"] == 0.97
    assert body["duration_s"] == 1.0
    assert body["segments_avg_logprob"] == pytest.approx((-0.2 * 0.6 + -0.4 * 0.4) / 1.0, abs=1e-3)
    assert set(body) >= {"text", "language", "language_probability", "duration_s", "segments_avg_logprob"}
    assert fake.calls[0]["language"] == "es"  # BCP-47 hint reduced to the Whisper code
    assert fake.calls[0]["samples"] == 16000
    assert body["compression_ratio"] == 1.3
    assert fake.calls[0]["temperature"] == (0.0, 0.4)


def test_unknown_hint_means_auto_detect(client, fake):
    assert post(client, make_wav(), language="mix").status_code == 200  # Mixtec: not a Whisper language
    assert fake.calls[0]["language"] is None


def test_hint_normalization():
    known = frozenset({"es", "tl", "no", "zh"})
    assert server.normalize_hint("es_419", known) == "es"
    assert server.normalize_hint("fil-PH", known) == "tl"
    assert server.normalize_hint("auto", known) is None
    assert server.normalize_hint("trs", known) is None


def test_too_large_upload(fake):
    with TestClient(server.create_app(model=fake, settings=settings(max_bytes=1000))) as c:
        r = post(c, make_wav(1.0))
    assert r.status_code == 413
    assert fake.calls == []


def test_too_long_audio(fake):
    with TestClient(server.create_app(model=fake, settings=settings(max_seconds=1))) as c:
        r = post(c, make_wav(2.0))
    assert r.status_code == 413
    assert fake.calls == []


def test_bad_audio_and_bad_requests(client, fake):
    assert post(client, b"definitely not audio" * 10).status_code == 400
    assert post(client, b"").status_code == 400
    r = client.post("/transcribe", content=make_wav(), headers={**auth(), "Content-Type": "audio/wav"})
    assert r.status_code == 415
    r = client.post("/transcribe", files={"other": ("x", b"123")}, headers=auth())
    assert r.status_code == 400
    assert fake.calls == []


def test_upload_stays_in_memory_and_transcript_not_logged(client, monkeypatch, caplog):
    created = []
    import tempfile
    real_spooled = tempfile.SpooledTemporaryFile

    class Spy(real_spooled):
        def rollover(self):  # called when a spooled file would move to disk
            created.append(True)
            return super().rollover()

    monkeypatch.setattr("starlette.formparsers.SpooledTemporaryFile", Spy)
    caplog.set_level(logging.DEBUG)
    big = make_wav(40.0)  # ~1.3 MB, above Starlette's default 1 MB spool size
    assert len(big) > 1024 * 1024
    r = post(client, big)
    assert r.status_code == 200, r.text
    assert created == [], "upload was spooled to disk"
    assert SECRET_WORDS not in caplog.text.lower()
