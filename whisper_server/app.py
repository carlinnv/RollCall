"""Self-hosted Whisper fallback for RollCall voice notes (PRD 2 section 3.1, PRD 3 section 3).

Run:  WHISPER_TOKEN=... uvicorn app:create_app --factory --host 0.0.0.0 --port 8000

Endpoints
  GET  /health      -> {"status": "ok", "model": "small", "model_loaded": true}   (no auth)
  POST /transcribe  -> {"text", "language", "language_probability", "duration_s",
                        "segments_avg_logprob", "no_speech_prob", "compression_ratio", "model",
                        "processing_ms", "language_hint_used"}
       multipart/form-data: file=<audio>, language=<optional hint, e.g. "es" or "es-MX">
       header: Authorization: Bearer $WHISPER_TOKEN

Privacy rules (PRD 3: "Whisper, if used, is self-hosted"):
  * Audio never touches disk: the body is read into memory with a hard size cap, the
    multipart parser keeps the file part in memory, and decoding works on a BytesIO.
    (The Docker image also runs with a read-only root and a tmpfs /tmp.)
  * Transcript text is never logged; logs carry only sizes, durations, language and timing.
  * WHISPER_TOKEN is required: the app refuses to start without one (min 16 chars).

Environment
  WHISPER_TOKEN         required bearer token
  WHISPER_MODEL         faster-whisper model size or path (default "small")
  WHISPER_DEVICE        default "cpu"
  WHISPER_COMPUTE_TYPE  default "int8"
  WHISPER_CPU_THREADS   default 0 (= all cores)
  WHISPER_BEAM_SIZE     default 1 (greedy; fastest on CPU, 5 is the faster-whisper default)
  WHISPER_VAD           default "1" (skip silence with the bundled Silero VAD)
  WHISPER_TEMPERATURES  default "0.0,0.4" (fallback ladder; faster-whisper's six-step default can take
                        10x longer on garbled audio, which is exactly the audio that goes to a human anyway)
  WHISPER_MAX_BYTES     default 10485760 (10 MB)
  WHISPER_MAX_SECONDS   default 120
  WHISPER_CONCURRENCY   default 1 (transcriptions at once; more just fight for the CPU)
"""
from __future__ import annotations

import asyncio
import hmac
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartParser

log = logging.getLogger("whisper_server")
SAMPLE_RATE = 16000
MULTIPART_OVERHEAD = 64 * 1024  # boundary lines and the small language field


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


class Settings:
    def __init__(self) -> None:
        self.token = os.environ.get("WHISPER_TOKEN", "").strip()
        self.model = os.environ.get("WHISPER_MODEL", "small").strip() or "small"
        self.device = os.environ.get("WHISPER_DEVICE", "cpu")
        self.compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
        self.cpu_threads = _env_int("WHISPER_CPU_THREADS", 0)
        self.beam_size = max(1, _env_int("WHISPER_BEAM_SIZE", 1))
        temps = os.environ.get("WHISPER_TEMPERATURES", "0.0,0.4")
        self.temperatures = tuple(float(t) for t in temps.split(",") if t.strip()) or (0.0,)
        self.vad = os.environ.get("WHISPER_VAD", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.max_bytes = _env_int("WHISPER_MAX_BYTES", 10 * 1024 * 1024)
        self.max_seconds = _env_int("WHISPER_MAX_SECONDS", 120)
        self.concurrency = max(1, _env_int("WHISPER_CONCURRENCY", 1))


def _whisper_languages() -> frozenset:
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        return frozenset(_LANGUAGE_CODES)
    except Exception:  # noqa: BLE001 - only used to validate hints
        return frozenset()


# Whisper's own codes for a few languages differ from ISO 639-1 / BCP-47.
_HINT_ALIASES = {"fil": "tl", "nb": "no", "iw": "he", "jv": "jw", "cmn": "zh"}


def normalize_hint(hint: Optional[str], known: frozenset) -> Optional[str]:
    """'es-MX' -> 'es'. Unknown or empty hints -> None (Whisper auto-detects)."""
    if not hint:
        return None
    base = hint.strip().lower().replace("_", "-").split("-")[0]
    base = _HINT_ALIASES.get(base, base)
    if base in {"", "auto", "und", "unknown"}:
        return None
    return base if (not known or base in known) else None


class _InMemoryMultiPartParser(MultiPartParser):
    """Starlette spools file parts over 1 MB to a temp file; keep the whole upload in memory."""

    def __init__(self, *args: Any, spool_bytes: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.spool_max_size = spool_bytes


def load_model(settings: Settings) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(settings.model, device=settings.device, compute_type=settings.compute_type,
                        cpu_threads=settings.cpu_threads)


def create_app(model: Any = None, settings: Optional[Settings] = None) -> FastAPI:
    """App factory. `model` may be any object with faster-whisper's `.transcribe()`; tests pass a fake."""
    settings = settings or Settings()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if len(settings.token) < 16:
        raise RuntimeError("WHISPER_TOKEN must be set (at least 16 characters); refusing to start without auth")
    languages = _whisper_languages()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if app.state.model is None:
            started = time.monotonic()
            app.state.model = await run_in_threadpool(load_model, settings)
            log.info("model %s loaded in %.1f s", settings.model, time.monotonic() - started)
        yield

    app = FastAPI(title="RollCall Whisper", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.model = model
    app.state.settings = settings
    gate = asyncio.Semaphore(settings.concurrency)

    def require_token(request: Request) -> None:
        header = request.headers.get("authorization", "")
        scheme, _, given = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(given.strip().encode(), settings.token.encode()):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token",
                                headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "model": settings.model, "model_loaded": app.state.model is not None}

    async def read_body_capped(request: Request) -> bytes:
        limit = settings.max_bytes + MULTIPART_OVERHEAD
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise HTTPException(status_code=413, detail=f"upload larger than {settings.max_bytes} bytes")
        chunks, size = [], 0
        async for chunk in request.stream():  # also covers chunked uploads with no Content-Length
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail=f"upload larger than {settings.max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    async def parse_upload(request: Request, body: bytes) -> tuple[bytes, Optional[str]]:
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            raise HTTPException(status_code=415, detail="send multipart/form-data with a 'file' part")

        async def replay() -> AsyncGenerator[bytes, None]:
            yield body

        parser = _InMemoryMultiPartParser(request.headers, replay(), max_files=1, max_fields=4,
                                          spool_bytes=settings.max_bytes + MULTIPART_OVERHEAD)
        try:
            form = await parser.parse()
        except Exception as exc:  # noqa: BLE001 - malformed multipart
            raise HTTPException(status_code=400, detail="malformed multipart body") from exc
        try:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(status_code=400, detail="missing 'file' part")
            audio = await upload.read()
            hint = form.get("language")
            return audio, (hint if isinstance(hint, str) else None)
        finally:
            await form.close()

    def run_transcription(audio: bytes, hint: Optional[str]) -> Dict[str, Any]:
        from faster_whisper.audio import decode_audio

        try:
            samples = decode_audio(io.BytesIO(audio), sampling_rate=SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001 - PyAV raises many types for bad input
            raise HTTPException(status_code=400, detail="could not decode audio") from exc
        duration = len(samples) / SAMPLE_RATE
        if duration > settings.max_seconds:
            raise HTTPException(status_code=413, detail=f"audio longer than {settings.max_seconds} s")
        if duration < 0.1:
            raise HTTPException(status_code=400, detail="audio is empty")
        segments, info = app.state.model.transcribe(
            samples, language=hint, beam_size=settings.beam_size, vad_filter=settings.vad,
            condition_on_previous_text=False, temperature=settings.temperatures)
        segments = list(segments)  # the generator does the decoding work
        texts = [s.text.strip() for s in segments if s.text.strip()]
        # Duration-weighted mean of per-segment avg_logprob (Whisper's usual confidence proxy).
        weights = [max(float(s.end) - float(s.start), 0.01) for s in segments]
        avg_logprob = (sum(float(s.avg_logprob) * w for s, w in zip(segments, weights)) / sum(weights)
                       if segments else None)
        no_speech = max((float(s.no_speech_prob) for s in segments), default=None)
        # High compression ratio = repeated tokens, Whisper's typical hallucination on unclear audio.
        compression = max((float(s.compression_ratio) for s in segments), default=None)
        return {
            "text": " ".join(texts),
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "duration_s": round(duration, 2),
            "segments_avg_logprob": round(avg_logprob, 4) if avg_logprob is not None else None,
            "no_speech_prob": round(no_speech, 4) if no_speech is not None else None,
            "compression_ratio": round(compression, 3) if compression is not None else None,
            "model": settings.model,
        }

    @app.post("/transcribe", dependencies=[Depends(require_token)])
    async def transcribe(request: Request) -> JSONResponse:
        started = time.monotonic()
        body = await read_body_capped(request)
        audio, raw_hint = await parse_upload(request, body)
        del body
        if not audio:
            raise HTTPException(status_code=400, detail="audio is empty")
        if len(audio) > settings.max_bytes:
            raise HTTPException(status_code=413, detail=f"upload larger than {settings.max_bytes} bytes")
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        hint = normalize_hint(raw_hint, languages)
        async with gate:
            result = await run_in_threadpool(run_transcription, audio, hint)
        result["processing_ms"] = int((time.monotonic() - started) * 1000)
        result["language_hint_used"] = hint
        # Metadata only: never the transcript text.
        log.info("transcribed bytes=%d duration_s=%s language=%s p=%s ms=%d", len(audio), result["duration_s"],
                 result["language"], result["language_probability"], result["processing_ms"])
        return JSONResponse(result)

    return app
