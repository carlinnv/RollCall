# Self-hosted Whisper fallback for RollCall. CPU only, int8.
FROM python:3.11-slim

ARG WHISPER_MODEL=small
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    WHISPER_MODEL=${WHISPER_MODEL} \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image so the container starts offline and never downloads at runtime.
RUN python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8')" \
    && chmod -R a+rX /models
ENV HF_HUB_OFFLINE=1

COPY app.py .
RUN useradd --system --uid 10001 whisper
USER whisper

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if b'\"model_loaded\":true' in urllib.request.urlopen('http://127.0.0.1:8000/health').read() else 1)"
# One worker: the model is loaded once and transcriptions are serialized (WHISPER_CONCURRENCY).
CMD ["uvicorn", "app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
