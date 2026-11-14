# RollCall Whisper fallback (self-hosted)

A small FastAPI service around [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2,
int8 on CPU). The backend (`backend/app/services/voice.py`) calls it only when Gemini's native audio
path fails, times out, or cannot handle the language (PRD 2 §3.1). PRD 3 §3 requires Whisper to be
self-hosted, so voice notes never go to a third-party speech API.

## API

| Method | Path | Auth | Body | Response |
|---|---|---|---|---|
| GET | `/health` | none | | `{"status":"ok","model":"small","model_loaded":true}` |
| POST | `/transcribe` | `Authorization: Bearer $WHISPER_TOKEN` | multipart: `file` (audio), optional `language` (`es`, `es-MX`, `vi`, ...) | `{text, language, language_probability, duration_s, segments_avg_logprob, no_speech_prob, compression_ratio, model, processing_ms, language_hint_used}` |

Errors: 401 missing or wrong token, 413 upload over 10 MB or audio over 120 s, 400 empty, undecodable or
malformed multipart, 415 not multipart, 503 model not loaded. A `language` hint Whisper does not know
(for example `mix` Mixtec, `trs` Triqui) is ignored and the language is auto-detected. The backend sends
those notes to an interpreter before it ever calls this service.

The backend treats a transcript as low-confidence (and routes the note to a human interpreter) when the
text is empty, `language_probability < 0.5` without a hint, `segments_avg_logprob < -1.0`,
`compression_ratio > 2.4` (repetition, Whisper's typical hallucination), or `no_speech_prob > 0.6`.

## Privacy and security

* **No audio on disk.** The body is read into memory with a hard cap. The multipart parser keeps the
  file part in memory (Starlette normally spools parts over 1 MB to a temp file), and decoding reads
  from a `BytesIO`. The container runs with a read-only root filesystem and a tmpfs `/tmp`.
* **No transcript text in logs.** The service logs bytes, duration, language and timing only. Caddy
  has no access log configured.
* **Auth is required.** The app refuses to start unless `WHISPER_TOKEN` is set (at least 16 characters).
  Tokens are compared in constant time. `/docs` and `/openapi.json` are disabled.
* **Only Caddy is exposed.** The whisper container publishes no ports. Caddy terminates HTTPS
  (automatic Let's Encrypt), caps request bodies at 11 MB, and proxies to it.
* The backend refuses to send audio to a non-HTTPS `WHISPER_URL` (plain `http://localhost` is allowed
  for local testing only).

## Configuration (environment)

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_TOKEN` | (required) | `openssl rand -hex 32` |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, ... (baked into the image at build time) |
| `WHISPER_COMPUTE_TYPE` | `int8` | |
| `WHISPER_CPU_THREADS` | `0` (all cores) | |
| `WHISPER_BEAM_SIZE` | `1` | greedy decoding; `5` is a little more accurate and about 2x slower |
| `WHISPER_TEMPERATURES` | `0.0,0.4` | shorter than faster-whisper's six-step ladder, which took 30 s on a 3 s garbled clip |
| `WHISPER_VAD` | `1` | Silero VAD skips silence |
| `WHISPER_MAX_BYTES` / `WHISPER_MAX_SECONDS` | `10485760` / `120` | |
| `WHISPER_CONCURRENCY` | `1` | transcriptions at once (more requests queue up) |

## Measured latency (local, 2026-11-14)

Laptop CPU: AMD Ryzen 9 5900HS (8 cores / 16 threads), int8, beam 1, VAD on. The test clips are
espeak-ng speech made locally. Times are server processing per request, warm model.

| Model | 7.4 s English note | 5.9 s Spanish note | Load time (first start) | Image size |
|---|---|---|---|---|
| tiny | 0.6-0.7 s | 0.6 s | ~7 s (incl. download) | 683 MB (podman build) |
| small | 2.9-3.3 s | 2.9-3.2 s | ~26 s (incl. download) | ~1.1 GB (est.) |

* 4 threads were as fast as 16, so a 4 vCPU server is enough. On a 4 vCPU Vultr instance, expect
  roughly **4-6 s for a short note with `small`** and about 1 s with `tiny`. Allow up to ~45 s for a
  2-minute note (the backend's `WHISPER_TIMEOUT_S` default).
* Quality: `small` transcribed both espeak clips almost word for word. `tiny` garbled the English clip
  ("Pana Fimer") and returned nothing for the Spanish one without a hint. **Use `small`.** Neither model
  could read the robotic Vietnamese clip. Both returned low-confidence output, which the backend routes
  to an interpreter as designed. Real human Vietnamese speech should do much better.

## Deploy on Vultr (Ubuntu 24.04, CPU)

1. **Create the server.** In the Vultr console go to Deploy, then Cloud Compute. Pick a High Performance
   or CPU Optimized plan with **4 vCPU / 8 GB**, Ubuntu 24.04 LTS x64, a US region near the backend, and
   your SSH key. Note the public IPv4, for example `203.0.113.7`.
2. **DNS name for HTTPS.** Either add an A record such as `whisper.<your-domain>` pointing to the IP, or
   use the zero-setup name `203-0-113-7.sslip.io` (it resolves to that IP and works with Let's Encrypt).
3. **Vultr firewall** (Network, then Firewall): create a group that allows inbound TCP 22 from your own
   IP only, plus TCP 80, TCP 443 and UDP 443 from anywhere. Attach it to the instance.
4. **On the server:**
   ```bash
   ssh root@203.0.113.7
   apt-get update && apt-get -y upgrade
   curl -fsSL https://get.docker.com | sh           # Docker Engine + compose plugin
   ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443 && ufw --force enable
   git clone <repo-url> rollcall && cd rollcall/whisper_server   # or: scp -r whisper_server root@IP:
   cp .env.example .env && chmod 600 .env
   sed -i "s/^WHISPER_TOKEN=.*/WHISPER_TOKEN=$(openssl rand -hex 32)/" .env
   sed -i "s/^WHISPER_DOMAIN=.*/WHISPER_DOMAIN=203-0-113-7.sslip.io/" .env
   docker compose up -d --build        # first build ~3-5 min (downloads the model into the image)
   docker compose logs -f whisper      # wait for "model small loaded"
   ```
   Docker-published ports bypass ufw. That is why only Caddy publishes ports (80/443) and the whisper
   container publishes none.
5. **Check it from your laptop:**
   ```bash
   curl https://203-0-113-7.sslip.io/health
   curl -H "Authorization: Bearer <token>" -F file=@note.ogg -F language=es \
        https://203-0-113-7.sslip.io/transcribe
   curl -o /dev/null -w "%{http_code}\n" -F file=@note.ogg https://203-0-113-7.sslip.io/transcribe  # 401
   ```
6. **Point the backend at it.** Add these lines to `backend/.env` (git-ignored):
   ```
   WHISPER_URL=https://203-0-113-7.sslip.io
   WHISPER_TOKEN=<the same token>
   ```

Update: `git pull && docker compose up -d --build`. Rotate the token: edit `.env`, run
`docker compose up -d`, then update `backend/.env`.

## Cost: delete the server after the hackathon

Third-party listings put Vultr High Performance 4 vCPU / 8 GB at about **$48/month (~$0.07/hour)** and
CPU Optimized 4 vCPU / 8 GB at about **$80/month (~$0.12/hour)**. Check vultr.com/pricing before you
deploy. A weekend costs a few dollars. **Vultr keeps billing a stopped instance**, so when the hackathon
is over, **destroy** it (Server, then Settings, then Destroy) instead of only powering it off. Remove the
DNS record and the firewall group too.

## Local development

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q test_app.py                       # fake model, no download
WHISPER_TOKEN=local-dev-token-123456 WHISPER_MODEL=tiny \
  .venv/bin/uvicorn app:create_app --factory --port 8000
```
Or with a container: `docker build --build-arg WHISPER_MODEL=tiny -t rollcall-whisper .` and
`docker run --read-only --tmpfs /tmp -e WHISPER_TOKEN=... -p 127.0.0.1:8000:8000 rollcall-whisper`.
