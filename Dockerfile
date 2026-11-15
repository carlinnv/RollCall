# Why Absent: one Dockerfile, two images (docker compose builds both).
#   backend -> FastAPI on :8000
#   edge    -> Caddy serving the built website and proxying /api to the backend (automatic HTTPS)

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS backend
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
# Public SocioPatterns contact graphs (CC BY-NC-SA), used by DATASET=sim and the counselor assistant.
RUN sh scripts/fetch_graphs.sh
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# One worker: the store is in memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM caddy:2 AS edge
COPY --from=web /web/dist /srv
COPY deploy/Caddyfile /etc/caddy/Caddyfile
