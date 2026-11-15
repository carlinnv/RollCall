# Deploying Why Absent (Docker)

One server runs two containers: the FastAPI backend, and Caddy, which serves the website,
proxies `/api` to the backend and gets an HTTPS certificate automatically.

## On a fresh Debian or Ubuntu server

```bash
curl -fsSL https://get.docker.com | sh
git clone https://github.com/carlinnv/RollCall.git /opt/whyabsent && cd /opt/whyabsent
cp .env.example .env && chmod 600 .env    # then edit .env (below)
docker compose up -d --build
```

## .env on the server (never commit it)

| Setting | Value |
|---|---|
| `SITE_ADDRESS` | your domain, e.g. `whyabsent.example.com` (DNS A record -> server IP, not proxied). Without a domain use `<ip-with-dashes>.sslip.io`, or `:80` for plain HTTP |
| `GEMINI_API_KEY` | Gemini key. Leave empty to run on the keyword rules |
| `DEMO_STORY` | `live` for the demo (Route 7 cluster appears on the live reply) |
| `CLUSTER_METHOD` | `rules` for clean demo clusters |
| `PHOTON_*` | Leave empty on the server unless this is the only place Photon runs. Two running copies would both answer each iMessage |
| `TIGER_DATABASE_URL` | Optional. TigerData Postgres connection string (`sslmode=require`). Saves the app's data so it survives restarts, and powers the daily-absences timeline. Leave empty to keep everything in memory |

If you set `TIGER_DATABASE_URL`, create its tables once after the containers are up:

```bash
docker compose exec backend python scripts/tiger_setup.py
docker compose restart backend
```

Ports 80 and 443 must be open. Update with `git pull && docker compose up -d --build`.
Without `TIGER_DATABASE_URL`, state is in memory and a restart reloads the demo data. With it, data is saved in TigerData;
do not point a local backend at the same database while the server is running, or the two will overwrite each other.

The site has no login. Anyone with the link can use the demo, including the Gemini-backed
pages, so take the server down after the event (`docker compose down`, then destroy the server).
