# RollCall — Why Absent

Princeton Hacks Fall 2026 Project: an app that helps school districts track student absences and come up with solutions.

**Why Absent** finds out why students miss school, traces clusters of absences to a root cause, and turns each cause into an action the school can take, priced in dollars per week. See PRD 1 (Product, UX & Business Impact) for the full spec.

> Synthetic data only. No real student records are used anywhere in this project.

## Stack

| Part | Tech | Location |
| --- | --- | --- |
| API | Python 3.9+, FastAPI, Pydantic v2, in-memory store | `backend/` |
| UI | React 18, TypeScript, Vite, React Router | `frontend/` |

## Run it

```bash
# API on :8000
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000 --reload
```

```bash
# UI on :5173 (proxies /api to :8000)
cd frontend
npm install
npm run dev
```

```bash
# Tests (PRD acceptance criteria)
cd backend && .venv/bin/pytest -q
```

### Optional: TigerData timeline

Mirrors absences into a TigerData (Timescale) hypertable. A continuous aggregate feeds the "Daily absences by bus route" heatmap on the Root causes page. Without it, that panel is hidden and nothing else changes.

1. Create a TigerData service and copy its Postgres connection string.
2. Add `TIGER_DATABASE_URL=postgres://...` to `.env` (project root) or `backend/.env`.
3. Create the hypertable and continuous aggregate once (schema in `backend/db/schema.sql`):
   ```bash
   cd backend && .venv/bin/python scripts/tiger_setup.py
   ```
4. Restart the API.

If the database is unreachable, the app logs a warning and carries on. Tests never connect to it.

## What's built (PRD 1 features)

| PRD | Surface | Route | Status |
| --- | --- | --- | --- |
| F1 Absence tracking & cause capture | Case queue, case detail, SIS CSV import | `/`, `/cases/:id` | Done (keyword classifier stub) |
| Safety gate | Escalations inbox; counselor takes over, no automated reply | `/escalations` | Done |
| Root-cause clusters | Bus route / classroom / household clusters with ablation | `/clusters` | Done (rule-based stub) |
| F4 School action plan | Draft → edit → approve & send, EN/ES messages | `/plans` | Done (templates) |
| F5 Learning plan | Pacing-guide grounded, load adapted to cause | `/learning/:studentId` | P0 done; P1 claim verification TODO |
| F6 Revenue dashboard | At risk, recovered, root causes by $; editable rate | `/revenue` | Done |

### Swap points for other PRDs

The logic behind each feature is a deliberately simple stub with a stable interface, so the model-backed versions can replace it without touching the API or UI:

- `backend/app/services/classifier.py` — `classify(text, names) -> Classification` (barrier, confidence, evidence quote, safety flag, language, source). Uses Gemini when `GEMINI_API_KEY` is set, with structured JSON output, a 6-second timeout and a disk cache in `backend/.cache/gemini`. Falls back to keyword rules when Gemini is not configured, slow or failing; seed data always uses the keyword rules. The keyword safety check always runs and Gemini can add a safety flag but never clear one. Known names are replaced with placeholders before the text is sent, and an evidence quote that is not an exact substring of the reply is replaced by the whole reply.
- `backend/app/services/clustering.py` — `detect_clusters(...) -> List[Cluster]`. Uses only school-recorded links (never inferred friendships).
- `backend/app/services/action_plan.py` — plan and message generation. Only `en` and `es` templates; other languages fall back to English (TODO translation).
- `backend/app/services/learning_plan.py` — `build(...) -> LearningPlan`. Never emits a topic that is not in the pacing guide.
- `backend/app/data/seed.py` — synthetic students, absences, replies and pacing guides for the week of Nov 9–13, 2026.

### iMessage channel (Photon 2.0 Beta)

**Setting it up:** follow [PHOTON_SETUP.md](PHOTON_SETUP.md). For the full list of features and rules, see [PHOTON_FEATURES.md](PHOTON_FEATURES.md).

When the `PHOTON_*` values are set in `.env`, the API polls Photon every 2 seconds on startup; without them the channel stays off and the "Simulate family reply" box still works.

- `backend/app/services/photon_channel.py` — an inbound text from `PHOTON_TEST_SENDER` is treated as a reply from the guardian of `PHOTON_DEMO_STUDENT_ID` (default `S002`) and goes through the same `store.record_reply` path as a simulated reply. The family gets a short acknowledgement, except on safety-flagged messages (counselor only) and after `STOP`.
- `backend/app/services/outreach.py` — the rules every outbound message must pass: no automated message to a household with an open safety escalation, none to a number that opted out, and no links. Also builds the first check-in text (no absence count; says who reads replies).
- `POST /api/absences/{id}/check-in` — sends the check-in for an absence with no reason on file. Approving a plan sends its family messages the same way. Each attempt is recorded in the outbox with a status: `sent`, `queued` (allowed but the family is not reachable yet), `blocked` (a rule forbids it) or `failed`.
- `GET /api/messages` — the message log: every inbound and outbound text, many per absence, with real timestamps and `latency_ms`. `GET /api/messages/latency` summarises reply latency against the 10-second target. The case page shows the full conversation.
- The UI refetches its data every 2 seconds (`frontend/src/hooks.ts`), so an incoming text appears on the case queue, case page, clusters and revenue without a reload.
- `GET /api/channel` — listener status, reachable families, iMessage latency, and recent iMessage log entries.
- `backend/scripts/photon_echo.py` — one-shot round-trip check for a new line or key.
- `backend/scripts/photon_send.py` — sends one test message to your own phone to check outbound.
- Shared lines cannot send the first message: the family phone must text the agent number once before anything can be sent to it.

- Safety escalations (`/escalations`): the counselor acknowledges a case, records the mandated-reporting decision, and closes it as a handled concern or a false alarm (relabeled). Messages to the household resume when its last open case closes. `POST /api/escalations/{id}/acknowledge`, `/report`, `/close`. The counselor can also generate an internal case report from the record, edit it, save versions and download it (`/case-report/draft`, `/case-reports`).

- Quiet hours: check-ins and plan messages are held between 8pm and 8am school time (`SCHOOL_TIMEZONE`, default `America/Los_Angeles`) and sent at 8am. Set `QUIET_HOURS=off` to test or demo outside those hours.
- Crisis-resource message: after a self-harm or imminent-danger flag, one fixed 988/911 message goes to the family, never when abuse is disclosed. The wording is a placeholder until the district approves it.

- On-call paging (`/oncall`): a roster of counselors, administrators and the principal, saved to `backend/.data/oncall.json` (kept across restarts and resets). A new safety case texts the on-call counselor during school hours (`SCHOOL_HOURS`, `SCHOOL_DAYS`) or the on-call administrator otherwise; if nobody acknowledges in 15 minutes the principal is paged. Staff reply `ACK` to acknowledge. Pages contain only the case ID and grade.

Not built yet: SMS fallback, voice notes (out of scope: Photon Beta iMessage carries text and images only), persistence (everything except the on-call roster is in memory).

## Design rules (from the PRD)

- Families are partners, not violators. All outreach asks what is getting in the way and offers help.
- Risk of harm sets priority (queue is ordered safety-first); revenue only justifies the program.
- Every outbound plan needs a clerk's approval.
- Default daily rate is $65 (midpoint of a $35–$100 vendor estimate, unverified); editable on the dashboard.

## 3-minute demo path

1. `/` Case queue: dozens of absences, many with no reason yet.
2. Open an awaiting-reply case and use **Simulate family reply** (e.g. a Spanish bus or illness message); label and evidence quote appear.
3. `/clusters`: Route 7 bus cluster and Room 104 illness cluster with ablation and weekly dollars at risk.
4. **Draft cluster action plan** → `/plans` → **Approve & send**; Spanish messages land in the sent log.
5. `/learning/S010`: catch-up plan adapted to illness; mark complete to count recovery days.
6. `/revenue`: total at risk and recovered; change the daily rate to see every figure update.

Use **Reset demo data** in the sidebar to start over.

### Offline demo mode (the backup)

If the Wi-Fi, Photon or Gemini fails during the demo, restart the API with `DEMO_MODE=1`:

```bash
cd backend && DEMO_MODE=1 SCHOOL_DAYS=all .venv/bin/uvicorn app.main:app --reload --port 8000
```

`SCHOOL_DAYS=all` means the on-call counselor is paged on any day, not only Monday to Friday. Before 8am or after 3:30pm, the administrator is still paged instead.

A **Demo** bar appears at the bottom of every page. **Play next message** sends the next scripted family text through the same path as a real iMessage:

1. Sofia's dad, in Spanish: the bus never came. Opens her case with the label and evidence quote.
2. Opens Root causes, where the Route 7 cluster has just appeared.
3. A custody-hearing reply, labeled an appointment. Opens that case.
4. The clerk corrects it to a family obligation (or do it by hand on the case page).
5. A similar reply comes back right, citing the correction. Opens that case.
6. A bullying disclosure goes to the counselor and pages the on-call staff. Opens Escalations.

Nothing leaves the laptop: Gemini's answers come from `backend/app/data/demo_answers.json`, and outbound texts are logged on a "demo replay" channel. After changing the script (`backend/app/data/demo_script.json`) or the classifier prompt, record the answers again with a Gemini key: `cd backend && .venv/bin/python -m app.services.demo record`.
