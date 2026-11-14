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

- `backend/app/services/classifier.py` — `classify(text) -> Classification` (barrier, confidence, evidence quote, safety flag). Safety check runs first and always wins.
- `backend/app/services/clustering.py` — `detect_clusters(...) -> List[Cluster]`. Uses only school-recorded links (never inferred friendships).
- `backend/app/services/action_plan.py` — plan and message generation. Only `en` and `es` templates; other languages fall back to English (TODO translation).
- `backend/app/services/learning_plan.py` — `build(...) -> LearningPlan`. Never emits a topic that is not in the pacing guide.
- `backend/app/data/seed.py` — synthetic students, absences, replies and pacing guides for the week of Nov 9–13, 2026.

### iMessage channel (Photon 2.0 Beta)

Copy `.env.example` to `.env` and fill in the `PHOTON_*` values. When they are set, the API polls Photon every 2 seconds on startup; without them the channel stays off and the "Simulate family reply" box still works.

- `backend/app/services/photon_channel.py` — an inbound text from `PHOTON_TEST_SENDER` is treated as a reply from the guardian of `PHOTON_DEMO_STUDENT_ID` (default `S002`) and goes through the same `store.record_reply` path as a simulated reply. The family gets a short acknowledgement, except on safety-flagged messages (counselor only) and after `STOP`.
- `GET /api/channel` — listener status and the message audit trail.
- `backend/scripts/photon_echo.py` — one-shot round-trip check for a new line or key.
- Shared lines cannot send the first message: the family phone must text the agent number once before anything can be sent to it.

Not built yet: outbound check-in and approved-plan messages over Photon, SMS fallback, voice notes (Photon Beta iMessage carries text and images only), speech-to-text, language detection, persistence.

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
