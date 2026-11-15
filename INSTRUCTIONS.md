# Running RollCall

All commands start from the project root (`RollCall/`). You need two terminals: one for the backend (API on port 8000) and one for the frontend (UI on port 5173). Start the backend first, then open http://localhost:5173.

Settings live in `.env` at the project root (copy `.env.example` to start). A setting given on the command line, like `DEMO_MODE=1 ...`, overrides `.env` for that run only.

## First-time setup

```bash
# Backend
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

## Frontend (all modes)

```bash
cd frontend
npm run dev
```

Open http://localhost:5173, choose **Lincoln Unified School District**, then pick a user. Use **Administrator** to see every page.

## Backend: pick one mode

### 1. Scripted demo, no live Gemini (recommended for presenting)

Synthetic school, recorded Gemini answers, no network, Photon off, quiet hours off, demo bar on every page.

```bash
cd backend
DEMO_MODE=1 SCHOOL_DAYS=all .venv/bin/uvicorn app.main:app --port 8000 --reload
```

`SCHOOL_DAYS=all` lets safety pages go to the on-call counselor on weekends too.

### 2. Demo school with keyword rules (no Gemini)

Synthetic school, labels from the built-in keyword rules, Photon on (real texts work).

```bash
cd backend
DATA_SOURCE=demo CLASSIFIER=keyword PHOTON_DEMO_STUDENT_ID=S002 .venv/bin/uvicorn app.main:app --port 8000 --reload
```

`PHOTON_DEMO_STUDENT_ID=S002` makes your phone the family of Sofia Ramírez, a student in the demo school.

### 3. Demo school with Gemini

Same as mode 2, labeled by Gemini (needs `GEMINI_API_KEY` in `.env`).

```bash
cd backend
DATA_SOURCE=demo PHOTON_DEMO_STUDENT_ID=S002 .venv/bin/uvicorn app.main:app --port 8000 --reload
```

### 4. Live data (the real app)

Starts empty; you import students and absences from a CSV. Saved in TigerData when `TIGER_DATABASE_URL` is set. No "Synthetic data" labels, and "Reset demo data" is disabled.

```bash
cd backend
DATA_SOURCE=live .venv/bin/uvicorn app.main:app --port 8000 --reload
```

Or put `DATA_SOURCE=live` in `.env` and run `.venv/bin/uvicorn app.main:app --port 8000 --reload`.

## Importing the sample CSV (live data)

Sample file: [`backend/data/sample/sis_import.csv`](backend/data/sample/sis_import.csv)
(full path: `RollCall/backend/data/sample/sis_import.csv`)

It holds 24 fictional students and 28 absences for Nov 9–13, 2026: a Route 5 bus problem (Tue–Wed), an illness run in Room 3B (Thu–Fri), two siblings out together, and a few single absences.

1. Run the backend in live mode (mode 4).
2. Sign in as **Administrator** or **Attendance clerk**.
3. Open **Case queue** (sidebar, under "Daily work").
4. Click **Import SIS CSV** (top right) and choose the sample file.

Importing the same file again skips absences already logged.

To see a cluster form: open three Route 5 absences from Nov 10 (Mateo Alvarez, Owen Fitzgerald, Isaiah Brooks) and use **Simulate family reply** with "The bus never came this morning". A Route 5 cluster appears on **Root causes**.

CSV columns: `student_id`, `date` (YYYY-MM-DD) required; optional `sis_code`, `first_name`, `last_name`, `grade`, `classroom`, `teacher`, `bus_route`, `household_id`, `preferred_language`, `guardian_name`, `guardian_phone`. A row with student details and an empty `date` adds a student with no absence.

## Sending check-in texts

On **Case queue**, the clerk or administrator clicks **Send check-ins (N)**. It texts every family whose absence is still awaiting a reply and has not had a check-in.

- Texts are held between 8pm and 8am school time. To send at any hour while testing, add `QUIET_HOURS=off` to `.env` (or the command line) and restart the backend. Set it back to `on` for real use.
- The shared Photon line cannot text a number first. Text the agent line once from your phone before testing.
- Your phone (`PHOTON_TEST_SENDER`) acts as the family of the student in `PHOTON_DEMO_STUDENT_ID`:
  - Live data with the sample CSV: `PHOTON_DEMO_STUDENT_ID=LU1003` (Owen Fitzgerald, English-speaking family)
  - Demo school: `PHOTON_DEMO_STUDENT_ID=S002` (Sofia Ramírez, Spanish-speaking family)

## TigerData (optional, live data only)

1. Put `TIGER_DATABASE_URL=postgres://...?sslmode=require` in `.env`.
2. Create the tables once:
   ```bash
   cd backend
   .venv/bin/python scripts/tiger_setup.py
   ```
3. Restart the backend. The log says `TigerData: loaded N records` (or `saved the seed data` the first time).

Run only one backend against a database at a time; two copies overwrite each other.

## Tests

```bash
cd backend
.venv/bin/pytest -q
```

Tests always use the demo school and never touch TigerData or Photon, whatever `.env` says.

## Production build and deploy

```bash
# Build the frontend (output in frontend/dist)
cd frontend
npm run build

# Deploy with Docker (see DEPLOY.md)
cp .env.example .env    # then fill it in
docker compose up -d --build
docker compose exec backend python scripts/tiger_setup.py   # once, if using TigerData
```

## Setting reference

| Setting | Values | What it does |
|---|---|---|
| `DATA_SOURCE` | `demo` (default) / `live` | Synthetic demo school, or empty start with CSV imports |
| `DEMO_MODE` | `0` / `1` | `1` = scripted demo with recorded Gemini answers; always uses demo data |
| `CLASSIFIER` | `gemini` / `keyword` | Who labels family replies (`gemini` needs `GEMINI_API_KEY`) |
| `QUIET_HOURS` | `on` / `off` | Hold texts between 8pm and 8am school time |
| `SCHOOL_DAYS` | `mon-fri` / `all` | Days the counselor is paged first for safety cases |
| `PHOTON_DEMO_STUDENT_ID` | a student ID | Whose family your test phone plays |
| `TIGER_DATABASE_URL` | Postgres URL | Save live data in TigerData |
