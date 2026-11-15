"""Erase the live data saved in TigerData, so the app starts empty again.

Deletes students, absences, replies, escalations, plans, messages, corrections and settings
(table app_records) and the absence history (table absence_events). Keeps the on-call staff roster.

Stop the backend first: a running backend still holds the old data in memory and would save
parts of it back.

    cd backend
    .venv/bin/python scripts/clear_live_data.py        # asks before deleting
    .venv/bin/python scripts/clear_live_data.py --yes  # no question
"""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND.parent / ".env")
load_dotenv(BACKEND / ".env")
url = os.environ.get("TIGER_DATABASE_URL")
if not url:
    raise SystemExit("TIGER_DATABASE_URL is not set. Add it to .env in the project root or backend/.env.")

with psycopg.connect(url, connect_timeout=10) as conn:
    counts = conn.execute(
        "SELECT collection, count(*) FROM app_records WHERE collection <> 'staff' GROUP BY 1 ORDER BY 1"
    ).fetchall()
    events = conn.execute("SELECT count(*) FROM absence_events").fetchone()[0]
    if not counts and not events:
        print("Nothing to clear: the live data is already empty.")
        sys.exit(0)
    print("This will delete:")
    for collection, n in counts:
        print(f"  {collection}: {n}")
    print(f"  absence history rows: {events}")
    print("On-call staff are kept.")
    if "--yes" not in sys.argv and input("Delete? Type yes: ").strip().lower() != "yes":
        print("Cancelled. Nothing was deleted.")
        sys.exit(1)
    with conn.transaction():
        conn.execute("DELETE FROM app_records WHERE collection <> 'staff'")
        conn.execute("DELETE FROM absence_events")
print("Cleared. Start the backend; it begins empty. Import the CSV again from the Case queue.")
