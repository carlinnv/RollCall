"""Create the TigerData hypertable and continuous aggregate. Run once: python scripts/tiger_setup.py"""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
# Same places the app reads settings from: the project root .env, then backend/.env.
load_dotenv(BACKEND.parent / ".env")
load_dotenv(BACKEND / ".env")
url = os.environ.get("TIGER_DATABASE_URL")
if not url:
    raise SystemExit("TIGER_DATABASE_URL is not set. Add it to .env in the project root or backend/.env.")
sql = (BACKEND / "db" / "schema.sql").read_text()
with psycopg.connect(url, autocommit=True, connect_timeout=10) as conn:
    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
        print("->", stmt.splitlines()[0])
        conn.execute(stmt)
print("TigerData schema ready.")
