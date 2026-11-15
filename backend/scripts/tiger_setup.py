"""Create the TigerData hypertable and continuous aggregate. Run once: python scripts/tiger_setup.py"""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
url = os.environ["TIGER_DATABASE_URL"]
sql = (Path(__file__).resolve().parents[1] / "db" / "schema.sql").read_text()
with psycopg.connect(url, autocommit=True, connect_timeout=10) as conn:
    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
        print("->", stmt.splitlines()[0])
        conn.execute(stmt)
print("TigerData schema ready.")
