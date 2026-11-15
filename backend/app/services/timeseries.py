"""TigerData sidecar: keeps absences in a hypertable; a continuous aggregate feeds the timeline.

`sync` is called by services/persist.py whenever absences change.

Optional. If TIGER_DATABASE_URL is unset or the database is unreachable, every call is a no-op,
so the demo never depends on it.
"""
import logging
import os
import threading
from datetime import datetime, time, timezone

log = logging.getLogger(__name__)
_lock = threading.Lock()
GROUPS = {"bus_route", "classroom", "barrier"}


def enabled() -> bool:
    """On only for live data. The synthetic demo school is never written to the database."""
    from ..store import data_source  # late import: the store imports the services package

    return bool(os.getenv("TIGER_DATABASE_URL")) and "PYTEST_CURRENT_TEST" not in os.environ and data_source() == "live"


def _connect():
    import psycopg

    return psycopg.connect(os.environ["TIGER_DATABASE_URL"], connect_timeout=3)


def _val(x):
    return getattr(x, "value", x)


def _rows(store):
    rows = []
    for a in store.absences.values():
        s = store.students.get(a.student_id)
        c = a.classification
        rows.append((
            datetime.combine(a.date, time(8), tzinfo=timezone.utc), a.id, a.student_id,
            _val(c.barrier) if c else None,
            s.bus_route if s else None, s.classroom if s else None, _val(a.status),
        ))
    return rows


def ensure_schema() -> bool:
    """One row per absence: a unique key lets sync update rows in place. Removes duplicate rows first."""
    if not enabled():
        return False
    try:
        with _lock, _connect() as conn:
            conn.execute(
                "DELETE FROM absence_events a USING absence_events b"
                " WHERE a.absence_id = b.absence_id AND a.time = b.time AND a.ctid < b.ctid"
            )
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS absence_events_key ON absence_events (absence_id, time)")
        return True
    except Exception as e:
        log.warning("TigerData schema update failed: %s", e)
        return False


def sync(store) -> bool:
    """Bring the hypertable in line with the store's absences.

    Each absence is one row, keyed by (absence_id, time). New absences are added and changed ones
    (a reply labels the barrier, a status moves) are updated in place; rows only go away when the
    absence itself is gone, which in practice means "Reset demo data".
    """
    if not enabled():
        return False
    try:
        rows = _rows(store)
        ids = [r[1] for r in rows]
        with _lock, _connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO absence_events (time, absence_id, student_id, barrier, bus_route, classroom, status)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (absence_id, time) DO UPDATE SET student_id = EXCLUDED.student_id,"
                " barrier = EXCLUDED.barrier, bus_route = EXCLUDED.bus_route, classroom = EXCLUDED.classroom,"
                " status = EXCLUDED.status"
                " WHERE (absence_events.student_id, absence_events.barrier, absence_events.bus_route,"
                " absence_events.classroom, absence_events.status) IS DISTINCT FROM"
                " (EXCLUDED.student_id, EXCLUDED.barrier, EXCLUDED.bus_route, EXCLUDED.classroom, EXCLUDED.status)",
                rows,
            )
            cur.execute("DELETE FROM absence_events WHERE NOT (absence_id = ANY(%s))", (ids,))
        return True
    except Exception as e:  # never break the app over the sidecar
        log.warning("TigerData sync failed: %s", e)
        return False


def daily(group: str):
    """Daily absence counts per group from the continuous aggregate."""
    if not enabled() or group not in GROUPS:
        return []
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT day::date, {group}, sum(absences)::int FROM absences_daily"  # group is whitelisted
                f" WHERE {group} IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2"
            )
            return [{"day": d.isoformat(), "key": k, "absences": n} for d, k, n in cur.fetchall()]
    except Exception as e:
        log.warning("TigerData read failed: %s", e)
        return []
