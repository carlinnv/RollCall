"""TigerData sidecar: mirrors absences into a hypertable; a continuous aggregate feeds the timeline.

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
    return bool(os.getenv("TIGER_DATABASE_URL")) and "PYTEST_CURRENT_TEST" not in os.environ


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


def sync(store) -> bool:
    """Replace the hypertable contents with the store's current absences (small demo data set)."""
    if not enabled():
        return False
    try:
        rows = _rows(store)
        with _lock, _connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM absence_events")
            cur.executemany(
                "INSERT INTO absence_events (time, absence_id, student_id, barrier, bus_route, classroom, status)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                rows,
            )
        return True
    except Exception as e:  # never break the app over the sidecar
        log.warning("TigerData sync failed: %s", e)
        return False


def sync_async(store) -> None:
    if enabled():
        threading.Thread(target=sync, args=(store,), daemon=True).start()


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
