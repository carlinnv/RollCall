"""Keep the store in TigerData (Postgres) so the app's data survives restarts.

The store stays in memory and is still the source of truth while the app runs. This module:

- loads it from the `app_records` table on startup (or saves the seed when the table is empty),
- saves changes from a background thread: about once a second it snapshots the store, compares it
  with what was last saved, and writes only the rows that changed. That catches every change path
  (API requests, the Photon thread, paging timers) without hooks in each one,
- keeps the `absence_events` hypertable up to date when absences change (services/timeseries.py).

Off unless TIGER_DATABASE_URL is set (and always off under pytest), so the app still runs in memory.
If the database cannot be read at startup, saving stays off for the whole run: an unloaded store
must never overwrite saved data.
"""
import json
import logging
import threading
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel

from ..models import (
    Absence, ActionPlan, CaseReport, Correction, Escalation, Message, RevenueSettings, Staff, Student,
)
from . import outreach, timeseries

log = logging.getLogger("uvicorn.error")

SAVE_EVERY_SECONDS = 1.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS app_records (
  collection  text        NOT NULL,
  id          text        NOT NULL,
  pos         integer     NOT NULL,
  data        jsonb       NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (collection, id)
)
"""

Key = Tuple[str, str]  # (collection, id)
Row = Tuple[int, str]  # (position, JSON text)

_lock = threading.Lock()  # one save at a time
_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_saved: Dict[Key, Row] = {}  # what the database holds, as of the last successful save
_active = False  # True only after a successful load


def enabled() -> bool:
    return timeseries.enabled()


def active() -> bool:
    return _active


def _dump(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        value = [v.model_dump(mode="json") if isinstance(v, BaseModel) else v for v in value]
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _rows(collection: str, items: Iterable[Tuple[str, object]]) -> Dict[Key, Row]:
    return {(collection, str(k)): (i, _dump(v)) for i, (k, v) in enumerate(items)}


# ---------- store <-> rows ----------

def snapshot(store) -> Dict[Key, Row]:
    """Every record in the store as {(collection, id): (position, json)}. Pure: no database."""
    rows: Dict[Key, Row] = {}
    rows.update(_rows("students", list(store.students.items())))
    rows.update(_rows("absences", list(store.absences.items())))
    rows.update(_rows("escalations", list(store.escalations.items())))
    rows.update(_rows("case_reports", list(store.case_reports.items())))
    rows.update(_rows("plans", list(store.plans.items())))
    # Messages are append-only; their position is their id, so order survives a reload.
    rows.update(_rows("messages", ((f"{i:06d}", m) for i, m in enumerate(list(store.messages)))))
    rows.update(_rows("corrections", list(store.corrections.by_school.items())))
    rows.update(_rows("staff", list(store.roster.staff.items())))
    rows.update(_rows("state", [
        ("settings", store.settings),
        ("recovery_log", list(store.recovery_log)),
        ("completed_learning", dict(store.completed_learning)),
        ("demo_seen", sorted(store.demo_seen)),
        ("held_keys", dict(store._held_keys)),
        # Families who texted STOP. Must outlive restarts; reset() does not clear it either.
        ("opted_out", sorted(outreach.opted_out)),
    ]))
    return rows


def restore(store, rows: Dict[Key, Row]) -> None:
    """Replace the store's contents with `rows` (as produced by `snapshot`). Pure: no database."""
    by: Dict[str, List[Tuple[int, str, object]]] = {}
    for (collection, rid), (pos, text) in rows.items():
        by.setdefault(collection, []).append((pos, rid, json.loads(text)))

    def items(collection: str) -> List[Tuple[str, object]]:
        return [(rid, data) for _, rid, data in sorted(by.get(collection, []), key=lambda r: r[0])]

    store.students = {k: Student.model_validate(v) for k, v in items("students")}
    store.absences = {k: Absence.model_validate(v) for k, v in items("absences")}
    store.escalations = {k: Escalation.model_validate(v) for k, v in items("escalations")}
    store.case_reports = {k: [CaseReport.model_validate(c) for c in v] for k, v in items("case_reports")}
    store.plans = {k: ActionPlan.model_validate(v) for k, v in items("plans")}
    store.messages = [Message.model_validate(v) for _, v in items("messages")]
    store.corrections.by_school = {k: [Correction.model_validate(c) for c in v] for k, v in items("corrections")}
    staff = {k: Staff.model_validate(v) for k, v in items("staff")}
    if staff:  # keep the roster file's staff when the database has none yet
        with store.roster._lock:
            store.roster.staff = staff
        store.roster._save()

    state = dict(items("state"))
    if "settings" in state:
        store.settings = RevenueSettings.model_validate(state["settings"])
    store.recovery_log = list(state.get("recovery_log", []))
    store.completed_learning = dict(state.get("completed_learning", {}))
    store.demo_seen = set(state.get("demo_seen", []))
    store._held_keys = dict(state.get("held_keys", {}))
    # Update in place: the Photon channel holds a reference to this set.
    outreach.opted_out.clear()
    outreach.opted_out.update(state.get("opted_out", []))
    store.touch()  # clusters are recomputed from the restored data


def _snapshot_safely(store) -> Optional[Dict[Key, Row]]:
    """Snapshot while other threads may be changing the store; None means try again next tick."""
    try:
        return snapshot(store)
    except RuntimeError:  # a dict or set changed size while it was being copied
        return None


# ---------- database ----------

def _read_all(conn) -> Dict[Key, Row]:
    cur = conn.execute("SELECT collection, id, pos, data::text FROM app_records")
    return {(c, i): (p, json.dumps(json.loads(d), sort_keys=True, separators=(",", ":"))) for c, i, p, d in cur.fetchall()}


def _write(conn, rows: Dict[Key, Row], before: Dict[Key, Row]) -> Tuple[int, int, bool]:
    """Upsert changed rows and delete removed ones in one transaction. Returns (upserted, deleted, absences changed)."""
    changed = [(c, i, p, d) for (c, i), (p, d) in rows.items() if before.get((c, i)) != (p, d)]
    removed = [k for k in before if k not in rows]
    if not changed and not removed:
        return 0, 0, False
    with conn.transaction(), conn.cursor() as cur:
        if changed:
            cur.executemany(
                "INSERT INTO app_records (collection, id, pos, data) VALUES (%s, %s, %s, %s::jsonb)"
                " ON CONFLICT (collection, id) DO UPDATE SET pos = EXCLUDED.pos, data = EXCLUDED.data, updated_at = now()",
                changed,
            )
        if removed:
            cur.executemany("DELETE FROM app_records WHERE collection = %s AND id = %s", removed)
    touched = {c for c, *_ in changed} | {c for c, _ in removed}
    return len(changed), len(removed), bool(touched & {"absences", "students"})


def save(store) -> bool:
    """Write whatever changed since the last save. Safe to call from any thread."""
    global _saved
    if not _active:
        return False
    rows = _snapshot_safely(store)
    if rows is None:
        return False
    with _lock:
        try:
            with timeseries._connect() as conn:
                up, down, absences_changed = _write(conn, rows, _saved)
            _saved = rows
        except Exception as e:  # keep running in memory; the next tick retries
            log.warning("TigerData save failed: %s", e)
            return False
    if up or down:
        log.debug("TigerData save: %d upserted, %d deleted", up, down)
    if absences_changed:
        timeseries.sync(store)
    return True


def _loop(store) -> None:
    while not _stop.wait(SAVE_EVERY_SECONDS):
        save(store)


def start(store) -> None:
    """Load the store from the database (or save the seed into it), then save changes in the background."""
    global _active, _saved, _thread
    if not enabled() or _active:
        return
    try:
        with timeseries._connect() as conn:
            conn.execute(SCHEMA)
            conn.commit()
            rows = _read_all(conn)
            if rows:
                restore(store, rows)
                _saved = rows
                log.info("TigerData: loaded %d records", len(rows))
            else:
                first = snapshot(store)
                _write(conn, first, {})
                _saved = first
                log.info("TigerData: empty database, saved the seed data (%d records)", len(first))
    except Exception as e:
        log.error("TigerData: could not load saved data (%s). Running in memory; nothing will be saved this run.", e)
        return
    _active = True
    timeseries.ensure_schema()
    timeseries.sync(store)
    _stop.clear()
    _thread = threading.Thread(target=_loop, args=(store,), name="tiger-save", daemon=True)
    _thread.start()


def stop(store) -> None:
    """Stop the saver and write any last changes."""
    global _thread
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=5)
        _thread = None
    save(store)
