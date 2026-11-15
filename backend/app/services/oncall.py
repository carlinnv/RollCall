"""On-call roster and paging rules for safety escalations (PRD 3 section 3).

- During school hours the on-call counselor is paged; after hours, the on-call administrator.
- If nobody acknowledges within 15 minutes, the principal is paged.
- If no one is on call for a role, the next role in the chain is paged instead.

The roster is staff configuration, not demo data, so it is saved to a JSON file
(backend/.data/oncall.json, git-ignored because it holds phone numbers) and survives
restarts and "Reset demo data".

Page texts carry no student name or family details: staff phones may be personal devices.
The details stay in the app.
"""
import json
import os
import threading
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional

from ..models import Staff
from . import outreach

ROLES = ("counselor", "administrator", "principal")
# Who to try, in order, for a first page and for an escalation.
CHAIN = {
    "counselor": ["counselor", "administrator", "principal"],
    "administrator": ["administrator", "principal"],
    "principal": ["principal"],
}
DEFAULT_SCHOOL_HOURS = "08:00-15:30"
DEFAULT_FILE = Path(__file__).resolve().parents[2] / ".data" / "oncall.json"


def roster_file() -> Path:
    return Path(os.environ.get("ONCALL_FILE") or DEFAULT_FILE)


def school_hours() -> tuple:
    """(start, end) from SCHOOL_HOURS, e.g. "08:00-15:30". Weekdays only."""
    raw = os.environ.get("SCHOOL_HOURS") or DEFAULT_SCHOOL_HOURS
    try:
        start, end = (time.fromisoformat(part.strip()) for part in raw.split("-"))
    except ValueError:
        start, end = (time.fromisoformat(part) for part in DEFAULT_SCHOOL_HOURS.split("-"))
    return start, end


def school_days_all() -> bool:
    """SCHOOL_DAYS=all counts weekends as school days (for a weekend demo). Default: Monday to Friday."""
    return (os.environ.get("SCHOOL_DAYS") or "mon-fri").strip().lower() == "all"


def in_school_hours(now: datetime) -> bool:
    start, end = school_hours()
    weekday = school_days_all() or now.weekday() < 5
    return weekday and start <= now.timetz().replace(tzinfo=None) < end


def first_role(now: datetime) -> str:
    """Who is paged first for a new case at this school time."""
    return "counselor" if in_school_hours(now) else "administrator"


def page_text(escalation_id: str, grade: int, escalated: bool) -> str:
    if escalated:
        return (f"RollCall: safety case {escalation_id} (grade {grade}) has not been acknowledged for 15 minutes. "
                f"Open the Escalations page. Reply ACK to acknowledge.")
    return (f"RollCall: new safety case {escalation_id} (grade {grade}). "
            f"Open the Escalations page. Reply ACK to acknowledge.")


class Roster:
    """Staff who can be paged. Thread-safe; every change is written to disk."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or roster_file()
        self._lock = threading.Lock()
        self.staff: Dict[str, Staff] = {}
        self.load()

    def load(self) -> None:
        try:
            rows = json.loads(self.path.read_text())
            self.staff = {row["staff_id"]: Staff(**row) for row in rows}
        except (OSError, ValueError, KeyError, TypeError):
            self.staff = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([s.model_dump() for s in self.staff.values()], indent=2))
        except OSError:
            pass  # still works for this session; the page shows what is in memory

    @staticmethod
    def _check(name: str, role: str, phone: str) -> None:
        if not name.strip():
            raise ValueError("name is required")
        if role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}")
        if len(outreach.digits(phone)) < 10:
            raise ValueError("phone needs at least 10 digits, with the country code, e.g. +15551234567")

    def add(self, name: str, role: str, phone: str, on_call: bool = True) -> Staff:
        self._check(name, role, phone)
        with self._lock:
            if any(outreach.digits(s.phone) == outreach.digits(phone) for s in self.staff.values()):
                raise ValueError("that phone number is already on the roster")
            n = max([int(i[1:]) for i in self.staff] + [0]) + 1
            member = Staff(staff_id=f"T{n:02d}", name=name.strip(), role=role, phone=phone.strip(), on_call=on_call)
            self.staff[member.staff_id] = member
            self._save()
            return member

    def update(self, staff_id: str, fields: dict) -> Staff:
        with self._lock:
            current = self.staff[staff_id]  # KeyError -> 404
            merged = {**current.model_dump(), **{k: v for k, v in fields.items() if k in ("name", "role", "phone", "on_call")}}
            self._check(merged["name"], merged["role"], merged["phone"])
            if any(s.staff_id != staff_id and outreach.digits(s.phone) == outreach.digits(merged["phone"])
                   for s in self.staff.values()):
                raise ValueError("that phone number is already on the roster")
            self.staff[staff_id] = Staff(**merged)
            self._save()
            return self.staff[staff_id]

    def remove(self, staff_id: str) -> None:
        with self._lock:
            del self.staff[staff_id]  # KeyError -> 404
            self._save()

    def by_phone(self, phone: str) -> Optional[Staff]:
        key = outreach.digits(phone)
        return next((s for s in self.staff.values() if outreach.digits(s.phone) == key), None)

    def on_call(self, role: str) -> List[Staff]:
        return [s for s in self.staff.values() if s.role == role and s.on_call]

    def targets(self, chain: List[str]) -> List[Staff]:
        """Everyone on call for the first role in the chain that has anyone on call."""
        for role in chain:
            people = self.on_call(role)
            if people:
                return people
        return []
