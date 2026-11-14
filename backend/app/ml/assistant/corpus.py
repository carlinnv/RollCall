"""Plain-data corpus for the F7 counselor assistant (PRD 2 section 3.4).

`build_corpus(run)` turns a SimRun into family-message records with the metadata the
assistant can filter on. `AssistantData.from_sim_run(run, alarms)` bundles those records with
absences, student metadata and detector clusters as plain Python data, so the assistant never
imports the app store and can be fed from any source (simulator, API payloads, tests).

Ids: one reply per absence, so message_id = "M-" + absence_id (e.g. "M-A00123").
Ground truth (true_label, event_id) stays out of `Message`; the benchmark reads it from the run.
"""
from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DAYS_PER_WEEK = 5  # school days; week w (1-based) covers days 5(w-1) .. 5(w-1)+4


def week_of(day: int) -> int:
    return day // DAYS_PER_WEEK + 1


def week_days(week: int) -> tuple:
    start = (week - 1) * DAYS_PER_WEEK
    return start, start + DAYS_PER_WEEK - 1


def message_id_for(absence_id: str) -> str:
    return f"M-{absence_id}"


@dataclass(frozen=True)
class Message:
    message_id: str
    absence_id: str
    student: str
    day: int                     # day of the absence the reply explains
    received_day: int
    language: str
    text: str
    observed_label: str          # pipeline label (classifier output / noisy proxy), not ground truth
    section: Optional[str] = None
    grade: Optional[int] = None
    route: Optional[str] = None
    neighborhood: Optional[str] = None
    household: Optional[str] = None

    @property
    def week(self) -> int:
        return week_of(self.day)

    def meta(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("text")
        d["week"] = self.week
        return d


def build_corpus(run: Any) -> List[Message]:
    """Messages from a SimRun's replies (sorted by absence day, then id)."""
    school = run.school
    by_id = {a.id: a for a in run.absences}
    out: List[Message] = []
    for r in run.replies:
        a = by_id[r.absence_id]
        s = a.student
        out.append(Message(
            message_id=message_id_for(a.id), absence_id=a.id, student=s, day=a.day,
            received_day=r.received_day, language=r.language, text=r.text, observed_label=r.observed_label,
            section=school.section.get(s), grade=school.grade.get(s), route=school.bus_route.get(s),
            neighborhood=school.neighborhood.get(s), household=school.household.get(s)))
    out.sort(key=lambda m: (m.day, m.message_id))
    return out


def group_sizes(students: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    """layer -> zone -> number of students, from {student: {"section":..., "route":..., ...}}."""
    sizes: Dict[str, Counter] = defaultdict(Counter)
    for meta in students.values():
        for layer in ("section", "route", "neighborhood"):
            z = meta.get(layer)
            if z is not None:
                sizes[layer][z] += 1
    return {k: dict(v) for k, v in sizes.items()}


def _alarm_dict(a: Any) -> Dict[str, Any]:
    """Accept app.ml.types.Alarm objects or dicts; return a JSON-friendly dict."""
    d = dataclasses.asdict(a) if dataclasses.is_dataclass(a) else dict(a)
    for key in ("zone", "absent_members"):
        if key in d and d[key] is not None:
            d[key] = sorted(d[key])
    d.setdefault("evidence_absence_ids", [])
    return d


@dataclass
class AssistantData:
    """Everything the assistant's read-only tools look at, as plain data."""

    messages: List[Message]
    students: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # id -> section/grade/route/...
    absences: List[Dict[str, Any]] = field(default_factory=list)       # {absence_id, student, day}
    clusters: List[Dict[str, Any]] = field(default_factory=list)       # detector alarms as dicts
    as_of: Optional[int] = None                                        # "today" for relative dates

    @classmethod
    def from_sim_run(cls, run: Any, alarms: Optional[Iterable[Any]] = None,
                     as_of: Optional[int] = None) -> "AssistantData":
        school = run.school
        students = {s: {"section": school.section.get(s), "grade": school.grade.get(s),
                        "route": school.bus_route.get(s), "neighborhood": school.neighborhood.get(s),
                        "household": school.household.get(s)} for s in school.students}
        absences = [{"absence_id": a.id, "student": a.student, "day": a.day} for a in run.absences]
        clusters = [_alarm_dict(a) for a in (alarms or [])]
        return cls(messages=build_corpus(run), students=students, absences=absences, clusters=clusters,
                   as_of=as_of if as_of is not None else run.n_days - 1)

    def group_sizes(self) -> Dict[str, Dict[str, int]]:
        return group_sizes(self.students)

    def message_ids(self) -> Sequence[str]:
        return [m.message_id for m in self.messages]
