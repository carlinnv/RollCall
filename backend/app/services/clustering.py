"""Root-cause clustering (stub).

Combines two signals:
  * reason text: the classified barrier of each absence
  * school contact network: students who share a bus route, classroom, or household

A cluster is a group of at least MIN_SIZE students (2 for a household) with the
same barrier who share one link within a short date window. We never infer
friendships between students; only school-recorded links are used.
"""
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

from ..models import Ablation, Absence, Barrier, Cluster, RevenueSettings, Student
from .revenue import dollars

MIN_SIZE = 3
WINDOW_DAYS = 3
LINKS = ("bus_route", "classroom", "household_id")
LINK_LABEL = {"bus_route": "bus_route", "classroom": "classroom", "household_id": "household"}
SKIP = {Barrier.UNKNOWN, Barrier.SAFETY}  # safety cases go to a counselor one by one


def _link_value(student: Student, link: str) -> Optional[str]:
    return getattr(student, link)


def detect_clusters(
    students: Dict[str, Student],
    absences: List[Absence],
    settings: RevenueSettings,
    week_start: date,
    week_end: date,
) -> List[Cluster]:
    labeled = [a for a in absences if a.classification and a.classification.barrier not in SKIP]
    clusters: List[Cluster] = []
    claimed = set()  # absence ids already in a stronger cluster

    for link in LINKS:
        groups: Dict[tuple, List[Absence]] = defaultdict(list)
        for a in labeled:
            value = _link_value(students[a.student_id], link)
            if value:
                groups[(a.classification.barrier, value)].append(a)

        for (barrier, value), members in groups.items():
            members = [a for a in members if a.id not in claimed]
            members.sort(key=lambda a: a.date)
            if not members:
                continue
            span = members[-1].date - members[0].date
            student_ids = sorted({a.student_id for a in members})
            min_size = 2 if link == "household_id" else MIN_SIZE
            if len(student_ids) < min_size or span > timedelta(days=WINDOW_DAYS):
                continue

            claimed.update(a.id for a in members)
            label = LINK_LABEL[link]
            week_days = sum(1 for a in members if week_start <= a.date <= week_end)
            clusters.append(Cluster(
                id=f"C-{label}-{value}-{barrier.value}".replace(" ", "_"),
                barrier=barrier,
                link_type=label,
                link_value=value,
                student_ids=student_ids,
                absence_ids=[a.id for a in members],
                first_date=members[0].date,
                last_date=members[-1].date,
                dollars_at_risk_week=dollars(week_days, settings),
                ablation=_ablation(students, absences, labeled, barrier, link, value, members, student_ids),
                summary=_summary(barrier, label, value, student_ids, members),
            ))

    clusters.sort(key=lambda c: len(c.student_ids), reverse=True)
    return clusters


def _ablation(students, all_absences, labeled, barrier, link, value, members, student_ids) -> Ablation:
    lo, hi = members[0].date - timedelta(days=WINDOW_DAYS), members[-1].date + timedelta(days=WINDOW_DAYS)
    text_only = {a.student_id for a in labeled if a.classification.barrier == barrier and lo <= a.date <= hi}
    network_only = {
        a.student_id for a in all_absences
        if _link_value(students[a.student_id], link) == value and lo <= a.date <= hi
    }
    return Ablation(text_only=len(text_only), network_only=len(network_only), combined=len(student_ids))


def _summary(barrier: Barrier, label: str, value: str, student_ids: List[str], members: List[Absence]) -> str:
    days = [d.strftime("%a %b %-d") for d in sorted({a.date for a in members})]
    where = {"bus_route": f"riders of {value}", "classroom": f"students in {value}", "household": "siblings in one household"}[label]
    return f"{len(student_ids)} {where} absent for {barrier.value} ({', '.join(days)})."
