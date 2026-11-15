"""Live adapter: the app's students / absences / replies -> PRD 2 detectors -> API clusters.

    clusters, meta = live.detect(students, absences, settings, week_start, week_end)

Datasets (env DATASET, default "demo")
  demo  The seed school in app/data/seed.py, as edited through the API.
  sim   The simulated Lyon primary school (graphs.load_school + assign_logistics + simulator.simulate
        with CONFIG_A), for scale. Read-only: plans cannot be drafted from it (no real families).

Zones for the demo school
  The seed school has no contact graph, so only logistics zones exist: class section (classroom),
  bus route and grade. Households are the counting unit, not a zone (siblings out together count
  once and are never a cluster on their own, PRD 2 section 3.2). No ego-net (contact) or neighborhood
  zones are built for the demo.

Methods (env CLUSTER_METHOD, default "auto"; every cluster says which one produced it in `method`)
  scan        The chosen PRD 2 detector (env DETECTOR; default "two_stage" when detectors.py has it,
              else "graph_labels") run prospectively for each day of the current week. The seed has
              one week of data and the Kulldorff scan needs a 40-day baseline, so for the demo the
              week is preceded by a SYNTHETIC QUIET BASELINE: 35 school days of background absences
              (3 % per student-day, the simulator's background barrier mix, 80 % replied, no planted
              events, fixed seed). Those clusters carry method "scan:<detector>+synthetic_baseline";
              their p-values are relative to that invented history and say so in method_note.
              With 28 students and one very unusual week the space-time scan is underpowered: the
              week's school-wide surge is conditioned out, so it often misses the Route 7 failure.
  rule_based  Fallback (PRD 2 section 3.2): at least 3 students (spanning >= 2 households) with the
              same barrier who share a bus route or classroom within 3 days. Each gets an exact
              one-sided hypergeometric p-value: of the n students with that barrier in the cluster's
              date span, how surprising is it that k or more are in this zone of K out of N students?
              observed = k, expected = n K / N. Uncorrected for the number of zones tried.
  auto        scan, plus every rule-based cluster whose (link, zone, cause) the scan did not report.
              This guarantees the PRD acceptance clusters (Route 7, Room 104) appear, each labeled
              with the method that actually found it.
  rules       rule_based only.

Safety: safety-labeled replies are hidden from the detectors (their absences still count as
all-cause absences) and no cluster is ever attributed to safety; those cases go to a counselor one
by one.
"""
from __future__ import annotations

import inspect
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..models import Ablation, Absence, Barrier, Cluster, RevenueSettings, Student
from ..services import clustering
from ..services.revenue import dollars
from . import detectors, scan
from .types import AbsenceEvent, Alarm, Reply, SchoolGraph, SimConfig, SimRun

# ---------------------------------------------------------------------------------------------
# configuration

DATASETS = ("demo", "sim")
METHODS = ("auto", "scan", "rules")
SYNTHETIC_BASELINE_DAYS = 35            # + the 5 days of the seed week = the scan's 40-day baseline
SYNTHETIC_RATE = 0.03
SYNTHETIC_REPLY_RATE = 0.8
SYNTHETIC_SEED = 0
SYNTHETIC_PREFIX = "SYN"                # absence ids of the synthetic baseline (never shown to users)
DEMO_N_PERM = 999                       # PRD 2 asks for 999 permutations; the demo school is tiny
# Copy of the simulator's background mix (kept local so simulator edits cannot change the demo).
BACKGROUND_MIX: Dict[str, float] = {
    "illness": 0.42, "appointment or travel": 0.15, "caregiving or family obligation": 0.10,
    "transportation": 0.08, "disengagement or academic struggle": 0.08, "housing or food insecurity": 0.04,
    "unknown": 0.13,
}
LINK_OF_LAYER = {"route": "bus_route", "section": "classroom", "grade": "grade", "neighborhood": "neighborhood",
                 "contact": "contact", "school": "school"}
LAYER_OF_LINK = {"bus_route": "route", "classroom": "section", "household": "household"}
NO_CLUSTER = {Barrier.SAFETY.value, Barrier.UNKNOWN.value}
SIM_PERM = 199


def dataset() -> str:
    d = os.environ.get("DATASET", "demo").strip().lower()
    return d if d in DATASETS else "demo"


def method() -> str:
    m = os.environ.get("CLUSTER_METHOD", "auto").strip().lower()
    return m if m in METHODS else "auto"


def detector_name() -> str:
    """DETECTOR env if it names a detector in detectors.py, else two_stage if present, else graph_labels."""
    wanted = os.environ.get("DETECTOR", "").strip()
    for name in (wanted, "two_stage", "graph_labels", "graph_only"):  # graph_labels: older detectors.py
        if name and callable(getattr(detectors, name, None)):
            return name
    return "graph_only"


def config_key() -> Tuple[str, str, str]:
    return dataset(), method(), detector_name()


# ---------------------------------------------------------------------------------------------
# calendar: school-day index <-> date (weekdays only)


@dataclass
class Calendar:
    days: List[date]

    def __post_init__(self) -> None:
        self.index = {d: i for i, d in enumerate(self.days)}

    def day_of(self, d: date) -> Optional[int]:
        return self.index.get(d)

    def last_on_or_before(self, d: date) -> int:
        """Index of the last school day <= d (clamped to the calendar)."""
        best = 0
        for i, x in enumerate(self.days):
            if x <= d:
                best = i
        return best


def _weekdays_back(end: date, n: int) -> List[date]:
    out: List[date] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _weekdays_between(start: date, end: date) -> List[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------------------------
# detector plumbing


def _run_detector(name: str, run: SimRun, days: Iterable[int], params: scan.ScanParams,
                  zones: Optional[scan.ZoneSet]) -> List[Alarm]:
    """Run detectors.<name> for each as_of in `days` (BH at q = 0.10, the detectors' default).
    A Pipeline is shared when the detector accepts one (two_stage may not)."""
    fn: Callable = getattr(detectors, name)
    accepts = inspect.signature(fn).parameters
    kwargs = {}
    if "pipeline" in accepts:
        kwargs["pipeline"] = detectors.Pipeline(run, params, zones=zones)
    out: List[Alarm] = []
    for t in days:
        out.extend(fn(run, t, params, **kwargs))
    return out


@dataclass
class _Group:
    layer: str
    zone_id: str
    cause: str
    alarms: List[Alarm]

    @property
    def best(self) -> Alarm:
        return min(self.alarms, key=lambda a: (a.p_value, -a.llr, a.day))

    @property
    def first_day(self) -> int:
        return min(a.day for a in self.alarms)

    @property
    def members(self) -> frozenset:
        return frozenset().union(*(a.absent_members for a in self.alarms))

    @property
    def evidence(self) -> List[str]:
        return sorted(set().union(*(a.evidence_absence_ids for a in self.alarms)))


def cause_now(alarm: Alarm, day: Optional[int]) -> str:
    """Cause as known on `day`: two_stage re-attributes as replies arrive (attribution_history)."""
    if day is not None and getattr(alarm, "attribution_history", None) and hasattr(detectors, "attribution_at"):
        return detectors.attribution_at(alarm, day)
    return alarm.barrier


def _group(alarms: Sequence[Alarm], now: Optional[int] = None) -> List[_Group]:
    """Alarms of the same (layer, zone, cause as of `now`) on different days are one cluster; then
    clusters with the same cause whose absent members overlap with Jaccard > 0.5 keep only the strongest."""
    by: Dict[Tuple[str, str, str], List[Alarm]] = defaultdict(list)
    for a in alarms:
        b = cause_now(a, now)
        if b in NO_CLUSTER:
            continue
        by[(a.layer, a.zone_id, b)].append(a)
    groups = sorted((_Group(l, z, b, al) for (l, z, b), al in by.items()),
                    key=lambda g: (g.best.p_value, -g.best.llr, g.zone_id))
    kept: List[_Group] = []
    for g in groups:
        if all(k.cause != g.cause or detectors.jaccard(k.members, g.members) <= 0.5 for k in kept):
            kept.append(g)
    return kept


def _slug(*parts: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "-".join(parts)).strip("_")


def cluster_id(link_type: str, link_value: str, cause: str, start: date) -> str:
    """Deterministic: same link, zone, cause and ISO week of onset -> same id across re-runs."""
    y, w, _ = start.isocalendar()
    return _slug("C", link_type, link_value, cause, f"{y}W{w:02d}")


# ---------------------------------------------------------------------------------------------
# demo school -> SimRun


def _label(a: Absence) -> Optional[str]:
    """The label the detectors may see: None when there is no reply; safety is hidden (unknown)."""
    if not a.classification:
        return None
    b = a.classification.barrier.value
    return Barrier.UNKNOWN.value if b == Barrier.SAFETY.value else b


def demo_school(students: Dict[str, Student]) -> SchoolGraph:
    ids = sorted(students)
    return SchoolGraph(
        name="seed_demo", students=ids,
        section={s: students[s].classroom for s in ids},
        grade={s: students[s].grade for s in ids},
        daily=[],  # no contact graph for the seed school
        bus_route={s: students[s].bus_route for s in ids},
        household={s: students[s].household_id for s in ids},
        neighborhood={},
    )


def synthetic_baseline(students: Sequence[str], n_days: int, seed: int = SYNTHETIC_SEED,
                       rate: float = SYNTHETIC_RATE, reply_rate: float = SYNTHETIC_REPLY_RATE
                       ) -> Tuple[List[AbsenceEvent], List[Reply]]:
    """Quiet background absences for days 0..n_days-1: no planted events, deterministic."""
    rng = np.random.default_rng(seed)
    labels = list(BACKGROUND_MIX)
    probs = np.array([BACKGROUND_MIX[b] for b in labels])
    probs = probs / probs.sum()
    absences: List[AbsenceEvent] = []
    replies: List[Reply] = []
    for d in range(n_days):
        draws = rng.random(len(students))
        for i in np.flatnonzero(draws < rate):
            b = labels[int(rng.choice(len(labels), p=probs))]
            aid = f"{SYNTHETIC_PREFIX}{len(absences):04d}"
            absences.append(AbsenceEvent(id=aid, student=students[int(i)], day=d, true_barrier=b))
            if rng.random() < reply_rate:
                replies.append(Reply(absence_id=aid, received_day=d, language="en", text="",
                                     true_label=b, observed_label=b))
    return absences, replies


def demo_run(students: Dict[str, Student], absences: Iterable[Absence], week_start: date, week_end: date,
             baseline_days: int = SYNTHETIC_BASELINE_DAYS) -> Tuple[SimRun, Calendar]:
    school = demo_school(students)
    absences = [a for a in absences if a.student_id in students]
    last = max([week_end] + [a.date for a in absences])
    cal = Calendar(_weekdays_back(week_start - timedelta(days=1), baseline_days) + _weekdays_between(week_start, last))
    events, replies = synthetic_baseline(school.students, baseline_days)
    for a in absences:
        day = cal.day_of(a.date)
        if day is None:  # weekend or before the baseline: not a school day the scan can place
            continue
        lab = _label(a)
        events.append(AbsenceEvent(id=a.id, student=a.student_id, day=day, true_barrier=lab or "unknown"))
        if lab is not None:
            received = date.fromisoformat(a.reply.received_at[:10]) if a.reply else a.date
            rd = max(day, cal.last_on_or_before(received))
            replies.append(Reply(absence_id=a.id, received_day=rd, language=a.reply.language if a.reply else "en",
                                 text="", true_label=lab, observed_label=lab))
    n = len(cal.days)
    run = SimRun(config=SimConfig(name="seed_demo", n_days=n, seed=SYNTHETIC_SEED), school=school, n_days=n,
                 absences=events, replies=replies, events=[], meta={"synthetic_baseline_days": baseline_days})
    return run, cal


def _demo_zones(school: SchoolGraph, params: scan.ScanParams) -> scan.ZoneSet:
    # All layers so every detector finds the layers it asks for; contact and neighborhood are empty
    # (ego={} skips the contact graph, which the seed school does not have).
    return scan.build_zones(school, scan.ALL_LAYERS, params.ks, ego={})


# ---------------------------------------------------------------------------------------------
# hypergeometric p-value for rule-based clusters


def hypergeom_p(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(population N, K successes, n draws)."""
    from scipy.stats import hypergeom
    return float(hypergeom.sf(k - 1, N, K, n))


# ---------------------------------------------------------------------------------------------
# public entry point


def detect(students: Dict[str, Student], absences: Iterable[Absence], settings: RevenueSettings,
           week_start: date, week_end: date, exclude: Iterable[str] = ()) -> Tuple[List[Cluster], dict]:
    """Safety-labeled absences never enter clustering, so no cluster cites a disclosure. `exclude`
    optionally drops students entirely (the store does not: staff still see them, plans withhold them)."""
    if dataset() == "sim":
        return detect_sim(settings, week_end)
    return detect_demo(students, list(absences), settings, week_start, week_end, exclude=exclude)


def detect_demo(students: Dict[str, Student], absences: List[Absence], settings: RevenueSettings,
                week_start: date, week_end: date, how: Optional[str] = None,
                detector: Optional[str] = None, exclude: Iterable[str] = ()) -> Tuple[List[Cluster], dict]:
    how = how or method()
    det = detector or detector_name()
    exclude = set(exclude)
    # Safety-labeled absences are dropped too, whoever they belong to (defense in depth).
    absences = [a for a in absences if a.student_id not in exclude
                and not (a.classification and a.classification.safety_critical)]
    by_id = {a.id: a for a in absences}
    clusters: List[Cluster] = []
    meta = {"dataset": "demo", "method": how, "detector": det, "zones": ["section", "route", "grade"],
            "household_unit": True, "contact_graph": False, "excluded_students": len(exclude),
            "notes": ["The seed school has no contact graph: zones are class section, bus route and grade only; "
                      "households are counted once and are never a cluster on their own."]}

    if how in ("auto", "scan"):
        params = scan.ScanParams(baseline_days=SYNTHETIC_BASELINE_DAYS + 5, n_perm=DEMO_N_PERM)
        run, cal = demo_run(students, absences, week_start, week_end)
        week_days = [cal.day_of(d) for d in _weekdays_between(week_start, cal.days[-1])]
        alarms = _run_detector(det, run, [d for d in week_days if d is not None], params, _demo_zones(run.school, params))
        for g in _group(alarms, now=run.n_days - 1):
            c = _demo_cluster_from_scan(g, det, run, cal, students, by_id, settings, week_start, week_end)
            if c is not None:
                clusters.append(c)
        meta["synthetic_baseline"] = {"days": SYNTHETIC_BASELINE_DAYS, "rate": SYNTHETIC_RATE,
                                      "reply_rate": SYNTHETIC_REPLY_RATE, "seed": SYNTHETIC_SEED,
                                      "barrier_mix": BACKGROUND_MIX}
        meta["scan_params"] = {"baseline_days": params.baseline_days, "n_perm": params.n_perm,
                               "windows": list(params.windows), "q": 0.10}
        meta["notes"].append("No 40-day history exists for the seed week, so the scan runs over a synthetic quiet "
                             "baseline; scan p-values are relative to that invented history.")

    if how in ("auto", "rules"):
        seen = {(c.link_type, c.link_value, c.barrier) for c in clusters}
        for c in _demo_rule_clusters(students, absences, settings, week_start, week_end):
            if (c.link_type, c.link_value, c.barrier) not in seen:
                clusters.append(c)
        if how == "auto":
            meta["notes"].append("Rule-based clusters are added for any (link, zone, cause) the scan did not report.")

    clusters.sort(key=lambda c: (c.p_value if c.p_value is not None else 1.0, -len(c.student_ids), c.id))
    meta["counts"] = {m: sum(1 for c in clusters if c.method == m) for m in sorted({c.method for c in clusters})}
    return clusters, meta


def _member_absences(members: Iterable[str], cause: str, start: date, end: date, absences: Iterable[Absence]) -> List[Absence]:
    """Absences of cluster members within [start, end] that carry the cause or are still unlabeled."""
    members = set(members)
    out = []
    for a in absences:
        if a.student_id in members and start <= a.date <= end:
            lab = a.classification.barrier.value if a.classification else None
            if lab is None or lab == cause:
                out.append(a)
    return sorted(out, key=lambda a: (a.date, a.id))


def _ablation(students: Dict[str, Student], absences: List[Absence], cause: str, layer: str, value: str,
              start: date, end: date, n_members: int) -> Ablation:
    """Students each signal alone would group (window +- 3 days): same reason anywhere / same zone any reason."""
    lo, hi = start - timedelta(days=clustering.WINDOW_DAYS), end + timedelta(days=clustering.WINDOW_DAYS)
    text_only = {a.student_id for a in absences
                 if a.classification and a.classification.barrier.value == cause and lo <= a.date <= hi}

    def in_zone(s: Student) -> bool:
        return {"route": s.bus_route, "section": s.classroom, "grade": f"G{s.grade}"}.get(layer) == value
    network_only = {a.student_id for a in absences if in_zone(students[a.student_id]) and lo <= a.date <= hi}
    return Ablation(text_only=len(text_only), network_only=len(network_only), combined=n_members)


def _provisional(evidence: List[Absence], cause: str, n_members: int) -> bool:
    support = len({a.student_id for a in evidence if a.classification and a.classification.barrier.value == cause})
    return support < 2 or support < 0.5 * n_members


def _messages(abs_: Iterable[Absence]) -> List[str]:
    return sorted({a.reply.message_id or f"M-{a.id}" for a in abs_ if a.reply})


def _where(link_type: str, value: str) -> str:
    return {"bus_route": f"riders of {value}", "classroom": f"students in {value}", "grade": f"students in grade {value[1:]}"
            }.get(link_type, f"students in {value}")


def _summary(link_type: str, value: str, cause: str, n: int, dates: Iterable[date], provisional: bool) -> str:
    days = ", ".join(d.strftime("%a %b %-d") for d in sorted(set(dates)))
    cause_txt = f"likely {cause} (provisional)" if provisional else f"for {cause}"
    return f"{n} {_where(link_type, value)} absent {cause_txt} ({days})."


def _demo_cluster_from_scan(g: _Group, det: str, run: SimRun, cal: Calendar, students: Dict[str, Student],
                            by_id: Dict[str, Absence], settings: RevenueSettings, week_start: date,
                            week_end: date) -> Optional[Cluster]:
    real = [by_id[i] for i in g.evidence if i in by_id]       # synthetic baseline ids are dropped
    members = sorted({a.student_id for a in real})
    if len({students[s].household_id for s in members}) < 2:
        return None
    best = g.best
    start, last = min(a.date for a in real), max(a.date for a in real)
    absn = _member_absences(members, g.cause, start, last, by_id.values())
    link = LINK_OF_LAYER.get(g.layer, g.layer)
    value = g.zone_id
    week = [a for a in absn if week_start <= a.date <= week_end]
    prov = _provisional(real, g.cause, len(members))
    revenue = dollars(len(week), settings)
    note = (f"Kulldorff space-time permutation scan ({det}), {best.window}-day window, family-wise Monte Carlo "
            f"p-value over {DEMO_N_PERM} permutations with BH q=0.10. The 35 school days before this week are a "
            f"synthetic quiet baseline (the seed has one week), so p is relative to that invented history.")
    return Cluster(
        id=cluster_id(link, value, g.cause, start), barrier=Barrier(g.cause), link_type=link, link_value=value,
        student_ids=members, absence_ids=[a.id for a in absn], first_date=start, last_date=last,
        dollars_at_risk_week=revenue,
        ablation=_ablation(students, list(by_id.values()), g.cause, g.layer, value, start, last, len(members)),
        summary=_summary(link, value, g.cause, len(members), [a.date for a in absn], prov),
        layers_involved=[g.layer] + ([] if det == "graph_only" else ["labels"]),
        zone=f"{g.layer}:{value}", member_ids=members, likely_cause=Barrier(g.cause), cause_provisional=prov,
        observed=float(best.observed), expected=round(float(best.expected), 3), p_value=float(best.p_value),
        start_date=start, detected_on=cal.days[g.first_day], weekly_revenue_at_risk=revenue,
        evidence_message_ids=_messages(a for a in real if a.classification),
        method=f"scan:{det}+synthetic_baseline", method_note=note, dataset="demo",
    )


def _demo_rule_clusters(students: Dict[str, Student], absences: List[Absence], settings: RevenueSettings,
                        week_start: date, week_end: date) -> List[Cluster]:
    """Rule-based fallback: same barrier + shared bus route or classroom, >= 3 students within 3 days.
    Household links are not clusters (PRD 2 section 3.2)."""
    out: List[Cluster] = []
    N = len(students)
    for c in clustering.detect_clusters(students, absences, settings, week_start, week_end):
        if c.link_type == "household":
            continue
        members = c.student_ids
        if len({students[s].household_id for s in members}) < 2:
            continue
        cause = c.barrier.value
        attr = "bus_route" if c.link_type == "bus_route" else "classroom"
        K = sum(1 for s in students.values() if getattr(s, attr) == c.link_value)
        span = [a for a in absences if a.classification and a.classification.barrier.value == cause
                and c.first_date <= a.date <= c.last_date]
        n = len({a.student_id for a in span})
        k = len(members)
        p = hypergeom_p(k, N, K, n)
        absn = _member_absences(members, cause, c.first_date, c.last_date, absences)
        week = [a for a in absn if week_start <= a.date <= week_end]
        revenue = dollars(len(week), settings)
        layer = LAYER_OF_LINK[c.link_type]
        evidence = [a for a in absences if a.id in set(c.absence_ids)]
        # The rule fires once MIN_SIZE distinct students have a labeled reply: the MIN_SIZE-th earliest.
        ready: Dict[str, date] = {}
        for a in evidence:
            got = date.fromisoformat(a.reply.received_at[:10]) if a.reply else a.date
            ready[a.student_id] = min(ready.get(a.student_id, got), max(got, a.date))
        detected = sorted(ready.values())[min(clustering.MIN_SIZE, len(ready)) - 1]
        out.append(c.model_copy(update=dict(
            id=cluster_id(c.link_type, c.link_value, cause, c.first_date),
            absence_ids=[a.id for a in absn], dollars_at_risk_week=revenue,
            layers_involved=[layer, "labels"], zone=f"{layer}:{c.link_value}", member_ids=members,
            likely_cause=c.barrier, cause_provisional=False, observed=float(k), expected=round(n * K / N, 3),
            p_value=p, start_date=c.first_date, detected_on=detected, weekly_revenue_at_risk=revenue,
            evidence_message_ids=_messages(evidence), method="rule_based",
            method_note=(f"Rule-based fallback (no 40-day baseline for the seed week): >= {clustering.MIN_SIZE} students "
                         f"with the same barrier sharing a {c.link_type.replace('_', ' ')} within "
                         f"{clustering.WINDOW_DAYS} days. p is an exact one-sided hypergeometric test: {k} of the {n} "
                         f"students with '{cause}' in the span are in this zone of {K} of {N} students "
                         f"(expected {n * K / N:.2f}); not corrected for the number of zones tried."),
            dataset="demo",
        )))
    return out


# ---------------------------------------------------------------------------------------------
# simulated Lyon school (DATASET=sim)

_SIM_CACHE: Dict[tuple, Tuple[SimRun, List[Alarm], Calendar]] = {}


def _sim_alarms(det: str, end: date) -> Tuple[SimRun, List[Alarm], Calendar]:
    key = (det, end)
    if key not in _SIM_CACHE:
        from . import graphs, simulator
        school = graphs.assign_logistics(graphs.load_school("lyon_primary_2009"), seed=0)
        run = simulator.simulate(school, simulator.CONFIG_A)
        params = scan.ScanParams(n_perm=SIM_PERM)
        days = range(min(params.first_day(), run.n_days - 1), run.n_days)
        alarms = _run_detector(det, run, days, params, None)
        _SIM_CACHE.clear()
        _SIM_CACHE[key] = (run, alarms, Calendar(_weekdays_back(end, run.n_days)))
    return _SIM_CACHE[key]


def detect_sim(settings: RevenueSettings, end: date, detector: Optional[str] = None) -> Tuple[List[Cluster], dict]:
    """Clusters over the whole prospective replay of a simulated Lyon run (day indices mapped to the
    weekdays ending at `end`). Revenue counts member absences in the cluster's last 5 school days."""
    det = detector or detector_name()
    run, alarms, cal = _sim_alarms(det, end)
    lab: Dict[str, str] = {}
    for r in sorted(run.replies, key=lambda r: r.received_day):
        lab[r.absence_id] = r.observed_label
    by_id = {a.id: a for a in run.absences}
    has_reply = {r.absence_id for r in run.replies}
    out: List[Cluster] = []
    for g in _group(alarms, now=run.n_days - 1):
        ev = [by_id[i] for i in g.evidence if i in by_id]
        if not ev:
            continue
        members = sorted({a.student for a in ev})
        d0, d1 = min(a.day for a in ev), max(a.day for a in ev)
        start, last = cal.days[d0], cal.days[d1]
        best = g.best
        support = len({a.student for a in ev if lab.get(a.id) == g.cause})
        prov = support < 2 or support < 0.5 * len(members)
        n_days = sum(1 for a in run.absences if a.student in set(members) and d1 - 4 <= a.day <= d1)
        revenue = dollars(n_days, settings)
        link = LINK_OF_LAYER.get(g.layer, g.layer)
        lo, hi = d0 - 3, d1 + 3
        text_only = {a.student for a in run.absences if lab.get(a.id) == g.cause and lo <= a.day <= hi}
        network_only = {a.student for a in run.absences if a.student in best.zone and lo <= a.day <= hi}
        out.append(Cluster(
            id=cluster_id(link, g.zone_id, g.cause, start), barrier=Barrier(g.cause), link_type=link,
            link_value=g.zone_id, student_ids=members, absence_ids=[a.id for a in ev], first_date=start,
            last_date=last, dollars_at_risk_week=revenue,
            ablation=Ablation(text_only=len(text_only), network_only=len(network_only), combined=len(members)),
            summary=f"{len(members)} students in {g.layer} zone {g.zone_id} absent, likely {g.cause}"
                    f"{' (provisional)' if prov else ''}.",
            layers_involved=[g.layer] + ([] if det == "graph_only" else ["labels"]),
            zone=f"{g.layer}:{g.zone_id}", member_ids=members, likely_cause=Barrier(g.cause), cause_provisional=prov,
            observed=float(best.observed), expected=round(float(best.expected), 3), p_value=float(best.p_value),
            start_date=start, detected_on=cal.days[g.first_day], weekly_revenue_at_risk=revenue,
            evidence_message_ids=sorted(f"M-{a.id}" for a in ev if a.id in has_reply),
            method=f"scan:{det}",
            method_note=(f"Kulldorff space-time permutation scan ({det}) on the simulated Lyon primary school "
                         f"(SocioPatterns contact graph + synthetic logistics, simulator CONFIG_A), 40-day baseline, "
                         f"{SIM_PERM} permutations, BH q=0.10, run prospectively every day."),
            dataset="sim",
        ))
    out.sort(key=lambda c: (c.detected_on, c.p_value, c.id), reverse=True)
    meta = {"dataset": "sim", "method": "scan", "detector": det, "school": run.school.name,
            "students": len(run.school.students), "days": run.n_days, "config": run.config.name,
            "planted_events": len(run.events), "zones": list(scan.SPATIAL_LAYERS) + ["grade"],
            "notes": ["Simulated school: student ids are re-keyed (N000...), there are no real families, "
                      "so action plans cannot be drafted from these clusters."],
            "counts": {f"scan:{det}": len(out)}}
    return out, meta
