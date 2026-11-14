"""Ablation metrics for the PRD 2 cluster detectors (PRD 2 section 4).

Definitions
  * Event e = (barrier b_e, absentees M_e, onset t_e, duration D_e); alarm a = (zone Z_a, barrier b_a, day t_a).
  * a is a true positive for e when b_a == b_e, purity(a, e) >= 0.5 and t_a in [t_e, t_e + D_e + 2].
    purity = |absent in Z_a that are absent because of e| / |absent in Z_a|, where "absent in Z_a" is the
    ground-truth all-cause set of zone members absent during the alarm's window (detector-independent,
    so a whole-school alarm only counts when the event dominates the school's absences).
  * Alarms with the same attributed barrier whose zones overlap with Jaccard > 0.5 on days at most
    `max_gap` apart merge into one episode. A false alarm is an episode with no true positive for any
    should_alarm event (alarms on school-wide flu waves, should_alarm=False, are therefore false alarms).
  * Event recall = should_alarm events with >= 1 true positive / evaluable should_alarm events
    (onset inside the scanned day range).
  * Equal false-alarm rate: each detector's p-value threshold is chosen on null runs so false alarms are
    at most 1 per 20 scanned school days, then applied to event runs.
  * Delay d = first TP day - onset; lead vs EARS-C1 = d_ears - d_ours, paired per event.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .detectors import DayResult, ears_c1, jaccard, select
from .scan import BARRIERS
from .types import Alarm, PlantedEvent, Reply, SimRun

SLACK_DAYS = 2
CATEGORIES = ("illness", "transportation", "other")


def category(barrier: Optional[str]) -> str:
    if barrier in ("illness", "transportation"):
        return barrier
    return "other"


# ----------------------------------------------------------------------------------------------
# matching


@dataclass
class TruthIndex:
    absent: Dict[int, Dict[str, Optional[str]]]          # day -> student -> event_id (None = background)


def truth_index(run: SimRun) -> TruthIndex:
    absent: Dict[int, Dict[str, Optional[str]]] = {}
    for a in run.absences:
        d = absent.setdefault(a.day, {})
        if d.get(a.student) is None:
            d[a.student] = a.event_id
    return TruthIndex(absent)


def _window_absent(alarm: Alarm, truth: TruthIndex) -> Dict[str, set]:
    """Zone members absent during the alarm window -> set of event ids behind those absences."""
    out: Dict[str, set] = {}
    for d in range(alarm.day - alarm.window + 1, alarm.day + 1):
        for s, eid in truth.absent.get(d, {}).items():
            if s in alarm.zone:
                out.setdefault(s, set()).add(eid)
    return out


def purity(alarm: Alarm, event: PlantedEvent, truth: TruthIndex) -> float:
    absent = _window_absent(alarm, truth)
    if not absent:
        return 0.0
    hit = sum(1 for s, eids in absent.items() if event.id in eids)
    return hit / len(absent)


def in_interval(day: int, event: PlantedEvent, slack: int = SLACK_DAYS) -> bool:
    return event.onset <= day <= event.onset + event.duration + slack


def is_true_positive(alarm: Alarm, event: PlantedEvent, truth: TruthIndex, min_purity: float = 0.5,
                     slack: int = SLACK_DAYS, check_barrier: bool = True) -> bool:
    if check_barrier and alarm.barrier != event.barrier:
        return False
    if not in_interval(alarm.day, event, slack):
        return False
    return purity(alarm, event, truth) >= min_purity


def merge_alarms(alarms: Iterable[Alarm], threshold: float = 0.5, max_gap: int = 2) -> List[List[Alarm]]:
    """Group alarms into episodes: same barrier, Jaccard > threshold with the episode's latest zone,
    and at most max_gap days after the episode's latest alarm."""
    episodes: List[List[Alarm]] = []
    for a in sorted(alarms, key=lambda x: (x.day, x.p_value)):
        best = None
        for ep in episodes:
            last = ep[-1]
            if last.barrier == a.barrier and 0 <= a.day - last.day <= max_gap and jaccard(a.zone, last.zone) > threshold:
                best = ep
                break
        if best is None:
            episodes.append([a])
        else:
            best.append(a)
    return episodes


@dataclass
class EventOutcome:
    event_id: str
    kind: str
    barrier: str
    onset: int
    duration: int
    first_tp_day: Optional[int] = None
    first_tp_layer: Optional[str] = None
    tp_layers: List[str] = field(default_factory=list)
    attributed: Optional[str] = None      # barrier of the first localized alarm (barrier ignored)

    @property
    def detected(self) -> bool:
        return self.first_tp_day is not None

    @property
    def delay(self) -> Optional[int]:
        return None if self.first_tp_day is None else self.first_tp_day - self.onset


@dataclass
class RunScore:
    n_days: int
    n_alarms: int
    n_episodes: int
    n_false_alarms: int
    events: List[EventOutcome]
    n_events_excluded: int = 0


def score_run(alarms: Sequence[Alarm], run: SimRun, days: Sequence[int], truth: Optional[TruthIndex] = None,
              min_purity: float = 0.5, slack: int = SLACK_DAYS, merge_gap: int = 2) -> RunScore:
    days = list(days)
    lo, hi = (min(days), max(days)) if days else (0, -1)
    truth = truth or truth_index(run)
    targets = [e for e in run.events if e.should_alarm and lo <= e.onset <= hi]
    excluded = sum(1 for e in run.events if e.should_alarm and not lo <= e.onset <= hi)
    outcomes = {e.id: EventOutcome(e.id, e.kind, e.barrier, e.onset, e.duration) for e in targets}
    tp_alarm = set()
    pur_cache: Dict[Tuple[int, str], float] = {}
    for i, a in enumerate(sorted(alarms, key=lambda x: (x.day, x.p_value))):
        for e in targets:
            if not in_interval(a.day, e, slack):
                continue
            key = (id(a), e.id)
            pur = pur_cache.get(key)
            if pur is None:
                pur = pur_cache[key] = purity(a, e, truth)
            if pur < min_purity:
                continue
            o = outcomes[e.id]
            if o.attributed is None:
                o.attributed = a.barrier
            if a.barrier == e.barrier:
                tp_alarm.add(id(a))
                if o.first_tp_day is None:
                    o.first_tp_day, o.first_tp_layer = a.day, a.layer
                if a.layer not in o.tp_layers:
                    o.tp_layers.append(a.layer)
    episodes = merge_alarms(alarms, max_gap=merge_gap)
    n_fa = sum(1 for ep in episodes if not any(id(a) in tp_alarm for a in ep))
    return RunScore(len(days), len(alarms), len(episodes), n_fa, list(outcomes.values()), excluded)


def ears_outcomes(run: SimRun, days: Sequence[int], slack: int = SLACK_DAYS) -> Dict[str, Optional[int]]:
    """EARS-C1 delay per evaluable should_alarm event: first EARS alarm day in the event interval - onset."""
    days = list(days)
    lo, hi = min(days), max(days)
    alarm_days = [a.day for a in ears_c1(run)]
    out: Dict[str, Optional[int]] = {}
    for e in run.events:
        if not e.should_alarm or not lo <= e.onset <= hi:
            continue
        hit = [d for d in alarm_days if in_interval(d, e, slack)]
        out[e.id] = (min(hit) - e.onset) if hit else None
    return out


def ears_false_alarms(run: SimRun, days: Sequence[int], slack: int = SLACK_DAYS) -> int:
    """EARS alarm days (within `days`) that fall in no should_alarm event interval."""
    days = set(days)
    targets = [e for e in run.events if e.should_alarm]
    return sum(1 for a in ears_c1(run) if a.day in days and not any(in_interval(a.day, e, slack) for e in targets))


# ----------------------------------------------------------------------------------------------
# thresholds and aggregation


def alarms_at(results: Sequence[DayResult], p_max=None, q: Optional[float] = None) -> List[Alarm]:
    out: List[Alarm] = []
    for r in results:
        out.extend(select(r, p_max=p_max, q=q))
    return out


def false_alarm_rate(per_run: Sequence[Sequence[DayResult]], runs: Sequence[SimRun], p_max=None,
                     q: Optional[float] = None, truths: Optional[Sequence[TruthIndex]] = None) -> Tuple[int, int]:
    """(false alarms, scanned days) summed over runs."""
    fa = nd = 0
    for i, (res, run) in enumerate(zip(per_run, runs)):
        days = [r.day for r in res]
        s = score_run(alarms_at(res, p_max, q), run, days, truths[i] if truths else None)
        fa += s.n_false_alarms
        nd += s.n_days
    return fa, nd


def calibrate_threshold(per_run: Sequence[Sequence[DayResult]], runs: Sequence[SimRun],
                        target_per_day: float = 1 / 20, cap: Optional[float] = None) -> Dict[str, object]:
    """Loosest threshold whose false-alarm rate on these (null) runs is <= target_per_day.

    Candidates are ordered by (p_value, -llr); the threshold is the last such key that keeps the
    rate on target (false alarms are non-decreasing along that order, up to episode merging), so
    ties in Monte Carlo p-values are broken by LLR. Returns the threshold as `threshold` = (p, llr)
    for detectors.select / alarms_at, plus the null false-alarm count it gives."""
    truths = [truth_index(r) for r in runs]
    keys = sorted({(a.p_value, -a.llr) for res in per_run for r in res for a in r.candidates
                   if cap is None or a.p_value <= cap})
    n_days = sum(len(res) for res in per_run)
    allowed = target_per_day * n_days

    def thr_of(i):
        return (keys[i][0], -keys[i][1])

    def fa(i):
        return false_alarm_rate(per_run, runs, p_max=thr_of(i), truths=truths)[0]

    if not keys:
        thr = (cap if cap is not None else 1.0, -math.inf)
    else:
        lo, hi = -1, len(keys) - 1               # invariant: fa(lo) <= allowed (lo = -1: nothing passes)
        if fa(hi) <= allowed:
            lo = hi
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if fa(mid) <= allowed:
                    lo = mid
                else:
                    hi = mid
        if lo < 0:
            thr = (keys[0][0], math.inf)
        elif lo == len(keys) - 1 and cap is not None:
            thr = (cap, -math.inf)
        else:
            thr = thr_of(lo)
    n_fa, nd = false_alarm_rate(per_run, runs, p_max=thr, truths=truths)
    return {"threshold": (float(thr[0]), float(np.clip(thr[1], -1e9, 1e9))), "p_threshold": float(thr[0]),
            "llr_tiebreak": float(np.clip(thr[1], -1e9, 1e9)), "null_false_alarms": n_fa, "null_days": nd,
            "null_fa_per_20_days": 20.0 * n_fa / max(nd, 1)}


def _quantiles(x: Sequence[float]) -> Dict[str, Optional[float]]:
    x = [v for v in x if v is not None]
    if not x:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    a = np.asarray(x, dtype=float)
    return {"n": int(a.size), "median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75))}


def summarize(scores: Sequence[RunScore], slack: int = SLACK_DAYS) -> dict:
    events = [e for s in scores for e in s.events]
    n = len(events)
    det = [e for e in events if e.detected]
    nd = sum(s.n_days for s in scores)
    fa = sum(s.n_false_alarms for s in scores)
    by_kind: Dict[str, Dict[str, int]] = {}
    for e in events:
        k = by_kind.setdefault(e.kind, {"n": 0, "detected": 0})
        k["n"] += 1
        k["detected"] += int(e.detected)
    confusion = {c: {a: 0 for a in CATEGORIES + ("missed",)} for c in CATEGORIES}
    for e in events:
        col = "missed" if e.attributed is None else category(e.attributed)
        confusion[category(e.barrier)][col] += 1
    layers: Dict[str, int] = {}
    for e in det:
        layers[e.first_tp_layer] = layers.get(e.first_tp_layer, 0) + 1
    contagion = [e for e in events if e.kind == "contagion"]
    return {
        "n_events": n,
        "n_events_excluded": sum(s.n_events_excluded for s in scores),
        "recall": (len(det) / n) if n else None,
        "recall_by_kind": {k: {**v, "recall": v["detected"] / v["n"]} for k, v in by_kind.items()},
        "pct_detected_by_onset_plus_2": (sum(1 for e in det if e.delay <= 2) / n) if n else None,
        "delay": _quantiles([e.delay for e in det]),
        "false_alarms": fa,
        "days": nd,
        "fa_per_20_days": 20.0 * fa / max(nd, 1),
        "attribution_confusion": confusion,
        "first_tp_layer": layers,
        "contagion_tp_via": {
            "contact_only": sum(1 for e in contagion if "contact" in e.tp_layers and "section" not in e.tp_layers),
            "section_only": sum(1 for e in contagion if "section" in e.tp_layers and "contact" not in e.tp_layers),
            "both": sum(1 for e in contagion if "contact" in e.tp_layers and "section" in e.tp_layers),
        },
    }


def lead_vs_baseline(scores: Sequence[RunScore], ears: Sequence[Dict[str, Optional[int]]]) -> dict:
    """Paired lead = d_ears - d_ours over events both detect; plus counts of one-sided detections."""
    leads, only_ours, only_ears, neither = [], 0, 0, 0
    ears_delays = []
    for s, eo in zip(scores, ears):
        for e in s.events:
            de = eo.get(e.event_id)
            if de is not None:
                ears_delays.append(de)
            if e.detected and de is not None:
                leads.append(de - e.delay)
            elif e.detected:
                only_ours += 1
            elif de is not None:
                only_ears += 1
            else:
                neither += 1
    return {"lead": _quantiles(leads), "only_ours": only_ours, "only_baseline": only_ears, "neither": neither}


def summarize_ears(runs: Sequence[SimRun], days_per_run: Sequence[Sequence[int]]) -> dict:
    delays, n, fa, nd = [], 0, 0, 0
    for run, days in zip(runs, days_per_run):
        eo = ears_outcomes(run, days)
        n += len(eo)
        delays.extend(eo.values())
        fa += ears_false_alarms(run, days)
        nd += len(list(days))
    det = [d for d in delays if d is not None]
    return {"n_events": n, "recall": (len(det) / n) if n else None,
            "pct_detected_by_onset_plus_2": (sum(1 for d in det if d <= 2) / n) if n else None,
            "delay": _quantiles(det), "false_alarm_days": fa, "days": nd,
            "fa_per_20_days": 20.0 * fa / max(nd, 1)}


# ----------------------------------------------------------------------------------------------
# label noise


def relabel(run: SimRun, accuracy: float, seed: int) -> SimRun:
    """Copy of `run` whose replies' observed_label is the true label with probability `accuracy`,
    otherwise a uniformly drawn different barrier. Events and absences are unchanged, so a sweep over
    `accuracy` is paired. Replies whose true label is not a barrier are left as they are."""
    rng = np.random.default_rng([seed, int(round(accuracy * 1000))])
    reps: List[Reply] = []
    for r in run.replies:
        if r.true_label not in BARRIERS:
            reps.append(r)
            continue
        if rng.random() < accuracy:
            lab = r.true_label
        else:
            others = [b for b in BARRIERS if b != r.true_label]
            lab = others[int(rng.integers(len(others)))]
        reps.append(dataclasses.replace(r, observed_label=lab))
    return dataclasses.replace(run, replies=reps)
