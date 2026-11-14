"""The three PRD 2 cluster detectors (same scan, different inputs) and the EARS-C1 baseline.

    content_only(run, as_of)  labeled observations, zones = whole school (barrier-as-space) + grade
    graph_only(run, as_of)    all-cause absences (pseudo-barrier "any"), zones = section / route /
                              neighborhood / contact; cause guessed from the zone type
    combined(run, as_of)      labeled observations, all zones (section / route / neighborhood /
                              contact / grade); cause = the scanned barrier
    graph_labels(run, as_of)  (beyond the PRD's three) graph-only's all-cause scan, but the cause is the
                              majority label among the alarm's absences with a reply by as_of
                              (zone-type guess when none has one): labels for attribution only
    run_prospective(run, detector, days)   the same, run every day (as replay would)
    ears_c1(run, days)        school-wide all-cause count > mean + 2 sd of the previous 7 days

By default each detector applies Benjamini-Hochberg (q = 0.10) across its (layer, barrier) families
at that as_of and returns the significant candidates, deduplicated. Pass p_max instead to get every
candidate with p <= p_max (what the evaluation does to calibrate thresholds).

For batch work use Pipeline: it builds zones and observation indexes once per run and computes
content-only and combined from one label scan (their families are identical where they overlap).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .scan import (ALL_LAYERS, ANY, BARRIER_CODE, BARRIERS, LABELED, SPATIAL_LAYERS, FamilyResult, Observations,
                   ScanParams, ZoneSet, bh_reject, build_zones, scan_day)
from .types import Alarm, SimRun

DETECTORS: Tuple[str, ...] = ("content_only", "graph_only", "combined")
ALL_DETECTORS: Tuple[str, ...] = DETECTORS + ("graph_labels",)
LABEL_DETECTORS: Tuple[str, ...] = ("content_only", "combined", "graph_labels")
DETECTOR_LAYERS: Dict[str, Tuple[str, ...]] = {
    "content_only": ("school", "grade"),
    "graph_only": SPATIAL_LAYERS,
    "graph_labels": SPATIAL_LAYERS,
    # All zones. The whole-school zone is left out: under the space-time permutation statistic its
    # LLR is identically 0 (c == mu), and the barrier-as-space substitute content-only uses would
    # re-admit school-wide flu waves, which the spatial scan is designed to condition out.
    "combined": SPATIAL_LAYERS + ("grade",),
}
# Graph-only cause guess by zone type (PRD 2 section 4, detector 2).
ZONE_ATTRIBUTION: Dict[str, str] = {
    "route": "transportation",
    "contact": "illness",
    "section": "illness",
    "neighborhood": "caregiving or family obligation",
}
_LAYER_RANK = {l: i for i, l in enumerate(("route", "contact", "section", "neighborhood", "grade", "school"))}


@dataclass
class DayResult:
    detector: str
    day: int
    candidates: List[Alarm]                    # every candidate with p <= params.report_p (not deduped)
    families: List[Tuple[str, str, float]]     # (layer, barrier, family p) of every evaluated family


def family_key(alarm: Alarm) -> Tuple[str, str]:
    if alarm.layer == "school":
        return ("school", LABELED)
    if alarm.detector in ("graph_only", "graph_labels"):
        return (alarm.layer, ANY)
    return (alarm.layer, alarm.barrier)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _rank(a: Alarm):
    return (a.p_value, -a.llr, _LAYER_RANK.get(a.layer, 9), a.zone_id)


def dedupe(alarms: Iterable[Alarm], threshold: float = 0.5) -> List[Alarm]:
    """Same-day non-maximum suppression: among alarms on the same day whose zones overlap with
    Jaccard > threshold keep the strongest (lowest p, then highest LLR). Ranking by p rather than raw
    LLR keeps dedupe consistent with p-value thresholding (LLRs of different families are not
    comparable)."""
    by_day: Dict[int, List[Alarm]] = {}
    for a in alarms:
        by_day.setdefault(a.day, []).append(a)
    out: List[Alarm] = []
    for day in sorted(by_day):
        kept: List[Alarm] = []
        for a in sorted(by_day[day], key=_rank):
            if all(jaccard(a.zone, k.zone) <= threshold for k in kept):
                kept.append(a)
        out.extend(kept)
    return out


Threshold = Union[float, Tuple[float, float]]


def passes(alarm: Alarm, threshold: Threshold) -> bool:
    """threshold p: p_value <= p. threshold (p, llr): lexicographic on (p_value, -llr), i.e. p_value < p,
    or p_value == p and llr >= llr. The LLR tie-break gives calibration a resolution finer than the
    1/(n_perm+1) steps of Monte Carlo p-values."""
    if isinstance(threshold, tuple):
        p, l = threshold
        return alarm.p_value < p or (alarm.p_value == p and alarm.llr >= l)
    return alarm.p_value <= threshold


def select(result: DayResult, p_max: Optional[Threshold] = None, q: Optional[float] = 0.10,
           jaccard_threshold: float = 0.5) -> List[Alarm]:
    """Alarms of one day. p_max given: every candidate passing that threshold (see passes()).
    Otherwise BH at q across the day's families; candidates of rejected families with p <= the BH
    cutoff are kept."""
    if p_max is not None:
        cands = [a for a in result.candidates if passes(a, p_max)]
    else:
        if not result.families:
            return []
        ps = np.array([f[2] for f in result.families])
        rej = bh_reject(ps, q if q is not None else 0.10)
        if not rej.any():
            return []
        cutoff = float(ps[rej].max())
        fams = {(f[0], f[1]) for f, r in zip(result.families, rej) if r}
        cands = [a for a in result.candidates if family_key(a) in fams and a.p_value <= cutoff]
    return dedupe(cands, jaccard_threshold)


class Pipeline:
    """Zones + observation indexes for one SimRun; scans any as_of for any detector."""

    def __init__(self, run: SimRun, params: Optional[ScanParams] = None, zones: Optional[ZoneSet] = None):
        self.run = run
        self.params = params or ScanParams()
        self.zones = zones if zones is not None else build_zones(run.school, ALL_LAYERS, self.params.ks)
        self.obs = Observations.from_run(run, self.zones)
        self._aidx = {a: i for i, a in enumerate(self.obs.a_ids)}

    def default_days(self) -> range:
        return range(min(self.params.first_day(), self.run.n_days - 1), self.run.n_days)

    def day(self, as_of: int, detectors: Sequence[str] = DETECTORS) -> Dict[str, DayResult]:
        out: Dict[str, DayResult] = {}
        label_dets = [d for d in detectors if d in ("content_only", "combined")]
        if label_dets:
            layers = sorted({l for d in label_dets for l in DETECTOR_LAYERS[d]}, key=ALL_LAYERS.index)
            fams = scan_day(self.obs, self.zones, as_of, "labels", layers, self.params)
            for d in label_dets:
                out[d] = self._result(d, as_of, [f for f in fams if f.layer in DETECTOR_LAYERS[d]])
        if "graph_only" in detectors or "graph_labels" in detectors:
            fams = scan_day(self.obs, self.zones, as_of, "any", SPATIAL_LAYERS, self.params)
            if "graph_only" in detectors:
                out["graph_only"] = self._result("graph_only", as_of, fams)
            if "graph_labels" in detectors:
                out["graph_labels"] = self._result("graph_labels", as_of, fams, self._label_vote(as_of))
        return out

    def _label_vote(self, as_of: int):
        lab = self.obs.labels_at(as_of)

        def vote(alarm: Alarm) -> str:
            codes = lab[[self._aidx[a] for a in alarm.evidence_absence_ids]]
            codes = codes[codes >= 0]
            if codes.size == 0:
                return ZONE_ATTRIBUTION[alarm.layer]
            counts = np.bincount(codes, minlength=len(BARRIERS))
            best = np.flatnonzero(counts == counts.max())
            guess = BARRIER_CODE.get(ZONE_ATTRIBUTION[alarm.layer])
            return BARRIERS[guess if guess in best else int(best[0])]   # ties -> zone-type guess
        return vote

    def prospective(self, days: Optional[Iterable[int]] = None,
                    detectors: Sequence[str] = DETECTORS) -> Dict[str, List[DayResult]]:
        days = self.default_days() if days is None else days
        out: Dict[str, List[DayResult]] = {d: [] for d in detectors}
        for t in days:
            for d, r in self.day(t, detectors).items():
                out[d].append(r)
        return out

    @staticmethod
    def _result(detector: str, as_of: int, fams: List[FamilyResult], vote=None) -> DayResult:
        cands: List[Alarm] = []
        for f in fams:
            for a in f.candidates:
                if detector == "graph_only":
                    b = ZONE_ATTRIBUTION[a.layer]
                elif detector == "graph_labels":
                    b = vote(a)
                else:
                    b = a.barrier
                cands.append(dataclasses.replace(a, detector=detector, barrier=b))
        return DayResult(detector, as_of, cands, [(f.layer, f.barrier, f.p_value) for f in fams])


def _detect(detector: str, run: SimRun, as_of: int, params: Optional[ScanParams], q: Optional[float],
            p_max: Optional[float], pipeline: Optional[Pipeline]) -> List[Alarm]:
    pipe = pipeline or Pipeline(run, params)
    return select(pipe.day(as_of, (detector,))[detector], p_max=p_max, q=q)


def content_only(run: SimRun, as_of: int, params: Optional[ScanParams] = None, q: Optional[float] = 0.10,
                 p_max: Optional[float] = None, pipeline: Optional[Pipeline] = None) -> List[Alarm]:
    return _detect("content_only", run, as_of, params, q, p_max, pipeline)


def graph_only(run: SimRun, as_of: int, params: Optional[ScanParams] = None, q: Optional[float] = 0.10,
               p_max: Optional[float] = None, pipeline: Optional[Pipeline] = None) -> List[Alarm]:
    return _detect("graph_only", run, as_of, params, q, p_max, pipeline)


def combined(run: SimRun, as_of: int, params: Optional[ScanParams] = None, q: Optional[float] = 0.10,
             p_max: Optional[float] = None, pipeline: Optional[Pipeline] = None) -> List[Alarm]:
    return _detect("combined", run, as_of, params, q, p_max, pipeline)


def graph_labels(run: SimRun, as_of: int, params: Optional[ScanParams] = None, q: Optional[float] = 0.10,
                 p_max: Optional[float] = None, pipeline: Optional[Pipeline] = None) -> List[Alarm]:
    return _detect("graph_labels", run, as_of, params, q, p_max, pipeline)


def run_prospective(run: SimRun, detector: str, days: Optional[Iterable[int]] = None,
                    params: Optional[ScanParams] = None, q: Optional[float] = 0.10,
                    p_max: Optional[float] = None, pipeline: Optional[Pipeline] = None) -> List[Alarm]:
    """Run `detector` every day in `days` (default: every day with a full baseline)."""
    pipe = pipeline or Pipeline(run, params)
    out: List[Alarm] = []
    for r in pipe.prospective(days, (detector,))[detector]:
        out.extend(select(r, p_max=p_max, q=q))
    return out


def daily_counts(run: SimRun) -> np.ndarray:
    counts = np.zeros(run.n_days, dtype=float)
    for a in run.absences:
        if 0 <= a.day < run.n_days:
            counts[a.day] += 1
    return counts


def ears_c1(run: SimRun, days: Optional[Iterable[int]] = None, lookback: int = 7, k: float = 2.0) -> List[Alarm]:
    """EARS-C1: alarm on day t when the school-wide all-cause count exceeds mean + k*sd of days t-7..t-1."""
    counts = daily_counts(run)
    days = range(lookback, run.n_days) if days is None else days
    everyone = frozenset(run.school.students)
    out: List[Alarm] = []
    for t in days:
        if t < lookback:
            continue
        base = counts[t - lookback:t]
        mu, sd = float(base.mean()), float(base.std(ddof=1))
        if counts[t] > mu + k * sd:
            z = (counts[t] - mu) / sd if sd > 0 else float("inf")
            out.append(Alarm(detector="ears_c1", layer="school", zone_id="school", zone=everyone,
                             barrier="unknown", day=int(t), window=1, observed=int(counts[t]), expected=mu,
                             llr=float(z), p_value=float("nan")))
    return out
