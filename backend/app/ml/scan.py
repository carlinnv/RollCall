"""Kulldorff space-time permutation scan over school zones (PRD 2 section 3.2).

Building blocks used by app.ml.detectors:

    llr(c, mu, C)                         Kulldorff log-likelihood ratio (0 where c <= mu)
    bh_reject(pvals, q) / bh_adjust(p)    Benjamini-Hochberg across (layer, barrier) families
    build_zones(school, layers, ks)       ZoneSet: zone list + sparse student x zone membership
    Observations.from_run(run, zones)     absences + replies indexed for fast as_of slicing
    scan_day(obs, zones, as_of, ...)      one prospective scan -> List[FamilyResult]

Records and households
    The scan counts *records*, not raw absences. A record is one household's absence spell for one
    barrier: consecutive absence days of any member of the household collapse into one record dated
    by the spell's first day (ScanParams.spells=False gives one record per household-day instead).
    So siblings absent together count once, and one long absence or a chronic absentee cannot pass
    for a cluster (date permutation would otherwise split a spell into "independent" days). A record
    counts once in a zone when any of its absent members belongs to that zone.

Statistic
    For a family (layer, barrier) at as_of t, the baseline is the last `baseline_days` days up to and
    including t. c = records in zone Z within the last w days, C_Z / C_W the zone / window totals,
    C the baseline total, mu = C_Z * C_W / C. LLR as in llr().

Null
    The days of the baseline records are shuffled across records (students and household grouping
    kept), n_perm times, all at once as a sparse (records x (perm, window)) indicator matrix. The
    family null is the max LLR over every zone and window (cells with c >= min_cases) in the family.
    Each candidate's p-value is its Monte Carlo rank against that family-wise null; candidates beyond
    every permutation get a Gumbel tail value capped at 1/(n_perm+1) (see gumbel_pvalues). The family
    p-value is the p-value of its best candidate.

The "school" layer is special: under a space-time permutation the whole-school zone always has
c == mu, so it carries no signal. For label-based detectors it is scanned with barrier as the
"space" dimension instead: records are all labeled records and zone b is "records labeled b".
mu = (labeled records in window) * (b records in baseline) / (labeled records in baseline), which
detects a barrier-specific school-wide surge while still conditioning out all-cause surges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.special import xlogy

from app.models import Barrier

from .types import Alarm, SchoolGraph, SimRun

BARRIERS: Tuple[str, ...] = tuple(b.value for b in Barrier if b != Barrier.UNKNOWN)
BARRIER_CODE: Dict[str, int] = {b: i for i, b in enumerate(BARRIERS)}
ANY = "any"                                  # pseudo-barrier for all-cause (graph-only) scans
LABELED = "*"                                # barrier key of the school barrier-as-space family

SPATIAL_LAYERS: Tuple[str, ...] = ("section", "route", "neighborhood", "contact")
ALL_LAYERS: Tuple[str, ...] = SPATIAL_LAYERS + ("grade", "school")
_EULER_GAMMA = 0.5772156649015329


@dataclass
class ScanParams:
    baseline_days: int = 40                  # B: baseline length, including the scan window
    windows: Tuple[int, ...] = (1, 2, 3)
    n_perm: int = 199                        # PRD asks for 999; 199 keeps the ablation fast
    min_window_obs: int = 2                  # only scan a barrier with >= this many records in max window
    min_cases: int = 2                       # candidate needs c >= this
    min_units: int = 2                       # candidate needs >= this many distinct households absent
    spells: bool = True                      # count new absence spells (onsets) instead of absence days
    report_p: float = 0.25                   # candidates with p above this are dropped early
    seed: int = 0
    ks: Tuple[int, ...] = (5, 10, 20)

    def first_day(self) -> int:
        """First as_of day with a full baseline."""
        return self.baseline_days - 1


# ----------------------------------------------------------------------------------------------
# statistics


def llr(c, mu, C):
    """Kulldorff LLR: c ln(c/mu) + (C-c) ln((C-c)/(C-mu)) when c > mu, else 0. Vectorized."""
    c = np.asarray(c, dtype=float)
    mu = np.asarray(mu, dtype=float)
    C = np.asarray(C, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = xlogy(c, c / mu) + xlogy(C - c, (C - c) / (C - mu))
    out = np.where(c > mu, val, 0.0)
    out = np.where(np.isfinite(out), out, 0.0)
    return out if out.ndim else float(out)


def bh_reject(pvals: Sequence[float], q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg step-up: boolean mask of rejected hypotheses at FDR level q."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    passed = p[order] <= q * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = int(np.nonzero(passed)[0].max())
        reject[order[: kmax + 1]] = True
    return reject


def bh_adjust(pvals: Sequence[float]) -> np.ndarray:
    """BH-adjusted p-values (q-values): reject at level q iff adjusted <= q."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def gumbel_pvalues(x: np.ndarray, null_max: np.ndarray) -> np.ndarray:
    """Family-wise p-values of observed LLRs `x` against the permutation null maxima.

    The Monte Carlo rank p-value (1 + #{null >= x}) / (n + 1) is used wherever it is resolved. Only an
    observation beyond every null maximum (rank p = 1/(n+1)) gets a finer value, from a Gumbel fitted
    to the null maxima by moments (as in SaTScan), capped at 1/(n+1). The cap matters: with few
    records the null is very discrete and the Gumbel tail is anti-conservative (checked against
    19999 permutations), so it is only trusted to order alarms the permutations cannot separate.
    """
    x = np.asarray(x, dtype=float)
    n = null_max.size
    srt = np.sort(null_max)
    mc = (1.0 + (n - np.searchsorted(srt, x - 1e-9, side="left"))) / (n + 1.0)
    floor = 1.0 / (n + 1.0)
    out = np.where(x <= 0, 1.0, mc)
    top = (mc <= floor) & (x > 0)
    if n < 2 or not top.any():
        return out
    sd = float(null_max.std(ddof=1))
    if not np.isfinite(sd) or sd < 1e-9:
        return out
    beta = sd * math.sqrt(6.0) / math.pi
    loc = float(null_max.mean()) - _EULER_GAMMA * beta
    z = np.clip(-(x[top] - loc) / beta, -700.0, 700.0)
    out[top] = np.clip(-np.expm1(-np.exp(z)), 1e-300, floor)
    return out


# ----------------------------------------------------------------------------------------------
# zones


_EGO_CACHE: Dict[Tuple[int, Tuple[int, ...]], Tuple[object, dict]] = {}


def _ego_nets_cached(school: SchoolGraph, ks: Tuple[int, ...]) -> dict:
    from .graphs import ego_nets  # local import: graphs is optional for callers that pass `ego`

    key = (id(school.daily), tuple(ks))
    hit = _EGO_CACHE.get(key)
    if hit is not None and hit[0] is school.daily:
        return hit[1]
    ego = ego_nets(school, ks)
    if len(_EGO_CACHE) > 8:
        _EGO_CACHE.clear()
    _EGO_CACHE[key] = (school.daily, ego)   # keep a reference so the id stays valid
    return ego


@dataclass
class ZoneSet:
    students: List[str]
    sidx: Dict[str, int]
    unit: np.ndarray                         # student index -> household unit index
    layer: List[str]                         # per zone
    zone_id: List[str]
    members: List[FrozenSet[str]]
    S2Z: sp.csr_matrix                       # students x zones, binary
    layer_cols: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_zones(self) -> int:
        return len(self.zone_id)


def build_zones(school: SchoolGraph, layers: Iterable[str] = ALL_LAYERS, ks: Sequence[int] = (5, 10, 20),
                ego: Optional[Dict[Tuple[str, int], FrozenSet[str]]] = None) -> ZoneSet:
    """Zones per layer. Identical member sets within a layer are kept once; zones made of a single
    household are dropped (siblings absent together are not a cluster). The "school" zone is kept
    for bookkeeping (see module doc)."""
    students = list(school.students)
    sidx = {s: i for i, s in enumerate(students)}
    hh = {}
    unit = np.array([hh.setdefault(school.household.get(s, s), len(hh)) for s in students], dtype=np.int64)

    groups: List[Tuple[str, str, FrozenSet[str]]] = []

    def by_attr(layer: str, attr: Dict[str, object], prefix: str = "") -> None:
        buckets: Dict[object, set] = {}
        for s in students:
            v = attr.get(s)
            if v is not None:
                buckets.setdefault(v, set()).add(s)
        for v in sorted(buckets, key=str):
            groups.append((layer, f"{prefix}{v}", frozenset(buckets[v])))

    layers = tuple(layers)
    for layer in layers:
        if layer == "section":
            by_attr("section", school.section)
        elif layer == "route":
            by_attr("route", school.bus_route)
        elif layer == "neighborhood":
            by_attr("neighborhood", school.neighborhood)
        elif layer == "grade":
            by_attr("grade", school.grade, "G")
        elif layer == "school":
            groups.append(("school", "school", frozenset(students)))
        elif layer == "contact":
            nets = ego if ego is not None else _ego_nets_cached(school, tuple(ks))
            for (s, k) in sorted(nets, key=lambda sk: (sk[1], sk[0])):
                groups.append(("contact", f"ego:{s}:k{k}", frozenset(m for m in nets[(s, k)] if m in sidx)))
        else:
            raise ValueError(f"unknown layer {layer!r}")

    layer_l, ids, mem = [], [], []
    seen = set()
    for layer, zid, m in groups:
        if layer != "school":
            if len({int(unit[sidx[s]]) for s in m}) < 2:
                continue
            if (layer, m) in seen:
                continue
            seen.add((layer, m))
        layer_l.append(layer)
        ids.append(zid)
        mem.append(m)
    rows, cols = [], []
    for j, m in enumerate(mem):
        for s in m:
            rows.append(sidx[s])
            cols.append(j)
    S2Z = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(students), len(mem)))
    lay = np.array(layer_l)
    layer_cols = {L: np.nonzero(lay == L)[0] for L in layers}
    return ZoneSet(students, sidx, unit, layer_l, ids, mem, S2Z, layer_cols)


# ----------------------------------------------------------------------------------------------
# observations


@dataclass
class Observations:
    """Absences of one SimRun, indexed for as_of slicing."""
    n_days: int
    a_student: np.ndarray
    a_unit: np.ndarray
    a_day: np.ndarray
    a_ids: List[str]
    r_abs: np.ndarray                         # reply rows, sorted by (absence, received_day)
    r_day: np.ndarray
    r_code: np.ndarray                        # BARRIER_CODE of observed_label, -1 = unknown

    @classmethod
    def from_run(cls, run: SimRun, zones: ZoneSet) -> "Observations":
        keep = [a for a in run.absences if a.student in zones.sidx]
        a_student = np.array([zones.sidx[a.student] for a in keep], dtype=np.int64)
        a_day = np.array([a.day for a in keep], dtype=np.int64)
        aidx = {a.id: i for i, a in enumerate(keep)}
        reps = [(aidx[r.absence_id], r.received_day, BARRIER_CODE.get(r.observed_label, -1))
                for r in run.replies if r.absence_id in aidx]
        reps.sort()
        arr = np.array(reps, dtype=np.int64).reshape(-1, 3)
        return cls(run.n_days, a_student, zones.unit[a_student] if len(keep) else np.zeros(0, np.int64),
                   a_day, [a.id for a in keep], arr[:, 0], arr[:, 1], arr[:, 2])

    def labels_at(self, as_of: int) -> np.ndarray:
        """Per-absence barrier code known at as_of (latest reply received by then), -1 = unknown."""
        lab = np.full(self.a_day.size, -1, dtype=np.int64)
        m = self.r_day <= as_of
        if m.any():
            a, c = self.r_abs[m], self.r_code[m]
            # rows are sorted by (absence, day): the last row of each absence is the latest reply
            last = np.r_[a[1:] != a[:-1], True]
            lab[a[last]] = c[last]
        return lab


# ----------------------------------------------------------------------------------------------
# scan


@dataclass
class FamilyResult:
    layer: str
    barrier: str                              # BARRIERS value, ANY, or LABELED for the school family
    p_value: float                            # family-wise p (best candidate); 1.0 if no candidate
    n_records: int
    candidates: List[Alarm]


def _records(sel: np.ndarray, obs: Observations, key_extra: Optional[np.ndarray] = None, spells: bool = True):
    """Collapse selected absences into household records.

    A record is a (unit[, label]) group's absence on one day, or with spells=True one run of
    consecutive absence days (dated by its first day inside the baseline). Returns
    (rec_of_absence, rec_day, rec_extra, n_rec); rec_of_absence maps each selected absence to its record.
    """
    nb = len(BARRIERS) + 1
    extra = np.zeros(sel.size, dtype=np.int64) if key_extra is None else np.asarray(key_extra, dtype=np.int64)
    group = obs.a_unit[sel] * nb + extra
    day = obs.a_day[sel]
    span = obs.n_days + 2
    uniq, inv = np.unique(group * span + day, return_inverse=True)   # sorted by (group, day)
    g, d = uniq // span, uniq % span
    if spells:
        start = np.r_[True, (g[1:] != g[:-1]) | (d[1:] != d[:-1] + 1)]
    else:
        start = np.ones(uniq.size, dtype=bool)
    rec = np.cumsum(start) - 1
    rec_extra = None if key_extra is None else (g[start] % nb)
    return rec[inv.ravel()], d[start], rec_extra, int(start.sum())


def _engine(days_rel: np.ndarray, M: sp.csr_matrix, L: int, windows: Sequence[int], n_perm: int,
            rng: np.random.Generator, fam_of_zone: np.ndarray, n_fam: int, min_cases: int = 1):
    """Observed LLRs and family null maxima.

    days_rel: (R,) record day relative to baseline start (as_of = L-1); M: (R x Z) binary.
    Returns obs (dict of arrays for observed zone-window cells with c > 0) and null_max (n_fam, n_perm).
    """
    R, Z = M.shape
    nw = len(windows)
    P = n_perm + 1
    maxw = max(windows)
    age0 = (L - 1) - days_rel                  # 0 = as_of
    C = float(R)
    C_Z = np.asarray(M.sum(axis=0)).ravel()
    C_W = np.array([(age0 < w).sum() for w in windows], dtype=float)

    # Shuffling all record days only matters for the records that land in the last maxw days, so
    # each permutation draws a uniformly random set of C_maxw records (smallest random keys) and hands
    # them the window ages in random order: O(n_perm * R) instead of a full sort per permutation.
    win_age = age0[age0 < maxw]
    k = win_age.size
    if n_perm and k:
        keys = rng.random((n_perm, R))
        pick = np.argpartition(keys, k - 1, axis=1)[:, :k] if k < R else np.tile(np.arange(R), (n_perm, 1))
        order = np.argsort(np.take_along_axis(keys, pick, axis=1), axis=1)
        rec = np.concatenate([np.nonzero(age0 < maxw)[0][None, :], np.take_along_axis(pick, order, axis=1)])
    else:
        rec = np.nonzero(age0 < maxw)[0][None, :]
    ages = np.broadcast_to(win_age, rec.shape)
    p_all = np.broadcast_to(np.arange(rec.shape[0])[:, None], rec.shape)
    rr, cc = [], []
    for wi, w in enumerate(windows):
        m = ages < w
        rr.append(rec[m])
        cc.append(p_all[m] * nw + wi)
    rr = np.concatenate(rr)
    cc = np.concatenate(cc)
    X = sp.csr_matrix((np.ones(rr.size, dtype=np.float32), (rr, cc)), shape=(R, P * nw))
    cnt = (M.T.tocsr() @ X).tocoo()
    # The statistic is the max LLR over cells with c >= min_cases, in the data and in every permutation
    # alike (a one-record "cluster" is never reported), so cells below that are dropped up front.
    big = cnt.data >= min_cases
    z, col, c = cnt.row[big], cnt.col[big], cnt.data[big].astype(float)
    wi = col % nw
    mu = C_Z[z] * C_W[wi] / C
    hot = c > mu                                # LLR is 0 elsewhere: evaluate logs only where needed
    z, col, c, wi, mu = z[hot], col[hot], c[hot], wi[hot], mu[hot]
    with np.errstate(divide="ignore", invalid="ignore"):
        val = c * np.log(c / mu) + xlogy(C - c, (C - c) / (C - mu))
    val[~np.isfinite(val)] = 0.0
    perm = col // nw
    null_max = np.zeros(n_fam * P)
    np.maximum.at(null_max, fam_of_zone[z] * P + perm, val)
    null_max = null_max.reshape(n_fam, P)[:, 1:]
    o = perm == 0
    obs = dict(zone=z[o], wi=wi[o], c=c[o], mu=mu[o], llr=val[o])
    return obs, null_max, age0


def scan_day(obs: Observations, zones: ZoneSet, as_of: int, mode: str, layers: Sequence[str],
             params: Optional[ScanParams] = None) -> List[FamilyResult]:
    """Run the scan at `as_of` and return one FamilyResult per evaluated (layer, barrier) family.

    mode "labels": absences carry the label of a reply received by as_of (others are unknown and
    not scanned); one family per (layer, barrier). The "school" layer becomes the barrier-as-space
    family (layer "school", barrier LABELED).
    mode "any": all-cause absences as the single pseudo-barrier ANY (school layer is ignored).
    """
    params = params or ScanParams()
    B = params.baseline_days
    lo = max(0, as_of - B + 1)
    L = as_of - lo + 1
    in_base = (obs.a_day >= lo) & (obs.a_day <= as_of)
    maxw = max(params.windows)
    out: List[FamilyResult] = []
    spatial = [l for l in layers if l != "school"]

    if mode == "any":
        groups = [(ANY, np.nonzero(in_base)[0])]
    elif mode == "labels":
        lab = obs.labels_at(as_of)
        groups = [(b, np.nonzero(in_base & (lab == code))[0]) for b, code in BARRIER_CODE.items()]
    else:
        raise ValueError(mode)

    if spatial:
        cols = np.concatenate([zones.layer_cols[l] for l in spatial])
        fam_of_zone = np.concatenate([np.full(zones.layer_cols[l].size, i) for i, l in enumerate(spatial)])
        S2Z = zones.S2Z[:, cols].tocsc().tocsr()
        for gi, (barrier, sel) in enumerate(groups):
            if sel.size == 0:
                continue
            inv, rec_day, _, n_rec = _records(sel, obs, spells=params.spells)
            if (rec_day > as_of - maxw).sum() < params.min_window_obs:
                continue
            R2S = sp.csr_matrix((np.ones(sel.size, dtype=np.float32), (inv, obs.a_student[sel])),
                                shape=(n_rec, len(zones.students)))
            M = R2S @ S2Z
            M.data[:] = 1.0
            rng = np.random.default_rng([params.seed, as_of, 1 + gi, 0])
            res, null_max, age0 = _engine(rec_day - lo, M, L, params.windows, params.n_perm, rng,
                                          fam_of_zone, len(spatial), params.min_cases)
            ctx = dict(sel=sel, inv=inv, age0=age0, M=M.tocsc(), cols=cols)
            for fi, layer in enumerate(spatial):
                out.append(_family(layer, barrier, fi, res, null_max[fi], fam_of_zone, ctx, obs, zones,
                                   params, as_of, n_rec))

    if mode == "labels" and "school" in layers:
        sel = np.nonzero(in_base & (lab >= 0))[0]
        if sel.size:
            inv, rec_day, rec_lab, n_rec = _records(sel, obs, key_extra=lab[sel], spells=params.spells)
            if (rec_day > as_of - maxw).sum() >= params.min_window_obs:
                M = sp.csr_matrix((np.ones(n_rec, dtype=np.float32), (np.arange(n_rec), rec_lab)),
                                  shape=(n_rec, len(BARRIERS)))
                rng = np.random.default_rng([params.seed, as_of, 0, 1])
                res, null_max, age0 = _engine(rec_day - lo, M, L, params.windows, params.n_perm, rng,
                                              np.zeros(len(BARRIERS), dtype=np.int64), 1, params.min_cases)
                ctx = dict(sel=sel, inv=inv, age0=age0, M=M.tocsc(), cols=None)
                out.append(_family("school", LABELED, 0, res, null_max[0], None, ctx, obs, zones,
                                   params, as_of, n_rec))
    return out


def _family(layer, barrier, fi, res, null_max, fam_of_zone, ctx, obs, zones, params, as_of, n_rec):
    if fam_of_zone is None:
        m = np.ones(res["zone"].size, dtype=bool)
    else:
        m = fam_of_zone[res["zone"]] == fi
    m &= (res["llr"] > 0) & (res["c"] >= params.min_cases)
    idx = np.nonzero(m)[0]
    if idx.size == 0:
        return FamilyResult(layer, barrier, 1.0, n_rec, [])
    p = gumbel_pvalues(res["llr"][idx], null_max)
    # best window per zone: lowest p, then highest LLR
    order = np.lexsort((res["wi"][idx], -res["llr"][idx], p, res["zone"][idx]))  # ties -> shortest window
    zsorted = res["zone"][idx][order]
    first = np.r_[True, zsorted[1:] != zsorted[:-1]]
    pick = order[first]
    fam_p = float(p.min())
    keep = pick[p[pick] <= params.report_p]
    keep = keep[np.lexsort((-res["llr"][idx][keep], p[keep]))]
    cands: List[Alarm] = []
    for k in keep:
        j = idx[k]
        zcol = int(res["zone"][j])
        w = params.windows[int(res["wi"][j])]
        alarm = _candidate(layer, barrier, zcol, w, float(res["c"][j]), float(res["mu"][j]),
                           float(res["llr"][j]), float(p[k]), ctx, obs, zones, params, as_of)
        if alarm is not None:
            cands.append(alarm)
    return FamilyResult(layer, barrier, fam_p, n_rec, cands)


def _candidate(layer, barrier, zcol, w, c, mu, llr_v, p, ctx, obs, zones, params, as_of):
    Mc = ctx["M"]
    recs = Mc.indices[Mc.indptr[zcol]:Mc.indptr[zcol + 1]]
    recs = recs[ctx["age0"][recs] < w]
    rec_set = np.zeros(ctx["age0"].size, dtype=bool)
    rec_set[recs] = True
    a_sel = ctx["sel"][rec_set[ctx["inv"]]]
    if ctx["cols"] is None:                    # school barrier-as-space family
        zone = frozenset(zones.students)
        zone_id = "school"
        b = BARRIERS[zcol]
        in_zone = a_sel
    else:
        g = int(ctx["cols"][zcol])
        zone = zones.members[g]
        zone_id = zones.zone_id[g]
        b = barrier
        stud = obs.a_student[a_sel]
        in_zone = a_sel[np.isin(stud, [zones.sidx[s] for s in zone])]
    if np.unique(obs.a_unit[in_zone]).size < params.min_units:
        return None
    absent = frozenset(zones.students[i] for i in obs.a_student[in_zone])
    return Alarm(detector="scan", layer=layer, zone_id=zone_id, zone=zone, barrier=b, day=as_of, window=w,
                 observed=int(round(c)), expected=float(mu), llr=float(llr_v), p_value=float(p),
                 absent_members=absent, evidence_absence_ids=sorted(obs.a_ids[i] for i in in_zone))
