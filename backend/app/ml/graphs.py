"""Contact graphs and synthetic logistics layers for the PRD 2 pipeline.

Public API (all pure and deterministic given their seed):

    load_school(name, top_k=10, data_dir=None) -> SchoolGraph
    assign_logistics(school, seed, n_neighborhoods=6, n_routes=8, rider_frac=0.6, sibling_rate=0.15) -> SchoolGraph
    aggregate_contacts(school) -> nx.Graph
    ego_nets(school, ks=(5, 10, 20)) -> dict[(student, k), frozenset[str]]
    synthetic_sbm(n_students, seed, students_per_section=25, n_days=3, top_k=10, reference=None) -> SchoolGraph
    summary(school) -> dict

Data: SocioPatterns face-to-face contacts in Netzschleuder CSV form (see scripts/fetch_graphs.sh).
Each edges.csv row is one active 20-second contact interval between two node indices.

Graph handling rules (PRD 2 section 2): node ids are re-keyed ("N000", ...), class names are replaced
by "SEC-00", ...; teachers are dropped; only the face-to-face layer is used; schools are never merged.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .types import SchoolGraph

CONTACT_INTERVAL_S = 20  # SocioPatterns sampling resolution: one row = 20 s of contact

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data_raw"

SCHOOLS: Dict[str, str] = {
    "marseille_hs_2013": "hs2013_proximity",
    "lyon_primary_2009": "primary2009",
}

TEACHER_CLASS = "Teachers"

# Default SBM calibration, measured on lyon_primary_2009 with top_k=10 (see summary() / _calibrate()).
# Used only when synthetic_sbm() gets no reference graph.
_DEFAULT_SBM_PARAMS = {
    "p_in": 0.455,           # P(edge) for a within-section pair on a given day (after top-k)
    "d_out": 2.07,           # expected between-section degree per student per day
    "f_same_grade": 0.52,    # share of between-section edges that stay inside the grade
    "n_grades": 5,
    "w_in": (5.99, 1.02),    # lognormal (mu, sigma) of within-section seconds/day
    "w_out": (6.18, 0.84),   # lognormal (mu, sigma) of between-section seconds/day
}


# ----------------------------------------------------------------------------------------------
# helpers


def _id_width(n: int, minimum: int) -> int:
    return max(minimum, len(str(max(n - 1, 0))))


def _student_ids(n: int) -> List[str]:
    w = _id_width(n, 3)
    return [f"N{i:0{w}d}" for i in range(n)]


def _section_ids(n: int) -> List[str]:
    w = _id_width(n, 2)
    return [f"SEC-{i:0{w}d}" for i in range(n)]


def _top_k_mask(u: np.ndarray, v: np.ndarray, w: np.ndarray, k: Optional[int]) -> np.ndarray:
    """Boolean mask over edges: True if the edge is among the top-k (by weight) of either endpoint.

    Ties are broken by the neighbour's index (lower first), so the result is deterministic.
    """
    m = len(u)
    if k is None or m == 0:
        return np.ones(m, dtype=bool)
    ends = np.concatenate([u, v])
    other = np.concatenate([v, u])
    ww = np.concatenate([w, w])
    eidx = np.concatenate([np.arange(m), np.arange(m)])
    order = np.lexsort((other, -ww, ends))  # last key is primary: group by endpoint, heaviest first
    se = ends[order]
    starts = np.flatnonzero(np.r_[True, se[1:] != se[:-1]])
    lengths = np.diff(np.r_[starts, len(se)])
    rank = np.arange(len(se)) - np.repeat(starts, lengths)
    keep = np.zeros(m, dtype=bool)
    keep[eidx[order][rank < k]] = True
    return keep


def _build_day_graph(names: Sequence[str], u: np.ndarray, v: np.ndarray, w: np.ndarray,
                     top_k: Optional[int]) -> nx.Graph:
    keep = _top_k_mask(u, v, w, top_k)
    g = nx.Graph()
    g.add_nodes_from(names)
    g.add_weighted_edges_from(
        (names[a], names[b], int(x)) for a, b, x in zip(u[keep].tolist(), v[keep].tolist(), w[keep].tolist())
    )
    return g


def _read_rows(path: Path) -> Iterable[List[str]]:
    with path.open(newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].lstrip().startswith("#"):
                continue
            yield [c.strip() for c in row]


def _hs_grade(cls: str) -> int:
    """Year of study for the Marseille high-school classes (classes preparatoires).

    Heuristic: a starred class (MP*1, MP*2, PC*, PSI*) is a second-year class, and so are the
    2BIO* classes (BCPST, labelled with a leading "2"); plain MP and PC are first-year classes.
    """
    if cls.startswith("2BIO") or "*" in cls:
        return 2
    return 1


def _primary_grade(cls: str) -> int:
    """Lyon primary classes are '1A'..'5B'; the leading digit is the grade."""
    return int(cls[0])


def _day_key_hs(ts: int) -> _dt.date:
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Europe/Paris")
    except Exception:  # pragma: no cover - no tz database; December in Paris is UTC+1 (no DST)
        tz = _dt.timezone(_dt.timedelta(hours=1))
    return _dt.datetime.fromtimestamp(ts, tz).date()


# ----------------------------------------------------------------------------------------------
# 1. real schools


def load_school(name: str, top_k: int = 10, data_dir: Optional[Path] = None) -> SchoolGraph:
    """Load a SocioPatterns school as one top-k-reduced contact graph per recorded day.

    * Teachers are excluded; students are re-keyed "N000".. sorted by original dataset id;
      sections "SEC-00".. sorted by original class name; the original names are dropped.
    * Days: marseille_hs_2013 uses the Europe/Paris calendar date of the UNIX timestamp;
      lyon_primary_2009 uses time // 86400 (time is seconds from an unspecified start).
    * Edge weight = 20 s x number of contact rows for that pair that day; then each node keeps its
      top_k heaviest edges and an edge survives if it is in the top_k of either endpoint
      (top_k=None disables the reduction). Every student is a node of every daily graph.
    """
    if name not in SCHOOLS:
        raise ValueError(f"unknown school {name!r}; expected one of {sorted(SCHOOLS)}")
    folder = Path(data_dir or DEFAULT_DATA_DIR) / SCHOOLS[name]
    nodes_path, edges_path = folder / "nodes.csv", folder / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"missing {folder}/nodes.csv or edges.csv (run scripts/fetch_graphs.sh)")

    # nodes.csv: index, id, class, gender, _pos
    raw = []
    for row in _read_rows(nodes_path):
        idx, orig_id, cls = int(row[0]), row[1], row[2]
        if cls == TEACHER_CLASS:
            continue
        raw.append((idx, orig_id, cls))

    def _id_sort_key(t):
        oid = t[1]
        return (0, int(oid), "") if oid.lstrip("-").isdigit() else (1, 0, oid)

    raw.sort(key=_id_sort_key)
    names = _student_ids(len(raw))
    classes = sorted({cls for _, _, cls in raw})
    sec_names = dict(zip(classes, _section_ids(len(classes))))
    grade_fn = _hs_grade if name == "marseille_hs_2013" else _primary_grade

    index_to_pos: Dict[int, int] = {}
    section: Dict[str, str] = {}
    grade: Dict[str, int] = {}
    for pos, (idx, _, cls) in enumerate(raw):
        index_to_pos[idx] = pos
        section[names[pos]] = sec_names[cls]
        grade[names[pos]] = grade_fn(cls)

    # edges.csv: source, target, time  -> count rows per (day, pair)
    day_of = _day_key_hs if name == "marseille_hs_2013" else (lambda t: t // 86400)
    day_cache: Dict[int, object] = {}
    keys: Dict[object, int] = {}
    a_list: List[int] = []
    b_list: List[int] = []
    d_list: List[int] = []
    for row in _read_rows(edges_path):
        a = index_to_pos.get(int(row[0]))
        b = index_to_pos.get(int(row[1]))
        if a is None or b is None or a == b:  # teacher endpoint or self loop
            continue
        t = int(row[2])
        bucket = t // 3600  # timestamps are hour-aligned for both calendars, so cache per hour
        dk = day_cache.get(bucket)
        if dk is None:
            dk = day_cache[bucket] = day_of(t)
        di = keys.setdefault(dk, len(keys))
        if a > b:
            a, b = b, a
        a_list.append(a)
        b_list.append(b)
        d_list.append(di)

    n = len(names)
    # map day keys to 0..D-1 in calendar order
    sorted_days = sorted(keys)
    remap = np.empty(len(keys), dtype=np.int64)
    for new, dk in enumerate(sorted_days):
        remap[keys[dk]] = new
    d_arr = remap[np.asarray(d_list, dtype=np.int64)] if d_list else np.empty(0, dtype=np.int64)
    code = (d_arr * n + np.asarray(a_list, dtype=np.int64)) * n + np.asarray(b_list, dtype=np.int64)
    uniq, counts = np.unique(code, return_counts=True)
    days_u = uniq // (n * n)
    a_u = (uniq // n) % n
    b_u = uniq % n
    w_u = counts * CONTACT_INTERVAL_S

    daily = []
    for d in range(len(sorted_days)):
        sel = days_u == d
        daily.append(_build_day_graph(names, a_u[sel], b_u[sel], w_u[sel], top_k))

    return SchoolGraph(name=name, students=names, section=section, grade=grade, daily=daily)


# ----------------------------------------------------------------------------------------------
# 2. synthetic logistics


def _pair_siblings(students: List[str], section: Dict[str, str], grade: Dict[str, int],
                   n_pairs: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Greedy random sibling pairs: always different sections, different grades whenever possible."""
    n = len(students)
    order = rng.permutation(n).tolist()
    sec = [section[s] for s in students]
    grd = [grade[s] for s in students]
    pools: Dict[int, List[int]] = defaultdict(list)
    for i in order:
        pools[grd[i]].append(i)
    remaining = {g: len(p) for g, p in pools.items()}
    grades = sorted(pools)
    paired = [False] * n
    pairs: List[Tuple[int, int]] = []

    def _pop(pool: List[int], exclude: int, exclude_sec: Optional[str]) -> Optional[int]:
        # lazy deletion from the end of a shuffled pool
        while pool and paired[pool[-1]]:
            pool.pop()
        for j in range(len(pool) - 1, -1, -1):
            c = pool[j]
            if paired[c] or c == exclude or (exclude_sec is not None and sec[c] == exclude_sec):
                continue
            pool.pop(j)
            return c
        return None

    for a in order:
        if len(pairs) >= n_pairs:
            break
        if paired[a]:
            continue
        ga = grd[a]
        others = [g for g in grades if g != ga and remaining[g] > 0]
        b = None
        if others:
            p = np.array([remaining[g] for g in others], dtype=float)
            g = others[int(rng.choice(len(others), p=p / p.sum()))]
            b = _pop(pools[g], a, None)
        if b is None:  # single-grade fallback: same grade, different section
            b = _pop(pools[ga], a, sec[a])
        if b is None:
            continue
        paired[a] = paired[b] = True
        remaining[grd[a]] -= 1
        remaining[grd[b]] -= 1
        pairs.append((a, b))
    return pairs


def assign_logistics(school: SchoolGraph, seed: int, n_neighborhoods: int = 6, n_routes: int = 8,
                     rider_frac: float = 0.6, sibling_rate: float = 0.15) -> SchoolGraph:
    """Return a copy of `school` with synthetic bus_route / household / neighborhood layers.

    * Neighborhoods "NB1".. are drawn at random with Dirichlet(4) sizes (no link to contacts).
    * Routes "R01".. each serve 1-2 neighborhoods; when n_routes >= n_neighborhoods every
      neighborhood is served by at least one route (otherwise unserved neighborhoods walk).
    * Households "H0000..": about `sibling_rate` of students live with one sibling from a
      different section (a different grade whenever the school has more than one); the rest
      live alone. Members of a household share neighborhood and route.
    * A household rides the bus with probability `rider_frac`, on a route serving its
      neighborhood; non-riders get bus_route None.
    The daily contact graphs are shared with the input (not copied).
    """
    rng = np.random.default_rng(seed)
    students = list(school.students)
    n = len(students)

    # households
    n_pairs = int(round(sibling_rate * n / 2))
    pairs = _pair_siblings(students, school.section, school.grade, n_pairs, rng)
    root = list(range(n))
    for a, b in pairs:
        root[max(a, b)] = min(a, b)
    hh_of_root: Dict[int, int] = {}
    hh_index = np.empty(n, dtype=np.int64)
    for i in range(n):
        hh_index[i] = hh_of_root.setdefault(root[i], len(hh_of_root))
    n_hh = len(hh_of_root)
    hw = _id_width(n_hh, 4)

    # neighborhoods and routes
    nb_names = [f"NB{i + 1}" for i in range(n_neighborhoods)]
    rw = _id_width(n_routes + 1, 2)
    route_names = [f"R{i + 1:0{rw}d}" for i in range(n_routes)]
    nb_p = rng.dirichlet(np.full(n_neighborhoods, 4.0))
    perm = rng.permutation(n_neighborhoods)
    serves: Dict[int, List[int]] = defaultdict(list)  # neighborhood -> routes
    for r in range(n_routes):
        primary = int(perm[r % n_neighborhoods])
        serves[primary].append(r)
        if n_neighborhoods > 1 and rng.random() < 0.5:
            second = int(rng.integers(0, n_neighborhoods - 1))
            second += second >= primary
            serves[second].append(r)

    hh_nb = rng.choice(n_neighborhoods, size=n_hh, p=nb_p)
    hh_rides = rng.random(n_hh) < rider_frac
    hh_pick = rng.random(n_hh)

    household: Dict[str, str] = {}
    neighborhood: Dict[str, str] = {}
    bus_route: Dict[str, Optional[str]] = {}
    for i, s in enumerate(students):
        h = int(hh_index[i])
        nb = int(hh_nb[h])
        household[s] = f"H{h:0{hw}d}"
        neighborhood[s] = nb_names[nb]
        routes = serves.get(nb, [])
        if hh_rides[h] and routes:
            bus_route[s] = route_names[routes[int(hh_pick[h] * len(routes))]]
        else:
            bus_route[s] = None

    return dataclasses.replace(school, bus_route=bus_route, household=household, neighborhood=neighborhood)


# ----------------------------------------------------------------------------------------------
# 3./4. aggregation and ego nets


def aggregate_contacts(school: SchoolGraph) -> nx.Graph:
    """Sum of the daily (top-k-reduced) contact weights over all recorded days."""
    agg = nx.Graph()
    agg.add_nodes_from(school.students)
    for g in school.daily:
        for a, b, w in g.edges(data="weight"):
            if agg.has_edge(a, b):
                agg[a][b]["weight"] += w
            else:
                agg.add_edge(a, b, weight=w)
    return agg


def ego_nets(school: SchoolGraph, ks: Sequence[int] = (5, 10, 20),
             agg: Optional[nx.Graph] = None) -> Dict[Tuple[str, int], FrozenSet[str]]:
    """For each student v and k in ks: {v} plus its k strongest neighbours in the aggregated graph.

    Ties are broken by neighbour id. Students with fewer than k neighbours get all of them.
    Pass a precomputed `agg` (from aggregate_contacts) to avoid recomputing it.
    """
    agg = aggregate_contacts(school) if agg is None else agg
    out: Dict[Tuple[str, int], FrozenSet[str]] = {}
    for v in school.students:
        nbrs = sorted(agg[v].items(), key=lambda kv: (-kv[1]["weight"], kv[0]))
        ranked = [u for u, _ in nbrs]
        for k in ks:
            out[(v, k)] = frozenset([v, *ranked[:k]])
    return out


# ----------------------------------------------------------------------------------------------
# 5. synthetic stochastic block model


def _lognormal_fit(w: np.ndarray, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if len(w) < 2:
        return fallback
    lw = np.log(w.astype(float))
    return float(lw.mean()), float(max(lw.std(), 1e-3))


def _calibrate(reference: SchoolGraph) -> dict:
    """Within/between-section edge probabilities and weight distributions of a reference school."""
    sec = reference.section
    grd = reference.grade
    sizes = defaultdict(int)
    for s in reference.students:
        sizes[sec[s]] += 1
    within_pairs = sum(c * (c - 1) // 2 for c in sizes.values())
    n = len(reference.students)
    e_in = e_out = e_same_grade = 0
    w_in: List[int] = []
    w_out: List[int] = []
    for g in reference.daily:
        for a, b, w in g.edges(data="weight"):
            if sec[a] == sec[b]:
                e_in += 1
                w_in.append(w)
            else:
                e_out += 1
                w_out.append(w)
                e_same_grade += grd[a] == grd[b]
    days = max(len(reference.daily), 1)
    d = _DEFAULT_SBM_PARAMS
    return {
        "p_in": (e_in / days) / within_pairs if within_pairs else d["p_in"],
        "d_out": 2.0 * (e_out / days) / n if n else d["d_out"],
        "f_same_grade": e_same_grade / e_out if e_out else d["f_same_grade"],
        "n_grades": len(set(grd.values())) or d["n_grades"],
        "w_in": _lognormal_fit(np.asarray(w_in), d["w_in"]),
        "w_out": _lognormal_fit(np.asarray(w_out), d["w_out"]),
        "w_in_pool": np.asarray(w_in, dtype=np.int64),
        "w_out_pool": np.asarray(w_out, dtype=np.int64),
    }


def _sample_weights(rng: np.random.Generator, m: int, pool: Optional[np.ndarray],
                    lognorm: Tuple[float, float]) -> np.ndarray:
    if pool is not None and len(pool):
        return pool[rng.integers(0, len(pool), size=m)]
    mu, sigma = lognorm
    raw = rng.lognormal(mu, sigma, size=m)
    return (np.maximum(1, np.ceil(raw / CONTACT_INTERVAL_S)) * CONTACT_INTERVAL_S).astype(np.int64)


def synthetic_sbm(n_students: int, seed: int, students_per_section: int = 25, n_days: int = 3,
                  top_k: int = 10, reference: Optional[SchoolGraph] = None) -> SchoolGraph:
    """Section-block stochastic block model for the runtime-vs-N benchmark (10k-50k students).

    Students are laid out in contiguous sections of `students_per_section`; sections are split into
    `n_grades` contiguous grade blocks (n_grades = number of grades in the reference, else 5).
    Each day independently:
      * every within-section pair is an edge with probability p_in (vectorised Bernoulli);
      * Poisson(n * d_out / 2) between-section edges are drawn per day: a uniform source, then a
        target in the same grade with probability f_same_grade, otherwise anywhere in another grade
        (self / same-section / wrong-grade draws are dropped, duplicates merged);
      * weights are resampled from the reference's within/between weight pools (or, without a
        reference, from lognormal fits measured on lyon_primary_2009), then reduced to top_k.
    Calibration (p_in, d_out, f_same_grade, weights) comes from `reference` when given.
    Memory and time are O(n * (students_per_section + d_out)) per day; no dense matrices.
    """
    if n_students <= 0:
        raise ValueError("n_students must be positive")
    rng = np.random.default_rng(seed)
    params = _calibrate(reference) if reference is not None else dict(_DEFAULT_SBM_PARAMS)
    p_in = float(min(max(params["p_in"], 0.0), 1.0))
    d_out = float(max(params["d_out"], 0.0))
    f_same = float(min(max(params["f_same_grade"], 0.0), 1.0))
    pool_in = params.get("w_in_pool")
    pool_out = params.get("w_out_pool")

    n = n_students
    spp = max(1, students_per_section)
    n_sec = math.ceil(n / spp)
    n_grades = int(max(1, min(params["n_grades"], n_sec)))
    names = _student_ids(n)
    sec_names = _section_ids(n_sec)
    sec_of = np.arange(n) // spp                         # section index per student
    grade_of_sec = (np.arange(n_sec) * n_grades) // n_sec  # contiguous grade blocks
    grade_of = grade_of_sec[sec_of]
    # student index range [lo, hi) of each grade (contiguous by construction)
    g_lo = np.searchsorted(grade_of, np.arange(n_grades), side="left")
    g_hi = np.searchsorted(grade_of, np.arange(n_grades), side="right")

    # all within-section pairs (sections are contiguous index blocks)
    iu, ju = np.triu_indices(spp, k=1)
    full = n // spp
    base = (np.arange(full) * spp)[:, None]
    pu = [(base + iu).ravel()]
    pv = [(base + ju).ravel()]
    rem = n - full * spp
    if rem > 1:
        ri, rj = np.triu_indices(rem, k=1)
        pu.append(ri + full * spp)
        pv.append(rj + full * spp)
    pair_u = np.concatenate(pu).astype(np.int64)
    pair_v = np.concatenate(pv).astype(np.int64)

    daily = []
    for _ in range(n_days):
        m_in = rng.random(len(pair_u)) < p_in
        u_in, v_in = pair_u[m_in], pair_v[m_in]

        m_out = int(rng.poisson(n * d_out / 2.0)) if n_sec > 1 else 0
        src = rng.integers(0, n, size=m_out)
        same = rng.random(m_out) < f_same
        gs = grade_of[src]
        span = (g_hi - g_lo)[gs]
        tgt_same = g_lo[gs] + (rng.random(m_out) * span).astype(np.int64)
        tgt_any = rng.integers(0, n, size=m_out)
        tgt = np.where(same, tgt_same, tgt_any)
        ok = (sec_of[tgt] != sec_of[src]) & np.where(same, True, grade_of[tgt] != gs)
        a = np.minimum(src, tgt)[ok]
        b = np.maximum(src, tgt)[ok]
        code = np.unique(a * n + b)
        u_out, v_out = code // n, code % n

        w_in = _sample_weights(rng, len(u_in), pool_in, params["w_in"])
        w_out = _sample_weights(rng, len(u_out), pool_out, params["w_out"])
        u = np.concatenate([u_in, u_out])
        v = np.concatenate([v_in, v_out])
        w = np.concatenate([w_in, w_out])
        daily.append(_build_day_graph(names, u, v, w, top_k))

    section = {names[i]: sec_names[s] for i, s in enumerate(sec_of.tolist())}
    grade = {names[i]: int(g) + 1 for i, g in enumerate(grade_of.tolist())}
    return SchoolGraph(name=f"sbm_{n}", students=names, section=section, grade=grade, daily=daily)


# ----------------------------------------------------------------------------------------------
# 6. summary


def summary(school: SchoolGraph) -> dict:
    """Headline numbers for a school graph (pitch: "contacts are mostly within a class")."""
    n = len(school.students)
    sec = school.section
    degs = [2 * g.number_of_edges() / n for g in school.daily] if n else [0.0]
    agg = aggregate_contacts(school)
    w_total = w_within = 0
    e_within = 0
    for a, b, w in agg.edges(data="weight"):
        w_total += w
        if sec[a] == sec[b]:
            w_within += w
            e_within += 1
    out = {
        "name": school.name,
        "n_students": n,
        "n_sections": len(set(sec.values())),
        "n_grades": len(set(school.grade.values())),
        "n_days": len(school.daily),
        "mean_daily_degree": round(float(np.mean(degs)), 2),
        "aggregate_degree": round(2 * agg.number_of_edges() / n, 2) if n else 0.0,
        "mean_daily_contact_minutes": round(w_total / max(n, 1) * 2 / max(len(school.daily), 1) / 60, 1),
        "within_section_weight_share": round(w_within / w_total, 3) if w_total else 0.0,
        "within_section_edge_share": round(e_within / agg.number_of_edges(), 3) if agg.number_of_edges() else 0.0,
    }
    if school.bus_route:
        riders = [r for r in school.bus_route.values() if r is not None]
        hh_sizes = defaultdict(int)
        for h in school.household.values():
            hh_sizes[h] += 1
        out.update({
            "n_routes": len(set(riders)),
            "rider_share": round(len(riders) / n, 3) if n else 0.0,
            "n_households": len(hh_sizes),
            "sibling_share": round(sum(c for c in hh_sizes.values() if c > 1) / n, 3) if n else 0.0,
            "n_neighborhoods": len(set(school.neighborhood.values())),
        })
    return out
