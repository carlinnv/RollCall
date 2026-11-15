"""Templated retrieval benchmark for the F7 assistant (PRD 2 sections 3.4 and 4). No hand labels.

Queries are generated from simulator ground truth (planted events, true barriers), and each
query's relevant set R(q) is the set of reply message ids whose absences belong to the event or
match the query's stated filter (true label, language, week). Retrieval only sees the reply text,
its metadata and the pipeline's *observed* (noisy) label, never the ground truth.

    Recall@5 = |top5 ∩ R| / min(5, |R|)          (PRD 2 section 3.4)
    MRR      = 1 / rank of the first relevant hit in the top 10 (0 if none)
    P@5      = |top5 ∩ R| / |top5|               (citation-precision proxy: an agent citing the top 5)

Weeks are 1-based school weeks: week w = school days 5(w-1) .. 5(w-1)+4.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np

from .corpus import build_corpus, group_sizes, message_id_for, week_days, week_of
from .retrieval import MessageIndex

ILLNESS = "illness"
TRANSPORTATION = "transportation"

# name -> MessageIndex.search kwargs
METHODS: Dict[str, Dict[str, Any]] = {
    "bm25": dict(method="bm25", expand=False, metadata=False),
    "bm25+expansion": dict(method="bm25", expand=True, metadata=False),
    "bm25+filter": dict(method="bm25", expand=False, metadata=True),
    "bm25+expansion+filter": dict(method="bm25", expand=True, metadata=True),
    # label ablation: no observed-label boost, so cause matching rests on text (+ lexicon) alone
    "bm25+filter (text only, no labels)": dict(method="bm25", expand=False, metadata=True, use_labels=False),
    "bm25+expansion+filter (no labels)": dict(method="bm25", expand=True, metadata=True, use_labels=False),
    "dense_hash": dict(method="dense", expand=True, metadata=False),
    "dense_hash+filter": dict(method="dense", expand=True, metadata=True),
    "hybrid+filter": dict(method="hybrid", expand=True, metadata=True),
}
DEFAULT_METHOD = "bm25+expansion+filter"


@dataclass
class BenchQuery:
    qid: str
    run: int
    text: str
    qtype: str
    relevant: List[str]
    as_of: Optional[int] = None
    cross_lingual: bool = False
    event_id: Optional[str] = None
    n_relevant_non_en: int = 0


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 5) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("empty relevant set")
    return len(set(ranked[:k]) & rel) / min(k, len(rel))


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    rel = set(relevant)
    for i, r in enumerate(ranked[:k]):
        if r in rel:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 5) -> float:
    top = list(ranked[:k])
    return len(set(top) & set(relevant)) / len(top) if top else 0.0


def citation_precision(cited: Iterable[str], relevant: Iterable[str]) -> Optional[float]:
    cited = list(dict.fromkeys(cited))
    if not cited:
        return None
    rel = set(relevant)
    return sum(c in rel for c in cited) / len(cited)


# ----------------------------------------------------------------------------------------------
# query generation


def _window(center: int, n_days: int, half: int = 3):
    return max(0, center - half), min(n_days - 1, center + half)


def make_queries(run: Any, run_index: int = 0) -> List[BenchQuery]:
    """Templated queries with ground-truth relevant sets for one SimRun."""
    replied = {r.absence_id: r for r in run.replies}
    lang_of = {r.absence_id: r.language for r in run.replies}
    school = run.school
    T = run.n_days
    out: List[BenchQuery] = []

    def mids(absences, as_of=None):
        return sorted(message_id_for(a.id) for a in absences
                      if a.id in replied and (as_of is None or replied[a.id].received_day <= as_of))

    def add(text, qtype, absences, as_of=None, cross=False, event_id=None):
        rel = mids(absences, as_of)
        if not rel:
            return
        non_en = sum(1 for m in rel if lang_of[m[2:]] != "en")
        out.append(BenchQuery(qid=f"r{run_index}-q{len(out):02d}", run=run_index, text=text, qtype=qtype,
                              relevant=rel, as_of=as_of, cross_lingual=cross, event_id=event_id,
                              n_relevant_non_en=non_en))

    ev_abs: Dict[str, List[Any]] = defaultdict(list)
    for a in run.absences:
        if a.event_id:
            ev_abs[a.event_id].append(a)
    info = {eid: m.get("info", {}) for eid, m in (run.meta or {}).items()}

    for e in run.events:
        A = ev_abs[e.id]
        if e.kind == "route_failure":
            route = info.get(e.id, {}).get("route")
            if not route:
                continue
            w = week_of(e.onset)
            ws, we = week_days(w)
            as_of = min(we, T - 1)
            add(f"Why are students on route {route} missing school this week?", "route_this_week",
                [a for a in A if ws <= a.day <= as_of], as_of=as_of, event_id=e.id)
            add(f"Transportation problems on bus {route} around day {e.onset}", "route_day", A, event_id=e.id)
            add(f"Which Spanish-speaking families said the bus did not come in week {w}?", "xl_language_week",
                [a for a in run.absences if a.true_barrier == TRANSPORTATION and ws <= a.day <= we
                 and lang_of.get(a.id) == "es"], cross=True)
            add(f"Families saying the school bus never showed up during week {w}", "xl_cause_week",
                [a for a in run.absences if a.true_barrier == TRANSPORTATION and ws <= a.day <= we], cross=True)
        elif e.kind == "contagion":
            mid = e.onset + e.duration // 2
            lo, hi = _window(mid, T)
            add(f"Which class has an illness cluster around day {mid}?", "which_class",
                [a for a in A if lo <= a.day <= hi], event_id=e.id)
            sec = Counter(school.section[a.student] for a in A).most_common(1)[0][0]
            d = e.onset + 1
            lo2, hi2 = _window(d, T)
            add(f"Students in {sec} out sick around day {d}", "section_illness",
                [a for a in A if school.section[a.student] == sec and lo2 <= a.day <= hi2], event_id=e.id)
            w = week_of(e.onset)
            ws, we = week_days(w)
            add(f"Vietnamese or Chinese families reporting a sick child in week {w}", "xl_language_week",
                [a for a in run.absences if a.true_barrier == ILLNESS and ws <= a.day <= we
                 and lang_of.get(a.id) in ("vi", "zh")], cross=True)
        elif e.kind == "caregiving_spike":
            nb = info.get(e.id, {}).get("neighborhood")
            lo, hi = _window(e.onset, T)
            inwin = [a for a in A if lo <= a.day <= hi]
            if nb:
                add(f"Childcare problems in neighborhood {nb} around day {e.onset}", "neighborhood_cause",
                    inwin, event_id=e.id)
            add(f"Which neighborhood has families keeping kids home to babysit around day {e.onset}?",
                "which_neighborhood", inwin, event_id=e.id)
        elif e.kind == "flu_wave":
            mid = e.onset + e.duration // 2
            lo, hi = _window(mid, T)
            add(f"Is there a school-wide flu outbreak around day {mid}?", "flu_window",
                [a for a in run.absences if a.true_barrier == ILLNESS and lo <= a.day <= hi])
    return out


# ----------------------------------------------------------------------------------------------
# evaluation


def _mean(xs: Sequence[float]) -> Optional[float]:
    return round(float(np.mean(xs)), 4) if len(xs) else None


def evaluate(index_by_run: Mapping[int, MessageIndex], queries: Sequence[BenchQuery],
             methods: Mapping[str, Mapping[str, Any]] = METHODS, k: int = 5) -> Dict[str, Any]:
    per_method: Dict[str, Any] = {}
    per_query: List[Dict[str, Any]] = []
    for q in queries:
        per_query.append({**{k_: v for k_, v in asdict(q).items() if k_ != "relevant"},
                          "n_relevant": len(q.relevant), "methods": {}})
    for name, kwargs in methods.items():
        r5, rr, p5 = [], [], []
        by_type: Dict[str, List[float]] = defaultdict(list)
        xl: List[float] = []
        for q, row in zip(queries, per_query):
            res = index_by_run[q.run].search(q.text, k=10, as_of=q.as_of, **kwargs)
            ranked = res.ids
            a, b, c = recall_at_k(ranked, q.relevant, k), reciprocal_rank(ranked, q.relevant), \
                precision_at_k(ranked, q.relevant, k)
            r5.append(a), rr.append(b), p5.append(c)
            by_type[q.qtype].append(a)
            if q.cross_lingual:
                xl.append(a)
            row["methods"][name] = {"recall@5": round(a, 3), "rr": round(b, 3), "p@5": round(c, 3),
                                    "top5": ranked[:5]}
        per_method[name] = {
            "recall@5": _mean(r5), "mrr": _mean(rr), "p@5": _mean(p5),
            "recall@5_cross_lingual": _mean(xl),
            "recall@5_by_type": {t: _mean(v) for t, v in sorted(by_type.items())},
            "share_queries_recall_1": _mean([float(x == 1.0) for x in r5]),
        }
    return {"methods": per_method, "queries": per_query}


def build_runs(config: Any, n_runs: int = 2, school_name: str = "lyon_primary_2009"):
    """Simulator runs as in scripts/run_ablation.py: run i uses seed config.seed + i for both
    the simulator and assign_logistics."""
    import dataclasses

    from app.ml.graphs import assign_logistics, load_school
    from app.ml.simulator import simulate

    base = load_school(school_name)
    runs = []
    for i in range(n_runs):
        seed = config.seed + i
        school = assign_logistics(base, seed)
        runs.append(simulate(school, dataclasses.replace(config, seed=seed)))
    return runs


def index_for_run(run: Any, dense: bool = True) -> MessageIndex:
    school = run.school
    students = {s: {"section": school.section.get(s), "route": school.bus_route.get(s),
                    "neighborhood": school.neighborhood.get(s)} for s in school.students}
    return MessageIndex(build_corpus(run), group_sizes=group_sizes(students), dense=dense)


def run_benchmark(config: Any, n_runs: int = 2, methods: Mapping[str, Mapping[str, Any]] = METHODS,
                  school_name: str = "lyon_primary_2009") -> Dict[str, Any]:
    runs = build_runs(config, n_runs, school_name)
    queries: List[BenchQuery] = []
    indexes = {}
    for i, run in enumerate(runs):
        indexes[i] = index_for_run(run)
        queries += make_queries(run, i)
    res = evaluate(indexes, queries, methods)
    res["setup"] = {
        "school": school_name, "config": config.name, "seeds": [config.seed + i for i in range(n_runs)],
        "n_queries": len(queries), "n_messages": [len(ix.messages) for ix in indexes.values()],
        "query_types": dict(Counter(q.qtype for q in queries)),
        "n_cross_lingual": sum(q.cross_lingual for q in queries),
        "mean_relevant": _mean([len(q.relevant) for q in queries]),
        "share_relevant_non_english": _mean([q.n_relevant_non_en / len(q.relevant) for q in queries]),
    }
    return res
