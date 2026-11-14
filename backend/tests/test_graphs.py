"""Tests for app.ml.graphs (PRD 2 section 2: contact graphs, logistics layers, SBM benchmark)."""
import re
import time
from collections import Counter, defaultdict

import pytest

from app.ml import graphs as G
from app.ml.types import SchoolGraph

HAS_DATA = all(
    (G.DEFAULT_DATA_DIR / sub / f).exists() for sub in G.SCHOOLS.values() for f in ("nodes.csv", "edges.csv")
)
needs_data = pytest.mark.skipif(not HAS_DATA, reason="data_raw missing (run scripts/fetch_graphs.sh)")


@pytest.fixture(scope="module")
def hs() -> SchoolGraph:
    if not HAS_DATA:
        pytest.skip("data_raw missing")
    return G.load_school("marseille_hs_2013")


@pytest.fixture(scope="module")
def primary() -> SchoolGraph:
    if not HAS_DATA:
        pytest.skip("data_raw missing")
    return G.load_school("lyon_primary_2009")


@pytest.fixture(scope="module")
def primary_full() -> SchoolGraph:
    if not HAS_DATA:
        pytest.skip("data_raw missing")
    return G.load_school("lyon_primary_2009", top_k=None)


def _check_rekeyed(school: SchoolGraph, n: int, n_sections: int, n_days: int):
    assert len(school.students) == n
    assert school.students == sorted(school.students)
    assert all(re.fullmatch(r"N\d{3}", s) for s in school.students)
    assert set(school.section) == set(school.students) == set(school.grade)
    secs = set(school.section.values())
    assert len(secs) == n_sections
    assert all(re.fullmatch(r"SEC-\d{2}", s) for s in secs)
    assert len(school.daily) == n_days
    for g in school.daily:
        assert set(g.nodes) == set(school.students)  # isolated students are still present
        assert all(w > 0 and w % 20 == 0 for _, _, w in g.edges(data="weight"))


# ---------------------------------------------------------------- real schools


@needs_data
def test_load_marseille(hs):
    _check_rekeyed(hs, 329, 9, 5)
    assert hs.name == "marseille_hs_2013"
    # year heuristic: MP (33) + PC (44) are first-year, everything else second-year
    assert Counter(hs.grade.values()) == {1: 77, 2: 252}


@needs_data
def test_load_lyon_excludes_teachers(primary):
    _check_rekeyed(primary, 232, 10, 2)
    assert primary.name == "lyon_primary_2009"
    assert set(primary.grade.values()) == {1, 2, 3, 4, 5}
    # each grade holds exactly two sections (A/B)
    per_grade = defaultdict(set)
    for s in primary.students:
        per_grade[primary.grade[s]].add(primary.section[s])
    assert all(len(v) == 2 for v in per_grade.values())


@needs_data
def test_top_k_rule(primary, primary_full):
    k = 10
    n = len(primary.students)
    for g_k, g_full in zip(primary.daily, primary_full.daily):
        assert g_k.number_of_edges() <= n * k
        assert g_k.number_of_edges() < g_full.number_of_edges()
        # an edge survives iff it is in the top-k of either endpoint (ties by neighbour id)
        chosen = set()
        for v in g_full.nodes:
            nbrs = sorted(g_full[v].items(), key=lambda kv: (-kv[1]["weight"], kv[0]))[:k]
            chosen.update(frozenset((v, u)) for u, _ in nbrs)
        assert {frozenset(e) for e in g_k.edges} == chosen
        for a, b, w in g_k.edges(data="weight"):
            assert g_full[a][b]["weight"] == w


@needs_data
def test_summary_contacts_mostly_within_section(hs, primary):
    for school in (hs, primary):
        s = G.summary(school)
        assert s["n_students"] == len(school.students)
        assert s["within_section_weight_share"] > 0.7


@needs_data
def test_ego_nets(primary):
    agg = G.aggregate_contacts(primary)
    nets = G.ego_nets(primary, ks=(5, 10, 20))
    assert len(nets) == 3 * len(primary.students)
    for v in primary.students:
        deg = agg.degree(v)
        for k in (5, 10, 20):
            ego = nets[(v, k)]
            assert v in ego
            assert len(ego) == 1 + min(k, deg)
        assert nets[(v, 5)] <= nets[(v, 10)] <= nets[(v, 20)]
        # strongest neighbour is always in the k=5 net
        if deg:
            best = max(agg[v].items(), key=lambda kv: (kv[1]["weight"], -int(kv[0][1:])))[0]
            assert best in nets[(v, 5)]


@needs_data
def test_aggregate_sums_days(primary):
    agg = G.aggregate_contacts(primary)
    total_daily = sum(w for g in primary.daily for _, _, w in g.edges(data="weight"))
    assert sum(w for _, _, w in agg.edges(data="weight")) == total_daily
    assert set(agg.nodes) == set(primary.students)


# ---------------------------------------------------------------- logistics


def _check_logistics(school: SchoolGraph, n_routes: int = 8, n_nb: int = 6):
    studs = set(school.students)
    assert set(school.bus_route) == set(school.household) == set(school.neighborhood) == studs
    routes = {r for r in school.bus_route.values() if r is not None}
    assert routes <= {f"R{i:02d}" for i in range(1, n_routes + 1)}
    assert set(school.neighborhood.values()) <= {f"NB{i}" for i in range(1, n_nb + 1)}
    assert all(re.fullmatch(r"H\d{4,}", h) for h in school.household.values())

    members = defaultdict(list)
    for s, h in school.household.items():
        members[h].append(s)
    sizes = Counter(len(m) for m in members.values())
    assert set(sizes) <= {1, 2}
    multi_grade = len(set(school.grade.values())) > 1
    for m in members.values():
        if len(m) == 2:
            a, b = m
            assert school.section[a] != school.section[b]
            if multi_grade:
                assert school.grade[a] != school.grade[b]
            assert school.neighborhood[a] == school.neighborhood[b]
            assert school.bus_route[a] == school.bus_route[b]
    sibling_share = sum(len(m) for m in members.values() if len(m) == 2) / len(studs)
    assert 0.10 <= sibling_share <= 0.20

    # routes are neighbourhood-correlated: each route serves at most two neighbourhoods
    route_nbs = defaultdict(set)
    for s, r in school.bus_route.items():
        if r is not None:
            route_nbs[r].add(school.neighborhood[s])
    assert all(1 <= len(v) <= 2 for v in route_nbs.values())
    rider_share = sum(r is not None for r in school.bus_route.values()) / len(studs)
    assert 0.4 <= rider_share <= 0.8


@needs_data
def test_logistics_real_schools(hs, primary):
    for school in (hs, primary):
        out = G.assign_logistics(school, seed=7)
        _check_logistics(out)
        assert not school.bus_route  # input not mutated
        assert out.daily is school.daily


def test_logistics_deterministic_per_seed():
    school = G.synthetic_sbm(600, seed=1)
    a = G.assign_logistics(school, seed=3)
    b = G.assign_logistics(school, seed=3)
    c = G.assign_logistics(school, seed=4)
    assert (a.bus_route, a.household, a.neighborhood) == (b.bus_route, b.household, b.neighborhood)
    assert (a.bus_route, a.household, a.neighborhood) != (c.bus_route, c.household, c.neighborhood)
    _check_logistics(a)


def test_logistics_single_grade_falls_back_to_different_section():
    school = G.synthetic_sbm(200, seed=0)
    school.grade = {s: 1 for s in school.students}
    out = G.assign_logistics(school, seed=0)
    _check_logistics(out)


# ---------------------------------------------------------------- SBM


def test_sbm_deterministic_and_structured():
    a = G.synthetic_sbm(500, seed=5, n_days=2)
    b = G.synthetic_sbm(500, seed=5, n_days=2)
    assert a.name == "sbm_500"
    assert len(a.daily) == 2
    assert a.section == b.section and a.grade == b.grade
    for ga, gb in zip(a.daily, b.daily):
        assert sorted(ga.edges(data="weight")) == sorted(gb.edges(data="weight"))
    assert len(set(a.section.values())) == 20
    assert set(a.grade.values()) == {1, 2, 3, 4, 5}
    assert G.summary(a)["within_section_weight_share"] > 0.7


def test_sbm_10k_builds_quickly():
    t0 = time.perf_counter()
    school = G.synthetic_sbm(10_000, seed=0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 15
    assert len(school.students) == 10_000
    assert school.students[0] == "N0000" and school.students[-1] == "N9999"
    assert len(school.daily) == 3
    for g in school.daily:
        assert g.number_of_nodes() == 10_000
        mean_deg = 2 * g.number_of_edges() / 10_000
        assert 5 <= mean_deg <= 20
        assert g.number_of_edges() <= 10_000 * 10


@needs_data
def test_sbm_calibrated_from_reference(hs):
    school = G.synthetic_sbm(1000, seed=0, reference=hs)
    assert set(school.grade.values()) == {1, 2}
    s, ref = G.summary(school), G.summary(hs)
    assert abs(s["mean_daily_degree"] - ref["mean_daily_degree"]) < 4
    assert s["within_section_weight_share"] > 0.7
