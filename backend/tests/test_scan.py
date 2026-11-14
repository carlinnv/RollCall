"""Unit tests for the PRD 2 scan, detectors and evaluator, on hand-built fixtures (no datasets needed)."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from app.ml import detectors as D
from app.ml.evaluate import (calibrate_threshold, merge_alarms, purity, score_run, truth_index)
from app.ml.scan import (Observations, ScanParams, bh_adjust, bh_reject, build_zones, llr, scan_day)
from app.ml.types import AbsenceEvent, Alarm, PlantedEvent, Reply, SchoolGraph, SimConfig, SimRun

N_SEC, PER_SEC, N_DAYS = 4, 20, 60
PARAMS = ScanParams(n_perm=99, seed=3)
BACKGROUND = (("illness", 0.5), ("appointment or travel", 0.25), ("caregiving or family obligation", 0.15),
              ("transportation", 0.10))


def make_school(route_equals_section: bool = False) -> SchoolGraph:
    n = N_SEC * PER_SEC
    students = [f"N{i:03d}" for i in range(n)]
    section = {s: f"SEC-{i // PER_SEC:02d}" for i, s in enumerate(students)}
    grade = {s: 1 + (i // PER_SEC) // 2 for i, s in enumerate(students)}
    g = nx.Graph()
    g.add_nodes_from(students)
    for sec in range(N_SEC):
        idx = list(range(sec * PER_SEC, (sec + 1) * PER_SEC))
        for a in idx:                                      # ring inside a section, weaker all-pairs
            for b in idx:
                if a < b:
                    dist = min(b - a, PER_SEC - (b - a))
                    g.add_edge(students[a], students[b], weight={1: 300, 2: 200, 3: 100}.get(dist, 10))
    for i in range(0, n - PER_SEC, 7):                     # a few weak cross-section ties
        g.add_edge(students[i], students[i + PER_SEC], weight=5)
    route, nb, hh = {}, {}, {}
    for i, s in enumerate(students):
        route[s] = None if i % 3 == 0 else f"R{(i * 7) % 4 + 1}"
        nb[s] = f"NB{(i * 3) % 5 + 1}"
        hh[s] = f"H{i:03d}"
    for a, b in ((0, 25), (5, 45), (10, 70)):              # sibling pairs share household / route / neighborhood
        hh[students[b]] = hh[students[a]]
        route[students[b]] = route[students[a]]
        nb[students[b]] = nb[students[a]]
    if route_equals_section:
        for i, s in enumerate(students):
            if section[s] == "SEC-03":
                route[s] = "R5"
        hh[students[70]] = f"H{70:03d}"                    # keep SEC-03 households inside SEC-03
    return SchoolGraph("fixture", students, section, grade, [g], route, hh, nb)


def make_run(school: SchoolGraph, seed: int = 0, planted=(), reply_delay: int = 0, rate: float = 0.03,
             quiet_from: int = N_DAYS) -> SimRun:
    """planted: iterable of (event_id, kind, barrier, members, days[, reply_day_offset][, should_alarm])."""
    rng = np.random.default_rng(seed)
    absences, replies, events = [], [], []
    taken = set()

    def add(student, day, barrier, eid, delay):
        aid = f"A{len(absences):05d}"
        absences.append(AbsenceEvent(aid, student, day, barrier, eid))
        replies.append(Reply(aid, day + delay, "en", "x", barrier, barrier))
        taken.add((student, day))

    for p in planted:
        eid, kind, barrier, members, days = p[:5]
        delay = p[5] if len(p) > 5 else reply_delay
        should = p[6] if len(p) > 6 else True
        events.append(PlantedEvent(eid, kind, barrier, frozenset(members), min(days), len(days), should))
        for d in days:
            for s in sorted(members):
                add(s, d, barrier, eid, delay)
    names = [b for b, _ in BACKGROUND]
    probs = np.array([w for _, w in BACKGROUND])
    for d in range(N_DAYS):
        if d >= quiet_from:
            continue
        for s in school.students:
            if (s, d) not in taken and rng.random() < rate:
                add(s, d, names[rng.choice(len(names), p=probs)], None, reply_delay)
    return SimRun(SimConfig(n_days=N_DAYS, seed=seed), school, N_DAYS, absences, replies, events)


def riders(school, route):
    return sorted(s for s in school.students if school.bus_route.get(s) == route)


# ----------------------------------------------------------------------------------------------
# statistics


def test_llr_hand_case():
    expected = 5 * math.log(5 / 2) + 15 * math.log(15 / 18)
    assert llr(5, 2.0, 20) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(1.846631, abs=1e-6)
    assert llr(2, 2.0, 20) == 0.0 and llr(1, 2.0, 20) == 0.0
    assert llr(20, 5.0, 20) == pytest.approx(20 * math.log(4))           # (C - c) = 0 term vanishes
    np.testing.assert_allclose(llr(np.array([5, 1]), np.array([2.0, 2.0]), 20), [expected, 0.0])


def test_bh_correctness():
    p = [0.01, 0.04, 0.03, 0.005, 0.5]                    # thresholds .02 .04 .06 .08 .10
    assert bh_reject(p, 0.10).tolist() == [True, True, True, True, False]
    p2 = [0.03, 0.04, 0.6]                                # step-up: 0.03 > 0.1/3 but rank 2 passes
    assert bh_reject(p2, 0.10).tolist() == [True, True, False]
    assert bh_reject([0.2, 0.3], 0.10).tolist() == [False, False]
    adj = bh_adjust(p)
    np.testing.assert_allclose(adj, [0.025, 0.05, 0.05, 0.025, 0.5])
    assert ((adj <= 0.10) == bh_reject(p, 0.10)).all()
    assert bh_reject([], 0.1).size == 0


# ----------------------------------------------------------------------------------------------
# zones and households


def test_zones_and_household_collapse():
    school = make_school()
    zs = build_zones(school, ks=(5,))
    assert {"section", "route", "neighborhood", "contact", "grade", "school"} <= set(zs.layer)
    assert zs.S2Z.shape == (80, zs.n_zones)
    # ego-net k=5 of N002 is itself plus its ring neighbours
    ego = zs.members[zs.zone_id.index("ego:N002:k5")]
    assert ego >= {"N000", "N001", "N002", "N003", "N004"} and len(ego) == 6

    # three households (one of them two siblings, N000 + N025) absent in the same neighborhood, same day
    nb = school.neighborhood["N000"]
    others = [s for s in school.students if school.neighborhood[s] == nb and s not in ("N000", "N025")][:2]
    members = ["N000", "N025"] + others
    run = make_run(school, seed=1, planted=[("E0", "caregiving_spike", "caregiving or family obligation",
                                             members, [50])], quiet_from=45)
    fams = scan_day(Observations.from_run(run, zs), zs, 50, "labels", ["neighborhood"], PARAMS)
    fam = next(f for f in fams if f.barrier == "caregiving or family obligation")
    alarm = next(a for a in fam.candidates if a.zone_id == nb and a.window == 1)
    assert alarm.observed == 3                             # 4 students, 3 households
    assert alarm.absent_members == frozenset(members)
    assert len(alarm.evidence_absence_ids) == 4

    # siblings alone are never a cluster (min 2 households)
    run2 = make_run(school, seed=1, planted=[("E0", "caregiving_spike", "caregiving or family obligation",
                                              ["N000", "N025"], [50])], quiet_from=45)
    alarms = D.combined(run2, 50, PARAMS, q=None, p_max=1.0)
    assert not [a for a in alarms if a.absent_members == frozenset({"N000", "N025"})]


# ----------------------------------------------------------------------------------------------
# detectors


def test_flat_null_has_no_alarms_at_strict_threshold():
    run = make_run(make_school(), seed=7)
    pipe = D.Pipeline(run, PARAMS)
    res = pipe.prospective(range(39, 60))
    for det, days in res.items():
        assert len(days) == 21
        for r in days:
            assert D.select(r, p_max=1e-4) == [], (det, r.day)
            assert all(0.0 <= f[2] <= 1.0 for f in r.families)


def test_route_failure_found_by_combined_as_transportation():
    school = make_school()
    rid = riders(school, "R1")
    members = rid[:7]
    run = make_run(school, seed=2, planted=[("E0", "route_failure", "transportation", members, [50])])
    alarms = D.combined(run, 50, PARAMS)                   # production setting: BH q = 0.10
    assert alarms and all(a.barrier == "transportation" for a in alarms[:2])
    top = next(a for a in alarms if a.layer == "route")
    assert top.zone_id == "R1" and top.barrier == "transportation"
    assert top.absent_members >= frozenset(members)
    assert top.p_value < 0.01 and top.observed >= 7 and top.expected < top.observed / 2
    # the graph-only detector also finds the route (its zone type says transportation)
    g = D.graph_only(run, 50, PARAMS)
    assert g and g[0].zone_id == "R1" and g[0].barrier == "transportation"


def test_contact_cluster_found_by_combined_as_illness():
    school = make_school()
    zs = build_zones(school, ks=(5,))
    members = zs.members[zs.zone_id.index("ego:N033:k5")]
    run = make_run(school, seed=4, planted=[("E0", "contagion", "illness", members, [50, 51])])
    alarms = D.combined(run, 51, PARAMS)
    hit = [a for a in alarms if a.barrier == "illness" and len(a.absent_members & members) >= 5]
    assert alarms[0].barrier == "illness"
    local = [a for a in hit if a.layer in ("contact", "section")]
    assert local and local[0].window >= 2                  # two consecutive days
    assert local[0].absent_members & members == members


def test_graph_only_cannot_distinguish_coincident_route_and_section():
    school = make_school(route_equals_section=True)
    sec3 = sorted(s for s in school.students if school.section[s] == "SEC-03")
    assert riders(school, "R5") == sec3
    members = sec3[::2]                                    # 10 of the 20
    run_route = make_run(school, seed=5, planted=[("E0", "route_failure", "transportation", members, [50])])
    run_ill = make_run(school, seed=5, planted=[("E0", "contagion", "illness", members, [50])])
    g_route = D.graph_only(run_route, 50, PARAMS)
    g_ill = D.graph_only(run_ill, 50, PARAMS)
    assert g_route and g_ill
    # graph-only sees identical inputs, so its answer is the same for both events: wrong for one of them
    assert [(a.zone_id, a.barrier) for a in g_route] == [(a.zone_id, a.barrier) for a in g_ill]
    assert g_route[0].barrier in ("transportation", "illness")
    # the combined detector reads the labels and gets both right
    assert D.combined(run_route, 50, PARAMS)[0].barrier == "transportation"
    assert D.combined(run_ill, 50, PARAMS)[0].barrier == "illness"
    # graph_labels: same all-cause scan as graph-only, cause read from the replies
    assert D.graph_labels(run_route, 50, PARAMS)[0].barrier == "transportation"
    assert D.graph_labels(run_ill, 50, PARAMS)[0].barrier == "illness"


def test_replies_after_as_of_are_ignored():
    school = make_school()
    members = riders(school, "R1")[:7]
    # every reply (planted and background) arrives 2 days late
    run = make_run(school, seed=2, planted=[("E0", "route_failure", "transportation", members, [50])],
                   reply_delay=2)
    early = D.combined(run, 50, PARAMS, q=None, p_max=1.0)
    assert not [a for a in early if a.barrier == "transportation" and a.absent_members & set(members)]
    late = [a for a in D.combined(run, 52, PARAMS) if a.layer == "route"]
    assert late and late[0].barrier == "transportation" and late[0].zone_id == "R1" and late[0].window == 3
    # graph-only uses all-cause absences, so it needs no replies
    assert D.graph_only(run, 50, PARAMS)[0].zone_id == "R1"
    # labels_at honours received_day
    obs = Observations.from_run(run, build_zones(school, ks=(5,)))
    assert (obs.labels_at(50)[obs.a_day == 50] == -1).all()
    assert (obs.labels_at(52)[obs.a_day == 50] >= 0).all()


def test_content_only_uses_school_and_grade_and_run_prospective():
    school = make_school()
    members = riders(school, "R1")[:9]
    run = make_run(school, seed=2, planted=[("E0", "route_failure", "transportation", members, [50])])
    alarms = D.content_only(run, 50, PARAMS)
    assert alarms and {a.layer for a in alarms} <= {"school", "grade"}
    assert alarms[0].barrier == "transportation"
    pro = D.run_prospective(run, "combined", range(48, 53), PARAMS)
    assert {a.day for a in pro if a.zone_id == "R1"} >= {50}
    assert all(48 <= a.day <= 52 for a in pro)


def test_ears_c1():
    school = make_school()
    run = make_run(school, seed=2, planted=[("E0", "flu_wave", "illness", school.students[:40], [50], 0, False)])
    days = [a.day for a in D.ears_c1(run)]
    assert 50 in days


# ----------------------------------------------------------------------------------------------
# evaluation


def _alarm(zone, barrier, day, window=1, p=0.001, absent=()):
    return Alarm("combined", "route", "Z", frozenset(zone), barrier, day, window, len(absent), 1.0, 5.0, p,
                 frozenset(absent))


def test_evaluator_tp_fp_matching():
    school = make_school()
    ev = ["N001", "N002", "N004", "N005"]                 # route-failure absentees
    bg = ["N007", "N008", "N010", "N011"]                 # background absentees in the same zone
    flu = school.students[40:60]
    run = make_run(school, seed=0, rate=0.0, planted=[
        ("E0", "route_failure", "transportation", ev, [50]),
        ("E1", "flu_wave", "illness", flu, [55], 0, False),
    ])
    for s in bg:
        run.absences.append(AbsenceEvent(f"B{s}", s, 50, "illness"))
    zone = ev + bg + ["N013"]
    truth = truth_index(run)
    tp = _alarm(zone, "transportation", 50, absent=ev)
    assert purity(tp, run.events[0], truth) == pytest.approx(0.5)
    impure = _alarm(zone + ["N020"], "transportation", 50)
    run.absences.append(AbsenceEvent("Bx", "N020", 50, "illness"))
    truth = truth_index(run)
    assert purity(impure, run.events[0], truth) < 0.5

    alarms = [
        tp,                                                        # TP, day 50
        _alarm(zone, "transportation", 51, window=2),               # same zone next day: merges with TP
        _alarm(zone, "illness", 50),                                # wrong barrier -> false alarm
        _alarm(ev, "transportation", 54, window=1),                 # outside [50, 53] and nobody absent -> FA
        _alarm(flu, "illness", 55),                                 # flu wave must not alarm -> FA
    ]
    score = score_run(alarms, run, range(39, 60), truth)
    assert score.n_events_excluded == 0 and len(score.events) == 1
    e = score.events[0]
    assert e.detected and e.delay == 0 and e.attributed == "transportation"
    assert score.n_episodes == 4 and score.n_false_alarms == 3
    # a later alarm whose window still covers the event day is a TP with a delay
    late = score_run([_alarm(ev, "transportation", 52, window=3)], run, range(39, 60), truth)
    assert late.events[0].delay == 2 and late.n_false_alarms == 0
    # wrong barrier only: event missed, but its attribution is recorded for the confusion matrix
    wrong = score_run([_alarm(ev, "illness", 50)], run, range(39, 60), truth)
    assert not wrong.events[0].detected and wrong.events[0].attributed == "illness"


def test_merge_alarms_jaccard():
    a = _alarm(["N001", "N002", "N003"], "illness", 50)
    b = _alarm(["N001", "N002", "N003", "N004"], "illness", 51)   # J = 0.75
    c = _alarm(["N001", "N009", "N010", "N011"], "illness", 52)   # J with b = 1/7
    assert [len(ep) for ep in merge_alarms([a, b, c])] == [2, 1]


def test_calibrate_threshold_hits_target():
    school = make_school()
    run = make_run(school, seed=0, rate=0.0)
    per_day = []
    for i, d in enumerate(range(40, 80)):                   # 40 days, one candidate per day, p = 0.001 * (i + 1)
        zone = frozenset(school.students[i:i + 3])
        a = Alarm("combined", "route", f"Z{i}", zone, "illness", d, 1, 3, 1.0, 5.0, 0.001 * (i + 1), zone)
        per_day.append(D.DayResult("combined", d, [a], [("route", "illness", a.p_value)]))
    cal = calibrate_threshold([per_day], [run], target_per_day=1 / 20)
    assert cal["null_false_alarms"] == 2 and cal["null_days"] == 40          # 40 days * 1/20
    assert cal["p_threshold"] == pytest.approx(0.002) and cal["threshold"] == (pytest.approx(0.002), 5.0)
    # ties in p are broken by LLR: same p everywhere, LLR decides
    for i, r in enumerate(per_day):
        r.candidates[0].p_value, r.candidates[0].llr = 0.005, 100.0 - i
    cal = calibrate_threshold([per_day], [run], target_per_day=1 / 20)
    assert cal["null_false_alarms"] == 2 and cal["threshold"] == (0.005, 99.0)


def test_spells_count_once_and_pvalues_respect_permutation_resolution():
    from app.ml.scan import gumbel_pvalues
    school = make_school()
    zs = build_zones(school, ks=(5,))
    nb = school.neighborhood["N001"]
    two = [s for s in school.students if school.neighborhood[s] == nb and school.household[s] != school.household["N001"]][:2]
    run = make_run(school, seed=1, quiet_from=45, planted=[
        ("E0", "contagion", "illness", ["N001"], [48, 49, 50]),          # one student, a 3-day spell
        ("E1", "contagion", "illness", two, [50]),
    ])
    fams = scan_day(Observations.from_run(run, zs), zs, 50, "labels", ["neighborhood"], PARAMS)
    fam = next(f for f in fams if f.barrier == "illness")
    a = next(a for a in fam.candidates if a.zone_id == nb and a.window == 3)
    assert a.observed == 3 and a.absent_members == frozenset(["N001", *two])   # spell = 1 record, not 3
    # rank p where the permutations resolve it; Gumbel only beyond every null maximum, capped at 1/(n+1)
    null = np.arange(1, 100, dtype=float) / 10                              # 99 maxima, 0.1 .. 9.9
    p = gumbel_pvalues(np.array([5.0, 9.9, 50.0, 0.0]), null)
    assert p[0] == pytest.approx(51 / 100) and p[1] == pytest.approx(2 / 100)
    assert 0 < p[2] < 1 / 100 and p[3] == 1.0
