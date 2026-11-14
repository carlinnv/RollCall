"""Tests for the PRD 2 absence simulator and the templated reply generator.

The SchoolGraph fixtures are built by hand (no dependency on app.ml.graphs).
"""
import dataclasses
import json
from collections import defaultdict

import networkx as nx
import numpy as np
import pytest

from app.ml.reply_text import LANGUAGES, STYLES, TEMPLATES, TERSE, make_reply
from app.ml.simulator import (BASELINE_DAYS, CONFIG_A, CONFIG_B, LANGUAGE_MIX, as_dict, null_config,
                              simulate, summary)
from app.ml.types import SchoolGraph
from app.models import Barrier

BARRIER_OF = {"contagion": Barrier.ILLNESS.value, "route_failure": Barrier.TRANSPORTATION.value,
              "caregiving_spike": Barrier.CAREGIVING.value, "flu_wave": Barrier.ILLNESS.value}


def make_school(n_sections=3, per_section=20, n_days=3, seed=7, with_edges=True, name="fixture"):
    rng = np.random.default_rng(seed)
    n = n_sections * per_section
    students = [f"N{i:03d}" for i in range(n)]
    section = {s: f"SEC-{i // per_section:02d}" for i, s in enumerate(students)}
    grade = {s: i // per_section + 1 for i, s in enumerate(students)}
    daily = []
    for _ in range(n_days):
        g = nx.Graph()
        g.add_nodes_from(students)
        if with_edges:
            for i, s in enumerate(students):
                sec = i // per_section
                mates = [j for j in range(sec * per_section, (sec + 1) * per_section) if j != i]
                for j in rng.choice(mates, size=5, replace=False):
                    g.add_edge(s, students[int(j)], weight=float(rng.lognormal(6.3, 0.9)))
                if rng.random() < 0.15:
                    j = int(rng.integers(n))
                    if j != i:
                        g.add_edge(s, students[j], weight=float(rng.lognormal(5.0, 0.8)))
        daily.append(g)
    school = SchoolGraph(name=name, students=students, section=section, grade=grade, daily=daily)

    # logistics: sibling pairs across sections, 3 neighborhoods, 3 routes (one per neighborhood)
    household = {s: f"H{i:04d}" for i, s in enumerate(students)}
    for k in range(0, min(per_section, n // 6)):
        a, b = students[k], students[k + per_section]  # same position, next section
        household[b] = household[a]
    hh = sorted(set(household.values()))
    hh_nb = {h: f"NB{1 + int(rng.integers(3))}" for h in hh}
    hh_route = {h: (f"R0{hh_nb[h][-1]}" if rng.random() < 0.7 else None) for h in hh}
    school.household = household
    school.neighborhood = {s: hh_nb[household[s]] for s in students}
    school.bus_route = {s: hh_route[household[s]] for s in students}
    return school


@pytest.fixture(scope="module")
def school():
    return make_school()


@pytest.fixture(scope="module")
def runs(school):
    return {seed: simulate(school, dataclasses.replace(CONFIG_A, seed=seed)) for seed in range(4)}


def union_graph(school):
    g = nx.Graph()
    for d in school.daily:
        g.add_edges_from(d.edges())
    return g


# --------------------------------------------------------------------------------------------
# determinism and structure


def test_deterministic_per_seed(school, runs):
    again = simulate(school, dataclasses.replace(CONFIG_A, seed=0))
    assert as_dict(again) == as_dict(runs[0])
    assert as_dict(runs[1]) != as_dict(runs[0])


def test_absence_ids_and_uniqueness(runs):
    for run in runs.values():
        ids = [a.id for a in run.absences]
        assert ids == [f"A{k:05d}" for k in range(len(ids))]
        cells = [(a.student, a.day) for a in run.absences]
        assert len(cells) == len(set(cells)), "one cause per student-day"
        assert all(0 <= a.day < run.n_days for a in run.absences)


def test_planted_events_members_and_barriers(runs):
    for run in runs.values():
        assert run.events, "config A plants events"
        by_event = defaultdict(list)
        for a in run.absences:
            if a.event_id is not None:
                by_event[a.event_id].append(a)
        assert set(by_event) == {e.id for e in run.events}
        for e in run.events:
            assert e.members
            assert e.barrier == BARRIER_OF[e.kind]
            evs = by_event[e.id]
            assert {a.student for a in evs} == set(e.members)
            assert all(a.true_barrier == e.barrier for a in evs)
            days = [a.day for a in evs]
            assert e.onset == min(days) and e.duration == max(days) - min(days) + 1
            assert e.onset >= BASELINE_DAYS  # n_days >= 60: keep a clean baseline
        assert "route_failure" in {e.kind for e in run.events}


def test_route_failure_members_are_riders(school, runs):
    seen = 0
    for run in runs.values():
        for e in run.events:
            if e.kind != "route_failure":
                continue
            routes = {school.bus_route[s] for s in e.members}
            assert len(routes) == 1 and None not in routes
            assert routes == {run.meta[e.id]["info"]["route"]}
            seen += 1
    assert seen >= 4


def test_contagion_members_reachable_via_contacts(school, runs):
    g = union_graph(school)
    seen = 0
    for run in runs.values():
        for e in run.events:
            if e.kind != "contagion":
                continue
            seen += 1
            sub = g.subgraph(e.members)
            # every infector is absent before its infectees, and a second seed is a contact of the first
            assert nx.number_connected_components(sub) == 1, e
    assert seen >= 4


def test_contagion_does_not_cross_missing_edges():
    school = make_school(with_edges=False)
    run = simulate(school, CONFIG_A)
    for e in run.events:
        if e.kind == "contagion":
            assert len(e.members) <= 2  # only the seeds can get sick without contacts


def test_flu_wave_does_not_alarm(runs):
    n_flu = 0
    for run in runs.values():
        flu = [e for e in run.events if e.kind == "flu_wave"]
        n_flu += len(flu)  # a flu wave on 60 students can (rarely) leave no absences and is dropped
        assert all(not e.should_alarm and e.barrier == Barrier.ILLNESS.value for e in flu)
        assert all(e.should_alarm for e in run.events if e.kind != "flu_wave")
    assert n_flu >= 3


def test_caregiving_spike_is_one_neighborhood(school, runs):
    for run in runs.values():
        for e in run.events:
            if e.kind == "caregiving_spike":
                assert len({school.neighborhood[s] for s in e.members}) == 1


def test_null_config_has_no_alarmable_events(school):
    cfg = null_config(CONFIG_A)
    assert (cfg.n_contagion, cfg.n_route_failures, cfg.n_caregiving_spikes) == (0, 0, 0)
    assert cfg.n_flu_waves == CONFIG_A.n_flu_waves and cfg.base_absence_rate == CONFIG_A.base_absence_rate
    for seed in range(3):
        run = simulate(school, dataclasses.replace(cfg, seed=seed))
        assert not [e for e in run.events if e.should_alarm]
        assert all(e.kind == "flu_wave" for e in run.events)


def test_config_b_differs_from_a():
    assert CONFIG_B.seed != CONFIG_A.seed
    assert CONFIG_B.beta != CONFIG_A.beta
    assert CONFIG_B.base_absence_rate != CONFIG_A.base_absence_rate
    assert CONFIG_B.route_absent_frac[1] <= CONFIG_A.route_absent_frac[0]
    assert CONFIG_B.reply_rate[1] <= CONFIG_A.reply_rate[0]


def test_background_rate_matches_config():
    big = make_school(n_sections=10, per_section=40, n_days=1, with_edges=False, name="big")
    cfg = dataclasses.replace(null_config(CONFIG_A), n_flu_waves=0)
    rates = []
    for seed in range(3):
        run = simulate(big, dataclasses.replace(cfg, seed=seed))
        assert not run.events
        rates.append(summary(run)["background_rate"])
    assert abs(np.mean(rates) - cfg.base_absence_rate) < 0.25 * cfg.base_absence_rate, rates


def test_siblings_sometimes_absent_together(school, runs):
    together = 0
    for run in runs.values():
        cells = {(a.student, a.day): a.true_barrier for a in run.absences if a.event_id is None}
        for (s, d), b in cells.items():
            for t in school.students:
                if t != s and school.household[t] == school.household[s] and cells.get((t, d)) == b:
                    together += 1
    assert together > 0


# --------------------------------------------------------------------------------------------
# replies


def test_replies_reference_absences_and_delays(runs):
    for run in runs.values():
        by_id = {a.id: a for a in run.absences}
        assert len({r.absence_id for r in run.replies}) == len(run.replies)
        for r in run.replies:
            a = by_id[r.absence_id]
            assert r.received_day >= a.day
            assert r.received_day - a.day <= run.config.reply_delay_max
            assert r.true_label == a.true_barrier
            assert r.text.strip()
        lo, hi = run.config.reply_rate
        assert lo - 0.1 <= len(run.replies) / len(run.absences) <= hi + 0.1


def test_reply_language_per_household(school, runs):
    for run in runs.values():
        by_id = {a.id: a for a in run.absences}
        lang_of = {}
        for r in run.replies:
            assert r.language in LANGUAGE_MIX
            h = school.household[by_id[r.absence_id].student]
            assert lang_of.setdefault(h, r.language) == r.language


def test_noisy_label_accuracy(runs):
    replies = [r for run in runs.values() for r in run.replies]
    acc = np.mean([r.observed_label == r.true_label for r in replies])
    assert abs(acc - CONFIG_A.label_accuracy) < 0.06
    valid = {b.value for b in Barrier}
    assert all(r.observed_label in valid for r in replies)


def test_classifier_hook(school):
    seen = []

    def clf(text):
        seen.append(text)
        return Barrier.UNKNOWN.value

    run = simulate(school, CONFIG_A, classifier=clf)
    assert run.replies and len(seen) == len(run.replies)
    assert all(r.observed_label == Barrier.UNKNOWN.value for r in run.replies)


def test_as_dict_is_json_serializable(runs):
    d = as_dict(runs[0])
    json.dumps(d, ensure_ascii=False)
    s = d["summary"]
    assert sum(s["absences_per_day"]) == s["n_absences"] == len(runs[0].absences)
    assert len(s["events"]) == len(runs[0].events)


# --------------------------------------------------------------------------------------------
# reply_text


GENERATED = [b.value for b in Barrier if b is not Barrier.SAFETY]


def test_template_coverage():
    for b in GENERATED:
        assert len(TEMPLATES["en"][b]) >= 6 and len(TEMPLATES["es"][b]) >= 6, b
        assert len(TEMPLATES["vi"][b]) >= 3, b
        for lang in LANGUAGES:
            assert TEMPLATES[lang][b] and TERSE[lang][b]
    assert Barrier.SAFETY.value not in TEMPLATES["en"]


def test_make_reply_all_combinations():
    rng = np.random.default_rng(3)
    for b in GENERATED:
        for lang in LANGUAGES + ("fr",):  # unsupported languages fall back to English
            for style in STYLES:
                for _ in range(5):
                    text = make_reply(b, lang, rng, style)
                    assert text.strip() and "{" not in text and "}" not in text


def test_make_reply_refuses_safety():
    with pytest.raises(ValueError):
        make_reply(Barrier.SAFETY.value, "en", np.random.default_rng(0))


def test_make_reply_variety_and_noise():
    rng = np.random.default_rng(5)
    normal = {make_reply(Barrier.ILLNESS.value, "es", rng, "normal") for _ in range(200)}
    assert len(normal) >= 10
    assert any("fever" in t or "flu" in t for t in normal)  # code-switching happens
    noisy = [make_reply(Barrier.ILLNESS.value, "es", rng, "noisy") for _ in range(200)]
    assert any(t == t.lower() for t in noisy)
    assert any(not t.endswith(".") for t in noisy)
