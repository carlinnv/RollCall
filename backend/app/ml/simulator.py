"""Absence simulator for PRD 2 (§2 "Simulator"): planted events on a real contact graph.

`simulate(school, config, classifier=None)` returns a SimRun with daily absences, the
family replies the pipeline would see, and the ground-truth planted events. Everything is
drawn from one numpy Generator seeded by `config.seed`, so a (school, config) pair always
gives the same run.

Mechanisms (the detectors must not encode these; see PRD 2 "circularity"):
  * Background: per-student propensity ~ Gamma(chronic_shape) with mean base_absence_rate;
    absences come in short spells with a realistic barrier mix (chronic absentees lean to
    disengagement/housing/caregiving). Siblings sometimes join a spell (same reason, same days).
  * Contagion (SEIR): 1-2 seeds; each school day susceptible j is infected with
    p = 1 - prod_i (1 - p_ij), p_ij = 1 - exp(-beta * w_ij), over infectious contacts i that
    are present in school.day_graph(day). Latent 1 day, infectious (at school, pre-symptomatic)
    `infectious_days`, then absent with illness for 2-3 days. Only a share of students
    (SUSCEPTIBLE_FRAC) is susceptible, which keeps outbreaks local.
  * Route failure: one bus route fails for 1-2 days; each rider is absent with probability
    f ~ U(route_absent_frac) per day ("transportation").
  * Caregiving spike (untargeted event type): a neighborhood daycare closes; 20-40% of that
    neighborhood's households with younger kids keep a child home 1-3 days.
  * Flu wave (should_alarm=False): the illness rate of every student rises x2-3 for ~5 days,
    uniformly at random (not along contacts).
  * Confounded events (bus_illness, road_closure, section_caregiving): events whose zone type
    points at the wrong cause. See CONFOUNDED_EVENTS below; they were pre-registered.
A student-day carries one cause. Planted events only take students who would otherwise be
in school, except contagion, which overrides any other cause on its symptomatic days.

Replies: households have a responsiveness ~ U(reply_rate), a language and a writing style.
Each absence gets a reply with that probability, received 0..reply_delay_max days later.
observed_label = classifier(text) when a classifier is given, else a noisy proxy that is
correct with probability label_accuracy and otherwise a plausible confusion or "unknown".
"""
from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from app.ml.reply_text import make_reply
from app.ml.types import AbsenceEvent, PlantedEvent, Reply, SchoolGraph, SimConfig, SimRun
from app.models import Barrier

ILLNESS = Barrier.ILLNESS.value
TRANSPORTATION = Barrier.TRANSPORTATION.value
HOUSING = Barrier.HOUSING_FOOD.value
CAREGIVING = Barrier.CAREGIVING.value
DISENGAGEMENT = Barrier.DISENGAGEMENT.value
APPOINTMENT = Barrier.APPOINTMENT.value
UNKNOWN = Barrier.UNKNOWN.value

# Background barrier mixes (safety is never simulated: it is evaluated on a frozen suite).
BACKGROUND_MIX: Dict[str, float] = {
    ILLNESS: 0.42, APPOINTMENT: 0.15, CAREGIVING: 0.10, TRANSPORTATION: 0.08,
    DISENGAGEMENT: 0.08, HOUSING: 0.04, UNKNOWN: 0.13,
}
CHRONIC_MIX: Dict[str, float] = {  # students with propensity > 2x the base rate
    ILLNESS: 0.32, APPOINTMENT: 0.07, CAREGIVING: 0.13, TRANSPORTATION: 0.10,
    DISENGAGEMENT: 0.18, HOUSING: 0.10, UNKNOWN: 0.10,
}
SPELL_LEN: Dict[str, Tuple[int, int]] = {  # inclusive range of a background spell's length (days)
    ILLNESS: (1, 3), APPOINTMENT: (1, 1), CAREGIVING: (1, 2), TRANSPORTATION: (1, 1),
    DISENGAGEMENT: (1, 2), HOUSING: (1, 3), UNKNOWN: (1, 1),
}
SIBLING_JOIN: Dict[str, float] = {  # P(a sibling is absent too, same reason and days)
    ILLNESS: 0.3, APPOINTMENT: 0.6, CAREGIVING: 0.4, TRANSPORTATION: 0.7,
    DISENGAGEMENT: 0.0, HOUSING: 0.9, UNKNOWN: 0.4,
}
MAX_DAILY_PROPENSITY = 0.5

SUSCEPTIBLE_FRAC = 0.6          # share of students who can catch a planted contagion
SYMPTOMATIC_DAYS = (2, 3)       # absent days after the infectious period
MIN_OUTBREAK = 3                # planted outbreaks are re-seeded until they reach this many cases
MAX_OUTBREAK_FRAC = 0.15        # ... and stay below this share of the school (keeps events local)
CONTAGION_TRIES = 20
ROUTE_DAYS = (1, 2)
MIN_ROUTE_RIDERS = 4
MIN_ROUTE_ABSENT = 2
CAREGIVING_HH_FRAC = (0.2, 0.4)
YOUNG_KIDS_FRAC = 0.5           # households with a younger (pre-school) child
CAREGIVING_DAYS = (1, 3)
FLU_DAYS = (4, 6)
FLU_MULT = (2.0, 3.0)
BASELINE_DAYS = 40              # no planted event starts before this day when n_days >= 60

# ----------------------------------------------------------------------------------------------
# Pre-registered confounded events.
CONFOUNDED_EVENTS = """
These three event types were written down and fixed on 2026-11-14 BEFORE any detector was evaluated
on them, and they must not be tuned afterwards to favour any detector. Each one breaks the
"zone type = cause" shortcut that graph-only attribution relies on (in the original simulator every
event type lined up with exactly one zone type, which is a simulator artifact, not reality):

  bus_illness (truth: illness; zone type: route -> graph-only says "transportation")
      An SEIR outbreak seeded in 1-2 riders of one bus route. Besides the school contact graph it
      transmits along co-ridership: every pair of riders of that route who are both at school that
      day shares BUS_CONTACT_S (~20 min) of contact, with the same p = 1 - exp(-beta * w). Same
      outbreak-size conditioning as contagion (each re-seeding try draws a route afresh); school
      contacts can still carry it off the bus. Realistic: respiratory and GI illness spreading on
      school buses is well documented; the absences then cluster by route although the barrier is
      illness, not a bus that failed to come.
  road_closure (truth: transportation; zone type: neighborhood -> graph-only says "caregiving")
      A road into one neighborhood served by (at least) two routes is closed for 1-2 days. Per
      household and day: riders of those two routes living in the neighborhood are absent with
      probability f ~ U(route_absent_frac); riders of those routes living elsewhere (bus delayed
      or rerouted) with ROAD_THROUGH_FRAC * f; non-riders living in the neighborhood (car trips)
      with ROAD_CAR_FRAC. Realistic: closures, floods, a broken bridge, or construction strand a
      whole neighborhood, riders and drivers alike, and do not respect route boundaries.
  section_caregiving (truth: caregiving or family obligation; zone type: section -> "illness")
      A cultural or religious observance kept by families concentrated in one class section (for
      example a language-program class): a share SECTION_OBS_FRAC of the section's households keep
      their child home on the same 1-2 days (siblings in other sections join with
      SECTION_SIBLING_JOIN), plus SECTION_OUTSIDE_RATE of other households in the school.
      Realistic: families sharing a language or culture are often placed in the same section, and
      observances fall on the same day(s) for all of them.

One revision, made before any detector had been run on these events: a simulator-only check of
event sizes showed bus_illness did not meet its own declared size conditioning (bus contacts on
every route exploded to ~40% of Marseille under config A's beta; a single fixed small route
fizzled to one case in Lyon under config B). Bus contacts were restricted to the seeded route and
each re-seeding try now draws the route afresh. Nothing has been changed since detectors were run.
"""
BUS_CONTACT_S = 1200.0          # seconds of contact per co-rider pair per school day (~20 min ride)
ROAD_DAYS = (1, 2)
ROAD_THROUGH_FRAC = 0.5         # riders of the closed routes living elsewhere: absent w.p. 0.5 * f
ROAD_CAR_FRAC = (0.15, 0.35)    # non-rider households of the closed neighborhood
MIN_ROAD_HOUSEHOLDS = 3
SECTION_OBS_FRAC = (0.25, 0.45)
SECTION_OUTSIDE_RATE = 0.03
SECTION_SIBLING_JOIN = 0.9
SECTION_TWO_DAY_P = 0.4         # observance lasts 2 days with this probability, else 1
MIN_SECTION_HOUSEHOLDS = 3
MIN_SECTION_SIZE = 8

ILLNESS_KINDS = ("contagion", "bus_illness")

LANGUAGE_MIX: Dict[str, float] = {"en": 0.50, "es": 0.35, "vi": 0.10, "zh": 0.05}
STYLE_MIX: Dict[str, float] = {"normal": 0.55, "terse": 0.30, "noisy": 0.15}
DELAY_DECAY = 0.5               # P(delay = d) proportional to DELAY_DECAY ** d

# Plausible confusions for the noisy label proxy (true -> candidates other than "unknown").
CONFUSION: Dict[str, List[str]] = {
    ILLNESS: [APPOINTMENT, CAREGIVING],
    APPOINTMENT: [ILLNESS],
    TRANSPORTATION: [CAREGIVING, HOUSING],
    CAREGIVING: [ILLNESS, TRANSPORTATION],
    HOUSING: [TRANSPORTATION, CAREGIVING],
    DISENGAGEMENT: [ILLNESS],
    UNKNOWN: [ILLNESS, APPOINTMENT],
}
P_CONFUSE_TO_UNKNOWN = 0.4

# Planted events are applied from the end of this tuple to its start; later-applied kinds win
# student-days (contagion and bus_illness override any other cause).
_PRIORITY_KINDS = ("contagion", "bus_illness", "route_failure", "road_closure", "caregiving_spike",
                   "section_caregiving", "flu_wave")
BARRIER_OF_KIND: Dict[str, str] = {
    "contagion": ILLNESS, "bus_illness": ILLNESS, "route_failure": TRANSPORTATION,
    "road_closure": TRANSPORTATION, "caregiving_spike": CAREGIVING, "section_caregiving": CAREGIVING,
    "flu_wave": ILLNESS,
}


# ----------------------------------------------------------------------------------------------
# presets

CONFIG_A = SimConfig(
    name="A", n_days=100, seed=0,
    base_absence_rate=0.03, chronic_shape=0.6,
    n_contagion=2, beta=0.0002, infectious_days=2,
    n_route_failures=2, route_absent_frac=(0.5, 0.8),
    n_caregiving_spikes=1, n_flu_waves=1,
    reply_rate=(0.6, 0.9), reply_delay_max=2, label_accuracy=0.85,
    n_bus_illness=1, n_road_closures=1, n_section_caregiving=1,
)

# Held-out config B: different (mostly non-overlapping) ranges, tested once.
CONFIG_B = SimConfig(
    name="B", n_days=100, seed=1000,
    base_absence_rate=0.045, chronic_shape=0.4,
    n_contagion=2, beta=0.00008, infectious_days=3,
    n_route_failures=3, route_absent_frac=(0.3, 0.5),
    n_caregiving_spikes=2, n_flu_waves=1,
    reply_rate=(0.4, 0.6), reply_delay_max=3, label_accuracy=0.80,
    n_bus_illness=1, n_road_closures=1, n_section_caregiving=1,
)


def null_config(cfg: SimConfig) -> SimConfig:
    """Same config with no alarmable planted events (flu waves kept): for false-alarm calibration."""
    return dataclasses.replace(cfg, name=f"{cfg.name}-null", n_contagion=0, n_route_failures=0,
                               n_caregiving_spikes=0, n_bus_illness=0, n_road_closures=0,
                               n_section_caregiving=0)


# ----------------------------------------------------------------------------------------------
# helpers


def _choice(rng: np.random.Generator, mix: Dict[str, float]) -> str:
    keys = list(mix)
    p = np.array([mix[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _rint(rng: np.random.Generator, lo_hi: Tuple[int, int]) -> int:
    lo, hi = lo_hi
    return int(rng.integers(lo, hi + 1))


def _runif(rng: np.random.Generator, lo_hi: Tuple[float, float]) -> float:
    lo, hi = lo_hi
    return float(lo + (hi - lo) * rng.random())


def _mean_spell(mix: Dict[str, float]) -> Tuple[float, float]:
    """(expected spell length, expected sibling-join probability weighted by length)."""
    z = sum(mix.values())
    length = sum(mix[b] / z * (SPELL_LEN[b][0] + SPELL_LEN[b][1]) / 2 for b in mix)
    join = sum(mix[b] / z * (SPELL_LEN[b][0] + SPELL_LEN[b][1]) / 2 * SIBLING_JOIN[b] for b in mix) / length
    return length, join


class _Grid:
    """One cause per student-day: barrier string and planted-event index (-1 = background)."""

    def __init__(self, n: int, n_days: int):
        self.n_days = n_days
        self.barrier: List[List[Optional[str]]] = [[None] * n_days for _ in range(n)]
        self.event = np.full((n, n_days), -1, dtype=np.int32)

    def free(self, i: int, d: int) -> bool:
        return self.barrier[i][d] is None

    def set(self, i: int, d: int, barrier: str, event: int = -1) -> None:
        self.barrier[i][d] = barrier
        self.event[i, d] = event


def _schedule(rng: np.random.Generator, cfg: SimConfig) -> List[Tuple[str, int]]:
    """Spread planted event onsets over [BASELINE_DAYS, n_days) in shuffled, non-overlapping slots."""
    kinds = (["contagion"] * cfg.n_contagion + ["route_failure"] * cfg.n_route_failures
             + ["caregiving_spike"] * cfg.n_caregiving_spikes + ["flu_wave"] * cfg.n_flu_waves
             + ["bus_illness"] * getattr(cfg, "n_bus_illness", 0)
             + ["road_closure"] * getattr(cfg, "n_road_closures", 0)
             + ["section_caregiving"] * getattr(cfg, "n_section_caregiving", 0))
    if not kinds:
        return []
    start = BASELINE_DAYS if cfg.n_days >= 60 else max(1, cfg.n_days // 3)
    end = max(start + 1, cfg.n_days - 6)  # leave room for an outbreak to show
    best, best_bad = None, None
    for _ in range(50):
        order = [kinds[i] for i in rng.permutation(len(kinds))]
        # avoid: an outbreak in the last slot; flu wave next to an outbreak (both illness)
        bad = (order[-1] in ILLNESS_KINDS and len(order) > 1) + sum(
            1 for a, b in zip(order, order[1:])
            if (a in ILLNESS_KINDS and b == "flu_wave") or (b in ILLNESS_KINDS and a == "flu_wave"))
        if best_bad is None or bad < best_bad:
            best, best_bad = order, bad
        if bad == 0:
            break
    edges = np.linspace(start, end, len(best) + 1)
    out = []
    for k, kind in enumerate(best):
        lo, hi = int(np.ceil(edges[k])), int(np.floor(edges[k + 1]))
        hi = max(lo, min(hi, lo + max(1, (hi - lo) // 2)))  # onset in the first half of its slot
        out.append((kind, int(rng.integers(lo, hi + 1))))
    return out


# ----------------------------------------------------------------------------------------------
# mechanisms


def _background(rng, cfg, school, idx, siblings, grid, propensity):
    n, T = len(school.students), cfg.n_days
    mix_len, mix_join = _mean_spell(BACKGROUND_MIX)
    hh_size = np.array([1 + len(siblings[i]) for i in range(n)], dtype=float)
    start_p = propensity / (mix_len * (1.0 + (hh_size - 1.0) * mix_join))
    chronic = propensity > 2.0 * cfg.base_absence_rate
    for d in range(T):
        draws = rng.random(n)
        for i in np.flatnonzero(draws < start_p):
            i = int(i)
            if not grid.free(i, d):
                continue
            b = _choice(rng, CHRONIC_MIX if chronic[i] else BACKGROUND_MIX)
            length = _rint(rng, SPELL_LEN[b])
            group = [i] + [j for j in siblings[i] if grid.free(j, d) and rng.random() < SIBLING_JOIN[b]]
            for j in group:
                for dd in range(d, min(T, d + length)):
                    if grid.free(j, dd):
                        grid.set(j, dd, b)


def _flu_wave(rng, cfg, n, grid, ev, onset, propensity_illness):
    T = cfg.n_days
    length = _rint(rng, FLU_DAYS)
    mult = _runif(rng, FLU_MULT)
    for t in range(length):
        d = onset + t
        if d >= T:
            break
        hump = np.sin(np.pi * (t + 0.5) / length)
        q = propensity_illness * (mult - 1.0) * hump / 1.5  # spells of 1-2 days
        for i in np.flatnonzero(rng.random(n) < q):
            i = int(i)
            if not grid.free(i, d):
                continue
            for dd in range(d, min(T, d + _rint(rng, (1, 2)))):
                if grid.free(i, dd):
                    grid.set(i, dd, ILLNESS, ev)


def _route_failure(rng, cfg, school, idx, grid, ev, onset, used_routes):
    riders: Dict[str, List[int]] = defaultdict(list)
    for s in school.students:
        r = school.bus_route.get(s)
        if r is not None:
            riders[r].append(idx[s])
    routes = sorted(r for r, m in riders.items() if len(m) >= MIN_ROUTE_RIDERS and r not in used_routes)
    if not routes:
        routes = sorted(r for r, m in riders.items() if len(m) >= 2)
    if not routes:
        return None
    route = routes[int(rng.integers(len(routes)))]
    used_routes.add(route)
    frac = _runif(rng, cfg.route_absent_frac)
    days = _rint(rng, ROUTE_DAYS)
    hit = 0
    for d in range(onset, min(cfg.n_days, onset + days)):
        for i in riders[route]:
            if grid.free(i, d) and rng.random() < frac:
                grid.set(i, d, TRANSPORTATION, ev)
                hit += 1
    if hit < MIN_ROUTE_ABSENT and onset < cfg.n_days:  # a failure nobody notices is not an event
        free = [i for i in riders[route] if grid.free(i, onset)]
        for k in rng.permutation(len(free))[: MIN_ROUTE_ABSENT - hit]:
            grid.set(free[int(k)], onset, TRANSPORTATION, ev)
    return {"route": route, "frac": frac}


def _caregiving_spike(rng, cfg, school, hh_members, grid, ev, onset, young, used_nbs):
    nb_households: Dict[str, List[str]] = defaultdict(list)
    for h, members in hh_members.items():
        nb = school.neighborhood.get(school.students[members[0]])
        if nb is not None and young[h]:
            nb_households[nb].append(h)
    nbs = sorted(nb for nb, hs in nb_households.items() if len(hs) >= 3 and nb not in used_nbs)
    if not nbs:
        nbs = sorted(nb for nb, hs in nb_households.items() if hs)
    if not nbs:
        return None
    nb = nbs[int(rng.integers(len(nbs)))]
    used_nbs.add(nb)
    frac = _runif(rng, CAREGIVING_HH_FRAC)
    hs = nb_households[nb]
    hit = [h for h in hs if rng.random() < frac]
    if len(hit) < 2:
        hit = [hs[int(k)] for k in rng.permutation(len(hs))[: min(2, len(hs))]]
    for h in hit:
        members = hh_members[h]
        kids = [members[int(rng.integers(len(members)))]]
        kids += [j for j in members if j != kids[0] and rng.random() < 0.25]
        length = _rint(rng, CAREGIVING_DAYS)
        start = onset + (1 if rng.random() < 0.2 else 0)
        for i in kids:
            for d in range(start, min(cfg.n_days, start + length)):
                if grid.free(i, d):
                    grid.set(i, d, CAREGIVING, ev)
    return {"neighborhood": nb, "households": len(hit)}


def _seir(rng, cfg, school, idx, grid, seed_day, seeds, susceptible, bus=None):
    """One SEIR run (read-only on grid). Returns {student index: exposure day}.

    bus: optional (route group per student, -1 = no route). Riders of the same route who are both
    at school that day add BUS_CONTACT_S seconds of contact per pair (bus_illness only).
    """
    T, names = cfg.n_days, school.students
    susc = susceptible.copy()
    exposed_on: Dict[int, int] = {}
    for s in seeds:
        exposed_on[s] = seed_day
        susc[s] = False
    for d in range(seed_day, T):
        # infectious (pre-symptomatic, at school) students present today
        infectious = [i for i, e in exposed_on.items()
                      if e + 1 <= d <= e + cfg.infectious_days and grid.free(i, d)]
        if not infectious:
            if all(e + cfg.infectious_days < d for e in exposed_on.values()):
                break
            continue
        g = school.day_graph(d)
        pressure: Dict[int, float] = defaultdict(float)
        for i in infectious:
            u = names[i]
            if u not in g:
                continue
            for v, data in g[u].items():
                j = idx.get(v)
                if j is not None and susc[j] and grid.free(j, d):
                    pressure[j] += float(data.get("weight", 1.0))
        if bus is not None:
            n_inf = defaultdict(int)
            for i in infectious:
                if bus[i] >= 0:
                    n_inf[int(bus[i])] += 1
            for r, k in n_inf.items():
                for j in np.flatnonzero((bus == r) & susc):
                    j = int(j)
                    if grid.free(j, d):
                        pressure[j] += k * BUS_CONTACT_S
        # 1 - prod_i exp(-beta w_ij) = 1 - exp(-beta * sum_i w_ij)
        for j in sorted(pressure):
            if rng.random() < 1.0 - np.exp(-cfg.beta * pressure[j]):
                exposed_on[j] = d
                susc[j] = False
    return exposed_on


def _contagion(rng, cfg, school, idx, grid, ev, onset, susceptible):
    """Planted SEIR outbreak whose seeds first miss school on `onset`.

    Outbreaks that fizzle (< MIN_OUTBREAK cases) or explode (> MAX_OUTBREAK_FRAC of the school)
    are re-seeded up to CONTAGION_TRIES times, i.e. planted events are conditioned on a local
    outbreak that takes off; the run closest to that range is kept otherwise.
    """
    n, T = len(school.students), cfg.n_days
    names = school.students
    seed_day = max(0, onset - 1 - cfg.infectious_days)
    g0 = school.day_graph(seed_day)
    candidates = [i for i in range(n) if susceptible[i] and grid.free(i, seed_day)
                  and names[i] in g0 and g0.degree(names[i]) > 0]
    if not candidates:
        candidates = [i for i in range(n) if susceptible[i]] or list(range(n))
    max_size = max(MIN_OUTBREAK, int(MAX_OUTBREAK_FRAC * n))

    def miss(size: int) -> int:
        return max(0, MIN_OUTBREAK - size) + max(0, size - max_size)

    best = None
    for _ in range(CONTAGION_TRIES):
        seeds = [candidates[int(rng.integers(len(candidates)))]]
        if rng.random() < 0.5 and names[seeds[0]] in g0:
            nbrs = [idx[v] for v in g0.neighbors(names[seeds[0]])
                    if v in idx and susceptible[idx[v]] and idx[v] != seeds[0]]
            if nbrs:
                seeds.append(nbrs[int(rng.integers(len(nbrs)))])
        exposed_on = _seir(rng, cfg, school, idx, grid, seed_day, seeds, susceptible)
        if best is None or miss(len(exposed_on)) < miss(len(best[1])):
            best = (seeds, exposed_on)
        if miss(len(exposed_on)) == 0:
            break
    seeds, exposed_on = best
    _apply_outbreak(rng, cfg, grid, ev, exposed_on, susceptible)
    return {"seeds": [names[s] for s in seeds], "infected": len(exposed_on)}


def _apply_outbreak(rng, cfg, grid, ev, exposed_on, susceptible):
    for i, e in sorted(exposed_on.items()):
        susceptible[i] = False
        first = e + 1 + cfg.infectious_days
        for dd in range(first, min(cfg.n_days, first + _rint(rng, SYMPTOMATIC_DAYS))):
            grid.set(i, dd, ILLNESS, ev)  # contagion overrides any other cause


def _riders(school, idx) -> Dict[str, List[int]]:
    riders: Dict[str, List[int]] = defaultdict(list)
    for s in school.students:
        r = school.bus_route.get(s)
        if r is not None:
            riders[r].append(idx[s])
    return riders


def _bus_illness(rng, cfg, school, idx, grid, ev, onset, susceptible, used_routes):
    """Pre-registered (CONFOUNDED_EVENTS): an outbreak seeded on one route, spreading on its bus too.

    Same conditioning as _contagion (re-seeded until MIN_OUTBREAK <= size <= MAX_OUTBREAK_FRAC * n,
    CONTAGION_TRIES tries); each try draws the route and its 1-2 rider seeds afresh.
    """
    n = len(school.students)
    riders = _riders(school, idx)
    routes = sorted(r for r, m in riders.items() if len(m) >= MIN_ROUTE_RIDERS and r not in used_routes)
    if not routes:
        routes = sorted(r for r, m in riders.items() if len(m) >= 2)
    if not routes:
        return None
    seed_day = max(0, onset - 1 - cfg.infectious_days)
    max_size = max(MIN_OUTBREAK, int(MAX_OUTBREAK_FRAC * n))

    def miss(size: int) -> int:
        return max(0, MIN_OUTBREAK - size) + max(0, size - max_size)

    best = None
    for _ in range(CONTAGION_TRIES):
        route = routes[int(rng.integers(len(routes)))]
        pool = [i for i in riders[route] if susceptible[i] and grid.free(i, seed_day)] or list(riders[route])
        seeds = [pool[int(rng.integers(len(pool)))]]
        if rng.random() < 0.5:
            mates = [i for i in pool if i != seeds[0]]
            if mates:
                seeds.append(mates[int(rng.integers(len(mates)))])
        bus = np.full(n, -1, dtype=np.int64)
        bus[riders[route]] = 0
        exposed_on = _seir(rng, cfg, school, idx, grid, seed_day, seeds, susceptible, bus)
        if best is None or miss(len(exposed_on)) < miss(len(best[2])):
            best = (route, seeds, exposed_on)
        if miss(len(exposed_on)) == 0:
            break
    route, seeds, exposed_on = best
    used_routes.add(route)
    _apply_outbreak(rng, cfg, grid, ev, exposed_on, susceptible)
    return {"route": route, "seeds": [school.students[s] for s in seeds], "infected": len(exposed_on)}


def _road_closure(rng, cfg, school, hh_members, grid, ev, onset, used_routes, used_nbs):
    """Pre-registered (CONFOUNDED_EVENTS): a neighborhood cut off; its 2 routes and car trips fail."""
    names = school.students
    nb_routes: Dict[str, set] = defaultdict(set)
    for s in names:
        r, nb = school.bus_route.get(s), school.neighborhood.get(s)
        if r is not None and nb is not None:
            nb_routes[nb].add(r)
    nbs = sorted(nb for nb, rs in nb_routes.items() if len(rs) >= 2 and nb not in used_nbs)
    if not nbs:
        nbs = sorted(nb for nb, rs in nb_routes.items() if len(rs) >= 2) or sorted(nb_routes)
    if not nbs:
        return None
    nb = nbs[int(rng.integers(len(nbs)))]
    used_nbs.add(nb)
    cand = sorted(nb_routes[nb])
    routes = sorted(cand[int(k)] for k in rng.permutation(len(cand))[:2])
    used_routes.update(routes)
    frac = _runif(rng, cfg.route_absent_frac)
    car = _runif(rng, ROAD_CAR_FRAC)
    days = _rint(rng, ROAD_DAYS)
    p_of: Dict[str, float] = {}
    for h, members in hh_members.items():
        s0 = names[members[0]]
        r, home = school.bus_route.get(s0), school.neighborhood.get(s0)
        if r in routes:
            p_of[h] = frac if home == nb else ROAD_THROUGH_FRAC * frac
        elif r is None and home == nb:
            p_of[h] = car
    hit = set()
    for d in range(onset, min(cfg.n_days, onset + days)):
        for h in sorted(p_of):
            if rng.random() < p_of[h]:
                for i in hh_members[h]:
                    if grid.free(i, d):
                        grid.set(i, d, TRANSPORTATION, ev)
                        hit.add(h)
    if len(hit) < MIN_ROAD_HOUSEHOLDS and onset < cfg.n_days:   # a closure nobody notices is not an event
        local = [h for h in sorted(p_of) if h not in hit
                 and school.neighborhood.get(names[hh_members[h][0]]) == nb]
        for k in rng.permutation(len(local))[: MIN_ROAD_HOUSEHOLDS - len(hit)]:
            for i in hh_members[local[int(k)]]:
                if grid.free(i, onset):
                    grid.set(i, onset, TRANSPORTATION, ev)
    return {"neighborhood": nb, "routes": routes, "frac": frac, "car_frac": car}


def _section_caregiving(rng, cfg, school, idx, hh_members, hh_of, grid, ev, onset, used_secs):
    """Pre-registered (CONFOUNDED_EVENTS): a family observance concentrated in one class section."""
    names = school.students
    by_sec: Dict[str, List[int]] = defaultdict(list)
    for s in names:
        by_sec[school.section[s]].append(idx[s])
    secs = sorted(k for k, m in by_sec.items() if len(m) >= MIN_SECTION_SIZE and k not in used_secs)
    if not secs:
        secs = sorted(by_sec)
    if not secs:
        return None
    sec = secs[int(rng.integers(len(secs)))]
    used_secs.add(sec)
    frac = _runif(rng, SECTION_OBS_FRAC)
    days = 2 if rng.random() < SECTION_TWO_DAY_P else 1
    in_sec = sorted({hh_of[i] for i in by_sec[sec]})
    in_sec_set = set(in_sec)
    observe = [h for h in in_sec if rng.random() < frac]
    if len(observe) < MIN_SECTION_HOUSEHOLDS:
        rest = [h for h in in_sec if h not in observe]
        observe += [rest[int(k)] for k in rng.permutation(len(rest))[: MIN_SECTION_HOUSEHOLDS - len(observe)]]
    outside = [h for h in sorted(hh_members) if h not in in_sec_set and rng.random() < SECTION_OUTSIDE_RATE]
    for h in sorted(observe) + outside:
        members = hh_members[h]
        if h in in_sec_set:
            anchor = [i for i in members if school.section[names[i]] == sec]
        else:
            anchor = [members[int(rng.integers(len(members)))]]
        kids = anchor + [i for i in members if i not in anchor and rng.random() < SECTION_SIBLING_JOIN]
        for i in kids:
            for d in range(onset, min(cfg.n_days, onset + days)):
                if grid.free(i, d):
                    grid.set(i, d, CAREGIVING, ev)
    return {"section": sec, "households": len(observe), "outside_households": len(outside), "frac": frac}


# ----------------------------------------------------------------------------------------------
# replies


def _noisy_label(rng: np.random.Generator, true: str, accuracy: float) -> str:
    if rng.random() < accuracy:
        return true
    if true != UNKNOWN and rng.random() < P_CONFUSE_TO_UNKNOWN:
        return UNKNOWN
    options = [b for b in CONFUSION.get(true, [UNKNOWN]) if b != true] or [UNKNOWN]
    return options[int(rng.integers(len(options)))]


# ----------------------------------------------------------------------------------------------
# main entry point


def simulate(school: SchoolGraph, config: SimConfig,
             classifier: Optional[Callable[[str], str]] = None) -> SimRun:
    """Simulate `config.n_days` school days of absences, replies and planted events on `school`.

    `school` must carry logistics layers (graphs.assign_logistics); missing entries mean
    "no route" / "own household" / "no neighborhood". Deterministic given config.seed
    (and a deterministic classifier).
    """
    cfg = config
    rng = np.random.default_rng(cfg.seed)
    names = list(school.students)
    n, T = len(names), cfg.n_days
    idx = {s: i for i, s in enumerate(names)}

    households: Dict[str, List[int]] = defaultdict(list)
    for s in names:
        households[school.household.get(s, f"_solo_{s}")].append(idx[s])
    hh_of = [school.household.get(s, f"_solo_{s}") for s in names]
    siblings = [[j for j in households[hh_of[i]] if j != i] for i in range(n)]
    hh_keys = list(households)

    # per-student propensity, per-household traits (drawn up front so they do not depend on events)
    shape = max(cfg.chronic_shape, 1e-3)
    propensity = np.minimum(rng.gamma(shape, cfg.base_absence_rate / shape, size=n), MAX_DAILY_PROPENSITY)
    lang_keys = list(LANGUAGE_MIX)
    lang_p = np.array([LANGUAGE_MIX[k] for k in lang_keys])
    style_keys = list(STYLE_MIX)
    style_p = np.array([STYLE_MIX[k] for k in style_keys])
    hh_lang = {h: lang_keys[int(k)] for h, k in zip(hh_keys, rng.choice(len(lang_keys), size=len(hh_keys), p=lang_p))}
    hh_style = {h: style_keys[int(k)] for h, k in zip(hh_keys, rng.choice(len(style_keys), size=len(hh_keys), p=style_p))}
    lo, hi = cfg.reply_rate
    hh_reply = {h: float(r) for h, r in zip(hh_keys, lo + (hi - lo) * rng.random(len(hh_keys)))}
    young = {h: bool(y) for h, y in zip(hh_keys, rng.random(len(hh_keys)) < YOUNG_KIDS_FRAC)}
    susceptible = rng.random(n) < SUSCEPTIBLE_FRAC

    grid = _Grid(n, T)
    _background(rng, cfg, school, idx, siblings, grid, propensity)

    # planted events, applied by kind (lower-priority kinds first; contagion overrides last)
    plan = _schedule(rng, cfg)
    meta: List[dict] = []
    for kind, onset in plan:
        meta.append({"kind": kind, "planned_onset": onset})
    order = sorted(range(len(plan)), key=lambda k: (_PRIORITY_KINDS.index(plan[k][0]) * -1, plan[k][1]))
    used_routes: set = set()
    used_nbs: set = set()
    used_secs: set = set()
    illness_share = BACKGROUND_MIX[ILLNESS] / sum(BACKGROUND_MIX.values())
    flu_q = np.full(n, cfg.base_absence_rate * illness_share)  # uniform, not propensity-weighted
    for k in order:
        kind, onset = plan[k]
        if kind == "flu_wave":
            _flu_wave(rng, cfg, n, grid, k, onset, flu_q)
            info = {}
        elif kind == "caregiving_spike":
            info = _caregiving_spike(rng, cfg, school, households, grid, k, onset, young, used_nbs)
        elif kind == "route_failure":
            info = _route_failure(rng, cfg, school, idx, grid, k, onset, used_routes)
        elif kind == "bus_illness":
            info = _bus_illness(rng, cfg, school, idx, grid, k, onset, susceptible, used_routes)
        elif kind == "road_closure":
            info = _road_closure(rng, cfg, school, households, grid, k, onset, used_routes, used_nbs)
        elif kind == "section_caregiving":
            info = _section_caregiving(rng, cfg, school, idx, households, hh_of, grid, k, onset, used_secs)
        else:
            info = _contagion(rng, cfg, school, idx, grid, k, onset, susceptible)
        meta[k]["info"] = info or {}

    # collect absences (day-major) and planted-event members
    members: Dict[int, set] = defaultdict(set)
    days_of: Dict[int, List[int]] = defaultdict(list)
    cells: List[Tuple[int, int, str, int]] = []
    for d in range(T):
        for i in range(n):
            b = grid.barrier[i][d]
            if b is None:
                continue
            e = int(grid.event[i, d])
            cells.append((i, d, b, e))
            if e >= 0:
                members[e].add(names[i])
                days_of[e].append(d)

    kept = sorted((k for k in range(len(plan)) if members[k]), key=lambda k: (min(days_of[k]), k))
    eid = {k: f"E{r:03d}" for r, k in enumerate(kept)}
    events = [PlantedEvent(id=eid[k], kind=plan[k][0], barrier=BARRIER_OF_KIND[plan[k][0]],
                           members=frozenset(members[k]), onset=min(days_of[k]),
                           duration=max(days_of[k]) - min(days_of[k]) + 1,
                           should_alarm=plan[k][0] != "flu_wave")
              for k in kept]

    absences: List[AbsenceEvent] = []
    for a, (i, d, b, e) in enumerate(cells):
        absences.append(AbsenceEvent(id=f"A{a:05d}", student=names[i], day=d, true_barrier=b,
                                     event_id=eid.get(e) if e >= 0 else None))

    # replies
    delays = np.arange(cfg.reply_delay_max + 1)
    delay_p = DELAY_DECAY ** delays
    delay_p = delay_p / delay_p.sum()
    replies: List[Reply] = []
    for ab in absences:
        h = hh_of[idx[ab.student]]
        if rng.random() >= hh_reply[h]:
            continue
        delay = int(rng.choice(len(delays), p=delay_p))
        style = hh_style[h] if rng.random() < 0.7 else style_keys[int(rng.choice(len(style_keys), p=style_p))]
        lang = hh_lang[h]
        text = make_reply(ab.true_barrier, lang, rng, style)
        observed = classifier(text) if classifier is not None else _noisy_label(rng, ab.true_barrier,
                                                                               cfg.label_accuracy)
        replies.append(Reply(absence_id=ab.id, received_day=ab.day + delay, language=lang, text=text,
                             true_label=ab.true_barrier, observed_label=observed))

    run = SimRun(config=cfg, school=school, n_days=T, absences=absences, replies=replies, events=events)
    run.meta = {eid[k]: meta[k] for k in kept}  # debug info (route, seeds)
    return run


# ----------------------------------------------------------------------------------------------
# summaries


def summary(run: SimRun) -> dict:
    """JSON-friendly summary stats of a run."""
    T = run.n_days
    n = len(run.school.students)
    per_day = [0] * T
    for a in run.absences:
        per_day[a.day] += 1
    by_id = {a.id: a for a in run.absences}
    background = [a for a in run.absences if a.event_id is None]
    delays = [r.received_day - by_id[r.absence_id].day for r in run.replies]
    meta = getattr(run, "meta", {})
    return {
        "config": run.config.name,
        "seed": run.config.seed,
        "school": run.school.name,
        "n_students": n,
        "n_days": T,
        "n_absences": len(run.absences),
        "mean_daily_rate": len(run.absences) / max(1, n * T),
        "background_rate": len(background) / max(1, n * T),
        "absences_per_day": per_day,
        "per_barrier": dict(Counter(a.true_barrier for a in run.absences).most_common()),
        "per_barrier_background": dict(Counter(a.true_barrier for a in background).most_common()),
        "n_replies": len(run.replies),
        "reply_rate": len(run.replies) / max(1, len(run.absences)),
        "mean_delay": float(np.mean(delays)) if delays else 0.0,
        "languages": dict(Counter(r.language for r in run.replies).most_common()),
        "observed_label_accuracy": (float(np.mean([r.observed_label == r.true_label for r in run.replies]))
                                    if run.replies else None),
        "events": [
            {"id": e.id, "kind": e.kind, "barrier": e.barrier, "size": len(e.members),
             "absence_days": sum(1 for a in run.absences if a.event_id == e.id),
             "onset": e.onset, "duration": e.duration, "should_alarm": e.should_alarm,
             **{k: v for k, v in meta.get(e.id, {}).get("info", {}).items()}}
            for e in run.events
        ],
    }


def as_dict(run: SimRun) -> dict:
    """Full JSON-friendly dump of a run (config, summary, events, absences, replies)."""
    return {
        "config": dataclasses.asdict(run.config),
        "summary": summary(run),
        "events": [{**dataclasses.asdict(e), "members": sorted(e.members)} for e in run.events],
        "absences": [dataclasses.asdict(a) for a in run.absences],
        "replies": [dataclasses.asdict(r) for r in run.replies],
    }
