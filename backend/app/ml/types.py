"""Shared data contracts for the PRD 2 pipeline (graph -> simulator -> detectors -> evaluation).

Every module in app/ml builds against these types. Change them only with care: the
graph loader, simulator, detectors and evaluation all depend on them.

Conventions
  * Student ids are re-keyed strings "N000", "N001", ... (never the original dataset ids).
  * Class sections are re-keyed "SEC-00", "SEC-01", ... (original class names dropped).
  * Days are integer school-day indices 0..n_days-1 (weekends are not represented).
  * Barrier values are the strings of app.models.Barrier (e.g. "illness", "transportation").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

import networkx as nx


@dataclass
class SchoolGraph:
    name: str                              # "lyon_primary_2009", "marseille_hs_2013", "sbm_<n>"
    students: List[str]                    # re-keyed ids, teachers excluded
    section: Dict[str, str]                # student -> re-keyed class section
    grade: Dict[str, int]                  # student -> grade level (int; derived from section)
    # One weighted undirected graph per recorded day. Edge attr "weight" = seconds of contact that day,
    # already reduced to each node's top-k edges by weight (union over endpoints).
    daily: List[nx.Graph]
    # Synthetic logistics layers, filled by graphs.assign_logistics(). None = walks/driven (no route).
    bus_route: Dict[str, Optional[str]] = field(default_factory=dict)
    household: Dict[str, str] = field(default_factory=dict)
    neighborhood: Dict[str, str] = field(default_factory=dict)

    def day_graph(self, day: int) -> nx.Graph:
        """Contact graph for simulated school day `day`, cycling the recorded days."""
        return self.daily[day % len(self.daily)]


@dataclass
class PlantedEvent:
    id: str                                # "E000", ...
    kind: str                              # "contagion" | "route_failure" | "caregiving_spike" | "flu_wave"
    barrier: str                           # Barrier value carried by its absences
    members: FrozenSet[str]                # students absent because of this event
    onset: int                             # first day of the event
    duration: int                          # days
    should_alarm: bool = True              # False for school-wide flu waves (must not raise alarms)


@dataclass
class AbsenceEvent:
    id: str                                # "A00000", ...
    student: str
    day: int
    true_barrier: str
    event_id: Optional[str] = None         # PlantedEvent.id, or None for background absences


@dataclass
class Reply:
    absence_id: str
    received_day: int                      # >= absence day; detectors only see replies with received_day <= as_of
    language: str                          # "en" | "es" | "vi" | ...
    text: str
    true_label: str                        # ground-truth barrier (what the family meant)
    observed_label: str                    # label the pipeline assigned (classifier output or noisy proxy)


@dataclass
class SimConfig:
    name: str = "A"
    n_days: int = 60                       # school days; >= 40 baseline days are needed by the scan
    seed: int = 0
    base_absence_rate: float = 0.03        # mean daily background absence probability
    chronic_shape: float = 0.6             # Gamma shape for per-student absence propensity (lower = more skew)
    n_contagion: int = 2
    beta: float = 0.0006                   # per-second transmission hazard on contact weight
    infectious_days: int = 2
    n_route_failures: int = 2
    route_absent_frac: tuple = (0.3, 0.8)  # share of riders absent when a route fails
    n_caregiving_spikes: int = 1           # untargeted event type (neighborhood daycare closure)
    n_flu_waves: int = 1                   # school-wide, must not alarm
    reply_rate: tuple = (0.4, 0.9)
    reply_delay_max: int = 3
    label_accuracy: float = 0.85           # used when no classifier is plugged in: P(observed == true)


@dataclass
class SimRun:
    config: SimConfig
    school: SchoolGraph
    n_days: int
    absences: List[AbsenceEvent]
    replies: List[Reply]
    events: List[PlantedEvent]


@dataclass
class Alarm:
    detector: str                          # "content_only" | "graph_only" | "combined"
    layer: str                             # "section" | "route" | "neighborhood" | "contact" | "school" | "grade"
    zone_id: str
    zone: FrozenSet[str]
    barrier: str                           # attributed cause ("unknown" allowed for graph-only before attribution)
    day: int                               # as_of day the alarm fired
    window: int                            # w in {1,2,3}
    observed: int
    expected: float
    llr: float
    p_value: float
    absent_members: FrozenSet[str] = frozenset()
    evidence_absence_ids: List[str] = field(default_factory=list)
