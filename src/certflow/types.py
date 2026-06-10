"""Shared types and interface contracts for CERT.

Spec: docs/specs/design-spec.md (section 6).
Theory: docs/theory/theory-notes.md.

Every module codes against these contracts; modules must not import each other's
internals. Nodes are arbitrary hashables (grid worlds use (row, col) tuples).
Edges are directed (u, v) tuples. Time is continuous (float seconds); each
replanning round advances time by the sensing period delta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterator, Protocol

Node = Hashable
Edge = tuple[Node, Node]


@dataclass
class EdgeBelief:
    """Planner-side belief about one edge. Intervals are derived, never stored."""

    c_hat: float          # point estimate from last observation
    t_obs: float          # time of last observation (-inf if never observed)
    rho: float            # assumed drift rate bound (A1), conservative
    sense_cost: float     # cost m_e of one observation of this edge
    observed: bool = True # False = c_hat is a prior, not a real observation;
                          # a score needs a real previous observation (pair)

    def age(self, t: float) -> float:
        return t - self.t_obs

    def lower(self, t: float, q: float, cost_floor: float = 1e-3) -> float:
        """ell_e(t) = c_hat - q - rho * age, clipped at the cost floor."""
        return max(cost_floor, self.c_hat - q - self.rho * self.age(t))

    def upper(self, t: float, q: float, cost_floor: float = 1e-3) -> float:
        """u_e(t) = c_hat + q + rho * age, clipped at the cost floor
        (a noisy observation can drive c_hat below zero; true costs cannot)."""
        return max(cost_floor, self.c_hat + q + self.rho * self.age(t))


@dataclass
class Certificate:
    """What CERT emits every round. confidence = 1 - alpha' - sum(Delta_stale)."""

    lb: float
    ub: float
    confidence: float          # <= 0 means INVALID (e.g. warm-up)
    path: list[Node]
    epsilon_attainable: bool   # T2' certifiability check result
    epsilon_floor: float       # smallest sustainable epsilon per T2'

    @property
    def gap(self) -> float:
        return self.ub - self.lb

    @property
    def valid(self) -> bool:
        return self.confidence > 0.0


class IncrementalSP(Protocol):
    """Contract for graphcore. One instance per metric (lower / upper / plan).

    Implementations must be incremental: after update_edges touching k edges,
    shortest_path() cost must scale with the locally affected region (T3),
    not with |E|. Costs must be positive. Unreachable goal -> (None, inf).
    """

    def update_edges(self, costs: dict[Edge, float]) -> None: ...

    def set_start(self, node: Node) -> None: ...

    def shortest_path(self) -> tuple[list[Node] | None, float]: ...


class World(Protocol):
    """Contract for drift: ground truth the planner never sees directly."""

    graph: dict[Node, dict[Node, float]]  # adjacency with INITIAL true costs

    def true_cost(self, e: Edge, t: float) -> float: ...

    def observe(self, e: Edge, t: float) -> float:
        """One noisy observation of c_e(t); draws from the world's RNG."""
        ...

    def edges(self) -> Iterator[Edge]: ...

    def rho_true(self, e: Edge) -> float:
        """Actual drift-rate bound for e (inf for off-model worlds)."""
        ...


@dataclass
class RoundLog:
    """One row per replanning round; harness aggregates these."""

    t: float
    lb: float
    ub: float
    confidence: float
    opt: float                 # oracle ground truth (nan outside simulation)
    covered: bool              # lb <= opt <= ub
    certified: bool            # gap <= epsilon and valid
    sensed_edge: Edge | None
    sense_spend: float
    replan_seconds: float      # wall-clock for the search repair this round


@dataclass
class EpisodeResult:
    rounds: list[RoundLog] = field(default_factory=list)
    travel_cost: float = 0.0
    sense_cost: float = 0.0
    reached_goal: bool = False
    oracle_cost: float = float("nan")  # clairvoyant walk cost (Tier-2 regret)
