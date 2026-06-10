"""Evaluation oracle for CERT experiments.

Provides:
- ``opt(world, t, start, goal)``: ground-truth shortest path at time t.
- ``CoverageLog``: accumulates ``RoundLog`` records and computes empirical
  coverage with Clopper–Pearson confidence intervals and a full summary.
"""
from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist

from certflow.types import Edge, Node, RoundLog

# ---------------------------------------------------------------------------
# Dijkstra (local fallback; used if graphcore is unavailable)
# ---------------------------------------------------------------------------

def _dijkstra(
    graph: dict[Node, dict[Node, float]],
    start: Node,
    goal: Node,
) -> tuple[list[Node] | None, float]:
    """Standard Dijkstra on a weighted directed graph.

    Parameters
    ----------
    graph:
        Adjacency map ``{u: {v: cost, ...}, ...}`` with positive costs.
    start, goal:
        Source and destination nodes.

    Returns
    -------
    ``(path, cost)`` where *path* is a list of nodes from start to goal
    (inclusive) or ``None`` if the goal is unreachable, and *cost* is the
    path cost (``inf`` if unreachable).
    """
    if start == goal:
        return [start], 0.0

    dist: dict[Node, float] = {start: 0.0}
    prev: dict[Node, Node | None] = {start: None}
    heap: list[tuple[float, Any]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if u == goal:
            # Reconstruct path.
            path: list[Node] = []
            cur: Node | None = goal
            while cur is not None:
                path.append(cur)
                cur = prev.get(cur)
            path.reverse()
            return path, dist[goal]
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return None, math.inf


# ---------------------------------------------------------------------------
# opt()
# ---------------------------------------------------------------------------

def opt(
    world: Any,
    t: float,
    start: Node,
    goal: Node,
) -> tuple[list[Node] | None, float]:
    """Ground-truth shortest path at time *t*.

    Builds a snapshot graph from ``world.true_cost(e, t)`` over all edges,
    then runs Dijkstra.

    Tries ``from certflow.graphcore import dijkstra`` first; falls back to
    the local ``_dijkstra`` implementation if that import fails.

    Parameters
    ----------
    world:
        Any object satisfying the ``World`` protocol.
    t:
        Query time (continuous float seconds).
    start, goal:
        Source and destination nodes in ``world.graph``.

    Returns
    -------
    ``(path, cost)`` — path is ``None`` and cost is ``inf`` if unreachable.
    """
    # Build snapshot graph at time t.
    snapshot: dict[Node, dict[Node, float]] = {}
    for e in world.edges():
        u, v = e
        c = world.true_cost(e, t)
        snapshot.setdefault(u, {})[v] = c

    # By design this evaluation oracle stays on the dict-based graphcore
    # Dijkstra, NOT the planner's flat engine: opt() is the ground-truth
    # reference that coverage/regret are scored against, so it must be an
    # implementation that is INDEPENDENT of the system under test. Sharing
    # fastgraph here would let an engine bug hide itself (both sides agreeing
    # on the same wrong number). This is the only intentional graphcore use on
    # the experiment hot path; the planner itself is fully ported to fastgraph.
    try:
        from certflow.graphcore import dijkstra as _gc_dijkstra  # type: ignore[import]
        return _gc_dijkstra(snapshot, start, goal)
    except (ImportError, ModuleNotFoundError):
        return _dijkstra(snapshot, start, goal)


# ---------------------------------------------------------------------------
# CoverageLog
# ---------------------------------------------------------------------------

class CoverageLog:
    """Accumulates per-round ``RoundLog`` records and computes coverage stats.

    Example::

        log = CoverageLog()
        log.record(round_log)
        cov = log.empirical_coverage()
        lo, hi = log.coverage_ci()
        print(log.summary())
    """

    def __init__(self) -> None:
        self._records: list[RoundLog] = []

    # ------------------------------------------------------------------
    def record(self, round_log: RoundLog) -> None:
        """Append one ``RoundLog`` to the accumulator."""
        self._records.append(round_log)

    # ------------------------------------------------------------------
    def empirical_coverage(self) -> float:
        """Fraction of rounds where ``lb <= opt <= ub``."""
        if not self._records:
            return float("nan")
        return float(np.mean([r.covered for r in self._records]))

    # ------------------------------------------------------------------
    def coverage_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """Clopper–Pearson interval for the empirical coverage.

        Parameters
        ----------
        confidence:
            Nominal confidence level (default 0.95).

        Returns
        -------
        ``(lower, upper)`` as floats in [0, 1].
        """
        n = len(self._records)
        if n == 0:
            return (float("nan"), float("nan"))
        k = sum(r.covered for r in self._records)
        alpha = 1.0 - confidence
        # Lower bound: beta quantile at alpha/2 (0 when k=0).
        lower = float(beta_dist.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
        # Upper bound: beta quantile at 1 - alpha/2 (1 when k=n).
        upper = float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
        return (lower, upper)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a summary dictionary with coverage, CI, and cost metrics.

        Keys
        ----
        coverage, ci_lower, ci_upper, mean_gap, median_gap,
        certified_fraction, total_sense_spend,
        latency_p50, latency_p95.
        """
        if not self._records:
            return {}

        records = self._records
        n = len(records)

        coverage = self.empirical_coverage()
        ci_lower, ci_upper = self.coverage_ci()

        gaps = np.array([r.ub - r.lb for r in records])
        mean_gap = float(np.mean(gaps))
        median_gap = float(np.median(gaps))

        certified_fraction = float(np.mean([r.certified for r in records]))

        total_sense_spend = float(sum(r.sense_spend for r in records))

        latencies = np.array([r.replan_seconds for r in records])
        latency_p50 = float(np.percentile(latencies, 50))
        latency_p95 = float(np.percentile(latencies, 95))

        return {
            "n_rounds": n,
            "coverage": coverage,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean_gap": mean_gap,
            "median_gap": median_gap,
            "certified_fraction": certified_fraction,
            "total_sense_spend": total_sense_spend,
            "latency_p50": latency_p50,
            "latency_p95": latency_p95,
        }
