"""Tests for certflow.egraph.EGraphPlanner.

Verifies the *actual* guarantee of weighted A* with the E-graph heuristic
(bounded suboptimality: cost <= w * optimal) and that accumulated experience
measurably reduces node expansions on a repeated identical query.
"""
from __future__ import annotations

import numpy as np
import pytest

from certflow.egraph import EGraphPlanner, manhattan_heuristic
from certflow.graphcore import dijkstra


def _grid_graph(rows, cols, rng, lo=0.5, hi=3.0):
    graph = {}
    for r in range(rows):
        for c in range(cols):
            graph.setdefault((r, c), {})
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    graph[(r, c)][(nr, nc)] = float(rng.uniform(lo, hi))
    return graph


def test_bounded_suboptimality_20_random_queries():
    """EGraphPlanner returns cost <= 1.2 * optimal on 20 random queries."""
    rows = cols = 12
    rng = np.random.default_rng(0)
    graph = _grid_graph(rows, cols, rng)
    h = manhattan_heuristic(0.5)  # 0.5 = lower bound on every edge cost
    w = 1.2
    planner = EGraphPlanner(graph, heuristic=h, w=w)

    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    qrng = np.random.default_rng(123)
    checked = 0
    while checked < 20:
        s = nodes[int(qrng.integers(0, len(nodes)))]
        gl = nodes[int(qrng.integers(0, len(nodes)))]
        if s == gl:
            continue
        path, cost = planner.plan(s, gl)
        opt_path, opt_cost = dijkstra(graph, s, gl)
        assert path is not None and opt_path is not None
        assert path[0] == s and path[-1] == gl
        # Path cost recomputed on the graph must match reported cost.
        summed = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
        assert summed == pytest.approx(cost, abs=1e-9)
        # The actual guarantee: bounded suboptimality.
        assert cost <= w * opt_cost + 1e-9, (
            f"cost {cost} exceeds {w} * optimal {opt_cost}"
        )
        checked += 1


def test_experience_reduces_expansions_on_repeated_query():
    """Repeating an identical query reuses experience -> fewer expansions."""
    rows = cols = 20
    rng = np.random.default_rng(7)
    # Near-uniform edge costs so Manhattan distance is an *informative*
    # admissible heuristic. This is the regime where the E-graphs reuse
    # mechanism delivers its expansion savings: a tight base heuristic plus a
    # tight experience incumbent lets OPEN pruning fire from the first
    # expansion. On a near-uninformative heuristic (wide cost spread) reuse
    # helps little -- that honest boundary is reported in the comparison doc.
    graph = _grid_graph(rows, cols, rng, lo=0.9, hi=1.1)
    h = manhattan_heuristic(0.9)
    planner = EGraphPlanner(graph, heuristic=h, w=1.2)

    s, gl = (0, 0), (rows - 1, cols - 1)

    # First (cold) plan: no experience yet.
    p1, c1 = planner.plan(s, gl)
    cold_exp = planner.expansions
    assert p1 is not None

    # Second (warm) plan: identical query, experience now contains the path.
    p2, c2 = planner.plan(s, gl)
    warm_exp = planner.expansions
    assert p2 is not None
    assert c2 == pytest.approx(c1, abs=1e-9)

    assert warm_exp < cold_exp, (
        f"expected experience to reduce expansions, "
        f"cold={cold_exp} warm={warm_exp}"
    )


def test_unreachable_returns_none():
    """Disconnected goal -> (None, inf)."""
    graph = {(0, 0): {(0, 1): 1.0}, (0, 1): {(0, 0): 1.0}, (5, 5): {}}
    h = manhattan_heuristic(0.5)
    planner = EGraphPlanner(graph, heuristic=h, w=1.2)
    path, cost = planner.plan((0, 0), (5, 5))
    assert path is None
    assert cost == float("inf")
