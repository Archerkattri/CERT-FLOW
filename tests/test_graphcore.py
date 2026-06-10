"""Tests for certflow.graphcore: D* Lite vs Dijkstra differential checks."""
from __future__ import annotations

import random

import pytest

from certflow.graphcore import DStarLite, dijkstra
from certflow.types import Node

INF = float("inf")


def _random_graph(rng: random.Random, n: int, p: float) -> dict[Node, dict[Node, float]]:
    """Random directed graph on nodes 0..n-1 with positive costs."""
    graph: dict[Node, dict[Node, float]] = {i: {} for i in range(n)}
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                graph[u][v] = round(rng.uniform(0.1, 10.0), 3)
    return graph


def _path_cost(graph: dict[Node, dict[Node, float]], path: list[Node]) -> float:
    return sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))


def _assert_agree(dsl: DStarLite, graph, start, goal) -> None:
    d_path, d_cost = dsl.shortest_path()
    ref_path, ref_cost = dijkstra(graph, start, goal)
    if ref_cost == INF:
        assert d_cost == INF and d_path is None
        return
    assert d_cost == pytest.approx(ref_cost, abs=1e-6), f"cost {d_cost} != {ref_cost}"
    assert d_path is not None
    assert d_path[0] == start and d_path[-1] == goal
    summed = _path_cost(graph, d_path)
    assert summed == pytest.approx(d_cost, abs=1e-6), f"path sum {summed} != {d_cost}"


def test_differential_random_batches():
    rng = random.Random(20260609)
    for trial in range(60):
        n = rng.randint(10, 200)
        p = rng.uniform(0.02, 0.12)
        graph = _random_graph(rng, n, p)
        start = rng.randrange(n)
        goal = rng.randrange(n)
        dsl = DStarLite(graph, start, goal)
        _assert_agree(dsl, graph, start, goal)
        edges = [(u, v) for u in graph for v in graph[u]]
        if not edges:
            continue
        for _ in range(22):
            k = rng.randint(1, max(1, len(edges) // 5))
            batch = {}
            for (u, v) in rng.sample(edges, min(k, len(edges))):
                w = round(rng.uniform(0.1, 12.0), 3)
                batch[(u, v)] = w
                graph[u][v] = w
            dsl.update_edges(batch)
            _assert_agree(dsl, graph, start, goal)


def test_moving_start():
    rng = random.Random(424242)
    for trial in range(25):
        n = rng.randint(10, 120)
        graph = _random_graph(rng, n, rng.uniform(0.05, 0.15))
        start = rng.randrange(n)
        goal = rng.randrange(n)
        dsl = DStarLite(graph, start, goal)
        edges = [(u, v) for u in graph for v in graph[u]]
        cur = start
        for _ in range(15):
            path, cost = dsl.shortest_path()
            ref_path, ref_cost = dijkstra(graph, cur, goal)
            assert cost == pytest.approx(ref_cost, abs=1e-6)
            if path is None or len(path) < 2 or cur == goal:
                break
            # Step the agent one node along the current path.
            cur = path[1]
            dsl.set_start(cur)
            # Apply a random cost update between moves.
            if edges:
                (u, v) = rng.choice(edges)
                w = round(rng.uniform(0.1, 12.0), 3)
                graph[u][v] = w
                dsl.update_edges({(u, v): w})
            _assert_agree(dsl, graph, cur, goal)


def test_unreachable_goal():
    graph = {0: {1: 1.0}, 1: {0: 1.0}, 2: {3: 1.0}, 3: {}}
    dsl = DStarLite(graph, 0, 3)
    path, cost = dsl.shortest_path()
    assert path is None and cost == INF
    ref_path, ref_cost = dijkstra(graph, 0, 3)
    assert ref_path is None and ref_cost == INF


def test_single_node():
    graph = {0: {}}
    dsl = DStarLite(graph, 0, 0)
    path, cost = dsl.shortest_path()
    assert path == [0] and cost == 0.0
    assert dijkstra(graph, 0, 0) == ([0], 0.0)


def test_nonpositive_cost_raises():
    graph = {0: {1: 1.0}, 1: {}}
    dsl = DStarLite(graph, 0, 1)
    with pytest.raises(ValueError):
        dsl.update_edges({(0, 1): 0.0})
    with pytest.raises(ValueError):
        dsl.update_edges({(0, 1): -5.0})


def test_cannot_add_new_edge():
    graph = {0: {1: 1.0}, 1: {}}
    dsl = DStarLite(graph, 0, 1)
    with pytest.raises(ValueError):
        dsl.update_edges({(1, 0): 2.0})  # edge absent from initial graph


def test_locality_fewer_pops():
    """A far-away update should expand fewer vertices than a from-scratch run."""
    rng = random.Random(7)
    # Long chain so an update near the goal end is "far" from start repair.
    n = 300
    graph: dict[Node, dict[Node, float]] = {i: {} for i in range(n)}
    for i in range(n - 1):
        graph[i][i + 1] = round(rng.uniform(1.0, 2.0), 3)
        # sparse shortcuts to make it a real graph, not a pure line
        if i + 5 < n:
            graph[i][i + 5] = round(rng.uniform(4.0, 6.0), 3)
    start, goal = 0, n - 1
    dsl = DStarLite(graph, start, goal)
    dsl.shortest_path()  # initial solve

    # From-scratch baseline: fresh instance solving the same query.
    fresh = DStarLite(graph, start, goal)
    fresh.shortest_path()
    scratch_pops = fresh.pops

    # A single far-away edge update (near the goal) then incremental repair.
    e = (goal - 1, goal)
    graph[goal - 1][goal] = graph[goal - 1][goal] + 0.5
    dsl.update_edges({e: graph[goal - 1][goal]})
    dsl.shortest_path()
    incremental_pops = dsl.pops

    assert incremental_pops < scratch_pops, (
        f"incremental {incremental_pops} not < scratch {scratch_pops}"
    )
