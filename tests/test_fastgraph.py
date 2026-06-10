"""Differential tests for certflow.fastgraph: the flat-array engine must match
graphcore.DStarLite and the dijkstra reference EXACTLY (cost), and fast_metrics
must equal the EdgeBelief lower/upper loop bit-for-bit.

Both the pure-Python and (when available) numba D* Lite paths are exercised.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from certflow.fastgraph import (
    FastDijkstra,
    FastDStarLite,
    FlatGraph,
    _HAVE_NUMBA,
    fast_metrics,
)
from certflow.graphcore import DStarLite, dijkstra
from certflow.types import EdgeBelief, Node

INF = float("inf")

# Run every D* Lite differential test on both backends; numba param is skipped
# cleanly if numba is not installed.
_BACKENDS = [False] + ([True] if _HAVE_NUMBA else [])


def _random_graph(rng: random.Random, n: int, p: float) -> dict[Node, dict[Node, float]]:
    graph: dict[Node, dict[Node, float]] = {i: {} for i in range(n)}
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                graph[u][v] = round(rng.uniform(0.1, 10.0), 3)
    return graph


def _path_cost(graph, path) -> float:
    return sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))


def _assert_cost_agrees(fast: FastDStarLite, old: DStarLite, graph, start, goal):
    f_path, f_cost = fast.shortest_path()
    o_path, o_cost = old.shortest_path()
    ref_path, ref_cost = dijkstra(graph, start, goal)
    if ref_cost == INF:
        assert f_cost == INF and f_path is None
        assert o_cost == INF and o_path is None
        return
    # FastDStarLite cost == DStarLite cost == dijkstra cost EXACTLY.
    assert f_cost == o_cost, f"fast {f_cost} != old {o_cost}"
    assert f_cost == pytest.approx(ref_cost, abs=1e-9), f"fast {f_cost} != ref {ref_cost}"
    assert f_path is not None
    assert f_path[0] == start and f_path[-1] == goal
    summed = _path_cost(graph, f_path)
    assert summed == pytest.approx(f_cost, abs=1e-9)


@pytest.mark.parametrize("use_numba", _BACKENDS)
def test_differential_random_batches(use_numba):
    """40+ random graphs x 20 update batches, fast == old == dijkstra exactly."""
    rng = random.Random(20260610)
    for trial in range(45):
        n = rng.randint(10, 200)
        p = rng.uniform(0.02, 0.12)
        graph = _random_graph(rng, n, p)
        start = rng.randrange(n)
        goal = rng.randrange(n)
        flat = FlatGraph(graph, extra_nodes=(start, goal))
        fast = FastDStarLite(graph, start, goal, flat=flat, use_numba=use_numba)
        old = DStarLite(graph, start, goal)
        _assert_cost_agrees(fast, old, graph, start, goal)
        edges = [(u, v) for u in graph for v in graph[u]]
        if not edges:
            continue
        for _ in range(20):
            k = rng.randint(1, max(1, len(edges) // 5))
            batch = {}
            for (u, v) in rng.sample(edges, min(k, len(edges))):
                w = round(rng.uniform(0.1, 12.0), 3)
                batch[(u, v)] = w
                graph[u][v] = w
            fast.update_edges(batch)
            old.update_edges(batch)
            _assert_cost_agrees(fast, old, graph, start, goal)


@pytest.mark.parametrize("use_numba", _BACKENDS)
def test_moving_start(use_numba):
    """Random walk: move start along the path, update an edge each step,
    fast cost == old cost == dijkstra cost exactly throughout."""
    rng = random.Random(424243)
    for trial in range(25):
        n = rng.randint(10, 120)
        graph = _random_graph(rng, n, rng.uniform(0.05, 0.15))
        start = rng.randrange(n)
        goal = rng.randrange(n)
        flat = FlatGraph(graph, extra_nodes=(start, goal))
        fast = FastDStarLite(graph, start, goal, flat=flat, use_numba=use_numba)
        old = DStarLite(graph, start, goal)
        edges = [(u, v) for u in graph for v in graph[u]]
        cur = start
        for _ in range(15):
            f_path, f_cost = fast.shortest_path()
            o_path, o_cost = old.shortest_path()
            ref_path, ref_cost = dijkstra(graph, cur, goal)
            assert f_cost == o_cost
            assert f_cost == pytest.approx(ref_cost, abs=1e-9)
            if f_path is None or len(f_path) < 2 or cur == goal:
                break
            cur = f_path[1]
            fast.set_start(cur)
            old.set_start(cur)
            if edges:
                (u, v) = rng.choice(edges)
                w = round(rng.uniform(0.1, 12.0), 3)
                graph[u][v] = w
                fast.update_edges({(u, v): w})
                old.update_edges({(u, v): w})


@pytest.mark.parametrize("use_numba", _BACKENDS)
def test_unreachable_goal(use_numba):
    graph = {0: {1: 1.0}, 1: {0: 1.0}, 2: {3: 1.0}, 3: {}}
    fast = FastDStarLite(graph, 0, 3, use_numba=use_numba)
    path, cost = fast.shortest_path()
    assert path is None and cost == INF
    old = DStarLite(graph, 0, 3)
    assert old.shortest_path() == (None, INF)


@pytest.mark.parametrize("use_numba", _BACKENDS)
def test_single_node(use_numba):
    graph = {0: {}}
    fast = FastDStarLite(graph, 0, 0, use_numba=use_numba)
    path, cost = fast.shortest_path()
    assert path == [0] and cost == 0.0
    assert dijkstra(graph, 0, 0) == ([0], 0.0)


@pytest.mark.parametrize("use_numba", _BACKENDS)
def test_start_equals_goal_with_edges(use_numba):
    graph = {0: {1: 1.0}, 1: {0: 2.0}}
    fast = FastDStarLite(graph, 0, 0, use_numba=use_numba)
    assert fast.shortest_path() == ([0], 0.0)


@pytest.mark.parametrize("use_numba", _BACKENDS)
def test_nonpositive_cost_raises(use_numba):
    graph = {0: {1: 1.0}, 1: {}}
    fast = FastDStarLite(graph, 0, 1, use_numba=use_numba)
    with pytest.raises(ValueError):
        fast.update_edges({(0, 1): 0.0})
    with pytest.raises(ValueError):
        fast.update_edges({(0, 1): -5.0})


@pytest.mark.parametrize("use_numba", _BACKENDS)
def test_cannot_add_new_edge(use_numba):
    graph = {0: {1: 1.0}, 1: {}}
    fast = FastDStarLite(graph, 0, 1, use_numba=use_numba)
    with pytest.raises(ValueError):
        fast.update_edges({(1, 0): 2.0})


@pytest.mark.parametrize("use_numba", _BACKENDS)
def test_bad_batch_leaves_state_untouched(use_numba):
    """A ValueError in update_edges must not have mutated any cost."""
    graph = {0: {1: 1.0, 2: 3.0}, 1: {2: 1.0}, 2: {}}
    fast = FastDStarLite(graph, 0, 2, use_numba=use_numba)
    _, c0 = fast.shortest_path()
    with pytest.raises(ValueError):
        fast.update_edges({(0, 1): 5.0, (9, 9): 1.0})  # second edge absent
    _, c1 = fast.shortest_path()
    assert c0 == c1  # batch rejected wholesale; cost unchanged


def test_pops_counter_is_local():
    """Far-away update expands fewer vertices than a from-scratch run
    (locality property, same as graphcore)."""
    rng = random.Random(7)
    n = 300
    graph = {i: {} for i in range(n)}
    for i in range(n - 1):
        graph[i][i + 1] = round(rng.uniform(1.0, 2.0), 3)
        if i + 5 < n:
            graph[i][i + 5] = round(rng.uniform(4.0, 6.0), 3)
    start, goal = 0, n - 1
    fast = FastDStarLite(graph, start, goal, use_numba=False)
    fast.shortest_path()
    fresh = FastDStarLite(graph, start, goal, use_numba=False)
    fresh.shortest_path()
    scratch_pops = fresh.pops
    graph[goal - 1][goal] += 0.5
    fast.update_edges({(goal - 1, goal): graph[goal - 1][goal]})
    fast.shortest_path()
    assert fast.pops < scratch_pops


# --------------------------------------------------------------------------- #
# FastDijkstra
# --------------------------------------------------------------------------- #


def test_fast_dijkstra_matches_reference():
    rng = random.Random(99)
    for _ in range(40):
        n = rng.randint(5, 150)
        graph = _random_graph(rng, n, rng.uniform(0.03, 0.15))
        start = rng.randrange(n)
        goal = rng.randrange(n)
        flat = FlatGraph(graph, extra_nodes=(start, goal))
        fd = FastDijkstra(flat, flat.index_of[start])
        path_idx, cost = fd.shortest_path(flat.index_of[goal])
        ref_path, ref_cost = dijkstra(graph, start, goal)
        if ref_cost == INF:
            assert cost == INF and path_idx is None
            continue
        assert cost == pytest.approx(ref_cost, abs=1e-9)
        # translate back and re-sum on the original graph
        nodes = [flat.node_of(i) for i in path_idx]
        assert nodes[0] == start and nodes[-1] == goal
        assert _path_cost(graph, nodes) == pytest.approx(cost, abs=1e-9)


def test_fast_dijkstra_source_equals_target():
    graph = {0: {1: 1.0}, 1: {}}
    flat = FlatGraph(graph)
    fd = FastDijkstra(flat, flat.index_of[0])
    assert fd.shortest_path(flat.index_of[0]) == ([flat.index_of[0]], 0.0)


# --------------------------------------------------------------------------- #
# fast_metrics vs EdgeBelief loop (bit-for-bit)
# --------------------------------------------------------------------------- #


def test_fast_metrics_matches_edgebelief_loop():
    rng = random.Random(31415)
    for _ in range(50):
        m = rng.randint(1, 500)
        t = rng.uniform(0.0, 100.0)
        q = rng.uniform(0.0, 5.0)
        cost_floor = rng.choice([1e-6, 1e-3, 0.0])
        c_hat = np.array([rng.uniform(-2.0, 20.0) for _ in range(m)])
        t_obs = np.array([rng.uniform(-50.0, t) for _ in range(m)])
        rho = np.array([rng.uniform(0.0, 0.2) for _ in range(m)])
        lo, up = fast_metrics(c_hat, t_obs, rho, t, q, cost_floor)
        for i in range(m):
            b = EdgeBelief(
                c_hat=float(c_hat[i]),
                t_obs=float(t_obs[i]),
                rho=float(rho[i]),
                sense_cost=0.1,
            )
            assert lo[i] == b.lower(t, q, cost_floor)
            assert up[i] == b.upper(t, q, cost_floor)


def test_fast_metrics_infinite_q():
    """Warm-up q=inf: lo pinned to floor, up infinite — matches the scalar."""
    c_hat = np.array([1.0, 5.0, 0.5])
    t_obs = np.array([0.0, 0.0, 0.0])
    rho = np.array([0.02, 0.0, 0.1])
    lo, up = fast_metrics(c_hat, t_obs, rho, t=10.0, q=INF, cost_floor=1e-3)
    for i in range(3):
        b = EdgeBelief(c_hat=float(c_hat[i]), t_obs=0.0, rho=float(rho[i]), sense_cost=0.1)
        assert lo[i] == b.lower(10.0, INF, 1e-3)
        assert up[i] == b.upper(10.0, INF, 1e-3)


# --------------------------------------------------------------------------- #
# Benchmark smoke (the real benchmark is run via __main__; here we just verify
# it runs and produces a speedup figure on a small grid quickly).
# --------------------------------------------------------------------------- #


def test_benchmark_runs_small():
    from certflow.fastgraph import run_benchmark

    res = run_benchmark(rows=12, cols=12, rounds=10, seed=0)
    assert res["n_edges"] > 0
    assert res["old"][0] > 0 and res["fast"][0] > 0
