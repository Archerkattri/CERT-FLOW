"""Differential exactness tests for the certified Contraction Hierarchy.

EXACTNESS is non-negotiable for CH, so these tests are pure differential checks
against `FastDijkstra` (the reference engine), covering:

* (a) synthetic graphs: 20 graphs x 200 random pairs, both cost-only and
  shortcut-unpacked paths, on random directed graphs and random-weight grids;
* (b) DIMACS NY: a contracted NY subgraph (the full-NY Python-side ordering is
  benchmarked in scripts/run_ch.py, not here, to keep the suite fast) x 200
  random pairs if the NY file is present;
* the bounded-change `CHPotentialOracle`: a CH built on 0.8x lower-bound costs
  used as an admissible A* heuristic stays EXACT under a fresh +-20% cost
  perturbation with no rebuild.

A failing path is verified by recomputing its cost from the original CSR (using
the minimum parallel-edge cost, since random digraphs may have parallel arcs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from certflow.fastgraph import FastDijkstra
from certflow.ch import ContractionHierarchy, CHPotentialOracle
from certflow.roadnet import _csr_from_arcs, load_dimacs

DATA = Path(__file__).resolve().parents[1] / "data" / "dimacs"
NY = DATA / "USA-road-d.NY.gr"


# --------------------------------------------------------------------------- #
# Graph builders
# --------------------------------------------------------------------------- #


def _grid(rows: int, cols: int, seed: int):
    rng = np.random.default_rng(seed)
    tails, heads, weights = [], [], []

    def nid(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    tails.append(nid(r, c))
                    heads.append(nid(nr, nc))
                    weights.append(float(rng.integers(1, 100)))
    return _csr_from_arcs(
        rows * cols,
        np.array(tails, dtype=np.int64),
        np.array(heads, dtype=np.int64),
        np.array(weights, dtype=np.float64),
    )


def _random_digraph(n: int, m: int, seed: int):
    rng = np.random.default_rng(seed)
    tails = rng.integers(0, n, size=m).astype(np.int64)
    heads = rng.integers(0, n, size=m).astype(np.int64)
    weights = rng.uniform(1.0, 100.0, size=m).astype(np.float64)
    return _csr_from_arcs(n, tails, heads, weights)


def _edge_min_cost(g, a: int, b: int) -> float:
    """Minimum cost among parallel edges a->b (or inf if none)."""
    best = np.inf
    for k in range(int(g.indptr[a]), int(g.indptr[a + 1])):
        if int(g.indices[k]) == b and float(g.cost[k]) < best:
            best = float(g.cost[k])
    return best


def _path_cost(g, path) -> float:
    return sum(_edge_min_cost(g, a, b) for a, b in zip(path[:-1], path[1:]))


# --------------------------------------------------------------------------- #
# (a) synthetic differential exactness: 20 graphs x 200 pairs
# --------------------------------------------------------------------------- #


def test_ch_exact_synthetic_20x200():
    cost_mism = 0
    path_mism = 0
    for gi in range(20):
        if gi < 12:
            g = _random_digraph(80, 320, seed=gi)
        else:
            g = _grid(8, 8, seed=gi)
        ch = ContractionHierarchy(g)
        ch.build()
        ch.warmup()
        rng = np.random.default_rng(1000 + gi)
        for _ in range(200):
            s = int(rng.integers(0, g.n))
            t = int(rng.integers(0, g.n))
            _, dij = FastDijkstra(g, s).shortest_path(t)
            q = ch.query(s, t)
            if not (abs(q - dij) < 1e-6 or (np.isinf(dij) and np.isinf(q))):
                cost_mism += 1
            p = ch.path(s, t)
            if np.isinf(dij):
                if p is not None:
                    path_mism += 1
            else:
                if p is None or abs(_path_cost(g, p) - dij) > 1e-6:
                    path_mism += 1
    assert cost_mism == 0, f"{cost_mism} cost mismatches"
    assert path_mism == 0, f"{path_mism} path mismatches"


def test_ch_self_query_zero():
    g = _grid(6, 6, seed=3)
    ch = ContractionHierarchy(g)
    ch.build()
    assert ch.query(5, 5) == 0.0
    assert ch.path(5, 5) == [5]


def test_ch_unreachable():
    # two disjoint components: 0..3 and 4..7
    tails = np.array([0, 1, 2, 4, 5, 6], dtype=np.int64)
    heads = np.array([1, 2, 3, 5, 6, 7], dtype=np.int64)
    w = np.ones(6, dtype=np.float64)
    g = _csr_from_arcs(8, tails, heads, w)
    ch = ContractionHierarchy(g)
    ch.build()
    ch.warmup()
    assert np.isinf(ch.query(0, 7))
    assert ch.path(0, 7) is None
    assert ch.query(0, 3) == 3.0


# --------------------------------------------------------------------------- #
# Error / degenerate paths (audit v2, check 4)
# --------------------------------------------------------------------------- #


def _tiny_chain(n: int):
    """A simple directed chain 0->1->...->(n-1), unit costs."""
    if n <= 1:
        return _csr_from_arcs(n, np.empty(0, np.int64), np.empty(0, np.int64),
                              np.empty(0, np.float64))
    tails = np.arange(n - 1, dtype=np.int64)
    heads = np.arange(1, n, dtype=np.int64)
    w = np.ones(n - 1, dtype=np.float64)
    return _csr_from_arcs(n, tails, heads, w)


def test_ch_query_before_build_raises():
    ch = ContractionHierarchy(_tiny_chain(4))
    with pytest.raises(RuntimeError, match="before build"):
        ch.query(0, 3)
    with pytest.raises(RuntimeError, match="before build"):
        ch.path(0, 3)


def test_ch_potential_query_before_build_raises():
    orc = CHPotentialOracle(_tiny_chain(4))
    with pytest.raises(RuntimeError, match="before build"):
        orc.query(0, 3)


def test_ch_single_node():
    g = _csr_from_arcs(1, np.empty(0, np.int64), np.empty(0, np.int64),
                       np.empty(0, np.float64))
    ch = ContractionHierarchy(g)
    ch.build()
    assert ch.query(0, 0) == 0.0
    assert ch.path(0, 0) == [0]


def test_ch_empty_graph():
    g = _csr_from_arcs(0, np.empty(0, np.int64), np.empty(0, np.int64),
                       np.empty(0, np.float64))
    ch = ContractionHierarchy(g)
    ch.build()        # must not raise
    ch.warmup()       # n < 2 -> no-op, must not raise
    assert ch.n == 0


def test_ch_repeated_build_is_stable():
    """Re-building on different cost arrays yields a hierarchy exact for the
    most recent costs (no stale-shortcut leakage across builds)."""
    g = _grid(8, 8, seed=5)
    ch = ContractionHierarchy(g)
    ch.build()
    ch.warmup()
    rng = np.random.default_rng(3)
    cost2 = g.cost * rng.uniform(0.5, 2.0, size=g.cost.size)
    ch.build(cost2)  # rebuild on new costs
    # exact on cost2: differential vs a FastDijkstra over a graph carrying cost2
    g2 = _csr_from_arcs(
        g.n,
        np.repeat(np.arange(g.n), np.diff(g.indptr.astype(np.int64))),
        np.asarray(g.indices, dtype=np.int64),
        cost2,
    )
    for _ in range(60):
        s = int(rng.integers(0, g.n))
        t = int(rng.integers(0, g.n))
        _, dij = FastDijkstra(g2, s).shortest_path(t)
        q = ch.query(s, t)
        assert abs(q - dij) < 1e-6 or (np.isinf(dij) and np.isinf(q)), (s, t, q, dij)


def test_ch_potential_build_rebuild():
    g = _grid(8, 8, seed=6)
    orc = CHPotentialOracle(g, lower_bound_factor=0.8)
    orc.build()
    orc.build()  # idempotent rebuild must not raise or corrupt
    rng = np.random.default_rng(9)
    for _ in range(60):
        s = int(rng.integers(0, g.n))
        t = int(rng.integers(0, g.n))
        _, dij = FastDijkstra(g, s).shortest_path(t)
        q = orc.query(s, t)
        assert abs(q - dij) < 1e-6 or (np.isinf(dij) and np.isinf(q)), (s, t, q, dij)


# --------------------------------------------------------------------------- #
# bounded-change CHPotentialOracle: admissible under +-20% with no rebuild
# --------------------------------------------------------------------------- #


def test_ch_potential_oracle_exact_and_bounded_change():
    g = _grid(12, 12, seed=1)
    orig = g.cost.copy()
    orc = CHPotentialOracle(g, lower_bound_factor=0.8)
    orc.build()
    orc.warmup()

    rng = np.random.default_rng(7)
    # exact on true (build-time) costs
    for _ in range(150):
        s = int(rng.integers(0, g.n))
        t = int(rng.integers(0, g.n))
        _, dij = FastDijkstra(g, s).shortest_path(t)
        q = orc.query(s, t)
        assert abs(q - dij) < 1e-6 or (np.isinf(dij) and np.isinf(q)), (s, t, q, dij)

    # +-20% perturbation -> every edge stays >= 0.8x original -> still exact,
    # NO rebuild of the hierarchy.
    factors = rng.uniform(0.8, 1.2, size=g.cost.size)
    g.cost[:] = orig * factors
    for _ in range(150):
        s = int(rng.integers(0, g.n))
        t = int(rng.integers(0, g.n))
        _, dij = FastDijkstra(g, s).shortest_path(t)
        q = orc.query(s, t)
        assert abs(q - dij) < 1e-6 or (np.isinf(dij) and np.isinf(q)), (s, t, q, dij)


# --------------------------------------------------------------------------- #
# (b) real-data: contracted NY subgraph differential check
# --------------------------------------------------------------------------- #


def _ny_subgraph(g, n_sub: int):
    """Induced subgraph on nodes [0, n_sub): keep arcs with both ends < n_sub."""
    indptr = np.asarray(g.indptr)
    indices = np.asarray(g.indices)
    cost = np.asarray(g.cost)
    tails, heads, weights = [], [], []
    for u in range(n_sub):
        for k in range(int(indptr[u]), int(indptr[u + 1])):
            v = int(indices[k])
            if v < n_sub:
                tails.append(u)
                heads.append(v)
                weights.append(float(cost[k]))
    return _csr_from_arcs(
        n_sub,
        np.array(tails, dtype=np.int64),
        np.array(heads, dtype=np.int64),
        np.array(weights, dtype=np.float64),
    )


@pytest.mark.skipif(not NY.exists(), reason="DIMACS NY graph not downloaded")
def test_ch_exact_on_ny_subgraph():
    g_full = load_dimacs(NY)
    sub = _ny_subgraph(g_full, 8000)
    ch = ContractionHierarchy(sub)
    ch.build()
    ch.warmup()
    rng = np.random.default_rng(123)
    mism = 0
    pmism = 0
    for _ in range(200):
        s = int(rng.integers(0, sub.n))
        t = int(rng.integers(0, sub.n))
        _, dij = FastDijkstra(sub, s).shortest_path(t)
        q = ch.query(s, t)
        if not (abs(q - dij) < 1e-6 or (np.isinf(dij) and np.isinf(q))):
            mism += 1
        p = ch.path(s, t)
        if np.isinf(dij):
            if p is not None:
                pmism += 1
        else:
            if p is None or abs(_path_cost(sub, p) - dij) > 1e-6:
                pmism += 1
    assert mism == 0, f"{mism} NY-subgraph cost mismatches"
    assert pmism == 0, f"{pmism} NY-subgraph path mismatches"
