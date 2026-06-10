"""Tests for the DIMACS road-network loader and the ALT accelerator.

These tests use a tiny synthetic DIMACS graph for the parser/CSR contract and a
small hand-built road graph for ALT exactness, so they run without the large
downloaded data. If the downloaded DIMACS files are present, an extra (marked)
exactness check runs against `FastDijkstra` on real NY pairs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from certflow.fastgraph import FastDijkstra
from certflow.roadnet import ALT, RoadGraph, _csr_from_arcs, load_dimacs

DATA = Path(__file__).resolve().parents[1] / "data" / "dimacs"
NY = DATA / "USA-road-d.NY.gr"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# A tiny DIMACS .gr file: 4 nodes, a directed cycle 1->2->3->4 plus 1->4 long.
_TINY = """c tiny test graph
c comment
p sp 4 5
a 1 2 1
a 2 3 1
a 3 4 1
a 1 4 10
a 4 1 1
"""


def _write_tiny(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.gr"
    p.write_text(_TINY)
    return p


def _grid_roadgraph(rows: int, cols: int, seed: int = 0) -> RoadGraph:
    """Build a 4-connected directed grid as a RoadGraph (via arc arrays)."""
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


# --------------------------------------------------------------------------- #
# Parser / CSR contract
# --------------------------------------------------------------------------- #


def test_load_dimacs_tiny(tmp_path):
    g = load_dimacs(_write_tiny(tmp_path))
    assert g.n == 4
    assert g.indices.size == 5
    # node 0 (DIMACS 1) has out-edges to node1 (cost1) and node3 (cost10)
    s, e = int(g.indptr[0]), int(g.indptr[1])
    succ = dict(zip(g.indices[s:e].tolist(), g.cost[s:e].tolist()))
    assert succ == {1: 1.0, 3: 10.0}


def test_reverse_csr_consistency(tmp_path):
    g = load_dimacs(_write_tiny(tmp_path))
    # For every reverse slot of head v, cost[r_to_fwd] must equal the forward
    # edge cost of (pred -> v), and the forward edge must exist.
    for v in range(g.n):
        rs, re = int(g.r_indptr[v]), int(g.r_indptr[v + 1])
        for t in range(rs, re):
            p = int(g.r_indices[t])
            c = float(g.cost[int(g.r_to_fwd[t])])
            fs, fe = int(g.indptr[p]), int(g.indptr[p + 1])
            fwd = dict(zip(g.indices[fs:fe].tolist(), g.cost[fs:fe].tolist()))
            assert v in fwd and fwd[v] == c


def test_dijkstra_on_roadgraph(tmp_path):
    g = load_dimacs(_write_tiny(tmp_path))
    # 0->3: direct edge is 10, but 0->1->2->3 is 3 (cheaper).
    path, cost = FastDijkstra(g, 0).shortest_path(3)
    assert cost == 3.0
    assert path == [0, 1, 2, 3]


def test_roadgraph_flatgraph_attribute_contract():
    """RoadGraph duck-types FlatGraph: it must expose every attribute that
    `FastDijkstra` (and its numba kernel) reads, with matching kinds.

    `FastDijkstra.shortest_path` reads `n, indptr, indices, cost, node_of`; the
    reverse CSR (`r_indptr, r_indices, r_to_fwd`) and `index_of` round out the
    surface `FastDStarLite` would touch. This test pins the contract so a future
    change to either class can't silently break the duck-typing. (`slot_of` /
    `set_cost` are FlatGraph-only: RoadGraph is a stand-in for the full-SSSP /
    ALT-preprocessing path, not for D* Lite's incremental cost updates.)"""
    from certflow.fastgraph import FlatGraph

    # what FastDijkstra and its kernel actually read off the graph object
    fastdijkstra_surface = {"n", "indptr", "indices", "cost", "node_of"}
    # the broader CSR surface used across the engines
    full_surface = fastdijkstra_surface | {
        "index_of", "r_indptr", "r_indices", "r_to_fwd"
    }

    rg = _grid_roadgraph(5, 5, seed=0)
    # reference: an equivalent FlatGraph over the same adjacency
    adj: dict = {i: {} for i in range(rg.n)}
    for u in range(rg.n):
        for k in range(int(rg.indptr[u]), int(rg.indptr[u + 1])):
            adj[u][int(rg.indices[k])] = float(rg.cost[k])
    fg = FlatGraph(adj)

    for attr in full_surface:
        assert hasattr(rg, attr), f"RoadGraph missing FlatGraph attr {attr!r}"
        assert hasattr(fg, attr), f"FlatGraph missing attr {attr!r}"

    # node_of is callable on both; index_of supports __getitem__/__contains__
    assert rg.node_of(3) == 3
    assert rg.index_of[3] == 3 and 3 in rg.index_of
    assert callable(fg.node_of)

    # CSR arrays must have the consistent shapes the kernels assume
    assert rg.indptr.shape[0] == rg.n + 1
    assert rg.r_indptr.shape[0] == rg.n + 1
    assert rg.indices.shape[0] == rg.cost.shape[0]
    assert rg.r_indices.shape[0] == rg.r_to_fwd.shape[0] == rg.indices.shape[0]

    # functional contract: FastDijkstra on RoadGraph == on the FlatGraph twin
    rng = np.random.default_rng(0)
    for _ in range(30):
        s = int(rng.integers(0, rg.n))
        t = int(rng.integers(0, rg.n))
        _, c_rg = FastDijkstra(rg, s).shortest_path(t)
        _, c_fg = FastDijkstra(fg, s).shortest_path(t)
        assert (c_rg == c_fg) or (np.isinf(c_rg) and np.isinf(c_fg)), (s, t)


def test_identity_map():
    g = _grid_roadgraph(3, 3)
    assert g.index_of[5] == 5
    assert 5 in g.index_of
    assert 100 not in g.index_of
    assert g.index_of.get(100, -1) == -1
    assert g.node_of(7) == 7


# --------------------------------------------------------------------------- #
# ALT exactness on a synthetic grid (no large data needed)
# --------------------------------------------------------------------------- #


def test_alt_exact_on_grid():
    g = _grid_roadgraph(15, 15, seed=1)
    alt = ALT(g, n_landmarks=6, lower_bound_factor=0.8, seed=0)
    alt.warmup()
    rng = np.random.default_rng(7)
    for _ in range(40):
        s = int(rng.integers(0, g.n))
        t = int(rng.integers(0, g.n))
        _, dij = FastDijkstra(g, s).shortest_path(t)
        a = alt.query(s, t)
        if np.isinf(dij):
            assert np.isinf(a)
        else:
            assert abs(a - dij) < 1e-6, (s, t, a, dij)


def test_alt_admissible_under_bounded_perturbation():
    """Landmarks built on 0.8x costs stay exact after a +-20% perturbation."""
    g = _grid_roadgraph(15, 15, seed=2)
    alt = ALT(g, n_landmarks=6, lower_bound_factor=0.8, seed=0)
    alt.warmup()
    # Perturb costs within +-20% (so they stay >= 0.8x originals).
    rng = np.random.default_rng(3)
    factors = rng.uniform(0.8, 1.2, size=g.cost.size)
    g.cost[:] = g.cost * factors  # in place; ALT shares the array
    for _ in range(40):
        s = int(rng.integers(0, g.n))
        t = int(rng.integers(0, g.n))
        _, dij = FastDijkstra(g, s).shortest_path(t)
        a = alt.query(s, t)
        if np.isinf(dij):
            assert np.isinf(a)
        else:
            assert abs(a - dij) < 1e-6, (s, t, a, dij)


def test_alt_self_query_zero():
    g = _grid_roadgraph(8, 8)
    alt = ALT(g, n_landmarks=4, seed=0)
    assert alt.query(5, 5) == 0.0


# --------------------------------------------------------------------------- #
# Real-data exactness (only if the DIMACS NY file is present)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not NY.exists(), reason="DIMACS NY graph not downloaded")
def test_alt_exact_on_ny_sample():
    g = load_dimacs(NY)
    alt = ALT(g, n_landmarks=16, lower_bound_factor=0.8, seed=0)
    alt.warmup()
    rng = np.random.default_rng(123)
    for _ in range(25):
        s = int(rng.integers(0, g.n))
        t = int(rng.integers(0, g.n))
        _, dij = FastDijkstra(g, s).shortest_path(t)
        a = alt.query(s, t)
        if np.isinf(dij):
            assert np.isinf(a)
        else:
            assert abs(a - dij) < 1e-6, (s, t, a, dij)
