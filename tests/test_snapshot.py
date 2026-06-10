"""Certified snapshot oracle: certificate-gated O(1) queries."""
import math

import pytest

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.fastgraph import FlatGraph
from certflow.graphcore import dijkstra
from certflow.snapshot import SnapshotOracle


def planner_on(kind: str, seed: int = 2, rho: float = 0.02):
    w = grid_world(6, 6, seed=seed, kind=kind, noise_scale=0.02,
                   **({"rho": rho} if kind == "bounded" else {}))
    p = CertPlanner(w, (0, 0), (5, 5),
                    PlannerConfig(epsilon=5.0, alpha_prime=0.2, eps_tv=1e-4))
    return w, p


def test_snapshot_exact_on_static_world():
    w, p = planner_on("static")
    for _ in range(250):
        p.round()
    res = p.snapshot_query((0, 0), (5, 5), tau=0.5)
    assert res is not None
    # exact on the snapshot costs: equals dijkstra on c_hat
    snap = {u: {v: p.beliefs[(u, v)].c_hat for v in nbrs}
            for u, nbrs in w.graph.items()}
    _, ref = dijkstra(snap, (0, 0), (5, 5))
    assert abs(res["cost"] - ref) < 1e-6
    # certificate: true cost of returned path within the reported slack
    truth = sum(w.true_cost(e, p.t)
                for e in zip(res["path"][:-1], res["path"][1:]))
    assert abs(truth - res["cost"]) <= res["slack"] + 1e-9
    # arbitrary other endpoints answered from the same build
    res2 = p.snapshot_query((0, 5), (5, 0), tau=0.5)
    assert res2 is not None and math.isfinite(res2["cost"])


def test_snapshot_gate_closes_under_drift():
    w, p = planner_on("bounded", rho=0.05)
    for _ in range(120):
        p.round()
    # tight tau: fast drift must keep the gate closed
    assert p.snapshot_query((0, 0), (5, 5), tau=1e-3) is None


def test_snapshot_invalidates_when_map_moves():
    w, p = planner_on("static")
    for _ in range(250):
        p.round()
    assert p.snapshot_query((0, 0), (5, 5), tau=0.5) is not None
    assert p._oracle is not None and p._oracle.ready
    # simulate a large estimate shift on one edge -> snapshot must expire
    e = next(iter(p.beliefs))
    i = p._edge_idx[e]
    p._arr_chat[i] += 10.0
    p._beliefs_version += 1  # direct array mutation is test-only; the real
    # channel (ingest_observation) bumps the version itself
    assert p.snapshot_query((0, 0), (5, 5), tau=0.5) is None or (
        abs(p._oracle_chat_snap[i] - p._arr_chat[i]) < 0.5
    )


# --------------------------------------------------------------------------- #
# SnapshotOracle error / degenerate paths (audit v2, check 4)
# --------------------------------------------------------------------------- #


def test_snapshot_query_before_build_raises():
    fg = FlatGraph({0: {1: 1.0}, 1: {2: 1.0}, 2: {}})
    o = SnapshotOracle(fg)
    assert o.ready is False
    with pytest.raises(RuntimeError, match="before build"):
        o.cost(0, 2)
    with pytest.raises(RuntimeError, match="before build"):
        o.path(0, 2)


def test_snapshot_single_node():
    o = SnapshotOracle(FlatGraph({0: {}}))
    o.build(0.0)
    assert o.cost(0, 0) == 0.0
    assert o.path(0, 0) == [0]


def test_snapshot_unreachable_target():
    # 0 -> 1, node 2 isolated
    o = SnapshotOracle(FlatGraph({0: {1: 1.0}, 1: {}, 2: {}}))
    o.build(0.0)
    assert math.isinf(o.cost(0, 2))
    assert o.path(0, 2) is None
    assert o.cost(0, 1) == 1.0
    assert o.path(0, 1) == [0, 1]


def test_snapshot_invalidate_rebuild_cycle():
    o = SnapshotOracle(FlatGraph({0: {1: 1.0}, 1: {2: 1.0}, 2: {}}))
    o.build(0.0)
    assert o.ready and o.cost(0, 2) == 2.0
    o.invalidate()
    assert o.ready is False
    with pytest.raises(RuntimeError, match="before build"):
        o.cost(0, 2)
    o.build(1.0)  # rebuild after invalidate
    assert o.ready and o.cost(0, 2) == 2.0
    assert o.built_at == 1.0
