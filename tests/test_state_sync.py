"""State-sync invariants for CertPlanner's parallel state (audit v2, check 1).

The planner maintains several views of the same belief/cost state in parallel
for speed:

* the canonical belief dicts ``beliefs[e].{c_hat,t_obs,rho,observed}``;
* belief ARRAYS ``_arr_chat/_arr_tobs/_arr_rho/_arr_obs`` (edge-ordered, for
  the vectorized refresh and the snapshot gate);
* the pre-widened metric cache dicts ``_cache_lo/_cache_up/_cache_due`` and the
  due ARRAY ``_arr_due``;
* the flat CSR cost arrays of the two D* Lite searches
  (``_flat_lo.cost`` / ``_flat_up.cost``).

Every mutation site (round, retarget, traversal/sensing ingest, snapshot query)
must keep these consistent. These tests run a mixed workload and assert the
cross-view equalities, which is the exact failure mode an isolated unit test
would miss (a divergence only shows after a particular interleaving).
"""
from __future__ import annotations

import math
import random

import numpy as np

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world


def _assert_dict_array_sync(p: CertPlanner) -> None:
    """beliefs dicts == belief arrays, edge for edge."""
    for e, i in p._edge_idx.items():
        b = p.beliefs[e]
        assert p._arr_chat[i] == b.c_hat, f"c_hat mismatch on {e}"
        assert p._arr_tobs[i] == b.t_obs, f"t_obs mismatch on {e}"
        assert p._arr_rho[i] == b.rho, f"rho mismatch on {e}"
        assert bool(p._arr_obs[i]) == bool(b.observed), f"observed mismatch on {e}"


def _assert_cache_flat_sync(p: CertPlanner) -> None:
    """_cache_lo/_cache_up == the costs actually loaded in the flat arrays.

    The flat cost arrays are what the D* Lite kernels read, so any drift
    between the cache and the flat costs is a soundness bug (the search would
    run on different costs than the cache the planner reasons about). Only
    checked when a cache exists (after the first refresh)."""
    if not p._cache_lo:
        return
    lo_flat = p._flat_lo.cost[p._slots_lo]
    up_flat = p._flat_up.cost[p._slots_up]
    for j, e in enumerate(p._edge_order):
        # lower search: flat cost must equal cached lower metric
        assert math.isclose(lo_flat[j], p._cache_lo[e], rel_tol=0, abs_tol=1e-9), (
            f"flat_lo vs cache_lo mismatch on {e}: {lo_flat[j]} != {p._cache_lo[e]}"
        )
        # upper search: flat cost is min(cache_up, _UB_CAP) (D* needs finite)
        from certflow.cert import _UB_CAP
        expect_up = min(p._cache_up[e], _UB_CAP)
        assert math.isclose(up_flat[j], expect_up, rel_tol=0, abs_tol=1e-6), (
            f"flat_up vs cache_up mismatch on {e}: {up_flat[j]} != {expect_up}"
        )


def _assert_due_sync(p: CertPlanner) -> None:
    """_arr_due == _cache_due in the predictor-free (vectorized) refresh path.

    _arr_due is the array twin of _cache_due for the vectorized staggered
    due-subset refresh. This contract holds ONLY when no predictor is supplied:
    with a predictor the dict path drives expiry off _cache_due alone and
    _arr_due is unused (documented at the _arr_due init site). All planners in
    this suite run predictor-free, so the equality must hold."""
    assert p.predictor is None, "due-sync invariant is scoped to the predictor-free path"
    if not p._cache_due:
        return
    for j, e in enumerate(p._edge_order):
        if e in p._cache_due:
            a = p._arr_due[j]
            d = p._cache_due[e]
            if math.isinf(a) and math.isinf(d):
                continue
            assert math.isclose(a, d, rel_tol=0, abs_tol=1e-9), (
                f"_arr_due vs _cache_due mismatch on {e}: {a} != {d}"
            )


def _check_all(p: CertPlanner) -> None:
    _assert_dict_array_sync(p)
    _assert_cache_flat_sync(p)
    _assert_due_sync(p)


def test_state_sync_mixed_workload():
    """Mixed workload: rounds + retargets + snapshot queries + traversal
    ingests, asserting every cross-view equality every 10 rounds."""
    rng = random.Random(7)
    w = grid_world(6, 6, seed=4, kind="bounded", rho=0.02, noise_scale=0.03)
    endpoints = [((0, 0), (5, 5)), ((0, 5), (5, 0)), ((5, 5), (0, 0)),
                 ((2, 0), (3, 5))]
    p = CertPlanner(w, *endpoints[0],
                    PlannerConfig(epsilon=5.0, alpha_prime=0.2, eps_tv=1e-4,
                                  rho_mode="online", adaptive_rate=True))

    _check_all(p)  # immediately after construction

    nodes = list(w.graph)
    for r in range(200):
        cert, sensed = p.round()

        # traversal-style free ingest: observe a couple of random edges
        if r % 3 == 0:
            for _ in range(2):
                u = rng.choice(nodes)
                nbrs = list(w.graph[u])
                if nbrs:
                    v = rng.choice(nbrs)
                    p.ingest_observation((u, v))

        # snapshot queries at varied tau (exercises the gate + oracle build)
        if r % 5 == 0:
            s, g = rng.choice(endpoints)
            p.snapshot_query(s, g, tau=rng.choice([1e-3, 0.5, 5.0]))

        # retarget mid-stream (lifelong operation), keep learned memory
        if r in (40, 110, 160):
            s, g = rng.choice(endpoints)
            p.t += 5.0
            p.retarget(s, g)

        if r % 10 == 0:
            _check_all(p)

    _check_all(p)


def test_state_sync_after_ingest_bumps_version():
    """ingest_observation keeps arrays in sync AND bumps the snapshot version
    (otherwise a stale gate verdict could be reused after the map moved)."""
    w = grid_world(5, 5, seed=1, kind="static", noise_scale=0.01)
    p = CertPlanner(w, (0, 0), (4, 4),
                    PlannerConfig(epsilon=5.0, alpha_prime=0.2, eps_tv=1e-4))
    for _ in range(30):
        p.round()
    _check_all(p)
    e = next(iter(p.beliefs))
    v0 = p._beliefs_version
    p.ingest_observation(e)
    assert p._beliefs_version == v0 + 1
    i = p._edge_idx[e]
    assert p._arr_chat[i] == p.beliefs[e].c_hat
    assert p._arr_obs[i] is np.True_ or bool(p._arr_obs[i]) is True
    assert p._arr_due[i] == p.t  # expired alongside _cache_due
    assert p._cache_due[e] == p.t
    _check_all(p)


def test_state_sync_online_rho_update():
    """_update_online_rho writes rho into every belief AND _arr_rho, and forces
    a full metric rebuild (the dict/array rho views must not diverge)."""
    w = grid_world(6, 6, seed=2, kind="bounded", rho=0.05, noise_scale=0.05)
    p = CertPlanner(w, (0, 0), (5, 5),
                    PlannerConfig(epsilon=5.0, alpha_prime=0.2, eps_tv=1e-4,
                                  rho_mode="online"))
    for _ in range(120):
        p.round()
        # sensing/traversal feeds rate samples; periodically check sync
    _check_all(p)
    # all beliefs share the pooled online rho; arrays must match
    for e, i in p._edge_idx.items():
        assert p._arr_rho[i] == p.beliefs[e].rho
