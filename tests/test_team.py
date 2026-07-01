"""Additive multi-agent certificate (TEAM-CERT survivor).

Soundness, coverage, and additive-gap tests on a shared-store N-agent grid.
Adapted from projects/certified-planning/experiments/tests/test_team_allocator.py
with the greedy-allocator assertions dropped (only the additive certificate is
ported to CERT-FLOW).
"""
import math

import numpy as np
import pytest

from certflow import (
    Certificate,
    CertPlanner,
    PlannerConfig,
    TeamCertificate,
    additive_certificate,
)
from certflow.drift import grid_world


# --------------------------------------------------------------------------- #
# unit-level properties on hand-built certificates
# --------------------------------------------------------------------------- #
def _cert(lb, ub, conf):
    return Certificate(
        lb=lb, ub=ub, confidence=conf, path=[], epsilon_attainable=True,
        epsilon_floor=0.0,
    )


def test_additive_sums_bounds_and_gap():
    certs = [_cert(1.0, 3.0, 0.95), _cert(2.0, 5.0, 0.9), _cert(0.5, 1.5, 0.99)]
    tc = additive_certificate(certs)
    assert isinstance(tc, TeamCertificate)
    assert tc.lb == pytest.approx(3.5)
    assert tc.ub == pytest.approx(9.5)
    # additive gap == sum of per-agent gaps
    assert tc.gap == pytest.approx(sum(c.gap for c in certs))
    assert tc.gap == pytest.approx(6.0)


def test_union_bound_confidence():
    certs = [_cert(0, 1, 0.95), _cert(0, 1, 0.90)]
    tc = additive_certificate(certs)
    # 1 - ((1-.95)+(1-.90)) = 1 - .15 = .85
    assert tc.confidence == pytest.approx(0.85)
    # floored at 0 when the budgets exhaust
    certs2 = [_cert(0, 1, 0.5), _cert(0, 1, 0.4), _cert(0, 1, 0.3)]
    tc2 = additive_certificate(certs2)
    assert tc2.confidence == 0.0
    assert not tc2.valid


def test_empty_raises():
    with pytest.raises(ValueError):
        additive_certificate([])


# --------------------------------------------------------------------------- #
# soundness + coverage on a real shared-store N-agent instance
# --------------------------------------------------------------------------- #
def _make_agents(n_agents, seed):
    """One SHARED world; N planners driven off it. The world's observe() is a
    single global RNG so every agent's observation refreshes the same ground
    truth -- the shared-store requirement of the additive certificate. Endpoints
    are spread across the grid so the per-agent optima differ."""
    world = grid_world(6, 6, seed=seed, kind="bounded", rho=0.02, noise_scale=0.05)
    cfg = PlannerConfig(epsilon=8.0, alpha_prime=0.1, rho_w=0.99)
    corners = [((0, 0), (5, 5)), ((0, 5), (5, 0)), ((5, 0), (0, 5)),
               ((5, 5), (0, 0)), ((0, 0), (5, 4)), ((1, 0), (4, 5))]
    planners = [
        CertPlanner(world, s, g, cfg) for (s, g) in corners[:n_agents]
    ]
    return world, planners


def _team_opt(world, planners, t):
    """Sum of true single-source optima (Dijkstra on true costs at time t)."""
    import heapq
    adj = {}
    for u in world.graph:
        for v in world.graph[u]:
            adj.setdefault(u, {})[v] = max(world.true_cost((u, v), t), 1e-9)
    total = 0.0
    for p in planners:
        dist = {p.start: 0.0}
        pq = [(0.0, p.start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            if u == p.goal:
                break
            for v, c in adj.get(u, {}).items():
                nd = d + c
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        total += dist.get(p.goal, math.inf)
    return total


@pytest.mark.parametrize("n_agents", [2, 3, 4])
def test_soundness_and_coverage(n_agents):
    hits, total, warm = 0, 0, 25
    for seed in range(6):
        world, planners = _make_agents(n_agents, seed)
        for r in range(warm + 60):
            certs = [p.round()[0] for p in planners]
            if r < warm:
                continue
            tc = additive_certificate(certs)
            if not tc.valid:
                continue
            # all planners advance the same shared world clock in lockstep
            t = planners[0].t - planners[0].cfg.delta
            opt = _team_opt(world, planners, t)
            if not math.isfinite(opt):
                continue
            # soundness: sum LB <= sum OPT <= sum UB
            sound = tc.lb <= opt + 1e-6 and opt <= tc.ub + 1e-6
            hits += int(sound)
            total += 1
    assert total > 50, f"too few valid team certificates ({total})"
    cov = hits / total
    assert cov >= 1 - 0.1, f"team coverage {cov:.3f} < 0.9 (n_agents={n_agents})"


def test_additive_gap_equals_sum_on_live_planners():
    world, planners = _make_agents(3, seed=1)
    for _ in range(40):
        certs = [p.round()[0] for p in planners]
    tc = additive_certificate(certs)
    assert tc.gap == pytest.approx(sum(c.gap for c in certs))
    assert tc.lb == pytest.approx(sum(c.lb for c in certs))
    assert tc.ub == pytest.approx(sum(c.ub for c in certs))
