"""Levy-Prokhorov distribution-shift quantile option (arXiv 2502.14105)."""
import math

import numpy as np
import pytest

from certflow import CertPlanner, PlannerConfig
from certflow.conformal import ConformalScorer
from certflow.drift import grid_world


def _filled(rho_w=1.0, n=200, seed=0):
    s = ConformalScorer(rho_w=rho_w)
    rng = np.random.default_rng(seed)
    for i in range(n):
        s.push(float(rng.exponential(1.0)), t=float(i))
    return s


def test_lp_reduces_to_plain_conformal_at_zero():
    s = _filled()
    for alpha in (0.05, 0.1, 0.2):
        q_plain = s.quantile(alpha, t=200.0)
        q_lp = s.quantile_lp(alpha, t=200.0, eps=0.0, rho=0.0)
        assert q_lp == pytest.approx(q_plain)


def test_lp_monotone_in_eps():
    s = _filled()
    prev = -math.inf
    for eps in (0.0, 0.5, 1.0, 2.0):
        q = s.quantile_lp(0.1, t=200.0, eps=eps, rho=0.0)
        assert q >= prev - 1e-12
        prev = q
    # exactly a flat offset
    base = s.quantile_lp(0.1, t=200.0, eps=0.0, rho=0.0)
    assert s.quantile_lp(0.1, t=200.0, eps=1.3, rho=0.0) == pytest.approx(base + 1.3)


def test_lp_monotone_in_rho():
    s = _filled(n=400)
    prev = -math.inf
    for rho in (0.0, 0.01, 0.03, 0.05):
        q = s.quantile_lp(0.1, t=400.0, eps=0.0, rho=rho)
        assert q >= prev - 1e-12
        prev = q
    # rho >= alpha => required level >= 1 => +inf (unsupportable)
    assert s.quantile_lp(0.1, t=400.0, eps=0.0, rho=0.1) == math.inf


def test_lp_widens_never_narrows():
    """LP intervals are always at least as wide as the exchangeable ones."""
    s = _filled()
    q_plain = s.quantile(0.1, t=200.0)
    q_lp = s.quantile_lp(0.1, t=200.0, eps=0.3, rho=0.02)
    assert q_lp >= q_plain


def test_coverage_formula_sanity():
    """Cov^WC(q) = F_P(q - eps) - rho, floored at 0, and consistent with the
    quantile: coverage at the LP quantile of level 1-alpha is >= 1-alpha-rho."""
    s = _filled(n=500)
    q = s.quantile(0.1, t=500.0)  # plain (1-0.1) quantile
    # at eps=rho=0 coverage of the plain quantile is >= 0.9
    assert s.coverage_lp(q, t=500.0, eps=0.0, rho=0.0) >= 0.9 - 1e-9
    # adding eps lowers the evaluated CDF point => coverage drops
    cov_eps = s.coverage_lp(q, t=500.0, eps=0.5, rho=0.0)
    assert cov_eps <= s.coverage_lp(q, t=500.0, eps=0.0, rho=0.0) + 1e-12
    # rho subtracts flatly
    assert s.coverage_lp(q, t=500.0, eps=0.0, rho=0.05) == pytest.approx(
        max(0.0, s.cdf(q, t=500.0) - 0.05))
    # the LP quantile guarantees the LP coverage floor
    eps, rho, alpha = 0.4, 0.02, 0.1
    q_lp = s.quantile_lp(alpha, t=500.0, eps=eps, rho=rho)
    assert s.coverage_lp(q_lp, t=500.0, eps=eps, rho=rho) >= 1 - alpha - rho - 1e-9


def test_finite_sample_floor():
    # rho=0 reduces to the usual split-conformal ceil(n(1-alpha))/(n+1)
    n, alpha = 99, 0.1
    assert ConformalScorer.lp_finite_sample_coverage(n, alpha, 0.0) == pytest.approx(
        math.ceil(n * 0.9) / (n + 1))
    # monotone: more adversarial mass rho lowers the floor (net of the ceil)
    f0 = ConformalScorer.lp_finite_sample_coverage(200, 0.1, 0.0)
    f1 = ConformalScorer.lp_finite_sample_coverage(200, 0.1, 0.05)
    assert f1 <= f0 + 1e-9


def test_lp_empirical_coverage_under_drift():
    """Synthetic mean-shift drift: the plain (1-alpha) quantile calibrated on
    the old regime under-covers the shifted test law; the LP quantile with a
    matched (eps, rho) restores coverage >= 1-alpha-rho."""
    rng = np.random.default_rng(3)
    alpha = 0.1
    shift = 0.6  # smooth mean drift the LP eps must absorb
    plain_hits, lp_hits, trials = 0, 0, 3000
    for _ in range(trials):
        s = ConformalScorer(rho_w=1.0)
        for i in range(150):
            s.push(float(rng.exponential(1.0)), t=float(i))  # calibration regime
        y = float(rng.exponential(1.0)) + shift                # shifted test point
        q_plain = s.quantile(alpha, t=150.0)
        q_lp = s.quantile_lp(alpha, t=150.0, eps=shift, rho=0.0)
        plain_hits += y <= q_plain
        lp_hits += y <= q_lp
    plain_cov = plain_hits / trials
    lp_cov = lp_hits / trials
    assert plain_cov < 0.9          # drift breaks the exchangeable quantile
    assert lp_cov >= 0.9 - 3 * math.sqrt(0.9 * 0.1 / trials)
    assert lp_cov > plain_cov


def _true_opt(world, s, g, t):
    import heapq
    adj = {}
    for u in world.graph:
        for v in world.graph[u]:
            adj.setdefault(u, {})[v] = max(world.true_cost((u, v), t), 1e-9)
    dist = {s: 0.0}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        for v, c in adj.get(u, {}).items():
            nd = d + c
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist.get(g, math.inf)


def test_lp_planner_holds_coverage():
    """shift_model='lp' planner stays sound (LB <= OPT <= UB) on a drift grid."""
    cfg = PlannerConfig(epsilon=5.0, alpha_prime=0.1, shift_model="lp",
                        eps_lp=0.05, rho_lp=0.001)
    hits = tot = 0
    for seed in range(4):
        w = grid_world(6, 6, seed=seed, kind="bounded", rho=0.02, noise_scale=0.05)
        p = CertPlanner(w, (0, 0), (5, 5), cfg)
        for r in range(120):
            c, _ = p.round()
            if r < 30 or not c.valid:
                continue
            o = _true_opt(w, (0, 0), (5, 5), p.t - cfg.delta)
            if math.isinf(o):
                continue
            hits += c.lb <= o + 1e-6 and o <= c.ub + 1e-6
            tot += 1
    assert tot > 50
    assert hits / tot >= 0.9
