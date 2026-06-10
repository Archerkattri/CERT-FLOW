"""Statistical and unit tests for the conformal certificate machinery."""
import math

import numpy as np
import pytest

from certflow.conformal import (
    ACITracker,
    ConformalScorer,
    path_alpha_edge,
    path_confidence,
)


def test_quantile_inf_during_warmup():
    s = ConformalScorer(rho_w=1.0)
    assert s.quantile(0.1, t=0.0) == math.inf  # empty buffer
    # n samples support alpha only if n >= (1-alpha)/alpha
    for i in range(5):
        s.push(1.0, t=0.0)
    assert s.quantile(0.05, t=0.0) == math.inf  # needs n >= 19
    assert not s.ready(0.05, t=0.0)
    assert s.ready(0.5, t=0.0)


def test_exchangeable_coverage_iid():
    """rho_w=1 on iid residuals: marginal coverage >= 1 - alpha."""
    rng = np.random.default_rng(0)
    alpha = 0.1
    hits, trials = 0, 4000
    for _ in range(trials):
        s = ConformalScorer(rho_w=1.0)
        res = rng.exponential(1.0, size=50)
        for r in res:
            s.push(r, t=0.0)
        q = s.quantile(alpha, t=0.0)
        hits += rng.exponential(1.0) <= q
    cov = hits / trials
    # exact validity gives >= 0.9; allow 3-sigma MC slack below
    assert cov >= 0.9 - 3 * math.sqrt(0.9 * 0.1 / trials)


def test_weighted_adapts_after_changepoint():
    """After a residual-scale changepoint, age-decay weights recover coverage
    faster than unweighted conformal."""
    rng = np.random.default_rng(1)
    alpha = 0.1

    def coverage_after_shift(rho_w: float) -> float:
        hits, trials = 0, 1500
        for _ in range(trials):
            s = ConformalScorer(rho_w=rho_w)
            for i in range(80):  # old regime: small residuals
                s.push(rng.exponential(0.2), t=float(i))
            for i in range(80, 100):  # new regime: 10x residuals
                s.push(rng.exponential(2.0), t=float(i))
            q = s.quantile(alpha, t=100.0)
            hits += rng.exponential(2.0) <= q
        return hits / trials

    cov_decay = coverage_after_shift(rho_w=0.9)
    cov_flat = coverage_after_shift(rho_w=1.0)
    assert cov_decay > cov_flat + 0.02
    assert cov_decay >= 0.85


def test_delta_stale_properties():
    s = ConformalScorer(rho_w=0.99, eps_tv=0.01)
    assert s.delta_stale(t=0.0) == 1.0  # empty buffer: no claim
    for i in range(20):
        s.push(1.0, t=float(i))
    d_now = s.delta_stale(t=20.0)
    d_later = s.delta_stale(t=200.0)
    assert 0.0 < d_now < d_later <= 1.0  # grows with age, capped
    s0 = ConformalScorer(rho_w=0.99, eps_tv=0.0)
    s0.push(1.0, t=0.0)
    assert s0.delta_stale(t=1000.0) == 0.0  # eps_tv=0: exchangeable, no gap


def test_delta_stale_below_corollary_bound():
    """Realized-age Delta_stale <= the age-uniform 2*eps_tv/(1-rho_w) bound."""
    s = ConformalScorer(rho_w=0.95, eps_tv=0.001)
    for i in range(200):
        s.push(1.0, t=float(i))
    assert s.delta_stale(t=200.0) <= 2 * 0.001 / (1 - 0.95) + 1e-9


def test_aci_long_run_bound_prop41():
    """Prop 4.1 holds for adversarial err sequences respecting the boundary
    convention (err=0 when raw alpha <= 0, err=1 when raw alpha >= 1)."""
    rng = np.random.default_rng(2)
    for gamma in (0.005, 0.05):
        aci = ACITracker(alpha_target=0.1, gamma=gamma)
        for _ in range(5000):
            if aci.alpha_raw <= 0.0:
                err = False
            elif aci.alpha_raw >= 1.0:
                err = True
            else:
                err = bool(rng.random() < 0.8)  # adversarially high miscoverage
            aci.update(err)
        dev = abs(aci.empirical_miscoverage() - aci.alpha_target)
        assert dev <= aci.coverage_bound() + 1e-12


def test_aci_working_alpha_clipped():
    aci = ACITracker(alpha_target=0.1, gamma=0.1)
    for _ in range(100):
        aci.update(True)  # hammer it downward
    assert 0.0 < aci.working_alpha() < 1.0


def test_path_helpers():
    assert path_alpha_edge(0.1, 20) == pytest.approx(0.005)
    assert path_confidence(0.1, [0.01, 0.02]) == pytest.approx(0.87)
    assert path_confidence(0.1, [0.5, 0.5]) < 0.0  # INVALID is representable
    with pytest.raises(ValueError):
        path_alpha_edge(0.1, 0)


def test_buffer_eviction_keeps_newest():
    s = ConformalScorer(rho_w=1.0, max_buffer=10)
    for i in range(25):
        s.push(float(i), t=float(i))
    assert len(s) == 10
    assert min(c.t for c in s._buf) == 15.0
