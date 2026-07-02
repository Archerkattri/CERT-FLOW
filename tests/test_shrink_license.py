"""ShrinkLicense: the anytime-valid test-then-tighten license (Waudby-Smith &
Ramdas 2023 betting confidence sequence) and its zero-effect planner wiring.

Mirrors the style of tests/test_pasc_watch.py.
"""
import math

import numpy as np
import pytest

from certflow.conformal import ShrinkLicense


# --------------------------------------------------------------------------- #
# (1) Confidence-sequence validity: time-uniform upper bound on the mean.
# --------------------------------------------------------------------------- #
def test_cs_validity_time_uniform():
    """iid Bernoulli(mu): the anytime-valid UCB under-covers the true mean over
    the whole horizon with probability <= delta (Ville). Checked empirically."""
    delta, mu, T, reps = 0.1, 0.3, 200, 300
    fails = 0
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)  # deterministic per rep
        sl = ShrinkLicense(grid=(0.5,), delta=delta, n_min=1)
        stream = rng.binomial(1, mu, size=T)
        under = False
        for x in stream:
            sl.update({0.5: float(x)})
            if sl.ucb(0.5) < mu:  # UCB dropped below the true mean: miscoverage
                under = True
                break
        fails += under
    assert fails / reps <= delta + 0.05  # Ville bound + MC slack


def test_cs_validity_small_mean():
    """Same validity check at a small mean (the regime that matters for a
    license: low violation rates)."""
    delta, mu, T, reps = 0.1, 0.1, 200, 300
    fails = 0
    for r in range(reps):
        rng = np.random.default_rng(2000 + r)
        sl = ShrinkLicense(grid=(0.8,), delta=delta, n_min=1)
        for x in rng.binomial(1, mu, size=T):
            sl.update({0.8: float(x)})
            if sl.ucb(0.8) < mu:
                fails += 1
                break
    assert fails / reps <= delta + 0.05


# --------------------------------------------------------------------------- #
# (2) The UCB converges toward the true mean FROM ABOVE on long streams.
# --------------------------------------------------------------------------- #
def test_ucb_converges_from_above():
    rng = np.random.default_rng(7)
    mu, T = 0.2, 5000
    sl = ShrinkLicense(grid=(0.7,), delta=0.05, n_min=1)
    for x in rng.binomial(1, mu, size=T):
        sl.update({0.7: float(x)})
    u = sl.ucb(0.7)
    assert u >= mu - 0.02        # from above (grid + MC slack)
    assert u <= mu + 0.12        # converged close to the truth


def test_ucb_tightens_monotone_in_n():
    """More low-violation evidence => the bound only gets tighter (in expectation
    a long all-zeros stream drives the UCB toward the grid floor)."""
    sl = ShrinkLicense(grid=(0.6,), delta=0.05, n_min=1)
    ucbs = []
    for i in range(400):
        sl.update({0.6: 0.0})
        if i in (20, 100, 399):
            ucbs.append(sl.ucb(0.6))
    assert ucbs[0] >= ucbs[1] >= ucbs[2]  # non-increasing with more evidence


# --------------------------------------------------------------------------- #
# (3) licensed_k: low-violation stream licenses k<1, a shift revokes to 1.0.
# --------------------------------------------------------------------------- #
def test_licensed_k_response_and_revocation():
    grid = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    alpha = 0.1
    sl = ShrinkLicense(grid=grid, delta=0.05, n_min=50)
    # warm-up: not enough samples -> no license yet
    assert sl.licensed_k(alpha) == 1.0
    # low-violation regime: even the tightest shrink holds -> license a k < 1
    for _ in range(200):
        sl.update({k: 0.0 for k in grid})
    lk_low = sl.licensed_k(alpha)
    assert lk_low < 1.0
    # regime shift: violations jump at every shrink factor
    for _ in range(300):
        sl.update({k: 1.0 for k in grid})
    assert sl.licensed_k(alpha) == 1.0  # license self-revoked


def test_licensed_k_n_min_gate():
    grid = (0.5, 1.0)
    sl = ShrinkLicense(grid=grid, delta=0.05, n_min=100)
    for _ in range(50):  # below n_min
        sl.update({0.5: 0.0, 1.0: 0.0})
    assert sl.licensed_k(0.1) == 1.0  # n < n_min: not yet licensed
    for _ in range(60):  # now past n_min
        sl.update({0.5: 0.0, 1.0: 0.0})
    assert sl.licensed_k(0.1) == 0.5


def test_bad_params():
    with pytest.raises(ValueError):
        ShrinkLicense(delta=0.0)
    with pytest.raises(ValueError):
        ShrinkLicense(n_min=0)
    with pytest.raises(ValueError):
        ShrinkLicense(c=1.0)


# --------------------------------------------------------------------------- #
# (4) Planner wiring: certificate bit-identical off AND on; diagnostics carry
#     the license fields when on.
# --------------------------------------------------------------------------- #
def _run(shrink_license, rounds=45):
    from certflow.cert import CertPlanner, PlannerConfig
    from certflow.drift import grid_world

    world = grid_world(8, 8, seed=5, kind="bounded", rho=0.02, noise_scale=0.05)
    cfg = PlannerConfig(
        epsilon=5.0, alpha_prime=0.2, shrink_license=shrink_license,
    )
    p = CertPlanner(world, start=(0, 0), goal=(7, 7), config=cfg)
    certs = []
    for _ in range(rounds):
        cert, _ = p.round()
        certs.append((cert.lb, cert.ub, cert.confidence))
    return certs, p


def test_planner_certificate_bit_identical():
    base, _ = _run(False)
    on, p_on = _run(True)
    assert base == on  # (lb, ub, confidence) identical every round

    d = p_on.diagnostics()
    for key in ("licensed_k", "shrink_ucb", "shrink_n", "shrunk_gap",
                "shrunk_lb", "shrunk_ub"):
        assert key in d
    assert 0.0 < d["licensed_k"] <= 1.0
    assert set(d["shrink_ucb"]) == set(p_on.cfg.shrink_grid)


def test_planner_shrink_feeds_only_when_on():
    _, p_off = _run(False)
    _, p_on = _run(True)
    # off: the license CS is never fed
    assert all(p_off.shrink.n(k) == 0 for k in p_off.cfg.shrink_grid)
    # on: it accumulates real per-edge outcomes
    assert any(p_on.shrink.n(k) > 0 for k in p_on.cfg.shrink_grid)


def test_planner_shrunk_gap_no_wider_than_gap():
    """The licensed shadow gap never exceeds the Tier-1 gap (licensed_k <= 1),
    and equals it when nothing is licensed."""
    _, p = _run(True, rounds=80)
    d = p.diagnostics()
    if p._prev_incumbent and math.isfinite(d["shrunk_gap"]):
        # full-radius (k=1) shadow gap on the same incumbent edges
        full = 0.0
        for e in __import__("certflow.cert", fromlist=["path_edges"]).path_edges(
            p._prev_incumbent
        ):
            b = p.beliefs[e]
            full += 2.0 * (p._shrink_q_edge + b.rho * b.age(p.t))
        assert d["shrunk_gap"] <= full + 1e-9
