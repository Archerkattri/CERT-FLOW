"""PASC joint per-edge calibration (arXiv 2605.18812) and the testability layer:
weighted conformal p-values, WATCH test martingale (arXiv 2505.04608), and
conformal e-values / merging (arXiv 2503.13050)."""
import math

import numpy as np
import pytest

from certflow.conformal import (
    ConformalTestMartingale,
    PASCCalibrator,
    conformal_e_value,
    conformal_p_value,
    effective_sample_size,
    merge_e_values,
    path_alpha_edge,
    residual_drift_score,
    score_ratio_e_value,
    weighted_group_quantile,
)


# --------------------------------------------------------------------------- #
# PASC
# --------------------------------------------------------------------------- #
def _paths(n_groups, L, sigma, rng):
    """Disjoint paths of L unique edges, each edge a fresh residual ~N(0,sigma)."""
    residuals = rng.normal(0.0, sigma, size=n_groups * L)
    groups = [list(range(k * L, (k + 1) * L)) for k in range(n_groups)]
    return groups, residuals


def test_pasc_Q_is_max_score_quantile():
    rng = np.random.default_rng(0)
    groups, residuals = _paths(300, 5, 1.0, rng)
    cal = PASCCalibrator(alpha=0.1).fit(groups, residuals)
    # Recompute the expected radius directly.
    max_scores = [max(abs(residuals[i]) for i in g) for g in groups]
    expected = weighted_group_quantile(max_scores, [1.0] * len(groups), 0.1)
    assert cal.Q() == pytest.approx(expected)


def test_pasc_joint_coverage_holds():
    """A fresh path's edges are ALL within +/-Q with prob >= 1 - alpha."""
    rng = np.random.default_rng(1)
    alpha, L = 0.1, 6
    covered = 0
    trials = 400
    for _ in range(trials):
        groups, residuals = _paths(300, L, 1.0, rng)
        cal = PASCCalibrator(alpha=alpha).fit(groups, residuals)
        Q = cal.Q()
        fresh = rng.normal(0.0, 1.0, size=L)  # a fresh path's edge residuals
        if np.all(np.abs(fresh) <= Q):
            covered += 1
    assert covered / trials >= 1 - alpha - 0.03  # joint coverage, small MC slack


def test_pasc_beats_bonferroni_width():
    """PASC's single radius is tighter than Bonferroni per-edge pricing at the
    same joint level: L * Q_pasc <= L * Q_bonf (per-edge alpha/L quantile)."""
    rng = np.random.default_rng(2)
    alpha, L = 0.1, 8
    groups, residuals = _paths(500, L, 1.0, rng)
    cal = PASCCalibrator(alpha=alpha).fit(groups, residuals)
    Q_pasc = cal.Q()
    # Bonferroni: price each edge at its own alpha/L absolute-residual quantile.
    per_edge_alpha = path_alpha_edge(alpha, L)
    all_abs = sorted(abs(residuals))
    r = math.ceil((len(all_abs) + 1) * (1 - per_edge_alpha))
    Q_bonf = math.inf if r > len(all_abs) else all_abs[r - 1]
    assert Q_pasc <= Q_bonf


def test_pasc_drift_retrofit_weights():
    rng = np.random.default_rng(3)
    groups, residuals = _paths(200, 4, 1.0, rng)
    times = list(range(len(groups)))
    cal = PASCCalibrator(alpha=0.1, rho_w=0.98).fit(
        groups, residuals, times=times, t=float(len(groups))
    )
    assert math.isfinite(cal.Q())


def test_pasc_warmup_returns_inf():
    cal = PASCCalibrator(alpha=0.1).fit([], [])
    assert cal.Q() == math.inf


def test_pasc_requires_fit():
    with pytest.raises(RuntimeError):
        PASCCalibrator().Q()


# --------------------------------------------------------------------------- #
# Weighted conformal p-value
# --------------------------------------------------------------------------- #
def test_p_value_uniform_under_null():
    """Conservative conformal p-values of exchangeable scores are ~uniform."""
    rng = np.random.default_rng(4)
    ps = []
    for _ in range(3000):
        cal_scores = list(rng.normal(size=50))
        test = rng.normal()
        ps.append(conformal_p_value(test, cal_scores))
    ps = np.array(ps)
    # super-uniform: P(p <= u) <= u + slack for a grid of u
    for u in (0.2, 0.5, 0.8):
        assert (ps <= u).mean() <= u + 0.05


def test_p_value_small_for_anomalously_large_score():
    cal = list(np.linspace(0, 1, 100))
    assert conformal_p_value(5.0, cal) < 0.02   # far above the buffer
    assert conformal_p_value(-5.0, cal) > 0.98  # far below


# --------------------------------------------------------------------------- #
# e-values and merging
# --------------------------------------------------------------------------- #
def test_e_value_expectation_at_most_one_under_null():
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 200000)
    e = np.array([conformal_e_value(pi, 0.5) for pi in p])
    assert e.mean() <= 1.0 + 0.02   # E[e] = 1 under uniform null


def test_e_value_large_for_small_p():
    assert conformal_e_value(1e-3, 0.5) > conformal_e_value(0.5, 0.5) > conformal_e_value(0.9, 0.5)


def test_merge_average_valid_under_dependence():
    # Average of e-values is itself an e-value (E <= 1) even if identical/dependent.
    rng = np.random.default_rng(6)
    means = []
    for _ in range(20000):
        p = rng.uniform()
        es = [conformal_e_value(p, 0.5)] * 4  # maximally dependent
        means.append(merge_e_values(es, "average"))
    assert np.mean(means) <= 1.0 + 0.03


def test_merge_bad_method():
    with pytest.raises(ValueError):
        merge_e_values([1.0], "median")


# --------------------------------------------------------------------------- #
# WATCH test martingale
# --------------------------------------------------------------------------- #
def test_martingale_flat_under_null():
    """Under the null (uniform p-values) the martingale rarely alarms:
    P(sup M >= 1/delta) <= delta by Ville."""
    rng = np.random.default_rng(7)
    alarms = 0
    runs = 200
    for _ in range(runs):
        m = ConformalTestMartingale(epsilon=0.5, alarm_delta=0.05)
        for _ in range(300):
            m.update(rng.uniform())
        if m.alarm():
            alarms += 1
    assert alarms / runs <= 0.05 + 0.03   # Ville bound (+ MC slack)


def test_martingale_grows_under_violation():
    """A stream of anomalously small p-values (persistent under-coverage) drives
    the martingale past the alarm threshold."""
    m = ConformalTestMartingale(epsilon=0.5, alarm_delta=0.01)
    for _ in range(200):
        m.update(0.01)   # every round the score is anomalously large
    assert m.alarm()
    assert m.value > 1.0 / 0.01


def test_martingale_value_and_running_max():
    m = ConformalTestMartingale()
    m.update(0.5)
    v1 = m.value
    m.update(0.999)   # near-1 p-value: e-value < 1, wealth drops
    assert m.value < v1
    assert m.running_max >= m.value


def test_martingale_bad_params():
    with pytest.raises(ValueError):
        ConformalTestMartingale(epsilon=1.5)
    with pytest.raises(ValueError):
        ConformalTestMartingale(alarm_delta=0.0)


# --------------------------------------------------------------------------- #
# Score-ratio e-value + drift diagnostics
# --------------------------------------------------------------------------- #
def test_score_ratio_e_value_expectation_under_null():
    """E[E] = 1 under exchangeability for the canonical conformal e-value."""
    rng = np.random.default_rng(8)
    es = []
    for _ in range(20000):
        scores = list(np.abs(rng.normal(size=40)))
        test = abs(rng.normal())
        # test point is exchangeable with the calibration scores here
        es.append(score_ratio_e_value(test, scores))
    assert abs(np.mean(es) - 1.0) < 0.05


def test_score_ratio_collapses_when_uninformative():
    # A typical score gives E ~ 1 (no evidence); a huge score gives E >> 1.
    cal = [1.0] * 100
    assert score_ratio_e_value(1.0, cal) == pytest.approx(1.0, abs=1e-6)
    assert score_ratio_e_value(50.0, cal) > 10.0


def test_score_ratio_rejects_negative():
    with pytest.raises(ValueError):
        score_ratio_e_value(-1.0, [1.0, 2.0])


def test_drift_score_zero_when_same_distribution():
    rng = np.random.default_rng(9)
    a = list(rng.normal(size=500))
    b = list(rng.normal(size=500))
    d_same = residual_drift_score(a, b)
    d_shift = residual_drift_score([x + 5.0 for x in a], b)
    assert d_shift > d_same
    assert d_same < 0.5   # same law -> small W1


def test_drift_score_empty():
    assert residual_drift_score([], [1.0]) == 0.0


def test_effective_sample_size():
    assert effective_sample_size([1.0] * 10) == pytest.approx(10.0)
    # One dominating weight -> n_eff near 1.
    assert effective_sample_size([100.0, 1.0, 1.0]) < 2.0
    assert effective_sample_size([]) == 0.0
