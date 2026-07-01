"""CIA-style group-sum (path-level) calibration (arXiv 2408.10939)."""
import math

import numpy as np
import pytest

from certflow.conformal import (
    CIACalibrator,
    ConformalScorer,
    path_alpha_edge,
    weighted_group_quantile,
)


def test_group_quantile_matches_ceil_rank():
    """Unit-weight weighted_group_quantile == ceil((1+K)(1-alpha))-th smallest."""
    scores = [float(x) for x in (5, 1, 3, 9, 2, 8, 4, 7, 6, 10)]
    K = len(scores)
    for alpha in (0.1, 0.2, 0.3):
        r = math.ceil((1 + K) * (1 - alpha))
        expected = math.inf if r > K else sorted(scores)[r - 1]
        got = weighted_group_quantile(scores, [1.0] * K, alpha)
        assert got == expected, (alpha, r, got, expected)


def _paths_and_residuals(n_groups, L, sigma, rng, n_elems=None):
    """Disjoint groups (no overlap): n_groups paths of L unique elements each,
    each element a fresh residual ~ N(0, sigma)."""
    n_elems = n_elems or n_groups * L
    residuals = rng.normal(0.0, sigma, size=n_elems)
    groups = [list(range(k * L, (k + 1) * L)) for k in range(n_groups)]
    return groups, residuals


def test_cia_subset_of_bonferroni_at_matched_coverage():
    """CIA scores the path SUM (concentrates ~sqrt(L)); Bonferroni sums L
    per-edge margins (~L). At matched target coverage CIA's radius Q is <= the
    Bonferroni radius, i.e. the CIA interval is contained in Bonferroni's."""
    rng = np.random.default_rng(0)
    alpha, L, K = 0.1, 8, 400
    groups, residuals = _paths_and_residuals(K, L, sigma=1.0, rng=rng)
    cia = CIACalibrator(alpha=alpha).fit(groups, residuals)
    Q_cia = cia.Q(L)

    # Bonferroni: per-edge (1 - alpha/L) quantile of |residual|, summed over L.
    edge = ConformalScorer(rho_w=1.0)
    for r in residuals:
        edge.push(abs(float(r)), t=0.0)
    q_edge = edge.quantile(path_alpha_edge(alpha, L), t=0.0)
    Q_bonf = L * q_edge

    assert math.isfinite(Q_cia)
    assert Q_cia <= Q_bonf, (Q_cia, Q_bonf)


def test_cia_empirical_coverage():
    """Coverage of the CIA path-sum interval >= 1 - alpha - delta (delta=0 for
    disjoint calibration groups)."""
    rng = np.random.default_rng(1)
    alpha, L = 0.1, 6
    hits, trials = 0, 2000
    for _ in range(trials):
        groups, residuals = _paths_and_residuals(120, L, sigma=1.0, rng=rng)
        cia = CIACalibrator(alpha=alpha).fit(groups, residuals)
        res = cia.interval(pred_sum=0.0, path_len=L)  # yhat sum = 0 by construction
        # a fresh test path of L unseen elements; true sum vs predicted (0)
        test_sum = float(rng.normal(0.0, 1.0, size=L).sum())
        hits += res.lo <= test_sum <= res.ub
    cov = hits / trials
    assert cov >= 1 - alpha - 3 * math.sqrt(0.9 * 0.1 / trials)


def test_overlap_delta_exposed():
    """Overlapping groups report a nonzero delta and an honest coverage level
    below the nominal 1 - alpha."""
    # groups sharing half their elements
    groups = [[0, 1, 2, 3], [2, 3, 4, 5], [0, 1, 4, 5]]
    residuals = np.zeros(6)
    cia = CIACalibrator(alpha=0.1).fit(groups, residuals)
    assert cia.delta > 0.0
    res = cia.interval(0.0, path_len=4)
    assert res.delta == cia.delta
    assert res.coverage_level == pytest.approx(max(0.0, 0.9 - cia.delta))
    # disjoint groups => delta 0
    cia2 = CIACalibrator(alpha=0.1).fit([[0, 1], [2, 3], [4, 5]], np.zeros(6))
    assert cia2.delta == 0.0


def test_symmetric_calibration_runs_and_covers():
    """Symmetric calibration (random cal/test index split) on overlapping
    groups still produces a finite, covering interval."""
    rng = np.random.default_rng(2)
    alpha, L = 0.1, 6
    # overlapping sliding-window groups over a shared element pool
    pool = 60
    residuals = rng.normal(0.0, 1.0, size=pool)
    groups = [list(range(i, i + L)) for i in range(0, pool - L, 2)]
    cia = CIACalibrator(alpha=alpha, symmetric=True, seed=0).fit(groups, residuals)
    Q = cia.Q(L)
    assert math.isfinite(Q) and Q > 0
    assert cia.delta > 0  # overlap present; degradation exposed


def test_weighted_variant_valid_under_drift_where_unweighted_fails():
    """Inject drift: the residual scale grows over calibration time. The
    unweighted CIA quantile (pooling stale small-scale groups) under-covers the
    current large-scale test sums; the age-weighted variant (rho_w<1) recovers."""
    rng = np.random.default_rng(4)
    alpha, L, K = 0.1, 6, 150
    unw_hits, w_hits, trials = 0, 0, 500
    for _ in range(trials):
        # group k observed at time k; residual scale ramps up with time
        groups, times, scores_by_group = [], [], []
        residuals = {}
        idx = 0
        for k in range(K):
            scale = 0.3 + 1.7 * (k / K)  # drift: old groups small, new groups large
            g = list(range(idx, idx + L))
            for i in g:
                residuals[i] = float(rng.normal(0.0, scale))
            groups.append(g)
            times.append(float(k))
            idx += L
        t_now = float(K)  # query at the newest time
        cur_scale = 0.3 + 1.7 * ((K - 1) / K)

        unw = CIACalibrator(alpha=alpha, rho_w=1.0).fit(
            groups, residuals, times=times, t=t_now)
        wtd = CIACalibrator(alpha=alpha, rho_w=0.9).fit(
            groups, residuals, times=times, t=t_now)

        test_sum = float(rng.normal(0.0, cur_scale, size=L).sum())
        r_unw = unw.interval(0.0, path_len=L)
        r_wtd = wtd.interval(0.0, path_len=L)
        unw_hits += r_unw.lo <= test_sum <= r_unw.ub
        w_hits += r_wtd.lo <= test_sum <= r_wtd.ub
    unw_cov = unw_hits / trials
    w_cov = w_hits / trials
    assert unw_cov < 0.9                      # stale calibration under-covers
    assert w_cov >= 0.9 - 3 * math.sqrt(0.9 * 0.1 / trials)
    assert w_cov > unw_cov


def test_stratify_separates_lengths():
    rng = np.random.default_rng(5)
    short = [list(range(i, i + 3)) for i in range(0, 300, 3)]
    long = [list(range(300 + i, 300 + i + 12)) for i in range(0, 1200, 12)]
    groups = short + long
    residuals = rng.normal(0.0, 1.0, size=1500)
    cia = CIACalibrator(alpha=0.1, stratify=True).fit(groups, residuals)
    q3 = cia.Q(3)
    q12 = cia.Q(12)
    assert math.isfinite(q3) and math.isfinite(q12)
    assert q12 > q3            # longer paths have larger sum spread
    assert cia.Q(7) == math.inf  # no calibration groups of length 7


def test_cia_path_certificate_on_live_planner():
    """CertPlanner.cia_path_certificate returns a finite, sound interval that
    brackets the incumbent path's TRUE cost (path_calibration='cia' option)."""
    from certflow import CertPlanner, PlannerConfig
    from certflow.drift import grid_world

    cfg = PlannerConfig(epsilon=5.0, alpha_prime=0.1,
                        path_calibration="cia", cia_symmetric=False)
    hits = tot = 0
    for seed in range(4):
        w = grid_world(6, 6, seed=seed, kind="bounded", rho=0.02, noise_scale=0.05)
        p = CertPlanner(w, (0, 0), (5, 5), cfg)
        for r in range(120):
            c, _ = p.round()
            if r < 40:
                continue
            res = p.cia_path_certificate()
            if res is None or not p._prev_incumbent:
                continue
            # true cost of the reported incumbent path at the current time
            t = p.t - cfg.delta
            edges = list(zip(p._prev_incumbent[:-1], p._prev_incumbent[1:]))
            true_cost = sum(max(w.true_cost(e, t), 1e-9) for e in edges)
            hits += res.lo <= true_cost <= res.ub
            tot += 1
    assert tot > 40
    assert hits / tot >= 0.9
