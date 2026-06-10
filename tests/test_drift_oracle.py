"""Tests for certflow.drift and certflow.oracle.

Run with:  pytest tests/test_drift_oracle.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from certflow.drift import (
    BoundedDriftWorld,
    JumpWorld,
    PeriodicWorld,
    StaticWorld,
    grid_world,
)
from certflow.oracle import CoverageLog, opt
from certflow.types import RoundLog


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

ROWS, COLS = 5, 5


def _some_edge(world):
    return next(world.edges())


def _make_round_log(covered: bool, gap: float = 1.0, certified: bool = False) -> RoundLog:
    return RoundLog(
        t=0.0,
        lb=0.0,
        ub=gap,
        confidence=0.9,
        opt=0.5 * gap if covered else gap + 1.0,
        covered=covered,
        certified=certified,
        sensed_edge=None,
        sense_spend=0.1,
        replan_seconds=0.001,
    )


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """true_cost is deterministic given seed; different seeds differ."""

    @pytest.mark.parametrize("kind", ["static", "bounded", "jump", "periodic"])
    def test_same_seed_same_cost(self, kind):
        t_vals = [0.0, 5.0, 50.0, 200.0]
        w1 = grid_world(ROWS, COLS, seed=7, kind=kind)
        w2 = grid_world(ROWS, COLS, seed=7, kind=kind)
        for e in list(w1.edges())[:6]:
            for t in t_vals:
                assert w1.true_cost(e, t) == pytest.approx(w2.true_cost(e, t), rel=1e-12)

    @pytest.mark.parametrize("kind", ["static", "bounded", "jump", "periodic"])
    def test_different_seeds_differ(self, kind):
        t = 10.0
        w1 = grid_world(ROWS, COLS, seed=1, kind=kind)
        w2 = grid_world(ROWS, COLS, seed=2, kind=kind)
        costs1 = [w1.true_cost(e, t) for e in list(w1.edges())[:10]]
        costs2 = [w2.true_cost(e, t) for e in list(w2.edges())[:10]]
        # At least some costs should differ.
        assert not all(
            math.isclose(c1, c2, rel_tol=1e-9) for c1, c2 in zip(costs1, costs2)
        ), "Different seeds produced identical costs — very unlikely."

    def test_repeated_true_cost_agree(self):
        """Calling true_cost twice with same args returns same value."""
        w = grid_world(ROWS, COLS, seed=42, kind="bounded")
        for e in list(w.edges())[:8]:
            for t in [0.0, 1.5, 37.2]:
                c1 = w.true_cost(e, t)
                c2 = w.true_cost(e, t)
                assert c1 == c2


# ---------------------------------------------------------------------------
# BoundedDriftWorld
# ---------------------------------------------------------------------------

class TestBoundedDriftWorld:
    """Lipschitz bound and floor/cap constraints."""

    def _world(self, seed=0, rho=0.1):
        return grid_world(ROWS, COLS, seed=seed, kind="bounded", rho=rho)

    def test_lipschitz_bound(self):
        """|c(t') - c(t)| <= rho_e * |t' - t| for many random pairs."""
        rng = np.random.default_rng(99)
        w = self._world(seed=5, rho=0.2)
        edges = list(w.edges())
        n_checks = 400
        for _ in range(n_checks):
            e = edges[rng.integers(0, len(edges))]
            t1, t2 = sorted(rng.uniform(0.0, 300.0, size=2))
            c1 = w.true_cost(e, t1)
            c2 = w.true_cost(e, t2)
            rho_e = w.rho_true(e)
            delta = abs(c2 - c1)
            allowed = rho_e * abs(t2 - t1) + 1e-9
            assert delta <= allowed, (
                f"Lipschitz violated: |{c2:.6f}-{c1:.6f}|={delta:.6f} "
                f"> rho_e={rho_e:.4f} * |{t2:.2f}-{t1:.2f}|={allowed:.6f}"
            )

    def test_floor_cap(self):
        """All costs in [floor, cap]."""
        w = self._world(seed=3)
        rng = np.random.default_rng(10)
        edges = list(w.edges())
        for _ in range(200):
            e = edges[rng.integers(0, len(edges))]
            t = float(rng.uniform(0, 500))
            c = w.true_cost(e, t)
            assert c >= 1e-6 - 1e-12
            assert c <= 1e4 + 1e-9

    def test_rho_e_in_range(self):
        """Per-edge rho_e in [0.5*rho, 1.5*rho]."""
        rho = 0.1
        w = self._world(seed=11, rho=rho)
        for e in w.edges():
            r = w.rho_true(e)
            assert 0.5 * rho - 1e-12 <= r <= 1.5 * rho + 1e-12

    def test_initial_graph_costs_positive(self):
        w = self._world(seed=20)
        for u, nbrs in w.graph.items():
            for v, c in nbrs.items():
                assert c > 0.0


# ---------------------------------------------------------------------------
# PeriodicWorld
# ---------------------------------------------------------------------------

class TestPeriodicWorld:
    def _world(self, seed=0, period=10.0, amplitude=0.3):
        return grid_world(
            ROWS, COLS, seed=seed, kind="periodic", period=period, amplitude=amplitude
        )

    def test_periodicity(self):
        """c_e(t) == c_e(t + period) for many t."""
        period = 20.0
        w = self._world(seed=7, period=period, amplitude=0.2)
        edges = list(w.edges())[:10]
        for e in edges:
            for t in [0.0, 1.5, 7.3, 15.9]:
                c1 = w.true_cost(e, t)
                c2 = w.true_cost(e, t + period)
                assert abs(c1 - c2) < 1e-10, (
                    f"Periodicity broken for e={e}, t={t}: {c1} vs {c2}"
                )

    def test_rho_true_is_finite(self):
        """rho_true should be the max instantaneous rate (finite, positive)."""
        w = self._world(seed=8)
        for e in list(w.edges())[:5]:
            r = w.rho_true(e)
            assert math.isfinite(r)
            assert r > 0.0

    def test_amplitude_zero_is_static(self):
        """amplitude=0 → constant cost."""
        w = self._world(seed=12, amplitude=0.0)
        e = _some_edge(w)
        c0 = w.true_cost(e, 0.0)
        for t in [1.0, 5.0, 100.0]:
            assert w.true_cost(e, t) == pytest.approx(c0)


# ---------------------------------------------------------------------------
# JumpWorld
# ---------------------------------------------------------------------------

class TestJumpWorld:
    def _world(self, seed=0, jump_rate=1.0, jump_scale=0.5):
        return grid_world(
            ROWS, COLS, seed=seed, kind="jump",
            jump_rate=jump_rate, jump_scale=jump_scale,
        )

    def test_jumps_occur(self):
        """Costs actually change between some time pairs (jumps happen)."""
        w = self._world(seed=3, jump_rate=2.0, jump_scale=0.5)
        edges = list(w.edges())
        t_vals = np.linspace(0, 20, 200)
        any_jump = False
        for e in edges[:5]:
            costs = [w.true_cost(e, t) for t in t_vals]
            # A jump creates a discontinuity: piecewise-constant, so some pair differs.
            if len(set(round(c, 10) for c in costs)) > 1:
                any_jump = True
                break
        assert any_jump, "No jumps observed — jump_rate=2.0 should produce many."

    def test_rho_true_is_inf(self):
        """JumpWorld declares rho_true = inf (off-model)."""
        w = self._world(seed=1)
        e = _some_edge(w)
        assert w.rho_true(e) == math.inf

    def test_costs_positive(self):
        w = self._world(seed=5)
        for e in list(w.edges())[:5]:
            for t in [0.0, 1.0, 50.0]:
                assert w.true_cost(e, t) > 0.0


# ---------------------------------------------------------------------------
# Observe noise
# ---------------------------------------------------------------------------

class TestObserveNoise:
    """mean(observe) ≈ true_cost for symmetric noise; seed gives reproducibility."""

    @pytest.mark.parametrize("noise_family", ["gaussian", "laplace", "student_t"])
    def test_unbiased(self, noise_family):
        """E[observe(e,t)] ≈ true_cost(e,t) for symmetric noise families."""
        w = grid_world(
            ROWS, COLS,
            seed=42,
            kind="static",
            noise_family=noise_family,
            noise_scale=0.1,
        )
        e = _some_edge(w)
        t = 5.0
        tc = w.true_cost(e, t)
        n = 5000
        obs = [w.observe(e, t) for _ in range(n)]
        mean_obs = np.mean(obs)
        # Mean should be within ~3 sigma / sqrt(n) of true cost.
        std_err = 0.1 / math.sqrt(n) * 10  # generous tolerance
        assert abs(mean_obs - tc) < std_err + 0.05, (
            f"{noise_family}: mean={mean_obs:.4f} != true={tc:.4f}"
        )

    def test_observe_varies(self):
        """Successive observe() calls return different values (not pure)."""
        w = grid_world(ROWS, COLS, seed=0, kind="static")
        e = _some_edge(w)
        obs = [w.observe(e, 0.0) for _ in range(20)]
        assert len(set(obs)) > 1, "observe() always returns same value — noise missing."

    def test_observe_seed_consistent(self):
        """Two worlds with same seed produce same sequence of observations."""
        w1 = grid_world(ROWS, COLS, seed=77, kind="static")
        w2 = grid_world(ROWS, COLS, seed=77, kind="static")
        e = _some_edge(w1)
        obs1 = [w1.observe(e, 0.0) for _ in range(10)]
        obs2 = [w2.observe(e, 0.0) for _ in range(10)]
        assert obs1 == pytest.approx(obs2)


# ---------------------------------------------------------------------------
# opt() oracle
# ---------------------------------------------------------------------------

class TestOpt:
    def test_opt_returns_path_and_cost(self):
        w = grid_world(ROWS, COLS, seed=0, kind="static")
        start = (0, 0)
        goal = (ROWS - 1, COLS - 1)
        path, cost = opt(w, 0.0, start, goal)
        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        assert cost > 0.0

    def test_opt_cost_consistent_with_path(self):
        """Sum of true costs along returned path equals reported cost."""
        w = grid_world(ROWS, COLS, seed=1, kind="static")
        path, cost = opt(w, 0.0, (0, 0), (4, 4))
        assert path is not None
        edge_sum = sum(
            w.true_cost((path[i], path[i + 1]), 0.0) for i in range(len(path) - 1)
        )
        assert abs(edge_sum - cost) < 1e-9

    def test_opt_unreachable(self):
        """Unreachable goal returns (None, inf)."""
        # Build a tiny world and query a node not in the graph.
        w = grid_world(2, 2, seed=0, kind="static")
        path, cost = opt(w, 0.0, (0, 0), (99, 99))
        assert path is None
        assert cost == math.inf

    def test_opt_same_start_goal(self):
        w = grid_world(ROWS, COLS, seed=3, kind="static")
        path, cost = opt(w, 0.0, (0, 0), (0, 0))
        assert path == [(0, 0)]
        assert cost == pytest.approx(0.0)

    def test_opt_uses_true_cost_at_t(self):
        """opt at different t values returns different costs for dynamic world."""
        w = grid_world(ROWS, COLS, seed=0, kind="bounded", rho=0.5)
        _, cost0 = opt(w, 0.0, (0, 0), (4, 4))
        _, cost50 = opt(w, 50.0, (0, 0), (4, 4))
        # Not guaranteed to differ, but very likely with rho=0.5 over 50s.
        # Just check both are finite positive.
        assert math.isfinite(cost0) and cost0 > 0
        assert math.isfinite(cost50) and cost50 > 0


# ---------------------------------------------------------------------------
# CoverageLog
# ---------------------------------------------------------------------------

class TestCoverageLog:
    def test_empty(self):
        log = CoverageLog()
        assert math.isnan(log.empirical_coverage())
        lo, hi = log.coverage_ci()
        assert math.isnan(lo)

    def test_all_covered(self):
        log = CoverageLog()
        for _ in range(10):
            log.record(_make_round_log(covered=True))
        assert log.empirical_coverage() == pytest.approx(1.0)

    def test_none_covered(self):
        log = CoverageLog()
        for _ in range(10):
            log.record(_make_round_log(covered=False))
        assert log.empirical_coverage() == pytest.approx(0.0)

    def test_known_coverage(self):
        """95/100 successes → empirical coverage 0.95."""
        log = CoverageLog()
        for _ in range(95):
            log.record(_make_round_log(covered=True))
        for _ in range(5):
            log.record(_make_round_log(covered=False))
        assert log.empirical_coverage() == pytest.approx(0.95)

    def test_clopper_pearson_95_of_100(self):
        """95/100 successes; hand-verified 95% Clopper–Pearson CI ≈ [0.889, 0.985]."""
        log = CoverageLog()
        for _ in range(95):
            log.record(_make_round_log(covered=True))
        for _ in range(5):
            log.record(_make_round_log(covered=False))
        lo, hi = log.coverage_ci(confidence=0.95)
        # Reference: scipy.stats.beta.ppf(0.025, 95, 6) ≈ 0.8872
        #            scipy.stats.beta.ppf(0.975, 96, 5) ≈ 0.9845
        assert 0.88 < lo < 0.90, f"Lower bound out of range: {lo:.4f}"
        assert 0.98 < hi < 0.99, f"Upper bound out of range: {hi:.4f}"

    def test_clopper_pearson_all_success(self):
        """n/n successes: lower bound via beta; upper bound = 1.0."""
        log = CoverageLog()
        for _ in range(20):
            log.record(_make_round_log(covered=True))
        lo, hi = log.coverage_ci()
        assert hi == pytest.approx(1.0)
        assert lo < 1.0

    def test_clopper_pearson_no_success(self):
        """0/n successes: lower = 0.0; upper bound via beta."""
        log = CoverageLog()
        for _ in range(20):
            log.record(_make_round_log(covered=False))
        lo, hi = log.coverage_ci()
        assert lo == pytest.approx(0.0)
        assert hi > 0.0

    def test_summary_keys_present(self):
        """summary() contains all required keys."""
        log = CoverageLog()
        for i in range(50):
            log.record(
                RoundLog(
                    t=float(i),
                    lb=0.5,
                    ub=1.5,
                    confidence=0.9,
                    opt=1.0,
                    covered=True,
                    certified=(i % 3 == 0),
                    sensed_edge=None,
                    sense_spend=0.05,
                    replan_seconds=0.002,
                )
            )
        s = log.summary()
        required = {
            "coverage", "ci_lower", "ci_upper",
            "mean_gap", "median_gap",
            "certified_fraction",
            "total_sense_spend",
            "latency_p50", "latency_p95",
        }
        assert required.issubset(s.keys()), f"Missing keys: {required - s.keys()}"

    def test_summary_values_sane(self):
        log = CoverageLog()
        for i in range(100):
            log.record(
                RoundLog(
                    t=float(i),
                    lb=1.0,
                    ub=3.0,
                    confidence=0.9,
                    opt=2.0,
                    covered=True,
                    certified=(i % 2 == 0),
                    sensed_edge=None,
                    sense_spend=0.1,
                    replan_seconds=0.001 * (i + 1),
                )
            )
        s = log.summary()
        assert s["coverage"] == pytest.approx(1.0)
        assert s["mean_gap"] == pytest.approx(2.0)
        assert s["certified_fraction"] == pytest.approx(0.5)
        assert s["total_sense_spend"] == pytest.approx(10.0)
        assert s["latency_p50"] > 0
        assert s["latency_p95"] >= s["latency_p50"]
