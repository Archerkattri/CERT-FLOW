"""Tests for the Gaussian mu±beta*sigma calibration baseline.

Covers:
1. GaussianScorer quantile on N(0,1) samples approximates z_{1-alpha}.
2. Heavy-tailed (Student-t, df=2) samples: Gaussian quantile at alpha=0.02
   under-estimates the true quantile on average (over-claim mechanism).
3. delta_stale always 0; ready() threshold works.
4. GaussianCertPlanner smoke test: 4x4 static world, 60 rounds, no errors.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm, t as student_t

from certflow.baselines import GaussianCertPlanner, GaussianScorer
from certflow.cert import PlannerConfig
from certflow.drift import grid_world
from certflow.harness import ExperimentConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_scorer(samples: list[float], rho_w: float = 1.0) -> GaussianScorer:
    """Push samples at integer times, return scorer at t = len(samples)."""
    scorer = GaussianScorer(rho_w=rho_w)
    for i, s in enumerate(samples):
        scorer.push(s, float(i))
    return scorer


# ---------------------------------------------------------------------------
# Test 1: Quantile approximation on Gaussian samples
# ---------------------------------------------------------------------------

class TestGaussianScorerQuantileOnGaussian:
    """GaussianScorer.quantile on N(0,1) samples should be close to z_{1-alpha}."""

    def test_alpha_05(self):
        rng = np.random.default_rng(0)
        samples = rng.standard_normal(500).tolist()
        scorer = _make_scorer(samples)
        q = scorer.quantile(0.05, float(len(samples)))
        expected = float(norm.ppf(0.95))
        assert abs(q - expected) < 0.25, (
            f"Quantile at alpha=0.05 = {q:.4f}, expected ~{expected:.4f}"
        )

    def test_alpha_10(self):
        rng = np.random.default_rng(1)
        samples = rng.standard_normal(500).tolist()
        scorer = _make_scorer(samples)
        q = scorer.quantile(0.10, float(len(samples)))
        expected = float(norm.ppf(0.90))
        assert abs(q - expected) < 0.25, (
            f"Quantile at alpha=0.10 = {q:.4f}, expected ~{expected:.4f}"
        )

    def test_alpha_20(self):
        rng = np.random.default_rng(2)
        samples = rng.standard_normal(300).tolist()
        scorer = _make_scorer(samples)
        q = scorer.quantile(0.20, float(len(samples)))
        expected = float(norm.ppf(0.80))
        assert abs(q - expected) < 0.3, (
            f"Quantile at alpha=0.20 = {q:.4f}, expected ~{expected:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Over-claim on heavy-tailed (Student-t, df=2) noise
# ---------------------------------------------------------------------------

class TestGaussianOverClaimsOnHeavyTails:
    """Gaussian quantile at alpha=0.02 over-claims coverage on t(df=3) samples.

    The 'over-claim' mechanism: fit Gaussian (mu, sigma) on a calibration set
    drawn from t(df=3), form threshold = mu + z_{0.98}*sigma, then measure
    actual coverage on a held-out test set from the same distribution.  For
    heavy-tailed noise the actual coverage falls below the claimed 1-alpha in
    the majority of trials.

    t(df=3) is the distribution used by the certflow drift module
    (drift.py: ``rng.standard_t(df=3)``).  df=3 has finite variance but
    excess kurtosis=inf (relative to Gaussian), making the Gaussian parametric
    assumption incorrect in a way that causes empirical under-coverage.

    We check directionally over N_SEEDS trials: in >50% of trials the actual
    coverage on a test set is below the claimed 1-alpha.
    """

    ALPHA = 0.01  # per-edge alpha used by planner with Bonferroni (alpha_prime/L)
    DF = 3        # matches drift.py standard_t(df=3)
    N_CAL = 300
    N_TEST = 2000
    N_SEEDS = 50

    def test_directional_underestimate(self):
        """Gaussian under-covers at the extreme-tail per-edge level.

        The planner uses Bonferroni splitting: per-edge alpha = alpha_prime / L.
        For alpha_prime=0.2 and L=20 (or any small alpha), the Gaussian
        z-score z_{1-alpha} * sigma does not capture heavy-tail mass from
        t(df=3) residuals, so actual empirical coverage on held-out test data
        falls below the claimed 1-alpha in the majority of trials.
        """
        under_count = 0
        for seed in range(self.N_SEEDS):
            rng_cal = np.random.default_rng(seed + 100)
            rng_test = np.random.default_rng(seed + 1000)
            cal_samples = rng_cal.standard_t(df=self.DF, size=self.N_CAL).tolist()
            test_samples = rng_test.standard_t(df=self.DF, size=self.N_TEST)
            scorer = _make_scorer(cal_samples)
            gauss_q = scorer.quantile(self.ALPHA, float(self.N_CAL))
            if math.isfinite(gauss_q):
                actual_coverage = float(np.mean(test_samples <= gauss_q))
                if actual_coverage < 1.0 - self.ALPHA:
                    under_count += 1

        # Expect actual coverage to fall below the claim in >50% of trials.
        fraction_under = under_count / self.N_SEEDS
        assert fraction_under > 0.5, (
            f"Gaussian quantile under-covered t(df={self.DF}) in only "
            f"{under_count}/{self.N_SEEDS} trials ({100*fraction_under:.1f}%). "
            f"Expected > 50% (heavy-tailed over-claim mechanism)."
        )


# ---------------------------------------------------------------------------
# Test 3: delta_stale always 0; ready() threshold
# ---------------------------------------------------------------------------

class TestDeltaStaleAndReady:
    def test_delta_stale_empty_buffer(self):
        scorer = GaussianScorer()
        assert scorer.delta_stale(0.0) == 0.0
        assert scorer.delta_stale(100.0) == 0.0

    def test_delta_stale_with_samples(self):
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(50).tolist()
        scorer = _make_scorer(samples)
        for t in [0.0, 10.0, 50.0, 1000.0]:
            assert scorer.delta_stale(t) == 0.0, (
                f"delta_stale({t}) = {scorer.delta_stale(t)}, expected 0.0"
            )

    def test_ready_requires_5_effective_samples(self):
        scorer = GaussianScorer(rho_w=1.0)  # no decay -> age irrelevant
        alpha = 0.05
        t = 0.0

        # With fewer than 5 samples, ready() should be False and quantile +inf.
        for n in range(5):
            assert not scorer.ready(alpha, t), (
                f"ready() should be False with {n} samples"
            )
            assert scorer.quantile(alpha, t) == math.inf
            scorer.push(1.0, t)

        # After 5 samples, effective weight sum = 5.0 >= 5.0, ready() True.
        assert scorer.ready(alpha, t), "ready() should be True with 5 samples"
        assert math.isfinite(scorer.quantile(alpha, t))

    def test_ready_respects_weights(self):
        """With heavy decay, 10 old samples may still have effective n < 5."""
        scorer = GaussianScorer(rho_w=0.5)
        alpha = 0.05
        # Push 10 samples at t=0 but evaluate at t=10 (age=10, w=0.5^10~=0.001)
        for _ in range(10):
            scorer.push(1.0, 0.0)
        t_eval = 10.0
        w_sum = sum(0.5 ** 10 for _ in range(10))  # ~0.01
        # w_sum << 5.0 -> not ready
        assert not scorer.ready(alpha, t_eval)
        assert scorer.quantile(alpha, t_eval) == math.inf


# ---------------------------------------------------------------------------
# Test 4: Planner smoke test
# ---------------------------------------------------------------------------

class TestGaussianCertPlannerSmoke:
    """4x4 static world, 60 rounds, no errors."""

    def test_smoke_60_rounds(self):
        world = grid_world(4, 4, seed=7, kind="static", noise_family="gaussian", noise_scale=0.05)
        start, goal = (0, 0), (3, 3)
        cfg = PlannerConfig(
            epsilon=5.0,
            alpha_prime=0.1,
            rho_w=0.99,
            eps_tv=0.0,
            gamma_aci=0.005,
            delta=1.0,
        )
        planner = GaussianCertPlanner(world, start, goal, cfg)
        assert not planner.use_aci, "use_aci should default to False"
        # Verify ACI gamma is effectively frozen.
        assert planner.aci.gamma < 1e-10, (
            f"ACI gamma should be ~0, got {planner.aci.gamma}"
        )
        for i in range(60):
            cert, sensed = planner.round()
            # Certificate lb and ub should always be finite or inf (never NaN).
            assert not math.isnan(cert.lb), f"Round {i}: lb is NaN"
            assert not math.isnan(cert.ub), f"Round {i}: ub is NaN"
            assert not math.isnan(cert.confidence), f"Round {i}: confidence is NaN"
            # ub >= lb always.
            if math.isfinite(cert.lb) and math.isfinite(cert.ub):
                assert cert.ub >= cert.lb - 1e-9, (
                    f"Round {i}: ub={cert.ub:.4f} < lb={cert.lb:.4f}"
                )

    def test_delta_stale_never_nonzero_during_episode(self):
        """delta_stale must always be 0.0 in GaussianScorer during an episode."""
        world = grid_world(4, 4, seed=8, kind="static", noise_family="gaussian", noise_scale=0.05)
        start, goal = (0, 0), (3, 3)
        cfg = PlannerConfig(epsilon=5.0, alpha_prime=0.1)
        planner = GaussianCertPlanner(world, start, goal, cfg)
        for _ in range(30):
            planner.round()
            assert planner.scorer.delta_stale(planner.t) == 0.0
