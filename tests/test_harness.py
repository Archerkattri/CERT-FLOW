"""Tests for certflow.harness.

Covers:
- ExperimentConfig.to_dict() and config_id() stability.
- spawn_seeds reproducibility.
- run_experiment aggregation correctness (coverage, gap, latency, sensing).
- Exception isolation: one bad seed does not kill the sweep.
- save/load round-trip equality including inf/nan fields.
- sweep cartesian product correctness.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from certflow.harness import (
    ExperimentConfig,
    ExperimentResult,
    _episode_result_from_dict,
    _episode_result_to_dict,
    _round_log_from_dict,
    _round_log_to_dict,
    load_results,
    run_experiment,
    save_results,
    spawn_seeds,
    sweep,
)
from certflow.types import EpisodeResult, RoundLog


# ---------------------------------------------------------------------------
# Helpers: deterministic fake episode functions
# ---------------------------------------------------------------------------

def _make_round(
    *,
    covered: bool,
    certified: bool,
    gap: float = 0.3,
    sense_spend: float = 0.5,
    replan_seconds: float = 0.01,
    opt: float = 5.0,
) -> RoundLog:
    lb = opt - gap / 2
    ub = opt + gap / 2
    return RoundLog(
        t=1.0,
        lb=lb,
        ub=ub,
        confidence=0.9,
        opt=opt,
        covered=covered,
        certified=certified,
        sensed_edge=((0, 0), (0, 1)),
        sense_spend=sense_spend,
        replan_seconds=replan_seconds,
    )


def _fake_episode(config: ExperimentConfig, seed: int) -> EpisodeResult:
    """Deterministically fabricate an EpisodeResult from *seed*.

    Layout (all episodes have 4 rounds):
      round 0: covered=True,  certified=False
      round 1: covered=True,  certified=True   <-- first certified
      round 2: covered=False, certified=True
      round 3: covered=True,  certified=True

    replan_seconds are seed-dependent but deterministic.
    sense_spend = 1.0 per round.
    """
    rng = np.random.default_rng(seed)
    base_latency = float(rng.uniform(0.01, 0.05))

    rounds = [
        RoundLog(
            t=float(i),
            lb=4.85,
            ub=5.15,
            confidence=0.9,
            opt=5.0,
            covered=(i != 2),
            certified=(i >= 1),
            sensed_edge=((0, 0), (0, 1)),
            sense_spend=1.0,
            replan_seconds=base_latency * (i + 1),
        )
        for i in range(4)
    ]
    return EpisodeResult(
        rounds=rounds,
        travel_cost=float(seed % 10) + 1.0,
        sense_cost=4.0,
        reached_goal=(seed % 2 == 0),
    )


def _raising_episode(config: ExperimentConfig, seed: int) -> EpisodeResult:
    raise RuntimeError(f"boom at seed {seed}")


def _mixed_episode(config: ExperimentConfig, seed: int) -> EpisodeResult:
    """Raises for odd seeds, succeeds for even seeds."""
    if seed % 2 != 0:
        raise ValueError(f"odd seed {seed}")
    return _fake_episode(config, seed)


# ---------------------------------------------------------------------------
# config_id and to_dict
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_to_dict_roundtrip(self):
        cfg = ExperimentConfig(rows=5, cols=5, rho=0.02, n_seeds=3)
        d = cfg.to_dict()
        assert d["rows"] == 5
        assert d["cols"] == 5
        assert d["rho"] == 0.02
        assert d["n_seeds"] == 3

    def test_config_id_stability(self):
        cfg = ExperimentConfig(rows=10, cols=10, rho=0.01)
        id1 = cfg.config_id()
        id2 = cfg.config_id()
        assert id1 == id2, "config_id must be idempotent"
        assert len(id1) == 12

    def test_config_id_differs_on_change(self):
        cfg1 = ExperimentConfig(rho=0.01)
        cfg2 = ExperimentConfig(rho=0.02)
        assert cfg1.config_id() != cfg2.config_id()

    def test_config_id_differs_on_boolean_change(self):
        cfg1 = ExperimentConfig(use_kappa=True)
        cfg2 = ExperimentConfig(use_kappa=False)
        assert cfg1.config_id() != cfg2.config_id()


# ---------------------------------------------------------------------------
# spawn_seeds
# ---------------------------------------------------------------------------

class TestSpawnSeeds:
    def test_length(self):
        seeds = spawn_seeds(42, 10)
        assert len(seeds) == 10

    def test_reproducibility(self):
        seeds_a = spawn_seeds(0, 5)
        seeds_b = spawn_seeds(0, 5)
        assert seeds_a == seeds_b

    def test_different_base_seeds_differ(self):
        assert spawn_seeds(1, 5) != spawn_seeds(2, 5)

    def test_all_integers(self):
        for s in spawn_seeds(99, 8):
            assert isinstance(s, int)

    def test_independence(self):
        # All generated seeds should be distinct (extremely unlikely to collide)
        seeds = spawn_seeds(0, 100)
        assert len(set(seeds)) == 100


# ---------------------------------------------------------------------------
# run_experiment — aggregation correctness
# ---------------------------------------------------------------------------

class TestRunExperiment:
    def setup_method(self):
        self.config = ExperimentConfig(n_seeds=4, base_seed=7)
        self.result = run_experiment(_fake_episode, self.config)

    def test_episode_count(self):
        assert len(self.result.episodes) == 4

    def test_no_failures(self):
        assert len(self.result.failures) == 0

    def test_coverage_exact(self):
        # 4 episodes × 4 rounds = 16 rounds total.
        # round 2 in each episode: covered=False → 4 uncovered.
        # covered = 12/16 = 0.75
        agg = self.result.aggregate()
        assert agg["n_rounds_total"] == 16
        assert agg["n_covered"] == 12
        assert math.isclose(agg["coverage"], 0.75)

    def test_coverage_ci_bounds(self):
        agg = self.result.aggregate()
        assert agg["coverage_ci_lo"] < agg["coverage"]
        assert agg["coverage_ci_hi"] > agg["coverage"]
        assert 0.0 <= agg["coverage_ci_lo"] <= 1.0
        assert 0.0 <= agg["coverage_ci_hi"] <= 1.0

    def test_gap_stats(self):
        # lb=4.85, ub=5.15 → gap=0.30 for all rounds
        agg = self.result.aggregate()
        assert math.isclose(agg["gap_mean"], 0.30, abs_tol=1e-9)
        assert math.isclose(agg["gap_median"], 0.30, abs_tol=1e-9)

    def test_certified_fraction(self):
        # rounds 0 is not certified; rounds 1,2,3 are → 12/16 = 0.75
        agg = self.result.aggregate()
        assert math.isclose(agg["certified_fraction"], 0.75)

    def test_time_to_first_cert(self):
        # First certified at round index 1 in every episode
        agg = self.result.aggregate()
        assert math.isclose(agg["time_to_first_cert_mean"], 1.0)
        assert math.isclose(agg["time_to_first_cert_median"], 1.0)

    def test_sense_spend(self):
        # 1.0 per round × 16 rounds = 16.0 total
        agg = self.result.aggregate()
        assert math.isclose(agg["sense_spend_total"], 16.0)
        assert math.isclose(agg["sense_spend_mean"], 1.0)

    def test_goal_reached_fraction(self):
        # reached_goal depends on seed: seeds from spawn_seeds(7, 4).
        # _fake_episode sets reached_goal = (seed % 2 == 0).
        seeds = spawn_seeds(7, 4)
        expected_fraction = sum(1 for s in seeds if s % 2 == 0) / 4
        agg = self.result.aggregate()
        assert math.isclose(agg["goal_reached_fraction"], expected_fraction)

    def test_replan_latency_percentiles_ordered(self):
        agg = self.result.aggregate()
        assert agg["replan_latency_p50"] <= agg["replan_latency_p95"]

    def test_travel_cost_mean_is_finite(self):
        agg = self.result.aggregate()
        assert math.isfinite(agg["travel_cost_mean"])


# ---------------------------------------------------------------------------
# Exception isolation
# ---------------------------------------------------------------------------

class TestExceptionIsolation:
    def test_all_raise_yields_zero_episodes(self):
        config = ExperimentConfig(n_seeds=3, base_seed=0)
        result = run_experiment(_raising_episode, config)
        assert len(result.episodes) == 0
        assert len(result.failures) == 3

    def test_failure_count_in_aggregate(self):
        config = ExperimentConfig(n_seeds=3, base_seed=0)
        result = run_experiment(_raising_episode, config)
        agg = result.aggregate()
        assert agg["failure_count"] == 3

    def test_mixed_episode_partial_success(self):
        # With 6 seeds, ~half will be odd (raise) and ~half will be even (succeed).
        config = ExperimentConfig(n_seeds=6, base_seed=0)
        result = run_experiment(_mixed_episode, config)
        total = len(result.episodes) + len(result.failures)
        assert total == 6
        assert len(result.failures) > 0
        assert len(result.episodes) > 0

    def test_one_bad_seed_doesnt_kill_sweep(self):
        """Specifically: first seed raises, rest succeed."""
        call_count = {"n": 0}

        def episode_fn(config: ExperimentConfig, seed: int) -> EpisodeResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first episode fails")
            return _fake_episode(config, seed)

        config = ExperimentConfig(n_seeds=5, base_seed=1)
        result = run_experiment(episode_fn, config)
        assert len(result.episodes) == 4
        assert len(result.failures) == 1
        assert call_count["n"] == 5


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def _make_result_with_nonfinite(self) -> ExperimentResult:
        """Construct a result that includes inf and nan in round logs."""
        config = ExperimentConfig(n_seeds=1, base_seed=0)
        round_nan_opt = RoundLog(
            t=0.0,
            lb=4.0,
            ub=6.0,
            confidence=0.9,
            opt=math.nan,   # outside sim
            covered=False,
            certified=False,
            sensed_edge=None,
            sense_spend=0.0,
            replan_seconds=math.inf,
        )
        round_normal = RoundLog(
            t=1.0,
            lb=4.5,
            ub=5.5,
            confidence=0.95,
            opt=5.0,
            covered=True,
            certified=True,
            sensed_edge=((1, 2), (1, 3)),
            sense_spend=1.0,
            replan_seconds=0.005,
        )
        ep = EpisodeResult(
            rounds=[round_nan_opt, round_normal],
            travel_cost=math.inf,
            sense_cost=1.0,
            reached_goal=True,
        )
        result = ExperimentResult(config=config, episodes=[ep], seeds=[42])
        return result

    def test_round_trip_basic(self):
        config = ExperimentConfig(n_seeds=2, base_seed=5)
        result = run_experiment(_fake_episode, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_results(result, tmp)
            loaded = load_results(path)

        assert loaded.config == result.config
        assert len(loaded.episodes) == len(result.episodes)
        for ep_orig, ep_load in zip(result.episodes, loaded.episodes):
            assert ep_orig.reached_goal == ep_load.reached_goal
            assert math.isclose(ep_orig.travel_cost, ep_load.travel_cost)
            assert len(ep_orig.rounds) == len(ep_load.rounds)
            for r_orig, r_load in zip(ep_orig.rounds, ep_load.rounds):
                assert math.isclose(r_orig.lb, r_load.lb)
                assert math.isclose(r_orig.ub, r_load.ub)
                assert r_orig.covered == r_load.covered
                assert r_orig.certified == r_load.certified

    def test_round_trip_nan_inf(self):
        result = self._make_result_with_nonfinite()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_results(result, tmp)
            loaded = load_results(path)

        r0_orig = result.episodes[0].rounds[0]
        r0_load = loaded.episodes[0].rounds[0]
        assert math.isnan(r0_orig.opt)
        assert math.isnan(r0_load.opt)
        assert math.isinf(r0_orig.replan_seconds)
        assert math.isinf(r0_load.replan_seconds)

        ep_orig = result.episodes[0]
        ep_load = loaded.episodes[0]
        assert math.isinf(ep_orig.travel_cost)
        assert math.isinf(ep_load.travel_cost)

    def test_filename_uses_config_id(self):
        config = ExperimentConfig(n_seeds=1, base_seed=0)
        result = run_experiment(_fake_episode, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_results(result, tmp)
            assert path.name == f"{config.config_id()}.json"

    def test_sensed_edge_tuple_roundtrip(self):
        round_log = RoundLog(
            t=0.0,
            lb=1.0,
            ub=2.0,
            confidence=0.9,
            opt=1.5,
            covered=True,
            certified=False,
            sensed_edge=((3, 4), (3, 5)),
            sense_spend=0.1,
            replan_seconds=0.002,
        )
        d = _round_log_to_dict(round_log)
        restored = _round_log_from_dict(d)
        assert restored.sensed_edge == ((3, 4), (3, 5))

    def test_sensed_edge_none_roundtrip(self):
        round_log = RoundLog(
            t=0.0,
            lb=1.0,
            ub=2.0,
            confidence=0.9,
            opt=1.5,
            covered=True,
            certified=False,
            sensed_edge=None,
            sense_spend=0.0,
            replan_seconds=0.001,
        )
        d = _round_log_to_dict(round_log)
        restored = _round_log_from_dict(d)
        assert restored.sensed_edge is None


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

class TestSweep:
    def test_cartesian_product_size(self):
        config = ExperimentConfig(n_seeds=1, base_seed=0)
        results = sweep(_fake_episode, config, {"rho": [0.0, 0.01, 0.1], "epsilon": [0.5, 1.0]})
        assert len(results) == 6  # 3 × 2

    def test_all_combos_present(self):
        config = ExperimentConfig(n_seeds=1, base_seed=0)
        rho_vals = [0.0, 0.1]
        eps_vals = [0.5, 1.0]
        results = sweep(_fake_episode, config, {"rho": rho_vals, "epsilon": eps_vals})
        combos_found = {(r.config.rho, r.config.epsilon) for r in results}
        expected = {(rho, eps) for rho in rho_vals for eps in eps_vals}
        assert combos_found == expected

    def test_unchanged_fields_preserved(self):
        config = ExperimentConfig(rows=7, cols=7, n_seeds=1, base_seed=0)
        results = sweep(_fake_episode, config, {"rho": [0.0, 0.05]})
        for r in results:
            assert r.config.rows == 7
            assert r.config.cols == 7

    def test_empty_grid_returns_one_result(self):
        config = ExperimentConfig(n_seeds=1, base_seed=0)
        results = sweep(_fake_episode, config, {})
        assert len(results) == 1
        assert results[0].config == config

    def test_single_param_sweep(self):
        config = ExperimentConfig(n_seeds=1, base_seed=0)
        results = sweep(_fake_episode, config, {"epsilon": [0.1, 0.5, 1.0, 2.0]})
        assert len(results) == 4
        epsilons = [r.config.epsilon for r in results]
        assert epsilons == [0.1, 0.5, 1.0, 2.0]

    def test_sweep_results_are_experiment_results(self):
        config = ExperimentConfig(n_seeds=2, base_seed=0)
        results = sweep(_fake_episode, config, {"rho": [0.0, 0.01]})
        for r in results:
            assert isinstance(r, ExperimentResult)
            assert len(r.episodes) == 2


# ---------------------------------------------------------------------------
# Edge cases for aggregate()
# ---------------------------------------------------------------------------

class TestAggregateEdgeCases:
    def test_no_episodes_aggregate(self):
        config = ExperimentConfig()
        result = ExperimentResult(config=config)
        agg = result.aggregate()
        assert agg["n_rounds_total"] == 0
        assert math.isnan(agg["coverage"])

    def test_never_certified_ttfc_is_nan(self):
        config = ExperimentConfig(n_seeds=1, base_seed=0)

        def never_certified(cfg, seed):
            rounds = [
                RoundLog(
                    t=float(i), lb=4.0, ub=6.0, confidence=0.9,
                    opt=5.0, covered=True, certified=False,
                    sensed_edge=None, sense_spend=0.5, replan_seconds=0.01,
                )
                for i in range(3)
            ]
            return EpisodeResult(rounds=rounds, travel_cost=1.0, sense_cost=1.5)

        result = run_experiment(never_certified, config)
        agg = result.aggregate()
        assert math.isnan(agg["time_to_first_cert_mean"])
        assert math.isnan(agg["time_to_first_cert_median"])

    def test_full_coverage_ci(self):
        """All rounds covered → CI upper bound == 1.0."""
        config = ExperimentConfig(n_seeds=1, base_seed=0)

        def all_covered(cfg, seed):
            rounds = [
                RoundLog(
                    t=float(i), lb=4.0, ub=6.0, confidence=0.9,
                    opt=5.0, covered=True, certified=True,
                    sensed_edge=None, sense_spend=0.5, replan_seconds=0.01,
                )
                for i in range(5)
            ]
            return EpisodeResult(rounds=rounds, travel_cost=1.0, sense_cost=2.5)

        result = run_experiment(all_covered, config)
        agg = result.aggregate()
        assert math.isclose(agg["coverage"], 1.0)
        assert math.isclose(agg["coverage_ci_hi"], 1.0)

    def test_zero_coverage_ci(self):
        """No rounds covered → CI lower bound == 0.0."""
        config = ExperimentConfig(n_seeds=1, base_seed=0)

        def none_covered(cfg, seed):
            rounds = [
                RoundLog(
                    t=float(i), lb=4.0, ub=6.0, confidence=0.9,
                    opt=5.0, covered=False, certified=False,
                    sensed_edge=None, sense_spend=0.5, replan_seconds=0.01,
                )
                for i in range(5)
            ]
            return EpisodeResult(rounds=rounds, travel_cost=1.0, sense_cost=2.5)

        result = run_experiment(none_covered, config)
        agg = result.aggregate()
        assert math.isclose(agg["coverage"], 0.0)
        assert math.isclose(agg["coverage_ci_lo"], 0.0)


class TestParallelExecution:
    def test_parallel_matches_sequential(self):
        from certflow.episodes import tier0_episode
        from certflow.harness import ExperimentConfig, run_experiment

        cfg = ExperimentConfig(
            rows=4, cols=4, kind="bounded", rho=0.01,
            noise_scale=0.05, epsilon=4.0, alpha_prime=0.2, rho_w=0.99,
            eps_tv=1e-4, n_seeds=4, max_rounds=40, base_seed=7,
        )
        seq = run_experiment(tier0_episode, cfg, workers=1)
        par = run_experiment(tier0_episode, cfg, workers=4)
        assert len(seq.episodes) == len(par.episodes) == 4
        a, b = seq.aggregate(), par.aggregate()
        for key in ("n_rounds_total", "coverage", "gap_median",
                    "certified_fraction", "sense_spend_total"):
            assert a[key] == b[key], key
