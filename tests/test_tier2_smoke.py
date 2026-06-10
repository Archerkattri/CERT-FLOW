"""Smoke tests for the Tier-2 sweep script.

Verifies that run_tier2.py exposes importable functions and that a micro-run
(2 seeds, 1 condition) produces an aggregate dict with the expected keys and
a goal fraction in [0, 1].
"""
from __future__ import annotations

import dataclasses
import math

import pytest

# Import the public API we need from run_tier2
from scripts.run_tier2 import BASE, aggregate_tier2, build_conditions
from certflow.episodes import tier2_episode
from certflow.harness import ExperimentConfig, run_experiment

EXPECTED_KEYS = {
    "goal_frac",
    "mission_rounds_mean",
    "regret_mean",
    "regret_median",
    "sense_spend_mean",
    "coverage_valid",
    "n_episodes",
    "n_reached",
}


def _micro_config() -> ExperimentConfig:
    """2-seed micro-run: cert sensing, when_certified, budget=20, fast termination."""
    return dataclasses.replace(
        BASE,
        n_seeds=2,
        max_rounds=80,
        sensing_policy="cert",
        move_policy="when_certified",
        sense_budget=20.0,
    )


def test_build_conditions_returns_nonempty():
    """build_conditions should return at least one (label, config) pair."""
    conds = build_conditions()
    assert len(conds) > 0
    for label, cfg in conds:
        assert isinstance(label, str) and len(label) > 0
        assert isinstance(cfg, ExperimentConfig)


def test_aggregate_tier2_keys_and_goal_fraction():
    """Micro-run (2 seeds): aggregate dict has expected keys; goal_frac in [0,1]."""
    cfg = _micro_config()
    result = run_experiment(tier2_episode, cfg)
    agg = aggregate_tier2(result)

    # All expected keys present
    assert EXPECTED_KEYS == set(agg.keys()), (
        f"Missing keys: {EXPECTED_KEYS - set(agg.keys())}, "
        f"Extra keys: {set(agg.keys()) - EXPECTED_KEYS}"
    )

    # goal_frac is a valid probability
    gf = agg["goal_frac"]
    assert not math.isnan(gf), "goal_frac must not be NaN for a completed run"
    assert 0.0 <= gf <= 1.0, f"goal_frac={gf} not in [0, 1]"

    # n_episodes must match config
    assert agg["n_episodes"] == cfg.n_seeds, (
        f"Expected {cfg.n_seeds} episodes, got {agg['n_episodes']}"
    )

    # n_reached is consistent with goal_frac
    assert agg["n_reached"] == round(gf * agg["n_episodes"])


def test_aggregate_tier2_empty_episodes():
    """aggregate_tier2 on a result with no episodes returns nan-filled dict."""
    from certflow.harness import ExperimentResult

    empty_result = ExperimentResult(config=_micro_config())
    agg = aggregate_tier2(empty_result)
    assert set(agg.keys()) == EXPECTED_KEYS
    assert agg["n_episodes"] == 0
    assert math.isnan(agg["goal_frac"])


def test_base_config_fields():
    """BASE config has the required Tier-2 field values."""
    assert BASE.rows == 10
    assert BASE.cols == 10
    assert BASE.kind == "bounded"
    assert abs(BASE.rho - 0.02) < 1e-9
    assert BASE.noise_family == "gaussian"
    assert abs(BASE.noise_scale - 0.05) < 1e-9
    assert abs(BASE.epsilon - 8.0) < 1e-9
    assert abs(BASE.alpha_prime - 0.2) < 1e-9
    assert abs(BASE.eps_tv - 1e-4) < 1e-9
    assert abs(BASE.gamma_aci - 0.01) < 1e-9
    assert BASE.use_kappa is True
    assert BASE.initial_survey is False
    assert BASE.max_rounds == 600
    assert BASE.base_seed == 2026
