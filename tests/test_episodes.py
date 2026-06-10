"""Smoke tests for the Tier-0 episode runner + harness integration."""
from certflow.episodes import coverage_among_valid, tier0_episode
from certflow.harness import ExperimentConfig, run_experiment


def small_config(**kw) -> ExperimentConfig:
    base = dict(
        rows=4, cols=4, kind="bounded", rho=0.01,
        noise_family="gaussian", noise_scale=0.05,
        epsilon=4.0, alpha_prime=0.2, rho_w=0.99, eps_tv=0.001,
        gamma_aci=0.01, delta=1.0, rho_hat_over_rho=1.0,
        n_seeds=3, max_rounds=120, base_seed=11,
    )
    base.update(kw)
    return ExperimentConfig(**base)


def test_tier0_episode_produces_rounds_and_valid_coverage():
    res = tier0_episode(small_config(), seed=3)
    assert len(res.rounds) == 120
    assert res.sense_cost > 0
    cov, valid = coverage_among_valid(res)
    assert valid > 10           # warm-up must end within the horizon
    assert cov / valid >= 0.8   # loose smoke bound


def test_harness_integration_aggregates():
    cfg = small_config(max_rounds=60)
    result = run_experiment(tier0_episode, cfg)
    agg = result.aggregate()
    assert agg["n_rounds_total"] == 3 * 60
    assert agg["failure_count"] == 0
    assert agg["sense_spend_total"] > 0


def test_off_model_worlds_run():
    for kind in ("static", "jump", "periodic"):
        res = tier0_episode(small_config(kind=kind, max_rounds=50), seed=1)
        assert len(res.rounds) == 50
