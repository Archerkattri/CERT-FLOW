"""Experiment configuration, seed management, orchestration, and metrics aggregation.

This module provides the harness layer described in spec §6 / §7. It does NOT
implement the planner episode loop (that lives in cert.py). It provides:

- ExperimentConfig: world + planner + run parameters with stable config_id().
- spawn_seeds: reproducible independent streams via numpy SeedSequence.
- run_experiment: multi-episode orchestration with exception isolation.
- ExperimentResult: aggregation (coverage, Clopper–Pearson CI, gap stats, latency).
- save_results / load_results: JSON round-trip with inf/nan sentinel handling.
- sweep: cartesian product over a param_grid.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import beta as beta_dist

from certflow.types import Edge, EpisodeResult, RoundLog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel strings for JSON serialisation of non-finite floats
# ---------------------------------------------------------------------------
_POS_INF_SENTINEL = "__inf__"
_NEG_INF_SENTINEL = "__-inf__"
_NAN_SENTINEL = "__nan__"


def _float_to_json(v: float) -> Any:
    """Convert a float to a JSON-safe value, using string sentinels for non-finite."""
    if isinstance(v, float):
        if math.isnan(v):
            return _NAN_SENTINEL
        if math.isinf(v):
            return _POS_INF_SENTINEL if v > 0 else _NEG_INF_SENTINEL
    return v


def _float_from_json(v: Any) -> Any:
    """Invert _float_to_json sentinel strings back to floats."""
    if v == _POS_INF_SENTINEL:
        return math.inf
    if v == _NEG_INF_SENTINEL:
        return -math.inf
    if v == _NAN_SENTINEL:
        return math.nan
    return v


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All knobs for one experiment sweep point.

    World params
    ------------
    rows, cols : grid dimensions
    kind : world/drift kind string (e.g. "grid_bounded_drift")
    rho : drift severity (A1 bound)
    noise_family : observation noise distribution name (e.g. "gaussian", "laplace")
    noise_scale : scale parameter for the noise distribution

    Planner params
    --------------
    epsilon : certificate gap target (UB − LB ≤ epsilon ⇒ stop sensing)
    alpha_prime : per-edge miscoverage budget α′
    rho_w : conformal calibration weight decay per unit time (NOT a drift rate)
    eps_tv : total-variation Lipschitz budget ε_TV (A2)
    gamma_aci : ACI step-size γ for online α-tracker
    delta : sensing period (seconds)
    rho_hat_over_rho : misspecification factor ρ̂/ρ (1.0 = correctly specified)
    use_kappa : enable κ kill-switch heuristic

    Run params
    ----------
    n_seeds : number of independent replications
    max_rounds : episode time horizon
    base_seed : root for SeedSequence
    """

    # world
    rows: int = 10
    cols: int = 10
    kind: str = "grid_bounded_drift"
    rho: float = 0.01
    noise_family: str = "gaussian"
    noise_scale: float = 0.1

    # planner
    epsilon: float = 0.5
    alpha_prime: float = 0.1
    rho_w: float = 0.99
    eps_tv: float = 0.05
    gamma_aci: float = 0.05
    delta: float = 1.0
    rho_hat_over_rho: float = 1.0
    use_kappa: bool = False
    sensing_policy: str = "cert"  # Tier-2: cert | random | max_age | max_width | none
    initial_survey: bool = True   # False = unknown-terrain start (Tier-2)
    move_policy: str = "always"   # Tier-2: always | when_certified (certify-then-go)
    sense_budget: float = float("inf")  # spend cap; exhausted -> depart anyway
    latent_margin: float = 1.0    # lambda: 1.0 = T1a semantics, 2.0 = provable T1b
    thinned_scores: bool = False  # disjoint-pair calibration (provable mode)
    use_aci: bool = True          # False = freeze working alpha (provable mode)
    sum_aware_ub: bool = False    # T4 block-quantile UB margin (theory.tex)
    hybrid_sensing: bool = False  # objective-matched sensing (VOI when eps unattainable)
    rho_mode: str = "given"       # "online" = estimate drift from observed rates
    adaptive_rate: bool = False   # T2'-derived k observations per round
    decision_uniform: bool = False  # alpha-spending over decision instants (T6)

    # run
    n_seeds: int = 10
    max_rounds: int = 200
    base_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of all fields (JSON-serialisable)."""
        return dataclasses.asdict(self)

    def config_id(self) -> str:
        """Stable 12-hex-char hash of the config for output naming.

        Only the content matters; the hash is reproducible across runs as long
        as the field values are unchanged. Python's built-in hash() is NOT used
        because it is randomised per-process.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def spawn_seeds(base_seed: int, n: int) -> list[int]:
    """Return n independent integer seeds derived from base_seed.

    Uses numpy.random.SeedSequence so the child streams are statistically
    independent and the sequence is fully reproducible from base_seed.
    """
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(n)
    # Take the first 32-bit integer from each child's generated state.
    return [int(child.generate_state(1)[0]) for child in children]


# ---------------------------------------------------------------------------
# ExperimentResult and aggregation
# ---------------------------------------------------------------------------

@dataclass
class _FailureRecord:
    """Lightweight record for an episode that raised an exception."""
    seed: int
    exc_type: str
    exc_message: str


@dataclass
class ExperimentResult:
    """Collects outcomes from all seeds of a single ExperimentConfig."""

    config: ExperimentConfig
    episodes: list[EpisodeResult] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    failures: list[_FailureRecord] = field(default_factory=list)

    def aggregate(self) -> dict[str, Any]:
        """Compute aggregate statistics across all rounds of all seeds.

        Returns a flat dict with the following keys:

        Coverage
        --------
        n_rounds_total        total rounds across all seeds
        n_covered             rounds where lb <= opt <= ub
        coverage              empirical fraction
        coverage_ci_lo        Clopper–Pearson 95% CI lower bound
        coverage_ci_hi        Clopper–Pearson 95% CI upper bound

        Certificate gap
        ---------------
        gap_mean              mean of (ub - lb) per round
        gap_median            median of (ub - lb) per round

        Certification
        -------------
        certified_fraction    fraction of rounds where certified == True
        time_to_first_cert_mean   mean rounds until first certified==True per episode (nan if never)
        time_to_first_cert_median median ditto

        Sensing
        -------
        sense_spend_total     sum of sense_spend over all rounds
        sense_spend_mean      per-round mean
        replan_latency_p50    50th percentile of replan_seconds
        replan_latency_p95    95th percentile of replan_seconds

        Episode-level
        -------------
        travel_cost_mean      mean EpisodeResult.travel_cost
        goal_reached_fraction fraction of episodes where reached_goal == True
        failure_count         number of episodes that raised an exception
        """
        all_rounds: list[RoundLog] = []
        for ep in self.episodes:
            all_rounds.extend(ep.rounds)

        n = len(all_rounds)

        if n == 0:
            return {
                "n_rounds_total": 0,
                "n_covered": 0,
                "coverage": math.nan,
                "coverage_ci_lo": math.nan,
                "coverage_ci_hi": math.nan,
                "gap_mean": math.nan,
                "gap_median": math.nan,
                "certified_fraction": math.nan,
                "time_to_first_cert_mean": math.nan,
                "time_to_first_cert_median": math.nan,
                "sense_spend_total": 0.0,
                "sense_spend_mean": math.nan,
                "replan_latency_p50": math.nan,
                "replan_latency_p95": math.nan,
                "travel_cost_mean": math.nan,
                "goal_reached_fraction": math.nan,
                "failure_count": len(self.failures),
            }

        # Coverage
        n_covered = sum(1 for r in all_rounds if r.covered)
        coverage = n_covered / n
        # Clopper–Pearson 95% CI via beta quantiles
        alpha_cp = 0.05
        if n_covered == 0:
            ci_lo = 0.0
        else:
            ci_lo = float(beta_dist.ppf(alpha_cp / 2, n_covered, n - n_covered + 1))
        if n_covered == n:
            ci_hi = 1.0
        else:
            ci_hi = float(beta_dist.ppf(1 - alpha_cp / 2, n_covered + 1, n - n_covered))

        # Certificate gap
        gaps = [r.ub - r.lb for r in all_rounds]
        gap_mean = float(np.mean(gaps))
        gap_median = float(np.median(gaps))

        # Certified fraction
        n_certified = sum(1 for r in all_rounds if r.certified)
        certified_fraction = n_certified / n

        # Time-to-first-certified per episode (in rounds, 0-indexed round number)
        ttfc: list[float] = []
        for ep in self.episodes:
            found = False
            for idx, r in enumerate(ep.rounds):
                if r.certified:
                    ttfc.append(float(idx))
                    found = True
                    break
            if not found:
                ttfc.append(math.nan)

        finite_ttfc = [v for v in ttfc if not math.isnan(v)]
        if finite_ttfc:
            ttfc_mean = float(np.mean(finite_ttfc))
            ttfc_median = float(np.median(finite_ttfc))
        else:
            ttfc_mean = math.nan
            ttfc_median = math.nan

        # Sensing
        sense_spends = [r.sense_spend for r in all_rounds]
        sense_spend_total = float(np.sum(sense_spends))
        sense_spend_mean = float(np.mean(sense_spends))

        # Replan latency
        replan_seconds = [r.replan_seconds for r in all_rounds]
        replan_p50 = float(np.percentile(replan_seconds, 50))
        replan_p95 = float(np.percentile(replan_seconds, 95))

        # Episode-level
        travel_cost_mean = float(np.mean([ep.travel_cost for ep in self.episodes])) if self.episodes else math.nan
        goal_reached_fraction = (
            float(np.mean([1.0 if ep.reached_goal else 0.0 for ep in self.episodes]))
            if self.episodes
            else math.nan
        )

        return {
            "n_rounds_total": n,
            "n_covered": n_covered,
            "coverage": coverage,
            "coverage_ci_lo": ci_lo,
            "coverage_ci_hi": ci_hi,
            "gap_mean": gap_mean,
            "gap_median": gap_median,
            "certified_fraction": certified_fraction,
            "time_to_first_cert_mean": ttfc_mean,
            "time_to_first_cert_median": ttfc_median,
            "sense_spend_total": sense_spend_total,
            "sense_spend_mean": sense_spend_mean,
            "replan_latency_p50": replan_p50,
            "replan_latency_p95": replan_p95,
            "travel_cost_mean": travel_cost_mean,
            "goal_reached_fraction": goal_reached_fraction,
            "failure_count": len(self.failures),
        }


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------

def run_experiment(
    episode_fn: Callable[[ExperimentConfig, int], EpisodeResult],
    config: ExperimentConfig,
    workers: int | None = None,
) -> ExperimentResult:
    """Run n_seeds episodes; catch per-episode exceptions gracefully.

    episode_fn must have signature (config: ExperimentConfig, seed: int) ->
    EpisodeResult and be importable at module top level (picklable) when
    workers > 1. Seeds are embarrassingly parallel and episodes are
    independent by construction, so results are identical to the sequential
    run up to list order (we re-sort by seed for determinism).

    workers: None -> CERTFLOW_WORKERS env var, else 1 (sequential).
    Per-episode wall-clock (replan_seconds) is still measured inside each
    process with perf_counter; parallelism affects only sweep wall-time.
    """
    import os

    if workers is None:
        workers = int(os.environ.get("CERTFLOW_WORKERS", "1"))
    seeds = spawn_seeds(config.base_seed, config.n_seeds)
    result = ExperimentResult(config=config, seeds=seeds)

    def _record(seed: int, outcome, error) -> None:
        if error is None:
            result.episodes.append(outcome)
        else:
            logger.warning("Episode with seed=%d raised %s: %s",
                           seed, type(error).__name__, error)
            result.failures.append(_FailureRecord(
                seed=seed, exc_type=type(error).__name__,
                exc_message=str(error)))

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(episode_fn, config, seed): seed for seed in seeds}
            done = []
            for fut, seed in futs.items():
                try:
                    done.append((seed, fut.result(), None))
                except Exception as exc:  # noqa: BLE001
                    done.append((seed, None, exc))
            for seed, ep, err in sorted(done, key=lambda x: seeds.index(x[0])):
                _record(seed, ep, err)
    else:
        for seed in seeds:
            try:
                _record(seed, episode_fn(config, seed), None)
            except Exception as exc:  # noqa: BLE001
                _record(seed, None, exc)

    return result


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _round_log_to_dict(r: RoundLog) -> dict[str, Any]:
    """Serialise a RoundLog to a JSON-safe dict.

    Edge tuples (u, v) are stored as lists (JSON arrays). Non-finite floats
    use string sentinels.
    """
    d = dataclasses.asdict(r)
    # sensed_edge is either None or a tuple of two nodes; asdict converts tuple
    # to list, which is already JSON-safe for simple hashable nodes. Keep as-is.
    for key in ("t", "lb", "ub", "confidence", "opt", "sense_spend", "replan_seconds"):
        if key in d:
            d[key] = _float_to_json(d[key])
    return d


def _round_log_from_dict(d: dict[str, Any]) -> RoundLog:
    """Deserialise a RoundLog from the dict produced by _round_log_to_dict."""
    for key in ("t", "lb", "ub", "confidence", "opt", "sense_spend", "replan_seconds"):
        if key in d:
            d[key] = _float_from_json(d[key])
    # Restore sensed_edge: JSON lists -> tuples (or None)
    se = d.get("sensed_edge")
    if se is not None:
        d["sensed_edge"] = tuple(
            tuple(n) if isinstance(n, list) else n for n in se
        )
    return RoundLog(**d)


def _episode_result_to_dict(ep: EpisodeResult) -> dict[str, Any]:
    return {
        "rounds": [_round_log_to_dict(r) for r in ep.rounds],
        "travel_cost": _float_to_json(ep.travel_cost),
        "sense_cost": _float_to_json(ep.sense_cost),
        "reached_goal": ep.reached_goal,
    }


def _episode_result_from_dict(d: dict[str, Any]) -> EpisodeResult:
    return EpisodeResult(
        rounds=[_round_log_from_dict(r) for r in d.get("rounds", [])],
        travel_cost=float(_float_from_json(d["travel_cost"])),
        sense_cost=float(_float_from_json(d["sense_cost"])),
        reached_goal=bool(d["reached_goal"]),
    )


def _failure_record_to_dict(f: _FailureRecord) -> dict[str, Any]:
    return dataclasses.asdict(f)


def _failure_record_from_dict(d: dict[str, Any]) -> _FailureRecord:
    return _FailureRecord(**d)


def _experiment_result_to_dict(result: ExperimentResult) -> dict[str, Any]:
    return {
        "config": result.config.to_dict(),
        "config_id": result.config.config_id(),
        "seeds": result.seeds,
        "episodes": [_episode_result_to_dict(ep) for ep in result.episodes],
        "failures": [_failure_record_to_dict(f) for f in result.failures],
        "aggregate": {
            k: _float_to_json(v) for k, v in result.aggregate().items()
        },
    }


def _experiment_result_from_dict(d: dict[str, Any]) -> ExperimentResult:
    config = ExperimentConfig(**d["config"])
    episodes = [_episode_result_from_dict(e) for e in d.get("episodes", [])]
    failures = [_failure_record_from_dict(f) for f in d.get("failures", [])]
    result = ExperimentResult(
        config=config,
        episodes=episodes,
        seeds=d.get("seeds", []),
        failures=failures,
    )
    return result


# ---------------------------------------------------------------------------
# save_results / load_results
# ---------------------------------------------------------------------------

def save_results(result: ExperimentResult, directory: str | os.PathLike) -> Path:
    """Serialise result to JSON in *directory*.

    The filename is ``<config_id>.json`` so different configs never collide.
    Returns the path of the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.config.config_id()}.json"
    payload = _experiment_result_to_dict(result)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_results(path: str | os.PathLike) -> ExperimentResult:
    """Deserialise an ExperimentResult from the JSON file at *path*."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    return _experiment_result_from_dict(d)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def sweep(
    episode_fn: Callable[[ExperimentConfig, int], EpisodeResult],
    base_config: ExperimentConfig,
    param_grid: dict[str, list],
) -> list[ExperimentResult]:
    """Run the cartesian product of param_grid values, each via run_experiment.

    Each point in the grid is produced by replacing the corresponding fields of
    base_config using dataclasses.replace. The order of results matches
    itertools.product order (first key varies slowest).

    Example
    -------
    sweep(fn, cfg, {"rho": [0.0, 0.01, 0.1], "epsilon": [0.5, 1.0]})
    → 6 ExperimentResults.
    """
    if not param_grid:
        return [run_experiment(episode_fn, base_config)]

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    results: list[ExperimentResult] = []
    for combo in itertools.product(*value_lists):
        overrides = dict(zip(keys, combo))
        config = dataclasses.replace(base_config, **overrides)
        results.append(run_experiment(episode_fn, config))

    return results
