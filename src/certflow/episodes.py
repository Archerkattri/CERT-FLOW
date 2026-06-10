"""Tier-0 episode runner: certification-only episodes for coverage validation.

Spec section 7, Tier 0: the robot is stationary; each round the planner emits a
certificate and (unless certified) one sensing action. The oracle records
ground truth so the harness can validate empirical coverage against the
claimed level. Coverage among VALID rounds is the paper metric; warm-up
rounds log covered=False with confidence<=0 and are filtered by the analysis.
"""
from __future__ import annotations

import time

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.harness import ExperimentConfig
from certflow.oracle import opt
from certflow.types import EpisodeResult, RoundLog


def planner_config(config: ExperimentConfig) -> PlannerConfig:
    return PlannerConfig(
        epsilon=config.epsilon,
        alpha_prime=config.alpha_prime,
        rho_w=config.rho_w,
        eps_tv=config.eps_tv,
        gamma_aci=config.gamma_aci,
        delta=config.delta,
        rho_hat_over_rho=config.rho_hat_over_rho,
        use_kappa=config.use_kappa,
        sensing_policy=config.sensing_policy,
        initial_survey=config.initial_survey,
        latent_margin=config.latent_margin,
        thinned_scores=config.thinned_scores,
        use_aci=config.use_aci,
        sum_aware_ub=config.sum_aware_ub,
        hybrid_sensing=config.hybrid_sensing,
        rho_mode=config.rho_mode,
        adaptive_rate=config.adaptive_rate,
        decision_uniform=config.decision_uniform,
    )


def tier0_episode(config: ExperimentConfig, seed: int) -> EpisodeResult:
    """One certification episode. config.kind uses drift.py names
    ("static", "bounded", "jump", "periodic")."""
    world_kwargs: dict = {
        "noise_family": config.noise_family,
        "noise_scale": config.noise_scale,
    }
    if config.kind == "bounded":
        world_kwargs["rho"] = config.rho
    world = grid_world(config.rows, config.cols, seed=seed, kind=config.kind, **world_kwargs)

    start, goal = (0, 0), (config.rows - 1, config.cols - 1)
    planner = CertPlanner(world, start, goal, planner_config(config))

    result = EpisodeResult()
    prev_spend = planner.sense_spend
    for _ in range(config.max_rounds):
        t_round = planner.t
        wall = time.perf_counter()
        cert, sensed = planner.round()
        wall = time.perf_counter() - wall

        _, true_opt = opt(world, t_round, start, goal)
        covered = bool(
            cert.valid and cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9
        )
        certified = bool(cert.valid and cert.gap <= config.epsilon)
        result.rounds.append(
            RoundLog(
                t=t_round,
                lb=cert.lb,
                ub=cert.ub,
                confidence=cert.confidence,
                opt=true_opt,
                covered=covered,
                certified=certified,
                sensed_edge=sensed,
                sense_spend=planner.sense_spend - prev_spend,
                replan_seconds=wall,
            )
        )
        prev_spend = planner.sense_spend

    result.sense_cost = planner.sense_spend
    result.reached_goal = False  # Tier 0 is stationary by design
    return result


def coverage_among_valid(result: EpisodeResult) -> tuple[int, int]:
    """(covered, valid) counts over rounds with a valid certificate."""
    valid = [r for r in result.rounds if r.confidence > 0.0]
    return sum(r.covered for r in valid), len(valid)


def oracle_walk_cost(world, start, goal, t0: float, delta: float) -> float:
    """Clairvoyant robot: replans on TRUE costs every step, moving one edge
    per period delta. The Tier-2 regret reference."""
    pos, t, total = start, t0, 0.0
    for _ in range(100_000):
        if pos == goal:
            return total
        path, _ = opt(world, t, pos, goal)
        if path is None or len(path) < 2:
            return float("inf")
        e = (path[0], path[1])
        total += world.true_cost(e, t)
        pos = path[1]
        t += delta
    return float("inf")


def tier2_episode(config: ExperimentConfig, seed: int) -> EpisodeResult:
    """Moving-robot episode: each round the planner certifies and senses
    (per its sensing_policy), then the robot traverses one edge of the
    incumbent, paying the TRUE cost and observing the traversed edge for
    free. Ends at goal or max_rounds. config.kind uses drift.py names;
    config metadata key "sensing_policy" selects the policy (default cert)."""
    world_kwargs: dict = {
        "noise_family": config.noise_family,
        "noise_scale": config.noise_scale,
    }
    if config.kind == "bounded":
        world_kwargs["rho"] = config.rho
    world = grid_world(config.rows, config.cols, seed=seed, kind=config.kind, **world_kwargs)

    start, goal = (0, 0), (config.rows - 1, config.cols - 1)
    planner = CertPlanner(world, start, goal, planner_config(config))

    result = EpisodeResult()
    pos = start
    prev_spend = 0.0
    for _ in range(config.max_rounds):
        t_round = planner.t
        wall = time.perf_counter()
        cert, sensed = planner.round()
        wall = time.perf_counter() - wall

        _, true_opt = opt(world, t_round, pos, goal)
        result.rounds.append(
            RoundLog(
                t=t_round,
                lb=cert.lb,
                ub=cert.ub,
                confidence=cert.confidence,
                opt=true_opt,
                covered=bool(cert.valid and cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9),
                certified=bool(cert.valid and cert.gap <= config.epsilon),
                sensed_edge=sensed,
                sense_spend=planner.sense_spend - prev_spend,
                replan_seconds=wall,
            )
        )
        prev_spend = planner.sense_spend

        certified_now = bool(cert.valid and cert.gap <= config.epsilon)
        may_move = (
            config.move_policy == "always"
            or certified_now
            or planner.sense_spend >= config.sense_budget  # deadline fallback
        )
        if may_move and cert.path and len(cert.path) >= 2 and cert.path[0] == pos:
            e = (cert.path[0], cert.path[1])
            # pay true cost at traversal time; the move is also a free look
            result.travel_cost += world.true_cost(e, planner.t)
            planner.ingest_observation(e)
            pos = cert.path[1]
            planner.advance_start(pos)
            if pos == goal:
                result.reached_goal = True
                break

    result.sense_cost = planner.sense_spend
    result.oracle_cost = oracle_walk_cost(world, start, goal, 0.0, config.delta)
    return result
