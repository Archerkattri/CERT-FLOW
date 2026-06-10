"""MovingAI benchmark experiment for CERT.

Three representative map environments:
  1. DAO dungeon: data/movingai/dao/arena.map  (49x49 full map, scen endpoints)
  2. Street:      64x64 crop of Berlin_0_256.map  (centre at row=128, col=128)
  3. Maze:        64x64 crop of maze512-1-0.map   (centre at row=320, col=256)

Policies compared per map:
  - cert      (certify-then-go, cert sensing)
  - random    (certify-then-go, random sensing)
  - max_age   (certify-then-go, max_age sensing)
  - blind     (no certificate, always move: no-cert baseline)

Episode semantics: moving-robot tier-2.
  certify-then-go: move only when epsilon-certified or sense_budget=20 exhausted.
  blind baseline:  sensing_policy="none", move_policy="always".

Run:
  cert_env/bin/python scripts/run_movingai.py [--quick]
  --quick: 4 seeds, 150 rounds (smoke test)
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from certflow.cert import CertPlanner, PlannerConfig
from certflow.episodes import coverage_among_valid, oracle_walk_cost
from certflow.graphcore import dijkstra
from certflow.harness import spawn_seeds
from certflow.movingai import (
    MovingAIBoundedDriftWorld,
    _build_graph_from_grid,
    crop,
    movingai_world_from_grid,
    parse_map,
    parse_scen,
    scenario_endpoints,
)
from certflow.oracle import opt
from certflow.types import EpisodeResult, RoundLog

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
QUICK = "--quick" in sys.argv
N_SEEDS = 4 if QUICK else 15
MAX_ROUNDS = 150 if QUICK else 400

# ---------------------------------------------------------------------------
# Experiment hyperparameters (matching tier2_episode conventions)
# ---------------------------------------------------------------------------
EPSILON = 8.0
ALPHA_PRIME = 0.2
RHO_W = 0.99
EPS_TV = 1e-4
DELTA = 1.0
USE_KAPPA = True
RHO = 0.02
NOISE_FAMILY = "gaussian"
NOISE_SCALE = 0.05
# sense_budget: sense_cost=0.1 per obs, budget=20 -> fallback fires at round 200.
# In quick mode (max_rounds=150) the full budget never fires; use 4.0 (fires at
# round 40) so there are >=100 rounds remaining for path traversal (~87 steps).
SENSE_BUDGET = 20.0 if not QUICK else 4.0
BASE_SEED = 3141

# ---------------------------------------------------------------------------
# Map definitions
# ---------------------------------------------------------------------------

_DATA = Path(__file__).resolve().parent.parent / "data" / "movingai"


def _load_dao_arena():
    """DAO arena.map — full 49x49 map, use scenario endpoints (min_length>=40)."""
    map_path = _DATA / "dao" / "arena.map"
    scen_path = _DATA / "dao" / "arena.map.scen"
    grid = parse_map(map_path)
    scen = parse_scen(scen_path)
    eps = scenario_endpoints(scen, grid, min_length=40.0)
    if not eps:
        raise RuntimeError("No arena.map scenario endpoints with min_length>=40")
    # Use first available endpoint pair (already verified passable by scenario_endpoints)
    start, goal = eps[0]
    return grid, start, goal


def _load_street_berlin():
    """64x64 crop of Berlin_0_256.map, centre (128,128); connected far endpoints."""
    map_path = _DATA / "street" / "Berlin_0_256.map"
    full_grid = parse_map(map_path)
    sub, r0, c0 = crop(full_grid, 128, 128, 64)

    # Find largest connected component, then pick far endpoint pair.
    G, _, _, _ = _build_graph_from_grid(sub)
    nodes = sorted(G.keys())

    visited: set = set()
    comps: list[frozenset] = []
    for s in nodes:
        if s in visited:
            continue
        comp: set = set()
        q = [s]
        while q:
            u = q.pop()
            if u in comp:
                continue
            comp.add(u)
            for v in G.get(u, {}):
                if v not in comp:
                    q.append(v)
        visited.update(comp)
        comps.append(frozenset(comp))

    biggest = max(comps, key=len)
    bc = sorted(biggest)
    start = bc[0]  # top-left-most
    # Farthest reachable node (verified connected above)
    goal = max(biggest, key=lambda n: abs(n[0] - start[0]) + abs(n[1] - start[1]))

    # Double-check path exists
    path, cost = dijkstra(G, start, goal)
    if path is None or math.isinf(cost):
        raise RuntimeError(f"Berlin crop: no path from {start} to {goal}")

    return sub, start, goal


def _load_maze():
    """64x64 crop of maze512-1-0.map, centre (384,128); connected far endpoints.

    Pre-identified crop: centre=(384,128), path_len=87 steps, component size=723.
    Verified by exploration: dijkstra confirms connectivity.
    """
    map_path = _DATA / "maze" / "maze512-1-0.map"
    full_grid = parse_map(map_path)
    sub, r0, c0 = crop(full_grid, 384, 128, 64)

    # Pre-identified connected endpoints (verified by exploration):
    start, goal = (15, 45), (45, 9)
    G, _, _, _ = _build_graph_from_grid(sub)
    path, cost = dijkstra(G, start, goal)
    if path is None or math.isinf(cost):
        raise RuntimeError(f"Maze crop: no path from {start} to {goal}")

    return sub, start, goal


# ---------------------------------------------------------------------------
# Per-map configuration
# ---------------------------------------------------------------------------

MAP_CONFIGS = [
    ("dao_arena",      _load_dao_arena),
    ("street_berlin",  _load_street_berlin),
    ("maze",           _load_maze),
]

# ---------------------------------------------------------------------------
# Local episode function (modelled on episodes.tier2_episode but uses
# movingai_world_from_grid instead of grid_world)
# ---------------------------------------------------------------------------

def _make_planner_config(sensing_policy: str) -> PlannerConfig:
    return PlannerConfig(
        epsilon=EPSILON,
        alpha_prime=ALPHA_PRIME,
        rho_w=RHO_W,
        eps_tv=EPS_TV,
        delta=DELTA,
        use_kappa=USE_KAPPA,
        sensing_policy=sensing_policy,
        initial_survey=False,   # unknown-terrain start (Tier-2 style)
        use_aci=True,
        sum_aware_ub=False,
    )


def movingai_episode(
    grid: list[str],
    start,
    goal,
    seed: int,
    sensing_policy: str,
    move_policy: str,
    sense_budget: float,
    max_rounds: int,
) -> EpisodeResult:
    """One moving-robot episode on a MovingAI grid world.

    certify-then-go semantics:
      - If move_policy == "when_certified": move only when epsilon-certified
        or sense_budget exhausted.
      - If move_policy == "always": move every round (blind baseline).
    Traversal is a free observation (matching tier2_episode).
    oracle_cost is computed at t=0 from start using oracle_walk_cost.
    """
    world = movingai_world_from_grid(
        grid,
        seed=seed,
        kind="bounded",
        rho=RHO,
        noise_family=NOISE_FAMILY,
        noise_scale=NOISE_SCALE,
    )
    cfg = _make_planner_config(sensing_policy)
    planner = CertPlanner(world, start, goal, cfg)

    result = EpisodeResult()
    pos = start
    prev_spend = 0.0

    for _ in range(max_rounds):
        t_round = planner.t
        cert, sensed = planner.round()

        _, true_opt = opt(world, t_round, pos, goal)
        result.rounds.append(
            RoundLog(
                t=t_round,
                lb=cert.lb,
                ub=cert.ub,
                confidence=cert.confidence,
                opt=true_opt,
                covered=bool(cert.valid and cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9),
                certified=bool(cert.valid and cert.gap <= EPSILON),
                sensed_edge=sensed,
                sense_spend=planner.sense_spend - prev_spend,
                replan_seconds=0.0,
            )
        )
        prev_spend = planner.sense_spend

        certified_now = bool(cert.valid and cert.gap <= EPSILON)
        may_move = (
            move_policy == "always"
            or certified_now
            or planner.sense_spend >= sense_budget
        )
        if may_move and cert.path and len(cert.path) >= 2 and cert.path[0] == pos:
            e = (cert.path[0], cert.path[1])
            result.travel_cost += world.true_cost(e, planner.t)
            planner.ingest_observation(e)
            pos = cert.path[1]
            planner.advance_start(pos)
            if pos == goal:
                result.reached_goal = True
                break

    result.sense_cost = planner.sense_spend
    result.oracle_cost = oracle_walk_cost(world, start, goal, 0.0, DELTA)
    return result


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

POLICIES = [
    ("cert",     "cert",   "when_certified", SENSE_BUDGET),
    ("random",   "random", "when_certified", SENSE_BUDGET),
    ("max_age",  "max_age","when_certified", SENSE_BUDGET),
    ("blind",    "none",   "always",         float("inf")),
]

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(episodes: list[EpisodeResult]) -> dict:
    n_eps = len(episodes)
    if n_eps == 0:
        return {
            "goal_frac": float("nan"), "mission_rounds_mean": float("nan"),
            "regret_mean": float("nan"), "regret_median": float("nan"),
            "sense_spend_mean": float("nan"), "coverage_valid": float("nan"),
            "n_episodes": 0, "n_reached": 0,
        }

    n_reached = sum(1 for ep in episodes if ep.reached_goal)
    goal_frac = n_reached / n_eps

    mission_rounds_list: list[float] = []
    regret_list: list[float] = []
    for ep in episodes:
        if ep.reached_goal:
            mission_rounds_list.append(float(len(ep.rounds)))
            if not math.isnan(ep.oracle_cost) and not math.isinf(ep.oracle_cost):
                regret = ep.travel_cost - ep.oracle_cost
                regret_list.append(regret)

    mission_rounds_mean = (
        float(sum(mission_rounds_list) / len(mission_rounds_list))
        if mission_rounds_list else float("nan")
    )
    if regret_list:
        regret_mean = float(sum(regret_list) / len(regret_list))
        sorted_r = sorted(regret_list)
        mid = len(sorted_r) // 2
        if len(sorted_r) % 2 == 0:
            regret_median = (sorted_r[mid - 1] + sorted_r[mid]) / 2.0
        else:
            regret_median = float(sorted_r[mid])
    else:
        regret_mean = float("nan")
        regret_median = float("nan")

    sense_spend_mean = float(sum(ep.sense_cost for ep in episodes) / n_eps)

    total_covered = total_valid = 0
    for ep in episodes:
        cov, val = coverage_among_valid(ep)
        total_covered += cov
        total_valid += val
    coverage_valid = (total_covered / total_valid) if total_valid > 0 else float("nan")

    return {
        "goal_frac": goal_frac,
        "mission_rounds_mean": mission_rounds_mean,
        "regret_mean": regret_mean,
        "regret_median": regret_median,
        "sense_spend_mean": sense_spend_mean,
        "coverage_valid": coverage_valid,
        "n_episodes": n_eps,
        "n_reached": n_reached,
    }


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'map':18} {'policy':8} {'goal%':>6} {'rounds':>7} "
        f"{'regret~':>8} {'regret|':>8} {'sense':>7} {'cov':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        def fmt(v, spec=".2f"):
            return f"{v:{spec}}" if (not math.isnan(v) and not math.isinf(v)) else "  nan"

        goal_str = f"{100 * r['goal_frac']:.0f}%" if not math.isnan(r["goal_frac"]) else " nan"
        print(
            f"{r['map']:18} {r['policy']:8} {goal_str:>6} {fmt(r['mission_rounds_mean'], '7.1f')} "
            f"{fmt(r['regret_mean'], '8.2f')} {fmt(r['regret_median'], '8.2f')} "
            f"{fmt(r['sense_spend_mean'], '7.2f')} {fmt(r['coverage_valid'], '6.3f')}"
        )


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__nan__"
        if math.isinf(obj):
            return "__inf__" if obj > 0 else "__-inf__"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all() -> list[dict]:
    seeds = spawn_seeds(BASE_SEED, N_SEEDS)
    rows: list[dict] = []

    for map_name, loader in MAP_CONFIGS:
        print(f"\n=== Map: {map_name} ===", flush=True)
        grid, start, goal = loader()
        print(f"  grid: {len(grid)}x{len(grid[0])}, start={start}, goal={goal}", flush=True)

        for pol_name, sensing_pol, move_pol, budget in POLICIES:
            print(f"  policy: {pol_name} ...", flush=True)
            episodes: list[EpisodeResult] = []
            t0 = time.perf_counter()
            for seed in seeds:
                ep = movingai_episode(
                    grid, start, goal,
                    seed=seed,
                    sensing_policy=sensing_pol,
                    move_policy=move_pol,
                    sense_budget=budget,
                    max_rounds=MAX_ROUNDS,
                )
                episodes.append(ep)
            elapsed = time.perf_counter() - t0

            agg = aggregate(episodes)
            row = {"map": map_name, "policy": pol_name, **agg}
            rows.append(row)

            goal_str = f"{100 * agg['goal_frac']:.0f}%" if not math.isnan(agg["goal_frac"]) else "nan"
            print(
                f"    done ({elapsed:.1f}s)  goal={goal_str} "
                f"regret_mean={agg['regret_mean']:.2f} "
                f"coverage={agg['coverage_valid']:.3f}",
                flush=True,
            )

    return rows


def main() -> None:
    outdir = Path("results/movingai")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = run_all()

    # Save JSON
    table_path = outdir / "table.json"
    table_path.write_text(json.dumps(_sanitize(rows), indent=2))
    print(f"\nSaved results to {table_path}", flush=True)

    print_table(rows)


if __name__ == "__main__":
    main()
