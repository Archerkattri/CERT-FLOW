"""Tier-2 comparison sweep: sensing policy x budget x move policy.

Spec section 7, Tier 2: unknown/drifting terrain, single robot.
The sweep compares certify-then-go (move_policy="when_certified") across sensing
policies (cert, random, max_age, max_width) at three budgets (10, 20, 40), plus
two baselines:
  - no-certificate: sensing_policy="none", move_policy="always"  (drives blind)
  - cert + always:  sensing_policy="cert",  move_policy="always"  (sense-while-driving)

Key claims the table must make visible:
  (a) at equal budget, cert sensing yields lower travel-regret than random/max_age/max_width
  (b) regret decreases with budget for cert
  (c) no-certificate baseline's regret (drives on prior)
  (d) coverage stays high during motion

Run:  cert_env/bin/python scripts/run_tier2.py [--quick]
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

from certflow.episodes import coverage_among_valid, oracle_walk_cost, tier2_episode
from certflow.harness import ExperimentConfig, run_experiment

QUICK = "--quick" in sys.argv

# ---------------------------------------------------------------------------
# Base configuration (Tier-2 spec §7)
# ---------------------------------------------------------------------------
BASE = ExperimentConfig(
    rows=10,
    cols=10,
    kind="bounded",
    rho=0.02,
    noise_family="gaussian",
    noise_scale=0.05,
    epsilon=8.0,
    alpha_prime=0.2,
    rho_w=0.99,   # conformal weight-decay: matches tier0/ablation scripts; rho=0.02 is world drift
    eps_tv=1e-4,
    gamma_aci=0.01,
    delta=1.0,
    rho_hat_over_rho=1.0,
    use_kappa=True,
    initial_survey=False,
    max_rounds=600,
    n_seeds=5 if QUICK else 25,
    base_seed=2026,
)

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
SENSING_POLICIES = ["cert", "random", "max_age", "max_width"]
BUDGETS = [10.0, 20.0, 40.0]


def build_conditions() -> list[tuple[str, ExperimentConfig]]:
    """Return (label, config) pairs for all conditions in the sweep."""
    conditions: list[tuple[str, ExperimentConfig]] = []

    # --- Certify-then-go: sensing_policy x budget ---
    for sp in SENSING_POLICIES:
        for budget in BUDGETS:
            label = f"cert-then-go | {sp:10s} | B={budget:.0f}"
            cfg = dataclasses.replace(
                BASE,
                sensing_policy=sp,
                move_policy="when_certified",
                sense_budget=budget,
            )
            conditions.append((label, cfg))

    # --- No-certificate baseline: drives immediately on prior, learns by traversal ---
    label = "no-cert baseline | none    | B=inf"
    cfg = dataclasses.replace(
        BASE,
        sensing_policy="none",
        move_policy="always",
        sense_budget=float("inf"),
    )
    conditions.append((label, cfg))

    # --- Cert + always (sense-while-driving) at infinite budget ---
    label = "cert+always      | cert    | B=inf"
    cfg = dataclasses.replace(
        BASE,
        sensing_policy="cert",
        move_policy="always",
        sense_budget=float("inf"),
    )
    conditions.append((label, cfg))

    return conditions


# ---------------------------------------------------------------------------
# Per-condition aggregation
# ---------------------------------------------------------------------------

def aggregate_tier2(result) -> dict:
    """Compute Tier-2-specific aggregates from an ExperimentResult."""
    episodes = result.episodes

    n_eps = len(episodes)
    if n_eps == 0:
        return {
            "goal_frac": float("nan"),
            "mission_rounds_mean": float("nan"),
            "regret_mean": float("nan"),
            "regret_median": float("nan"),
            "sense_spend_mean": float("nan"),
            "coverage_valid": float("nan"),
            "n_episodes": 0,
            "n_reached": 0,
        }

    # goal fraction
    n_reached = sum(1 for ep in episodes if ep.reached_goal)
    goal_frac = n_reached / n_eps

    # mission rounds and regret among reached episodes
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
        if mission_rounds_list
        else float("nan")
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

    # mean sense spend
    sense_spend_mean = (
        float(sum(ep.sense_cost for ep in episodes) / n_eps)
        if episodes
        else float("nan")
    )

    # coverage among valid rounds
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
# Main
# ---------------------------------------------------------------------------

def run_sweep() -> list[dict]:
    """Run the full Tier-2 sweep and return a list of row dicts."""
    conditions = build_conditions()
    rows: list[dict] = []

    for label, cfg in conditions:
        print(f"running: {label} ...", flush=True)
        result = run_experiment(tier2_episode, cfg)
        agg = aggregate_tier2(result)
        row = {"label": label, **agg}
        rows.append(row)
        goal_str = f"{100 * agg['goal_frac']:.0f}%" if not math.isnan(agg["goal_frac"]) else " nan"
        print(
            f"  done  goal={goal_str} regret_mean={agg['regret_mean']:.2f} "
            f"coverage={agg['coverage_valid']:.3f}",
            flush=True,
        )

    return rows


def print_table(rows: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    # Header
    hdr = (
        f"{'condition':42} {'goal%':>6} {'rounds':>7} "
        f"{'regret~':>8} {'regret|':>8} {'sense':>7} {'cov':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        def fmt(v, spec=".2f"):
            return f"{v:{spec}}" if not math.isnan(v) and not math.isinf(v) else "  nan"

        goal_str = f"{100 * r['goal_frac']:.0f}%" if not math.isnan(r["goal_frac"]) else " nan"
        print(
            f"{r['label']:42} {goal_str:>6} {fmt(r['mission_rounds_mean'], '7.1f')} "
            f"{fmt(r['regret_mean'], '8.2f')} {fmt(r['regret_median'], '8.2f')} "
            f"{fmt(r['sense_spend_mean'], '7.2f')} {fmt(r['coverage_valid'], '6.3f')}"
        )


def _sanitize_for_json(obj):
    """Recursively convert non-finite floats to sentinels for JSON."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "__nan__"
        if math.isinf(obj):
            return "__inf__" if obj > 0 else "__-inf__"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def main() -> None:
    outdir = Path("results/tier2")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = run_sweep()

    # Save JSON
    table_path = outdir / "table.json"
    table_path.write_text(json.dumps(_sanitize_for_json(rows), indent=2))
    print(f"\nSaved results to {table_path}", flush=True)

    print_table(rows)


if __name__ == "__main__":
    main()
