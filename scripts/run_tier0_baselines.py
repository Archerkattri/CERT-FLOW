"""Tier-0 coverage comparison: CERT vs Gaussian mu±beta*sigma baseline.

Compares conformal CERT (tier0_episode) against the Gaussian baseline
(gaussian_tier0_episode) on four conditions:

  1. bounded rho=0.02 + gaussian noise
  2. bounded rho=0.02 + student_t noise
  3. jump world + student_t noise
  4. static + student_t noise

For each condition x planner, reports:
  valid%           fraction of rounds with a valid certificate
  coverage         empirical coverage among valid rounds
  95% CI           Clopper-Pearson 95% CI on coverage
  claimed          mean claimed confidence among valid rounds
  gap~             median gap (UB - LB) among valid rounds
  cert%            fraction of rounds where certified (gap <= epsilon)

The Gaussian baseline should show lower coverage on student_t conditions:
the parametric claim over-reaches where conformal is distribution-free.

Usage
-----
    cert_env/bin/python scripts/run_tier0_baselines.py [--quick]

--quick : 5 seeds / 100 rounds (fast); default 25 seeds / 300 rounds.

Output saved to results/tier0_baselines/<config_id>.json.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from scipy.stats import beta as beta_dist

from certflow.baselines import gaussian_tier0_episode
from certflow.episodes import tier0_episode
from certflow.harness import ExperimentConfig, run_experiment, save_results

QUICK = "--quick" in sys.argv

BASE = ExperimentConfig(
    rows=6, cols=6, kind="bounded", rho=0.02,
    noise_family="gaussian", noise_scale=0.05,
    epsilon=5.0, alpha_prime=0.2, rho_w=0.99, eps_tv=1e-4,
    gamma_aci=0.01, delta=1.0, rho_hat_over_rho=1.0,
    n_seeds=5 if QUICK else 25,
    max_rounds=100 if QUICK else 300,
    base_seed=2026,
)

# Four conditions: (label, overrides)
CONDITIONS: list[tuple[str, dict]] = [
    ("bounded rho=0.02 gaussian",  dict(kind="bounded", rho=0.02, noise_family="gaussian")),
    ("bounded rho=0.02 student_t", dict(kind="bounded", rho=0.02, noise_family="student_t")),
    ("jump student_t",             dict(kind="jump",    noise_family="student_t")),
    ("static student_t",           dict(kind="static",  noise_family="student_t")),
]

PLANNERS = [
    ("CERT",     tier0_episode),
    ("Gaussian", gaussian_tier0_episode),
]


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


def compute_row(
    label: str,
    planner_name: str,
    condition_overrides: dict,
    episode_fn,
) -> dict:
    cfg = dataclasses.replace(BASE, **condition_overrides)
    result = run_experiment(episode_fn, cfg)

    covered = valid = 0
    claimed_sum = 0.0
    all_gaps = []
    valid_gaps = []

    for ep in result.episodes:
        if not hasattr(ep, "rounds"):
            continue
        for r in ep.rounds:
            all_gaps.append(r.ub - r.lb)
            if r.confidence > 0.0:
                valid += 1
                covered += r.covered
                claimed_sum += r.confidence
                valid_gaps.append(r.ub - r.lb)

    agg = result.aggregate()
    total_rounds = agg["n_rounds_total"]
    lo, hi = clopper_pearson(covered, valid)

    return {
        "condition": label,
        "planner": planner_name,
        "valid_rounds": valid,
        "total_rounds": total_rounds,
        "valid_fraction": valid / total_rounds if total_rounds else 0.0,
        "coverage": covered / valid if valid else float("nan"),
        "cov_ci_lo": lo,
        "cov_ci_hi": hi,
        "claimed_mean": claimed_sum / valid if valid else float("nan"),
        "gap_median": sorted(valid_gaps)[len(valid_gaps) // 2] if valid_gaps else float("nan"),
        "certified_fraction": agg["certified_fraction"],
        "failure_count": agg["failure_count"],
        "config_id": cfg.config_id(),
    }, result, cfg


def main() -> None:
    outdir = Path("results/tier0_baselines")
    outdir.mkdir(parents=True, exist_ok=True)

    table = []
    for cond_label, cond_overrides in CONDITIONS:
        for planner_name, episode_fn in PLANNERS:
            print(f"running: {cond_label} / {planner_name} ...", flush=True)
            row, result, cfg = compute_row(
                cond_label, planner_name, cond_overrides, episode_fn
            )
            table.append(row)
            save_results(result, str(outdir))
            print(f"  done: valid={row['valid_rounds']}/{row['total_rounds']} "
                  f"cov={row['coverage']:.3f}", flush=True)

    # Save table JSON (replace nan with null for JSON compliance)
    def _json_safe(v):
        if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
            return None
        return v

    json_table = [{k: _json_safe(v) for k, v in row.items()} for row in table]
    (outdir / "table.json").write_text(json.dumps(json_table, indent=2))

    # Print combined table
    hdr = (
        f"{'condition':28} {'planner':9} {'valid%':>6} {'coverage':>9} "
        f"{'95% CI':>16} {'claimed':>8} {'gap~':>6} {'cert%':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in table:
        import math
        ci = (
            f"[{r['cov_ci_lo']:.3f},{r['cov_ci_hi']:.3f}]"
            if not math.isnan(r['cov_ci_lo'])
            else "     [n/a]     "
        )
        coverage_str = (
            f"{r['coverage']:9.3f}" if not math.isnan(r['coverage']) else "      nan"
        )
        claimed_str = (
            f"{r['claimed_mean']:8.3f}" if not math.isnan(r['claimed_mean']) else "     nan"
        )
        gap_str = (
            f"{r['gap_median']:6.2f}" if not math.isnan(r['gap_median']) else "   nan"
        )
        print(
            f"{r['condition']:28} {r['planner']:9} "
            f"{100 * r['valid_fraction']:>5.1f}% {coverage_str} {ci:>16} "
            f"{claimed_str} {gap_str} {100 * r['certified_fraction']:>5.1f}%"
        )

    print(f"\nResults saved to {outdir.resolve()}/")


if __name__ == "__main__":
    main()
