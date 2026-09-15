"""Full v4 fleet/sensing audit with reproducible provenance.

This is the release-facing companion to the correlated-world stress test.  It
compares the upgraded CERT-FLOW portfolio against budget-matched controls on
the same drifting worlds and reports decision quality together with held-out
certificate coverage.  The run is deliberately larger than the historical
Tier-2 quick table and writes a manifest so partial/quick results cannot be
promoted by accident.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import source_manifest  # noqa: E402

from certflow.episodes import coverage_among_valid, tier2_episode
from certflow.harness import ExperimentConfig, run_experiment, spawn_seeds

BASE = ExperimentConfig(
    rows=10,
    cols=10,
    kind="bounded",
    rho=0.02,
    noise_family="gaussian",
    noise_scale=0.05,
    epsilon=8.0,
    alpha_prime=0.2,
    rho_w=0.99,
    eps_tv=1e-4,
    gamma_aci=0.01,
    delta=1.0,
    rho_hat_over_rho=1.0,
    use_kappa=True,
    initial_survey=False,
    # Match the existing Tier-2 release horizon; B=40 otherwise truncates
    # before arrival and turns a budget comparison into a horizon comparison.
    max_rounds=600,
    n_seeds=12,
    base_seed=2026,
)

POLICIES = ("cert", "hybrid", "voi", "random", "max_age", "max_width")
BUDGETS = (10.0, 20.0, 40.0)


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _manifest() -> dict:
    data_root = ROOT / "data"
    files = []
    if data_root.exists():
        for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
            digest = hashlib.sha256()
            with path.open("rb") as fh:
                for block in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(block)
            files.append({"path": str(path.relative_to(ROOT)), "sha256": digest.hexdigest()})
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except Exception:
        dirty = None
    return {
        "schema": "certflow.benchmark-manifest.1",
        **source_manifest(ROOT),
        "source_revision": _revision(),
        "dirty_tree": dirty,
        "data_files": files,
        "complete": True,
    }


def _coverage_lcb(covered: int, valid: int, alpha: float = 0.05) -> float:
    if valid <= 0:
        return math.nan
    return float(beta_dist.ppf(alpha, covered, valid - covered + 1)) if covered else 0.0


def aggregate(result) -> dict:
    n_reached = sum(1 for ep in result.episodes if ep.reached_goal)
    regrets = []
    n_valid = n_covered = 0
    for ep in result.episodes:
        if ep.reached_goal and math.isfinite(ep.oracle_cost):
            regrets.append(ep.travel_cost - ep.oracle_cost)
        covered, valid = coverage_among_valid(ep)
        n_covered += covered
        n_valid += valid
    rounds = sum(len(ep.rounds) for ep in result.episodes)
    reached_rounds = [len(ep.rounds) for ep in result.episodes if ep.reached_goal]
    spend = [ep.sense_cost for ep in result.episodes]
    return {
        "goal_frac": n_reached / len(result.episodes) if result.episodes else math.nan,
        "mission_rounds_mean": sum(reached_rounds) / len(reached_rounds) if reached_rounds else math.nan,
        "regret_mean": sum(regrets) / len(regrets) if regrets else math.nan,
        "regret_median": sorted(regrets)[len(regrets) // 2] if regrets else math.nan,
        "sense_spend_mean": sum(spend) / len(spend) if spend else math.nan,
        "coverage_valid": n_covered / n_valid if n_valid else math.nan,
        "coverage_lcb": _coverage_lcb(n_covered, n_valid),
        "n_valid": n_valid,
        "n_covered": n_covered,
        "n_rounds": rounds,
        "n_episodes": len(result.episodes),
        "n_reached": n_reached,
        "failures": len(result.failures),
    }


def conditions() -> list[tuple[str, ExperimentConfig]]:
    out = []
    for policy in POLICIES:
        for budget in BUDGETS:
            sensing = "cert" if policy == "hybrid" else policy
            out.append((
                f"{policy}|B={budget:.0f}",
                dataclasses.replace(
                    BASE,
                    sensing_policy=sensing,
                    move_policy="when_certified",
                    sense_budget=budget,
                    hybrid_sensing=(policy == "hybrid"),
                    refine_after_certify=(policy == "hybrid"),
                ),
            ))
    out.extend([
        ("no-cert|B=inf", dataclasses.replace(
            BASE, sensing_policy="none", move_policy="always", sense_budget=math.inf)),
        ("cert-always|B=inf", dataclasses.replace(
            BASE, sensing_policy="cert", move_policy="always", sense_budget=math.inf)),
    ])
    return out


def _json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "__nan__" if math.isnan(value) else ("__inf__" if value > 0 else "__-inf__")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CERT-FLOW budgeted single-agent sensing benchmark."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="run the reduced development matrix (not a release artifact)",
    )
    args = parser.parse_args()
    quick = args.quick
    cfg_base = dataclasses.replace(BASE, n_seeds=5 if quick else 12, max_rounds=180 if quick else 600)
    rows = []
    for label, cfg in conditions():
        cfg = dataclasses.replace(cfg, n_seeds=cfg_base.n_seeds, max_rounds=cfg_base.max_rounds)
        print(f"running {label} ({cfg.n_seeds} seeds x {cfg.max_rounds} rounds)", flush=True)
        result = run_experiment(tier2_episode, cfg)
        agg = aggregate(result)
        rows.append({"label": label, "policy": label.split("|")[0].strip(), "budget": label.split("=")[-1], **agg,
                     "config_id": cfg.config_id(), "seeds": spawn_seeds(cfg.base_seed, cfg.n_seeds)})
        print(f"  goal={agg['goal_frac']:.3f} regret={agg['regret_mean']:.4f} "
              f"coverage={agg['coverage_valid']:.4f} LCB={agg['coverage_lcb']:.4f}", flush=True)
    payload = {
        "schema": "certflow.v4.fleet-sensing.1",
        "quick": quick,
        "seeds": cfg_base.n_seeds,
        "rounds": cfg_base.max_rounds,
        "budgets": list(BUDGETS),
        "policies": list(POLICIES),
        "manifest": _manifest(),
        "rows": rows,
    }
    out = ROOT / "results" / "v4_validity" / ("fleet-sensing-quick.json" if quick else "fleet-sensing-full.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")
    print(f"WROTE_JSON {out}")


if __name__ == "__main__":
    main()
