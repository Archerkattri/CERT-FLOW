"""Full-loop CERT scale benchmark.

Purpose
-------
Benchmarks the complete CERT planner (conformal + dual D* Lite + pre-widening
+ sensing) across grid sizes 10x10 .. 60x60, three planner configurations, and
three random seeds x 150 rounds each.

Configurations
--------------
A: defaults (B=10, k_alternatives=3, no online rho)
B: B=0, k_alternatives=0 (exact metrics, no alternatives Dijkstras)
C: recommended_config() (online rho, hybrid sensing, kappa, adaptive_rate,
   sum_aware_ub) — the production configuration
D: defaults but k_alternatives=0 (isolates the alternatives Dijkstra overhead
   relative to config A, keeping everything else equal)

World: BoundedDrift, rho=0.02, noise_scale=0.05, initial_survey=True.

Usage
-----
    cert_env/bin/python scripts/run_scale.py          # full (3 seeds x 150)
    cert_env/bin/python scripts/run_scale.py --quick  # smoke (1 seed x 50)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

from certflow.cert import CertPlanner, PlannerConfig, recommended_config
from certflow.drift import grid_world
from certflow.oracle import opt


# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

SIZES = [(10, 10), (20, 20), (40, 40), (60, 60)]
SEEDS_FULL = [2026, 2027, 2028]
SEEDS_QUICK = [2026]
N_ROUNDS_FULL = 150
N_ROUNDS_QUICK = 50

RHO = 0.02
NOISE_SCALE = 0.05


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------

def config_A() -> PlannerConfig:
    """Defaults: B=10, k_alternatives=3."""
    return PlannerConfig(
        epsilon=5.0,
        alpha_prime=0.1,
        rho_w=0.99,
        rho=0.02 if False else 0.0,  # rho set via world; rho_mode="given" reads rho_true
        noise_scale=NOISE_SCALE if False else 0.0,
        prewiden_rounds=10,
        k_alternatives=3,
    )


def _base_planner_cfg(**overrides) -> PlannerConfig:
    """Shared baseline params then apply overrides."""
    base = dict(
        epsilon=5.0,
        alpha_prime=0.1,
        rho_w=0.99,
        prewiden_rounds=10,
        k_alternatives=3,
    )
    base.update(overrides)
    return PlannerConfig(**base)


CONFIGS: dict[str, PlannerConfig] = {
    "A_defaults_B10": _base_planner_cfg(),
    "B_exact_B0_k0": _base_planner_cfg(prewiden_rounds=0, k_alternatives=0),
    "C_recommended":  recommended_config(epsilon=5.0, alpha_prime=0.1, rho_w=0.99,
                                         prewiden_rounds=10, k_alternatives=3),
    "D_defaults_k0":  _base_planner_cfg(k_alternatives=0),
}


# ---------------------------------------------------------------------------
# Per-round timing + RSS measurement
# ---------------------------------------------------------------------------

def _rss_kb() -> int:
    """Peak RSS in KiB via getrusage (Linux: ru_maxrss in kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@dataclass
class RoundRecord:
    wall_s: float
    gap: float
    valid: bool
    certified: bool


@dataclass
class RunResult:
    size: tuple[int, int]
    config_name: str
    seed: int
    rounds: list[RoundRecord] = field(default_factory=list)
    peak_rss_kb: int = 0
    init_wall_s: float = 0.0


def run_one(
    rows: int,
    cols: int,
    cfg: PlannerConfig,
    seed: int,
    n_rounds: int,
) -> RunResult:
    rss_before = _rss_kb()
    t0_init = time.perf_counter()
    world = grid_world(rows, cols, seed=seed, kind="bounded",
                       rho=RHO, noise_scale=NOISE_SCALE)
    start, goal = (0, 0), (rows - 1, cols - 1)
    planner = CertPlanner(world, start, goal, cfg)
    init_wall = time.perf_counter() - t0_init

    records: list[RoundRecord] = []
    for _ in range(n_rounds):
        w0 = time.perf_counter()
        cert, _ = planner.round()
        w1 = time.perf_counter()
        records.append(RoundRecord(
            wall_s=w1 - w0,
            gap=cert.gap if math.isfinite(cert.gap) else float("inf"),
            valid=cert.valid,
            certified=bool(cert.valid and cert.gap <= cfg.epsilon),
        ))

    rss_after = _rss_kb()
    res = RunResult(
        size=(rows, cols),
        config_name="",
        seed=seed,
        rounds=records,
        peak_rss_kb=max(rss_after, rss_before),  # conservative; getrusage is cumulative peak
        init_wall_s=init_wall,
    )
    return res


# ---------------------------------------------------------------------------
# Aggregate a list of RunResult for one (size, config)
# ---------------------------------------------------------------------------

@dataclass
class CellStats:
    size: tuple[int, int]
    config_name: str
    n_rounds: int
    p50_ms: float
    p95_ms: float
    peak_rss_kb: int   # max across seeds
    valid_pct: float   # % rounds with cert.valid
    certified_pct: float
    gap_median: float  # median over valid rounds (inf if none)


def aggregate(runs: list[RunResult], config_name: str) -> CellStats:
    all_walls: list[float] = []
    all_valid: list[bool] = []
    all_cert: list[bool] = []
    all_gaps: list[float] = []
    max_rss = 0

    for r in runs:
        for rec in r.rounds:
            all_walls.append(rec.wall_s)
            all_valid.append(rec.valid)
            all_cert.append(rec.certified)
            if rec.valid and math.isfinite(rec.gap):
                all_gaps.append(rec.gap)
        max_rss = max(max_rss, r.peak_rss_kb)

    n = len(all_walls)
    p50 = float(np.percentile(all_walls, 50)) * 1000.0
    p95 = float(np.percentile(all_walls, 95)) * 1000.0
    valid_pct = 100.0 * sum(all_valid) / n if n else 0.0
    cert_pct = 100.0 * sum(all_cert) / n if n else 0.0
    gap_med = float(np.median(all_gaps)) if all_gaps else float("inf")

    size = runs[0].size if runs else (0, 0)
    return CellStats(
        size=size,
        config_name=config_name,
        n_rounds=n,
        p50_ms=p50,
        p95_ms=p95,
        peak_rss_kb=max_rss,
        valid_pct=valid_pct,
        certified_pct=cert_pct,
        gap_median=gap_med,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CERT full-loop scale benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke run: 1 seed x 50 rounds")
    args = parser.parse_args()

    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL
    n_rounds = N_ROUNDS_QUICK if args.quick else N_ROUNDS_FULL
    mode = "quick" if args.quick else "full"

    print(f"[scale] mode={mode}  sizes={[f'{r}x{c}' for r,c in SIZES]}  "
          f"seeds={seeds}  rounds/seed={n_rounds}  configs={list(CONFIGS)}")

    all_stats: list[CellStats] = []
    raw: list[dict[str, Any]] = []

    for rows, cols in SIZES:
        size_label = f"{rows}x{cols}"
        n_edges = 2 * (rows * (cols - 1) + (rows - 1) * cols)
        for cfg_name, cfg in CONFIGS.items():
            runs: list[RunResult] = []
            for seed in seeds:
                print(f"  {size_label} {cfg_name} seed={seed} ...", end="", flush=True)
                r = run_one(rows, cols, cfg, seed, n_rounds)
                r.config_name = cfg_name
                runs.append(r)
                total_ms = sum(rec.wall_s for rec in r.rounds) * 1000
                print(f" done  total={total_ms:.0f}ms  rss={r.peak_rss_kb}kB")

            stats = aggregate(runs, cfg_name)
            all_stats.append(stats)
            raw.append({
                "size": size_label,
                "n_edges": n_edges,
                "config": cfg_name,
                "seeds": seeds,
                "n_rounds": stats.n_rounds,
                "p50_ms": round(stats.p50_ms, 3),
                "p95_ms": round(stats.p95_ms, 3),
                "peak_rss_kb": stats.peak_rss_kb,
                "valid_pct": round(stats.valid_pct, 1),
                "certified_pct": round(stats.certified_pct, 1),
                "gap_median": round(stats.gap_median, 2) if math.isfinite(stats.gap_median) else None,
            })

    # ------------------------------------------------------------------
    # Save raw JSON
    # ------------------------------------------------------------------
    out_dir = _repo / "results" / "scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "table.json"
    with open(json_path, "w") as fh:
        json.dump(raw, fh, indent=2)
    print(f"\n[scale] Raw results -> {json_path}")

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'size':<8} {'config':<25} {'n_rounds':>8} {'p50 ms':>8} {'p95 ms':>8} "
          f"{'RSS MB':>7} {'valid%':>7} {'cert%':>7} {'gap p50':>8}")
    print("-" * 100)
    for s in all_stats:
        rss_mb = s.peak_rss_kb / 1024.0
        gap_str = f"{s.gap_median:.2f}" if math.isfinite(s.gap_median) else " inf"
        print(f"{s.size[0]}x{s.size[1]:<4}  {s.config_name:<25} {s.n_rounds:>8} "
              f"{s.p50_ms:>8.2f} {s.p95_ms:>8.2f} {rss_mb:>7.1f} "
              f"{s.valid_pct:>7.1f} {s.certified_pct:>7.1f} {gap_str:>8}")
    print("=" * 100)


if __name__ == "__main__":
    main()
