"""Ablation suite (spec section 4.5 kill-gate + module ablations).

Each module added to the v1 loop gets an on/off (or sweep) axis, measured on
the same worlds: coverage, valid%, certified%, gap, churn (Fox edge
symmetric-difference per round), replan latency, sensing spend.

Conditions:
  full          B=10, maintenance on, backstop on, kappa ON
  no-kappa      kappa OFF (the Design-1 kill-gate comparison)
  no-maint      maintenance sensing off
  B=0           pre-widening off (exact metrics)
  B=20          aggressive pre-widening
  no-backstop   round-robin backstop disabled (greedy sensing only)

Run: cert_env/bin/python scripts/run_ablations.py [--quick]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.oracle import opt
from certflow.sensing import path_edges

QUICK = "--quick" in sys.argv
EPS = 12.0 if "--eps12" in sys.argv else 5.0  # --eps12: above the T2' floor so
# maintenance/backstop rows are informative (ablations doc finding 3)
SEEDS = 5 if QUICK else 20
ROUNDS = 100 if QUICK else 300
ROWS = COLS = 8

BASE = dict(
    epsilon=EPS, alpha_prime=0.2, rho_w=0.99, eps_tv=1e-4,
    gamma_aci=0.01, delta=1.0, prewiden_rounds=10, use_kappa=True,
)

CONDITIONS: list[tuple[str, dict]] = [
    ("full",        {}),
    ("no-kappa",    dict(use_kappa=False)),
    ("no-maint",    dict(maintenance_every=10**9, maintenance_lookahead=0.0)),
    ("B=0",         dict(prewiden_rounds=0)),
    ("B=20",        dict(prewiden_rounds=20)),
    ("no-backstop", dict(backstop_slack=1e9)),
]


def run_condition(overrides: dict) -> dict:
    cfg = PlannerConfig(**{**BASE, **overrides})
    covered = valid = certified = rounds_total = 0
    churn_diffs: list[int] = []
    gaps: list[float] = []
    latencies: list[float] = []
    spend = 0.0
    for seed in range(SEEDS):
        w = grid_world(ROWS, COLS, seed=seed, kind="bounded", rho=0.02, noise_scale=0.05)
        p = CertPlanner(w, (0, 0), (ROWS - 1, COLS - 1), cfg)
        prev = None
        for _ in range(ROUNDS):
            t_round = p.t
            t0 = time.perf_counter()
            cert, _ = p.round()
            latencies.append(time.perf_counter() - t0)
            rounds_total += 1
            cur = set(path_edges(cert.path)) if cert.path else None
            if prev and cur:
                churn_diffs.append(len(prev ^ cur))
            prev = cur
            if cert.valid:
                valid += 1
                gaps.append(cert.gap)
                certified += cert.gap <= cfg.epsilon
                _, o = opt(w, t_round, (0, 0), (ROWS - 1, COLS - 1))
                covered += cert.lb - 1e-9 <= o <= cert.ub + 1e-9
        spend += p.sense_spend
    churn_diffs.sort()
    n = len(churn_diffs)
    return dict(
        coverage=covered / valid if valid else float("nan"),
        valid_pct=valid / rounds_total,
        cert_pct=certified / valid if valid else 0.0,
        gap_median=statistics.median(gaps) if gaps else float("nan"),
        churn_mean=sum(churn_diffs) / n if n else float("nan"),
        churn_p95=churn_diffs[int(0.95 * n)] if n else float("nan"),
        churn_rounds_pct=sum(d > 0 for d in churn_diffs) / n if n else float("nan"),
        latency_p50_ms=1e3 * statistics.median(latencies),
        sense_spend=spend / SEEDS,
    )


def main() -> None:
    out = {}
    hdr = (
        f"{'condition':12} {'coverage':>8} {'valid%':>7} {'cert%':>6} {'gap~':>6} "
        f"{'churn':>6} {'chp95':>5} {'flap%':>6} {'p50ms':>6} {'spend':>6}"
    )
    rows = []
    for label, overrides in CONDITIONS:
        m = run_condition(overrides)
        out[label] = m
        rows.append(
            f"{label:12} {m['coverage']:>8.3f} {100*m['valid_pct']:>6.1f}% "
            f"{100*m['cert_pct']:>5.1f}% {m['gap_median']:>6.2f} "
            f"{m['churn_mean']:>6.2f} {m['churn_p95']:>5} "
            f"{100*m['churn_rounds_pct']:>5.1f}% {m['latency_p50_ms']:>6.2f} "
            f"{m['sense_spend']:>6.1f}"
        )
        print(f"done: {label}", flush=True)
    print("\n" + hdr)
    print("-" * len(hdr))
    print("\n".join(rows))
    outdir = Path("results/ablations")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "table.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
