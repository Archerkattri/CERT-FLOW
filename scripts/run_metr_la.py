"""Real-data validation: CERT on replayed METR-LA traffic.

The recording is ground truth, so the oracle is exact on REAL drifting costs:
this is the experiment where the certificate meets data we did not generate.
A1 is violated by real incidents at a measured rate (rho = per-edge p95 of
|dc/dt|), so this is simultaneously the honest off-model stress test.

Per seed: one replay day (288 bins), stationary certification at a far
endpoint pair, one observation per 5-minute bin. CERT vs the Gaussian
baseline, plus an adaptive-rate CERT variant.

Run: cert_env/bin/python scripts/run_metr_la.py [--quick]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import beta as beta_dist

from certflow.baselines import GaussianCertPlanner
from certflow.cert import CertPlanner
from certflow.graphcore import dijkstra
from certflow.realworld import TrafficWorld, far_endpoints, traffic_planner_config

QUICK = "--quick" in sys.argv
DATASET = "pems-bay" if "--pems-bay" in sys.argv else "metr-la"
SEEDS = 4 if QUICK else 20
ROUNDS = 100 if QUICK else 288  # one day full

# (name, class, planner overrides, world overrides). rho_quantile is the
# drift-model aggressiveness: smaller -> tighter widths but more A1
# violations, which the conformal layer absorbs as larger scores (measured:
# coverage 1.000 from p95 down to p50; p75 is the width-optimal point).
PLANNERS = [
    ("CERT p95", CertPlanner, {}, {}),
    ("CERT p75", CertPlanner, {}, dict(rho_quantile=0.75)),
    ("CERT p75+adaptive", CertPlanner, dict(adaptive_rate=True), dict(rho_quantile=0.75)),
    ("Gaussian p95", GaussianCertPlanner, {}, {}),
]


def cp_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


def true_opt(world: TrafficWorld, t: float, s, g) -> float:
    snap = {u: {v: world.true_cost((u, v), t) for v in nbrs}
            for u, nbrs in world.graph.items()}
    _, cost = dijkstra(snap, s, g)
    return cost


def main() -> None:
    rows = []
    a1_rates = []
    for pname, cls, over, wover in PLANNERS:
        covered = valid = certn = rounds_total = 0
        claimed_sum = 0.0
        gaps = []
        spend = 0.0
        for seed in range(SEEDS):
            w = TrafficWorld(dataset=DATASET, seed=seed, n_bins=ROUNDS, **wover)
            a1_rates.append(w.a1_violation_rate)
            s, g = far_endpoints(w)
            cfg = traffic_planner_config(**over)
            p = cls(w, s, g, cfg)
            for _ in range(ROUNDS):
                t_round = p.t
                cert, _ = p.round()
                rounds_total += 1
                if cert.valid:
                    valid += 1
                    o = true_opt(w, t_round, s, g)
                    covered += cert.lb - 1e-9 <= o <= cert.ub + 1e-9
                    claimed_sum += cert.confidence
                    certn += (
                        cert.gap <= cfg.epsilon
                        and cert.confidence >= cfg.min_certify_confidence
                    )
                    gaps.append(cert.gap)
            spend += p.sense_spend
        lo, hi = cp_ci(covered, valid)
        rows.append(dict(
            planner=pname,
            valid_pct=valid / rounds_total,
            coverage=covered / valid if valid else float("nan"),
            ci_lo=lo, ci_hi=hi,
            claimed=claimed_sum / valid if valid else float("nan"),
            gap_median=sorted(gaps)[len(gaps) // 2] if gaps else float("nan"),
            cert_pct=certn / rounds_total,
            spend=spend / SEEDS,
        ))
        print(f"done: {pname}", flush=True)

    outdir = Path(f"results/{DATASET.replace(chr(45), chr(95))}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "table.json").write_text(json.dumps(
        dict(rows=rows, a1_violation_rate=sum(a1_rates) / len(a1_rates)), indent=2))

    print(f"\nmean A1-violation rate (real incidents vs p95 rho): "
          f"{sum(a1_rates)/len(a1_rates):.3f}")
    hdr = (f"{'planner':14} {'valid%':>7} {'coverage':>9} {'95% CI':>16} "
           f"{'claimed':>8} {'gap~ (s)':>9} {'cert%':>6} {'spend':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        print(f"{r['planner']:14} {100*r['valid_pct']:>6.1f}% {r['coverage']:>9.3f} "
              f"{ci:>16} {r['claimed']:>8.3f} {r['gap_median']:>9.1f} "
              f"{100*r['cert_pct']:>5.1f}% {r['spend']:>6.0f}")


if __name__ == "__main__":
    main()
