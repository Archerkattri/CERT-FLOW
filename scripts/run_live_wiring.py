"""Real-data benchmark of the live-wired round-2 calibrators on METR-LA.

Runs the existing TrafficWorld / traffic_planner_config harness (same replay,
endpoints, oracle as scripts/run_metr_la.py) in three modes and reports, per
mode: valid%, the LB <= OPT <= UB violation rate against the recording's true
costs, and the mean/median certified gap (width). For the watch_monitor mode it
also reports whether the test martingale / Shiryaev-Roberts detector stayed
quiet across seeds and the end-of-run drift / effective-sample-size diagnostics.

  (i)   default Bonferroni per-edge pricing (the shipped behaviour)
  (ii)  path_calibration="pasc"  (joint per-edge radius)
  (iii) default + watch_monitor=True (live validity monitor; pricing unchanged)

Run: PYTHONPATH=src python scripts/run_live_wiring.py [--quick]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

from certflow.cert import CertPlanner
from certflow.graphcore import dijkstra
from certflow.realworld import TrafficWorld, far_endpoints, traffic_planner_config

QUICK = "--quick" in sys.argv
SEEDS = 5 if QUICK else 20
ROUNDS = 100 if QUICK else 288  # one replay day

MODES = [
    ("Bonferroni (default)", dict()),
    ("PASC", dict(path_calibration="pasc")),
    ("Bonferroni + watch", dict(watch_monitor=True, sr_threshold=5000.0)),
]


def true_opt(world: TrafficWorld, t: float, s, g) -> float:
    snap = {u: {v: world.true_cost((u, v), t) for v in nbrs}
            for u, nbrs in world.graph.items()}
    _, cost = dijkstra(snap, s, g)
    return cost


def main() -> None:
    rows = []
    for mname, over in MODES:
        valid = violations = certn = rounds_total = 0
        gaps, cert_gaps = [], []
        quiet_watch = quiet_sr = 0
        drift_scores, ess_vals = [], []
        for seed in range(SEEDS):
            w = TrafficWorld(dataset="metr-la", seed=seed, n_bins=ROUNDS)
            s, g = far_endpoints(w)
            cfg = traffic_planner_config(**over)
            p = CertPlanner(w, s, g, cfg)
            for _ in range(ROUNDS):
                t_round = p.t
                cert, _ = p.round()
                rounds_total += 1
                if cert.valid:
                    valid += 1
                    o = true_opt(w, t_round, s, g)
                    covered = cert.lb - 1e-9 <= o <= cert.ub + 1e-9
                    violations += not covered
                    gaps.append(cert.gap)
                    is_cert = (cert.gap <= cfg.epsilon
                               and cert.confidence >= cfg.min_certify_confidence)
                    certn += is_cert
                    if is_cert:
                        cert_gaps.append(cert.gap)
            if over.get("watch_monitor"):
                quiet_watch += not p.watch.alarm()
                quiet_sr += not p.sr.alarm()
                d = p.diagnostics()
                drift_scores.append(d["residual_drift_score"])
                ess_vals.append(d["effective_sample_size"])
        row = dict(
            mode=mname,
            valid_pct=valid / rounds_total,
            violation_rate=violations / valid if valid else float("nan"),
            n_valid=valid,
            gap_mean=st.mean(gaps) if gaps else float("nan"),
            gap_median=st.median(gaps) if gaps else float("nan"),
            cert_pct=certn / rounds_total,
            cert_gap_median=st.median(cert_gaps) if cert_gaps else float("nan"),
        )
        if over.get("watch_monitor"):
            row.update(
                quiet_watch=f"{quiet_watch}/{SEEDS}",
                quiet_sr=f"{quiet_sr}/{SEEDS}",
                drift_score_mean=st.mean(drift_scores) if drift_scores else float("nan"),
                ess_mean=st.mean(ess_vals) if ess_vals else float("nan"),
            )
        rows.append(row)
        print(f"done: {mname}", flush=True)

    outdir = Path("results/live_wiring")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "table.json").write_text(
        json.dumps(dict(seeds=SEEDS, rounds=ROUNDS, rows=rows), indent=2))

    hdr = (f"{'mode':22} {'valid%':>7} {'viol.rate':>9} {'gap~med':>8} "
           f"{'gap~mean':>8} {'cert%':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['mode']:22} {100*r['valid_pct']:>6.1f}% "
              f"{r['violation_rate']:>9.4f} {r['gap_median']:>8.1f} "
              f"{r['gap_mean']:>8.1f} {100*r['cert_pct']:>5.1f}%")
    for r in rows:
        if "quiet_watch" in r:
            print(f"\n{r['mode']}: martingale quiet {r['quiet_watch']}, "
                  f"SR quiet {r['quiet_sr']}, mean drift score "
                  f"{r['drift_score_mean']:.3f}, mean ESS {r['ess_mean']:.1f}")

    # honest width verdict
    b = next(r for r in rows if r["mode"].startswith("Bonferroni (default)"))
    pasc = next(r for r in rows if r["mode"] == "PASC")
    if b["gap_median"] and pasc["gap_median"] == pasc["gap_median"]:
        red = 100 * (1 - pasc["gap_median"] / b["gap_median"])
        print(f"\nPASC median-gap change vs Bonferroni: {red:+.1f}% "
              f"(negative = wider) at violation rate "
              f"{pasc['violation_rate']:.4f} vs {b['violation_rate']:.4f}")


if __name__ == "__main__":
    main()
