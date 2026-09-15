"""Matched calibration-shift recovery benchmark for CERT-FLOW v4.

The path certificate can remain covered while edge-level residual bands are
already under-covering after a changepoint.  This runner therefore reports both
levels and compares the existing heuristic recovery manager, the formal
e-value recovery gate, and an ACI-only control on identical world seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "extval"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import source_manifest  # noqa: E402

from certflow.cert import CertPlanner, PlannerConfig  # noqa: E402
from certflow.graphcore import dijkstra  # noqa: E402
from certflow.oracle import opt  # noqa: E402
from shift_world import ShiftWorld, _draw_one  # noqa: E402


def _snapshot(world: ShiftWorld, t: float) -> dict:
    return {u: {v: world.true_cost((u, v), t) for v in nbrs}
            for u, nbrs in world.graph.items()}


def _run_one(method: str, seed: int, rounds: int, t_cp: float,
             recovery_samples: int = 10, scalar_formal: bool = False,
             formal_epsilon: float = 0.5) -> dict:
    formal = method == "formal"
    recovery = method in {"heuristic", "formal"}
    use_aci = method == "aci"
    cfg = PlannerConfig(
        epsilon=5.0,
        alpha_prime=0.2,
        rho_w=0.99,
        eps_tv=0.0,
        delta=1.0,
        sensing_policy="cert",
        initial_survey=True,
        maintenance_every=8,
        use_aci=use_aci,
        latent_margin=1.0,
        regime_recovery=recovery,
        recovery_formal=formal,
        recovery_samples=recovery_samples,
        recovery_alarm_delta=0.01,
        watch_epsilon=formal_epsilon,
        recovery_betting_epsilons=(None if scalar_formal else (0.1, 0.25, 0.5, 0.75, 0.9)),
    )
    world = ShiftWorld(
        6, 6, np.random.default_rng(20000 + seed), t_cp=t_cp,
        rho_pre=0.005, rho_post=0.06,
        family_pre="gaussian", scale_pre=0.04,
        family_post="student_t", scale_post=0.25,
        rho_true_mode="pre", max_t=max(rounds + 10, 400),
    )
    planner = CertPlanner(world, (0, 0), (5, 5), cfg)
    probe_rng = np.random.default_rng(900000 + seed)
    path_valid = path_covered = 0
    edge_n = edge_miss = edge_n_post = edge_miss_post = 0
    pre_alarm = False
    detection_round: int | None = None
    recovery_round: int | None = None
    first_valid_after_recovery: int | None = None
    alarm_seen = False
    for r in range(rounds):
        t = planner.t
        cert, _ = planner.round()
        snap = _snapshot(world, t)
        _, optimum = opt(world, t, (0, 0), (5, 5))
        monitor = planner.formal_regime if formal else planner.regime
        is_alarm = recovery and monitor.state != "stable"
        if t < t_cp and is_alarm:
            pre_alarm = True
        if t >= t_cp and detection_round is None and (recovery and (is_alarm or monitor.alarm_count > 0)):
            detection_round = r
            alarm_seen = True
        if alarm_seen and recovery_round is None and monitor.state == "stable":
            recovery_round = r
        if recovery_round is not None and first_valid_after_recovery is None and cert.valid:
            first_valid_after_recovery = r
        if cert.valid:
            path_valid += 1
            path_covered += int(cert.lb - 1e-9 <= optimum <= cert.ub + 1e-9)
        alpha_edge = getattr(planner, "_last_alpha_edge", None)
        q = planner.scorer.quantile(alpha_edge, t) if alpha_edge else math.inf
        if not (alpha_edge and math.isfinite(q) and cert.path and len(cert.path) >= 2):
            continue
        for u, v in zip(cert.path, cert.path[1:]):
            edge = (u, v)
            belief = planner.beliefs[edge]
            half = q + belief.rho * belief.age(t)
            family = world._family_pre if t < t_cp else world._family_post
            scale = world._scale_pre if t < t_cp else world._scale_post
            heldout = world.true_cost(edge, t) + _draw_one(probe_rng, family, scale)
            miss = not (belief.c_hat - half - 1e-12 <= heldout <= belief.c_hat + half + 1e-12)
            edge_n += 1
            edge_miss += int(miss)
            if t >= t_cp:
                edge_n_post += 1
                edge_miss_post += int(miss)
    return {
        "method": method,
        "seed": seed,
        "rounds": rounds,
        "path_valid_fraction": path_valid / rounds,
        "path_coverage": path_covered / path_valid if path_valid else math.nan,
        "edge_coverage": 1.0 - edge_miss / edge_n if edge_n else math.nan,
        "post_edge_coverage": 1.0 - edge_miss_post / edge_n_post if edge_n_post else math.nan,
        "edge_n": edge_n,
        "post_edge_n": edge_n_post,
        "pre_alarm": pre_alarm,
        "detection_delay_rounds": (
            detection_round - int(t_cp) if detection_round is not None else math.nan
        ),
        "recovery_delay_rounds": (
            recovery_round - detection_round
            if recovery_round is not None and detection_round is not None else math.nan
        ),
        "recovery_to_valid_rounds": (
            first_valid_after_recovery - recovery_round
            if first_valid_after_recovery is not None and recovery_round is not None else math.nan
        ),
        "formal_diagnostics": planner.formal_regime.diagnostics(),
        "heuristic_diagnostics": planner.regime.diagnostics(),
    }


def _aggregate(rows: list[dict]) -> dict:
    out = {}
    for method in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == method]

        def med(key: str):
            vals = [r[key] for r in rs if math.isfinite(float(r[key]))]
            return st.median(vals) if vals else math.nan

        out[method] = {
            "seeds": len(rs),
            "path_valid_fraction_mean": st.mean(r["path_valid_fraction"] for r in rs),
            "path_coverage_mean": st.mean(r["path_coverage"] for r in rs),
            "edge_coverage_mean": st.mean(r["edge_coverage"] for r in rs),
            "post_edge_coverage_mean": st.mean(r["post_edge_coverage"] for r in rs),
            "pre_false_alarm_fraction": st.mean(bool(r["pre_alarm"]) for r in rs),
            # Keep the denominator visible.  A delay median over only the
            # successful seeds can otherwise make an always-failing detector
            # look fast or complete.
            "detections": sum(math.isfinite(float(r["detection_delay_rounds"])) for r in rs),
            "detection_rate": st.mean(math.isfinite(float(r["detection_delay_rounds"])) for r in rs),
            "recoveries": sum(math.isfinite(float(r["recovery_delay_rounds"])) for r in rs),
            "recovery_rate": st.mean(math.isfinite(float(r["recovery_delay_rounds"])) for r in rs),
            "valid_reentries": sum(math.isfinite(float(r["recovery_to_valid_rounds"])) for r in rs),
            "valid_reentry_rate": st.mean(math.isfinite(float(r["recovery_to_valid_rounds"])) for r in rs),
            "censored_detections": sum(not math.isfinite(float(r["detection_delay_rounds"])) for r in rs),
            "censored_recoveries": sum(not math.isfinite(float(r["recovery_delay_rounds"])) for r in rs),
            "censored_valid_reentries": sum(not math.isfinite(float(r["recovery_to_valid_rounds"])) for r in rs),
            "detection_delay_median": med("detection_delay_rounds"),
            "recovery_delay_median": med("recovery_delay_rounds"),
            "recovery_to_valid_median": med("recovery_to_valid_rounds"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--changepoint", type=float, default=110.0)
    ap.add_argument("--recovery-samples", type=int, default=5)
    ap.add_argument("--scalar-formal", action="store_true",
                    help="use the historical single-epsilon SR monitor (default)")
    ap.add_argument("--mixture-formal", action="store_true",
                    help="use the fixed multi-epsilon SR research variant")
    ap.add_argument("--formal-epsilon", type=float, default=0.5)
    args = ap.parse_args()
    if args.scalar_formal and args.mixture_formal:
        ap.error("--scalar-formal and --mixture-formal are mutually exclusive")
    use_scalar_formal = args.scalar_formal or not args.mixture_formal
    if args.quick:
        args.seeds, args.rounds = 3, 180
    rows = [
        _run_one(method, seed, args.rounds, args.changepoint,
                 args.recovery_samples,
                 use_scalar_formal,
                 args.formal_epsilon)
        for method in ("aci", "heuristic", "formal")
        for seed in range(args.seeds)
    ]
    aggregate = _aggregate(rows)
    formal = aggregate["formal"]
    out = {
        "schema": "certflow.v4.shift-recovery.1",
        "quick": args.quick,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "changepoint": args.changepoint,
        "methods": ["aci", "heuristic", "formal"],
        "recovery_samples": args.recovery_samples,
        "scalar_formal": use_scalar_formal,
        "formal_epsilon": args.formal_epsilon,
        "manifest": source_manifest(ROOT),
        "aggregate": aggregate,
        "rows": rows,
        "gates": {
            "full_provenance": not args.quick and args.seeds == 12 and args.rounds == 300,
            "formal_no_pre_shift_false_alarms": formal["pre_false_alarm_fraction"] == 0.0,
            "formal_detects_shift": formal["detection_rate"] == 1.0,
            "formal_reenters": formal["recovery_rate"] == 1.0 and formal["valid_reentry_rate"] == 1.0,
        },
    }
    target = ROOT / "results" / "v4_validity" / (
        "shift-recovery-quick.json" if args.quick else "shift-recovery-full.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"wrote {target}")
    return 0 if all(out["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
