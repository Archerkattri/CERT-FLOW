"""Reproducible CERT-FLOW v3 scorecard.

This is the promotion harness for the seven v3 upgrade primitives.  It keeps
the cheap, deterministic primitive checks in one place and adds paired planner
episodes for validity, gap, regret, sensing, recovery, and fleet metrics.

Quick diagnostic:
    python scripts/run_v3_benchmarks.py --quick

The full invocation is intentionally longer.  Results are written under
``results/v3_scorecard`` with raw rows plus an aggregate JSON.  No result from
this runner is a release claim unless the full (non-quick) configuration is
used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import source_manifest  # noqa: E402

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.graphcore import dijkstra
from certflow.team import additive_certificate, joint_fleet_certificate
from certflow.upgrades import (
    ActiveSensingPolicy,
    DecisionRiskController,
    EvidenceConditionalCalibrator,
    JointFleetCalibrator,
    RegimeRecoveryManager,
    SelectionConditionalCalibrator,
    SelectionLedger,
    SensingAction,
    TrajectoryConformalCalibrator,
    congestion_penalty,
)


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _manifest() -> dict:
    """Capture enough provenance to detect stale or non-reproducible runs."""
    files = []
    data_root = ROOT / "data"
    for path in sorted(data_root.rglob("*")) if data_root.exists() else []:
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": str(path.relative_to(ROOT)), "sha256": digest,
                      "bytes": path.stat().st_size})
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        status = "unknown"
    source = source_manifest(ROOT)
    return {
        "schema": "certflow.benchmark-manifest.1",
        **source,
        "revision": _revision(),
        "working_tree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "data_files": files,
        "complete": True,
    }


def _ci(k: int, n: int) -> tuple[float, float]:
    # Wilson interval avoids an extra scipy dependency in the scorecard.
    if n <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - rad), min(1.0, ctr + rad)


def _json_safe(value):
    """Make diagnostics JSON-safe without changing the runtime APIs."""
    if isinstance(value, dict):
        return {repr(k) if not isinstance(k, (str, int, float, bool, type(None))) else k:
                _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _snapshot(world, t: float) -> dict:
    return {u: {v: world.true_cost((u, v), t) for v in nbrs}
            for u, nbrs in world.graph.items()}


def _path_cost(snapshot: dict, path: list) -> float:
    return sum(snapshot[u][v] for u, v in zip(path, path[1:]))


def _planner_condition(name: str) -> PlannerConfig:
    base = dict(
        epsilon=12.0, alpha_prime=0.2, rho_w=0.99, eps_tv=1e-4,
        gamma_aci=0.01, delta=1.0, prewiden_rounds=10,
        min_certify_confidence=0.5,
    )
    if name == "legacy":
        return PlannerConfig(**base)
    if name == "recommended":
        return PlannerConfig(**base, rho_mode="online", rho_online_quantile=0.5,
                             hybrid_sensing=True,
                             use_kappa=True, adaptive_rate=True,
                             sum_aware_ub=True, warmup_sense_per_round=12,
                             max_sense_per_round=12,
                             latent_margin=1.05)
    if name == "recommended-cia":
        return PlannerConfig(**base, rho_mode="online", hybrid_sensing=True,
                             use_kappa=True, adaptive_rate=True,
                             cia_ub=True)
    if name == "v3-active-recovery":
        return PlannerConfig(**base, rho_mode="online", hybrid_sensing=True,
                             use_kappa=True, adaptive_rate=True,
                             sum_aware_ub=True, active_sensing=True,
                             regime_recovery=True, trajectory_tubes=True)
    if name == "v3-active":
        return PlannerConfig(**base, rho_mode="online", hybrid_sensing=True,
                             use_kappa=True, adaptive_rate=True,
                             sum_aware_ub=True, active_sensing=True,
                             active_sensing_min_improvement=0.25)
    if name == "v3-recovery":
        return PlannerConfig(**base, rho_mode="online", hybrid_sensing=True,
                             use_kappa=True, adaptive_rate=True,
                             sum_aware_ub=True, regime_recovery=True)
    if name == "v3-evidence":
        return PlannerConfig(**base, rho_mode="online", hybrid_sensing=True,
                             use_kappa=True, adaptive_rate=True,
                             sum_aware_ub=True, evidence_model=True)
    raise ValueError(name)


def _row_key(row: dict) -> tuple:
    return (row.get("kind"), row.get("dataset", "synthetic"),
            row.get("condition"), row.get("seed"))


def planner_rows(quick: bool, done: set[tuple] | None = None,
                 emit=None) -> list[dict]:
    seeds = 3 if quick else 10
    rounds = 80 if quick else 300
    rows: list[dict] = []
    for condition in ("legacy", "recommended", "recommended-cia", "v3-evidence",
                      "v3-active", "v3-recovery",
                      "v3-active-recovery"):
        for seed in range(seeds):
            key = ("planner", "synthetic", condition, seed)
            if done and key in done:
                continue
            world = grid_world(8, 8, seed=seed, kind="bounded",
                               rho=0.02, noise_scale=0.05)
            start, goal = (0, 0), (7, 7)
            cfg = _planner_condition(condition)
            planner = CertPlanner(world, start, goal, cfg)
            covered = valid = 0
            gaps: list[float] = []
            regrets: list[float] = []
            certs = 0
            elapsed = 0.0
            for _ in range(rounds):
                t = planner.t
                tick = time.perf_counter()
                cert, _ = planner.round()
                elapsed += time.perf_counter() - tick
                snap = _snapshot(world, t)
                _, optimum = dijkstra(snap, start, goal)
                if cert.valid:
                    valid += 1
                    covered += int(cert.lb - 1e-9 <= optimum <= cert.ub + 1e-9)
                    gaps.append(cert.gap)
                    certs += int(cert.gap <= cfg.epsilon)
                if cert.path:
                    regrets.append(_path_cost(snap, cert.path) - optimum)
            lo, hi = _ci(covered, valid)
            rows.append({
                "kind": "planner", "condition": condition, "seed": seed,
                "rounds": rounds, "valid_fraction": valid / rounds,
                "coverage": covered / valid if valid else math.nan,
                "coverage_ci_lo": lo, "coverage_ci_hi": hi,
                "certified_fraction": certs / rounds,
                "gap_median": st.median(gaps) if gaps else math.nan,
                "gap_p90": float(np.percentile(gaps, 90)) if gaps else math.nan,
                "regret_median": st.median(regrets) if regrets else math.nan,
                "regret_mean": st.mean(regrets) if regrets else math.nan,
                "sense_spend": planner.sense_spend,
                "replan_p50_ms": 1000.0 * elapsed / rounds,
                "recovery": planner.recovery_diagnostics(),
            })
            if emit is not None:
                emit(rows[-1])
            print(f"planner {condition} seed={seed} done", flush=True)
    return rows


def traffic_rows(quick: bool, done: set[tuple] | None = None,
                 emit=None, rho_quantile: float = 0.95,
                 dataset_suffix: str = "", conditions: tuple[str, ...] | None = None) -> list[dict]:
    """Paired real-data replay rows using the p95 configuration family."""
    try:
        from certflow.baselines import GaussianCertPlanner
        from certflow.realworld import TrafficWorld, far_endpoints, traffic_planner_config
    except Exception as exc:
        return [{"kind": "traffic", "skipped": True, "reason": repr(exc)}]

    seeds = 3 if quick else 10
    rounds = 60 if quick else 288
    rows: list[dict] = []
    conditions = conditions or ("legacy", "recommended", "recommended-no-sum",
                                "recommended-given", "recommended-evidence", "gaussian")
    for dataset in ("metr-la", "pems-bay"):
        result_dataset = f"{dataset}{dataset_suffix}"
        for condition in conditions:
            for seed in range(seeds):
                key = ("traffic", result_dataset, condition, seed)
                if done and key in done:
                    continue
                try:
                    world = TrafficWorld(dataset=dataset, seed=seed, n_bins=rounds,
                                         rho_quantile=rho_quantile)
                except (FileNotFoundError, OSError, ImportError) as exc:
                    rows.append({"kind": "traffic", "dataset": result_dataset,
                                 "condition": condition, "seed": seed,
                                 "skipped": True, "reason": repr(exc)})
                    continue
                start, goal = far_endpoints(world)
                # Match scripts/run_metr_la.py and the committed full tables:
                # p95/p75 names the drift-rate quantile, while the claim
                # budget remains alpha_prime=0.2.
                common = dict(alpha_prime=0.2, epsilon=120.0)
                if condition in ("legacy", "p75"):
                    cfg = traffic_planner_config(**common)
                    planner_cls = CertPlanner
                elif condition == "recommended":
                    cfg = traffic_planner_config(
                        **common, rho_mode="online", rho_online_quantile=0.5,
                        hybrid_sensing=True,
                        use_kappa=True, kappa_slack_frac=1.0,
                        adaptive_rate=True,
                        sum_aware_ub=True, warmup_sense_per_round=12,
                        max_sense_per_round=12,
                        mean_path_execution=True,
                        latent_margin=1.05,
                    )
                    planner_cls = CertPlanner
                elif condition == "p75-adaptive":
                    # Exact prior competitor: p75 drift quantile plus the
                    # adaptive sensing-rate change, with all v3 policy knobs
                    # otherwise at their legacy defaults.
                    cfg = traffic_planner_config(**common, adaptive_rate=True)
                    planner_cls = CertPlanner
                elif condition in ("recommended-no-sum",):
                    cfg = traffic_planner_config(
                        **common, rho_mode="online", hybrid_sensing=True,
                        use_kappa=True, adaptive_rate=True,
                    )
                    planner_cls = CertPlanner
                elif condition in ("recommended-given", "p75-given"):
                    cfg = traffic_planner_config(
                        **common, hybrid_sensing=True, use_kappa=True,
                        adaptive_rate=True, sum_aware_ub=True,
                        warmup_sense_per_round=12, max_sense_per_round=12,
                        mean_path_execution=True,
                        latent_margin=1.05,
                    )
                    planner_cls = CertPlanner
                elif condition in ("recommended-evidence", "p75-evidence"):
                    evidence_kwargs = dict(
                        **common, hybrid_sensing=True, use_kappa=True,
                        adaptive_rate=True, sum_aware_ub=True,
                        evidence_model=True, warmup_sense_per_round=12,
                        max_sense_per_round=12,
                        latent_margin=1.05,
                    )
                    if condition in ("recommended-evidence", "p75-evidence"):
                        evidence_kwargs["rho_mode"] = "online"
                        evidence_kwargs["rho_online_quantile"] = 0.5
                    if condition == "p75-evidence":
                        # PEMS has a smaller route-regret margin than METR-LA.
                        # Let a certified mean-path candidate compete with the
                        # incumbent over a wider, still certificate-priced
                        # kappa band; this changes execution choice, not the
                        # coverage accounting.
                        evidence_kwargs["mean_path_execution"] = True
                        evidence_kwargs["kappa_slack_frac"] = 2.0
                    cfg = traffic_planner_config(**evidence_kwargs)
                    planner_cls = CertPlanner
                else:
                    cfg = traffic_planner_config(**common)
                    planner_cls = GaussianCertPlanner
                planner = planner_cls(world, start, goal, cfg)
                valid = covered = certified = 0
                gaps: list[float] = []
                regrets: list[float] = []
                wall = 0.0
                for _ in range(rounds):
                    t = planner.t
                    tick = time.perf_counter()
                    cert, _ = planner.round()
                    wall += time.perf_counter() - tick
                    snap = _snapshot(world, t)
                    _, optimum = dijkstra(snap, start, goal)
                    if cert.valid:
                        valid += 1
                        covered += int(cert.lb - 1e-9 <= optimum <= cert.ub + 1e-9)
                        gaps.append(cert.gap)
                        certified += int(cert.gap <= cfg.epsilon)
                    if cert.path:
                        regrets.append(_path_cost(snap, cert.path) - optimum)
                lo, hi = _ci(covered, valid)
                rows.append({
                    "kind": "traffic", "dataset": result_dataset,
                    "condition": condition, "seed": seed, "rounds": rounds,
                    "valid_fraction": valid / rounds,
                    "coverage": covered / valid if valid else math.nan,
                    "coverage_ci_lo": lo, "coverage_ci_hi": hi,
                    "certified_fraction": certified / rounds,
                    "gap_median": st.median(gaps) if gaps else math.nan,
                    "gap_p90": float(np.percentile(gaps, 90)) if gaps else math.nan,
                    "regret_median": st.median(regrets) if regrets else math.nan,
                    "regret_mean": st.mean(regrets) if regrets else math.nan,
                    "sense_spend": planner.sense_spend,
                    "replan_p50_ms": 1000.0 * wall / rounds,
                    "a1_violation_rate": world.a1_violation_rate,
                })
                if emit is not None:
                    emit(rows[-1])
                print(f"traffic {result_dataset} {condition} seed={seed} done", flush=True)
    return rows


def primitive_rows() -> list[dict]:
    """Exercise each upgrade with a validity-scope-aware microbenchmark."""
    rows: list[dict] = []
    ledger = SelectionLedger()
    event = ledger.record("chosen", ["a", "b"], context=[1.0, 2.0])
    sc = SelectionConditionalCalibrator(min_audit=5).certify(
        event.selected, 10.0, [0.1, 0.2, 0.15, 0.3, 0.05, 0.2],
        alpha=0.1, ledger=ledger)
    rows.append({"kind": "selection", "valid": sc.valid,
                 "confidence": sc.confidence, "radius": sc.radius,
                 "audit_count": sc.audit_count})

    rng = np.random.default_rng(11)
    features = rng.normal(size=(80, 6))
    scale = np.exp(0.15 * features[:, 0])
    residuals = scale * rng.normal(size=80)
    ec = EvidenceConditionalCalibrator(min_calibration=20)
    ec.fit(features[:40], residuals[:40], features[40:], residuals[40:])
    rows.append({"kind": "evidence", "valid": ec.ready,
                 "median_radius": float(np.median([
                     ec.radius(x, 0.1) for x in features[40:]]))})

    risk = DecisionRiskController(delta=0.05).select(
        {"safe": [0.0] * 120, "cheap": [1.0] * 120},
        {"safe": 5.0, "cheap": 1.0}, target_risk=0.1)
    rows.append({"kind": "decision-risk", "valid": risk.valid,
                 "selected": risk.action, "risk_bound": risk.risk_bound})

    pred = np.zeros((30, 6, 2))
    obs = pred + rng.normal(0.0, 0.05, size=pred.shape)
    tc = TrajectoryConformalCalibrator(min_calibration=20).fit(pred, obs)
    tube = tc.tube(pred[0], 0.1)
    rows.append({"kind": "trajectory", "valid": tube.valid,
                 "contains_calibration_point": tube.contains(obs[0]),
                 "radius_max": float(np.max(tube.radius))})

    policy = ActiveSensingPolicy(exploration=0.25)
    actions = [SensingAction("high", 2.0, 1.0), SensingAction("low", 0.5, 1.0)]
    first = policy.select(actions)
    if first:
        policy.update(first.key, 1.0)
    policy.update("high", 4.0)
    chosen = policy.select(actions)
    rows.append({"kind": "active-sensing", "valid": chosen is not None,
                 "chosen": repr(chosen.key) if chosen else None,
                 # Grid-edge keys are tuples; stringify them at the artifact
                 # boundary rather than weakening the in-memory policy API.
                 "stats": {repr(k): v for k, v in policy.stats().items()}})

    manager = RegimeRecoveryManager(alarm_delta=0.01, recovery_samples=4,
                                    evidence_threshold=2.0)
    for _ in range(4):
        manager.observe_pvalue(1e-8)
    alarmed = not manager.certificate_allowed
    for _ in range(4):
        manager.observe_pvalue(1.0)
    rows.append({"kind": "recovery", "valid": alarmed and manager.certificate_allowed,
                 "alarmed": alarmed, "final_state": manager.state})

    fleet = JointFleetCalibrator(mode="sum", min_calibration=20)
    matrix = rng.normal(0.0, 0.2, size=(40, 3))
    fleet.fit(matrix)
    jc = joint_fleet_certificate(
        fleet, [10.0, 11.0, 12.0], alpha=0.1,
        route_loads={"shared": 3.0}, capacities={"shared": 2.0},
        congestion_coefficient=2.0)
    rows.append({"kind": "fleet", "valid": jc.valid,
                 "gap": jc.gap, "congestion_cost": jc.congestion_cost})
    rows.append({"kind": "fleet-penalty", "valid": congestion_penalty(
        {"r": 1.0}, {"r": 1.0}) == 0.0})
    return rows


def aggregate(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    groups = sorted({(r["kind"], r.get("dataset", "synthetic"), r["condition"])
                     for r in rows if r.get("condition") and not r.get("skipped")})
    for kind, dataset, condition in groups:
        rs = [r for r in rows if r.get("kind") == kind
              and r.get("dataset", "synthetic") == dataset
              and r.get("condition") == condition]
        key = f"{dataset}/{condition}" if kind == "traffic" else condition
        out[key] = {
            "seeds": len(rs),
            "valid_fraction_mean": st.mean(r["valid_fraction"] for r in rs),
            "coverage_mean": st.mean(r["coverage"] for r in rs),
            "coverage_ci_lo_min": min(r["coverage_ci_lo"] for r in rs),
            "gap_median_median": st.median(r["gap_median"] for r in rs),
            "gap_p90_median": st.median(r["gap_p90"] for r in rs),
            "regret_median_median": st.median(r["regret_median"] for r in rs),
            "sense_spend_mean": st.mean(r["sense_spend"] for r in rs),
            "replan_p50_ms_mean": st.mean(r["replan_p50_ms"] for r in rs),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--p75-only", action="store_true",
                    help="run only the matched rho-quantile=0.75 sweep")
    ap.add_argument("--p95-only", action="store_true",
                    help="run only the matched rho-quantile=0.95 traffic sweep")
    ap.add_argument("--synthetic-only", action="store_true",
                    help="run only the synthetic planner and primitive scorecard")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing checkpoint for this run")
    args = ap.parse_args()
    outdir = ROOT / "results" / "v3_scorecard"
    outdir.mkdir(parents=True, exist_ok=True)
    if sum(bool(x) for x in (args.p75_only, args.p95_only, args.synthetic_only)) > 1:
        raise SystemExit("choose at most one scorecard subset")
    if args.p75_only:
        label = "p75-quick" if args.quick else "p75-full"
    elif args.p95_only:
        label = "p95-quick" if args.quick else "p95-full"
    elif args.synthetic_only:
        label = "synthetic-quick" if args.quick else "synthetic-full"
    else:
        label = "quick" if args.quick else "full"
    partial = outdir / f"{label}.partial.json"
    existing: list[dict] = []
    if partial.exists() and not args.fresh:
        try:
            existing = json.loads(partial.read_text(encoding="utf-8")).get("rows", [])
        except (OSError, ValueError):
            existing = []
    if args.p95_only:
        existing = [r for r in existing
                    if not str(r.get("dataset", "")).endswith("-rho75")]
    elif args.p75_only:
        existing = [r for r in existing
                    if str(r.get("dataset", "")).endswith("-rho75")]
    # Keep one copy of every checkpointed row.  A rerun resumes completed
    # cells automatically; changing the condition list only adds new cells.
    checkpoint_rows = {_row_key(r): r for r in existing}
    done = set(checkpoint_rows)

    def emit(row: dict) -> None:
        checkpoint_rows[_row_key(row)] = row
        partial.write_text(json.dumps(_json_safe({"rows": list(checkpoint_rows.values())}),
                                      indent=2, default=str), encoding="utf-8")

    # Primitive checks are cheap and are regenerated each invocation so code
    # changes cannot leave their diagnostics stale.
    for row in primitive_rows():
        checkpoint_rows[(_row_key(row)[0], "primitive", row["kind"], None)] = row
    rows = list(checkpoint_rows.values())
    if args.synthetic_only:
        rows.extend(planner_rows(args.quick, done=done, emit=emit))
    elif not args.p75_only and not args.p95_only:
        rows.extend(planner_rows(args.quick, done=done, emit=emit))
        rows.extend(traffic_rows(args.quick, done=done, emit=emit))
    elif args.p95_only:
        rows.extend(traffic_rows(args.quick, done=done, emit=emit))
    if not args.p95_only and not args.synthetic_only:
        rows.extend(traffic_rows(
            args.quick, done=done, emit=emit, rho_quantile=0.75,
            dataset_suffix="-rho75",
            conditions=("p75", "p75-adaptive", "p75-given", "p75-evidence", "gaussian"),
        ))
    # The callbacks already inserted new rows; de-duplicate the returned rows
    # before producing the final artifact.
    rows = list({_row_key(r): r for r in rows}.values())
    payload = {
        "config": {"quick": args.quick, "python": sys.version,
                   "platform": platform.platform(), "revision": _revision(),
                   "manifest": _manifest()},
        "aggregate": aggregate(rows), "rows": rows,
    }
    if args.synthetic_only:
        outfile = outdir / ("synthetic-quick.json" if args.quick else "synthetic-full.json")
    elif args.p75_only:
        outfile = outdir / ("p75-quick.json" if args.quick else "p75-full.json")
    elif args.p95_only:
        outfile = outdir / ("p95-quick.json" if args.quick else "p95-full.json")
    else:
        outfile = outdir / ("quick.json" if args.quick else "full.json")
    outfile.write_text(json.dumps(_json_safe(payload), indent=2, default=str), encoding="utf-8")
    if partial.exists():
        partial.unlink()
    print(json.dumps(payload["aggregate"], indent=2, default=str))
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main()
