"""Full real-traffic conditional and post-selection audit.

Selection happens during a planner warm-up.  Only after the route is fixed do
we collect a fresh audit stream.  The first audit block calibrates the selected
route and selected-edge group certificates; the second block is never used for
calibration and supplies coverage evidence.
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
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import source_manifest  # noqa: E402

from certflow.cert import CertPlanner  # noqa: E402
from certflow.graphcore import dijkstra  # noqa: E402
from certflow.realworld import (  # noqa: E402
    TrafficWorld,
    far_endpoints,
    traffic_planner_config,
)
from certflow.upgrades import (  # noqa: E402
    GroupConditionalCalibrator,
    SelectionConditionalCalibrator,
    SelectionLedger,
)


def _selected_path(planner: CertPlanner) -> list:
    path = list(getattr(planner, "_prev_incumbent", []) or [])
    if len(path) >= 2 and path[-1] == planner.goal:
        return path
    path, _ = planner._mean_search.shortest_path()
    return list(path or [])


def _run_seed(dataset: str, seed: int, probe_cycles: int) -> dict:
    world = TrafficWorld(dataset=dataset, seed=seed, n_bins=288, rho_quantile=0.95)
    start, goal = far_endpoints(world)
    cfg = traffic_planner_config(
        rho_mode="online", rho_online_quantile=0.5,
        hybrid_sensing=True, use_kappa=True, adaptive_rate=True,
        sum_aware_ub=True, warmup_sense_per_round=12,
        max_sense_per_round=12, mean_path_execution=True,
        latent_margin=1.05, selection_conditional=True,
        selection_audit_min=20,
    )
    planner = CertPlanner(world, start, goal, cfg)
    for _ in range(96):
        planner.round()
    path = _selected_path(planner)
    edges = list(zip(path, path[1:]))
    if not edges:
        return {"dataset": dataset, "seed": seed, "status": "OPEN", "reason": "no selected path"}

    # Selection is complete before any audit observation below.
    ledger = SelectionLedger()
    event = ledger.record(tuple(path), [tuple(path)], context=[float(len(edges))])
    point = sum(planner.beliefs[e].c_hat for e in edges)
    cal_route: list[float] = []
    test_route: list[float] = []
    cal_groups: list[object] = []
    cal_residuals: list[float] = []
    test_groups: list[object] = []
    test_residuals: list[float] = []
    base_t = planner.t
    for cycle in range(2 * probe_cycles):
        t = base_t + cycle * world._BIN_SECONDS if hasattr(world, "_BIN_SECONDS") else base_t + cycle * 300.0
        # TrafficWorld is finite; keep the audit inside its recorded horizon.
        t = min(t, (world._speeds.shape[0] - 1.001) * 300.0)
        route_residual = 0.0
        for edge in edges:
            observation = world.observe(edge, t)
            residual = observation - planner.beliefs[edge].c_hat
            route_residual += residual
            if cycle < probe_cycles:
                cal_groups.append(edge)
                cal_residuals.append(residual)
            else:
                test_groups.append(edge)
                test_residuals.append(residual)
        (cal_route if cycle < probe_cycles else test_route).append(route_residual)

    route_cal = SelectionConditionalCalibrator(min_audit=max(20, probe_cycles // 2))
    # The selected point is frozen during the independent audit.  Traffic is
    # nonstationary, so the audit certificate must pay the declared A1 drift
    # allowance from selection time to the end of the audit; otherwise this is
    # a fresh-selection audit with an exchangeability claim the data cannot
    # support.
    audit_horizon = 2.0 * probe_cycles * 300.0
    route_drift_margin = sum(
        world.rho_true(edge) * audit_horizon for edge in edges
    )
    route_cert = route_cal.certify(
        event.selected, point, cal_route, alpha=0.2, ledger=ledger,
        drift_margin=route_drift_margin,
    )
    route_truth = [
        sum(world.true_cost(edge, base_t + (probe_cycles + i) * 300.0) for edge in edges)
        for i in range(probe_cycles)
    ]
    route_covered = [
        route_cert.lower - 1e-9 <= truth <= route_cert.upper + 1e-9
        for truth in route_truth
    ]

    groups = GroupConditionalCalibrator(min_support=max(20, probe_cycles // 2))
    groups.fit(cal_groups, cal_residuals)
    group_certificates = {
        edge: groups.certify(
            edge, planner.beliefs[edge].c_hat, alpha=0.2,
            drift_margin=world.rho_true(edge) * audit_horizon,
        )
        for edge in edges
    }
    group_report = groups.coverage_report(
        test_groups,
        [
            abs(residual) <= group_certificates[edge].radius + 1e-12
            for edge, residual in zip(test_groups, test_residuals)
        ],
        alpha=0.2,
        target_coverage=0.8,
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "status": "PASS",
        "selected_path_edges": len(edges),
        "selection_digest_matches": route_cert.selection_digest == ledger.digest,
        "selection_audit_count": route_cert.audit_count,
        "selection_certificate_valid": route_cert.valid,
        "selection_audit_coverage": sum(route_covered) / len(route_covered),
        "selection_audit_lcb": max(
            0.0,
            sum(route_covered) / len(route_covered)
            - math.sqrt(math.log(1.0 / 0.05) / (2.0 * len(route_covered))),
        ),
        "group_count": len(group_report),
        "group_all_valid": all(item.valid for item in group_report.values()),
        "group_min_lcb": min(
            item.lower_confidence_bound for item in group_report.values()
        ),
        "group_min_coverage": min(item.coverage for item in group_report.values()),
        "group_reports": {
            repr(edge): {
                "support": item.support,
                "coverage": item.coverage,
                "lower_confidence_bound": item.lower_confidence_bound,
                "valid": item.valid,
            }
            for edge, item in group_report.items()
        },
    }


def _aggregate(rows: list[dict]) -> dict:
    out = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        rs = [row for row in rows if row["dataset"] == dataset]
        out[dataset] = {
            "seeds": len(rs),
            "selection_certificate_valid_fraction": st.mean(
                bool(row.get("selection_certificate_valid")) for row in rs
            ),
            "selection_digest_match_fraction": st.mean(
                bool(row.get("selection_digest_matches")) for row in rs
            ),
            "selection_audit_coverage_mean": st.mean(
                row.get("selection_audit_coverage", math.nan) for row in rs
            ),
            "selection_audit_lcb_min": min(
                row.get("selection_audit_lcb", math.nan) for row in rs
            ),
            "group_all_valid_fraction": st.mean(
                bool(row.get("group_all_valid")) for row in rs
            ),
            "group_min_lcb_min": min(row.get("group_min_lcb", math.nan) for row in rs),
            "group_min_coverage_min": min(
                row.get("group_min_coverage", math.nan) for row in rs
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--probe-cycles", type=int, default=60)
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.probe_cycles = 2, 30
    rows = [
        _run_seed(dataset, seed, args.probe_cycles)
        for dataset in ("metr-la", "pems-bay")
        for seed in range(args.seeds)
    ]
    aggregate = _aggregate(rows)
    gates = {
        "full_provenance": not args.quick and args.seeds == 10 and args.probe_cycles == 60,
        "all_rows_completed": all(row.get("status") == "PASS" for row in rows),
        "selection_audits_valid": all(
            row.get("selection_certificate_valid")
            and row.get("selection_digest_matches")
            and row.get("selection_audit_lcb", 0.0) >= 0.8
            for row in rows
        ),
        "group_audits_valid": all(
            row.get("group_all_valid") and row.get("group_min_lcb", 0.0) >= 0.8
            for row in rows
        ),
    }
    out = {
        "schema": "certflow.v4.realworld-audit.1",
        "quick": args.quick,
        "seeds": args.seeds,
        "probe_cycles": args.probe_cycles,
        "datasets": ["metr-la", "pems-bay"],
        "manifest": source_manifest(ROOT),
        "aggregate": aggregate,
        "rows": rows,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "OPEN",
    }
    target = ROOT / "results" / "v4_validity" / (
        "realworld-audit-quick.json" if args.quick else "realworld-audit-full.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"wrote {target}")
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
