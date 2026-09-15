"""Audit the CERT-FLOW v3 promotion gates from persisted full-run artifacts.

This is intentionally a checker, not a score aggregator.  It refuses to call a
quick artifact a release result, compares promoted conditions against the
recorded baselines, and reports open gates instead of hiding objective
tradeoffs in one composite number.

Usage::

    python scripts/check_v3_gate.py
    python scripts/check_v3_gate.py --strict  # exit 1 if any gate is open
    python scripts/check_v3_gate.py --json results/v3_scorecard/gate.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_COVERAGE_LCB = 0.80
FULL_SEEDS = 10


def load(relative: str) -> Any:
    path = ROOT / relative
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def row(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return payload["aggregate"][name]


def check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    evidence: str,
    *,
    severity: str = "release",
) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else "OPEN",
            "severity": severity,
            "evidence": evidence,
        }
    )


def content_bound_manifest(payload: Any) -> dict[str, Any] | None:
    """Find a source manifest that identifies dirty source content."""
    if not isinstance(payload, dict):
        return None
    manifest = payload.get("manifest")
    if manifest is None and isinstance(payload.get("config"), dict):
        manifest = payload["config"].get("manifest")
    if not isinstance(manifest, dict):
        return None
    if not isinstance(manifest.get("source_digest"), str):
        return None
    if not isinstance(manifest.get("source_file_count"), int):
        return None
    if manifest["source_file_count"] < 1:
        return None
    return manifest


def compare_traffic(
    checks: list[dict[str, Any]],
    payload: dict[str, Any],
    dataset: str,
    promoted: str,
    baselines: list[str],
) -> None:
    p = row(payload, promoted)
    label = f"{dataset} / {promoted}"
    full = payload.get("config", {}).get("quick") is False and p.get("seeds") == FULL_SEEDS
    check(
        checks,
        f"{label}: full-run provenance",
        full,
        f"quick={payload.get('config', {}).get('quick')!r}, seeds={p.get('seeds')!r}",
    )
    cov = p.get("coverage_ci_lo_min")
    check(
        checks,
        f"{label}: coverage lower bound",
        finite(cov) and cov >= TARGET_COVERAGE_LCB,
        f"LCB={cov:.4f} >= {TARGET_COVERAGE_LCB:.2f}" if finite(cov) else f"LCB={cov!r}",
    )
    gap = p.get("gap_median_median")
    regret = p.get("regret_median_median")
    for baseline in baselines:
        b = row(payload, baseline)
        bgap = b.get("gap_median_median")
        bregret = b.get("regret_median_median")
        check(
            checks,
            f"{label}: median gap < {baseline}",
            finite(gap) and finite(bgap) and gap < bgap,
            f"{gap:.3f} < {bgap:.3f}" if finite(gap) and finite(bgap) else f"{gap!r} vs {bgap!r}",
        )
        check(
            checks,
            f"{label}: median regret < {baseline}",
            finite(regret) and finite(bregret) and regret < bregret,
            f"{regret:.3f} < {bregret:.3f}" if finite(regret) and finite(bregret) else f"{regret!r} vs {bregret!r}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="return 1 when any release gate is open")
    parser.add_argument("--json", type=Path, help="also write the audit report to this path")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    p95 = load("results/v3_scorecard/p95-full.json")
    p75 = load("results/v3_scorecard/p75-full.json")
    synthetic = load("results/v3_scorecard/synthetic-full.json")
    width = load("scripts/out/width_attack.json")
    scale = load("results/scale/table.json")
    movingai = load("results/movingai/table.json")
    routing = load("results/roadnet/ch_ny_sub50000.json")
    fleet = load("results/fleet/correlated-student_t-full.json")
    v4_validity = load("results/v4_validity/audit.json")
    shift_recovery = load("results/v4_validity/shift-recovery-full.json")
    shift_stress = load("results/v4_validity/shift_stress-full.json")
    realworld_audit = load("results/v4_validity/realworld-audit-full.json")
    fleet_sensing = load("results/v4_validity/fleet-sensing-full.json")
    extern = load("results/extern_baselines/table.json")

    # A revision/complete flag alone cannot bind a persisted result to a dirty
    # checkout.  These are the artifacts used for promotion-facing claims.
    for artifact_name, artifact in (
        ("p95 scorecard", p95),
        ("p75 scorecard", p75),
        ("synthetic scorecard", synthetic),
        ("validity audit", v4_validity),
        ("shift recovery", shift_recovery),
        ("real-world audit", realworld_audit),
        ("budgeted sensing", fleet_sensing),
    ):
        manifest = content_bound_manifest(artifact)
        check(
            checks,
            f"{artifact_name}: content-bound source provenance",
            manifest is not None,
            (
                f"digest={manifest.get('source_digest', '')[:12]}, "
                f"files={manifest.get('source_file_count')}"
                if manifest is not None
                else "missing source_digest/source_file_count"
            ),
        )

    if v4_validity is None:
        check(checks, "v4 validity audit present", False,
              "missing results/v4_validity/audit.json")
    else:
        check(
            checks,
            "v4 finite-family conditional validity",
            v4_validity.get("gates", {}).get("finite_family_group_certificates") is True,
            f"status={v4_validity.get('status')!r}",
        )

    if extern is None:
        check(checks, "external decision-quality battery present", False,
              "missing results/extern_baselines/table.json")
    else:
        part_b = {item.get("policy"): item for item in (extern.get("part_b") or [])}
        full_external = (
            extern.get("quick") is False
            and extern.get("seeds_b") == 15
            and extern.get("rounds_b") == 600
        )
        check(
            checks,
            "external decision-quality battery is full",
            full_external,
            f"quick={extern.get('quick')!r}, seeds={extern.get('seeds_b')!r}, rounds={extern.get('rounds_b')!r}",
        )
        check(
            checks,
            "external battery manifest complete",
            extern.get("manifest", {}).get("complete") is True,
            f"manifest={extern.get('manifest', {})}",
        )
        required = {"cert", "voi", "hybrid", "max_age", "max_width"}
        complete = required <= set(part_b)
        check(
            checks,
            "external decision-quality rows complete",
            complete and all(
                math.isfinite(float(part_b[p].get("regret_mean")))
                and part_b[p].get("goal", 0.0) > 0.0
                for p in required
            ),
            f"policies={sorted(part_b)}, goals={{p: part_b[p].get('goal') for p in sorted(part_b)}}",
        )
        if complete:
            competitors = required - {"cert", "hybrid"}
            strongest = min(float(part_b[p]["regret_mean"]) for p in competitors)
            check(
                checks,
                "CERT-FLOW hybrid regret beats strongest decision baseline",
                math.isfinite(float(part_b["hybrid"].get("regret_mean")))
                and float(part_b["hybrid"]["regret_mean"]) < strongest,
                f"hybrid={part_b['hybrid'].get('regret_mean')!r} < strongest={strongest:.6f}",
            )

    if v4_validity is not None:
        check(
            checks,
            "v4 formal recovery alarm/re-entry",
            v4_validity.get("gates", {}).get("formal_recovery_alarm_and_reentry") is True,
            f"status={v4_validity.get('status')!r}",
        )

    if shift_recovery is None:
        check(checks, "v4 full shift-recovery artifact present", False,
              "missing results/v4_validity/shift-recovery-full.json")
    else:
        gates = shift_recovery.get("gates", {})
        check(checks, "v4 shift-recovery full provenance",
              gates.get("full_provenance") is True,
              f"quick={shift_recovery.get('quick')!r}, seeds={shift_recovery.get('seeds')!r}, rounds={shift_recovery.get('rounds')!r}")
        check(checks, "v4 formal recovery no pre-shift false alarms",
              gates.get("formal_no_pre_shift_false_alarms") is True,
              f"formal_false_alarm_fraction={shift_recovery.get('aggregate', {}).get('formal', {}).get('pre_false_alarm_fraction')!r}")
        check(checks, "v4 formal recovery detects and re-enters",
              gates.get("formal_detects_shift") is True and gates.get("formal_reenters") is True,
              f"formal={shift_recovery.get('aggregate', {}).get('formal', {})}")
        aci_post = shift_recovery.get("aggregate", {}).get("aci", {}).get("post_edge_coverage_mean")
        formal_post = shift_recovery.get("aggregate", {}).get("formal", {}).get("post_edge_coverage_mean")
        check(checks, "v4 formal recovery improves ACI post-shift edge coverage",
              finite(formal_post) and finite(aci_post) and formal_post > aci_post,
              f"formal={formal_post!r} > aci={aci_post!r}")
        check(checks, "v4 accelerated recovery gate",
              shift_recovery.get("recovery_samples") == 5
              and shift_recovery.get("scalar_formal") is True
              and finite(shift_recovery.get("aggregate", {}).get("formal", {}).get("recovery_delay_median"))
              and shift_recovery["aggregate"]["formal"]["recovery_delay_median"] <= 5
              and shift_recovery["aggregate"]["formal"]["recovery_to_valid_median"] <= 5,
              f"fresh_samples={shift_recovery.get('recovery_samples')!r}, "
              f"recovery={shift_recovery.get('aggregate', {}).get('formal', {}).get('recovery_delay_median')!r}, "
              f"valid_reentry={shift_recovery.get('aggregate', {}).get('formal', {}).get('recovery_to_valid_median')!r}")

    if shift_stress is None:
        check(checks, "v4 full calibration-shift stress artifact present", False,
              "missing results/v4_validity/shift_stress-full.json", severity="open-experiment")
    else:
        check(checks, "v4 calibration-shift stress full provenance",
              shift_stress.get("quick") is False
              and shift_stress.get("seeds") == 12
              and shift_stress.get("rounds") == 300
              and shift_stress.get("manifest", {}).get("complete") is True,
              f"quick={shift_stress.get('quick')!r}, seeds={shift_stress.get('seeds')!r}, rounds={shift_stress.get('rounds')!r}",
              severity="open-experiment")

    if realworld_audit is None:
        check(checks, "v4 full real-world conditional audit present", False,
              "missing results/v4_validity/realworld-audit-full.json")
    else:
        gates = realworld_audit.get("gates", {})
        check(checks, "v4 real-world audit full provenance",
              gates.get("full_provenance") is True,
              f"quick={realworld_audit.get('quick')!r}, seeds={realworld_audit.get('seeds')!r}, probe_cycles={realworld_audit.get('probe_cycles')!r}")
        check(checks, "v4 real-world audit rows completed",
              gates.get("all_rows_completed") is True,
              f"status={realworld_audit.get('status')!r}")
        check(checks, "v4 real-world post-selection audits valid",
              gates.get("selection_audits_valid") is True,
              f"gate={gates.get('selection_audits_valid')!r}")
        check(checks, "v4 real-world group audits valid",
              gates.get("group_audits_valid") is True,
              f"gate={gates.get('group_audits_valid')!r}")
        check(
            checks,
            "v4 fresh post-selection audit",
            v4_validity.get("gates", {}).get("fresh_post_selection_audit") is True,
            f"status={v4_validity.get('status')!r}",
        )

    if p95 is None:
        check(checks, "p95 scorecard present", False, "missing results/v3_scorecard/p95-full.json")
    else:
        manifest = p95.get("config", {}).get("manifest", {})
        check(checks, "p95 scorecard manifest complete",
              manifest.get("complete") is True and bool(manifest.get("data_files")),
              f"manifest_complete={manifest.get('complete')!r}, data_files={len(manifest.get('data_files', []))}")
        compare_traffic(checks, p95, "METR-LA", "metr-la/recommended", ["metr-la/legacy", "metr-la/gaussian"])
        compare_traffic(checks, p95, "PEMS-BAY", "pems-bay/recommended", ["pems-bay/legacy", "pems-bay/gaussian"])

    if p75 is None:
        check(checks, "p75 scorecard present", False, "missing results/v3_scorecard/p75-full.json")
    else:
        manifest = p75.get("config", {}).get("manifest", {})
        check(checks, "p75 scorecard manifest complete",
              manifest.get("complete") is True and bool(manifest.get("data_files")),
              f"manifest_complete={manifest.get('complete')!r}, data_files={len(manifest.get('data_files', []))}")
        # These are specialist regimes: evidence/given wins on METR, while
        # the p75 baseline remains the sharper PEMS choice.  Keep both facts.
        compare_traffic(checks, p75, "METR-LA", "metr-la-rho75/p75-given", ["metr-la-rho75/p75", "metr-la-rho75/gaussian"])
        compare_traffic(checks, p75, "PEMS-BAY", "pems-bay-rho75/p75-evidence", ["pems-bay-rho75/gaussian"])
        pems = row(p75, "pems-bay-rho75/p75-evidence")
        base = row(p75, "pems-bay-rho75/p75")
        check(
            checks,
            "p75 PEMS specialist: regret < p75 baseline",
            pems["regret_median_median"] < base["regret_median_median"],
            f"{pems['regret_median_median']:.3f} < {base['regret_median_median']:.3f}; gap tradeoff is retained explicitly",
        )
        check(
            checks,
            "p75 PEMS specialist: median gap < p75 baseline",
            pems["gap_median_median"] < base["gap_median_median"],
            f"{pems['gap_median_median']:.3f} < {base['gap_median_median']:.3f}",
        )

    if synthetic is None:
        check(checks, "synthetic scorecard present", False, "missing results/v3_scorecard/synthetic-full.json")
    else:
        manifest = synthetic.get("config", {}).get("manifest", {})
        check(checks, "synthetic scorecard manifest complete",
              manifest.get("complete") is True and bool(manifest.get("data_files")),
              f"manifest_complete={manifest.get('complete')!r}, data_files={len(manifest.get('data_files', []))}")
        s = synthetic["aggregate"]
        r = s["recommended"]
        check(checks, "synthetic recommended: full-run provenance", synthetic.get("config", {}).get("quick") is False and r.get("seeds") == FULL_SEEDS, f"quick={synthetic.get('config', {}).get('quick')!r}, seeds={r.get('seeds')!r}")
        check(checks, "synthetic recommended: coverage lower bound", r.get("coverage_ci_lo_min", 0) >= TARGET_COVERAGE_LCB, f"LCB={r.get('coverage_ci_lo_min')}")
        check(checks, "synthetic recommended: gap_median_median < legacy", r["gap_median_median"] < s["legacy"]["gap_median_median"], f"{r['gap_median_median']:.4f} < {s['legacy']['gap_median_median']:.4f}")
        check(checks, "synthetic recommended: regret < legacy", r["regret_median_median"] < s["legacy"]["regret_median_median"], f"{r['regret_median_median']:.4f} < {s['legacy']['regret_median_median']:.4f}")
        # Evidence-conditioned pricing remains available as a stronger
        # decision-quality portfolio member, but the promoted lane also wins
        # the primary synthetic regret comparison.
        e = s["v3-evidence"]
        check(checks, "synthetic evidence: coverage lower bound", e.get("coverage_ci_lo_min", 0) >= TARGET_COVERAGE_LCB, f"LCB={e.get('coverage_ci_lo_min')}")
        check(checks, "synthetic evidence: gap < legacy", e["gap_median_median"] < s["legacy"]["gap_median_median"], f"{e['gap_median_median']:.4f} < {s['legacy']['gap_median_median']:.4f}")
        check(checks, "synthetic evidence: regret < legacy", e["regret_median_median"] < s["legacy"]["regret_median_median"], f"{e['regret_median_median']:.4f} < {s['legacy']['regret_median_median']:.4f}")

    if width is None:
        check(checks, "width attack present", False, "missing scripts/out/width_attack.json")
    else:
        metr = {x["mode"]: x for x in width["metr_la"]}
        grid = {x["mode"]: x for x in width["grid"]}
        check(checks, "width attack is full", not width.get("quick", True) and width.get("seeds") == FULL_SEEDS, f"quick={width.get('quick')!r}, seeds={width.get('seeds')!r}")
        check(checks, "METR sum-aware: no validity violations", metr["sum_aware"]["violation_rate"] == 0.0, f"violation_rate={metr['sum_aware']['violation_rate']}")
        check(checks, "METR sum-aware: tighter certified gap", metr["sum_aware"]["gap_ratio"] < 1.0, f"ratio={metr['sum_aware']['gap_ratio']:.4f}")
        check(checks, "grid stress: no validity violations", all(x["violation_rate"] == 0.0 for x in grid.values()), "all five modes report 0.0")

    if scale is None:
        check(checks, "scale scorecard present", False, "missing results/scale/table.json")
    else:
        recommended = {x["size"]: x for x in scale if x["config"] == "C_recommended"}
        for size, item in recommended.items():
            # 0.80 is the declared target; 60x60 is deliberately an open
            # scaling gate until the planner keeps validity at that size.
            check(checks, f"scale {size}: declared validity", item["valid_pct"] / 100.0 >= TARGET_COVERAGE_LCB, f"valid={item['valid_pct']:.1f}%")
        check(checks, "scale 60x60: latency budget", recommended["60x60"]["p95_ms"] <= 50.0, f"p95={recommended['60x60']['p95_ms']:.3f} ms <= 50 ms")

    if movingai is None:
        check(checks, "MovingAI scorecard present", False, "missing results/movingai/table.json")
    else:
        cert = [x for x in movingai if x["policy"] == "cert"]
        check(checks, "MovingAI cert: every map reached goal", bool(cert) and all(x["goal_frac"] == 1.0 for x in cert), f"goal_fracs={[x['goal_frac'] for x in cert]}")
        check(checks, "MovingAI cert: every map covered", bool(cert) and all(x["coverage_valid"] >= TARGET_COVERAGE_LCB for x in cert), f"coverage={[x['coverage_valid'] for x in cert]}")

    if routing is None:
        check(checks, "routing native-query gate", False, "missing persisted results/roadnet/ch_ny_sub50000.json", severity="open-systems")
    else:
        check(checks, "routing CH exactness", routing.get("ch_mismatches") == 0, f"mismatches={routing.get('ch_mismatches')}")
        check(checks, "routing bounded-change exactness", routing.get("potential_perturbed_mismatches") == 0, f"mismatches={routing.get('potential_perturbed_mismatches')}")
        p50 = routing.get("ch_cost_ms", {}).get("p50")
        reference = routing.get("published_ch_p50_ms", 0.110)
        check(checks, "routing CH p50 <= published reference", finite(p50) and p50 <= reference, f"p50={p50:.4f} ms <= {reference:.3f} ms" if finite(p50) else f"p50={p50!r}")
    if fleet is None:
        check(checks, "correlated fleet full release gate", False, "missing results/fleet/correlated-student_t-full.json", severity="open-experiment")
    else:
        corr = fleet.get("correlated", {})
        indep = fleet.get("independent", {})
        full = fleet.get("seeds") == 12 and fleet.get("rounds") == 250
        check(checks, "correlated fleet: full stress provenance", full, f"seeds={fleet.get('seeds')}, rounds={fleet.get('rounds')}")
        check(checks, "correlated fleet: coverage lower bound", corr.get("coverage", 0) >= TARGET_COVERAGE_LCB, f"correlated coverage={corr.get('coverage')}")
        check(checks, "correlated fleet: no correlated coverage drop", corr.get("coverage") >= indep.get("coverage", 0), f"correlated={corr.get('coverage')} independent={indep.get('coverage')}")
        check(checks, "correlated fleet: path-gap finite", finite(corr.get("gap_median")), f"gap_median={corr.get('gap_median')}")

    if fleet_sensing is None:
        check(checks, "v4 full fleet/sensing audit present", False,
              "missing results/v4_validity/fleet-sensing-full.json", severity="open-experiment")
    else:
        fs_rows = fleet_sensing.get("rows", [])
        fs_full = (fleet_sensing.get("quick") is False
                   and fleet_sensing.get("seeds") == 12
                   and fleet_sensing.get("rounds") == 600)
        check(checks, "v4 fleet/sensing full provenance", fs_full,
              f"quick={fleet_sensing.get('quick')!r}, seeds={fleet_sensing.get('seeds')!r}, rounds={fleet_sensing.get('rounds')!r}")
        check(checks, "v4 fleet/sensing manifest complete",
              fleet_sensing.get("manifest", {}).get("complete") is True,
              f"complete={fleet_sensing.get('manifest', {}).get('complete')!r}, "
              f"data_files={len(fleet_sensing.get('manifest', {}).get('data_files', []))}")
        required = {f"{p}|B={b:.0f}" for p in ("cert", "hybrid", "voi", "random", "max_age", "max_width") for b in (10.0, 20.0, 40.0)}
        labels = {r.get("label") for r in fs_rows}
        complete = required <= labels and all(r.get("failures", 1) == 0 for r in fs_rows)
        check(checks, "v4 fleet/sensing rows complete", complete,
              f"rows={len(fs_rows)}, required={len(required)}")
        by = {r.get("label"): r for r in fs_rows}
        valid_policies = ["cert", "hybrid", "voi", "random", "max_age", "max_width"]
        check(checks, "v4 fleet/sensing valid coverage LCBs",
              complete and all(float(by[f"{p}|B={b:.0f}"]["coverage_lcb"]) >= TARGET_COVERAGE_LCB
                               for p in valid_policies for b in (10.0, 20.0, 40.0)),
              "all budget/policy LCBs >= 0.80")
        for budget in (10.0, 20.0, 40.0):
            h = by.get(f"hybrid|B={budget:.0f}")
            competitors = [by.get(f"{p}|B={budget:.0f}") for p in ("voi", "random", "max_age", "max_width")]
            check(checks, f"v4 hybrid regret beats controls at B={budget:.0f}",
                  h is not None and all(c is not None and float(h["regret_mean"]) < float(c["regret_mean"]) for c in competitors),
                  f"hybrid={h.get('regret_mean') if h else None}, controls={[c.get('regret_mean') if c else None for c in competitors]}")

    report = {
        "status": "PASS" if all(x["status"] == "PASS" for x in checks if x["severity"] == "release") else "OPEN",
        "declared_target_coverage_lcb": TARGET_COVERAGE_LCB,
        "checks": checks,
    }
    if args.json:
        target = args.json if args.json.is_absolute() else ROOT / args.json
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in checks:
        print(f"{item['status']:>4} [{item['severity']:<14}] {item['name']}: {item['evidence']}")
    print(f"\nPROMOTION STATUS: {report['status']}")
    if args.strict and report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
