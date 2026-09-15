"""Run reproducible v4 validity and recovery audits.

This is intentionally separate from the route scorecard: it tests the scope of
the new claims directly on fresh calibration/audit streams.  It never tunes on
the audit observations.  The output is a release artifact consumed by
``check_v3_gate.py`` until the scorecard is renamed in a future release.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import source_manifest  # noqa: E402

from certflow.upgrades import (
    GroupConditionalCalibrator,
    SelectionConditionalCalibrator,
    SelectionLedger,
    SequentialRecoveryMonitor,
)


def _group_audit() -> dict:
    rng = np.random.default_rng(7401)
    train_groups = ["easy"] * 240 + ["hard"] * 240
    train_residuals = list(rng.normal(0.0, 0.35, 240)) + list(
        rng.normal(0.0, 1.0, 240)
    )
    cal = GroupConditionalCalibrator(min_support=50).fit(
        train_groups, train_residuals
    )
    certificates = {
        group: cal.certify(group, 0.0, alpha=0.2)
        for group in ("easy", "hard")
    }

    audit_groups = ["easy"] * 240 + ["hard"] * 240
    audit_residuals = list(rng.normal(0.0, 0.35, 240)) + list(
        rng.normal(0.0, 1.0, 240)
    )
    covered = [
        abs(residual) <= certificates[group].radius + 1e-12
        for group, residual in zip(audit_groups, audit_residuals)
    ]
    report = cal.coverage_report(audit_groups, covered, alpha=0.2)
    return {
        "certificates": {
            group: {
                "valid": cert.valid,
                "support": cert.support,
                "radius": cert.radius,
                "confidence": cert.confidence,
                "alpha_allocated": cert.alpha_allocated,
            }
            for group, cert in certificates.items()
        },
        "audit": {
            group: {
                "support": item.support,
                "coverage": item.coverage,
                "lower_confidence_bound": item.lower_confidence_bound,
                "target_coverage": item.target_coverage,
                "valid": item.valid,
            }
            for group, item in report.items()
        },
    }


def _recovery_audit() -> dict:
    monitor = SequentialRecoveryMonitor(
        alarm_threshold=100.0, recovery_samples=5, betting_epsilon=0.5
    )
    stable_states = [monitor.observe_pvalue(0.5) for _ in range(100)]
    stable_false_alarm = monitor.formal_alarm
    detection_delay = None
    for i in range(1, 51):
        monitor.observe_pvalue(1e-8)
        if monitor.formal_alarm:
            detection_delay = i
            break
    recovery_delay = None
    for i in range(1, 51):
        monitor.observe_pvalue(1.0)
        if monitor.certificate_allowed:
            recovery_delay = i
            break
    return {
        "stable_rounds": len(stable_states),
        "stable_false_alarm": stable_false_alarm,
        "detection_delay": detection_delay,
        "recovery_delay": recovery_delay,
        "formal_alarm_after_recovery": monitor.formal_alarm,
        "diagnostics": monitor.diagnostics(),
    }


def _post_selection_audit() -> dict:
    """Exercise the selected-object path using an independent audit stream."""
    rng = np.random.default_rng(7402)
    ledger = SelectionLedger()
    event = ledger.record(
        "route-b", ["route-a", "route-b", "route-c"], context=[12.0, 24.0]
    )
    audit_residuals = list(rng.normal(0.0, 0.5, 80))
    cert = SelectionConditionalCalibrator(min_audit=40).certify(
        event.selected, 10.0, audit_residuals, alpha=0.1, ledger=ledger
    )
    return {
        "valid": cert.valid,
        "audit_count": cert.audit_count,
        "radius": cert.radius,
        "selection_digest": cert.selection_digest,
        "digest_matches": cert.selection_digest == ledger.digest,
        "validity_scope": cert.validity_scope,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic CERT-FLOW validity audit."
    )
    parser.parse_args()
    group = _group_audit()
    recovery = _recovery_audit()
    post_selection = _post_selection_audit()
    group_pass = all(
        item["valid"] for item in group["certificates"].values()
    ) and all(item["valid"] for item in group["audit"].values())
    recovery_pass = (
        recovery["stable_false_alarm"] is False
        and recovery["detection_delay"] is not None
        and recovery["recovery_delay"] is not None
        and recovery["recovery_delay"] <= 10
        and recovery["formal_alarm_after_recovery"] is False
    )
    post_selection_pass = (
        post_selection["valid"]
        and post_selection["digest_matches"]
        and post_selection["audit_count"] >= 40
    )
    out = {
        "schema": "certflow.v4.validity-audit.1",
        "seed": 7401,
        "status": "PASS" if group_pass and recovery_pass else "OPEN",
        "group_conditional": group,
        "formal_recovery": recovery,
        "post_selection": post_selection,
        "manifest": source_manifest(ROOT),
        "gates": {
            "finite_family_group_certificates": group_pass,
            "formal_recovery_alarm_and_reentry": recovery_pass,
            "fresh_post_selection_audit": post_selection_pass,
        },
    }
    target = ROOT / "results" / "v4_validity" / "audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
