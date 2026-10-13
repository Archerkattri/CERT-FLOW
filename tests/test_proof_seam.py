"""Adversarial checks for the frozen-selection proof boundary."""

import math

import pytest

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.upgrades import SelectionConditionalCalibrator


@pytest.mark.parametrize("point", [math.nan, math.inf, -math.inf])
def test_selection_point_must_be_finite(point):
    with pytest.raises(ValueError, match="point must be finite"):
        SelectionConditionalCalibrator(2).certify("route", point, [0.1, 0.2], 0.5)


@pytest.mark.parametrize("drift", [math.nan, math.inf, -math.inf, -0.01])
def test_selection_drift_must_be_finite_and_nonnegative(drift):
    with pytest.raises(ValueError, match="drift_margin must be finite and >= 0"):
        SelectionConditionalCalibrator(2).certify(
            "route", 1.0, [0.1, 0.2], 0.5, drift_margin=drift
        )


@pytest.mark.parametrize("drift", [0.0, 0.3])
def test_finite_selection_point_and_drift_preserve_the_full_audit_sample(drift):
    certificate = SelectionConditionalCalibrator(2).certify(
        "route", 1.0, [0.1, -0.2], 0.5, drift_margin=drift
    )
    assert certificate.valid
    assert certificate.audit_count == 2
    assert certificate.lower == pytest.approx(1.0 - 0.2 - drift)
    assert certificate.upper == pytest.approx(1.0 + 0.2 + drift)


@pytest.mark.parametrize("residual", [math.nan, math.inf, -math.inf])
def test_audit_sample_cannot_silently_drop_nonfinite_residuals(residual):
    with pytest.raises(ValueError, match="finite"):
        SelectionConditionalCalibrator(2).certify(
            "route", 1.0, [0.1, 0.2, residual], 0.5
        )


def test_internal_audit_scores_cannot_certify_another_route():
    planner = CertPlanner(
        grid_world(3, 3, seed=22, kind="static", noise_scale=0.0),
        (0, 0), (2, 2),
        PlannerConfig(epsilon=3.0, selection_conditional=True, selection_audit_min=2),
    )
    for _ in range(30):
        planner.round()
    event = planner._last_selection_event
    assert event is not None
    for index in range(2):
        planner.record_selection_audit(
            0.1, decision_id=event.decision_id, observation_id=f"independent-{index}"
        )
    assert planner.selection_certificate(alpha=0.5).valid
    other = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    if tuple(other) == event.selected:
        other = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    with pytest.raises(ValueError, match="audit path"):
        planner.selection_certificate(other, alpha=0.5)
    # A caller-supplied, separately collected stream remains a supported seam,
    # with no false claim that the planner selected this route.
    manual = planner.selection_certificate(other, audit_residuals=[0.1, 0.1], alpha=0.5)
    assert manual.valid
    assert manual.selection_digest == "no-ledger"
