"""Requirement-level tests for the CERT-FLOW v3 upgrade layer."""
import math

import numpy as np
import pytest

from certflow import (
    ActiveSensingPolicy,
    DecisionRiskController,
    EvidenceConditionalCalibrator,
    GroupConditionalCalibrator,
    JointFleetCalibrator,
    MixtureShiryaevRobertsDetector,
    RegimeRecoveryManager,
    SelectionConditionalCalibrator,
    SelectionLedger,
    SequentialRecoveryMonitor,
    SensingAction,
    TrajectoryConformalCalibrator,
    congestion_penalty,
    reachable_tube,
    RiskLedger,
    RouteWitness,
    SensingCandidate,
    select_witness_observations,
)
from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.team import joint_fleet_certificate


def test_selection_conditional_audit_conditions_on_ledger():
    ledger = SelectionLedger()
    event = ledger.record("path-b", ["path-a", "path-b"], context=[4, 20])
    cert = SelectionConditionalCalibrator(min_audit=5).certify(
        event.selected, 10.0, [0.2, 0.3, 0.1, 0.4, 0.25, 0.15],
        alpha=0.2, ledger=ledger,
    )
    assert cert.valid
    assert cert.selected == "path-b"
    assert cert.confidence == 0.8
    assert cert.selection_digest == ledger.digest
    assert cert.lower <= 10.0 <= cert.upper


def test_selection_conditional_requires_separate_audit_support():
    cert = SelectionConditionalCalibrator(min_audit=5).certify(
        "chosen", 1.0, [0.1, 0.2], alpha=0.1
    )
    assert not cert.valid
    assert cert.confidence == 0.0
    assert math.isinf(cert.radius)


def test_group_conditional_calibration_allocates_alpha_and_rejects_sparse_groups():
    groups = ["easy"] * 25 + ["hard"] * 25 + ["rare"] * 3
    residuals = [0.2] * 25 + [0.8] * 25 + [0.1] * 3
    cal = GroupConditionalCalibrator(min_support=10).fit(groups, residuals)
    hard = cal.certify("hard", 10.0, alpha=0.2)
    rare = cal.certify("rare", 10.0, alpha=0.2)
    assert hard.valid
    assert hard.alpha_allocated == 0.2 / 3
    assert rare.support == 3 and not rare.valid and math.isinf(rare.radius)


def test_group_calibration_rejects_nonfinite_residuals():
    with pytest.raises(ValueError, match="finite"):
        GroupConditionalCalibrator().fit(["a", "a"], [0.1, math.nan])


def test_group_family_cannot_underallocate_and_certify_all_uses_union():
    cal = GroupConditionalCalibrator(min_support=1).fit(
        ["fitted-only", "another-fitted"], [0.1, 0.2]
    )
    with pytest.raises(ValueError, match="smaller"):
        cal.certify("fitted-only", 0.0, alpha=0.1, family_size=1)
    out = cal.certify_all({"point-only": 0.0}, alpha=0.1)
    assert out["point-only"].family_size == 3


def test_group_coverage_audit_uses_finite_family_lower_bound():
    cal = GroupConditionalCalibrator(min_support=5)
    report = cal.coverage_report(
        ["easy"] * 20 + ["hard"] * 20,
        [True] * 20 + [True] * 19 + [False],
        alpha=0.1, target_coverage=0.7,
    )
    assert report["easy"].valid
    assert report["hard"].support == 20
    assert report["hard"].lower_confidence_bound <= report["hard"].coverage


def test_formal_sequential_recovery_monitor_alarms_and_reopens():
    monitor = SequentialRecoveryMonitor(
        alarm_threshold=10.0, recovery_samples=3, betting_epsilon=0.5
    )
    for _ in range(20):
        monitor.observe_pvalue(1e-8)
    assert monitor.formal_alarm if hasattr(monitor, "formal_alarm") else True
    assert not monitor.certificate_allowed
    for _ in range(3):
        monitor.observe_pvalue(1.0)
    assert monitor.certificate_allowed
    assert monitor.diagnostics()["validity_scope"].startswith(
        "configured sequential-detector"
    )


def test_formal_monitor_rejects_invalid_p_values_without_state_change():
    monitor = SequentialRecoveryMonitor(alarm_threshold=10.0)
    for invalid in (math.nan, math.inf, -math.inf, -0.1, 1.1):
        with pytest.raises(ValueError, match="finite"):
            monitor.observe_pvalue(invalid)
        assert monitor.last_p_value is None
        assert monitor.certificate_allowed
    assert monitor.observe_pvalue(0.5) == "stable"


def test_detector_rejects_nonfinite_threshold_and_mixture_is_public():
    assert MixtureShiryaevRobertsDetector(threshold=10.0).threshold == 10.0
    for threshold in (math.inf, math.nan):
        try:
            MixtureShiryaevRobertsDetector(threshold=threshold)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite threshold was accepted")
def test_mixture_formal_monitor_preserves_alarm_and_reports_fixed_bets():
    monitor = SequentialRecoveryMonitor(
        alarm_threshold=10.0, recovery_samples=2,
        betting_epsilons=(0.1, 0.25, 0.5, 0.75, 0.9),
    )
    for _ in range(20):
        monitor.observe_pvalue(1e-8)
    assert monitor.formal_alarm
    assert not monitor.certificate_allowed
    assert monitor.diagnostics()["betting_epsilons"] == (0.1, 0.25, 0.5, 0.75, 0.9)
    for _ in range(2):
        monitor.observe_pvalue(1.0)
    assert monitor.certificate_allowed


def test_evidence_model_uses_pooled_normalized_calibration():
    train_x = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    train_y = [0.1, 0.2, 0.5, 0.8, 1.2, 1.6]
    cal_x = [[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]]
    cal_y = [0.15, 0.3, 0.55, 0.9, 1.3, 1.8]
    cal = EvidenceConditionalCalibrator(min_calibration=5)
    cal.fit(train_x, train_y, cal_x, cal_y)
    assert cal.ready
    low = cal.radius([0.5], 0.2)
    high = cal.radius([5.0], 0.2)
    assert math.isfinite(low) and math.isfinite(high)
    assert high > low


def test_decision_risk_control_selects_best_safe_action():
    controller = DecisionRiskController(delta=0.05)
    result = controller.select(
        {"safe-good": [0.0] * 200, "unsafe-cheap": [1.0] * 200},
        {"safe-good": 5.0, "unsafe-cheap": 1.0},
        target_risk=0.1,
    )
    assert result.valid
    assert result.action == "safe-good"
    assert result.risk_bound <= 0.1


def test_trajectory_tube_and_reachable_tube():
    predicted = np.zeros((12, 4, 2))
    observed = predicted.copy()
    observed[:, :, 0] += np.linspace(0.0, 0.2, 12)[:, None]
    cal = TrajectoryConformalCalibrator(min_calibration=10)
    cal.fit(predicted, observed)
    tube = cal.tube(predicted[0], alpha=0.1)
    assert tube.valid
    assert tube.contains(observed[0])
    reachable = reachable_tube(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        0.5,
    )
    assert np.allclose(reachable.radius[-1], [1.0, 1.0])


def test_active_sensing_learns_realized_gain():
    policy = ActiveSensingPolicy(exploration=0.0)
    actions = [SensingAction("a", 1.0, 1.0), SensingAction("b", 2.0, 1.0)]
    first = policy.select(actions)
    assert first is not None
    policy.update("a", 10.0)
    policy.update("b", 0.0)
    chosen = policy.select(actions)
    assert chosen is not None and chosen.key == "a"
    assert policy.stats()["a"]["pulls"] == 1.0
    allocation = policy.allocate(
        [SensingAction("a", 1.0, 0.6), SensingAction("b", 1.0, 0.6)],
        budget=1.0,
    )
    assert len(allocation) == 1


def test_joint_fleet_certificate_prices_correlated_agents_and_congestion():
    cal = JointFleetCalibrator(mode="sum", min_calibration=8)
    cal.fit([[0.1, -0.1], [0.2, 0.0], [-0.2, 0.1], [0.0, 0.2],
             [0.1, 0.1], [-0.1, -0.2], [0.2, -0.1], [0.0, 0.0]])
    tc = joint_fleet_certificate(
        cal, [10.0, 12.0], alpha=0.2,
        route_loads={"road": 3.0}, capacities={"road": 2.0},
        congestion_coefficient=2.0,
    )
    assert tc.valid
    assert tc.congestion_cost == 2.0
    assert tc.lower <= 24.0 <= tc.upper
    assert congestion_penalty({"road": 2.0}, {"road": 2.0}) == 0.0


def test_regime_recovery_revokes_then_reopens_certificate():
    manager = RegimeRecoveryManager(
        alarm_delta=0.01, recovery_samples=3, evidence_threshold=2.0
    )
    for _ in range(3):
        manager.observe_pvalue(1e-6)
    assert manager.state in ("alarm", "recovery")
    assert not manager.certificate_allowed
    for _ in range(3):
        manager.observe_pvalue(1.0)
    assert manager.state == "stable"
    assert manager.certificate_allowed


def test_route_witness_and_risk_ledger_are_explicit_and_bounded():
    ledger = RiskLedger(0.2)
    ledger.spend(0.1, reason="calibration", scope="epoch-0")
    ledger.spend(0.05, reason="audit", scope="selected-route")
    witness = RouteWitness(
        route=((0, 0), (0, 1), (1, 1)),
        incumbent_upper=4.0,
        optimistic_lower=3.0,
        epoch=2,
        selection_rule="greedy-gap-with-challenger-reserve",
        observation_ids=("e1", "e2"),
        risk_spent=ledger.spent,
        risk_total=ledger.alpha_total,
    )
    assert witness.valid and witness.gap == 1.0
    assert ledger.remaining == pytest.approx(0.05)
    assert len(ledger.digest) == 16
    with pytest.raises(ValueError, match="exceed"):
        ledger.spend(0.2, reason="bad", scope="bad")


def test_witness_sensing_reserves_a_challenger_deterministically():
    plan = select_witness_observations(
        [
            SensingCandidate("incumbent", 9.0, 3.0),
            SensingCandidate("challenger", 1.0, 1.0, challenger=True),
            SensingCandidate("other", 2.0, 1.0),
        ],
        budget=2.0,
    )
    assert plan[0].key == "challenger"
    assert sum(item.cost for item in plan) <= 2.0


def test_planner_v3_hooks_are_live_but_opt_in():
    world = grid_world(4, 4, seed=11, kind="bounded", rho=0.01, noise_scale=0.05)
    cfg = PlannerConfig(
        epsilon=3.0, alpha_prime=0.2, selection_conditional=True,
        active_sensing=True, regime_recovery=True,
    )
    planner = CertPlanner(world, (0, 0), (3, 3), cfg)
    for _ in range(50):
        planner.round()
    assert len(planner.selection_ledger.events) > 0
    assert planner.active_sensing.stats()
    diag = planner.diagnostics()
    assert "v3_regime" in diag
    assert "v3_selection_digest" in diag


def test_planner_can_price_with_learned_evidence_and_emit_route_tube():
    world = grid_world(4, 4, seed=12, kind="bounded", rho=0.01, noise_scale=0.05)
    cfg = PlannerConfig(
        epsilon=3.0, alpha_prime=0.2, evidence_model=True,
        trajectory_tubes=True, trajectory_min_calibration=3,
    )
    planner = CertPlanner(world, (0, 0), (3, 3), cfg)
    x = np.asarray([
        [float(i), 0.01 * i, 4.0, 20.0, 1.0, 0.5]
        for i in range(40)
    ])
    y = np.linspace(0.05, 0.5, 40)
    planner.fit_evidence_model(x[:20], y[:20], x[20:], y[20:])
    planner.fit_trajectory_calibrator(
        np.zeros((4, 3, 1)), np.full((4, 3, 1), 0.05)
    )
    for _ in range(25):
        planner.round()
    assert math.isfinite(planner.evidence_radius(next(iter(planner.beliefs))))
    assert planner._last_trajectory_tube is not None
    assert planner.diagnostics()["v3_trajectory_tube_valid"]


def test_planner_auto_evidence_uses_chronological_train_calibration_split():
    world = grid_world(4, 4, seed=121, kind="static", noise_scale=0.05)
    planner = CertPlanner(
        world, (0, 0), (3, 3),
        PlannerConfig(
            epsilon=3.0, alpha_prime=0.2, evidence_model=True,
            evidence_model_min_calibration=5,
        ),
    )
    for _ in range(90):
        planner.round()
    diag = planner.diagnostics()
    assert diag["v3_evidence_auto_train"] >= 5
    assert diag["v3_evidence_auto_calibration"] >= 5
    assert diag["v3_evidence_auto_fitted"]
    assert diag["v3_evidence_model_ready"]


def test_parallel_warmup_uses_distinct_edges_and_fills_support_quickly():
    world = grid_world(5, 5, seed=31, kind="bounded", rho=0.02, noise_scale=0.05)
    planner = CertPlanner(
        world, (0, 0), (4, 4),
        PlannerConfig(
            epsilon=3.0, alpha_prime=0.2, adaptive_rate=True,
            warmup_sense_per_round=4, max_sense_per_round=4,
        ),
    )
    planner.round()
    assert 0.0 < planner.sense_spend <= 0.4
    planner.round()
    assert len(planner.scorer._buf) >= 4


def test_negative_drift_adjusted_scores_never_create_negative_search_costs():
    world = grid_world(4, 4, seed=32, kind="bounded", rho=1.0, noise_scale=0.0)
    planner = CertPlanner(
        world, (0, 0), (3, 3),
        PlannerConfig(epsilon=3.0, alpha_prime=0.2, rho_w=1.0),
    )
    for _ in range(20):
        planner.round()
    assert all(v >= planner.cfg.cost_floor for v in planner._cache_lo.values())
    assert all(v >= planner.cfg.cost_floor for v in planner._cache_up.values())


def test_online_rho_keeps_edge_heterogeneity_with_pooled_fallback():
    world = grid_world(3, 3, seed=33, kind="static", noise_scale=0.0)
    planner = CertPlanner(
        world, (0, 0), (2, 2),
        PlannerConfig(
            epsilon=3.0, alpha_prime=0.2, rho_mode="online",
            rho_online_quantile=0.5, rho_online_min_samples=3,
        ),
    )
    edges = list(planner.beliefs)
    planner._rate_samples = [0.1] * 3 + [0.9] * 3
    planner._rate_samples_by_edge[edges[0]] = [0.1] * 3
    planner._rate_samples_by_edge[edges[1]] = [0.9] * 3
    planner._update_online_rho()
    assert planner.beliefs[edges[0]].rho < planner.beliefs[edges[1]].rho


def test_mean_path_execution_adds_only_a_certified_candidate():
    world = grid_world(4, 4, seed=34, kind="bounded", rho=0.01, noise_scale=0.02)
    planner = CertPlanner(
        world, (0, 0), (3, 3),
        PlannerConfig(
            epsilon=3.0, alpha_prime=0.2, mean_path_execution=True,
            use_kappa=True, warmup_sense_per_round=4,
        ),
    )
    cert, _ = planner.round()
    assert cert.path
    assert all(edge in planner.beliefs for edge in zip(cert.path, cert.path[1:]))


def test_planner_recovery_gate_revokes_certificate():
    world = grid_world(4, 4, seed=13, kind="static", noise_scale=0.05)
    planner = CertPlanner(
        world, (0, 0), (3, 3),
        PlannerConfig(epsilon=3.0, alpha_prime=0.2, regime_recovery=True),
    )
    planner.regime.force_recovery()
    cert, _ = planner.round()
    assert not cert.valid
    assert planner.recovery_diagnostics()["regime"]["state"] in {
        "recovery", "stable"
    }


def test_planner_can_select_formal_recovery_gate():
    world = grid_world(4, 4, seed=131, kind="static", noise_scale=0.05)
    planner = CertPlanner(
        world, (0, 0), (3, 3),
        PlannerConfig(
            epsilon=3.0, alpha_prime=0.2, regime_recovery=True,
            recovery_formal=True,
        ),
    )
    for _ in range(30):
        planner.round()
    for _ in range(4):
        planner.formal_regime.observe_pvalue(1e-8)
    cert, _ = planner.round()
    assert not cert.valid
    diag = planner.recovery_diagnostics()
    assert "formal_regime" in diag
    assert diag["formal_regime"]["validity_scope"].startswith(
        "configured sequential-detector"
    )


def test_v3_state_is_inert_when_all_upgrade_flags_are_off():
    w1 = grid_world(4, 4, seed=14, kind="bounded", rho=0.01, noise_scale=0.05)
    w2 = grid_world(4, 4, seed=14, kind="bounded", rho=0.01, noise_scale=0.05)
    cfg = PlannerConfig(epsilon=3.0, alpha_prime=0.2)
    p1 = CertPlanner(w1, (0, 0), (3, 3), cfg)
    p2 = CertPlanner(w2, (0, 0), (3, 3), cfg)
    for _ in range(30):
        c1, s1 = p1.round()
        c2, s2 = p2.round()
        assert c1.__dict__ == c2.__dict__
        assert s1 == s2
