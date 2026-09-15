"""CERT-FLOW v3 upgrade primitives.

The original planner deliberately keeps its default certificate small and
auditable.  This module contains the larger decision layer that can be opted
into without changing that default stream:

* selection-conditional audit certificates;
* evidence-quality calibration without calibration-set shredding;
* direct decision/risk control;
* trajectory-level conformal tubes and linear reachable tubes;
* budgeted active sensing;
* joint fleet calibration;
* regime detection and recovery state management.

The implementations are intentionally dependency-light (NumPy only) and make
their validity scope explicit.  In particular, an evidence model is not
labelled "conditional" unless the caller supplies a separate calibration
sample and a stated conditioning protocol.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence

import numpy as np


def _finite_sample_quantile(values: Sequence[float], alpha: float) -> float:
    """Conformal upper order statistic, including the test-point convention."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    x = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if x.size == 0:
        return math.inf
    x.sort()
    # k is the zero-based index of ceil((n+1)(1-alpha)); k==n means +inf.
    k = int(math.ceil((x.size + 1) * (1.0 - alpha)) - 1)
    return math.inf if k >= x.size else float(x[max(k, 0)])


@dataclass(frozen=True)
class SelectionEvent:
    """Immutable record of one adaptive selection decision."""

    decision_id: int
    selected: Hashable
    candidates: tuple[Hashable, ...]
    context: tuple[float, ...] = ()


class SelectionLedger:
    """Audit trail for adaptive candidate selection.

    The digest is included in certificates so a downstream evaluator can
    verify exactly which selection history a post-selection audit conditioned
    on.  It is not a cryptographic security boundary.
    """

    def __init__(self) -> None:
        self._events: list[SelectionEvent] = []

    def record(
        self,
        selected: Hashable,
        candidates: Sequence[Hashable],
        context: Sequence[float] = (),
    ) -> SelectionEvent:
        event = SelectionEvent(
            decision_id=len(self._events),
            selected=selected,
            candidates=tuple(candidates),
            context=tuple(float(x) for x in context),
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[SelectionEvent, ...]:
        return tuple(self._events)

    @property
    def digest(self) -> str:
        payload = repr(self.events).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class SelectionConditionalCertificate:
    """Certificate for a selected object using a separate post-selection audit."""

    selected: Hashable
    lower: float
    upper: float
    confidence: float
    audit_count: int
    selection_digest: str
    validity_scope: str = "conditional on recorded selection history and audit exchangeability"

    @property
    def valid(self) -> bool:
        return self.confidence > 0.0 and math.isfinite(self.lower) and math.isfinite(self.upper)

    @property
    def radius(self) -> float:
        return (self.upper - self.lower) / 2.0


class SelectionConditionalCalibrator:
    """Build a post-selection interval from an independent audit stream.

    ``audit_residuals`` must be collected after selection, or from a split
    stream that was not used by the selection rule.  This is a conservative,
    operational route to selection-conditional validity; it does not claim
    that reusing the original adaptive calibration buffer is valid.
    """

    def __init__(self, min_audit: int = 10) -> None:
        if min_audit < 1:
            raise ValueError("min_audit must be >= 1")
        self.min_audit = int(min_audit)

    def certify(
        self,
        selected: Hashable,
        point: float,
        audit_residuals: Sequence[float],
        alpha: float,
        ledger: SelectionLedger | None = None,
        drift_margin: float = 0.0,
    ) -> SelectionConditionalCertificate:
        if drift_margin < 0.0:
            raise ValueError("drift_margin must be >= 0")
        finite = [abs(float(x)) for x in audit_residuals if math.isfinite(float(x))]
        q = _finite_sample_quantile(finite, alpha)
        conf = max(0.0, 1.0 - alpha) if len(finite) >= self.min_audit and math.isfinite(q) else 0.0
        radius = q + drift_margin if math.isfinite(q) else math.inf
        digest = ledger.digest if ledger is not None else "no-ledger"
        return SelectionConditionalCertificate(
            selected=selected,
            lower=float(point) - radius,
            upper=float(point) + radius,
            confidence=conf,
            audit_count=len(finite),
            selection_digest=digest,
        )


@dataclass(frozen=True)
class GroupCertificate:
    """A certificate for one member of a declared, finite group family.

    The family size is part of the certificate because the alpha allocation is
    Bonferroni-over-groups.  This is deliberately a finite-family guarantee;
    it does not claim universal conditional coverage over arbitrary covariates.
    """

    group: Hashable
    lower: float
    upper: float
    confidence: float
    support: int
    alpha_allocated: float
    family_size: int
    validity_scope: str = (
        "conditional on the declared finite group family and independent "
        "group calibration exchangeability"
    )

    @property
    def valid(self) -> bool:
        return (
            self.support > 0
            and self.confidence > 0.0
            and math.isfinite(self.lower)
            and math.isfinite(self.upper)
        )

    @property
    def radius(self) -> float:
        return (self.upper - self.lower) / 2.0


@dataclass(frozen=True)
class GroupCoverageReport:
    """Held-out group coverage audit with a fixed-time concentration bound.

    The bound is valid for the declared finite family at this audit call.  It
    is not automatically valid after repeatedly inspecting or replacing the
    audit set; callers that need optional-stopping validity must use a
    confidence sequence or an explicit alpha-spending protocol.
    """

    group: Hashable
    support: int
    coverage: float
    lower_confidence_bound: float
    target_coverage: float
    confidence: float
    valid: bool


class GroupConditionalCalibrator:
    """Finite-family conditional calibration with explicit sparse-group gates.

    A single global interval cannot honestly promise coverage within a hard
    subgroup.  This calibrator accepts a *declared finite family* and fits one
    split-conformal residual distribution per group.  Alpha is allocated as
    ``alpha / G`` across the family, while groups below ``min_support`` return
    an invalid infinite-radius certificate instead of silently borrowing a
    marginal guarantee.  The held-out audit method reports a one-sided
    Hoeffding lower bound, also Bonferroni allocated over the family.
    """

    def __init__(self, min_support: int = 20) -> None:
        if min_support < 1:
            raise ValueError("min_support must be >= 1")
        self.min_support = int(min_support)
        self._scores: dict[Hashable, list[float]] = {}

    def fit(
        self,
        groups: Sequence[Hashable],
        residuals: Sequence[float],
    ) -> "GroupConditionalCalibrator":
        if len(groups) != len(residuals):
            raise ValueError("groups/residuals length mismatch")
        scores: dict[Hashable, list[float]] = {}
        for group, residual in zip(groups, residuals):
            value = float(residual)
            if not math.isfinite(value):
                raise ValueError(
                    "residuals must be finite; refusing to silently shrink "
                    "the declared calibration sample"
                )
            scores.setdefault(group, []).append(abs(value))
        self._scores = scores
        return self

    @property
    def groups(self) -> tuple[Hashable, ...]:
        return tuple(self._scores)

    def _family_size(self, family_size: int | None) -> int:
        size = len(self._scores) if family_size is None else int(family_size)
        if size < 1:
            raise ValueError("family_size must be >= 1")
        if size < len(self._scores):
            raise ValueError(
                "family_size cannot be smaller than the fitted group family"
            )
        return size

    def certify(
        self,
        group: Hashable,
        point: float,
        alpha: float,
        drift_margin: float = 0.0,
        family_size: int | None = None,
    ) -> GroupCertificate:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if drift_margin < 0.0:
            raise ValueError("drift_margin must be >= 0")
        size = self._family_size(family_size)
        alpha_group = alpha / size
        scores = self._scores.get(group, [])
        q = _finite_sample_quantile(scores, alpha_group)
        supported = len(scores) >= self.min_support and math.isfinite(q)
        radius = q + drift_margin if supported else math.inf
        return GroupCertificate(
            group=group,
            lower=float(point) - radius,
            upper=float(point) + radius,
            confidence=(1.0 - alpha_group) if supported else 0.0,
            support=len(scores),
            alpha_allocated=alpha_group,
            family_size=size,
        )

    def certify_all(
        self,
        points: Mapping[Hashable, float],
        alpha: float,
        drift_margin: float = 0.0,
    ) -> dict[Hashable, GroupCertificate]:
        declared_family = set(points) | set(self._scores)
        size = max(len(declared_family), 1)
        return {
            group: self.certify(
                group, point, alpha, drift_margin=drift_margin,
                family_size=size,
            )
            for group, point in points.items()
        }

    def coverage_report(
        self,
        groups: Sequence[Hashable],
        covered: Sequence[bool | int],
        alpha: float,
        target_coverage: float | None = None,
    ) -> dict[Hashable, GroupCoverageReport]:
        """Audit held-out indicators without treating the audit as calibration.

        ``covered`` must be computed on fresh outcomes.  The lower bound is a
        one-sided fixed-time Hoeffding bound at confidence ``1 - alpha/G`` for
        every declared group, so ``valid`` is a transparent finite-family
        gate. Repeated calls or optional inspection require a separate
        confidence-sequence or alpha-spending design.
        """
        if len(groups) != len(covered):
            raise ValueError("groups/covered length mismatch")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        family = tuple(dict.fromkeys(groups))
        if not family:
            return {}
        target = 1.0 - alpha if target_coverage is None else float(target_coverage)
        if not 0.0 <= target <= 1.0:
            raise ValueError("target_coverage must be in [0, 1]")
        delta_group = alpha / len(family)
        values_by_group = {group: [] for group in family}
        for group, value in zip(groups, covered):
            values_by_group[group].append(bool(value))
        out: dict[Hashable, GroupCoverageReport] = {}
        for group in family:
            vals = values_by_group[group]
            n = len(vals)
            phat = sum(vals) / n if n else 0.0
            margin = math.sqrt(math.log(1.0 / delta_group) / (2.0 * n)) if n else math.inf
            lcb = max(0.0, phat - margin) if n else 0.0
            out[group] = GroupCoverageReport(
                group=group,
                support=n,
                coverage=phat,
                lower_confidence_bound=lcb,
                target_coverage=target,
                confidence=1.0 - delta_group if n else 0.0,
                valid=n >= self.min_support and lcb >= target,
            )
        return out


class SequentialRecoveryMonitor:
    """Formal alarm gate using a conformal e-value change detector.

    The e-value process has a stated false-alarm threshold under the supplied
    conformal null.  Recovery remains an operational policy: after an alarm,
    the certificate stays revoked until ``recovery_samples`` fresh, non-small
    p-values arrive.  This separates the formal alarm claim from the practical
    re-entry rule and avoids presenting a heuristic CUSUM as a theorem.
    """

    def __init__(
        self,
        alarm_threshold: float = 100.0,
        recovery_samples: int = 10,
        betting_epsilon: float = 0.5,
        betting_epsilons: Sequence[float] | None = None,
        safe_p_value: float = 0.1,
    ) -> None:
        if not math.isfinite(float(alarm_threshold)) or alarm_threshold <= 1.0:
            raise ValueError("alarm_threshold must be finite and > 1")
        if recovery_samples < 1:
            raise ValueError("invalid recovery monitor parameters")
        if not 0.0 < betting_epsilon < 1.0:
            raise ValueError("betting_epsilon must be in (0, 1)")
        if not 0.0 < safe_p_value <= 1.0:
            raise ValueError("safe_p_value must be in (0, 1]")
        self.alarm_threshold = float(alarm_threshold)
        self.recovery_samples = int(recovery_samples)
        self.safe_p_value = float(safe_p_value)
        self._detector = None
        self._betting_epsilon = float(betting_epsilon)
        self._betting_epsilons = (
            tuple(float(e) for e in betting_epsilons)
            if betting_epsilons is not None else None
        )
        if self._betting_epsilons is not None and (
            not self._betting_epsilons
            or any(not math.isfinite(e) or not 0.0 < e < 1.0
                   for e in self._betting_epsilons)
        ):
            raise ValueError("betting_epsilons must be a non-empty sequence in (0, 1)")
        self.state = RecoveryState.STABLE
        self.safe_samples = 0
        self.alarm_count = 0
        self.last_p_value: float | None = None

    def _ensure_detector(self):
        if self._detector is None:
            if self._betting_epsilons is None:
                from certflow.conformal import ShiryaevRobertsDetector
                self._detector = ShiryaevRobertsDetector(
                    threshold=self.alarm_threshold,
                    epsilon=self._betting_epsilon,
                )
            else:
                from certflow.conformal import MixtureShiryaevRobertsDetector
                self._detector = MixtureShiryaevRobertsDetector(
                    threshold=self.alarm_threshold,
                    epsilons=self._betting_epsilons,
                )
        return self._detector

    @property
    def certificate_allowed(self) -> bool:
        return self.state == RecoveryState.STABLE

    @property
    def formal_alarm(self) -> bool:
        """Whether the configured sequential detector crossed its threshold."""
        return bool(self._detector is not None and self._detector.alarm())

    def observe_pvalue(self, p_value: float) -> str:
        p = float(p_value)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("p_value must be finite and in [0, 1]")
        # A computed zero is numerically representable but is not a literal
        # probability under the detector's positive-p-value interface. Keep
        # the historical lower clamp only for this valid boundary case.
        p = max(p, 1e-12)
        self.last_p_value = p
        detector = self._ensure_detector()
        detector.update(p)
        if self.state == RecoveryState.STABLE and detector.alarm():
            self.state = RecoveryState.ALARM
            self.alarm_count += 1
            self.safe_samples = 0
        elif self.state in (RecoveryState.ALARM, RecoveryState.RECOVERY):
            self.state = RecoveryState.RECOVERY
            self.safe_samples = self.safe_samples + 1 if p >= self.safe_p_value else 0
            if self.safe_samples >= self.recovery_samples:
                self.state = RecoveryState.STABLE
                self.safe_samples = 0
                # Start a fresh formal testing epoch.  The old alarm remains
                # historical evidence, but must not immediately retrigger the
                # newly recovered certificate on the next observation.
                self._detector = None
        return self.state

    def diagnostics(self) -> dict[str, Any]:
        detector = self._ensure_detector()
        return {
            "state": self.state,
            "certificate_allowed": self.certificate_allowed,
            "formal_alarm": detector.alarm(),
            "formal_statistic": detector.R,
            "formal_peak": detector.peak,
            "alarm_round": detector.alarm_round,
            "alarm_threshold": self.alarm_threshold,
            "betting_epsilon": self._betting_epsilon,
            "betting_epsilons": self._betting_epsilons,
            "safe_samples": self.safe_samples,
            "alarm_count": self.alarm_count,
            "last_p_value": self.last_p_value,
            "validity_scope": (
                "configured sequential-detector threshold under the supplied "
                "conformal p-value null; recovery is an operational gate; "
                "no automatic Ville claim"
            ),
        }


def evidence_features(
    age: float,
    drift_exposure: float,
    path_length: float,
    effective_sample_size: float,
    overlap: float = 1.0,
    sensing_density: float = 0.0,
) -> np.ndarray:
    """Canonical evidence vector used by the optional learned calibrator."""
    return np.asarray(
        [
            max(float(age), 0.0),
            max(float(drift_exposure), 0.0),
            max(float(path_length), 0.0),
            max(float(effective_sample_size), 0.0),
            min(max(float(overlap), 0.0), 1.0),
            max(float(sensing_density), 0.0),
        ],
        dtype=float,
    )


class EvidenceScoreModel:
    """Ridge model for a positive, case-dependent residual scale.

    The model is fitted on a proper training split.  Its output is only a
    scale; conformal calibration still happens on a separate residual split.
    """

    def __init__(self, ridge: float = 1e-3) -> None:
        if ridge < 0.0:
            raise ValueError("ridge must be >= 0")
        self.ridge = float(ridge)
        self.coef_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, features: Sequence[Sequence[float]], residuals: Sequence[float]) -> "EvidenceScoreModel":
        x = np.asarray(features, dtype=float)
        y = np.log1p(np.abs(np.asarray(residuals, dtype=float)))
        if x.ndim != 2 or x.shape[0] != y.size or x.shape[0] < 2:
            raise ValueError("features must be (n, d) with at least two rows")
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        z = (x - self.mean_) / self.std_
        design = np.column_stack([np.ones(x.shape[0]), z])
        reg = self.ridge * np.eye(design.shape[1])
        reg[0, 0] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + reg, design.T @ y)
        return self

    @property
    def ready(self) -> bool:
        return self.coef_ is not None and self.mean_ is not None and self.std_ is not None

    def scale(self, features: Sequence[float]) -> float:
        if not self.ready:
            raise RuntimeError("EvidenceScoreModel must be fitted first")
        z = (np.asarray(features, dtype=float) - self.mean_) / self.std_  # type: ignore[operator]
        pred = float(np.r_[1.0, z] @ self.coef_)  # type: ignore[operator]
        return max(math.expm1(min(pred, 50.0)), 1e-6)

    def scale_many(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        """Vectorized counterpart of :meth:`scale` for graph refreshes."""
        if not self.ready:
            raise RuntimeError("EvidenceScoreModel must be fitted first")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        z = (x - self.mean_) / self.std_  # type: ignore[operator]
        pred = np.column_stack([np.ones(x.shape[0]), z]) @ self.coef_  # type: ignore[operator]
        return np.maximum(np.expm1(np.minimum(pred, 50.0)), 1e-6)


class EvidenceConditionalCalibrator:
    """Pooled conformal calibration on model-normalized residuals."""

    def __init__(self, model: EvidenceScoreModel | None = None, min_calibration: int = 20) -> None:
        if min_calibration < 1:
            raise ValueError("min_calibration must be >= 1")
        self.model = model or EvidenceScoreModel()
        self.min_calibration = int(min_calibration)
        self._normalized: list[float] = []

    def fit(
        self,
        train_features: Sequence[Sequence[float]],
        train_residuals: Sequence[float],
        calibration_features: Sequence[Sequence[float]],
        calibration_residuals: Sequence[float],
    ) -> "EvidenceConditionalCalibrator":
        self.model.fit(train_features, train_residuals)
        if len(calibration_features) != len(calibration_residuals):
            raise ValueError("calibration features/residuals length mismatch")
        self._normalized = [
            abs(float(r)) / self.model.scale(f)
            for f, r in zip(calibration_features, calibration_residuals)
            if math.isfinite(float(r))
        ]
        return self

    @property
    def ready(self) -> bool:
        return self.model.ready and len(self._normalized) >= self.min_calibration

    def radius(self, features: Sequence[float], alpha: float) -> float:
        if not self.ready:
            return math.inf
        return self.model.scale(features) * _finite_sample_quantile(self._normalized, alpha)

    def radius_many(self, features: Sequence[Sequence[float]], alpha: float) -> np.ndarray:
        """Vectorized normalized-conformal radius for all graph edges."""
        if not self.ready:
            return np.full(len(features), math.inf, dtype=float)
        q = _finite_sample_quantile(self._normalized, alpha)
        return self.model.scale_many(features) * q

    def normalized_score(self, features: Sequence[float], residual: float) -> float:
        return abs(float(residual)) / self.model.scale(features)


@dataclass
class DecisionRiskCertificate:
    action: Hashable | None
    objective: float
    empirical_risk: float
    risk_bound: float
    target_risk: float
    confidence: float
    sample_count: int
    valid: bool
    validity_scope: str = "finite-sample Hoeffding upper bound on the supplied loss stream"


class DecisionRiskController:
    """Select the best action whose calibrated risk is below a target."""

    def __init__(self, delta: float = 0.05) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.delta = float(delta)

    def select(
        self,
        actions: Mapping[Hashable, Sequence[float]],
        objectives: Mapping[Hashable, float],
        target_risk: float,
        alpha: float | None = None,
    ) -> DecisionRiskCertificate:
        if not 0.0 <= target_risk <= 1.0:
            raise ValueError("target_risk must be in [0, 1]")
        delta = self.delta if alpha is None else alpha
        if not 0.0 < delta < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        candidates: list[tuple[float, Hashable, float, float, int]] = []
        for action, raw in actions.items():
            losses = np.asarray([float(x) for x in raw], dtype=float)
            losses = losses[np.isfinite(losses)]
            if losses.size == 0:
                continue
            losses = np.clip(losses, 0.0, 1.0)
            empirical = float(losses.mean())
            bonus = math.sqrt(math.log(1.0 / delta) / (2.0 * losses.size))
            bound = min(1.0, empirical + bonus)
            if bound <= target_risk:
                candidates.append((float(objectives[action]), action, empirical, bound, int(losses.size)))
        if not candidates:
            return DecisionRiskCertificate(None, math.inf, 1.0, 1.0, target_risk, 0.0, 0, False)
        objective, action, empirical, bound, n = min(candidates, key=lambda x: x[0])
        return DecisionRiskCertificate(
            action=action,
            objective=objective,
            empirical_risk=empirical,
            risk_bound=bound,
            target_risk=target_risk,
            confidence=1.0 - delta,
            sample_count=n,
            valid=True,
        )


@dataclass
class TrajectoryTube:
    """Uniform conformal tube around a sequence of predicted states/costs."""

    center: np.ndarray
    radius: np.ndarray
    confidence: float
    coverage_level: float
    validity_scope: str = "trajectory-level conformity score"

    @property
    def lower(self) -> np.ndarray:
        return self.center - self.radius

    @property
    def upper(self) -> np.ndarray:
        return self.center + self.radius

    @property
    def valid(self) -> bool:
        return self.confidence > 0.0

    def contains(self, observed: Sequence[Sequence[float]] | Sequence[float]) -> bool:
        y = np.asarray(observed, dtype=float)
        return bool(np.all(y >= self.lower - 1e-12) and np.all(y <= self.upper + 1e-12))

    def inflate(self, extra: float | Sequence[float]) -> "TrajectoryTube":
        return TrajectoryTube(self.center.copy(), self.radius + np.asarray(extra), self.confidence, self.coverage_level, self.validity_scope)


class TrajectoryConformalCalibrator:
    """Calibrate one trajectory-level max norm instead of edgewise intervals."""

    def __init__(self, min_calibration: int = 10) -> None:
        if min_calibration < 1:
            raise ValueError("min_calibration must be >= 1")
        self.min_calibration = int(min_calibration)
        self._scores: list[float] = []
        self._shape: tuple[int, ...] | None = None

    def fit(self, predicted: Sequence[Sequence[Sequence[float]]], observed: Sequence[Sequence[Sequence[float]]]) -> "TrajectoryConformalCalibrator":
        p = np.asarray(predicted, dtype=float)
        y = np.asarray(observed, dtype=float)
        if p.shape != y.shape or p.ndim < 2:
            raise ValueError("predicted and observed must have equal shape (n, horizon, ...)" )
        self._shape = p.shape[1:]
        self._scores = [float(np.max(np.abs(a - b))) for a, b in zip(p, y)]
        return self

    @property
    def ready(self) -> bool:
        return len(self._scores) >= self.min_calibration

    def tube(self, center: Sequence[Sequence[float]] | Sequence[float], alpha: float) -> TrajectoryTube:
        c = np.asarray(center, dtype=float)
        q = _finite_sample_quantile(self._scores, alpha) if self.ready else math.inf
        conf = max(0.0, 1.0 - alpha) if math.isfinite(q) else 0.0
        return TrajectoryTube(c, np.full_like(c, q, dtype=float), conf, 1.0 - alpha)


def reachable_tube(
    centers: Sequence[Sequence[float]],
    dynamics: Sequence[Sequence[float]] | np.ndarray,
    disturbance_radius: float | Sequence[float],
    confidence: float = 1.0,
) -> TrajectoryTube:
    """Propagate a componentwise disturbance tube through linear dynamics."""
    c = np.asarray(centers, dtype=float)
    if c.ndim != 2 or c.shape[0] == 0:
        raise ValueError("centers must have shape (horizon, state_dim)")
    a = np.asarray(dynamics, dtype=float)
    if a.shape != (c.shape[1], c.shape[1]):
        raise ValueError("dynamics must be square with state dimension")
    w = np.broadcast_to(np.asarray(disturbance_radius, dtype=float), (c.shape[1],))
    radii = np.zeros_like(c)
    for t in range(1, c.shape[0]):
        radii[t] = np.abs(a) @ radii[t - 1] + w
    return TrajectoryTube(c, radii, max(0.0, min(1.0, confidence)), confidence, "componentwise linear reachable tube")


@dataclass
class SensingAction:
    key: Hashable
    expected_gain: float
    cost: float
    context: tuple[float, ...] = ()


@dataclass
class _Arm:
    pulls: int = 0
    gain_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.gain_sum / self.pulls if self.pulls else 0.0


class ActiveSensingPolicy:
    """Budgeted UCB sensing policy with an explicit exploration bonus."""

    def __init__(self, exploration: float = 0.5) -> None:
        if exploration < 0.0:
            raise ValueError("exploration must be >= 0")
        self.exploration = float(exploration)
        self._arms: dict[Hashable, _Arm] = {}
        self._round = 0

    def select(self, actions: Sequence[SensingAction]) -> SensingAction | None:
        if not actions:
            return None
        self._round += 1
        for action in actions:
            self._arms.setdefault(action.key, _Arm())
        def score(a: SensingAction) -> float:
            arm = self._arms[a.key]
            bonus = self.exploration * math.sqrt(math.log(self._round + 1.0) / (arm.pulls + 1))
            prior = max(float(a.expected_gain), 0.0)
            return (arm.mean + prior + bonus) / max(float(a.cost), 1e-12)
        return max(actions, key=score)

    def score(self, action: SensingAction) -> float:
        """Return the current UCB gain-per-cost score without advancing time.

        This lets a caller use active sensing as a safe improvement gate over
        an already-valid route-critical choice instead of unconditionally
        replacing that choice.
        """
        self._arms.setdefault(action.key, _Arm())
        arm = self._arms[action.key]
        bonus = self.exploration * math.sqrt(
            math.log(self._round + 1.0) / (arm.pulls + 1)
        )
        return (arm.mean + max(float(action.expected_gain), 0.0) + bonus) / max(float(action.cost), 1e-12)

    def update(self, key: Hashable, realized_gain: float) -> None:
        arm = self._arms.setdefault(key, _Arm())
        arm.pulls += 1
        arm.gain_sum += max(float(realized_gain), 0.0)

    def allocate(self, actions: Sequence[SensingAction], budget: float) -> list[SensingAction]:
        """Greedily fill a hard sensing budget using the live UCB scores."""
        if budget < 0.0:
            raise ValueError("budget must be >= 0")
        remaining = float(budget)
        pool = list(actions)
        chosen: list[SensingAction] = []
        while pool:
            affordable = [a for a in pool if a.cost <= remaining + 1e-12]
            pick = self.select(affordable)
            if pick is None:
                break
            chosen.append(pick)
            remaining -= pick.cost
            pool.remove(pick)
        return chosen

    def stats(self) -> dict[Hashable, dict[str, float]]:
        return {k: {"pulls": float(v.pulls), "mean_gain": v.mean} for k, v in self._arms.items()}


class RecoveryState:
    STABLE = "stable"
    ALARM = "alarm"
    RECOVERY = "recovery"


class RegimeRecoveryManager:
    """CUSUM-like p-value monitor with explicit certificate revocation."""

    def __init__(self, alarm_delta: float = 0.01, recovery_samples: int = 10, evidence_threshold: float = 4.0) -> None:
        if not 0.0 < alarm_delta < 1.0:
            raise ValueError("alarm_delta must be in (0, 1)")
        if recovery_samples < 1 or evidence_threshold <= 0.0:
            raise ValueError("invalid recovery parameters")
        self.alarm_delta = float(alarm_delta)
        self.recovery_samples = int(recovery_samples)
        self.evidence_threshold = float(evidence_threshold)
        self.state = RecoveryState.STABLE
        self.cumulative_evidence = 0.0
        self.safe_samples = 0
        self.alarm_count = 0
        self.last_p_value: float | None = None

    @property
    def certificate_allowed(self) -> bool:
        return self.state == RecoveryState.STABLE

    def observe_pvalue(self, p_value: float) -> str:
        p = min(max(float(p_value), 1e-12), 1.0)
        self.last_p_value = p
        # Repeated raw ``-log(p)`` accumulation treats every ordinary p-value
        # as evidence and creates predictable false revocations in a stable
        # stream.  Only p-values smaller than the configured alarm budget add
        # evidence; ordinary p-values dissipate it.  This is still a compact
        # CUSUM-style monitor (not a replacement for a formally calibrated
        # sequential test), but its null reference is the declared alarm
        # level rather than an arbitrary constant.
        increment = max(0.0, math.log(self.alarm_delta / p))
        self.cumulative_evidence = max(0.0, 0.9 * self.cumulative_evidence + increment)
        if self.state == RecoveryState.STABLE and self.cumulative_evidence >= self.evidence_threshold:
            self.state = RecoveryState.ALARM
            self.alarm_count += 1
            self.safe_samples = 0
        elif self.state in (RecoveryState.ALARM, RecoveryState.RECOVERY):
            self.state = RecoveryState.RECOVERY
            if p >= 0.1:
                self.safe_samples += 1
            else:
                self.safe_samples = 0
            if self.safe_samples >= self.recovery_samples:
                self.state = RecoveryState.STABLE
                self.cumulative_evidence = 0.0
                self.safe_samples = 0
        return self.state

    def force_recovery(self) -> None:
        self.state = RecoveryState.RECOVERY
        self.safe_samples = 0
        self.cumulative_evidence = 0.0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "certificate_allowed": self.certificate_allowed,
            "cumulative_evidence": self.cumulative_evidence,
            "safe_samples": self.safe_samples,
            "alarm_count": self.alarm_count,
            "last_p_value": self.last_p_value,
        }


@dataclass
class JointFleetCertificate:
    lower: float
    upper: float
    confidence: float
    agent_count: int
    radius: float
    mode: str
    congestion_cost: float = 0.0
    validity_scope: str = "joint fleet score calibrated on independent fleet episodes"

    @property
    def gap(self) -> float:
        return self.upper - self.lower

    @property
    def valid(self) -> bool:
        return self.confidence > 0.0 and math.isfinite(self.lower) and math.isfinite(self.upper)


class JointFleetCalibrator:
    """Calibrate correlated fleet residuals as one joint episode score."""

    def __init__(self, mode: str = "sum", min_calibration: int = 10) -> None:
        if mode not in ("sum", "max"):
            raise ValueError("mode must be 'sum' or 'max'")
        if min_calibration < 1:
            raise ValueError("min_calibration must be >= 1")
        self.mode = mode
        self.min_calibration = int(min_calibration)
        self._scores: list[float] = []
        self._agents: int | None = None

    def fit(self, residual_matrix: Sequence[Sequence[float]]) -> "JointFleetCalibrator":
        x = np.asarray(residual_matrix, dtype=float)
        if x.ndim != 2 or x.shape[0] < 1:
            raise ValueError("residual_matrix must have shape (episodes, agents)")
        self._agents = int(x.shape[1])
        self._scores = [
            float(abs(row).sum() if self.mode == "sum" else abs(row).max())
            for row in x
        ]
        return self

    @property
    def ready(self) -> bool:
        return len(self._scores) >= self.min_calibration and self._agents is not None

    def certify(
        self,
        predicted_costs: Sequence[float],
        alpha: float,
        congestion_cost: float = 0.0,
    ) -> JointFleetCertificate:
        if congestion_cost < 0.0:
            raise ValueError("congestion_cost must be >= 0")
        p = np.asarray(predicted_costs, dtype=float)
        if p.ndim != 1:
            raise ValueError("predicted_costs must be one-dimensional")
        if not self.ready or p.size != self._agents:
            return JointFleetCertificate(math.inf, math.inf, 0.0, int(p.size), math.inf, self.mode, congestion_cost)
        q = _finite_sample_quantile(self._scores, alpha)
        if not math.isfinite(q):
            return JointFleetCertificate(math.inf, math.inf, 0.0, int(p.size), q, self.mode, congestion_cost)
        margin = q if self.mode == "sum" else p.size * q
        total = float(p.sum() + congestion_cost)
        return JointFleetCertificate(total - margin, total + margin, 1.0 - alpha, int(p.size), q, self.mode, congestion_cost)


def congestion_penalty(
    route_loads: Mapping[Hashable, float],
    capacities: Mapping[Hashable, float] | None = None,
    coefficient: float = 1.0,
) -> float:
    """Deterministic overload penalty for shared fleet resources."""
    if coefficient < 0.0:
        raise ValueError("coefficient must be >= 0")
    total = 0.0
    for resource, raw_load in route_loads.items():
        load = max(float(raw_load), 0.0)
        cap = float(capacities[resource]) if capacities and resource in capacities else 1.0
        if cap <= 0.0:
            raise ValueError("resource capacities must be > 0")
        total += coefficient * max(0.0, load - cap) ** 2
    return total
