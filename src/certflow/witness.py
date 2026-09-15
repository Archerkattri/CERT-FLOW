"""Selection-aware route witnesses and auditable risk spending.

This module is a conservative CF1/CF2 seam.  It does not claim conditional
coverage by itself: a witness is only valid when its declared calibration and
drift assumptions hold, and the risk ledger records the finite spending plan
that an external sequential construction must justify.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Hashable, Sequence


def _finite(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


@dataclass(frozen=True)
class RiskSpend:
    amount: float
    reason: str
    scope: str


class RiskLedger:
    """Finite, append-only risk-spending record for an operation."""

    def __init__(self, alpha_total: float) -> None:
        alpha = _finite(alpha_total, "alpha_total")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha_total must be in (0, 1)")
        self.alpha_total = alpha
        self._spends: list[RiskSpend] = []

    def spend(self, amount: float, *, reason: str, scope: str) -> RiskSpend:
        value = _finite(amount, "amount")
        if value <= 0.0:
            raise ValueError("amount must be > 0")
        if not reason or not scope:
            raise ValueError("reason and scope must be non-empty")
        if self.spent + value > self.alpha_total + 1e-15:
            raise ValueError("risk ledger would exceed alpha_total")
        item = RiskSpend(value, str(reason), str(scope))
        self._spends.append(item)
        return item

    @property
    def spends(self) -> tuple[RiskSpend, ...]:
        return tuple(self._spends)

    @property
    def spent(self) -> float:
        return float(sum(item.amount for item in self._spends))

    @property
    def remaining(self) -> float:
        return max(0.0, self.alpha_total - self.spent)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "alpha_total": self.alpha_total,
                "spends": [item.__dict__ for item in self._spends],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class RouteWitness:
    """A finite graph certificate witness with explicit selection history."""

    route: tuple[Hashable, ...]
    incumbent_upper: float
    optimistic_lower: float
    epoch: int
    selection_rule: str
    observation_ids: tuple[str, ...] = ()
    risk_spent: float = 0.0
    risk_total: float = 1.0
    validity_scope: str = "finite graph; declared calibration and drift assumptions"

    def __post_init__(self) -> None:
        if len(self.route) < 2:
            raise ValueError("route must contain at least two nodes")
        if self.epoch < 0:
            raise ValueError("epoch must be >= 0")
        if not self.selection_rule:
            raise ValueError("selection_rule must be non-empty")
        upper = _finite(self.incumbent_upper, "incumbent_upper")
        lower = _finite(self.optimistic_lower, "optimistic_lower")
        spent = _finite(self.risk_spent, "risk_spent")
        total = _finite(self.risk_total, "risk_total")
        if lower > upper:
            raise ValueError("optimistic_lower cannot exceed incumbent_upper")
        if not 0.0 <= spent <= total <= 1.0:
            raise ValueError("risk values must satisfy 0 <= spent <= total <= 1")

    @property
    def gap(self) -> float:
        return float(self.incumbent_upper - self.optimistic_lower)

    @property
    def valid(self) -> bool:
        return self.gap >= 0.0 and bool(self.validity_scope)


@dataclass(frozen=True)
class SensingCandidate:
    key: Hashable
    expected_gap_reduction: float
    cost: float
    challenger: bool = False

    def __post_init__(self) -> None:
        reduction = _finite(self.expected_gap_reduction, "expected_gap_reduction")
        cost = _finite(self.cost, "cost")
        if reduction < 0.0 or cost <= 0.0:
            raise ValueError("candidate reduction must be >= 0 and cost must be > 0")


def select_witness_observations(
    candidates: Sequence[SensingCandidate],
    *,
    budget: float,
    exploration_reserve: float = 0.25,
    max_items: int | None = None,
) -> tuple[SensingCandidate, ...]:
    """Greedily spend a sensing budget while reserving challenger coverage.

    The returned plan is an acquisition heuristic, not a coverage theorem.
    Ties are deterministic and the reserve prevents a route-only policy from
    spending every observation on the incumbent path.
    """
    cap = _finite(budget, "budget")
    reserve = _finite(exploration_reserve, "exploration_reserve")
    if cap <= 0.0 or not 0.0 <= reserve <= 1.0:
        raise ValueError("budget must be > 0 and exploration_reserve must be in [0, 1]")
    unique = {candidate.key: candidate for candidate in candidates}
    ordered = sorted(
        unique.values(),
        key=lambda item: (-(item.expected_gap_reduction / item.cost), str(item.key)),
    )
    chosen: list[SensingCandidate] = []
    spent = 0.0
    challengers = [item for item in ordered if item.challenger]
    if challengers and reserve > 0.0:
        first = challengers[0]
        if first.cost <= cap:
            chosen.append(first)
            spent += first.cost
    for item in ordered:
        if item in chosen or spent + item.cost > cap + 1e-12:
            continue
        chosen.append(item)
        spent += item.cost
        if max_items is not None and len(chosen) >= max_items:
            break
    return tuple(chosen)

