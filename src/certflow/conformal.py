"""Conformal certificate machinery.

Implements the three statistical layers from the theory note
(docs/theory/theory-notes.md, sections 2, 4, 5):

1. Age-weighted non-exchangeable split conformal quantile
   (Barber, Candes, Ramdas, Tibshirani 2023, Theorem 2): weights are a fixed,
   data-independent function of sample age only (section 5.4.1 constraint).
2. The staleness coverage correction Delta_stale, from their independence
   corollary: coverage gap <= sum_i w~_i * d_TV_i with
   d_TV_i <= min(1, 2 * eps_tv * age_i) under the TV-Lipschitz drift
   assumption A2.
3. Adaptive Conformal Inference safety net (Gibbs & Candes 2021, Eq. 2 +
   Prop 4.1): assumption-free long-run miscoverage control.

The quantile returns +inf while the buffer cannot support level alpha
(warm-up): the certificate is INVALID by construction, never silently wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CalSample:
    residual: float  # nonconformity score at collection time (may be negative,
                     # e.g. the drift-adjusted score |obs - c_hat| - rho * age)
    t: float         # collection time


class ConformalScorer:
    """Rolling buffer of residuals with age-geometric weights.

    rho_w is the decay per unit time: w_i = rho_w ** age_i. Calibrate rho_w to
    the sensing period (e.g. rho_w = 0.99 ** (1/delta) gives the paper's 0.99
    per round). eps_tv is the assumed TV-Lipschitz drift rate (A2); 0 recovers
    the exchangeable guarantee.
    """

    def __init__(
        self,
        rho_w: float = 0.99,
        eps_tv: float = 0.0,
        max_buffer: int = 2000,
    ) -> None:
        if not 0.0 < rho_w <= 1.0:
            raise ValueError("rho_w must be in (0, 1]")
        if eps_tv < 0.0:
            raise ValueError("eps_tv must be >= 0")
        self.rho_w = rho_w
        self.eps_tv = eps_tv
        self.max_buffer = max_buffer
        self._buf: list[CalSample] = []
        self._signed: list[CalSample] = []  # signed deviations (sum-aware UB)

    def push(self, residual: float, t: float) -> None:
        self._buf.append(CalSample(residual, t))
        if len(self._buf) > self.max_buffer:
            # drop the oldest by collection time (buffer arrives ~ordered)
            self._buf.sort(key=lambda s: s.t)
            del self._buf[0 : len(self._buf) - self.max_buffer]

    def __len__(self) -> int:
        return len(self._buf)

    def _weights(self, t: float) -> list[float]:
        return [self.rho_w ** max(0.0, t - s.t) for s in self._buf]

    def quantile(self, alpha: float, t: float) -> float:
        """Weighted (1-alpha)-quantile with test mass w~_{n+1} at +inf.

        Barber et al. split conformal: q = Quantile_{1-alpha} of
        sum_i w~_i delta_{R_i} + w~_{n+1} delta_{+inf}. Returns +inf when the
        finite mass cannot reach 1-alpha (warm-up / too-stale buffer).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not self._buf:
            return math.inf
        w = self._weights(t)
        total = sum(w) + 1.0  # +1.0 is the test point's unnormalized weight
        target = (1.0 - alpha) * total
        pairs = sorted(zip((s.residual for s in self._buf), w))
        acc = 0.0
        for r, wi in pairs:
            acc += wi
            if acc >= target - 1e-12:
                return r
        return math.inf

    def delta_stale(self, t: float) -> float:
        """Coverage correction: sum_i w~_i * min(1, 2 * eps_tv * age_i).

        Uses the realized ages of the calibration buffer (sharper than the
        age-uniform 2*eps_tv/(1-rho_w) corollary bound). Capped at 1.
        """
        if not self._buf:
            return 1.0
        w = self._weights(t)
        total = sum(w) + 1.0
        gap = sum(
            wi * min(1.0, 2.0 * self.eps_tv * max(0.0, t - s.t))
            for wi, s in zip(w, self._buf)
        ) / total
        return min(1.0, gap)

    def ready(self, alpha: float, t: float) -> bool:
        return math.isfinite(self.quantile(alpha, t))

    def effective_mass(self, t: float) -> float:
        """Sum of (unnormalized) weights: the buffer's effective sample size.
        The smallest supportable per-sample level is 1/(mass+1) — below that
        the weighted quantile is +inf (alpha-annealing uses this floor)."""
        return sum(self._weights(t))

    def push_signed(self, deviation: float, t: float) -> None:
        """Signed drift-bracketed deviation (obs - c_hat_prev); feeds the
        sum-aware block quantile. Kept separate from the absolute scores."""
        self._signed.append(CalSample(deviation, t))
        if len(self._signed) > self.max_buffer:
            self._signed.sort(key=lambda s: s.t)
            del self._signed[0 : len(self._signed) - self.max_buffer]

    def block_quantile(self, alpha: float, t: float, block_len: int) -> float:
        """Sum-aware path margin (theory.tex T4): one-sided (1-alpha)
        weighted conformal quantile of sums of `block_len` signed deviations.

        Blocks are consecutive in collection time (newest first) so member
        ages are similar; the block weight is the minimum member weight (a
        data-independent function of ages, as Barber requires). Test mass at
        +inf as usual; returns +inf while fewer than ~1/alpha blocks exist.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if block_len < 1:
            raise ValueError("block_len must be >= 1")
        samples = sorted(self._signed, key=lambda s: s.t, reverse=True)
        n_blocks = len(samples) // block_len
        if n_blocks == 0:
            return math.inf
        sums, weights = [], []
        for b in range(n_blocks):
            block = samples[b * block_len : (b + 1) * block_len]
            sums.append(sum(s.residual for s in block))
            weights.append(min(self.rho_w ** max(0.0, t - s.t) for s in block))
        total = sum(weights) + 1.0
        target = (1.0 - alpha) * total
        acc = 0.0
        for v, w in sorted(zip(sums, weights)):
            acc += w
            if acc >= target - 1e-12:
                return v
        return math.inf

    def block_delta_stale(self, t: float, block_len: int) -> float:
        """Coverage correction at block level: TV of a block factorizes over
        members (independence), so the per-block term is min(1, sum of member
        2*eps_tv*age terms), weighted like block_quantile."""
        samples = sorted(self._signed, key=lambda s: s.t, reverse=True)
        n_blocks = len(samples) // block_len
        if n_blocks == 0:
            return 1.0
        terms, weights = [], []
        for b in range(n_blocks):
            block = samples[b * block_len : (b + 1) * block_len]
            tv = sum(2.0 * self.eps_tv * max(0.0, t - s.t) for s in block)
            terms.append(min(1.0, tv))
            weights.append(min(self.rho_w ** max(0.0, t - s.t) for s in block))
        total = sum(weights) + 1.0
        return min(1.0, sum(w * d for w, d in zip(weights, terms)) / total)


class ACITracker:
    """Adaptive Conformal Inference (Gibbs & Candes 2021).

    Update: alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t).
    Prop 4.1: |mean(err) - alpha_target| <= (max(a1, 1-a1) + gamma) / (T*gamma)
    for ANY sequence, provided the caller respects the boundary convention:
    working_alpha() clips into (0, 1); when raw alpha_t <= 0 the prediction set
    is everything (so err must be 0), when >= 1 it is empty (err must be 1).
    """

    def __init__(self, alpha_target: float, gamma: float = 0.005) -> None:
        if not 0.0 < alpha_target < 1.0:
            raise ValueError("alpha_target must be in (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0")
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.alpha_raw = alpha_target
        self._t = 0
        self._errs = 0

    def working_alpha(self, lo: float = 1e-4) -> float:
        return min(1.0 - lo, max(lo, self.alpha_raw))

    def update(self, err: bool) -> None:
        self.alpha_raw += self.gamma * (self.alpha_target - (1.0 if err else 0.0))
        self._t += 1
        self._errs += 1 if err else 0

    def empirical_miscoverage(self) -> float:
        return self._errs / self._t if self._t else math.nan

    def coverage_bound(self) -> float:
        """RHS of Prop 4.1 for the current horizon."""
        if self._t == 0:
            return math.inf
        a1 = self.alpha_target
        return (max(a1, 1.0 - a1) + self.gamma) / (self._t * self.gamma)


def path_alpha_edge(alpha_prime: float, path_len: int) -> float:
    """Bonferroni split: per-edge level for a path-level alpha_prime."""
    if path_len <= 0:
        raise ValueError("path_len must be >= 1")
    return alpha_prime / path_len


def path_confidence(
    alpha_prime: float, deltas_stale: list[float]
) -> float:
    """Certificate confidence: 1 - alpha' - sum(Delta_stale over path edges).

    May be <= 0, which marks the certificate INVALID (types.Certificate.valid).
    """
    return 1.0 - alpha_prime - sum(deltas_stale)


class AgeBinnedScorer:
    """Per-age-bin conformal calibration for predictor-mode intervals.

    The spatial-predictor study (docs/results/spatial-predictor-study.md)
    showed predictor residual scale is strongly age-dependent; a pooled
    buffer would miscalibrate every regime. Scores are routed to the bin of
    the edge's age at observation; queries use the bin of the edge's current
    age. Within-bin exchangeability replaces the global A4 assumption
    (assumption A4': residual distribution depends on age only through the
    bin). An unready bin returns +inf — annealing handles it upstream.
    """

    def __init__(
        self,
        bin_edges: tuple[float, ...] = (6.0, 12.0, 24.0, 48.0),
        rho_w: float = 0.99,
        eps_tv: float = 0.0,
        max_buffer: int = 2000,
    ) -> None:
        self.bin_edges = tuple(bin_edges)
        self._bins = [
            ConformalScorer(rho_w=rho_w, eps_tv=eps_tv, max_buffer=max_buffer)
            for _ in range(len(self.bin_edges) + 1)
        ]

    def _bin(self, age: float) -> "ConformalScorer":
        for i, edge in enumerate(self.bin_edges):
            if age < edge:
                return self._bins[i]
        return self._bins[-1]

    def push(self, score: float, t: float, age: float) -> None:
        self._bin(age).push(score, t)

    def quantile(self, alpha: float, t: float, age: float) -> float:
        return self._bin(age).quantile(alpha, t)

    def delta_stale(self, t: float, age: float) -> float:
        return self._bin(age).delta_stale(t)

    def effective_mass(self, t: float, age: float) -> float:
        return self._bin(age).effective_mass(t)

    def __len__(self) -> int:
        return sum(len(b) for b in self._bins)
