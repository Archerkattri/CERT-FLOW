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
        shift_model: str = "tv",
        eps_lp: float = 0.0,
        rho_lp: float = 0.0,
    ) -> None:
        if not 0.0 < rho_w <= 1.0:
            raise ValueError("rho_w must be in (0, 1]")
        if eps_tv < 0.0:
            raise ValueError("eps_tv must be >= 0")
        if shift_model not in ("tv", "lp"):
            raise ValueError("shift_model must be 'tv' or 'lp'")
        if eps_lp < 0.0:
            raise ValueError("eps_lp must be >= 0")
        if not 0.0 <= rho_lp < 1.0:
            raise ValueError("rho_lp must be in [0, 1)")
        self.rho_w = rho_w
        self.eps_tv = eps_tv
        self.max_buffer = max_buffer
        # Levy-Prokhorov distribution-shift model (arXiv 2502.14105): an
        # alternative to the TV-Lipschitz staleness bound. eps_lp is the
        # smooth-drift budget (a flat additive offset on the score quantile,
        # units of the score) and rho_lp is the mass of abruptly / adversarially
        # changed edges (a flat level shift + coverage penalty). Defaults 0
        # recover plain conformal even under shift_model="lp".
        self.shift_model = shift_model
        self.eps_lp = eps_lp
        self.rho_lp = rho_lp
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

    def cdf(self, x: float, t: float) -> float:
        """Weighted empirical CDF F_P(x) of the buffered scores, using the
        same age-geometric normalized weights as :meth:`quantile` (the test
        point's +inf mass is NOT counted here: this is the calibration CDF)."""
        if not self._buf:
            return 0.0
        w = self._weights(t)
        total = sum(w) + 1.0  # match quantile's normalization (test mass at inf)
        acc = sum(wi for r, wi in zip((s.residual for s in self._buf), w) if r <= x)
        return acc / total

    def quantile_lp(
        self,
        alpha: float,
        t: float,
        eps: float | None = None,
        rho: float | None = None,
    ) -> float:
        """Levy-Prokhorov worst-case (1-alpha) quantile (arXiv 2502.14105).

        Under an LP(eps, rho) ambiguity ball around the calibration law P, the
        worst-case quantile at coverage level ``beta = 1 - alpha`` is

            Quant^WC_{eps,rho}(beta; P) = Quant(beta + rho; P) + eps

        i.e. shift the quantile *level* up by ``rho`` (to absorb the mass of
        abruptly-changed edges) and add the flat smooth-drift offset ``eps``.
        Since :meth:`quantile` returns the ``1 - alpha`` weighted quantile,
        ``Quant(beta + rho)`` is ``quantile(alpha - rho)``. When ``alpha - rho
        <= 0`` the required level is >= 1 and no finite quantile supports it
        (returns +inf, which marks the certificate invalid -- never wrong).

        eps=rho=0 recovers plain conformal exactly. The result is monotone
        non-decreasing in both eps and rho, so LP intervals are always at least
        as wide as (hence at least as sound as) the exchangeable quantile.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        eps = self.eps_lp if eps is None else eps
        rho = self.rho_lp if rho is None else rho
        if eps < 0.0:
            raise ValueError("eps must be >= 0")
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        alpha_eff = alpha - rho
        if alpha_eff <= 0.0:
            return math.inf  # required coverage level >= 1: unsupportable
        q = self.quantile(alpha_eff, t)
        return q + eps if math.isfinite(q) else math.inf

    def coverage_lp(
        self,
        q: float,
        t: float,
        eps: float | None = None,
        rho: float | None = None,
    ) -> float:
        """Levy-Prokhorov worst-case coverage of a set of radius ``q``:

            Cov^WC_{eps,rho}(q; P) = F_P(q - eps) - rho,

        floored at 0. F_P is the weighted calibration CDF (:meth:`cdf`)."""
        eps = self.eps_lp if eps is None else eps
        rho = self.rho_lp if rho is None else rho
        return max(0.0, self.cdf(q - eps, t) - rho)

    @staticmethod
    def lp_finite_sample_coverage(n: int, alpha: float, rho: float) -> float:
        """Finite-sample LP coverage floor (arXiv 2502.14105):

            P{Y in C} >= ceil(n * (1 - alpha + rho)) / (n + 1) - rho.

        Reduces to the usual split-conformal ceil(n(1-alpha))/(n+1) at rho=0."""
        if n < 0:
            raise ValueError("n must be >= 0")
        return math.ceil(n * (1.0 - alpha + rho)) / (n + 1) - rho

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

    def __init__(
        self,
        alpha_target: float,
        gamma: float = 0.005,
        mode: str = "fixed",
        eta: float = 0.1,
    ) -> None:
        if not 0.0 < alpha_target < 1.0:
            raise ValueError("alpha_target must be in (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0")
        if mode not in ("fixed", "sf-ogd"):
            raise ValueError("mode must be 'fixed' or 'sf-ogd'")
        if eta <= 0.0:
            raise ValueError("eta must be > 0")
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.mode = mode
        self.eta = eta
        self.alpha_raw = alpha_target
        self._t = 0
        self._errs = 0
        self._grad_sq_sum = 0.0  # SF-OGD adaptive denominator sum_i ||g_i||^2

    def working_alpha(self, lo: float = 1e-4) -> float:
        return min(1.0 - lo, max(lo, self.alpha_raw))

    def update(self, err: bool) -> None:
        # ACI gradient of the pinball loss w.r.t. the level: g_t = err_t - target
        # (so the fixed step alpha += gamma*(target - err) is -gamma * g_t).
        g = (1.0 if err else 0.0) - self.alpha_target
        if self.mode == "sf-ogd":
            # Scale-free OGD (SAOCP, arXiv 2302.07869, Alg. 2): step
            # s_{t+1} = s_t - eta * g_t / sqrt(sum_{i<=t} ||g_i||^2). Anytime,
            # scale-free: no fixed gamma to tune to the score/err magnitude.
            self._grad_sq_sum += g * g
            denom = math.sqrt(self._grad_sq_sum)
            if denom > 0.0:
                self.alpha_raw -= self.eta * g / denom
        else:
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


def weighted_group_quantile(
    scores: list[float], weights: list[float], alpha: float
) -> float:
    """Weighted (1-alpha) quantile of ``scores`` with a test point of unit
    weight at +inf (the split-conformal ``union {+inf}`` convention).

    With uniform unit weights this is exactly the ``ceil((1+K)(1-alpha))``-th
    smallest score (returning +inf when that rank exceeds K), i.e. CIA's
    ``(1-alpha)(1+n)/n`` calibration percentile. With age-geometric weights it
    is the non-exchangeable weighted quantile (Barber et al. 2023), which is
    what the drift retrofit uses.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not scores:
        return math.inf
    total = sum(weights) + 1.0  # +1.0 = the test point's unnormalized weight
    target = (1.0 - alpha) * total
    acc = 0.0
    for s, w in sorted(zip(scores, weights)):
        acc += w
        if acc >= target - 1e-12:
            return s
    return math.inf  # finite mass cannot reach 1-alpha: rank fell on the +inf pt


@dataclass
class CIAResult:
    """A CIA group-sum certificate for one predicted path sum."""

    lo: float
    ub: float
    Q: float
    delta: float           # overlap degradation: coverage >= 1 - alpha - delta
    coverage_level: float  # 1 - alpha - delta (the HONEST level, floored at 0)


class CIACalibrator:
    """CIA-style group-sum (path-level) calibration (arXiv 2408.10939).

    Replaces per-edge Bonferroni with one nonconformity score on the whole
    path sum, so the margin concentrates like ``sqrt(L)`` instead of summing
    ``L`` per-edge margins. For calibration groups (observed paths / edge-sets)
    ``S_k`` with per-index residuals ``y_i - yhat_i``::

        s_k = | sum_{i in S_k} (y_i - yhat_i) |
        Q   = ceil((1+K)(1-alpha))-th smallest of {s_k} u {+inf}
        C(pred) = [ pred - Q , pred + Q ]

    Options
    -------
    symmetric:
        Symmetric calibration for overlapping groups (CIA Sec. "Group
        sampling"): assign each element index to cal / test independently with
        prob 0.5; score group k on ``S_k intersect cal`` only; Q from the
        cal-half scores union {+inf}. This decorrelates the calibration and
        test sums when groups share elements.
    rho_w:
        Age-geometric decay for the DRIFT RETROFIT: when < 1 (and per-group
        ``times`` are supplied to :meth:`fit`), Q is pulled from the AGE-WEIGHTED
        empirical CDF using CERT-FLOW's existing non-exchangeable weights
        (weight ``rho_w ** (t - time_k)``), not the unweighted one. This
        inherits the weighted-coverage argument of Barber et al. 2023 Thm 2 --
        the same argument the single-agent certificate already relies on -- so
        coverage stays valid under drift where the unweighted CIA quantile
        (calibrated to a stale slice) fails.
    stratify:
        Compute a separate Q per path length (number of elements in the group),
        so short and long paths are not calibrated against each other's
        (differently-scaled) sums.

    Overlap degradation
    -------------------
    Groups that share elements have correlated sums; the coverage guarantee
    degrades to ``>= 1 - alpha - delta`` where ``delta`` is the max pairwise
    overlap fraction ``max_{k != l} |S_k intersect S_l| / |S_k|``. The honest
    level ``1 - alpha - delta`` is exposed on every :class:`CIAResult` so callers
    never read the nominal level off a certificate whose groups overlap.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        rho_w: float = 1.0,
        symmetric: bool = False,
        stratify: bool = False,
        seed: int = 0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 < rho_w <= 1.0:
            raise ValueError("rho_w must be in (0, 1]")
        self.alpha = alpha
        self.rho_w = rho_w
        self.symmetric = symmetric
        self.stratify = stratify
        self.seed = seed
        self._fitted = False
        self.delta = 0.0
        # per-length-bucket (scores, weights); key None = pooled (no stratify)
        self._buckets: dict[object, tuple[list[float], list[float]]] = {}

    @staticmethod
    def _overlap_delta(groups: list[list[int]]) -> float:
        sets = [set(g) for g in groups]
        delta = 0.0
        for k in range(len(sets)):
            if not sets[k]:
                continue
            for l in range(len(sets)):
                if k == l:
                    continue
                inter = len(sets[k] & sets[l])
                if inter:
                    delta = max(delta, inter / len(sets[k]))
        return min(1.0, delta)

    def fit(
        self,
        groups: list[list[int]],
        residuals,
        times: list[float] | None = None,
        t: float = 0.0,
    ) -> "CIACalibrator":
        """Calibrate on groups ``S_k`` (element-index lists) with per-index
        ``residuals`` (indexable by element id). ``times[k]`` and ``t`` drive
        the age weight when ``rho_w < 1``."""
        import numpy as _np

        rng = _np.random.default_rng(self.seed)
        self.delta = self._overlap_delta(groups)
        self._buckets = {}
        for k, g in enumerate(groups):
            if not g:
                continue
            if self.symmetric:
                in_cal = {i: bool(rng.integers(0, 2)) for i in g}
                members = [i for i in g if in_cal[i]]
            else:
                members = list(g)
            if not members:
                continue
            s_k = abs(sum(float(residuals[i]) for i in members))
            if times is not None and self.rho_w < 1.0:
                w_k = self.rho_w ** max(0.0, t - times[k])
            else:
                w_k = 1.0
            key = len(g) if self.stratify else None
            self._buckets.setdefault(key, ([], []))
            self._buckets[key][0].append(s_k)
            self._buckets[key][1].append(w_k)
        self._fitted = True
        return self

    def Q(self, path_len: int | None = None) -> float:
        """CIA radius Q for a path of ``path_len`` elements (stratified) or the
        pooled radius (``stratify=False`` / ``path_len=None``)."""
        if not self._fitted:
            raise RuntimeError("call fit() first")
        key = path_len if self.stratify else None
        bucket = self._buckets.get(key)
        if bucket is None:
            return math.inf  # no calibration groups of this length
        scores, weights = bucket
        return weighted_group_quantile(scores, weights, self.alpha)

    def interval(self, pred_sum: float, path_len: int | None = None) -> CIAResult:
        """CIA interval ``[pred_sum - Q, pred_sum + Q]`` with the honest
        overlap-degraded coverage level attached."""
        q = self.Q(path_len)
        return CIAResult(
            lo=pred_sum - q,
            ub=pred_sum + q,
            Q=q,
            delta=self.delta,
            coverage_level=max(0.0, 1.0 - self.alpha - self.delta),
        )


@dataclass
class PASCResult:
    """A PASC joint per-edge radius for a fresh path."""

    Q: float               # single radius applied to EVERY edge of the path
    coverage_level: float  # 1 - alpha (joint over all edges, under group exch.)
    delta: float           # overlap diagnostic (see PASCCalibrator)


class PASCCalibrator:
    """PASC-style joint per-edge calibration (arXiv 2605.18812).

    Replaces per-edge **Bonferroni** pricing (``alpha / L`` per edge, then union)
    for the case where CERT-FLOW needs *all* edge prices on a path to be
    simultaneously valid -- which is exactly what the dual optimistic/conservative
    searches consume. Instead of an ``alpha / L`` quantile per edge, PASC
    calibrates ONE quantile of the **joint maximum nonconformity score** across a
    path's edges::

        s_k = max_{i in S_k} | y_i - yhat_i |          (per calibration path S_k)
        Q   = weighted (1 - alpha) quantile of {s_k} u {+inf}
        price every edge of a fresh path at  yhat_e +/- Q

    Guarantee (under exchangeability of the calibration groups): for a fresh path,
    ``P( for all edges e:  |y_e - yhat_e| <= Q ) >= 1 - alpha`` -- joint over the
    whole path in a SINGLE scalar quantile, not an ``alpha/L`` correction. Because
    ``Q`` is the standard split-conformal weighted quantile (the same valid
    ``weighted_group_quantile`` the rest of this module uses, with the ``u{+inf}``
    test point), soundness does not depend on any PASC-specific level constant.

    Drift retrofit (``rho_w < 1`` with per-group ``times``): ``Q`` is pulled from
    the AGE-WEIGHTED quantile, inheriting the Barber et al. 2023 weighted-coverage
    argument -- the same non-exchangeable retrofit as :class:`CIACalibrator`.

    ``delta`` is reported for honesty (max pairwise edge-overlap fraction across
    calibration groups): the clean guarantee assumes the calibration paths are
    exchangeable with the test path; heavy edge reuse across groups is the regime
    where that assumption is weakest.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        rho_w: float = 1.0,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 < rho_w <= 1.0:
            raise ValueError("rho_w must be in (0, 1]")
        self.alpha = alpha
        self.rho_w = rho_w
        self._fitted = False
        self.delta = 0.0
        self._scores: list[float] = []
        self._weights: list[float] = []

    def fit(
        self,
        groups: list[list[int]],
        residuals,
        times: list[float] | None = None,
        t: float = 0.0,
    ) -> "PASCCalibrator":
        """Calibrate on paths ``S_k`` (edge-index lists) with per-edge signed
        ``residuals`` (indexable by edge id). The per-group score is the MAX
        absolute residual over the path's edges."""
        self.delta = CIACalibrator._overlap_delta(groups)
        self._scores, self._weights = [], []
        for k, g in enumerate(groups):
            if not g:
                continue
            s_k = max(abs(float(residuals[i])) for i in g)
            if times is not None and self.rho_w < 1.0:
                w_k = self.rho_w ** max(0.0, t - times[k])
            else:
                w_k = 1.0
            self._scores.append(s_k)
            self._weights.append(w_k)
        self._fitted = True
        return self

    def Q(self) -> float:
        """The single joint per-edge radius (weighted 1-alpha quantile of the
        per-path max scores). ``+inf`` while too few groups exist (warm-up)."""
        if not self._fitted:
            raise RuntimeError("call fit() first")
        if not self._scores:
            return math.inf
        return weighted_group_quantile(self._scores, self._weights, self.alpha)

    def result(self) -> PASCResult:
        return PASCResult(
            Q=self.Q(),
            coverage_level=1.0 - self.alpha,
            delta=self.delta,
        )


# --------------------------------------------------------------------------- #
# Testability layer: making the pinned-at-1.0 coverage an observable quantity.
#
# CERT-FLOW's certificate is sound but over-conservative: empirical coverage
# sits at exactly 1.0, so calibration is untestable and there is no signal for
# how much tighter one could safely go. The tools below add that signal without
# weakening any guarantee:
#   * conformal_p_value: weighted conformal p-value (uniform under the weighted-
#     exchangeability null the certificate already assumes);
#   * ConformalTestMartingale (WATCH, arXiv 2505.04608; Vovk test martingales):
#     alarms when that null is violated (staleness model breaking / under-cover),
#     with Ville false-alarm control -- and doubles as a tightness stress test
#     (shrink the radius until the martingale approaches its alarm threshold);
#   * conformal_e_value / merge_e_values (Vovk-Wang; arXiv 2503.13050): e-values
#     and their admissible merges (average = valid under arbitrary dependence;
#     product = the sequential test martingale).
# --------------------------------------------------------------------------- #


def conformal_p_value(
    score: float,
    cal_scores: list[float],
    cal_weights: list[float] | None = None,
    test_weight: float = 1.0,
    u: float | None = None,
) -> float:
    """Weighted conformal p-value of a fresh nonconformity ``score`` against the
    weighted calibration scores.

        p = ( sum_i w_i 1[R_i > R] + tau (w_test + sum_i w_i 1[R_i = R]) )
            / ( sum_i w_i + w_test )

    with ``tau = u`` for the smoothed p-value (``u ~ U(0,1)``; exactly uniform
    under the null) or ``tau = 1`` for the conservative (super-uniform) p-value
    when ``u is None``. A *small* p means the score is anomalously large --
    evidence the calibration under-predicts, i.e. the drift/staleness model is
    breaking. Under weighted exchangeability (Barber et al. 2023) the p-value is
    (super-)uniform, which is exactly what the test martingale needs.
    """
    n = len(cal_scores)
    w = [1.0] * n if cal_weights is None else list(cal_weights)
    if len(w) != n:
        raise ValueError("cal_weights length must match cal_scores")
    greater = sum(wi for r, wi in zip(cal_scores, w) if r > score)
    equal = sum(wi for r, wi in zip(cal_scores, w) if r == score)
    tau = 1.0 if u is None else float(u)
    total = sum(w) + test_weight
    if total <= 0.0:
        return 1.0
    return (greater + tau * (test_weight + equal)) / total


def conformal_e_value(p: float, epsilon: float = 0.5) -> float:
    """Conformal e-value from a p-value via the power betting density
    ``f(p) = epsilon * p**(epsilon-1)`` (``epsilon in (0,1)``). It is a valid
    e-value: ``E[f(p)] = integral_0^1 f = 1`` under the uniform null, and since
    ``f`` is decreasing it stays an e-value (``E <= 1``) under a super-uniform
    (conservative) p-value too. By Markov, ``P(e >= 1/alpha) <= alpha``. Large e
    = evidence the score was anomalously large (null violated)."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    p = min(1.0, max(1e-12, p))  # clamp; p=0 would be +inf evidence
    return epsilon * p ** (epsilon - 1.0)


def merge_e_values(evalues: list[float], method: str = "average") -> float:
    """Merge e-values into one e-value.

    ``average`` (arithmetic mean) is valid under ARBITRARY dependence (Vovk &
    Wang 2021) -- the safe default. ``product`` is valid only under sequential /
    independence structure (it is the test martingale). Both preserve
    ``E[merged] <= 1`` under their respective assumptions, so the Markov alarm
    ``merged >= 1/alpha`` keeps false-alarm probability ``<= alpha``.
    """
    if not evalues:
        return 1.0
    if method == "average":
        return sum(evalues) / len(evalues)
    if method == "product":
        out = 1.0
        for e in evalues:
            out *= e
        return out
    raise ValueError("method must be 'average' or 'product'")


class ConformalTestMartingale:
    """Weighted conformal test martingale (WATCH, arXiv 2505.04608).

    Feed it the per-round weighted conformal p-values (:func:`conformal_p_value`).
    It bets against the exchangeability null with the power betting function
    ``f(p) = epsilon * p**(epsilon-1)`` (``epsilon in (0,1)`` bets on small p, i.e.
    on anomalously large scores = under-coverage). The wealth process
    ``M_t = prod_{s<=t} f(p_s)`` is a non-negative (super)martingale with
    ``M_0 = 1``, so Ville's inequality gives ``P(sup_t M_t >= 1/delta) <= delta``
    under the null. :meth:`alarm` fires at ``M_t >= 1/delta``: the staleness /
    weighting model the certificate relies on is being violated, with false-alarm
    probability at most ``delta``.

    Two uses:
      1. **Validity monitor.** An alarm means the certificate's assumptions broke
         (e.g. a regime change the age-weighting did not absorb) -- act on it.
      2. **Tightness stress test.** Replay with a *shrunken* radius (tighter
         certificate). The martingale stays flat while the tighter certificate is
         still valid and climbs once it over-tightens; the largest shrink that
         keeps ``M_t`` below the alarm is the tightest safe certificate. This is
         the observable that the pinned-at-1.0 coverage never provided.

    Works in log-space for numerical stability over long horizons.
    """

    def __init__(self, epsilon: float = 0.5, alarm_delta: float = 0.01) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        if not 0.0 < alarm_delta < 1.0:
            raise ValueError("alarm_delta must be in (0, 1)")
        self.epsilon = epsilon
        self.alarm_delta = alarm_delta
        self._log_m = 0.0
        self._t = 0
        self._log_max = 0.0

    def update(self, p: float) -> float:
        """Bet on one conformal p-value; return the current martingale value."""
        e = conformal_e_value(p, self.epsilon)
        self._log_m += math.log(max(e, 1e-300))
        self._log_max = max(self._log_max, self._log_m)
        self._t += 1
        return self.value

    @property
    def value(self) -> float:
        return math.exp(self._log_m)

    @property
    def running_max(self) -> float:
        return math.exp(self._log_max)

    def alarm(self) -> bool:
        """True once the martingale has ever crossed the Ville threshold
        ``1/alarm_delta`` (evidence the exchangeability null is violated)."""
        return self._log_max >= math.log(1.0 / self.alarm_delta)


class ShiryaevRobertsDetector:
    """Shiryaev-Roberts e-detector for a change after a possibly LONG null period
    (WATCH Prop 3.3, arXiv 2505.04608).

    A plain conformal test martingale (:class:`ConformalTestMartingale`) can random-
    walk toward zero over a long null run, so a *late* violation has to overcome
    that accumulated decay before it can alarm. The Shiryaev-Roberts statistic
    restarts implicitly at every step and does not::

        R_0 = 0 ;  R_t = (1 + R_{t-1}) * e_t

    with ``e_t`` the conformal e-value (:func:`conformal_e_value`). Under the null
    ``E[R_t] = t`` (it grows only linearly), and the average run length to a false
    alarm at threshold ``c`` is ``>= c`` (Pollak/Tartakovsky; WATCH Prop 3.3). After
    a change ``e_t >> 1`` sustains and ``R_t`` explodes, so :meth:`alarm` fires
    quickly regardless of how long the null ran first. Pick ``threshold`` as the
    target false-alarm ARL (expected rounds between false alarms).
    """

    def __init__(self, threshold: float, epsilon: float = 0.5) -> None:
        if threshold <= 1.0:
            raise ValueError("threshold must be > 1")
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        self.threshold = threshold
        self.epsilon = epsilon
        self.R = 0.0
        self._t = 0
        self._alarm_t: int | None = None
        self._peak = 0.0

    def update(self, p: float) -> float:
        e = conformal_e_value(p, self.epsilon)
        self.R = (1.0 + self.R) * e
        self._peak = max(self._peak, self.R)
        if self._alarm_t is None and self.R >= self.threshold:
            self._alarm_t = self._t
        self._t += 1
        return self.R

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def alarm_round(self) -> int | None:
        return self._alarm_t

    def alarm(self) -> bool:
        return self._alarm_t is not None


def score_ratio_e_value(
    test_score: float,
    cal_scores: list[float],
    cal_weights: list[float] | None = None,
    test_weight: float = 1.0,
) -> float:
    """Canonical conformal e-value (Balinsky & Balinsky 2024; arXiv 2503.13050
    Eq. 4) for NON-NEGATIVE nonconformity scores::

        E = S_test / mean_{cal u {test}} S

    ``E[E] = 1`` under exchangeability, so ``P(E >= 1/alpha) <= alpha`` (Markov).
    ``E ~ 1`` when the test score is typical -- an uninformative / trivially-wide
    certificate produces no evidence -- and ``E >> 1`` when the test score is
    anomalously large (a stressed-tight regime or an outright violation). This is
    the "collapses toward the null when uninformative" diagnostic. The weighted
    mean uses the age-weights (with the test point weighted ``test_weight``), so
    it inherits the non-exchangeable weighting the rest of the module uses.
    """
    if test_score < 0.0:
        raise ValueError("score_ratio_e_value needs non-negative scores")
    n = len(cal_scores)
    w = [1.0] * n if cal_weights is None else list(cal_weights)
    if len(w) != n:
        raise ValueError("cal_weights length must match cal_scores")
    wsum = sum(w) + test_weight
    if wsum <= 0.0:
        return 1.0
    wmean = (sum(wi * s for wi, s in zip(w, cal_scores)) + test_weight * test_score) / wsum
    if wmean <= 0.0:
        return 1.0  # all scores zero: no evidence
    return test_score / wmean


def residual_drift_score(recent_scores: list[float], cal_scores: list[float]) -> float:
    """DASC drift magnitude ``D_t`` (arXiv 2606.15953, Sec. 5): the 1-D
    Wasserstein-1 distance between the recent-window and calibration score
    distributions. Larger = more distribution drift.

    A MONITORING signal only -- it pairs with :class:`ConformalTestMartingale`
    for a drift dashboard. It is deliberately NOT wired into the certificate's
    coverage weights: DASC's coverage theorem is not distribution-free (it depends
    on unknown Lipschitz/mismatch constants), and gating the calibration weights
    on a label-dependent drift score would forfeit CERT-FLOW's hard
    LB <= OPT <= UB guarantee. Kept as an observable instead.
    """
    if not recent_scores or not cal_scores:
        return 0.0
    try:
        from scipy.stats import wasserstein_distance
    except Exception:  # scipy always present as a core dep, but stay defensive
        # Fallback: |mean difference| is a lower bound on W1.
        return abs(sum(recent_scores) / len(recent_scores)
                   - sum(cal_scores) / len(cal_scores))
    return float(wasserstein_distance(recent_scores, cal_scores))


def effective_sample_size(weights: list[float]) -> float:
    """Kish effective sample size ``n_eff = (sum w)^2 / sum w^2`` (DASC ``n_eff,t``
    with normalized weights reduces to ``1 / sum w_norm^2``). Small ``n_eff``
    warns that the weighted quantile rests on few effective samples -- a
    fragility signal for the drift dashboard."""
    s1 = sum(weights)
    s2 = sum(w * w for w in weights)
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


class ShrinkLicense:
    """Test-then-tighten license: an ANYTIME-VALID, a-posteriori license to
    report a tighter "Tier-2" radius ALONGSIDE (never instead of) the
    distribution-free Tier-1 certificate. It converts the WATCH observability
    layer into measured width reduction with an explicit, honest validity claim.

    For each candidate shrink factor ``k`` in ``grid`` it maintains an
    anytime-valid upper confidence sequence (CS) on the mean violation rate of
    the ``k``-shrunk per-edge interval, fed the stream of Bernoulli outcomes
    ``x_t(k) = 1{fresh score_t > k * radius_t}`` where ``radius_t`` is the full
    per-edge radius (``q + rho * age``) the planner would have priced that edge
    with at that moment. The CS is the hedged/betting capital process of
    Waudby-Smith & Ramdas (2023, "Estimating means of bounded random variables
    by betting") for bounded means: with a predictable plug-in betting fraction
    ``lambda_t`` (a function of past outcomes only, so the same value at every
    candidate mean ``m``), the lower-tail capital
    ``K_t^-(m) = prod_{i<=t} (1 + lambda_i (m - x_i))`` is, at the TRUE mean, a
    non-negative martingale with ``K_0 = 1``. Ville's inequality gives
    ``P(sup_t K_t^-(mu) >= 1/delta) <= delta``, so the upper edge of
    ``{m : K_t^-(m) < 1/delta}`` is a time-uniform upper bound on ``mu``
    (``K_t^-`` is monotone non-decreasing in ``m`` for bounded ``x``, so the edge
    is a single crossing point, tracked on a fixed ``m``-grid).

    THE CLAIM (do not soften): The licensed radius carries an anytime-valid,
    time-uniform upper bound on the OBSERVED deployment stream's violation rate
    (P(ucb ever under-covers the running mean) <= delta by Ville). Under drift
    this is an a-posteriori empirical claim about the deployment so far, together
    with an alarmable monitor -- it is NOT an a-priori distribution-free guarantee
    for the next round; that guarantee remains with the Tier-1 certificate. The
    license self-revokes: under a regime shift, violations inflate the CS and
    licensed_k returns to 1.0.
    """

    def __init__(
        self,
        grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        delta: float = 0.05,
        n_min: int = 50,
        m_grid: int = 200,
        c: float = 0.5,
    ) -> None:
        import numpy as _np

        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if n_min < 1:
            raise ValueError("n_min must be >= 1")
        if not 0.0 < c < 1.0:
            raise ValueError("c must be in (0, 1)")
        if m_grid < 2:
            raise ValueError("m_grid must be >= 2")
        self.grid = tuple(sorted(grid))
        self.delta = float(delta)
        self.n_min = int(n_min)
        self.c = float(c)  # betting-fraction cap: |lambda| <= c keeps every
        # capital factor 1 + lambda*(m - x) strictly positive for x in {0,1},
        # m in (0,1) (worst case 1 - c > 0).
        self._log_thresh = math.log(1.0 / delta)
        # Candidate mean values in (0, 1) at which each per-k capital process is
        # tracked; the UCB is the smallest m whose lower-tail capital has crossed
        # the Ville threshold (evidence the mean is below m).
        self._m = _np.linspace(1.0 / (m_grid + 1), m_grid / (m_grid + 1), m_grid)
        self._logK: dict[float, "_np.ndarray"] = {
            k: _np.zeros(m_grid) for k in self.grid
        }
        self._n = {k: 0 for k in self.grid}
        self._sum = {k: 0.0 for k in self.grid}    # running sum of outcomes
        self._sumsq = {k: 0.0 for k in self.grid}  # sum (x - muhat_prev)^2

    def update(self, violations) -> None:
        """Absorb one round's outcomes. ``violations`` is a dict ``{k: x_t(k)}``
        or a sequence aligned with ``grid``; each ``x_t(k)`` is the Bernoulli
        ``1{fresh score > k * radius}`` for that shrink factor."""
        import numpy as _np

        items = (
            violations.items() if hasattr(violations, "items")
            else zip(self.grid, violations)
        )
        for k, x in items:
            arr = self._logK.get(k)
            if arr is None:
                continue
            x = float(x)
            n_prev = self._n[k]
            # Predictable plug-in (WSR 2023, eq. 3.9-3.10): mean/variance from
            # PAST outcomes only, with a 1/2, 1/4 prior; lambda_t is thus a
            # predictable bet, so K_t^-(mu) stays a martingale at the true mean.
            muhat = (0.5 + self._sum[k]) / (1.0 + n_prev)
            sigma2 = (0.25 + self._sumsq[k]) / (1.0 + n_prev)
            t = n_prev + 1
            lam = math.sqrt(
                2.0 * self._log_thresh / (sigma2 * t * math.log(t + 1.0))
            )
            lam = min(lam, self.c)
            arr += _np.log1p(lam * (self._m - x))
            self._sum[k] += x
            self._sumsq[k] += (x - muhat) ** 2
            self._n[k] = t

    def ucb(self, k: float) -> float:
        """Current anytime-valid upper bound on the violation mean for factor
        ``k`` (the smallest m whose lower-tail capital has crossed 1/delta).
        Returns 1.0 while no candidate is rejected (nothing tighter licensed)."""
        import numpy as _np

        arr = self._logK.get(k)
        if arr is None or self._n[k] <= 0:
            return 1.0
        crossed = arr >= self._log_thresh
        if not bool(crossed.any()):
            return 1.0
        return float(self._m[int(_np.argmax(crossed))])

    def n(self, k: float) -> int:
        return self._n.get(k, 0)

    def licensed_k(self, alpha_target: float) -> float:
        """Smallest ``k`` with ``n >= n_min`` and ``ucb(k) <= alpha_target``,
        else 1.0 (no tightening licensed / license revoked)."""
        for k in self.grid:  # ascending: smallest (tightest) first
            if k >= 1.0:
                break
            if self._n[k] >= self.n_min and self.ucb(k) <= alpha_target:
                return k
        return 1.0
