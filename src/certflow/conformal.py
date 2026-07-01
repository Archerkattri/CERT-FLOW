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
