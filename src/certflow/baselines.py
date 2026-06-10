"""Gaussian mu±beta*sigma calibration baseline for CERT.

This module provides the honest strawman for the paper's coverage claim: the
same planner loop, same widening, same D* Lite dual search — only the quantile
machinery is swapped.  The key contrast:

- ConformalScorer: distribution-free, uses the empirical weighted quantile.
  Coverage degrades gracefully on heavy-tailed or drifted noise.
- GaussianScorer: parametric, uses mu + z_{1-alpha} * sigma.  Claims full
  confidence (delta_stale = 0) regardless of buffer age.  Over-claims coverage
  on heavy-tailed distributions — that is the mechanism under test.

The GaussianCertPlanner disables ACI (gamma -> 0) because the purpose of this
baseline is to expose the raw Gaussian claim without the assumption-free ACI
safety net widening it back.  Leaving ACI active would partially mask the
over-claim and muddy the comparison.  This is documented via the ``use_aci``
flag attribute; the underlying trick is replacing the ACITracker with one that
has gamma=1e-12 so working_alpha stays effectively fixed at alpha_target.
"""
from __future__ import annotations

import math

from scipy.stats import norm

from certflow.cert import CertPlanner, PlannerConfig
from certflow.conformal import ACITracker, CalSample
from certflow.drift import grid_world
from certflow.harness import ExperimentConfig
from certflow.oracle import opt
from certflow.types import EpisodeResult, RoundLog


# ---------------------------------------------------------------------------
# GaussianScorer
# ---------------------------------------------------------------------------

class GaussianScorer:
    """Gaussian mu±z*sigma quantile using the same age-geometric weights as
    ConformalScorer.

    Interface mirrors ConformalScorer exactly:
      push(residual, t), quantile(alpha, t), delta_stale(t), ready(alpha, t),
      __len__.

    Key differences from ConformalScorer
    -------------------------------------
    - quantile: parametric Gaussian formula using weighted mean and weighted
      std, then z_{1-alpha} (scipy.stats.norm.ppf).  No distribution-free
      quantile computation.
    - delta_stale: always 0.0.  The Gaussian baseline silently assumes full
      confidence regardless of how stale the buffer is.  This over-claim is
      the exact mechanism the paper tests against.
    - ready: requires sum of weights >= 5 (effective sample count >= 5).
      Returns False until threshold is met; quantile returns +inf before then.

    Parameters
    ----------
    rho_w : float
        Age-geometric weight decay per unit time (same meaning as
        ConformalScorer.rho_w).
    max_buffer : int
        Rolling buffer capacity (same as ConformalScorer).
    """

    def __init__(
        self,
        rho_w: float = 0.99,
        max_buffer: int = 2000,
    ) -> None:
        if not 0.0 < rho_w <= 1.0:
            raise ValueError("rho_w must be in (0, 1]")
        self.rho_w = rho_w
        self.max_buffer = max_buffer
        self._buf: list[CalSample] = []

    def push(self, residual: float, t: float) -> None:
        self._buf.append(CalSample(residual, t))
        if len(self._buf) > self.max_buffer:
            self._buf.sort(key=lambda s: s.t)
            del self._buf[0 : len(self._buf) - self.max_buffer]

    def __len__(self) -> int:
        return len(self._buf)

    def _weights(self, t: float) -> list[float]:
        return [self.rho_w ** max(0.0, t - s.t) for s in self._buf]

    def _effective_n(self, t: float) -> float:
        return sum(self._weights(t))

    def quantile(self, alpha: float, t: float) -> float:
        """Weighted Gaussian (1-alpha)-quantile: mu_w + z_{1-alpha} * sigma_w.

        Returns +inf when ready() is False (fewer than 5 effective samples).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not self._buf:
            return math.inf
        w = self._weights(t)
        w_sum = sum(w)
        if w_sum < 5.0:
            return math.inf
        # Weighted mean
        residuals = [s.residual for s in self._buf]
        mu = sum(wi * r for wi, r in zip(w, residuals)) / w_sum
        # Weighted variance (reliability weights: divide by w_sum, no Bessel)
        var = sum(wi * (r - mu) ** 2 for wi, r in zip(w, residuals)) / w_sum
        sigma = math.sqrt(var) if var > 0.0 else 0.0
        z = float(norm.ppf(1.0 - alpha))
        return mu + z * sigma

    def delta_stale(self, t: float) -> float:
        """Always 0.0.

        The Gaussian baseline claims full confidence regardless of how old the
        calibration buffer is.  This is the over-claim the paper is designed to
        expose: there is no staleness correction, so the reported confidence
        never degrades with buffer age.
        """
        return 0.0

    def push_signed(self, deviation: float, t: float) -> None:
        """Interface parity with ConformalScorer; the Gaussian baseline does
        not implement the sum-aware UB (T4), so signed deviations are dropped."""

    def block_quantile(self, alpha: float, t: float, block_len: int) -> float:
        return float("inf")

    def block_delta_stale(self, t: float, block_len: int) -> float:
        return 1.0

    def effective_mass(self, t: float) -> float:
        """Interface parity: weighted effective sample size (for annealing)."""
        return sum(self._weights(t)) if hasattr(self, "_weights") else float(len(self))


    def ready(self, alpha: float, t: float) -> bool:
        """True when the effective sample weight sum >= 5."""
        return math.isfinite(self.quantile(alpha, t))


# ---------------------------------------------------------------------------
# GaussianCertPlanner
# ---------------------------------------------------------------------------

class GaussianCertPlanner(CertPlanner):
    """CertPlanner subclass with GaussianScorer and ACI disabled.

    Construction: calls super().__init__ then replaces self.scorer with a
    GaussianScorer (same rho_w) and replaces self.aci with an ACITracker
    whose gamma=1e-12 so working_alpha stays essentially pinned at alpha_target
    (the change in raw alpha per update is ~1e-12, negligible).

    The ``use_aci`` flag (False by default) documents intent: the Gaussian
    baseline intentionally does not use the assumption-free ACI safety net.
    Enabling ACI (use_aci=True) is valid but changes the comparison by letting
    the ACI net partially compensate for Gaussian over-claims.
    """

    def __init__(
        self,
        world,
        start,
        goal,
        config: PlannerConfig,
        t0: float = 0.0,
        use_aci: bool = False,
    ) -> None:
        super().__init__(world, start, goal, config, t0=t0)
        # Swap the conformal scorer for the Gaussian one (same rho_w).
        self.scorer = GaussianScorer(
            rho_w=config.rho_w,
            max_buffer=2000,
        )
        self.use_aci = use_aci
        if not use_aci:
            # Freeze working_alpha by using a vanishingly small gamma.
            # ACITracker still accepts updates (no structural changes to the
            # round() loop) but alpha_raw moves by at most ~1e-12 per step,
            # so working_alpha() is effectively constant at alpha_target.
            self.aci = ACITracker(
                alpha_target=config.alpha_prime,
                gamma=1e-12,
            )


# ---------------------------------------------------------------------------
# gaussian_tier0_episode
# ---------------------------------------------------------------------------

def gaussian_tier0_episode(config: ExperimentConfig, seed: int) -> EpisodeResult:
    """Tier-0 episode using GaussianCertPlanner.

    Mirror of episodes.tier0_episode; constructs a GaussianCertPlanner
    instead of CertPlanner.  Reuses planner_config and coverage_among_valid
    from certflow.episodes without modifying that module.
    """
    from certflow.episodes import planner_config  # refactor-free reuse

    import time

    world_kwargs: dict = {
        "noise_family": config.noise_family,
        "noise_scale": config.noise_scale,
    }
    if config.kind == "bounded":
        world_kwargs["rho"] = config.rho
    world = grid_world(config.rows, config.cols, seed=seed, kind=config.kind, **world_kwargs)

    start, goal = (0, 0), (config.rows - 1, config.cols - 1)
    planner = GaussianCertPlanner(
        world, start, goal, planner_config(config), use_aci=False
    )

    result = EpisodeResult()
    prev_spend = planner.sense_spend
    for _ in range(config.max_rounds):
        t_round = planner.t
        wall = time.perf_counter()
        cert, sensed = planner.round()
        wall = time.perf_counter() - wall

        _, true_opt = opt(world, t_round, start, goal)
        covered = bool(
            cert.valid and cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9
        )
        certified = bool(cert.valid and cert.gap <= config.epsilon)
        result.rounds.append(
            RoundLog(
                t=t_round,
                lb=cert.lb,
                ub=cert.ub,
                confidence=cert.confidence,
                opt=true_opt,
                covered=covered,
                certified=certified,
                sensed_edge=sensed,
                sense_spend=planner.sense_spend - prev_spend,
                replan_seconds=wall,
            )
        )
        prev_spend = planner.sense_spend

    result.sense_cost = planner.sense_spend
    result.reached_goal = False  # Tier 0 is stationary by design
    return result


def adstar_bound(
    beliefs,
    graph_struct,
    start,
    goal,
    w: float = 1.5,
    cost_floor: float = 1e-3,
):
    """AD*/ARA*-semantics suboptimality interval from current point estimates.

    Bounded-suboptimal search with inflation w returns a path P-hat whose
    cost on ITS map satisfies cost(P-hat) <= w * OPT_map, i.e. the standard
    claim OPT_map in [cost(P-hat)/w, cost(P-hat)]. That claim is sound on
    the searcher's own (stale, noisy point-estimate) map and has no
    mechanism for observation noise or staleness; we evaluate it as-is
    against the TRUE drifting optimum. The comparison targets the bound
    SEMANTICS under staleness, not the search algorithm (on these graph
    sizes exact search is instant, so anytime behavior is not the axis).
    """
    from certflow.graphcore import dijkstra

    g = {
        u: {v: max(beliefs[(u, v)].c_hat, cost_floor) for v in nbrs}
        for u, nbrs in graph_struct.items()
    }
    path, c = dijkstra(g, start, goal)
    if path is None:
        return 0.0, float("inf")
    return c / w, c
