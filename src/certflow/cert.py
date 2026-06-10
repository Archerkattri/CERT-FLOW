"""CERT main loop: the 8-step replanning round of spec section 4.3.

Integrates graphcore (dual incremental searches), conformal (certificate
substrate), and sensing (route-critical observation selection). The planner
never sees true costs; it interacts with the world only through observe().
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from certflow.conformal import (
    ACITracker,
    AgeBinnedScorer,
    ConformalScorer,
    path_alpha_edge,
    path_confidence,
)
from certflow.fastgraph import FastDStarLite, FlatGraph
from certflow.sensing import baseline_select, near_optimal_alternatives, path_edges, select_observation
from certflow.types import Certificate, Edge, EdgeBelief, Node, World

# Finite upper-cost cap for unbounded edges. An UNOBSERVED edge (or warm-up
# u-cost at q=inf) has no coverage theorem pricing its upper bound, so it is
# conceptually +inf; D* Lite needs strictly finite, positive costs to maintain
# its g/rhs invariants, so we cap at this sentinel. It must dominate any real
# path cost (so capped edges are never chosen unless unavoidable) yet stay well
# below float overflow when summed over a path.
_UB_CAP = 1e9


@dataclass
class PlannerConfig:
    epsilon: float = 5.0          # target certificate gap
    alpha_prime: float = 0.1      # path-level miscoverage target
    rho_w: float = 0.99           # conformal weight decay per unit time
    eps_tv: float = 0.0           # A2 TV-Lipschitz rate (0 = exchangeable claim)
    gamma_aci: float = 0.005      # ACI step size
    delta: float = 1.0            # sensing period (time units per round)
    rho_hat_over_rho: float = 1.0 # drift misspecification factor (A1 sweep)
    sense_cost: float = 0.1       # uniform m_e for v1
    k_alternatives: int = 3
    delta_subopt: float = 0.1
    backstop_slack: float = 1.5   # backstop_age = slack * L * delta
    cost_floor: float = 1e-3
    # Maintenance sensing while certified (T2': certification is sustained by
    # a sensing rate, not achieved once). lookahead: sense when the projected
    # gap crosses epsilon within this many rounds. every: calibration-freshness
    # floor, sense at least once per this many certified rounds.
    maintenance_lookahead: float = 2.0
    maintenance_every: int = 5
    # kappa corridor hysteresis (Design 1): among incumbent candidates whose
    # u-cost is within kappa_slack_frac*epsilon of the tightest UB, execute the
    # one with the highest mean edge-conductivity instead of the raw argmin.
    # UB itself is always the tightest bound, so the certificate is untouched;
    # execution suboptimality stays bounded by gap + slack. kappa is reinforced
    # on the executed incumbent and decays geometrically each round.
    use_kappa: bool = False
    kappa_decay: float = 0.95
    kappa_slack_frac: float = 0.5

    # Sensing policy: "cert" (gap-shrink VOI + backstop, the contribution),
    # or Tier-2 baselines: "random", "max_age" (global freshness round-robin),
    # "max_width" (global info-gain proxy, not route-critical), "none".
    sensing_policy: str = "cert"

    # Unknown-terrain start (Tier-2): skip the t0 survey; every edge begins
    # at a weak prior with a large age, so intervals start wide and sensing
    # allocation actually matters.
    initial_survey: bool = True
    prior_cost: float = 1.0
    prior_age: float = 200.0

    # Margin factor lambda (paper/theory.tex): 1.0 = observable-coverage
    # semantics (T1a, the empirically-validated default); 2.0 = provable
    # latent-cost coverage (T1b). Scales the conformal quantile everywhere.
    latent_margin: float = 1.0

    # Thinned calibration (theory.tex, honest accounting item 1): consecutive
    # scores on the same edge share a noise draw (one-dependent). Thinning
    # keeps only scores from disjoint observation pairs (2nd, 4th, ... obs of
    # each edge), restoring independence at a factor-2 sample cost. Part of
    # the provable mode together with latent_margin=2.
    thinned_scores: bool = False

    # ACI adapts the working alpha from realized edge-coverage events. It
    # CANCELS static margins (with lambda=2, errs vanish, alpha climbs, q
    # shrinks until errs return to target) — so the provable T1b mode must
    # freeze it: use_aci=False pins the working level at alpha_prime and the
    # quantile is the raw weighted-conformal quantile the theorem assumes.
    use_aci: bool = True

    # Sum-aware upper certificate (theory.tex T4): replace the incumbent's
    # Bonferroni UB value (sum of per-edge u_e, margin ~ L*q_{a'/L}) with
    # sum(c_hat) + block-quantile margin at level alpha' (~ sqrt(L)*q) +
    # sum(rho*a). UB side only — the LB must hold uniformly over paths and
    # keeps its per-edge construction. Tightens the gap and the T2' floor.
    sum_aware_ub: bool = False

    # Alpha annealing: report the best currently-supportable claim instead of
    # INVALID during warm-up. The effective sample size m floors the per-edge
    # level at 1/(m+1); the path level anneals from coarse to the target as
    # evidence accrues. Claims always state the annealed (weaker) level, so
    # nothing is overclaimed; certification additionally requires the claim
    # to have reached min_certify_confidence (never stop sensing on a weak
    # claim).
    anneal_alpha: bool = True
    min_certify_confidence: float = 0.5

    # Adaptive sensing rate (T2'): sense k <= max_sense_per_round edges per
    # round, with k chosen so the sustained gap floor 2*L*q + rho*Delta*
    # L*(L-1)/k meets epsilon when possible. Also focuses sensing on P_lb
    # (gap decomposition) and adapts the pre-widening horizon B so the cache
    # spends at most prewiden_slack_frac of the epsilon-slack on width
    # (at high drift B drops to 0: exact metrics, slower, certifiable).
    # Off by default (changes spend and latency semantics).
    adaptive_rate: bool = False
    max_sense_per_round: int = 4
    prewiden_slack_frac: float = 0.25

    # Objective-matched sensing: when T2' says epsilon is unattainable at the
    # current rate, certificate-gap sensing buys nothing — spend observations
    # on the EXPECTED-best route instead (VOI), which is what determines
    # departure quality; switch back to gap-directed sensing when epsilon is
    # attainable. Measured: matches the VOI baseline's regret (5x better than
    # pure gap sensing in unattainable regimes) while keeping certificate
    # behavior where certification is possible.
    hybrid_sensing: bool = False

    # Online drift-rate estimation: rho_mode="online" frees the planner from
    # a world-supplied rho. Pooled rate samples |obs - c_hat_prev| / age from
    # re-observations (noise inflates them -> conservative); rho_hat is their
    # rho_online_quantile. Until the estimator warms, rho ~ 0 and the
    # conformal layer absorbs unmodeled drift into the scores (validated on
    # real traffic at up to 49% A1-violation rates: width cost, not coverage).
    rho_mode: str = "given"          # "given" | "online"

    # Stabilized sensing target (the P_lb-churn factor): under drift the
    # optimistic path is a moving target and focused sensing chases it,
    # leaving realized gaps ~2x above the T2' floor. Keep sensing the SAME
    # path while its ell-cost stays within (1+tol) of LB; the gap bound pays
    # at most tol*LB extra (u(P_s) - ell(P_s) + [ell(P_s) - LB]) and the
    # ages on the stable target obey the round-robin analysis.
    stabilize_sensing: bool = False
    sense_path_tol: float = 0.1

    # Churn-aware certification (T7): the optimistic path hops over a CHURN
    # SET of K >= L edges under drift; the T2' floor and the sensing target
    # must use K, not the instantaneous path length, or realized gaps run
    # ~K/L above the floor (the measured ~1.6x residual). K is tracked over
    # a sliding window and reported; adaptive k solves the K-floor.
    churn_window: int = 50

    # Refine-after-certify: certification stops gap-sensing, but the
    # certified incumbent can be far from optimal WITHIN epsilon (measured
    # in lifelong runs: memory-carried incumbents certify at regret 0.4-0.6
    # vs 0.025 for fresh exploration). When on, certified rounds keep
    # sensing the EXPECTED-best route (VOI) to improve the incumbent; the
    # certificate is untouched (sensing only ever tightens it).
    refine_after_certify: bool = False

    # Strict LB level (theory GAP-A): the lower bound must cover the UNKNOWN
    # optimum's edges, whose count can exceed |P_lb|; the airtight per-edge
    # level divides by (|V|-1), not L. Off by default (the deployed planner
    # operates at alpha'/L, validated against ground truth at 1.000 across
    # all conditions — the conservatism slack absorbs the difference); ON in
    # the provable recipe, where every constant must be theorem-exact.
    strict_lb_alpha: bool = False

    # Decision-uniform certificates: per-round claims are marginal; a robot
    # that ACTS whenever certified peeks every round, and across T rounds the
    # chance that SOME acted-on certificate failed exceeds alpha'. Full
    # per-round time-uniformity is quantifiably impractical (stitched-DKW
    # needs n >~ 63k scores at Bonferroni levels — theory.tex T6), but the
    # certificate is only USED at decision instants (stop sensing, depart):
    # alpha-spending over a decision budget gives simultaneous validity of
    # ALL decisions at level alpha' for the width cost of alpha'/N_dec.
    decision_uniform: bool = False
    max_decisions: int = 5

    # Predictor mode (spatial-predictor study): when a point predictor is
    # supplied to the planner, edges older than predictor_age_gate*delta use
    # (predicted center, LEARNED age-binned conformal width) instead of
    # (last obs, q + rho*age). Per-edge fallback chain: prediction available
    # AND its age-bin quantile supportable, else the model-based path —
    # separate calibration buffers per model, so scores never mix regimes
    # (assumption A4': within-bin exchangeability). Bin edges in delta units.
    predictor_age_gate: float = 12.0
    predictor_bins: tuple = (6.0, 12.0, 24.0, 48.0)
    rho_online_quantile: float = 0.9
    rho_online_min_samples: int = 10

    # Lazy pre-widening (T3 locality): cache edge metrics at age + B*delta so
    # they stay valid (conservatively wide) for B rounds and D* Lite repair
    # touches ~|E|/B edges per round instead of all of them. Soundness:
    # cached ell <= true ell and cached u >= true u throughout the window.
    # Width cost: +2*rho*B*delta per edge. 0 disables (exact, slow).
    prewiden_rounds: int = 10

    # Staggered pre-widening (predictor-free vectorized path): per-edge horizon
    # factors are drawn uniformly from [stagger_lo, stagger_hi] to spread cache
    # expiries across rounds instead of all firing on one synchronized round.
    # Soundness is independent of the spread — each entry's width is computed at
    # ITS OWN horizon — so this is purely a latency-smoothing knob.
    stagger_lo: float = 0.75
    stagger_hi: float = 1.25


def recommended_config(**overrides) -> "PlannerConfig":
    """The best-known configuration from the full ablation/benchmark program:
    online drift estimation (coverage-neutral, 1.7-2.4x tighter gaps),
    objective-matched hybrid sensing (regret matches/beats the strongest
    baseline while keeping the certificate), kappa hysteresis (-70% churn),
    adaptive rate + adaptive pre-widening, gated sum-aware UB. Annealing is
    already the default. decision_uniform stays a claim-semantics choice."""
    base = dict(
        rho_mode="online",
        hybrid_sensing=True,
        use_kappa=True,
        adaptive_rate=True,
        sum_aware_ub=True,
    )
    base.update(overrides)
    return PlannerConfig(**base)


class CertPlanner:
    """Holds beliefs, the two D* Lite instances, and the certificate state."""

    def __init__(
        self,
        world: World,
        start: Node,
        goal: Node,
        config: PlannerConfig,
        t0: float = 0.0,
        predictor=None,
    ) -> None:
        self.cfg = config
        self.world = world
        if not config.anneal_alpha and config.rho_w < 1.0:
            ess_cap = 1.0 / (1.0 - config.rho_w)
            if ess_cap < 1.0 / config.alpha_prime:
                import warnings
                warnings.warn(
                    f"rho_w={config.rho_w} caps effective sample size at "
                    f"~{ess_cap:.0f} < 1/alpha_prime={1/config.alpha_prime:.0f}: "
                    "without annealing the certificate may never become valid",
                    stacklevel=2,
                )
        self.start = start
        self.goal = goal
        self.t = t0

        # Initial survey: one observation per edge at t0 (spec: warm-up phase;
        # the certificate stays INVALID until the calibration buffer fills).
        # With initial_survey=False (unknown terrain), edges start at a weak
        # prior with a large age instead.
        self.beliefs: dict[Edge, EdgeBelief] = {}
        for e in world.edges():
            if config.initial_survey:
                c0, t_obs0, seen = max(world.observe(e, t0), config.cost_floor), t0, True
            else:
                c0, t_obs0, seen = config.prior_cost, t0 - config.prior_age, False
            self.beliefs[e] = EdgeBelief(
                c_hat=c0,
                t_obs=t_obs0,
                rho=self._rho_hat(e),
                sense_cost=config.sense_cost,
                observed=seen,
            )

        self.scorer = ConformalScorer(rho_w=config.rho_w, eps_tv=config.eps_tv)
        self.predictor = predictor
        self.binned = AgeBinnedScorer(
            bin_edges=tuple(b * config.delta for b in config.predictor_bins),
            rho_w=config.rho_w, eps_tv=config.eps_tv,
        )
        self.pred_used_rounds = 0  # diagnostic: edges priced by the predictor
        self._edge_alpha_extra: dict[Edge, float] = {}  # per-bin annealing charge
        self.aci = ACITracker(alpha_target=config.alpha_prime, gamma=config.gamma_aci)
        self.sense_spend = 0.0
        self._round_idx = 0
        self._obs_count: dict[Edge, int] = {}  # real observations per edge
        self._rate_samples: list[float] = []    # online rho: |dc|/age samples
        self._rho_online = 1e-9
        self._last_gap = math.inf               # gap-stall feedback for k
        self._stall = 0
        self._churn_seen: dict[Edge, int] = {}  # edge -> last round on P_lb
        self.cal_rho_a_max = 0.0  # max rho_e*a_e among pushed scores (pi_cal diagnostic)

        # lazy pre-widening cache (see PlannerConfig.prewiden_rounds)
        self._cache_lo: dict[Edge, float] = {}
        self._cache_up: dict[Edge, float] = {}
        self._cache_due: dict[Edge, float] = {}   # absolute expiry time
        self._cache_q: float = -1.0               # q the cache was built with

        # certified snapshot oracle (snapshot.py): built on point estimates
        # when the certificate proves the map tight; O(1) queries thereafter
        self._oracle = None
        self._oracle_chat_snap = None
        self._flat_mid = None
        self._beliefs_version = 0

        # kappa corridor hysteresis state (see PlannerConfig.use_kappa)
        self._p_sense: list[Node] = []  # stabilized sensing target path
        self._kappa: dict[Edge, float] = {}
        self._prev_incumbent: list[Node] = []
        self._incumbent_since = t0  # when the incumbent edge-set last changed
        self._rng = random.Random(0)  # baseline sensing policies only

        nodes = set(world.graph) | {v for n in world.graph for v in world.graph[n]}
        lo, up = self._metrics(q=math.inf)  # warm-up: q=inf -> ell at floor, u at inf
        # D* Lite needs finite costs; cap warm-up upper costs.
        up = {e: min(c, _UB_CAP) for e, c in up.items()}
        adj_lo = self._to_adj(nodes, lo)
        adj_up = self._to_adj(nodes, up)
        self._flat_lo = FlatGraph(adj_lo, extra_nodes=(start, goal))
        self._flat_up = FlatGraph(adj_up, extra_nodes=(start, goal))
        self.sp_lower = FastDStarLite(adj_lo, start, goal, flat=self._flat_lo)
        self.sp_upper = FastDStarLite(adj_up, start, goal, flat=self._flat_up)
        self._graph_lower_cache = adj_lo
        # fixed edge order + CSR slots for vectorized cache->flat cost sync
        # (the shared-flat constructor does NOT read costs from the adjacency,
        # so scratch rebuilds must write the cache into the arrays themselves)
        import numpy as _np
        self._edge_order = list(self.beliefs)
        ix_lo, ix_up = self._flat_lo.index_of, self._flat_up.index_of
        self._slots_lo = _np.array(
            [self._flat_lo.slot_of(ix_lo[u], ix_lo[v]) for u, v in self._edge_order],
            dtype=_np.int64)
        self._slots_up = _np.array(
            [self._flat_up.slot_of(ix_up[u], ix_up[v]) for u, v in self._edge_order],
            dtype=_np.int64)
        # belief arrays in edge order (vectorized full-refresh: fast_metrics)
        self._edge_idx = {e: i for i, e in enumerate(self._edge_order)}
        self._arr_chat = _np.array(
            [self.beliefs[e].c_hat for e in self._edge_order])
        self._arr_tobs = _np.array(
            [self.beliefs[e].t_obs for e in self._edge_order])
        self._arr_rho = _np.array(
            [self.beliefs[e].rho for e in self._edge_order])
        self._arr_obs = _np.array(
            [self.beliefs[e].observed for e in self._edge_order], dtype=bool)
        # staggered pre-widening horizons (audit seam 3): synchronized expiry
        # made one round per cycle pay a full-|E| refresh loop; per-edge
        # horizon factors in [0.75, 1.25] spread expiries across rounds.
        # Soundness: each entry's width is computed at ITS OWN horizon.
        self._arr_stagger = _np.random.default_rng(0).uniform(
            config.stagger_lo, config.stagger_hi, len(self._edge_order))
        # _arr_due is the expiry clock for the VECTORIZED (predictor-free) refresh
        # path ONLY; it is the array twin of _cache_due there. When a predictor is
        # supplied the dict path drives expiry off _cache_due alone and never
        # reads _arr_due, so the two intentionally do not track each other in that
        # mode (the array view is simply unused). ingest_observation expires both.
        self._arr_due = _np.full(len(self._edge_order), -_np.inf)

    def _adaptive_B(self, q_eff: float) -> int:
        """Pre-widening horizon: cap width spend at a fraction of the
        epsilon-slack when certification is in play; keep the configured
        latency-optimal horizon during warm-up or unattainable epsilon."""
        cfg = self.cfg
        B = cfg.prewiden_rounds
        if not cfg.adaptive_rate or B <= 0:
            return max(B, 0)
        L_b = max(getattr(self, "_last_L", 1), 1)
        rho_b = max((self.beliefs[e].rho for e in self.beliefs), default=0.0)
        slack = cfg.epsilon - 2 * L_b * q_eff
        if rho_b > 0 and slack > 0 and q_eff > 0:
            b_cap = int(cfg.prewiden_slack_frac * slack
                        / (2 * rho_b * L_b * cfg.delta))
            B = max(0, min(B, b_cap))
        return B

    def _rebuild_searches(self) -> None:
        """Fresh D* Lite instances from the current cached metrics (used when
        a change touches most edges; incremental repair of ~|E| inconsistent
        vertices is strictly slower than one scratch compute)."""
        import numpy as _np
        # sync the cache into the flat cost arrays FIRST: the shared-flat
        # constructor keeps existing costs, and the scratch-rebuild path
        # skips update_edges — without this write the engines resurrect
        # stale costs (a divergence the degenerate ablation caught)
        self._flat_lo.cost[self._slots_lo] = _np.fromiter(
            (self._cache_lo[e] for e in self._edge_order),
            dtype=_np.float64, count=len(self._edge_order))
        self._flat_up.cost[self._slots_up] = _np.fromiter(
            (min(self._cache_up[e], _UB_CAP) for e in self._edge_order),
            dtype=_np.float64, count=len(self._edge_order))
        # reuse the FlatGraphs: CSR stays built, numba kernel stays warm;
        # structure-only adjacency suffices (engines read flat.cost)
        self.sp_lower = FastDStarLite(self._graph_lower_cache, self.start,
                                      self.goal, flat=self._flat_lo)
        self.sp_upper = FastDStarLite(self._graph_lower_cache, self.start,
                                      self.goal, flat=self._flat_up)

    def _rho_hat(self, e: Edge) -> float:
        if self.cfg.rho_mode == "online":
            return 1e-9  # estimator warms from observed rates (see round())
        rho_true = self.world.rho_true(e)
        if not math.isfinite(rho_true):
            rho_true = 0.0  # off-model worlds: planner assumes its A1 model anyway
        return max(rho_true * self.cfg.rho_hat_over_rho, 1e-9)

    def _update_online_rho(self) -> None:
        """Pooled drift-rate estimate from re-observation rate samples; on a
        material change, update all beliefs and force a metric rebuild."""
        cfg = self.cfg
        if cfg.rho_mode != "online" or len(self._rate_samples) < cfg.rho_online_min_samples:
            return
        if len(self._rate_samples) < 1.1 * getattr(self, "_rho_sorted_at", 0):
            return  # re-estimate only when the sample set grew 10%
        self._rho_sorted_at = len(self._rate_samples)
        rates = sorted(self._rate_samples)
        rho = max(rates[int(cfg.rho_online_quantile * (len(rates) - 1))], 1e-9)
        if abs(rho - self._rho_online) > 0.05 * max(self._rho_online, 1e-9):
            self._rho_online = rho
            for b in self.beliefs.values():
                b.rho = rho
            self._arr_rho[:] = rho
            self._cache_q = -1.0  # rho changed everywhere: full metric rebuild

    def _to_adj(self, nodes, costs: dict[Edge, float]) -> dict[Node, dict[Node, float]]:
        adj: dict[Node, dict[Node, float]] = {n: {} for n in nodes}
        for (u, v), c in costs.items():
            adj[u][v] = c
        return adj

    def _pred_interval(self, e: Edge, age: float) -> tuple[float, float] | None:
        """(center, halfwidth) from the predictor path, or None to fall back.
        Requires: predictor supplied, age past the gate, a prediction for e,
        and a supportable age-bin quantile at the current per-edge level."""
        cfg = self.cfg
        if self.predictor is None or age < cfg.predictor_age_gate * cfg.delta:
            return None
        pred = self.predictor(e, self.t, self.beliefs)
        if pred is None:
            return None
        alpha_edge = getattr(self, "_last_alpha_edge", self._alpha_prime_eff)
        # per-bin annealing: query at the bin's supportable level and charge
        # the weakening to the claim (weakest-link accounting, summed over
        # the certifying path's predictor-priced edges in round())
        mass = self.binned.effective_mass(self.t, age)
        if mass <= 0.0:
            return None
        alpha_bin = max(alpha_edge, (1.0 + 1e-9) / (mass + 1.0))
        if alpha_bin >= 0.5:
            return None  # bin too immature to be worth a claim
        qb = self.binned.quantile(alpha_bin, self.t, age)
        if not math.isfinite(qb):
            return None
        self._edge_alpha_extra[e] = max(0.0, alpha_bin - alpha_edge)
        return max(pred, cfg.cost_floor), cfg.latent_margin * qb

    def _metrics(self, q: float) -> tuple[dict[Edge, float], dict[Edge, float]]:
        lo, up = {}, {}
        for e, b in self.beliefs.items():
            if not b.observed:
                # an unobserved edge is UNKNOWN: the prior is not an
                # observation and no coverage theorem prices it — ell at the
                # floor (it could be cheap), u unbounded (it could be awful).
                # Certification therefore requires a fully-OBSERVED path:
                # exactly the Traversing-Mars 'prove the path' semantics,
                # which T2's degenerate corollary claims (and a noise-free
                # test exposed: the prior-centered interval was a soundness
                # hole masked by noise everywhere else).
                lo[e] = self.cfg.cost_floor
                up[e] = _UB_CAP
            elif math.isfinite(q):
                pi = self._pred_interval(e, b.age(self.t))
                if pi is not None:
                    c, h = pi
                    lo[e] = max(self.cfg.cost_floor, c - h)
                    up[e] = max(self.cfg.cost_floor, c + h)
                else:
                    lo[e] = b.lower(self.t, q, self.cfg.cost_floor)
                    up[e] = b.upper(self.t, q, self.cfg.cost_floor)
            else:
                lo[e] = self.cfg.cost_floor
                up[e] = _UB_CAP
        return lo, up

    def _refresh_metrics(self, q_eff: float) -> None:
        """Maintain the pre-widened metric cache; push only changed edges to
        the two searches. Soundness: entries are computed at age + B*delta,
        so cached ell <= true ell and cached u >= true u until expiry; a grown
        quantile forces a full rebuild (a cached-too-small q would be unsound,
        a cached-too-large q is only conservative)."""
        cfg = self.cfg
        B = cfg.prewiden_rounds
        if self.predictor is None:
            # vectorized full-refresh fast path (fast_metrics): exact mode
            # recomputes everything every round, and full rebuilds touch all
            # edges — both were a Python per-edge loop (~15ms at 14k edges)
            import numpy as _np
            from certflow.fastgraph import fast_metrics
            full_now = (
                B <= 0
                or not self._cache_lo
                or q_eff > self._cache_q + 1e-12
                or self._cache_q > 1.30 * q_eff + 1e-12
            )
            if full_now:
                horizon = 0.0 if B <= 0 else self._adaptive_B(q_eff) * cfg.delta
                q_used = q_eff if B <= 0 else 1.15 * q_eff
                lo_a, up_a = fast_metrics(
                    self._arr_chat, self._arr_tobs, self._arr_rho,
                    self.t, q_used, cfg.cost_floor)
                if horizon > 0.0:
                    # per-edge staggered horizons: widen each entry to cover
                    # its OWN expiry time (linear in age, so additive here)
                    h = horizon * self._arr_stagger
                    widen = self._arr_rho * h
                    lo_a = _np.maximum(lo_a - widen, cfg.cost_floor)
                    up_a = up_a + widen
                    dues = self.t + h
                else:
                    dues = _np.full(len(self._edge_order), self.t)
                unobs = ~self._arr_obs
                lo_a[unobs] = cfg.cost_floor
                up_a[unobs] = _UB_CAP
                _np.minimum(up_a, _UB_CAP, out=up_a)
                self._cache_lo = dict(zip(self._edge_order, lo_a.tolist()))
                self._cache_up = dict(zip(self._edge_order, up_a.tolist()))
                self._cache_due = dict(zip(self._edge_order, dues.tolist()))
                self._arr_due = dues
                if B > 0:
                    self._cache_q = q_used
                self._flat_lo.cost[self._slots_lo] = lo_a
                self._flat_up.cost[self._slots_up] = up_a
                self.sp_lower = FastDStarLite(
                    self._graph_lower_cache, self.start, self.goal,
                    flat=self._flat_lo)
                self.sp_upper = FastDStarLite(
                    self._graph_lower_cache, self.start, self.goal,
                    flat=self._flat_up)
                # adjacency VALUES are consumed only by the alternatives
                # helper, which refreshes them on demand (_graph_lower_with);
                # the engines read costs from the flat arrays — skip the
                # O(|E|) dict-of-dicts rebuild here
                return
            # vectorized staggered due-subset (small by construction)
            mask = self._arr_due <= self.t
            if mask.any():
                idx = _np.nonzero(mask)[0]
                B_eff = self._adaptive_B(q_eff)
                h = B_eff * cfg.delta * self._arr_stagger[idx]
                widen = self._arr_rho[idx] * (
                    (self.t - self._arr_tobs[idx]) + h)
                q_used = self._cache_q
                lo_sub = _np.maximum(
                    self._arr_chat[idx] - q_used - widen, cfg.cost_floor)
                up_sub = _np.minimum(
                    self._arr_chat[idx] + q_used + widen, _UB_CAP)
                unobs = ~self._arr_obs[idx]
                lo_sub[unobs] = cfg.cost_floor
                up_sub[unobs] = _UB_CAP
                self._arr_due[idx] = self.t + h
                lo_chg, up_chg = {}, {}
                for j, li, ui in zip(idx.tolist(), lo_sub.tolist(),
                                     up_sub.tolist()):
                    e = self._edge_order[j]
                    self._cache_due[e] = self._arr_due[j]
                    if li != self._cache_lo.get(e):
                        lo_chg[e] = self._cache_lo[e] = li
                    if ui != self._cache_up.get(e):
                        up_chg[e] = self._cache_up[e] = ui
                if len(lo_chg) > 0.3 * len(self.beliefs):
                    self._rebuild_searches()
                elif lo_chg or up_chg:
                    if lo_chg:
                        self.sp_lower.update_edges(lo_chg)
                    if up_chg:
                        self.sp_upper.update_edges(up_chg)
            return
        if cfg.adaptive_rate and B > 0:
            B = self._adaptive_B(q_eff)
        if B <= 0:
            lo, up = self._metrics(q_eff)
            self._cache_lo, self._cache_up = lo, up
            self._rebuild_searches()  # all edges changed: scratch beats repair
            return

        full = (
            not self._cache_lo
            or q_eff > self._cache_q + 1e-12          # unsound to keep: rebuild
            or self._cache_q > 1.30 * q_eff + 1e-12   # too loose: rebuild
        )
        # headroom 1.15: a growing quantile forces a full rebuild, and on
        # large graphs each rebuild is an O(|E|) + scratch-search event (the
        # p95 spikes); more headroom = fewer events at ~15% width on the
        # noise term only (the drift term dominates under drift anyway)
        q_used = 1.15 * q_eff if full else self._cache_q
        horizon = B * cfg.delta
        lo_chg: dict[Edge, float] = {}
        up_chg: dict[Edge, float] = {}
        for e, b in self.beliefs.items():
            if not (full or self.t >= self._cache_due.get(e, -math.inf)):
                continue
            if not b.observed:
                # unknown edge (see _metrics): floor / unbounded until seen
                lo_v, up_v = cfg.cost_floor, _UB_CAP
                if lo_v != self._cache_lo.get(e):
                    lo_chg[e] = self._cache_lo[e] = lo_v
                if up_v != self._cache_up.get(e):
                    up_chg[e] = self._cache_up[e] = up_v
                self._cache_due[e] = self.t + horizon
                continue
            a_pre = b.age(self.t) + horizon
            pi = self._pred_interval(e, a_pre)  # pre-widened age: conservative
            if pi is not None:
                c_pi, h_pi = pi
                self.pred_used_rounds += 1
                lo_v = max(cfg.cost_floor, c_pi - h_pi)
                up_v = max(cfg.cost_floor, c_pi + h_pi)
            else:
                self._edge_alpha_extra.pop(e, None)
                lo_v = max(cfg.cost_floor, b.c_hat - q_used - b.rho * a_pre)
                up_v = max(cfg.cost_floor, b.c_hat + q_used + b.rho * a_pre)
            if lo_v != self._cache_lo.get(e):
                lo_chg[e] = self._cache_lo[e] = lo_v
            if up_v != self._cache_up.get(e):
                up_chg[e] = self._cache_up[e] = up_v
            self._cache_due[e] = self.t + horizon
        if full:
            self._cache_q = q_used
        # When most of the graph changed (full rebuilds, B=0 mode), repairing
        # ~|E| inconsistencies through the priority queue costs far more than
        # one fresh compute — rebuild the search instances from scratch
        # instead (measured: p95 spikes 33-96 ms -> scratch cost ~1-5 ms).
        if len(lo_chg) > 0.3 * len(self.beliefs):
            self._rebuild_searches()
            return
        if lo_chg:
            self.sp_lower.update_edges(lo_chg)
        if up_chg:
            self.sp_upper.update_edges({e: min(c, _UB_CAP) for e, c in up_chg.items()})

    @property
    def _alpha_prime_eff(self) -> float:
        """Claim level: alpha'/N_dec under decision-uniform alpha-spending."""
        if self.cfg.decision_uniform:
            return self.cfg.alpha_prime / max(self.cfg.max_decisions, 1)
        return self.cfg.alpha_prime

    def _q(self, path_len: int) -> float:
        alpha_path = (
            self.aci.working_alpha() if self.cfg.use_aci else self._alpha_prime_eff
        )
        path_len = max(path_len, 1)
        if self.cfg.strict_lb_alpha:
            # GAP-A: cover the unknown optimum's edges too — divide by the
            # max possible simple-path length, not the current path's
            path_len = max(path_len, len(self._graph_lower_cache) - 1)
        # annealing floor: the smallest per-edge level the buffer supports
        self._alpha_claim = self._alpha_prime_eff
        if self.cfg.anneal_alpha:
            m = self.scorer.effective_mass(self.t)
            if m <= 0.0:
                self._alpha_claim = 1.0  # empty buffer: nothing supportable
            else:
                alpha_edge_min = (1.0 + 1e-9) / (m + 1.0)
                supportable = min(1.0, path_len * alpha_edge_min)
                alpha_path = max(alpha_path, supportable)
                self._alpha_claim = max(self._alpha_prime_eff, supportable)
        alpha_edge = path_alpha_edge(alpha_path, path_len)
        self._last_alpha_edge = alpha_edge
        return self.scorer.quantile(alpha_edge, self.t)

    def round(self) -> tuple[Certificate, Edge | None]:
        """One replanning round. Returns the certificate and the sensed edge."""
        cfg = self.cfg

        # Step 1-2: iterate q <-> path-length coupling once (L feeds Bonferroni).
        # Start from last known L or a Dijkstra-free guess of 1.
        self._update_online_rho()
        L_guess = getattr(self, "_last_L", 1)
        q = self._q(L_guess)
        q_eff = (q if math.isfinite(q) else 0.0) * cfg.latent_margin
        # warm-up: intervals exist but the certificate is INVALID via confidence
        self._refresh_metrics(q_eff)

        sum_aware_L = 0
        p_lb, lb = self.sp_lower.shortest_path()
        lb_edges = path_edges(p_lb)
        L = max(len(lb_edges), 1)
        if L != L_guess:  # one refinement pass with the right Bonferroni level
            q = self._q(L)
            q_eff = (q if math.isfinite(q) else 0.0) * cfg.latent_margin
            self._refresh_metrics(q_eff)
            p_lb, lb = self.sp_lower.shortest_path()
            lb_edges = path_edges(p_lb)
            L = max(len(lb_edges), 1)
        self._last_L = L
        lo, up = self._cache_lo, self._cache_up

        # stabilized sensing target (see PlannerConfig.stabilize_sensing)
        sense_path = p_lb
        if cfg.stabilize_sensing and p_lb is not None:
            ps = self._p_sense
            if (
                ps
                and ps[0] == self.start
                and ps[-1] == self.goal
                and sum(lo[e] for e in path_edges(ps))
                <= (1.0 + cfg.sense_path_tol) * lb
            ):
                sense_path = ps
            self._p_sense = list(sense_path)
        sense_edges = path_edges(sense_path) if sense_path else lb_edges

        # Step 3: UB = min over (u-cost of optimistic path, u-cost of
        # conservative shortest path); any path's u-cost upper-bounds OPT.
        p_ub, _ = self.sp_upper.shortest_path()
        ub_edges = path_edges(p_ub) if p_ub is not None else []
        ub_candidates = []
        if p_lb is not None:
            ub_candidates.append((sum(up[e] for e in lb_edges), p_lb))
        if p_ub is not None:
            ub_candidates.append((sum(up[e] for e in ub_edges), p_ub))
        prev = self._trimmed_prev_incumbent()
        if prev is not None:
            ub_candidates.append(
                (sum(up[e] for e in path_edges(prev)), prev)
            )
        if cfg.stabilize_sensing and sense_path is not None and sense_path is not p_lb:
            # the stabilized sensing target's edges are the fresh ones; its
            # u-cost completes the gap bound u(P_s) - LB <= width(P_s) + tol*LB
            ub_candidates.append(
                (sum(up[e] for e in sense_edges), sense_path)
            )
        if not ub_candidates:
            ub, incumbent = math.inf, []
        else:
            if cfg.sum_aware_ub and math.isfinite(q) and prev is not None:
                # T4: tighter UB on the standing incumbent ONLY, gated on
                # freshness — every edge re-observed since this path became
                # the incumbent. Post-selection observations are independent
                # of the selection event, so the fixed-path theorem applies
                # conditionally; without the gate the winner's curse breaks
                # coverage (measured: 0.823 in the noise-dominated regime).
                pe = path_edges(prev)
                fresh = pe and all(
                    self.beliefs[e].t_obs >= self._incumbent_since for e in pe
                )
                if fresh:
                    alpha_path = (
                        self.aci.working_alpha() if cfg.use_aci else cfg.alpha_prime
                    )
                    m = self.scorer.block_quantile(alpha_path, self.t, len(pe))
                    if math.isfinite(m):
                        sum_aware_L = len(pe)
                        c_sum = (
                            sum(self.beliefs[e].c_hat for e in pe)
                            + cfg.latent_margin * m
                            + sum(self.beliefs[e].rho * self.beliefs[e].age(self.t)
                                  for e in pe)
                        )
                        ub_candidates = [
                            (min(c, c_sum), p) if p is prev else (c, p)
                            for c, p in ub_candidates
                        ]
            # the certificate always reports the tightest bound
            ub = min(c for c, _ in ub_candidates)
            if cfg.use_kappa:
                slack = cfg.kappa_slack_frac * cfg.epsilon
                eligible = [p for c, p in ub_candidates if c <= ub + slack]
                incumbent = max(eligible, key=self._kappa_score)
            else:
                incumbent = min(ub_candidates, key=lambda x: x[0])[1]
        incumbent_edges = path_edges(incumbent)
        if cfg.use_kappa:
            decay = cfg.kappa_decay
            for e in self._kappa:
                self._kappa[e] *= decay
            for e in incumbent_edges:
                self._kappa[e] = self._kappa.get(e, 0.0) + 1.0
        if set(incumbent_edges) != set(path_edges(self._prev_incumbent)):
            self._incumbent_since = self.t  # freshness gate resets (T4)
        self._prev_incumbent = list(incumbent) if incumbent else []

        # Churn set (T7): edges recently on the optimistic path; the floor
        # and the sensing rotation must cover this set, not just today's path
        for e in lb_edges:
            self._churn_seen[e] = self._round_idx
        cutoff = self._round_idx - cfg.churn_window
        self._churn_seen = {
            e: r for e, r in self._churn_seen.items() if r >= cutoff
        }
        churn_edges = list(self._churn_seen)
        K = max(len(churn_edges), L, 1)

        # Step 4: churn-aware T2' certifiability floor (T7): round-robin over
        # the K-edge churn set at rate k bounds every current-path age by
        # (K-1)*Delta/k, so the sustainable floor uses K, not L
        rho_bar = max((self.beliefs[e].rho for e in lb_edges), default=0.0)
        k_now = 1
        eps_floor = 2 * L * q_eff + 2 * rho_bar * cfg.delta * L * (K - 1) / k_now
        attainable = cfg.epsilon >= eps_floor and math.isfinite(q)

        # Confidence: 1 - alpha_claim - sum of Delta_stale over the certifying
        # path. The CLAIM is the annealed level (>= alpha_prime; equals it
        # once the buffer supports the target) — never ACI's working alpha,
        # which only modulates interval width.
        d_stale = self.scorer.delta_stale(self.t)
        stale_total = L * d_stale
        if sum_aware_L:
            # T4's UB-side staleness is the BLOCK-level term (audit GAP-B;
            # block_delta_stale was dead code): charge the larger of the two
            # accountings — conservative, hence sound
            stale_total = max(
                stale_total,
                self.cfg.latent_margin
                * self.scorer.block_delta_stale(self.t, sum_aware_L),
            )
        alpha_claim = getattr(self, "_alpha_claim", self.cfg.alpha_prime)
        alpha_claim += sum(self._edge_alpha_extra.get(e, 0.0) for e in lb_edges)
        confidence = (
            max(0.0, 1.0 - alpha_claim - stale_total)
            if math.isfinite(q)
            else 0.0
        )

        cert = Certificate(
            lb=lb if p_lb is not None else math.inf,
            ub=ub,
            confidence=confidence,
            path=incumbent or [],
            epsilon_attainable=attainable,
            epsilon_floor=eps_floor,
        )

        # Step 5-6: sense unless certified; certified rounds still perform
        # maintenance sensing (projected-expiry + calibration-freshness floor),
        # otherwise the buffer ages and the claim self-extinguishes even in a
        # static world (observed in Tier-0).
        sensed: Edge | None = None
        # certification requires the claim to have annealed past the floor:
        # a gap <= epsilon at confidence 0.1 must not stop sensing
        certified = (
            cert.valid
            and cert.gap <= cfg.epsilon
            and cert.confidence >= cfg.min_certify_confidence
        )
        maintain = False
        if certified and p_lb is not None:
            growth = 2.0 * cfg.delta * sum(self.beliefs[e].rho for e in lb_edges)
            expiring = cert.gap + cfg.maintenance_lookahead * growth > cfg.epsilon
            cal_floor = self._round_idx % max(cfg.maintenance_every, 1) == 0
            maintain = expiring or cal_floor or cfg.refine_after_certify
        # Adaptive rate (T2'): choose k so the sustainable floor
        # 2*L*q + rho*Delta*L*(L-1)/k meets epsilon when possible.
        n_sense = 1
        if (
            cfg.adaptive_rate
            and math.isfinite(q)
            and not certified
            and cfg.sensing_policy == "cert"
        ):
            noise_floor = 2 * L * q_eff
            if cfg.epsilon > noise_floor and rho_bar > 0:
                k_needed = math.ceil(
                    2 * rho_bar * cfg.delta * L * (K - 1)
                    / max(cfg.epsilon - noise_floor, 1e-9)
                )
                if k_needed <= cfg.max_sense_per_round:
                    n_sense = max(1, k_needed)
                # else: epsilon unattainable even at max rate — do not burn
                # budget chasing it (T2' says no rate <= max can sustain it)
            # gap-stall feedback: the floor formula assumes a fixed path, but
            # optimism attracts the LB to the stalest region and the target
            # churns; when the gap visibly stalls above epsilon, raise the
            # rate (bounded by max_sense_per_round)
            if cert.gap >= self._last_gap - 1e-9:
                self._stall += 1
            else:
                self._stall = 0
            n_sense = min(
                cfg.max_sense_per_round, n_sense + self._stall // 5
            )
        self._last_gap = cert.gap if math.isfinite(cert.gap) else self._last_gap

        sensed_list: list[Edge] = []
        alt: set[Edge] | None = None
        for i in range(n_sense):
            pick: Edge | None = None
            if cfg.sensing_policy != "cert":
                if not certified:
                    mean_graph = None
                    if cfg.sensing_policy == "voi":
                        mean_graph = self._mean_graph()
                    pick = baseline_select(
                        cfg.sensing_policy, self.beliefs, self.t, self._rng,
                        mean_graph=mean_graph, start=self.start, goal=self.goal,
                    )
            elif (not certified or maintain) and p_lb is not None and sense_edges:
                if not math.isfinite(q):
                    # Warm-up: alternate MAPPING (round-robin the optimistic
                    # path) with CALIBRATION (re-observe the oldest already-
                    # observed edge — only repeat observations form scores).
                    # Without the alternation, unknown-terrain warm-up chases
                    # the churning P_lb onto first-touch edges and the buffer
                    # starves (measured: 26 scores from 120 observations).
                    seen = [
                        e for e, b in self.beliefs.items() if b.observed
                    ]
                    if (self._round_idx + i) % 2 == 1 and seen:
                        pick = max(seen, key=lambda e: self.beliefs[e].age(self.t))
                    else:
                        pick = sense_edges[(self._round_idx + i) % len(sense_edges)]
                elif (cfg.hybrid_sensing and not attainable) or (
                        cfg.refine_after_certify and certified):
                    # objective-matched: epsilon unattainable -> VOI on the
                    # expected-best route (departure quality is the objective)
                    if alt is None:  # latch the mean graph once per round
                        self._mean_graph_round = self._mean_graph()
                        alt = set()
                    mean_graph = self._mean_graph_round
                    pick = baseline_select(
                        "voi", self.beliefs, self.t, self._rng,
                        mean_graph=mean_graph, start=self.start, goal=self.goal,
                    )
                else:
                    if cfg.adaptive_rate:
                        # Focused mode, churn-measured (T7): focused sensing
                        # SUPPRESSES churn (measured: K 59 -> 11 ~ L) — far
                        # better than rotating over the churn set, which
                        # spreads observations thin (same cert%, +20% spend).
                        # K still feeds the floor and the rate honestly.
                        pick = select_observation(
                            self.beliefs, sense_edges, [], set(),
                            q_eff, self.t,
                            backstop_age=cfg.backstop_slack * L * cfg.delta,
                        )
                    else:
                        if alt is None:
                            alt = near_optimal_alternatives(
                                self._graph_lower_with(lo), self.start,
                                self.goal, lb, k=cfg.k_alternatives,
                                delta_subopt=cfg.delta_subopt,
                            )
                        pick = select_observation(
                            self.beliefs, sense_edges, ub_edges, alt,
                            q_eff, self.t,
                            backstop_age=cfg.backstop_slack * L * cfg.delta,
                        )
                    if pick is None and maintain:
                        # static-world maintenance: zero gap-recovery, but the
                        # calibration buffer still needs fresh residuals
                        pick = max(
                            sense_edges, key=lambda e: self.beliefs[e].age(self.t)
                        )
            if pick is None:
                break
            # Observe, score, ACI feedback, belief update. The err event uses
            # the UNCLIPPED interval (T1a observable semantics): the cost-floor
            # clip is justified by latent positivity (c > 0) and is sound
            # inside the search metrics, but the observable y = c + eta can be
            # negative under heavy-tailed noise — testing observables against
            # clipped bounds manufactures spurious miscoverage.
            b_pre = self.beliefs[pick]
            was_observed = b_pre.observed
            half = q_eff + b_pre.rho * b_pre.age(self.t)
            lo_obs, up_obs = b_pre.c_hat - half, b_pre.c_hat + half
            obs = self.ingest_observation(pick)
            covered = lo_obs - 1e-12 <= obs <= up_obs + 1e-12
            if math.isfinite(q) and was_observed:
                self.aci.update(err=not covered)
            self.sense_spend += self.beliefs[pick].sense_cost
            sensed_list.append(pick)
        self._round_idx += 1
        sensed = sensed_list[0] if sensed_list else None

        self.t += cfg.delta
        return cert, sensed

    def _mean_graph(self) -> dict[Node, dict[Node, float]]:
        """Point-estimate adjacency (max(c_hat, cost_floor)) for VOI sensing.
        Cached by beliefs-version: rebuilt only when an observation changed a
        c_hat since the last build (the dict-of-dicts is O(|E|) to construct)."""
        if (getattr(self, "_mean_graph_version", None) == self._beliefs_version
                and getattr(self, "_mean_graph_cache", None) is not None):
            return self._mean_graph_cache
        floor = self.cfg.cost_floor
        beliefs = self.beliefs
        mg: dict[Node, dict[Node, float]] = {}
        for u, nbrs in self._graph_lower_cache.items():
            mg[u] = {v: max(beliefs[(u, v)].c_hat, floor) for v in nbrs}
        self._mean_graph_cache = mg
        self._mean_graph_version = self._beliefs_version
        return mg

    def _graph_lower_with(self, lo: dict[Edge, float]) -> dict[Node, dict[Node, float]]:
        for (u, v), c in lo.items():
            self._graph_lower_cache[u][v] = c
        return self._graph_lower_cache

    def ingest_observation(self, e: Edge) -> float:
        """Observe edge e now and absorb it: drift-adjusted nonconformity
        score into the calibration buffer (theory note: the deterministic
        widening is removed so scores stay ~exchangeable under A1), belief
        update projected into the feasible set, metric-cache expiry. Used by
        sensing (paid) and by traversal (free observation while moving)."""
        b = self.beliefs[e]
        obs = self.world.observe(e, self.t)
        if self.predictor is not None and b.observed:
            pred = self.predictor(e, self.t, self.beliefs)
            if pred is not None:
                self.binned.push(abs(obs - pred), self.t, b.age(self.t))
        old_count = self._obs_count.get(e, 1 if b.observed else 0)
        self._obs_count[e] = old_count + 1
        # A score is only a valid noise-pair score when a real previous
        # observation exists (never against a prior). Thinned mode keeps only
        # disjoint pairs: the 2nd, 4th, ... observation of each edge.
        if b.observed and (
            not self.cfg.thinned_scores or self._obs_count[e] % 2 == 0
        ):
            score = abs(obs - b.c_hat) - b.rho * b.age(self.t)
            self.scorer.push(score, self.t)
            self.scorer.push_signed(obs - b.c_hat, self.t)
            self.cal_rho_a_max = max(self.cal_rho_a_max, b.rho * b.age(self.t))
            age = b.age(self.t)
            if self.cfg.rho_mode == "online" and age >= self.cfg.delta:
                self._rate_samples.append(abs(obs - b.c_hat) / age)
                if len(self._rate_samples) > 2000:
                    del self._rate_samples[0]
        b.c_hat = max(obs, self.cfg.cost_floor)
        b.t_obs = self.t
        b.observed = True
        i = self._edge_idx[e]
        self._arr_chat[i] = b.c_hat
        self._arr_tobs[i] = b.t_obs
        self._arr_obs[i] = True
        self._arr_due[i] = self.t  # expire alongside _cache_due
        self._beliefs_version += 1  # invalidates the cached snapshot gate
        self._cache_due[e] = self.t  # expire: fresh metric next round
        return obs

    def _kappa_score(self, path: list[Node]) -> float:
        """Mean conductivity over a path's edges (mean, not sum, so longer
        paths are not favored merely for having more reinforced edges)."""
        edges = path_edges(path)
        if not edges:
            return 0.0
        return sum(self._kappa.get(e, 0.0) for e in edges) / len(edges)

    def _trimmed_prev_incumbent(self) -> list[Node] | None:
        """Previous incumbent re-rooted at the current start, or None if the
        start is no longer on it (it is then not a valid s-g path)."""
        p = self._prev_incumbent
        if not p or p[-1] != self.goal:
            return None
        try:
            i = p.index(self.start)
        except ValueError:
            return None
        trimmed = p[i:]
        return trimmed if len(trimmed) >= 2 else None

    def snapshot_query(self, s: Node, g: Node, tau: float):
        """Certified O(1) route query via certificate-gated preprocessing.

        Gate: for every edge, the CURRENT interval fits inside the snapshot
        estimate +/- tau (|c_hat_now - c_hat_snap| + lambda*q + rho*a <= tau).
        On the coverage event this puts every true cost within tau of the
        snapshot costs, so the returned (snapshot-optimal) path's true cost
        is within |P|*tau of its reported cost and within 2|P|*tau of the
        true optimum. Returns dict(path, cost, slack, confidence) or None
        when the gate is closed (the oracle then needs a rebuild or the map
        is genuinely too uncertain — fall back to round()).
        """
        import numpy as _np
        from certflow.fastgraph import FlatGraph
        from certflow.snapshot import SnapshotOracle

        # the gate verdict is constant within a planner round: cache it
        stamp = (self._round_idx, self.t, tau, self._beliefs_version)
        if getattr(self, "_gate_stamp", None) == stamp:
            if not self._gate_ok:
                return None
            return self._answer_query(s, g, tau)

        # re-anneal at query time: weighted mass decays between rounds, so
        # the stored per-edge level can fall just below the supportable floor
        mass = self.scorer.effective_mass(self.t)
        if mass <= 0:
            return None
        alpha_edge_q = max(
            getattr(self, "_last_alpha_edge", self._alpha_prime_eff),
            (1.0 + 1e-6) / (mass + 1.0),
        )
        q = self.scorer.quantile(alpha_edge_q, self.t)
        if not math.isfinite(q):
            return None
        half = (self.cfg.latent_margin * q
                + self._arr_rho * (self.t - self._arr_tobs))
        if not self._arr_obs.all():
            self._gate_stamp = stamp
            self._gate_ok = False
            return None
        if self._oracle is None or self._oracle_chat_snap is None:
            drift_ok = half <= tau  # fresh build: snap == now
        else:
            drift_ok = (_np.abs(self._arr_chat - self._oracle_chat_snap)
                        + half) <= tau
        if not bool(drift_ok.all()):
            # widths/drift exceed tau on some edge: snapshot (if any) expires
            if self._oracle is not None:
                self._oracle.invalidate()
                self._oracle_chat_snap = None
            # rebuild is allowed only when the CURRENT map fits the gate
            if not bool((half <= tau).all()):
                self._gate_stamp = stamp
                self._gate_ok = False
                return None
        if self._oracle is None or not self._oracle.ready:
            if self._flat_mid is None:
                self._flat_mid = FlatGraph(
                    self._graph_lower_cache,
                    extra_nodes=(self.start, self.goal))
                ix = self._flat_mid.index_of
                self._slots_mid = _np.array(
                    [self._flat_mid.slot_of(ix[u], ix[v])
                     for u, v in self._edge_order], dtype=_np.int64)
            self._flat_mid.cost[self._slots_mid] = self._arr_chat
            self._oracle = self._oracle or SnapshotOracle(self._flat_mid)
            self._oracle.build(self.t)
            self._oracle_chat_snap = self._arr_chat.copy()
        self._gate_stamp = stamp
        self._gate_ok = True
        self._gate_alpha_edge = alpha_edge_q
        self._gate_dstale = self.scorer.delta_stale(self.t)
        return self._answer_query(s, g, tau)

    def _answer_query(self, s: Node, g: Node, tau: float):
        ix = self._flat_mid.index_of
        si, gi = ix.get(s), ix.get(g)
        if si is None or gi is None:
            return None
        pi = self._oracle.path(si, gi)
        if pi is None:
            return None
        path = [self._flat_mid.node_of(i) for i in pi]
        cost = self._oracle.cost(si, gi)
        L_p = len(path) - 1
        alpha_claim = max(
            getattr(self, "_alpha_claim", self._alpha_prime_eff),
            min(1.0, L_p * self._gate_alpha_edge),
        )
        return dict(
            path=path, cost=cost, slack=L_p * tau,
            opt_slack=2 * L_p * tau,
            confidence=path_confidence(
                alpha_claim, [self._gate_dstale] * max(L_p, 1)),
        )

    def retarget(self, start: Node, goal: Node) -> None:
        """New mission in the same environment (lifelong operation): keep the
        learned memory — beliefs, calibration buffer, ACI state, kappa — and
        drop mission-specific state (incumbent, sensing target, gap-stall).
        Searches are rebuilt from scratch at the new endpoints (a global
        change; scratch beats repair)."""
        self.start, self.goal = start, goal
        self._prev_incumbent = []
        self._p_sense = []
        self._incumbent_since = self.t
        self._last_gap = math.inf
        self._stall = 0
        if hasattr(self, "_last_L"):
            del self._last_L
        if self._cache_lo:
            self._rebuild_searches()
        else:  # retarget before any round: warm-up metrics, fresh engines
            self.sp_lower = FastDStarLite(
                self._graph_lower_cache, start, goal, flat=self._flat_lo)
            self.sp_upper = FastDStarLite(
                self._to_adj(set(self._graph_lower_cache),
                             {e: _UB_CAP for e in self.beliefs}),
                start, goal, flat=self._flat_up)

    def advance_start(self, node: Node) -> None:
        """Robot moved: shift both searches' start (D* Lite km offset)."""
        self.start = node
        self.sp_lower.set_start(node)
        self.sp_upper.set_start(node)
