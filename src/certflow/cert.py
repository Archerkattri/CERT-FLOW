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
    ConformalTestMartingale,
    ShiryaevRobertsDetector,
    ShrinkLicense,
    conformal_p_value,
    effective_sample_size,
    path_alpha_edge,
    path_confidence,
    residual_drift_score,
    weighted_group_quantile,
)
from certflow.fastgraph import FastDijkstraSearch, FastDStarLite, FlatGraph
from certflow.graphcore import dijkstra
from certflow.sensing import baseline_select, near_optimal_alternatives, path_edges, select_observation
from certflow.types import Certificate, Edge, EdgeBelief, Node, World
from certflow.upgrades import (
    ActiveSensingPolicy,
    DecisionRiskController,
    EvidenceConditionalCalibrator,
    RegimeRecoveryManager,
    SelectionConditionalCalibrator,
    SelectionLedger,
    SequentialRecoveryMonitor,
    SensingAction,
    TrajectoryConformalCalibrator,
    TrajectoryTube,
    evidence_features,
)

# Finite upper-cost cap for unbounded edges. An UNOBSERVED edge (or warm-up
# u-cost at q=inf) has no coverage theorem pricing its upper bound, so it is
# conceptually +inf; D* Lite needs strictly finite, positive costs to maintain
# its g/rhs invariants, so we cap at this sentinel. It must dominate any real
# path cost (so capped edges are never chosen unless unavoidable) yet stay well
# below float overflow when summed over a path.
_UB_CAP = 1e9


def _grid_heuristic(flat: FlatGraph, graph: dict[Node, dict[Node, float]],
                    goal: Node, cost_floor: float):
    """Return an admissible grid lower bound, or ``None`` for generic graphs.

    The bound is enabled only for four-neighbour integer-coordinate graphs.
    Every planner edge costs at least ``cost_floor``, so Manhattan distance
    times that floor is consistent for both optimistic and conservative metric
    graphs.  Arbitrary road/task graphs retain the zero-heuristic kernel.
    """
    if not isinstance(goal, tuple) or len(goal) != 2:
        return None
    if any(
        not isinstance(n, tuple) or len(n) != 2
        or not all(isinstance(x, int) for x in n)
        for n in flat.nodes
    ):
        return None
    for u, nbrs in graph.items():
        for v in nbrs:
            if sum(abs(int(a) - int(b)) for a, b in zip(u, v)) != 1:
                return None
    import numpy as _np
    return _np.asarray(
        [cost_floor * (abs(n[0] - goal[0]) + abs(n[1] - goal[1]))
         for n in flat.nodes],
        dtype=_np.float64,
    )


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

    # Add the current point-estimate shortest path to the certified candidate
    # set and execute it only inside the existing kappa corridor.  Its
    # conservative upper cost remains part of the certificate UB.
    mean_path_execution: bool = False

    # Search backend. ``dstar`` preserves the incremental default; ``dijkstra``
    # uses exact fresh flat queries; ``auto`` selects the latter only for large
    # verified grid topologies where queue repair is empirically slower.
    search_backend: str = "dstar"

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

    # CIA upper certificate: use a path-sum conformal UB for the fresh,
    # selected incumbent. It spends a separate alpha budget from the LB
    # (default half/half) and falls back to the ordinary UB when unsupported.
    # This is the composition-safe alternative to independently trusting an
    # LB and a path-sum UB at the same nominal alpha. If both CIA and
    # sum-aware UB are enabled, CIA takes precedence for the incumbent.
    cia_ub: bool = False
    cia_ub_alpha_fraction: float = 0.5

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

    # Parallel calibration warm-up: before a finite conformal quantile exists,
    # observe distinct edges in a round so long paths do not spend one whole
    # day merely filling the path-level support requirement.  The schedule is
    # deterministic and alternates mapping with repeat observations; it never
    # repeats an edge at the same timestamp, so it does not manufacture extra
    # independent calibration scores.  Off by default for legacy accounting.
    warmup_sense_per_round: int = 1

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

    # Evidence-conditioned edge pricing (Varun/PASC follow-up): route
    # calibration by the age of the evidence being priced. A path uses the
    # maximum supported within-bin quantile across its edges; if any path bin
    # is immature, pricing falls back to the pooled quantile for the whole
    # path. This is opt-in because the conditional claim requires the stated
    # within-bin exchangeability assumption and enough support per bin.
    age_stratify: bool = False
    age_bins: tuple = (1.0, 3.0, 6.0, 12.0, 24.0, 48.0)

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

    # --- Distribution-shift staleness model (arXiv 2502.14105) ----------------
    # shift_model="tv" (default) keeps the TV-Lipschitz Delta_stale coverage
    # correction. shift_model="lp" swaps in the Levy-Prokhorov worst-case
    # quantile instead: the per-edge conformal quantile becomes
    #   quantile(alpha_edge - rho_lp) + eps_lp
    # (level shift + flat offset) and the confidence penalty per edge becomes
    # the flat LP mass rho_lp rather than 2*eps_tv*age. eps_lp is the smooth-
    # drift budget per query (units of the score), rho_lp the mass of abruptly
    # / adversarially changed edges. Both 0 => LP reduces to plain conformal.
    # LP intervals are always >= the exchangeable ones, so it is a purely
    # conservative (sound) alternative. Default reproduces the TV behavior.
    shift_model: str = "tv"
    eps_lp: float = 0.0
    rho_lp: float = 0.0

    # --- ACI step-size rule (SAOCP, arXiv 2302.07869, Alg. 2) -----------------
    # aci_mode="fixed" (default) is the Gibbs-Candes gamma-step ACI. "sf-ogd"
    # uses the scale-free OGD step s_{t+1} = s_t - eta*g_t/sqrt(sum ||g_i||^2),
    # which is anytime and needs no gamma tuned to the err/score magnitude.
    aci_mode: str = "fixed"
    aci_eta: float = 0.1

    # --- Path-level calibration (CIA, arXiv 2408.10939; PASC, arXiv 2605.18812)-
    # path_calibration="bonferroni" (default) is the per-edge Bonferroni UB.
    # "cia" is the experimental group-sum path calibration exposed via
    # CertPlanner.cia_path_certificate(); it does NOT change round()'s default
    # certificate. Set cia_ub=True to wire a composition-safe CIA UB into
    # round() with an explicit split alpha budget. rho_w for the CIA drift
    # retrofit is taken from cfg.rho_w.
    #
    # "pasc" (experimental, live-wired 2026): _q() prices edges with the PASC
    # JOINT per-edge radius instead of the per-edge Bonferroni quantile. The
    # radius Q is the age-weighted (1-alpha_path) quantile of the per-block
    # MAX |score|, blocks of length path_len drawn newest-first from the
    # absolute-score buffer (the same construction as pasc_edge_radius()). A
    # single joint quantile replaces the alpha/L per-edge union bound, so it is
    # tighter-or-equal while still delivering joint per-edge coverage >=
    # 1-alpha_path. HONESTY: PASC's joint guarantee assumes the calibration
    # BLOCKS are exchangeable with the fresh path (block exchangeability); the
    # age-weighted retrofit inherits exactly the same weighted-coverage argument
    # (Barber et al. 2023 Thm 2) as the rest of this module -- no stronger, no
    # weaker. Until the buffer holds a full block at the required level the
    # radius is +inf and _q() falls back to Bonferroni (warm-up, unchanged).
    path_calibration: str = "bonferroni"
    cia_symmetric: bool = False
    cia_stratify: bool = False

    # --- Live validity monitor (WATCH, arXiv 2505.04608) ----------------------
    # watch_monitor=True feeds every new (weighted) conformal p-value into a
    # ConformalTestMartingale (planner.watch) AND a ShiryaevRobertsDetector
    # (planner.sr), owned by the planner, turning the pinned-at-1.0 coverage
    # into an OBSERVABLE, alarming quantity. It changes NO certificate and NO
    # pricing -- purely diagnostic. planner.diagnostics() exposes the martingale
    # value/alarm, the SR peak/alarm, the recent-vs-buffer residual drift score
    # (W1) and the calibration weights' effective sample size. Ville's
    # inequality bounds the martingale false-alarm probability at
    # watch_alarm_delta; sr_threshold is the SR false-alarm ARL target.
    watch_monitor: bool = False
    watch_epsilon: float = 0.5       # power betting exponent (bets on small p)
    watch_alarm_delta: float = 0.01  # Ville false-alarm budget for planner.watch
    sr_threshold: float = 100.0      # Shiryaev-Roberts ARL threshold
    watch_window: int = 50           # recent-window size for the drift score

    # All-pairs snapshot preflight. The snapshot path is an optimization only;
    # exceeding this cap raises a named resource diagnostic before n*n tables
    # are allocated. Callers can choose a larger explicit cap or use the normal
    # bounded online/ALT path; no over-budget snapshot is certified.
    snapshot_max_bytes: int = 512 * 1024 * 1024

    # --- Test-then-tighten license (ShrinkLicense) ----------------------------
    # shrink_license=True maintains, per shrink factor k in shrink_grid, an
    # anytime-valid upper confidence sequence on the OBSERVED per-edge violation
    # rate of the k-shrunk full radius (Waudby-Smith & Ramdas 2023 betting CS,
    # Ville time-uniform at level shrink_delta). It LICENSES a tighter Tier-2
    # radius k*radius ALONGSIDE (never instead of) the Tier-1 certificate:
    # licensed_k is the smallest k whose UCB <= shrink_alpha_target after
    # shrink_n_min samples, else 1.0 (self-revoking under a regime shift). It
    # changes NO certificate and NO pricing -- purely observational, surfaced in
    # diagnostics() (licensed_k, per-k ucb+n, and the shadow shrunk gap). See
    # conformal.ShrinkLicense for the exact (honest) validity claim.
    shrink_license: bool = False
    shrink_grid: tuple = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    shrink_delta: float = 0.05
    shrink_n_min: int = 50
    shrink_alpha_target: float | None = None  # None -> use alpha_prime

    # --- CERT-FLOW v3 upgrade layer ------------------------------------------
    # These switches are deliberately opt-in: the legacy certificate stream
    # remains byte-identical when they are all false.
    selection_conditional: bool = False
    selection_audit_min: int = 10
    evidence_model: bool = False
    evidence_model_min_calibration: int = 20
    decision_risk_target: float | None = None
    decision_risk_delta: float = 0.05
    trajectory_tubes: bool = False
    trajectory_min_calibration: int = 10
    active_sensing: bool = False
    active_sensing_exploration: float = 0.5
    active_sensing_min_improvement: float = 0.25
    regime_recovery: bool = False
    recovery_formal: bool = False
    recovery_alarm_delta: float = 0.01
    recovery_samples: int = 5
    recovery_evidence_threshold: float = 4.0
    recovery_betting_epsilons: tuple | None = None


def recommended_config(**overrides) -> "PlannerConfig":
    """The best-known configuration from the full ablation/benchmark program:
    online drift estimation (coverage-neutral, 1.7-2.4x tighter gaps),
    objective-matched hybrid sensing (regret matches/beats the strongest
    baseline while keeping the certificate), kappa hysteresis (-70% churn),
    adaptive rate + adaptive pre-widening, gated sum-aware UB. Annealing is
    already the default. decision_uniform stays a claim-semantics choice."""
    base = dict(
        rho_mode="online",
        # Median online-rate pricing is the robust point on the traffic
        # crossover: the 0.9 tail overreacts to sensor noise on PEMS-BAY,
        # while conformal residuals still absorb the unmodeled remainder.
        rho_online_quantile=0.5,
        hybrid_sensing=True,
        use_kappa=True,
        adaptive_rate=True,
        sum_aware_ub=True,
        # Long paths can consume the weighted support faster than a small
        # warm-up replenishes it.  A 12-edge burst reaches stable support
        # earlier than 8/8 on the 60x60 stress case, reducing both unsupported
        # rounds and total long-run repair time.
        warmup_sense_per_round=12,
        max_sense_per_round=12,
        search_backend="auto",
        # Small safety cushion for finite-sample and measurement-noise tail
        # effects.  It costs only a few percent of the conformal radius while
        # retaining the large sharpness gain of the sum-aware bound.
        latent_margin=1.05,
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

        self.scorer = ConformalScorer(
            rho_w=config.rho_w, eps_tv=config.eps_tv,
            shift_model=config.shift_model, eps_lp=config.eps_lp,
            rho_lp=config.rho_lp,
        )
        self.predictor = predictor
        self.binned = AgeBinnedScorer(
            bin_edges=tuple(b * config.delta for b in config.predictor_bins),
            rho_w=config.rho_w, eps_tv=config.eps_tv,
        )
        self.evidence_binned = AgeBinnedScorer(
            bin_edges=tuple(b * config.delta for b in config.age_bins),
            rho_w=config.rho_w, eps_tv=config.eps_tv,
        )
        # v3 upgrade state.  None of these objects affects the legacy
        # certificate unless its corresponding config flag is enabled.
        self.selection_ledger = SelectionLedger()
        self.selection_calibrator = SelectionConditionalCalibrator(
            min_audit=config.selection_audit_min
        )
        self.evidence_calibrator = EvidenceConditionalCalibrator(
            min_calibration=config.evidence_model_min_calibration
        )
        self.decision_risk = DecisionRiskController(delta=config.decision_risk_delta)
        self.trajectory_calibrator = TrajectoryConformalCalibrator(
            min_calibration=config.trajectory_min_calibration
        )
        self.active_sensing = ActiveSensingPolicy(
            exploration=config.active_sensing_exploration
        )
        self.regime = RegimeRecoveryManager(
            alarm_delta=config.recovery_alarm_delta,
            recovery_samples=config.recovery_samples,
            evidence_threshold=config.recovery_evidence_threshold,
        )
        self.formal_regime = SequentialRecoveryMonitor(
            alarm_threshold=1.0 / max(config.recovery_alarm_delta, 1e-6),
            recovery_samples=config.recovery_samples,
            betting_epsilon=config.watch_epsilon,
            betting_epsilons=config.recovery_betting_epsilons,
        )
        self._selection_audit_scores: list[float] = []
        self._selection_audit_observation_ids: set[str] = set()
        self._selection_audit_key: tuple[Node, ...] | None = None
        self._selection_audit_decision_id: int | None = None
        self._last_selection_event = None
        # Automatic evidence calibration uses a chronological train/calibration
        # split.  It never prices with a model fitted on the same residuals
        # that calibrate its normalized radius.  The explicit
        # fit_evidence_model() API remains available for externally supplied
        # held-out data and marks this state as manually fitted.
        self._evidence_train_features: list[list[float]] = []
        self._evidence_train_residuals: list[float] = []
        self._evidence_cal_features: list[list[float]] = []
        self._evidence_cal_residuals: list[float] = []
        self._evidence_auto_fitted = False
        self.pred_used_rounds = 0  # diagnostic: edges priced by the predictor
        self._edge_alpha_extra: dict[Edge, float] = {}  # per-bin annealing charge
        self.aci = ACITracker(
            alpha_target=config.alpha_prime, gamma=config.gamma_aci,
            mode=config.aci_mode, eta=config.aci_eta,
        )
        # Live validity monitor (WATCH, arXiv 2505.04608). Always constructed so
        # planner.watch / planner.sr exist; only FED when cfg.watch_monitor. The
        # martingale/detector consume the per-round weighted conformal p-values
        # pushed in ingest_observation. Purely observational: no pricing effect.
        self.watch = ConformalTestMartingale(
            epsilon=config.watch_epsilon, alarm_delta=config.watch_alarm_delta,
        )
        self.sr = ShiryaevRobertsDetector(
            threshold=config.sr_threshold, epsilon=config.watch_epsilon,
        )
        # Test-then-tighten license (ShrinkLicense). Always constructed so
        # planner.shrink exists; only FED when cfg.shrink_license. Purely
        # observational: no pricing effect, certificate untouched.
        self.shrink = ShrinkLicense(
            grid=tuple(config.shrink_grid),
            delta=config.shrink_delta,
            n_min=config.shrink_n_min,
        )
        # last per-edge quantile the planner priced with (q_eff = lambda*q); the
        # shrink outcome uses radius_t = q_edge + rho*age. inf = not yet priced.
        self._shrink_q_edge = math.inf
        self._recent_scores: list[float] = []  # rolling window for the drift score
        self.sense_spend = 0.0
        self._round_idx = 0
        self._obs_count: dict[Edge, int] = {}  # real observations per edge
        self._rate_samples: list[float] = []    # pooled online-rate fallback
        self._rate_samples_by_edge: dict[Edge, list[float]] = {}
        self._rho_online = 1e-9
        self._rho_dirty_edges: set[Edge] = set()
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
        self._last_cia_ub_used = False
        self._rng = random.Random(0)  # baseline sensing policies only
        self._warmup_cursor = 0

        nodes = set(world.graph) | {v for n in world.graph for v in world.graph[n]}
        lo, up = self._metrics(q=math.inf)  # warm-up: q=inf -> ell at floor, u at inf
        # D* Lite needs finite costs; cap warm-up upper costs.
        up = {e: min(c, _UB_CAP) for e, c in up.items()}
        adj_lo = self._to_adj(nodes, lo)
        adj_up = self._to_adj(nodes, up)
        self._flat_lo = FlatGraph(adj_lo, extra_nodes=(start, goal))
        self._flat_up = FlatGraph(adj_up, extra_nodes=(start, goal))
        self._heuristic = _grid_heuristic(
            self._flat_lo, adj_lo, goal, config.cost_floor
        )
        if config.search_backend not in {"dstar", "dijkstra", "auto"}:
            raise ValueError("search_backend must be 'dstar', 'dijkstra', or 'auto'")
        self._use_fresh_dijkstra = (
            config.search_backend == "dijkstra"
            or (
                config.search_backend == "auto"
                and self._heuristic is not None
                and self._flat_lo.n >= 1600
            )
        )
        self.sp_lower = self._new_search(adj_lo, self._flat_lo, start, goal)
        self.sp_upper = self._new_search(adj_up, self._flat_up, start, goal)
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
        self._flat_mean = FlatGraph(
            self._graph_lower_cache, extra_nodes=(start, goal)
        )
        ix_mean = self._flat_mean.index_of
        self._slots_mean = _np.array(
            [self._flat_mean.slot_of(ix_mean[u], ix_mean[v])
             for u, v in self._edge_order],
            dtype=_np.int64)
        self._mean_search = FastDijkstraSearch(
            self._flat_mean, start, goal
        )
        self._mean_flat_version = -1
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

    def _new_search(self, graph, flat: FlatGraph, start: Node, goal: Node):
        if self._use_fresh_dijkstra:
            return FastDijkstraSearch(flat, start, goal)
        return FastDStarLite(
            graph, start, goal, flat=flat, heuristic=self._heuristic
        )

    def _mean_path(self) -> list[Node] | None:
        """Exact point-estimate route using the numba flat backend."""
        if self._mean_flat_version != self._beliefs_version:
            import numpy as _np
            values = _np.maximum(self._arr_chat, self.cfg.cost_floor)
            self._flat_mean.cost[self._slots_mean] = values
            self._mean_flat_version = self._beliefs_version
        path, _ = self._mean_search.shortest_path()
        return path

    def _mean_path_edges(self) -> list[Edge]:
        return path_edges(self._mean_path())

    def _warmup_calibration_edge(self, excluded: set[Edge]) -> Edge | None:
        """Pick a previously observed edge in O(1) on surveyed grids.

        Warm-up used to scan and age-rank the entire graph for every repeat
        observation.  A deterministic round-robin over the fixed edge order
        supplies the same independent-score purpose without making a 12-edge
        burst pay twelve full-graph Python scans.
        """
        n = len(self._edge_order)
        if n == 0:
            return None
        for offset in range(n):
            i = (self._warmup_cursor + offset) % n
            e = self._edge_order[i]
            if self.beliefs[e].observed and e not in excluded:
                self._warmup_cursor = (i + 1) % n
                return e
        return None

    def _adaptive_B(self, q_eff: float) -> int:
        """Pre-widening horizon: cap width spend at a fraction of the
        epsilon-slack when certification is in play; keep the configured
        latency-optimal horizon during warm-up or unattainable epsilon."""
        cfg = self.cfg
        B = cfg.prewiden_rounds
        if not cfg.adaptive_rate or B <= 0:
            return max(B, 0)
        L_b = max(getattr(self, "_last_L", 1), 1)
        # rho is mirrored in the fixed edge-order array; avoid a Python
        # generator scan over every edge on each q/metric coupling pass.
        rho_b = float(self._arr_rho.max()) if len(self._arr_rho) else 0.0
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
        self.sp_lower = self._new_search(
            self._graph_lower_cache, self._flat_lo, self.start, self.goal
        )
        self.sp_upper = self._new_search(
            self._graph_lower_cache, self._flat_up, self.start, self.goal
        )

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
        # An upward radius change can make an existing cached metric
        # optimistic, so it must invalidate immediately.  A downward estimate
        # is already covered by the old (wider) cache and can take effect at
        # the ordinary edge-expiry boundary; rebuilding the whole graph for
        # every downward refinement was the source of the 60x60 p95 spikes.
        changed = rho > self._rho_online * 1.05
        self._rho_online = rho
        # A pooled scalar is a safe cold-start fallback, but it discards the
        # spatial heterogeneity that matters on real traffic.  Once an edge
        # has enough independent re-observation rates, use its own robust
        # quantile; sparse edges continue using the pooled estimate.  This is
        # still model-free: neither branch consults world.rho_true().
        min_local = max(3, int(cfg.rho_online_min_samples))
        for e, b in self.beliefs.items():
            local = self._rate_samples_by_edge.get(e, [])
            if len(local) >= min_local:
                ordered = sorted(local)
                target = max(
                    ordered[int(cfg.rho_online_quantile * (len(ordered) - 1))],
                    1e-9,
                )
            else:
                target = rho
            if abs(target - b.rho) > 0.05 * max(b.rho, 1e-9):
                upward = target > b.rho
                b.rho = target
                self._arr_rho[self._edge_idx[e]] = target
                if upward:
                    # The next metric refresh updates only this edge.  Keeping
                    # a dirty set avoids rebuilding the entire graph whenever
                    # one local rate estimate moves upward.
                    self._rho_dirty_edges.add(e)

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

    def _evidence_interval(self, e: Edge, age: float) -> tuple[float, float] | None:
        """Case-dependent interval from the opt-in learned evidence model."""
        if not self.cfg.evidence_model or not self.evidence_calibrator.ready:
            return None
        alpha_edge = getattr(self, "_last_alpha_edge", self._alpha_prime_eff)
        features = evidence_features(
            age,
            self.beliefs[e].rho * age,
            getattr(self, "_last_L", 1),
            self.scorer.effective_mass(self.t),
            overlap=1.0,
        )
        radius = self.evidence_calibrator.radius(features, alpha_edge)
        if not math.isfinite(radius):
            return None
        half = self.cfg.latent_margin * radius + self.beliefs[e].rho * age
        return max(self.cfg.cost_floor, self.beliefs[e].c_hat - half), max(
            self.cfg.cost_floor, self.beliefs[e].c_hat + half
        )

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
        if self.predictor is None and cfg.evidence_model and self.evidence_calibrator.ready:
            # Evidence-conditioned pricing used to fall through to the
            # dict/edge loop below, turning a width improvement into a large
            # latency regression on METR-LA/PEMS-BAY.  The model and the
            # normalized conformal factor are both independent of edge order,
            # so refresh the whole metric vector with NumPy and rebuild the
            # shared flat searches exactly as the ordinary fast path does.
            import numpy as _np
            # FastDStarLite is imported at module scope; keep it out of this
            # local branch so ordinary (non-evidence) refreshes see the same
            # symbol and do not trigger Python's local-name shadowing rule.

            horizon = 0.0 if B <= 0 else self._adaptive_B(q_eff) * cfg.delta
            ages = _np.maximum(0.0, self.t - self._arr_tobs)
            if horizon > 0.0:
                ages = ages + horizon
                h = horizon * self._arr_stagger
                dues = self.t + h
            else:
                dues = _np.full(len(self._edge_order), self.t)
            features = _np.column_stack((
                ages,
                self._arr_rho * ages,
                _np.full(len(ages), max(getattr(self, "_last_L", 1), 1.0)),
                _np.full(len(ages), self.scorer.effective_mass(self.t)),
                _np.ones(len(ages)),
                _np.zeros(len(ages)),
            ))
            alpha_edge = min(
                1.0 - 1e-9,
                max(1e-9, getattr(self, "_last_alpha_edge", self._alpha_prime_eff)),
            )
            radius = self.evidence_calibrator.radius_many(features, alpha_edge)
            # The scalar evidence path falls back to pooled pricing when the
            # normalized quantile lacks support at a long-path alpha.  Keep
            # that sound warm-up behavior in the vector path as well; using
            # inf directly would turn every affected edge into _UB_CAP.
            finite_radius = _np.isfinite(radius)
            half = _np.where(
                finite_radius,
                cfg.latent_margin * radius,
                q_eff,
            ) + self._arr_rho * ages
            lo_a = _np.maximum(self._arr_chat - half, cfg.cost_floor)
            up_a = _np.minimum(self._arr_chat + half, _UB_CAP)
            unobs = ~self._arr_obs
            lo_a[unobs] = cfg.cost_floor
            up_a[unobs] = _UB_CAP
            self._cache_lo = dict(zip(self._edge_order, lo_a.tolist()))
            self._cache_up = dict(zip(self._edge_order, up_a.tolist()))
            self._cache_due = dict(zip(self._edge_order, dues.tolist()))
            self._arr_due = dues
            self._cache_q = q_eff
            self._flat_lo.cost[self._slots_lo] = lo_a
            self._flat_up.cost[self._slots_up] = up_a
            self.sp_lower = self._new_search(
                self._graph_lower_cache, self._flat_lo, self.start, self.goal
            )
            self.sp_upper = self._new_search(
                self._graph_lower_cache, self._flat_up, self.start, self.goal
            )
            return
        if self.predictor is None and not (
            cfg.evidence_model and self.evidence_calibrator.ready
        ):
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
                self._rho_dirty_edges.clear()
                self._flat_lo.cost[self._slots_lo] = lo_a
                self._flat_up.cost[self._slots_up] = up_a
                self.sp_lower = self._new_search(
                    self._graph_lower_cache, self._flat_lo, self.start, self.goal
                )
                self.sp_upper = self._new_search(
                    self._graph_lower_cache, self._flat_up, self.start, self.goal
                )
                # adjacency VALUES are consumed only by the alternatives
                # helper, which refreshes them on demand (_graph_lower_with);
                # the engines read costs from the flat arrays — skip the
                # O(|E|) dict-of-dicts rebuild here
                return
            # vectorized staggered due-subset (small by construction)
            mask = self._arr_due <= self.t
            dirty_idx = [self._edge_idx[e] for e in self._rho_dirty_edges
                         if e in self._edge_idx]
            if dirty_idx:
                mask[dirty_idx] = True
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
                self._rho_dirty_edges.difference_update(
                    self._edge_order[j] for j in dirty_idx
                    if j < len(self._edge_order)
                )
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
                ei = self._evidence_interval(e, a_pre)
                if ei is not None:
                    lo_v, up_v = ei
                else:
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

    def _cia_ub_alpha(self) -> float:
        """Miscoverage budget reserved for the optional CIA upper bound."""
        fraction = self.cfg.cia_ub_alpha_fraction
        if not 0.0 < fraction < 1.0:
            raise ValueError("cia_ub_alpha_fraction must be in (0, 1)")
        return self._alpha_prime_eff * fraction

    def _pricing_alpha(self) -> float:
        """Alpha used by live conformal pricing.

        Decision-uniform mode spends a frozen share of ``alpha_prime`` per
        decision.  ACI remains available for ordinary marginal pricing, but it
        must not override that spent share in any decision-uniform path.
        """
        if self.cfg.decision_uniform:
            alpha = self._alpha_prime_eff
        else:
            alpha = self.aci.working_alpha() if self.cfg.use_aci else self.cfg.alpha_prime
        if self.cfg.cia_ub:
            fraction = self.cfg.cia_ub_alpha_fraction
            if not 0.0 < fraction < 1.0:
                raise ValueError("cia_ub_alpha_fraction must be in (0, 1)")
            # Keep the LB + CIA-UB budgets within the configured claim even
            # when ACI has adapted above its target.
            alpha = min(alpha, self._alpha_prime_eff)
            alpha *= 1.0 - fraction
        return alpha

    def _q(self, path_len: int, path: list[Node] | None = None) -> float:
        alpha_path = self._pricing_alpha()
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
        if self.cfg.path_calibration == "pasc" and 0.0 < alpha_path < 1.0:
            # PASC joint per-edge radius (arXiv 2605.18812): one age-weighted
            # (1-alpha_path) quantile of the per-block MAX score prices EVERY
            # edge of the path jointly at >= 1-alpha_path, replacing the alpha/L
            # per-edge Bonferroni quantile with a single (tighter-or-equal) joint
            # quantile. Returns +inf while the buffer holds no full block
            # supportable at this level -> fall through to the Bonferroni path
            # below (warm-up: current behavior, unchanged). The alpha_path==1
            # annealing corner (nothing supportable) also falls through -- the
            # certificate is invalid there anyway.
            Q = self._pasc_radius(path_len, alpha_path)
            if math.isfinite(Q):
                self._last_alpha_edge = alpha_path
                return Q
        if self.cfg.shift_model == "lp":
            # Levy-Prokhorov worst-case per-edge quantile: level shift by rho_lp,
            # flat offset eps_lp. Always >= the exchangeable quantile (sound).
            # The shifted level consumes an extra rho_lp of per-edge mass, so
            # the annealing floor must guarantee the SHIFTED level is supportable
            # (else quantile_lp returns +inf). Raise only the quantile level, not
            # the reported claim (rho_lp is charged separately in the confidence
            # via the LP staleness term) — a purely conservative widening.
            if self.cfg.anneal_alpha:
                m = self.scorer.effective_mass(self.t)
                if m > 0.0:
                    floor_q = min(
                        0.999,
                        path_len * ((1.0 + 1e-9) / (m + 1.0) + self.cfg.rho_lp),
                    )
                    alpha_path = max(alpha_path, floor_q)
            alpha_edge = path_alpha_edge(alpha_path, path_len)
            self._last_alpha_edge = alpha_edge
            return self.scorer.quantile_lp(
                alpha_edge, self.t, eps=self.cfg.eps_lp, rho=self.cfg.rho_lp)
        alpha_edge = path_alpha_edge(alpha_path, path_len)
        self._last_alpha_edge = alpha_edge
        q = self.scorer.quantile(alpha_edge, self.t)
        if self.cfg.age_stratify and path is not None:
            edges = path_edges(path)
            if edges:
                age_quantiles = [
                    self.evidence_binned.quantile(
                        alpha_edge, self.t, self.beliefs[e].age(self.t)
                    )
                    for e in edges
                ]
                # A conditional path price is only useful when every edge's
                # evidence stratum is supported. Otherwise retain the pooled
                # quantile rather than mixing conditional and pooled claims.
                if all(math.isfinite(v) for v in age_quantiles):
                    q = max(age_quantiles)
        return q

    def _pasc_radius(self, path_len: int, alpha_path: float) -> float:
        """PASC joint per-edge radius for a path of ``path_len`` edges at level
        ``alpha_path`` (arXiv 2605.18812), computed from the live score buffer.
        Shares pasc_edge_radius()'s block construction but takes the block length
        directly (round() knows L, not a concrete calibration path) and matches
        the LIVE score convention:

        The buffer stores the DRIFT-ADJUSTED nonconformity score
        ``s_e = |obs_e - c_hat_e| - rho_e * age_e`` (already an absolute
        deviation minus the drift allowance). Bonferroni prices an edge at
        ``c_hat +/- (q + rho*age)`` with ``q = scorer.quantile(alpha/L)`` = the
        (1-alpha/L) weighted quantile of these SIGNED scores; coverage of edge e
        needs ``q + rho*age_e >= |obs_e - c_hat_e|`` i.e. ``q >= s_e``. The joint
        (all edges at once) analogue is therefore the (1-alpha) weighted quantile
        of the per-block MAX of the SIGNED scores ``max_e s_e`` -- NOT
        ``max_e |s_e|``. The standalone pasc_edge_radius()/PASCCalibrator take
        ``|residual|`` because they were written for RAW symmetric residuals
        (``yhat +/- Q``); applying abs to these already-drift-adjusted (hence
        often negative) scores would double-count the drift subtraction and blow
        the radius up. Signed-max keeps PASC the tighter-or-equal joint sibling
        of the per-edge Bonferroni quantile.

        Soundness: ``Q >= max_e s_e`` jointly (prob >= 1-alpha under block
        exchangeability) gives ``Q + rho*age_e >= |obs_e - c_hat_e|`` for every
        edge simultaneously -- the SAME per-edge magnitude bound Bonferroni
        certifies, just calibrated jointly instead of via the alpha/L union.
        Returns +inf while fewer than one full block exists (warm-up)."""
        pl = max(path_len, 1)
        samples = sorted(self.scorer._buf, key=lambda s: s.t, reverse=True)
        n_blocks = len(samples) // pl
        if n_blocks == 0:
            return math.inf
        scores, weights = [], []
        for b in range(n_blocks):
            block = samples[b * pl : (b + 1) * pl]
            scores.append(max(s.residual for s in block))
            weights.append(
                min(self.cfg.rho_w ** max(0.0, self.t - s.t) for s in block)
            )
        return weighted_group_quantile(scores, weights, alpha_path)

    def diagnostics(self) -> dict:
        """Live validity/tightness dashboard for watch_monitor runs (WATCH,
        arXiv 2505.04608 + DASC diagnostics, arXiv 2606.15953). All quantities
        are OBSERVATIONAL -- none feed the certificate or the pricing:

        * ``watch_value`` / ``watch_running_max`` / ``watch_alarm``: the
          conformal test-martingale wealth and its Ville alarm (the null being
          tested is the weighted-exchangeability the certificate assumes; an
          alarm means the staleness/weighting model is breaking).
        * ``sr_peak`` / ``sr_alarm`` / ``sr_alarm_round``: the Shiryaev-Roberts
          e-detector, which restarts implicitly so a LATE change still alarms
          quickly after a long quiet null.
        * ``residual_drift_score``: DASC W1 distance between the recent-window
          scores and the full calibration buffer (drift magnitude).
        * ``effective_sample_size``: Kish n_eff of the age weights (how many
          samples the weighted quantile effectively rests on).
        """
        w = self.scorer._weights(self.t)
        cal = [s.residual for s in self.scorer._buf]
        recent = self._recent_scores[-self.cfg.watch_window:]
        out = dict(
            watch_value=self.watch.value,
            watch_running_max=self.watch.running_max,
            watch_alarm=self.watch.alarm(),
            sr_peak=self.sr.peak,
            sr_alarm=self.sr.alarm(),
            sr_alarm_round=self.sr.alarm_round,
            residual_drift_score=residual_drift_score(recent, cal),
            effective_sample_size=effective_sample_size(w),
            n_scores=len(cal),
        )
        # Tier-2 shrink license (ShrinkLicense): observational only, never feeds
        # the certificate. licensed_k, the per-k upper CS bound + sample counts,
        # and the SHADOW gap the current certificate would carry if every
        # per-edge radius were multiplied by licensed_k.
        alpha_t = (
            self.cfg.shrink_alpha_target
            if self.cfg.shrink_alpha_target is not None
            else self.cfg.alpha_prime
        )
        lk = self.shrink.licensed_k(alpha_t)
        # shrunk_lb/ub: APPROXIMATION -- the incumbent path is held FIXED and its
        # per-edge shadow bounds are recomputed at k*(q_edge + rho*age) using the
        # last-priced quantile; it does NOT re-run the shortest-path searches
        # under the shrunk metric, so it can differ from a full shrunk re-plan.
        inc_edges = (
            path_edges(self._prev_incumbent) if self._prev_incumbent else []
        )
        shrunk_lb = shrunk_ub = float("nan")
        if inc_edges and math.isfinite(self._shrink_q_edge):
            lb_s = ub_s = 0.0
            for e in inc_edges:
                b = self.beliefs[e]
                half_k = lk * (self._shrink_q_edge + b.rho * b.age(self.t))
                lb_s += max(self.cfg.cost_floor, b.c_hat - half_k)
                ub_s += b.c_hat + half_k
            shrunk_lb, shrunk_ub = lb_s, ub_s
        out.update(
            licensed_k=lk,
            shrink_ucb={k: self.shrink.ucb(k) for k in self.cfg.shrink_grid},
            shrink_n={k: self.shrink.n(k) for k in self.cfg.shrink_grid},
            shrunk_lb=shrunk_lb,
            shrunk_ub=shrunk_ub,
            shrunk_gap=(
                shrunk_ub - shrunk_lb if shrunk_lb == shrunk_lb else float("nan")
            ),
        )
        out.update(
            v3_regime=self.regime.diagnostics(),
            v3_formal_regime=self.formal_regime.diagnostics(),
            v3_active_sensing=self.active_sensing.stats(),
            v3_selection_events=len(self.selection_ledger.events),
            v3_selection_digest=self.selection_ledger.digest,
            v3_selection_decision_id=self._selection_audit_decision_id,
            v3_selection_audit_count=len(self._selection_audit_scores),
            v3_evidence_model_ready=self.evidence_calibrator.ready,
            v3_evidence_auto_train=len(self._evidence_train_residuals),
            v3_evidence_auto_calibration=len(self._evidence_cal_residuals),
            v3_evidence_auto_fitted=self._evidence_auto_fitted,
            v3_trajectory_calibrator_ready=self.trajectory_calibrator.ready,
            v3_trajectory_tube_valid=(
                self._last_trajectory_tube.valid
                if getattr(self, "_last_trajectory_tube", None) is not None
                else False
            ),
        )
        return out

    def round(self) -> tuple[Certificate, Edge | None]:
        """One replanning round. Returns the certificate and the sensed edge."""
        cfg = self.cfg
        self._last_cia_ub_used = False

        # Step 1-2: iterate q <-> path-length coupling once (L feeds Bonferroni).
        # Start from last known L or a Dijkstra-free guess of 1.
        self._update_online_rho()
        L_guess = getattr(self, "_last_L", 1)
        # Drift-adjusted scores can be negative when the A1 allowance exceeds
        # the observed residual.  They are valid calibration scores, but a
        # negative radius would invert an interval and can create non-positive
        # search costs (and negative cycles in the oracle).  A zero radius is
        # the sound projection because |residual| <= q + rho*age is implied by
        # |residual| <= max(q, 0) + rho*age.
        q = self._q(L_guess)
        if math.isfinite(q):
            q = max(0.0, q)
        q_eff = (q if math.isfinite(q) else 0.0) * cfg.latent_margin
        # warm-up: intervals exist but the certificate is INVALID via confidence
        self._refresh_metrics(q_eff)

        sum_aware_L = 0
        p_lb, lb = self.sp_lower.shortest_path()
        lb_edges = path_edges(p_lb)
        L = max(len(lb_edges), 1)
        if L != L_guess or cfg.age_stratify:
            # Refine once with the right Bonferroni level. In age-stratified
            # mode the path is also the evidence-overlap proxy.
            q = self._q(L, p_lb if cfg.age_stratify else None)
            if math.isfinite(q):
                q = max(0.0, q)
            q_eff = (q if math.isfinite(q) else 0.0) * cfg.latent_margin
            self._refresh_metrics(q_eff)
            p_lb, lb = self.sp_lower.shortest_path()
            lb_edges = path_edges(p_lb)
            L = max(len(lb_edges), 1)
        self._last_L = L
        # cache the full per-edge radius term for the shrink license (the width
        # the planner is pricing with this round); inf during warm-up -> the
        # license update is skipped until a finite quantile exists.
        self._shrink_q_edge = q_eff if math.isfinite(q) else math.inf
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
        mean_path = None
        mean_path_cost = math.inf
        if cfg.mean_path_execution:
            mean_path = self._mean_path()
            if mean_path is not None:
                mean_edges = path_edges(mean_path)
                if mean_edges and all(e in up for e in mean_edges):
                    mean_path_cost = sum(up[e] for e in mean_edges)
                    ub_candidates.append((mean_path_cost, mean_path))
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
            if (cfg.sum_aware_ub or cfg.cia_ub) and math.isfinite(q) and prev is not None:
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
                    if cfg.cia_ub:
                        # CIA calibrates the selected path sum at its own
                        # alpha budget. The LB budget is already reduced in
                        # _pricing_alpha(), so the two events compose by a
                        # union bound at the configured total alpha.
                        cia = self.cia_path_certificate(
                            prev, alpha=self._cia_ub_alpha()
                        )
                        c_sum = cia.ub if cia is not None else math.inf
                        if cia is not None:
                            sum_aware_L = len(pe)
                            self._last_cia_ub_used = True
                    else:
                        alpha_path = self._pricing_alpha()
                        m = self.scorer.block_quantile(alpha_path, self.t, len(pe))
                        if math.isfinite(m):
                            sum_aware_L = len(pe)
                            c_sum = (
                                sum(self.beliefs[e].c_hat for e in pe)
                                + cfg.latent_margin * m
                                + sum(self.beliefs[e].rho * self.beliefs[e].age(self.t)
                                      for e in pe)
                            )
                        else:
                            c_sum = math.inf
                    if math.isfinite(c_sum):
                        ub_candidates = [
                            (min(c, c_sum), p) if p is prev else (c, p)
                            for c, p in ub_candidates
                        ]
            # the certificate always reports the tightest bound
            ub = min(c for c, _ in ub_candidates)
            if cfg.use_kappa:
                slack = cfg.kappa_slack_frac * cfg.epsilon
                if mean_path is not None and mean_path_cost <= ub + slack:
                    # This is a certificate candidate, not an uncertified
                    # shortcut: its upper cost was included before UB was
                    # minimized, and corridor membership bounds execution.
                    incumbent = mean_path
                else:
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
        if cfg.selection_conditional and incumbent:
            # Store the complete candidate set, not just the winner.  The
            # post-selection audit API can then condition on the exact choice
            # made by the planner instead of pretending it was a random path.
            candidate_keys = [tuple(p) for _, p in ub_candidates if p]
            selected_key = tuple(incumbent)
            self._last_selection_event = self.selection_ledger.record(
                selected=selected_key,
                candidates=candidate_keys,
                context=(float(L), float(self.scorer.effective_mass(self.t))),
            )
            # Every adaptive comparison is a new selection event, even when
            # the same route wins again. Evidence collected after an earlier
            # decision is not automatically independent of this later one.
            # Reset unconditionally so a repeated winner cannot silently
            # recycle its post-selection audit stream.
            self._selection_audit_scores = []
            self._selection_audit_observation_ids = set()
            self._selection_audit_key = selected_key
            self._selection_audit_decision_id = self._last_selection_event.decision_id

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
        if cfg.shift_model == "lp":
            # LP model prices staleness as the flat mass of abruptly-changed
            # edges rho_lp per edge (the smooth-drift part is already in the
            # widened quantile), replacing the TV Delta_stale coverage penalty.
            d_stale = cfg.rho_lp
        else:
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
        if cfg.regime_recovery and not self._recovery_monitor().certificate_allowed:
            # Recovery is a hard safety gate, not merely a diagnostic.  Old
            # calibration is intentionally unavailable until fresh evidence
            # returns the manager to STABLE.
            cert.confidence = 0.0
        self._last_trajectory_tube = None
        if cfg.trajectory_tubes and incumbent:
            centers = [self.beliefs[e].c_hat for e in incumbent_edges]
            self._last_trajectory_tube = self.trajectory_tube(
                centers,
                calibrate=self.trajectory_calibrator.ready,
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
            not math.isfinite(q)
            and not certified
            and cfg.warmup_sense_per_round > 1
        ):
            n_sense = min(
                max(1, int(cfg.warmup_sense_per_round)),
                max(1, int(cfg.max_sense_per_round)),
            )
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
                    # path) with CALIBRATION (re-observe a rotating already-
                    # observed edge — only repeat observations form scores).
                    # Without the alternation, unknown-terrain warm-up chases
                    # the churning P_lb onto first-touch edges and the buffer
                    # starves (measured: 26 scores from 120 observations).
                    excluded = set(sensed_list)
                    map_candidates = [e for e in sense_edges if e not in excluded]
                    if (self._round_idx + i) % 2 == 1:
                        pick = self._warmup_calibration_edge(excluded)
                    else:
                        if map_candidates:
                            pick = map_candidates[(self._round_idx + i) % len(map_candidates)]
                        else:
                            pick = self._warmup_calibration_edge(excluded)
                elif cfg.hybrid_sensing or (
                        cfg.refine_after_certify and certified):
                    # Objective-matched portfolio: VOI on the expected-best
                    # route is useful even when the epsilon floor is formally
                    # attainable.  The certificate remains sound because the
                    # normal path-age/backstop accounting still prices every
                    # certifying edge; this only chooses additional sensing
                    # from the route-utility side of the portfolio.
                    # Recompute after each observation. A latched VOI route can
                    # become stale inside a sensing burst; the external VOI
                    # comparator reacts to updated point estimates, so a fair
                    # certificate-aware portfolio must retain that response.
                    voi_edges = [
                        e for e in self._mean_path_edges() if e in self.beliefs
                    ]
                    candidates = voi_edges or list(self.beliefs)
                    pick = max(
                        candidates,
                        key=lambda e: self.beliefs[e].rho * self.beliefs[e].age(self.t),
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
            if cfg.active_sensing:
                # Formal active sensing layer: the legacy route-critical choice
                # supplies priors, while the UCB policy learns realized gain
                # per unit sensing cost.  Keep the learner inside the
                # route-critical sensing set: allowing UB alternatives and
                # unrelated candidate edges to replace the certified choice
                # made active sensing degrade route regret in the first v3
                # full run.
                candidates = []
                for edge in dict.fromkeys(list(sense_edges) + ([pick] if pick is not None else [])):
                    b = self.beliefs[edge]
                    gain = 1e6 if not b.observed else 2.0 * b.rho * b.age(self.t)
                    candidates.append(
                        SensingAction(edge, gain, b.sense_cost, (float(L),))
                    )
                chosen = self.active_sensing.select(candidates)
                baseline = next((a for a in candidates if a.key == pick), None)
                if (
                    chosen is not None
                    and baseline is not None
                    and chosen.key != pick
                    and self.active_sensing.score(chosen)
                    > self.active_sensing.score(baseline)
                    * (1.0 + max(cfg.active_sensing_min_improvement, 0.0))
                ):
                    pick = chosen.key
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
            if cfg.active_sensing:
                # This is a conservative immediate proxy.  A downstream
                # caller can update the policy with a recomputed gap when it
                # has an expensive post-observation replanning step.
                realized_gain = max(0.0, (up_obs - lo_obs) -
                                    2.0 * self.beliefs[pick].rho *
                                    self.beliefs[pick].age(self.t))
                self.active_sensing.update(pick, realized_gain)
        self._round_idx += 1
        sensed = sensed_list[0] if sensed_list else None

        self.t += cfg.delta
        # An alarm can be triggered by the final observation in this round.
        # Re-check after sensing so the caller never receives a still-valid
        # certificate from the same round that revoked its calibration.
        if cfg.regime_recovery and not self._recovery_monitor().certificate_allowed:
            cert.confidence = 0.0
        return cert, sensed

    def cia_path_certificate(
        self, path: list[Node] | None = None, alpha: float | None = None
    ):
        """Experimental CIA group-sum certificate for a fixed path (config
        flag path_calibration="cia"; does NOT alter round()'s default).

        Builds calibration groups as disjoint blocks of the signed-deviation
        buffer (newest-first, block length = |path|), scores each block sum,
        and pulls the radius Q from the AGE-WEIGHTED empirical CDF (the drift
        retrofit: weights are cfg.rho_w ** age, inheriting the existing
        weighted-coverage argument). The reported UB is
            sum(c_hat) + Q + sum(rho * age)
        so the deterministic A1 drift term is charged on top of the conformal
        radius, exactly as the Bonferroni UB does. Returns a CIAResult with the
        interval on the path sum, or None while the buffer is too small.

        Blocks are disjoint => overlap delta = 0; the honest coverage level is
        1 - alpha (still degraded by the per-round staleness accounted upstream).
        """
        from certflow.conformal import CIAResult, weighted_group_quantile

        p = path if path is not None else self._prev_incumbent
        edges = path_edges(p) if p else []
        if not edges:
            return None
        L = len(edges)
        samples = sorted(self.scorer._signed, key=lambda s: s.t, reverse=True)
        n_blocks = len(samples) // L
        if n_blocks == 0:
            return None
        alpha_path = (
            self._cia_ub_alpha()
            if alpha is None and self.cfg.cia_ub
            else self._pricing_alpha() if alpha is None else alpha
        )
        scores, weights = [], []
        for b in range(n_blocks):
            block = samples[b * L : (b + 1) * L]
            scores.append(abs(sum(s.residual for s in block)))
            weights.append(
                min(self.cfg.rho_w ** max(0.0, self.t - s.t) for s in block)
            )
        Q = weighted_group_quantile(scores, weights, alpha_path)
        if not math.isfinite(Q):
            return None
        pred = sum(self.beliefs[e].c_hat for e in edges)
        drift = sum(self.beliefs[e].rho * self.beliefs[e].age(self.t) for e in edges)
        q_margin = self.cfg.latent_margin * Q + drift
        return CIAResult(
            lo=pred - q_margin, ub=pred + q_margin, Q=q_margin, delta=0.0,
            coverage_level=max(0.0, 1.0 - alpha_path),
        )

    def pasc_edge_radius(self, path: list[Node] | None = None) -> float | None:
        """PASC joint per-edge radius for the current path (arXiv 2605.18812).

        Returns a single radius ``Q`` such that pricing EVERY edge of an
        ``L``-edge path at ``c_hat_e +/- Q`` gives JOINT per-edge coverage
        ``>= 1 - alpha`` -- i.e. with high probability all edges are within ``Q``
        at once, in one weighted quantile, instead of the ``alpha/L`` per-edge
        Bonferroni correction ``round()`` uses. Built from disjoint length-``L``
        blocks of the absolute-score buffer, taking the per-block MAX and its
        age-weighted ``(1-alpha)`` quantile (the same drift retrofit as
        :meth:`cia_path_certificate`).

        This is PASC's genuinely useful quantity: joint per-edge validity. It does
        NOT give a tighter path-SUM bound (that sum is ``L * Q``, same order as
        Bonferroni) -- use :meth:`cia_path_certificate` for the ``sqrt(L)`` sum.
        Returns ``None`` while the buffer holds fewer than one full block.
        """
        from certflow.conformal import weighted_group_quantile

        p = path if path is not None else self._prev_incumbent
        edges = path_edges(p) if p else []
        if not edges:
            return None
        L = len(edges)
        samples = sorted(self.scorer._buf, key=lambda s: s.t, reverse=True)
        n_blocks = len(samples) // L
        if n_blocks == 0:
            return None
        alpha_path = self._pricing_alpha()
        scores, weights = [], []
        for b in range(n_blocks):
            block = samples[b * L : (b + 1) * L]
            scores.append(max(abs(s.residual) for s in block))
            weights.append(
                min(self.cfg.rho_w ** max(0.0, self.t - s.t) for s in block)
            )
        Q = weighted_group_quantile(scores, weights, alpha_path)
        return Q if math.isfinite(Q) else None

    # ------------------------------------------------------------------ v3 API
    def selection_certificate(
        self,
        path: list[Node] | None = None,
        audit_residuals: list[float] | None = None,
        point: float | None = None,
        alpha: float | None = None,
    ):
        """Certify the selected route with an independent post-selection audit.

        ``audit_residuals`` are path-level residuals collected after the route
        was selected (or from a held-out audit stream).  Reusing the ordinary
        adaptive edge buffer here is intentionally not supported: that is the
        winner's-curse failure mode this API is designed to expose.
        """
        p = path if path is not None else self._prev_incumbent
        if not p:
            return None
        selected = tuple(p)
        event = self._last_selection_event
        if event is None or event.selected != selected:
            # A caller can certify a manually supplied selected path, but the
            # returned digest explicitly records that no planner event exists.
            ledger = None
        else:
            ledger = self.selection_ledger
        if point is None:
            point = sum(self.beliefs[e].c_hat for e in path_edges(p))
        if audit_residuals is None:
            audit_residuals = list(self._selection_audit_scores)
        if alpha is None:
            alpha = self._alpha_prime_eff
        drift = sum(
            self.beliefs[e].rho * self.beliefs[e].age(self.t)
            for e in path_edges(p)
        )
        return self.selection_calibrator.certify(
            selected, point, audit_residuals, alpha, ledger=ledger,
            drift_margin=drift,
        )

    def record_selection_audit(
        self,
        residual: float,
        path: list[Node] | None = None,
        *,
        decision_id: int,
        observation_id: str,
    ) -> None:
        """Append one independently collected residual for the selected path.

        This method is the bridge for a real post-selection audit sensor.  It
        requires the exact selection decision and a unique observation ID. It
        rejects stale decisions, duplicate observations and different paths,
        preventing accidental reuse across adaptive route comparisons.
        """
        p = path if path is not None else self._prev_incumbent
        key = tuple(p) if p else None
        if key is None or self._selection_audit_key != key:
            raise ValueError("audit path does not match the current selected path")
        if self._selection_audit_decision_id != int(decision_id):
            raise ValueError("audit decision_id is stale or does not match current selection")
        identity = str(observation_id).strip()
        if not identity:
            raise ValueError("observation_id must be non-empty")
        if identity in self._selection_audit_observation_ids:
            raise ValueError("observation_id has already been used for this selection")
        value = float(residual)
        if not math.isfinite(value):
            raise ValueError("residual must be finite")
        self._selection_audit_scores.append(abs(value))
        self._selection_audit_observation_ids.add(identity)

    def _auto_evidence_update(self, edge: Edge, age: float, residual: float) -> None:
        """Collect a disjoint chronological split for the evidence model.

        This is intentionally conservative: automatic fitting happens once,
        after two equal support blocks, and the fitted model is frozen for the
        episode.  A deployment with a longer independent stream can call
        fit_evidence_model() again at an explicit split boundary.
        """
        if not self.cfg.evidence_model or self._evidence_auto_fitted:
            return
        if not math.isfinite(residual):
            return
        features = evidence_features(
            age, self.beliefs[edge].rho * age,
            getattr(self, "_last_L", 1),
            self.scorer.effective_mass(self.t), overlap=1.0,
        ).tolist()
        minimum = max(2, self.cfg.evidence_model_min_calibration)
        # Auto-fit needs a nontrivial training/calibration split, but should
        # remain usable in short diagnostic episodes as well as full runs.
        minimum = min(minimum, 20)
        if len(self._evidence_train_residuals) < minimum:
            self._evidence_train_features.append(features)
            self._evidence_train_residuals.append(abs(float(residual)))
            return
        self._evidence_cal_features.append(features)
        self._evidence_cal_residuals.append(abs(float(residual)))
        if len(self._evidence_cal_residuals) >= minimum:
            self.evidence_calibrator.fit(
                self._evidence_train_features,
                self._evidence_train_residuals,
                self._evidence_cal_features,
                self._evidence_cal_residuals,
            )
            self._evidence_auto_fitted = True

    def fit_evidence_model(
        self,
        train_features,
        train_residuals,
        calibration_features,
        calibration_residuals,
    ):
        """Fit the optional learned overlap/evidence calibrator.

        The caller supplies a proper training split and a disjoint calibration
        split.  The model learns a residual scale from observables; the final
        radius remains conformal through normalized calibration residuals.
        """
        result = self.evidence_calibrator.fit(
            train_features, train_residuals,
            calibration_features, calibration_residuals,
        )
        self._evidence_auto_fitted = True
        return result

    def evidence_radius(
        self,
        edge: Edge,
        path_length: int | None = None,
        overlap: float = 1.0,
        sensing_density: float = 0.0,
        alpha: float | None = None,
    ) -> float:
        """Return a case-dependent conformal radius for one edge.

        This is an explicit v3 opt-in API.  It does not alter the legacy
        planner pricing unless a caller deliberately uses this radius.
        """
        b = self.beliefs[edge]
        ess = self.scorer.effective_mass(self.t)
        features = evidence_features(
            b.age(self.t), b.rho * b.age(self.t),
            path_length if path_length is not None else getattr(self, "_last_L", 1),
            ess, overlap, sensing_density,
        )
        return self.evidence_calibrator.radius(
            features, self._pricing_alpha() if alpha is None else alpha
        )

    def decision_certificate(
        self,
        actions,
        objectives,
        target_risk: float | None = None,
        alpha: float | None = None,
    ):
        """Select a route/action by directly controlling supplied risk losses."""
        target = (
            self.cfg.decision_risk_target
            if target_risk is None else target_risk
        )
        if target is None:
            raise ValueError("target_risk must be supplied or configured")
        return self.decision_risk.select(actions, objectives, target, alpha)

    def fit_trajectory_calibrator(self, predicted, observed):
        """Fit the trajectory-level max-score calibrator on held-out episodes."""
        return self.trajectory_calibrator.fit(predicted, observed)

    def trajectory_tube(
        self,
        center,
        alpha: float | None = None,
        calibrate: bool = True,
    ):
        """Return a conformal trajectory tube from the v3 calibrator."""
        level = self._alpha_prime_eff if alpha is None else alpha
        if calibrate:
            return self.trajectory_calibrator.tube(center, level)
        import numpy as _np
        c = _np.asarray(center, dtype=float)
        p = self._prev_incumbent
        edges = path_edges(p) if p else []
        q = self._q(len(edges), p) if edges else math.inf
        radii = _np.asarray([
            q + self.beliefs[e].rho * self.beliefs[e].age(self.t)
            for e in edges
        ], dtype=float)
        return TrajectoryTube(c, radii, self._alpha_prime_eff, 1.0 - level,
                              "CERT-FLOW route edge-cost tube")

    def recovery_diagnostics(self) -> dict:
        """Return the v3 regime/recovery state and active-sensing ledger."""
        return {
            "regime": self.regime.diagnostics(),
            "formal_regime": self.formal_regime.diagnostics(),
            "active_sensing": self.active_sensing.stats(),
            "selection_events": len(self.selection_ledger.events),
            "selection_digest": self.selection_ledger.digest,
        }

    def _recovery_monitor(self):
        """Return the configured operational recovery gate."""
        return self.formal_regime if self.cfg.recovery_formal else self.regime

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
            pair_age = b.age(self.t)
            raw_residual = abs(obs - b.c_hat)
            score = raw_residual - b.rho * pair_age
            self._auto_evidence_update(e, pair_age, raw_residual)
            # Live validity monitor (WATCH): the weighted conformal p-value of
            # this fresh score against the CURRENT calibration buffer (before it
            # is pushed) is (super-)uniform under the weighted-exchangeability
            # null the certificate assumes. Feed it to the test martingale and
            # the Shiryaev-Roberts detector. Purely observational -- no pricing.
            if self.scorer._buf:
                w_cal = self.scorer._weights(self.t)
                cal = [s.residual for s in self.scorer._buf]
                p = conformal_p_value(score, cal, w_cal)
                if self.cfg.watch_monitor:
                    self.watch.update(p)
                    self.sr.update(p)
                    self._recent_scores.append(score)
                    if len(self._recent_scores) > self.cfg.watch_window:
                        del self._recent_scores[0]
                if self.cfg.regime_recovery:
                    monitor = self._recovery_monitor()
                    previous_state = monitor.state
                    new_state = monitor.observe_pvalue(p)
                    if (previous_state == "stable" and new_state == "alarm"):
                        # A detected regime break revokes old calibration.  The
                        # manager then requires a fresh recovery sample run
                        # before the certificate can become live again.
                        self.scorer.clear()
                        self._recent_scores.clear()
            elif self.cfg.regime_recovery:
                # During recovery the pre-alarm buffer is intentionally empty;
                # fresh observations count as non-surprising evidence until a
                # new buffer has formed, allowing the hard gate to reopen.
                monitor = self._recovery_monitor()
                monitor.observe_pvalue(1.0)
            # Test-then-tighten license (ShrinkLicense): per shrink factor k, the
            # Bernoulli outcome x_t(k) = 1{|obs - c_hat| > k * radius_t} on the
            # UNADJUSTED deviation vs the k-shrunk FULL per-edge radius
            # radius_t = q_edge + rho*age (the width the planner is pricing with,
            # cached in round()). b.c_hat/b.age here are still the pre-update
            # values, i.e. exactly the price this edge carried this moment. Feeds
            # an anytime-valid upper CS on the observed violation rate. Purely
            # observational -- no pricing, no change to the emitted certificate.
            # Skipped until a finite per-edge quantile has been priced (warm-up).
            if self.cfg.shrink_license and math.isfinite(self._shrink_q_edge):
                dev = abs(obs - b.c_hat)
                radius = self._shrink_q_edge + b.rho * b.age(self.t)
                if radius > 0.0:
                    self.shrink.update(
                        {k: 1.0 if dev > k * radius else 0.0
                         for k in self.cfg.shrink_grid}
                    )
            self.scorer.push(score, self.t)
            self.evidence_binned.push(score, self.t, b.age(self.t))
            self.scorer.push_signed(obs - b.c_hat, self.t)
            self.cal_rho_a_max = max(self.cal_rho_a_max, b.rho * b.age(self.t))
            age = b.age(self.t)
            if self.cfg.rho_mode == "online" and age >= self.cfg.delta:
                rate = abs(obs - b.c_hat) / age
                self._rate_samples.append(rate)
                local = self._rate_samples_by_edge.setdefault(e, [])
                local.append(rate)
                if len(local) > 256:
                    del local[:-256]
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
            self._oracle = self._oracle or SnapshotOracle(
                self._flat_mid, max_bytes=self.cfg.snapshot_max_bytes
            )
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
        self._heuristic = _grid_heuristic(
            self._flat_lo, self._graph_lower_cache, goal, self.cfg.cost_floor
        )
        self._mean_search = FastDijkstraSearch(self._flat_mean, start, goal)
        self._mean_flat_version = -1
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
            self.sp_lower = self._new_search(
                self._graph_lower_cache, self._flat_lo, start, goal
            )
            self.sp_upper = self._new_search(
                self._to_adj(set(self._graph_lower_cache),
                             {e: _UB_CAP for e in self.beliefs}),
                self._flat_up, start, goal
            )

    def advance_start(self, node: Node) -> None:
        """Robot moved: shift both searches' start (D* Lite km offset)."""
        self.start = node
        self.sp_lower.set_start(node)
        self.sp_upper.set_start(node)
