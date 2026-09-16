# CERT-FLOW mathematical review and runtime trace

Status: internal review boundary, 2026-09-15. This document states the
strongest claims supported by the current implementation. It is not an
independent proof review and does not upgrade an empirical audit into a
conditional-coverage theorem.

## Random objects and information flow

Let `G=(V,E)` be a finite directed graph. At planner round `t`, edge `e` has a
latent cost `C_e(t)`, a noisy observation `Y_e(t)` when sensed, and a belief
center `c_hat_e(t)`. The planner's filtration `F_t` contains the graph, all
past sensing choices and observations, internal randomization, fitted/frozen
calibration state, route comparisons, and the complete selection ledger up to
the current decision. The selected route `P_t`, its candidate family and the
sensing action are `F_t`-measurable.

For a frozen selection time `tau`, a post-selection audit stream has route
scores `S_1,...,S_n,S_{n+1}`. The implemented selected-route certificate needs
these scores to be exchangeable conditional on `F_tau`, or to satisfy a
separately justified weighted-exchangeability model. Merely observing the
same route after another adaptive comparison does not establish this. The
online API therefore starts a new audit epoch at every comparison, even when
the same route wins again.

The default weighted edge scorer is motivated by conformal methods beyond
exchangeability, but recency weighting is not by itself a theorem for an
arbitrary traffic process. The relevant assumptions must be declared and
audited. See [Barber et al., *Conformal prediction beyond exchangeability*](https://arxiv.org/abs/2202.13415)
and [Tibshirani et al., *Conformal prediction under covariate shift*](https://arxiv.org/abs/1904.06019).

## The theorem the runtime actually supports

### Deterministic finite-graph witness

For one frozen round, suppose every edge used to form the optimistic
shortest-path lower bound satisfies `ell_e <= C_e`, and every edge of a
reported incumbent candidate satisfies `C_e <= u_e`. Let

- `LB = min_P sum_{e in P} ell_e`, and
- `UB = min_{P in C} sum_{e in P} u_e` for the finite reported candidate set.

Then `LB <= OPT <= UB`. If the returned candidate has upper cost `UB` and
`UB-LB <= epsilon`, its true cost is at most `OPT+epsilon`. This is a
pointwise implication, not a probabilistic statement.

A probability statement follows only after proving a simultaneous event that
covers the lower-bound search space and the returned upper path at the stated
risk. The current path-wise `alpha/L` allocation does not automatically cover
every edge that could define the global optimistic lower path. Therefore the
default runtime must not claim a global route-optimum theorem from path-local
coverage alone. `RouteWitness` and `RiskLedger` expose the objects needed for
a restricted candidate-subgraph proof, but they are not yet wired into a
completed sequential construction.

### Frozen selected-route certificate

Condition on `F_tau` and a frozen selected route `P_tau`. If the post-selection
audit scores and the next score are exchangeable under that conditional law,
the split-conformal order statistic used by
`SelectionConditionalCalibrator` gives the usual finite-sample marginal
coverage for the next score, conditional on the recorded selection history.
A declared nonnegative drift margin may then enlarge the interval. This is a
fixed-selection, next-audit statement; it is not an always-valid confidence
sequence and it does not survive arbitrary optional stopping or audit reuse.

Universal finite-sample conditional coverage is not claimed. Finite declared
groups may receive simultaneous guarantees under their own support,
calibration and Bonferroni assumptions, consistent with the restricted
conditional setting described by [Gibbs, Cherian and Candès](https://arxiv.org/abs/2305.12616).

## Risk allocation and failure behavior

| Runtime object | Risk scope | Failure behavior |
|---|---|---|
| `ConformalScorer` and path pricing | marginal/path-local edge event under the configured shift model | nonfinite quantile, inadequate mass or stale support keeps the certificate invalid |
| CIA incumbent upper bound | separate configured alpha slice, freshness-gated | unavailable or stale CIA is ignored; the ordinary upper candidate remains |
| `SelectionConditionalCalibrator` | separate frozen-selection audit alpha | too few independent scores returns confidence zero and infinite radius |
| finite group calibrator | alpha divided over the declared finite family | absent/unsupported groups are invalid rather than pooled silently |
| `RiskLedger` | explicit finite spending record | overspend raises; it does not authorize unrecorded sequential reuse |

The selection-audit alpha is not automatically composed with every default
planner risk. A publication theorem must state the joint allocation and prove
the simultaneous event; the ledger digest alone is an audit trail, not that
proof.

## Runtime-to-proof trace

| Mathematical object | Implementation | Auditable evidence |
|---|---|---|
| `F_t` selection history | `SelectionLedger`, `SelectionEvent` | decision ID, selected route, full candidate tuple, context and digest |
| frozen route and epoch | `CertPlanner._last_selection_event` and audit decision ID | stale decision IDs and path mismatches are rejected |
| distinct audit observations | `CertPlanner.record_selection_audit` | mandatory unique observation ID; duplicates are rejected |
| path point estimate and drift allowance | `CertPlanner.selection_certificate` | belief sum, per-edge age/rho margin and independent audit count |
| optimistic/upper route witness | `RouteWitness` | route, epoch, bounds, observation IDs, rule and risk totals |
| alpha spending | `RiskLedger` | append-only spends, hard total cap and deterministic digest |
| adaptive sensing | `select_witness_observations`, `ActiveSensingPolicy` | deterministic challenger reserve and policy diagnostics; no theorem inferred |
| locked external endpoint | `results/protocols/independent-day-map-v1.json` | data/map hashes, disjoint calendar/map roles and `LOCKED_UNRUN` status |

## Adversarial checks

`tests/test_upgrades.py` checks that duplicate post-selection observations,
wrong paths and stale decision IDs fail; that a repeated adaptive comparison
clears prior audit support; and that the active-sensing path remains explicit.
`tests/test_independent_protocol.py` checks disjoint day windows, distinct map
identities and the locked/unrun endpoint state. Existing CIA tests exercise
stale-calibration undercoverage and the freshness/weighting response.

## Claim replacement

The supported statement is:

> CERT-FLOW emits an auditable deterministic route witness. Its probabilistic
> interpretation is marginal and assumption-bound. A selected route can be
> audited with a separate frozen post-selection stream, and declared finite
> groups can be audited with separate group calibration. The software does not
> provide universal conditional coverage, an always-valid sequential
> selected-route certificate, or a global route-optimum theorem from
> path-local calibration.

Independent proof review and execution of the locked day/map endpoint remain
release gates.
