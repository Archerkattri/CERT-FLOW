# External-algorithm comparisons

*Head-to-head against published baselines (AD*/ARA* bound semantics, CTP-RS-style VOI, E-Graphs, networkx) — where CERT wins, where it loses, and why, with every honest negative kept.*

**Reproduce:** `scripts/run_extern_baselines.py` (full, 15 seeds); Part D via `scripts/run_repeated_queries.py`.

> **Finding —** Under staleness CERT is the only bound that stays valid where AD*-semantics intervals collapse, and objective-matched hybrid sensing turns a several-fold regret loss into a win — but on sparse queries under global drift, scratch Dijkstra beats both D* Lite persistence and E-Graphs, and we say so plainly.

**Fairness design.** Part A evaluates competing BOUND CONSTRUCTIONS from the
same belief state on a shared observation stream (neutral max_age sensing),
so only the bound semantics differ. Part B compares SENSING POLICIES inside
the same planner. Part C is the computation-only row. Part D pits CERT's
engine against E-Graphs and D* Lite on repeated queries over a drifting map.

## Part A — bound semantics under staleness (shared stream)

AD*/ARA*-style bounded-suboptimal search reports the standard claim
OPT in [c(P-hat)/w, c(P-hat)] on its current point-estimate map. Sound on
its own map; evaluated as-is against the true drifting optimum. Rows are
grouped by world and, within each world, ordered best -> worst on **validity**
(the primary axis — a route certificate is useful only if it holds):

| world | bound | n · | validity ↑ | median width ↓ |
|---|---|---:|---:|---:|
| synthetic rho=0.02 | **CERT** | 4340 | **1.000** | 16.47 |
| | AD* w=2.0 | 4340 | 0.589 | 3.73 |
| | AD* w=1.5 | 4340 | 0.589 | 2.48 |
| | AD* w=1.2 | 4340 | 0.570 | 1.24 |
| METR-LA (real) | **CERT** | 4273 | **1.000** | 46247 s |
| | AD* w=2.0 | 4273 | 0.066 | 537 s |
| | AD* w=1.5 | 4273 | 0.024 | 358 s |
| | AD* w=1.2 | 4273 | 0.018 | 179 s |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

Validity is the ranked column and is bolded; the narrow AD* widths are *not*
marked best, because narrow-and-wrong is the trap this table exposes — see
the width-trade finding below.

Findings: (1) AD*-semantics intervals are narrow and wrong — 57-59% valid
on synthetic drift, **2-7% on real traffic at any inflation**. The
mechanism is diagnosable: c(P-hat) is a MIN over noisy stale estimates
(optimistically biased), and w-inflation extends the interval downward
only — it hedges search suboptimality, never estimate optimism. The
published semantics has no channel for map error; that channel is exactly
what CERT adds. (2) The width trade is stated, not hidden: CERT's valid
intervals are 1-2 orders of magnitude wider. Narrow-and-wrong vs
wide-and-sound is the actual choice on stale maps. (3) This is a critique
of applying map-conditional bounds as route certificates under staleness,
not of AD* as a search algorithm (on these graphs exact search is instant;
anytime behavior is not the axis tested).

## Part B — sensing policies, certify-then-go regret

*10x10 unknown terrain, rho=0.02, budget 20, 15 seeds, all goal=100%. Ordered best -> worst on regret mean.*

| policy | regret mean ↓ | regret median ↓ |
|---|---:|---:|
| **hybrid (objective-matched)** | **-0.12** | **-0.20** |
| voi (CTP-RS-style, expected-route) | 0.48 | 0.47 |
| cert (gap-directed) | 2.35 | 2.22 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

The honest loss and what it taught: the CTP-RS-style VOI baseline BEATS
pure certificate-gap sensing by ~5x in this regime, because epsilon is
T2'-unattainable before budget exhaustion — gap-directed observations are
spent on certificate-relevant but route-marginal edges while VOI pours
everything into the expected-best route, and departure quality is all that
matters when the certificate cannot close. The fix is objective-matched
sensing (hybrid, now a config flag): VOI while T2' says epsilon is
unattainable, gap-directed when attainable. Hybrid beats even pure VOI
(better warm-up route knowledge from the mapping/calibration alternation)
while remaining the only policy that also produces a valid certificate.
Negative mean regret = matches the greedy clairvoyant oracle on average.

## Part C — TASP-degenerate (computation-only tightening)

Never-sense in a drifting world: 0/300 valid rounds. TASP tightens bounds
by spending computation across estimators; under drift with a single
sensing channel, computation cannot substitute for observation — there is
nothing to tighten with.

## Scoreboard summary (all head-to-head comparisons to date)

- vs Gaussian mu+z*sigma: 2.2-2.7x tighter on real traffic at equal
  coverage; edge-level Gaussian breaks 4.8-10x where CERT stays calibrated.
- vs AD*-semantics bounds: 1.000 vs 0.02-0.59 validity under staleness.
- vs CTP-RS-style VOI: lost on regret with pure gap sensing (honest),
  now wins with objective-matched hybrid (-0.12 vs 0.48) plus the
  certificate VOI lacks.
- vs freshness/info-gain/random/blind: 2-3x regret advantage (synthetic),
  lowest regret on MovingAI maps with route choice.
- vs scratch Dijkstra (engine): 9.7-23.8x faster incremental repair.
- vs TASP-degenerate / no-sensing: certification impossible without
  observation; the comparison is categorical.

## Part D — E-Graphs (RSS 2012) on repeated queries over a drifting map

`scripts/run_repeated_queries.py`, 3 seeds, POOL_SIZE=6, 50 queries/run,
drift advanced 10 time-units between queries, rho=0.02.
The named opponent (Phillips et al., "E-Graphs: Bootstrapping Planning with
Experience Graphs", RSS 2012) in the setting where it applies: a *sequence*
of planning queries on a *changing* map with start/goal pairs drawn from a
small reusable pool. `EGraphPlanner` is weighted A* (w=1.2) whose heuristic is
h_E(s) = min(h(s), min_v[h(s,v)+h_exp(v)]) plus an ARA*-style experience
incumbent that snaps onto and reuses prior returned paths (faithful grid
simplifications documented in `src/certflow/egraph.py`; its actual guarantee
is bounded suboptimality cost <= 1.2*optimal). Fairness: every bounded planner
shares one admissible Manhattan heuristic; D* Lite is our engine's reuse
answer (one persistent instance per pool entry). Timing convention matches the
Tier-1 latency doc (perf_counter around the plan call only, ms, p50/p95).
Two regimes by log-normal cost spread: "spread" sigma=0.5, "uniform"
sigma=0.05. Within each (regime, size) block, planners are ordered best -> worst
on median latency; bold marks the fastest planner in that block.

| regime · | size · | planner · | p50 (ms) ↓ | p95 (ms) ↓ | cost ratio mean ↓ | cost ratio max ↓ | mean expansions ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| spread | 20x20 | Dijkstra (scratch, optimal) | **0.1796** | **0.2724** | 1.0000 | 1.0000 | - |
| spread | 20x20 | weighted A* w=1.2 (scratch) | 0.2359 | 0.3508 | 1.0000 | 1.0000 | 235 |
| spread | 20x20 | networkx dijkstra (extern) | 0.2508 | 0.3763 | - | - | - |
| spread | 20x20 | D* Lite (persistent) | 1.6824 | 3.1701 | 1.0000 | 1.0000 | - |
| spread | 20x20 | EGraphPlanner (experience) | 3.5300 | 5.9294 | 1.0000 | 1.0000 | 234 |
| spread | 40x40 | Dijkstra (scratch, optimal) | **0.8556** | **1.3269** | 1.0000 | 1.0000 | - |
| spread | 40x40 | weighted A* w=1.2 (scratch) | 1.1044 | 1.6942 | 1.0000 | 1.0000 | 949 |
| spread | 40x40 | networkx dijkstra (extern) | 1.1742 | 1.8186 | - | - | - |
| spread | 40x40 | D* Lite (persistent) | 7.4794 | 13.1197 | 1.0000 | 1.0000 | - |
| spread | 40x40 | EGraphPlanner (experience) | 12.8636 | 58.3983 | 1.0000 | 1.0000 | 948 |
| uniform | 20x20 | Dijkstra (scratch, optimal) | **0.1803** | **0.2673** | 1.0000 | 1.0000 | - |
| uniform | 20x20 | weighted A* w=1.2 (scratch) | 0.2360 | 0.3398 | 1.0000 | 1.0000 | 243 |
| uniform | 20x20 | networkx dijkstra (extern) | 0.2503 | 0.3689 | - | - | - |
| uniform | 20x20 | D* Lite (persistent) | 1.6623 | 2.9033 | 1.0000 | 1.0000 | - |
| uniform | 20x20 | EGraphPlanner (experience) | 3.5611 | 7.3273 | 1.0000 | 1.0000 | 242 |
| uniform | 40x40 | Dijkstra (scratch, optimal) | **0.8044** | **1.4086** | 1.0000 | 1.0000 | - |
| uniform | 40x40 | weighted A* w=1.2 (scratch) | 1.0071 | 1.6093 | 1.0000 | 1.0000 | 939 |
| uniform | 40x40 | networkx dijkstra (extern) | 1.1148 | 1.8768 | - | - | - |
| uniform | 40x40 | D* Lite (persistent) | 7.5654 | 11.8262 | 1.0000 | 1.0000 | - |
| uniform | 40x40 | EGraphPlanner (experience) | 17.3080 | 67.2352 | 1.0000 | 1.0000 | 938 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

Latency columns are ranked per block. The cost ratio is identical (optimal) for
every planner that reports it — see the heuristic-collapse finding below — so
nothing is marked best there; expansions track weighted A* to within a single
node and are likewise left unmarked, because parity (not the one-node gap) is
the point.
(0 unreachable queries across all runs.)

Findings (honest, both axes):

1. **On this workload scratch Dijkstra wins outright, and we say so plainly.**
   D* Lite persistence (1.7-7.5 ms p50) *loses* to from-scratch Dijkstra
   (0.18-0.86 ms) by ~4-9x. The reason is diagnosable and is exactly the
   honest boundary from the Tier-1 doc: between queries the *entire map*
   drifts, so every persistent D* Lite instance re-applies all |E| edge
   changes and repairs a globally-affected region — incremental locality (T3)
   buys nothing when the change is not local. D* Lite is the right tool for
   *local* perturbations between *contiguous* replans (the moving-robot regime,
   where it is 9.7-23.8x faster — see tier1-latency.md), not for sparse
   queries separated by global drift.

2. **E-Graphs provides no expansion savings here, and its experience scan
   makes it the slowest by 15-60x.** Expansions track plain weighted A*
   exactly (234 vs 235, 948 vs 949) — the experience valley never forms. The
   mechanism is a genuine and somewhat surprising drift result: an *admissible*
   heuristic must lower-bound every edge cost over the whole horizon, but under
   rho=0.02 drift some edge is driven down toward the cost floor (observed
   horizon-min ~8e-4 vs ~1.0 nominal), collapsing Manhattan to ~0. With h~0,
   weighted A* degenerates to Dijkstra (hence cost ratio == 1.0000 everywhere,
   no inflation effect) and E-graphs has no informativeness to exploit. The
   per-node O(|V_exp|) experience-vertex scan is then pure overhead — that is
   the 3.5-17 ms. The "uniform" regime does not rescue this: drift collapses
   the heuristic regardless of the initial spread.

3. **Where E-Graphs *does* win is validated separately.** When the heuristic
   stays informative (near-uniform costs, no horizon-wide drift-down), the
   experience incumbent fires from the first expansion and a repeated identical
   query is solved in **0 expansions** (warm) vs 39 (cold) — see
   `tests/test_egraph.py::test_experience_reduces_expansions_on_repeated_query`.
   E-Graphs is a reuse accelerator for repeated queries under an *informative*
   heuristic; persistent global drift is precisely the condition that voids
   both its and D* Lite's reuse advantage.

4. **External speed anchor.** networkx `shortest_path_length` (dijkstra,
   weight) is the third-party reference: 0.25 ms (20x20) / 1.17 ms (40x40)
   p50 on the same updated graphs. Our engine's `dijkstra()` is 0.18 / 0.86
   ms — i.e. ~1.3-1.4x faster than a mature third-party implementation, so the
   Tier-1 "fast" claims are anchored to an external baseline, not only to our
   own code.
