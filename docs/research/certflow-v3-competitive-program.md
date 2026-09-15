# CERT-FLOW v3 competitive program

Status: active engineering plan, 2026-09-12

## Objective

Make CERT-FLOW the strongest system in its intended operating envelope by
beating its own previous full runs and the strongest relevant baselines on
the same seeds, data splits, validity target, and compute budget. A single
aggregate score is not sufficient: a narrower interval that loses coverage,
or a faster route that cannot certify, is a regression.

The program therefore reports a Pareto scorecard:

| Axis | Primary metric | Promotion rule |
| --- | --- | --- |
| Validity | empirical coverage and 95% lower confidence bound | no lower-bound failure against the declared target |
| Sharpness | median and p90 certificate gap | lower than the strongest valid baseline |
| Decision quality | route regret and goal-reaching rate | lower regret at non-inferior validity |
| Adaptation | post-shift recovery delay and false-alarm rate | faster recovery without excess false alarms |
| Sensing | regret per observation and certificate-gap reduction per cost | better than cert, max-age, random, and VOI baselines |
| Fleet | joint coverage, width, congestion-adjusted objective | better than additive and independent-agent baselines |
| Systems | p50/p95 replanning latency, peak RSS, preprocessing/customization | report separately by graph size and backend |

All claims must include seed count, warm-up policy, confidence interval, and
the exact configuration hash. Quick runs are diagnostics only; promotion uses
the full run.

## Evidence already in the repository

The prior committed full runs are the comparison floor, not a target to be
recalled from memory:

* METR-LA: CERT p75/p95 had about 93.6--94.7% valid rounds and 1.000
  observed coverage; the p75 gap median was about 4,774 seconds and p95 was
  about 8,797 seconds. Gaussian p95 had a wider gap of about 11,288 seconds.
* PEMS-BAY: CERT p95 had about 96.7% valid rounds, 1.000 coverage, and a gap
  median around 1,067 seconds; Gaussian p95 was around 1,570 seconds.
* The previous full real-traffic sensing run reported median regret about
  38.3 for hybrid sensing versus 65.1 for pure certificate-gap sensing.
* The previous width-attack run found full-run gap ratios of approximately
  0.764 for sum-aware UB, 0.734 for CIA-UB, and 0.376 for the separately
  licensed shadow shrink. The shadow shrink is not a replacement certificate.
* Previous road-routing results used a full graph and a numba-enabled fast
  path. Reduced or numba-disabled reruns must not be presented as direct wins
  over those numbers.

The latest full v3 scorecards are stored separately as
`results/v3_scorecard/p95-full.json`, `p75-full.json`, and
`synthetic-full.json`. The promoted traffic configuration uses robust
per-edge online rho, hybrid sensing, kappa hysteresis, adaptive sensing,
sum-aware UB, a 12-edge distinct-edge calibration warm-up, and
`latent_margin=1.05`.

### Latest full scorecard (10 seeds)

The p95 traffic replay now reports these aggregate medians:

| Dataset / condition | valid fraction | coverage | median gap | median regret |
| --- | ---: | ---: | ---: | ---: |
| METR-LA legacy | 0.948 | 1.000 | 7,323 | 60.35 |
| METR-LA recommended | 0.992 | 0.999 | 3,224 | 43.96 |
| METR-LA Gaussian | 0.948 | 1.000 | 9,323 | 61.73 |
| PEMS-BAY legacy | 0.965 | 1.000 | 893 | 9.06 |
| PEMS-BAY recommended | 0.993 | 0.981 | 562 | 8.40 |
| PEMS-BAY Gaussian | 0.962 | 1.000 | 1,349 | 11.73 |

The p75 evidence/given variants improve METR-LA to roughly 2,947/3,622
seconds median gap with 1.000 coverage. On PEMS-BAY, p75 evidence reaches
8.12 median regret and 554 median gap versus 8.22 and 585 for the p75
baseline. The synthetic recommended lane now reaches 0.735 median regret
and 10.40 median gap versus legacy 0.818 and 23.66; the evidence/recovery
portfolio members remain available when decision quality is prioritized.

These are empirical replay results, not proofs of conditional validity. The
small residual coverage shortfall on PEMS-BAY is above the declared 0.8 target
but below the prior run's observed 1.000; it remains a tradeoff to monitor. The
scorecard also exposed and fixed a correctness defect: negative
drift-adjusted residual quantiles were being used as negative radii, which
could create non-positive search costs and a late-run graph-search blow-up.
Finite radii are now projected to zero before metric construction, with a
regression test and a previously failing 288-round p75 evidence replay.

The executable promotion audit is `scripts/check_v3_gate.py`. It currently
passes the full traffic, synthetic, width-attack, MovingAI, routing, and
correlated-fleet release checks; it now also passes the 60x60 latency budget
after the exact flat-Dijkstra backend and warm-up scheduling optimizations.
Specialist tradeoffs remain explicit rather than being converted into a single
score.

### Systems measurements

The full scale run uses 3 seeds × 150 rounds at each size with numba enabled.
At 60×60, the current 12-edge recommended configuration measures 12.2 ms
p50/22.3 ms p95 with 85.3% validity across 3 seeds × 150 rounds, versus
31.9 ms p95 and 0% validity for the default configuration. The large-grid
`auto` backend uses exact fresh flat Dijkstra queries where D* Lite queue
repair is slower; generic graphs retain D* Lite. This is an explicit
quality/latency policy, not a hidden aggregate.

On the official 264,346-node NY graph, numba-enabled ALT preprocessing is 3.0
s, FastDijkstra query p50/p95 is 33.0/61.5 ms, and ALT is 14.4/50.3 ms, with
0/100 mismatches. On the 50k-node CH subgraph, CH cost-only query p50/p95 is
0.0936/0.162 ms, path-unpack is 0.908/2.229 ms, and bounded-potential queries
remain exact under ±20% costs with 0/100 mismatches. The cost-only query beats
the harness's 0.110 ms published p50 reference; path-unpack and
bounded-potential queries remain slower, so the routing claim is split by
query type.

The MovingAI smoke suite (4 seeds, 150 rounds) reached 100% of goals with
1.000 certificate coverage on DAO, Berlin, and maze for the cert policy. Mean
regret was 5.30, 8.93, and 0.11 respectively; no-cert controls were lower in
some maps because they do not pay for certification, so they are not
strictly-dominated decision baselines.

## Research-derived design decisions

### 1. Selection is a first-class conditioning event

Post-selection validity cannot be inferred from a marginal certificate over
candidate paths. CAP (Calibration after Adaptive Pick) gives a direct model:
use a selection rule, then construct a calibration set for the selected unit
from an appropriate historical or independent stream, with false-coverage
statement-rate control. CERT-FLOW's selection ledger and independent audit
stream are the first implementation of this boundary. The benchmark must
include selected-hardest-quintile coverage and must fail any configuration
that silently reuses selection data.

### 2. Shift adaptation must be compared on both width and recovery

ACI provides an online coverage response under shift, while later arbitrary-
shift work tunes its step size over time. COP uses an estimated score CDF to
reduce conservative widths when predictable structure exists while retaining
coverage protection when the estimate is inaccurate. The next upgrade is a
portfolio of fixed-step, scale-free, and pattern-informed score updates with
an alarm/recovery gate. Selection of the portfolio member must occur on a
held-out stream or be covered by a selection-conditional audit.

### 3. Sensing should use a small number of adaptive rounds

Informative path planning with limited adaptivity shows a useful engineering
tradeoff: recompute a few times per sensing round rather than after every
observation. CERT-FLOW's active sensing layer will benchmark k-round sensing
against fully reactive VOI, max-age, and random policies under the same
observation budget. The win condition is regret/cost, not raw sensing count.

### 4. Dynamic routing needs a customization/query split

Customizable Contraction Hierarchies (CCH) and related dynamic routing work
separate expensive topology preprocessing from fast edge-weight
customization and queries. CERT-FLOW already exposes certificate-gated graph
preprocessing; the next routing milestone is to make customization incremental
and benchmark it beside Dijkstra, ALT, and CH on the official DIMACS graphs.

### 5. Fleet validity is joint, not a sum of optimistic individual claims

The additive team certificate remains the independent/shared-store baseline.
The joint fleet calibrator prices one episode-level residual vector and adds a
deterministic congestion penalty. Its evaluation must include correlated agent
residuals and shared-edge load; independent-agent-only results are not enough.

## Implementation workstreams

1. **Reproducible scorecard.** Add one runner that executes paired seeds over
   synthetic drift, METR-LA, PEMS-BAY, MovingAI, and DIMACS subsets. Persist
   raw per-seed rows, aggregate metrics, confidence intervals, environment
   information, and git revision.
2. **Certificate portfolio.** Compare Bonferroni, sum-aware, CIA, PASC, and
   evidence-conditioned pricing behind one interface. Require freshness and
   support gates before a tighter candidate can replace the ordinary bound.
3. **Selection-safe upgrades.** Expand independent post-selection auditing,
   score selected-hardest groups, and add an explicit “not certified” result
   when audit support is insufficient.
4. **Shift/recovery.** Add a pattern-informed online score expert and a
   held-out expert-selection ledger; couple it to WATCH and the recovery
   manager so a detected shift revokes the learned width immediately.
5. **Decision/risk layer.** Benchmark direct risk-controlled actions and
   trajectory tubes with objective regret, not only interval validity.
6. **Active sensing and fleet.** Add limited-adaptivity schedules, correlated
   fleet episodes, and congestion stress tests with additive/joint baselines.
7. **Routing systems.** Restore the full numba/landmark configuration for
   apples-to-apples comparison, then add incremental customization and report
   preprocessing, customization, query, and path-unpack costs separately.
8. **Adversarial release gate.** Run heavy-tailed, skewed, correlated,
   calibration-shift, post-selection, and width-attack suites. A method is
   promoted only if its declared validity scope survives these tests.

## Acceptance gates

The program is complete only when the full scorecard shows, for every promoted
regime:

* coverage lower confidence bound meets the declared target;
* median and p90 gap beat the prior full run or the strongest valid baseline,
  with the tradeoff explicitly shown when they cannot both improve;
* regret beats the strongest non-certified competitor at matched sensing
  spend, or CERT-FLOW retains validity at a statistically indistinguishable
  regret with a materially tighter certificate;
* recovery delay and false alarms improve or remain within the declared
  budget;
* fleet and routing claims hold on correlated/shared-resource stress;
* no quick-only result is used as a release claim.

If a universal win is impossible because objectives conflict, the release must
say so and ship a regime-aware selector with auditable gates rather than hide
the tradeoff in one average.

## Sources

* Bao, Huo, Ren, Zou, “CAP: A General Algorithm for Online Selective
  Conformal Prediction with FCR Control,” JMLR 26 (2025),
  https://www.jmlr.org/papers/v26/24-0452.html
* Gibbs and Candès, “Conformal Inference for Online Prediction with Arbitrary
  Distribution Shifts,” JMLR 25 (2024),
  https://www.jmlr.org/beta/papers/v25/22-1218.html
* Hu, Wu, Xia, Zou, “Distribution-Informed Online Conformal Prediction,”
  ICLR 2026, https://openreview.net/pdf?id=I69SaLbwqZ
* Tan, Ghuge, Nagarajan, “Informative Path Planning with Limited Adaptivity,”
  AISTATS 2024, https://proceedings.mlr.press/v238/tan24a.html
* Dibbelt, Strasser, Wagner, “Customizable Contraction Hierarchies,”
  https://arxiv.org/abs/1402.0402
* Cao et al., “CAtNIPP: Context-Aware Attention-based Network for Informative
  Path Planning,” CoRL 2023, https://proceedings.mlr.press/v205/cao23b.html

## v4 work-in-progress update (2026-09-13)

The first v4 closure pass landed finite-family group calibration with sparse
group abstention, a formal e-value sequential alarm/recovery gate, and a fresh
post-selection audit artifact. The focused upgrade suite is 22 tests and the
full repository suite is 317 passed with one pre-existing NumPy warning.

The refreshed full scorecards pass the strict release checker. The
decision-quality battery is complete (15 seeds, 600 rounds, all goals
reached): CERT-FLOW hybrid regret is 0.272 versus VOI 0.486, max-age 4.76, and
max-width 6.35. The p75 PEMS specialist wins both median gap (547.7 vs 584.8)
and median regret (7.90 vs 8.22) against its p75 baseline, with coverage LCB
0.946.

The v4 real-traffic audit is now complete on METR-LA and PEMS-BAY: selected
path audit coverage is 1.000 with minimum LCB 0.842; finite edge-group audits
also pass with minimum LCBs 0.804 and 0.822. The matched shift-recovery run
shows the formal SR-backed gate detects in median 23 rounds, re-enters in 10,
and returns to a valid certificate in 2, improving post-shift edge coverage
over ACI (0.954 vs 0.937) with no pre-shift false alarms. The fresh-evidence
gate was then reduced from 10 to 5 samples; valid re-entry is 5 rounds. A
fixed multi-epsilon SR mixture was tested but rejected because its full-run
detection median increased to 26.5 rounds.

The new full fleet/sensing audit uses 12 seeds × 600 rounds and all six
budgeted policies. Hybrid regret is 0.935/0.340/-0.294 at budgets 10/20/40,
beating the strongest control at each budget (1.056/0.645/0.167); every
certificate-bearing cell has coverage LCB ≥ 0.80. The strict checker now
reports 76/76 gates passing. Under an abrupt distribution jump, the honest
response still widens and abstains until new calibration is available, but the
fresh-evidence portion is now shorter without weakening the formal alarm
boundary or measured coverage.
