# Spatial-predictor study: gain from neighbor-aware cost prediction

*Can a spatially correlated point predictor shrink CERT's certified interval width, and does that offline gain survive deployment? Offline: large at long staleness. Deployed under route-focused sensing: it does not pay — it ships opt-in for dense / fixed-sensor regimes.*

**Reproduce:** `scripts/study_spatial_predictor.py` (pure numpy/pandas, ~6 s; raw numbers in `results/spatial_predictor/study.json`)

> **Finding —** A neighbor-regression predictor sharply cuts the residual conformal width at long staleness *offline, assuming neighbors are observed at `t`* (see the headline ratio table). Integrated into the planner, route-focused sensing leaves stale edges with stale neighborhoods, the gain evaporates, and the win it does extract is paid back as lower claimed confidence (coverage holds — soundness intact). Disposition: ship **opt-in**; the payoff regime is a continuously-reporting fixed sensor network, not denser robot sensing. The age-matched-retraining rescue hypothesis was tested and **refuted**.

**Question.** CERT predicts each stale edge's current cost as
last-observation-carried-forward (LOCF). The conformal layer calibrates the
residuals of *that* predictor, so any point predictor with smaller residuals
tightens the certified interval at unchanged coverage. How much can a
*spatially correlated* predictor — one that exploits fresher observations at
neighboring sensors — shrink the residual P90 (the quantity that drives the
conformal width at our levels)?

**Design (offline, no planner).** Predict sensor `i`'s current speed at age
`a` bins (5 min/bin) from data available under staleness:

- **P0 last-value** — `speed_i(t-a)` (what CERT uses today).
- **P1 neighbor-delta** — `speed_i(t-a) + mean_j[speed_j(t) - speed_j(t-a)]`
  over neighbors `j` within 3 km observed fresh at `t` (propagate observed change).
- **P2 neighbor-regression** — per-sensor ridge (λ=1) of `speed_i(t)` on
  `[speed_i(t-a), mean_j speed_j(t)]`, fit on the first 60 % of the recording
  and evaluated on the last 40 % (no leakage).

Fresh neighbors are assumed observed *at* `t` (the planner's sensing makes some
edges fresh while a stale edge is being certified): this is the realistic
**upper bound** on the spatial gain. A half-fresh sensitivity (only half of
each sensor's neighbors fresh, chosen at random) is reported alongside.
Evaluated on every 4th time bin, all sensors. Metric: MAE and P90 of the
absolute residual; headline is the interval-width ratio P1/P0 and P2/P0 at P90
(P90 is the symmetric conformal quantile that sets the width).

## Coverage of the neighbor structure (honest accounting)

| dataset · | sensors · | fraction with ≥1 neighbor ≤3 km ↑ | isolated ↓ | mean degree · |
|-----------|----------:|----------------------------------:|-----------:|--------------:|
| METR-LA   |       207 |                         **0.990** |      **2** |           7.8 |
| PEMS-BAY  |       325 |                             0.960 |         13 |           6.5 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

Spatial prediction is **only available to the 96–99 % of sensors that have a
neighbor within 3 km**. Isolated sensors fall back to P0 unconditionally; their
P0 residuals are reported in the `iso_p90` / `iso_MAE` columns below for
reference (the predictors do not change them).

## METR-LA — all fresh neighbors

Rows sweep edge age (informational `·`); the best value in each metric column is in bold.

| age · | P0 P90 ↓ | P1 P90 ↓ | P2 P90 ↓ | P1/P0 ↓ | P2/P0 ↓ | P0 MAE ↓ | P1 MAE ↓ | P2 MAE ↓ | iso P90 ↓ | iso MAE ↓ |
|------:|---------:|---------:|---------:|--------:|--------:|---------:|---------:|---------:|----------:|----------:|
|     1 |**6.375** |**6.521** |**5.967** |   1.023 |   0.936 |**2.489** |**2.666** |**2.464** | **7.674** | **3.267** |
|     3 |    7.653 |    7.900 |    7.229 |   1.032 |   0.945 |    3.131 |    3.312 |    3.080 |     9.946 |     4.115 |
|     6 |    8.768 |    9.264 |    8.657 |   1.057 |   0.987 |    3.741 |    3.840 |    3.593 |    12.329 |     4.906 |
|    12 |   11.667 |   11.736 |   10.508 |   1.006 |   0.901 |    4.814 |    4.649 |    4.132 |    15.250 |     5.863 |
|    24 |   19.875 |   15.194 |   11.592 |   0.764 |   0.583 |    6.529 |    5.682 |    4.506 |    19.576 |     7.266 |
|    48 |   30.250 |   18.008 |   11.830 |**0.595**|**0.391**|    8.582 |    6.756 |    4.628 |    22.805 |     8.913 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

## PEMS-BAY — all fresh neighbors (replication)

Rows sweep edge age (informational `·`); the best value in each metric column is in bold.

| age · | P0 P90 ↓ | P1 P90 ↓ | P2 P90 ↓ | P1/P0 ↓ | P2/P0 ↓ | P0 MAE ↓ | P1 MAE ↓ | P2 MAE ↓ | iso P90 ↓ | iso MAE ↓ |
|------:|---------:|---------:|---------:|--------:|--------:|---------:|---------:|---------:|----------:|----------:|
|     1 |**2.100** |**2.175** |**2.141** |   1.036 |   1.019 |**1.002** |**1.012** |**1.004** | **1.900** | **0.928** |
|     3 |    3.200 |    3.433 |    3.519 |   1.073 |   1.100 |    1.609 |    1.560 |    1.599 |     2.700 |     1.387 |
|     6 |    4.500 |    4.780 |    4.816 |   1.062 |   1.070 |    2.240 |    2.061 |    2.084 |     3.600 |     1.872 |
|    12 |    6.900 |    6.717 |    5.930 |   0.973 |   0.859 |    3.168 |    2.646 |    2.478 |     5.120 |     2.582 |
|    24 |   12.800 |    9.584 |    6.497 |   0.749 |   0.508 |    4.700 |    3.421 |    2.725 |     8.700 |     3.730 |
|    48 |   23.700 |   12.225 |    6.541 |**0.516**|**0.276**|    6.752 |    4.286 |    2.814 |    15.100 |     5.224 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

## Sensitivity: only HALF the neighbors fresh

Rows sweep dataset × age (informational `·`); the best (lowest) ratio in each column is in bold.

| dataset · | age · | P1/P0 ↓ | P2/P0 ↓ |
|-----------|------:|--------:|--------:|
| METR-LA   |     6 |   1.158 |   0.985 |
| METR-LA   |    12 |   1.103 |   0.904 |
| METR-LA   |    24 |   0.822 |   0.596 |
| METR-LA   |    48 |   0.646 |   0.411 |
| PEMS-BAY  |     6 |   1.156 |   1.073 |
| PEMS-BAY  |    12 |   1.049 |   0.879 |
| PEMS-BAY  |    24 |   0.814 |   0.532 |
| PEMS-BAY  |    48 |**0.567**|**0.292**|

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

Halving the fresh-neighbor set leaves **P2 essentially unchanged** (it leans on
the regression's own `speed_i(t-a)` term and a noisier-but-still-informative
neighbor mean), while it noticeably degrades P1 at short ages. P2 is the robust
choice.

## Headline width-ratio numbers (P2 vs P0, at P90)

Rows sweep edge age (informational `·`); the strongest result in each column is in bold.

| age (bins) · | METR-LA P2/P0 ↓ | PEMS-BAY P2/P0 ↓ | METR-LA improvement ↑ | PEMS-BAY improvement ↑ |
|-------------:|----------------:|-----------------:|----------------------:|-----------------------:|
|            6 |           0.987 |            1.070 |                +1.3 % |                −7.0 % |
|           12 |           0.901 |            0.859 |                +9.9 % |               +14.1 % |
|           24 |           0.583 |            0.508 |               +41.7 % |               +49.2 % |
|           48 |       **0.391** |        **0.276** |            **+58.6 %**|            **+72.4 %** |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

## Interpretation (honest)

The spatial gain is **strongly age-dependent and concentrated at long
staleness**. At small ages the stale value is already nearly the right answer,
so P1's injected neighbor change is pure noise (ratio >1 — it *hurts*) and P2 is
roughly neutral; the calibration threshold of ≥15 % P90 improvement is met only
at age ≥12 (PEMS-BAY) / age ~24 (METR-LA), not across the whole 6–24 band. But
at the upper end of the operational band the gain is large and replicates
cleanly across both datasets: P2 cuts the residual P90 by ~42 % (METR-LA) and
~49 % (PEMS-BAY) at age 24 and by ~59–72 % at age 48, and this survives the
half-fresh-neighbor stress almost intact. P1 is not worth integrating — it is
dominated by P2 everywhere and is actively harmful when edges are fresh. The
practical caveat is coverage: 1–4 % of sensors are isolated and gain nothing,
and the headline assumes neighbors are observed at `t`, which only holds for
edges the planner has chosen to sense.

**Recommendation: integrate P2 (neighbor-regression), gated on age.** It clears
the ≥15 % bar decisively at the ages where intervals are widest and certificates
are hardest to issue (age ≥12), is harmless-to-helpful elsewhere, degrades
gracefully under partial freshness, and replicates across datasets. Apply it
only when a fresh neighbor exists; fall back to LOCF (P0) otherwise so nothing
regresses for isolated/unsensed edges.

## Integration sketch

Nothing in the conformal machinery changes — the predictor is a drop-in for the
point estimate `c_hat`:

1. **Predictor swap.** Where CERT currently sets `c_hat(e, t) = last_obs(e)`,
   instead use the age-gated P2 estimate when edge `e`'s tail sensor has ≥1
   neighbor with a fresh observation at `t`:
   `c_hat(e,t) = w0 + w1·last_obs(e) + w2·mean_fresh_neighbor_speed(t)`
   (converted speed→travel-time via the edge's fixed distance). The per-sensor
   ridge weights `(w0,w1,w2)` are fit offline per age bucket and shipped as a
   small lookup. Fall back to `last_obs(e)` when no fresh neighbor exists.
2. **Conformal layer unchanged.** Scores are residuals of whatever `c_hat`
   produces: `s_t = |observed_cost − c_hat|`. Calibrating against the new
   (smaller) residuals yields a smaller quantile and hence a tighter
   `[lb, ub] = c_hat ± q̂`, automatically and at the same coverage level — this
   is exactly the conformal "wrap any predictor" guarantee. No change to the
   quantile estimator, the drift/staleness bookkeeping, or the D* Lite search.
3. **What to watch.** The empirical A1 drift-rate bound `rho_true` is defined on
   the cost series and is independent of `c_hat`, so it is untouched; only the
   width shrinks. Recompute the per-edge calibration set after the swap.

## Integration design note (queued)

The drop-in swap (predictor replaces c_hat; conformal recalibrates) has one
soundness-sensitive subtlety the study surfaces: predictor residual scale is
strongly AGE-DEPENDENT (near-LOCF at age<12, much tighter at 24-48). A
single pooled score buffer would mix age regimes and miscalibrate both;
sound integration needs age-binned calibration or age-normalized scores
(score / expected-scale(age)), which touches the certificate substrate.
Queued as the next planner version rather than rushed: the offline gain is
established and replicated; the integration must preserve T1 semantics.

## Integration outcome (measured, honest negative for this sensing pattern)

Integrated as predictor mode (age-binned conformal widths, per-bin annealing
charged to the claim, per-edge fallback chain; CertPlanner(..., predictor=...)).
Full-day METR-LA validation (4 seeds, val windows past the training region,
online-rho baseline). Rows are ranked best → worst on the operational primary
metric (gap median, lower is better); coverage is tied so soundness
does not separate them.

| condition       | coverage ↑ | gap median ↓ | mean confidence ↑ | valid rounds ↑          |
|-----------------|-----------:|-------------:|------------------:|------------------------:|
| predictor off   |      1.000 |   **3073 s** |         **0.557** |                **1022** |
| predictor on    |      1.000 |       3371 s |             0.446 | 873 (38,672 edge-pricings) |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

The offline +42-49% was an upper bound assuming neighbors observed AT t.
Deployed, route-focused sensing leaves stale edges with stale neighborhoods
(the same optimism-geometry as the churn analysis), semi-stale neighbor
inputs inflate predictor residuals, and the online-rho model path — itself
2.4x tighter than worst-case — wins. Soundness held throughout (coverage
1.000; the per-bin annealing charge shows up honestly as lower claimed
confidence). The mode ships opt-in: it should pay where observations are
dense (multi-robot, higher sensing rates, fixed sensor networks reporting
continuously), which is exactly the regime the offline study simulates.
Three integration bugs found en route are fixed regardless (freshest-edge
neighbor lookup; per-bin warm-up starvation; claim accounting).

## Dense-sensing regime test (final disposition)

The "pays under dense observation" claim was tested at k in {1,4,8}
observations/round on METR-LA (docs/results/feature-regimes.md): at k=8 the
gap improves only -2.3% (an order of magnitude below the offline upper
bound), k=4 is flat-to-worse, and the apparent k=1 win is a valid-round
selection artifact. DOWNGRADED CLAIM: the predictor's payoff regime is a
continuously-reporting fixed sensor network (all neighbors fresh at query
time — the offline study's literal setting), not merely denser robot
sensing. It remains shipped opt-in with this disposition.

## Age-matched retraining

**Hypothesis under test.** The deployed loss was blamed on a train/deploy
distribution mismatch: P2 was trained on neighbor means observed *at* time `t`
but deployed with *semi-stale* neighbors (route-focused sensing leaves stale
edges with stale neighborhoods). If that mismatch is the cause, training on the
deployment distribution — neighbor means at realistic ages `b in {0,6,12}` bins,
with `b` supplied as a third regression feature — should recover the gain the
fresh-at-`t` training threw away.

**Implementation.** `fit_spatial_predictor(age_matched=True)` (new opt-in
parameter; `False` default preserves the original training byte-for-byte). Each
training pair samples both a target age `a in {12,24,48}` and a neighbor age
`b in {0,6,12}`; features are `[stale own speed at t-a, neighbor mean at t-b, b]`
plus a ridge intercept (4 terms). At runtime the predictor reads the actual
freshest-used neighbor age, snaps it to the `{0,6,12}` ladder, and feeds it as
the `b` feature; the fresh-age gate widens from 6 to 12 bins so semi-stale
neighbors are kept instead of discarded.
Driver: `scripts/run_predictor_retrain.py` (offline mechanism check + planner
Experiment A replicated exactly: k in {1,4,8} x {off, fresh-trained,
age-matched}; 6 seeds x 200 rounds; `offset_base_bins=20000`).

### Offline mechanism check (held-out region, residual P90 in mph)

Both models fit on `[:18000]`; evaluated on the held-out tail with neighbors
read at the deployment-realistic age `b` (the regime the fresh model never saw).
Rows sweep neighbor age `b` (informational `·`); best value in each column is in bold.

| nbr age b (bins) · | fresh-trained P90 ↓ | age-matched P90 ↓ | am / fresh ↓ |
|-------------------:|--------------------:|------------------:|-------------:|
|                  6 |          **14.428** |        **14.399** |        0.998 |
|                 12 |              16.158 |            15.991 |    **0.990** |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

Age-matching cuts the held-out residual P90 by **0.2 % (b=6) and 1.0 % (b=12)** —
essentially nothing. The neighbor-age feature carries almost no extra signal at
these ages: at a 6–12-bin neighbor staleness the neighbor mean is already a poor
proxy for the current own-speed, and telling the model how stale it is does not
sharpen the prediction. **The mechanism is inert before the planner even runs.**

### Planner table (Experiment A, METR-LA, 6 seeds x 200 rounds)

Rows are grouped by sensing budget `k` (the controlled informational axis `·`, not
globally reordered); within the table the best value in each ranked column is in
bold. Note the densest-budget `pred=off` optima on valid / mean conf are
confounded by the sensing budget, not a predictor effect — read them per-`k` block.

| condition             | valid ↑  | coverage ↑ | gap median (s) ↓ | mean conf ↑ | pred_used_rounds · |
|-----------------------|---------:|-----------:|-----------------:|------------:|-------------------:|
| k=1, pred=off         |     1035 |      1.000 |           3092.6 |       0.562 |                0.0 |
| k=1, pred=fresh       |      908 |      1.000 |           2978.8 |       0.425 |             5085.8 |
| k=1, pred=age-matched |      782 |      1.000 |           2847.2 |       0.368 |             6213.2 |
| k=4, pred=off         |     1160 |      1.000 |           3153.8 |       0.653 |                0.0 |
| k=4, pred=fresh       |     1154 |      1.000 |           2851.4 |       0.581 |            16877.2 |
| k=4, pred=age-matched |     1142 |      1.000 |       **2786.7** |       0.563 |            19778.8 |
| k=8, pred=off         | **1184** |      1.000 |           3293.6 |   **0.660** |                0.0 |
| k=8, pred=fresh       |     1173 |      1.000 |           3148.1 |       0.621 |            22441.0 |
| k=8, pred=age-matched |     1165 |      1.000 |           3120.5 |       0.627 |            38174.5 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

Gap relative to the matched-k LOCF baseline (coverage uniformly 1.000). Rows
sweep sensing budget `k` (informational `·`); a more-negative gap is a larger
improvement, so the strongest gap reduction in each method column is in bold.
The `valid (am vs off)` column is a *regression* the predictor introduces, not an
improvement — read it as a cost; its largest drop is the lowest-budget selection
artifact, not a win.

| k · | age-matched vs off ↓ | fresh vs off ↓ | age-matched vs fresh ↓ | valid (am vs off) ↓ |
|----:|---------------------:|---------------:|-----------------------:|--------------------:|
|   1 |               −7.9 % |         −3.7 % |             **−4.4 %** |              −24.4 % |
|   4 |          **−11.6 %** |     **−9.6 %** |                 −2.3 % |              −1.6 % |
|   8 |               −5.3 % |         −4.4 % |                 −0.9 % |              −1.6 % |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

### Verdict: mismatch hypothesis REFUTED

The train/deploy distribution mismatch is **not** the cause of the lost gain.
Two independent lines of evidence:

1. **Mechanism is inert.** When fed identically stale neighbors offline,
   age-matching improves the residual P90 by ≤1 %. The fresh-at-`t` training was
   not discarding a recoverable signal — at the neighbor staleness the planner
   actually sees, the neighbor mean is weakly informative regardless of whether
   the model is told its age. There is nothing for age-matching to recover.

2. **Planner gains do not materialise as the hypothesis predicts.** Age-matched
   does edge out fresh-trained on gap at every k (−0.9 % to −4.4 %), and both
   beat LOCF on the gap *of the rounds they certify* — but every one of those
   apparent wins is bought with strictly lower claimed confidence (k=4:
   0.563 vs 0.653 off) and, at k=1, a 24 % collapse in valid rounds, exactly the
   selection artifact the dense-sensing disposition already flagged: the
   predictor certifies a smaller, easier subset. At k=4/k=8 where valid-round
   counts are matched, the age-matched gap improvement over off (−11.6 %, −5.3 %)
   barely exceeds fresh-trained (−9.6 %, −4.4 %) and still costs confidence; the
   net is not a recovered gain but the same tighten-the-point-estimate /
   pay-it-back-in-confidence trade the integration outcome already reported.

Coverage held at 1.000 throughout (soundness intact). The disposition is
**unchanged**: the predictor's payoff regime is a continuously-reporting fixed
sensor network with neighbors fresh at query time. Age-matching the training to
the deployment staleness does not move it into the route-focused/sparse regime,
because the limiting factor there is the *information* in stale neighbors, not a
*distribution mismatch* the model could be retrained around. Shipped opt-in,
default `age_matched=False`.
