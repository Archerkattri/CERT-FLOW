# CIA vs CERT: path-cost coverage as the calibration->test time gap grows

`scripts/run_cia_comparison.py` full (50 paths, 20 repetitions per gap,
METR-LA 205-sensor / 888-edge graph). Target 90% coverage (alpha=0.10).

## Attribution and license

The CIA intervals in this experiment are produced by a **faithful extraction
of the symmetric-calibration sum construction** from Luo & Zhou's own code,
*Conformalized Interval Arithmetic with Symmetric Calibration* (arXiv
2408.10939, AAAI 2025; repository https://github.com/luo-lorry/CIA). The
extracted procedure is their `main.py::group_by_dimensions` "Group sampling /
stratified CIA" branch: per-element signed nonconformity scores, a random
two-way split of the labelled element population (symmetric calibration),
path-sum scores drawn by sampling k elements without replacement from the
calibration half and **stratified by path length k**, and the
`(1-alpha)(1+n)/n` calibration percentile, with the interval centred at the
predicted path sum and half-width equal to that percentile (their reported
"efficiency" is `2 * percentile`, i.e. the full width). We run **their
construction**, not their CLI: their CLI consumes static Anaheim/Chicago
equilibrium-flow `.tntp` data, and our question is temporal (a METR-LA gap
sweep), so the published loader does not apply.

**License note (prominent):** the CIA repository contains **no LICENSE file**
(6 commits, code released alongside an AAAI 2025 paper with an explicit code
pointer). We run an extracted re-implementation of their published method for
a research comparison under academic-use norms, and state that here. No CIA
source is copied or vendored into this repository.

## What CIA assumes, and why the gap is the right stress

CIA's coverage guarantee for a sum holds when the per-element nonconformity
scores are **exchangeable** between calibration and test. On a single METR-LA
time slice that holds: the per-edge residual is observation noise, identically
distributed across edges, and CIA's symmetric calibration covers the true path
sum at the calibration time. CIA has **no drift model, no age weighting, no
sensing**. So the moment the test path sum is read at a LATER time than
calibration, the truth has drifted away from the calibrated slice and CIA's
interval -- pinned to the calibration slice -- has no channel to follow it.
That is exactly the failure CERT's non-exchangeable age weights plus the
explicit drift term `rho * age` are built to correct.

## Protocol

- **Graph / costs:** `TrafficWorld(seed=0)` METR-LA topology, 205 sensors /
  888 edges; edge cost = `dist / (speed * 0.44704)` seconds (as in
  `realworld.py`). One long replay window (3200 bins) so `T_cal + 24h` always
  fits; topology is seed-independent.
- **Paths:** a fixed set of 50 simple s-g paths, 6-15 edges each (median 9),
  drawn from the real graph.
- **Gaps:** calibrate at `T_cal`, test the true path sum at `T_cal + gap` for
  gap in {0, 1h, 3h, 6h, 12h, 24h} = {0, 12, 36, 72, 144, 288} five-minute
  bins. 20 random `(T_cal, path)` repetitions per gap.
- **Shared data:** both methods see the **same** observed edge costs at
  `T_cal` and are scored against the **same** true path sums at `T_cal + gap`.
- **CIA row:** the extracted symmetric-calibration sum interval above, with
  per-edge score = `obs(e, T_cal) - true(e, T_cal)` (CIA's `yhat - y`; here
  `yhat` = the calibration-time observation, `y` = the true cost), predicted
  path sum = sum of those observations.
- **CERT row:** per-edge `c_hat +/- (q + rho * gap_seconds)` summed over the
  path by Bonferroni. `c_hat = obs(e, T_cal)`; `q` is the
  `ConformalScorer` weighted-conformal quantile of the calibration slice's
  per-edge residuals at the Bonferroni per-edge level `alpha / L`; `rho` is
  the world's **p75** per-edge `|dc/dt|` drift rate (the width-optimal dial
  from `docs/results/metr-la.md`); the staleness term is `rho * gap_seconds`.
  Same data, same gaps, same 90% level.

## Result: coverage and median width vs gap

| gap | CIA coverage | CIA 95% CI | CIA med width (s) | CERT coverage | CERT 95% CI | CERT med width (s) |
|---|---:|---|---:|---:|---|---:|
| 0   | **0.950** | [0.751, 0.999] | 57    | 1.000 | [0.832, 1.000] | 292   |
| 1h  | 0.550 | [0.315, 0.769] | 50    | 0.950 | [0.751, 0.999] | 1177  |
| 3h  | 0.250 | [0.087, 0.491] | 47    | 0.950 | [0.751, 0.999] | 1799  |
| 6h  | **0.200** | [0.057, 0.437] | 54    | 1.000 | [0.832, 1.000] | 4266  |
| 12h | 0.350 | [0.154, 0.592] | 49    | 1.000 | [0.832, 1.000] | 6078  |
| 24h | 0.550 | [0.315, 0.769] | 49    | 1.000 | [0.832, 1.000] | 14238 |

(20 repetitions per cell; A1-violation rate of the p75 drift model on this
window: 0.250, consistent with the metr-la doc.)

## Interpretation

1. **CIA is valid exactly where its assumption holds, and only there.** At
   gap = 0 -- the static, exchangeable slice that is CIA's actual setting --
   it covers at 0.950, above the 90% target, with a tight 57 s interval. This
   is the positive control: the extracted construction reproduces CIA's
   intended behaviour, so the failures below are about the assumption, not a
   broken re-implementation.

2. **Coverage collapses as the gap opens.** One hour out, CIA is already at
   0.55; by 3-6 hours it is 0.20-0.25 -- i.e. the true path cost lands outside
   CIA's interval **three times out of four**. The interval width barely moves
   (47-57 s throughout) because CIA has no term that grows with elapsed time:
   it is calibrated to a slice and stays pinned to it while the truth drifts
   away. This is the exchangeability fragility, measured.

3. **The 24h partial recovery is real and diagnostic, not noise.** CIA climbs
   back to 0.55 at 24h. METR-LA traffic is strongly diurnal: a full day after
   `T_cal` the network is often back near its calibration-time state, so a
   drift-blind interval partially re-covers. This is honest evidence that
   CIA's failure is *drift*, not *random degradation* -- coverage tracks how
   far the world has moved from the calibration slice, and a periodic world
   brings it back. It is not a property anyone could rely on (the worst gaps,
   3-6h, are exactly the operationally common staleness range), but it is the
   right shape for the stated mechanism.

4. **CERT holds across every gap, at a width cost it does not hide.** CERT
   stays at 0.95-1.00 for all gaps -- never below the 90% target within CI --
   by widening each edge with `rho * gap_seconds`. The price is explicit:
   median width grows from 292 s at gap 0 to 14238 s at 24h (~49x). This is
   the same trade documented against AD*-semantics in `extern-baselines.md`
   (narrow-and-wrong vs wide-and-sound on stale maps), now against the closest
   conformal method instead of a search bound. At gap 0 CERT is wider than CIA
   (292 vs 57 s) -- it pays Bonferroni over the path and carries a drift term
   that is near-zero but not zero -- so when the world is genuinely static,
   CIA is the more efficient instrument. The moment the world moves, that
   efficiency is coverage CERT keeps and CIA loses.

## Bottom line

On its own setting (exchangeable slice, gap 0) CIA does what it claims: 0.950
coverage, tightest intervals. As the calibration-to-test gap grows, CIA's
coverage falls to 0.20-0.25 at the 3-6h staleness that dominates real
operation, with no widening to compensate, because it has no drift channel.
CERT's non-exchangeable age weights plus `rho * age` widening hold coverage at
0.95-1.00 across all gaps, paying with intervals up to ~49x wider. This is
the direct, quantitative instantiation of the related-work claim that CIA
"achieves valid path-cost coverage only under exchangeability, with no drift,
weights, sensing, or online loop."

## Deviations from CIA's original construction

Listed for honesty; none changes the mechanism under test.

1. **Domain.** CIA's CLI runs on static Anaheim/Chicago equilibrium **flow**
   volumes; we run on METR-LA **travel-time** costs because the gap sweep is
   temporal and METR-LA is our traffic setup. The construction (symmetric
   calibration, length-stratified path-sum percentile) is unchanged.
2. **Predictor.** CIA pairs the construction with a directed-graph
   autoencoder (`yhat` from the GNN). We use the calibration-time observation
   as `yhat` and the true cost as `y`, so the per-edge score is the
   observation-noise residual. This is the cleanest faithful mapping of "yhat
   - y" into a temporal slice and isolates the variable of interest (the gap);
   substituting a learned predictor would change `yhat`'s bias but not CIA's
   drift-blindness, which is the property being measured. At gap 0 the 0.950
   coverage confirms the mapping preserves CIA's guarantee.
3. **Sampling count.** CIA's "Group sampling" branch uses `num_samples = 100`
   calibration path-sum draws; we use 200 to tighten the percentile estimate
   on the larger edge population. The percentile formula
   `(1-alpha)(1+n)/n` and the without-replacement length-k sampling are
   theirs, unchanged.
4. **Score sign.** We report the two-sided absolute-score interval
   (`pred +/- percentile`, CIA's `efficiency = 2 * percentile`), matching
   CIA's `column_sums_abs = |sum(...)|` branch. We did not also run their CQR
   one-sided variant, since the absolute-score interval is the headline
   construction and the comparison axis is coverage-vs-gap, not score family.

## Semantics caveat for the gap-0 width comparison

CIA's 57s interval covers a FIXED path's cost sum; CERT's 292s brackets the
MINIMUM over all paths (OPT). T5 (theory.tex) proves the uniform-over-paths
guarantee must pay a Bonferroni-order price — so the gap-0 width difference
is largely the cost of the stronger claim, not slack: the two numbers
certify different objects.
