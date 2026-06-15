# Real-data validation: METR-LA traffic replay

*The certificate holds on two real traffic recordings it was never tuned for — empirical coverage stays at or above every claimed confidence, even where a real-incident drift model is violated at a measured rate.*

**Reproduce:** `scripts/run_metr_la.py` (add `--pems-bay` for the held-out city)

> **Finding —** A misspecified drift assumption (drift the model misses) costs interval *width*, never coverage: understated drift lands in the drift-adjusted conformal scores and inflates `q`. This self-absorption was invisible in synthetic worlds (where the assumption held by construction) and is the architecture's central redundancy paying off in the wild.

`scripts/run_metr_la.py` full (20 seeds = 20 distinct replay days, 288 rounds
each = one observation per 5-minute bin, 205-sensor LA highway graph, 888 edges,
costs = travel-time seconds from recorded speeds, oracle exact on the recording).
`--pems-bay` runs the same harness on the held-out city (325 sensors, 20 replay
days). epsilon=120s, alpha'=0.2, unit-aware config (`traffic_planner_config`).
rho_e = per-edge empirical quantile of |dc/dt|; A1 violations are real incidents
the drift model misses, at a measured rate (mean A1-violation rate LA 0.151, Bay
0.150). Raw: `results/metr_la/table.json`, `results/pems_bay/table.json`. Probe
sweep (4 seeds x 100 rounds): rho at p95/p75/p50 gives measured A1-violation
rates 5%/25%/49% — coverage 1.000 at all three.

## Results

Rows are ordered best -> worst on the primary metric, **coverage** (target-relative:
each row must meet/exceed its own *claimed* confidence — all rows do). Where coverage
is saturated at its ceiling for several rows, ties are broken by the efficiency metric
the experiment optimizes, **gap median (lower is better)**.

### METR-LA (20 replay days)

| planner | valid% ↑ | coverage ↑ | 95% CI · | claimed ↑ | gap median (s) ↓ | spend ↓ |
|---|---:|---:|:--|---:|---:|---:|
| CERT, p75 + adaptive | 93.6% | 1.000 | [0.999,1.000] | 0.584 | **4330** | 336 |
| CERT, rho=p75        | 93.6% | 1.000 | [0.999,1.000] | 0.572 | 4774 | **288** |
| CERT, rho=p95        | 94.7% | 1.000 | [0.999,1.000] | 0.588 | 8797 | **288** |
| Gaussian, rho=p95    | **94.8%** | 1.000 | [0.999,1.000] | **0.742** | 11288 | **288** |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*
<br>*coverage is target-relative (meet/exceed the row's `claimed`); all four rows tie at the coverage ceiling. `claimed` is honesty-bound — the bolded best-in-column claim belongs to the **Gaussian** strawman and is unjustified (no staleness correction), not a virtue.*

### PEMS-BAY (20 replay days)

| planner | valid% ↑ | coverage ↑ | 95% CI · | claimed ↑ | gap median (s) ↓ | spend ↓ |
|---|---:|---:|:--|---:|---:|---:|
| CERT, rho=p95        | **96.7%** | **1.000** | [0.999,1.000] | 0.680 | 1067 | **288** |
| Gaussian, rho=p95    | 96.6% | **1.000** | [0.999,1.000] | **0.772** | 1570 | **288** |
| CERT, rho=p75        | 95.5% | 0.993 | [0.991,0.995] | 0.644 | **679**  | **288** |
| CERT, p75 + adaptive | 95.5% | 0.987 | [0.984,0.990] | 0.643 | 683  | 296 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*
<br>*Ranked by coverage (the certificate metric), gap breaking ties among ceiling-coverage rows. Coverage is target-relative — every row sits above its `claimed` value (the certificate holds); `claimed` is honesty-bound, so Gaussian's bolded best-in-column claim is unjustified (no staleness correction). Deployment nuance: `CERT, rho=p75` is the **tightest valid** config (gap 679, coverage 0.993 ≫ its 0.644 claim) — the practical pick when a tighter certificate is worth a thinner coverage margin.*

## Findings

1. **The certificate holds on real data it was never tuned for.** Coverage
   >= 0.987 across 20 replay days per city with the oracle computed exactly
   from the recording — including under a drift model that is violated by real
   incidents 5-49% of the time. Mechanism: understated drift lands in the
   drift-adjusted conformal scores instead of the rho*a widening, inflating
   q — A1 misspecification costs width, never coverage. This self-absorption
   was invisible in synthetic worlds (where A1 held by construction) and is
   the architecture's central redundancy paying off in the wild.
2. **The drift-model dial has an interior optimum on real data:** on LA, p75
   (25% violations) gives gaps 46% tighter than the conservative p95 (4774s vs
   8797s); pushing to p50 backfires as the score mass explodes. Tuning
   aggressiveness against measured violation rates is a real operational knob
   this experiment surfaces.
3. **Conformal beats Gaussian on width at equal coverage:** CERT p95 is tighter
   than Gaussian p95 on identical worlds (8797 vs 11288 on LA, 1067 vs 1570 on
   Bay), and the best CERT variant is 1.3-2.6x tighter — while Gaussian claims
   0.742/0.772 confidence with no staleness correction (an unjustified claim
   that happens to hold here because drift-widths dominate).
4. **Held-out replication.** Every PEMS-BAY row's coverage sits above its claim;
   the aggressive variants visibly spend slack toward the claimed level (0.987
   vs claim 0.643), the intended efficiency behavior. Bay Area traffic is far
   gentler than LA (gaps ~8x tighter); the adaptive variant trims the LA gap
   -9% (4774 -> 4330) and is gap-neutral on the already-gentle Bay (683 vs 679)
   — matching T2' across two real regimes.
5. **Honest negatives:** cert% = 0 at epsilon=120s — LA traffic drift makes
   2-minute route certification unattainable at one observation per 5 minutes
   (T2' floor in the thousands of seconds; the planner declares this rather
   than chasing it). The adaptive variant cannot help when epsilon is
   unattainable at every k <= max (guard working as intended). Observation
   noise is synthetic (the recording does not separate sensor noise from
   state) — stated, not hidden.
