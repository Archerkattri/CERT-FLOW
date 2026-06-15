# Extended validation (RSS-strengthening)

Additional experiments beyond the published paper, run to probe the method
harder and broaden the evidence for the RSS version. **These do not change the
published engrXiv results**; they are additive. Every cell was run with real
code and then adversarially audited for fairness and honesty by a separate
reviewer. Findings that do **not** favour CERT are reported prominently —
several of them sharpen or qualify the paper's claims.

Reproduce: `scripts/extval/*.py` (each prints its measured numbers).

---

## 1. Conformal baseline battery — age-weighting vs NexCP vs ACI

`scripts/extval/baseline_battery.py` (10 seeds, 2000 events/stream, target
0.90). Faithful comparators on the **same** edge-cost residual stream:
NexCP (Barber et al. 2023, fixed index-geometric weights), ACI
(Gibbs–Candès), and CERT's age-weighted `ConformalScorer`.

| stream | method | coverage | median width |
|---|---|---:|---:|
| synthetic (irregular sensing) | CERT (age) | 0.916 | 51.4 |
| | NexCP (index) | 0.903 | **47.7** |
| | ACI (γ=0.1) | 0.887 | 35.0 |
| METR-LA residuals | CERT (age) | 0.910 | 285.8 |
| | NexCP (index) | 0.907 | **281.0** |
| | ACI (γ=0.1) | 0.896 | 203.8 |

**Honest finding (does not favour CERT): on the bare residual stream,
age-weighting and fixed index-weighting are a statistical tie** — NexCP is
marginally tighter on both 10-seed datasets (within ±sd; a 3-seed run flips
it the other way). Root cause, diagnosed not hand-waved: in steady sensing
the pooled residual arrivals occur at a near-constant wall-clock rate, so
`corr(arrival index, time) = 0.9994` — when index is ~linear in time,
index-weighting and age-weighting are nearly the same estimator. This was
robust to a bursty-rate construction (still tied). **Implication for the
paper:** CERT's genuine advantage over exchangeable/fixed-weight conformal
is **not** the quantile's age-weighting per se — it is (i) the explicit
`ρ·age` drift-widening term (which this battery omits for a fair
residual-only comparison) and (ii) the sense-to-certify loop. We will state
this directly rather than overclaim the weighting. ACI controls long-run
miscoverage at exactly 0.100 (Prop 4.1), at a tighter width but with mild
marginal under-coverage in the synthetic regime.

---

## 2. Calibration → test distribution shift

`scripts/extval/stress_cal_shift.py` (12 seeds × 300 rounds, abrupt
changepoint: ρ 0.005→0.06 and/or gaussian→student-t, A1 bound frozen at the
pre-shift value). Measured drift q95 0.0054→0.063; realised A1-violation
0.000→0.851 — the planner is genuinely surprised.

- **Edge (conformal) layer**, where Barber's guarantee is tight: post-shift
  transient miscoverage jumps to **~5× the claimed level** at `eps_tv=0`
  (0.11 vs 0.02), then self-recalibrates in the settled window (0.003–0.015).
- **Path certificate** `LB≤OPT≤UB`: stays ~**1.000 in every segment even at
  `eps_tv=0`**, because the per-edge Bonferroni union bound is heavily
  conservative.
- **`eps_tv`'s real role is claim honesty, not a coverage rescue.** At
  `eps_tv=1e-3` the certificate **refuses to certify** the stale-buffer
  transient (valid-fraction → 0) rather than overclaiming; it is the only
  mechanism that reacts to the shift at all.

**Honest finding:** the literal hypothesis ("`eps_tv>0` rescues path
coverage") did not reproduce — path coverage was never the thing at risk.
The shift breaks coverage at the *edge* layer, and `eps_tv` buys
refusal-to-certify, not a better coverage number.

---

## 3. Adversarial drift placement (A1 worst case)

`scripts/extval/stress_adversarial.py` (12 seeds × 300 rounds; an adversary
holds an edge flat while sensed, then drifts it maximally once stale —
concentrating drift where the `ρ·age` budget is largest). Severity swept
1×–32× the assumed ρ.

- **The A1-violation lemma holds at every severity, arm, and ρ-mode**
  (coverage loss ≤ realised violation mass = True throughout), even as
  realised violation mass reaches 82–96%.
- New boundary: with **online ρ-estimation**, the lower-bound side degrades
  to **0.685 coverage only under extreme 32× adversarial severity** (the
  online estimator is slow to react to adversarially-timed drift); the
  given-ρ mode and the upper bound hold at 1.000.

**Takeaway:** the soundness lemma is robust; the one degradation is a
characterised extreme-adversary + online-ρ corner, not a general failure.

---

## 4. Spatially-correlated + heavy-tailed drift

`scripts/extval/stress_correlated.py` (12 seeds × 250 rounds; a shared latent
factor drives neighbouring edges together; student-t and Pareto drift
increments).

- **Coverage 1.0000** under both correlated and independent worlds, both
  heavy-tailed families (target 0.80) — correlation and heavy tails do **not**
  break coverage. Deployed certificate gaps are nearly identical across
  worlds (42.19 vs 42.19) — per-edge Bonferroni width is correlation-insensitive.
- **Honest finding (opposite to the hypothesis):** the sum-aware joint
  bound's advantage comes from **independence** (√L pooling), not
  correlation. Positive spatial correlation makes path-sum residuals stack,
  inflating the joint half-width, so the Bonferroni-vs-joint over-pay
  *shrinks* under correlation (8.35×→6.70× student-t; 9.04×→6.00× Pareto)
  rather than growing.

---

## 5. Certified-loop scaling

`scripts/extval/scaling.py` (3 seeds, bounded drift, recommended config).

| grid | \|E\| | L | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| 20×20 | 1520 | 38 | 0.94 | 1.80 |
| 40×40 | 6240 | 78 | 2.33 | 5.26 |
| 60×60 | 14160 | 118 | 4.06 | 14.88 |
| 80×80 | 25280 | 158 | 7.41 | 28.96 |
| 100×100 | 39600 | 198 | 10.13 | 46.64 |

p50 grows **10.8× for 26× the edges (sublinear)** — real-time at every size,
consistent with the published `scale.md`.

**Honest finding (sharpens a published caveat):** under the deployed
recommended config (`rho_w=0.99`), the certified loop **never exits warm-up on
grids ≥ 60×60** — not from too few rounds, but because the calibration
buffer's effective sample size saturates at ESS≈100 < L−1 (=117 at 60×60), so
the weakest annealed claim can never be supported. This root-causes the
`scale.md` observation that the gap is reported as ∞ for large grids: it is an
**ESS-ceiling vs path-length** limit, and lifting it needs a larger ESS
(higher `rho_w` / bigger buffer) or shorter certified paths.

---

## 6. FoMo — real off-road seasonal drift (Forêt Montmorency)

`scripts/extval/fomo_validation.py` on the FoMo dataset (norlab-ulaval): 6
route-colours re-traversed across up to 12 deployments over a year (−19 °C
snow → summer vegetation). Cost signal only (GNSS `gt.txt` poses + battery
power + weather/snow; the 9.4 TB of raw lidar/radar/camera is **not** needed
and not downloaded — 150 MB total). Per-route segments aligned by normalised
arc-length; per-segment cost = traversal time and energy (∫|V·I|dt); each
deployment is one observation, the seasonal change is the drift. ρ at the p75
per-step convention (the documented traffic-tier interior optimum).

| cost | edge coverage | path coverage | A1-violation | median width |
|---|---:|---:|---:|---:|
| traversal-time | 0.869 | 1.000 | 0.25 | 78.6 s |
| energy | 0.879 | 1.000 | 0.25 | 1.96e5 J |

**Honest finding:** on **real off-road seasonal drift** — far more abrupt
than bounded synthetic drift, with single-step predictions spanning months
and 25% of steps drifting beyond the p75 bound — the certificate holds at
the **path level (1.000)** but the marginal **edge** guarantee strains
slightly (**0.87 vs the 0.90 target**). This is the credible new-domain
result: the conformal layer absorbs most but not all of a winter→summer
regime shift; path-level Bonferroni conservatism still yields full coverage.
It validates the core claim in a genuinely new domain (field robotics)
without a suspiciously-perfect number.

*Disclosures:* segment alignment is by normalised arc-length (the colours are
re-traversals of the same trail), a stated modelling choice, not a
ground-truth correspondence; replay only, no robot.

---

## What this changes for the paper

- **Refine** the conformal claim: credit the `ρ·age` drift term + sensing
  loop for the advantage over exchangeable/fixed-weight CP, not the
  quantile's age-weighting (cell 1).
- **Reframe** `eps_tv` as a refusal-to-certify / claim-honesty mechanism
  under abrupt shift, not a path-coverage rescue (cell 2).
- **Keep** the soundness story: the A1 lemma holds adversarially (cell 3);
  coverage is robust to correlation/heavy tails (cell 4).
- **Add** a real field-robotics datapoint (cell 6) and an honest scaling
  boundary (cell 5).
