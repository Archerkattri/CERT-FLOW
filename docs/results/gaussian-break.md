# Gaussian-break experiment: edge-level calibration audit

`scripts/run_gaussian_break.py` full (25 seeds x 400 rounds, 10x10 grids,
alpha'=0.1 so alpha_edge ~ 0.0056, rho_w=0.999, ACI frozen).
The audit draws a uniformly random edge each valid round and tests a fresh
observation (never fed back) against the planner's UNCLIPPED nominal interval
— the fixed-edge observable guarantee (T1a). Planner-SELECTED edge miss rates
are reported separately (selection diagnostic).

| condition | planner | audit n | miss rate | 95% CI | ratio vs claim | verdict |
|---|---|---:|---:|---|---:|---|
| gaussian noise (control) | CERT     | 3484 | 0.0052 | [0.0031,0.0082] | 0.9  | ok |
|                          | Gaussian | 9850 | 0.0268 | [0.0237,0.0302] | 4.8  | **BROKEN** |
| student_t (df=3)         | CERT     | 4955 | 0.0071 | [0.0049,0.0098] | 1.3  | ok |
|                          | Gaussian | 9850 | 0.0409 | [0.0371,0.0450] | 7.4  | **BROKEN** |
| skewed (lognormal)       | CERT     | 4399 | 0.0073 | [0.0050,0.0103] | 1.3  | ok |
|                          | Gaussian | 9850 | 0.0554 | [0.0510,0.0601] | 10.0 | **BROKEN** |
| drift 0.02 + skewed      | CERT     | 4881 | 0.0016 | [0.0007,0.0032] | 0.3  | ok |
|                          | Gaussian | 9850 | 0.0041 | [0.0029,0.0055] | 0.7  | ok |

## Findings

1. **The Gaussian building block breaks 4.8-10x — even when the noise IS
   Gaussian.** The 4.8x control break shows this is plug-in inference
   failure (sigma-hat estimation error at the 0.6% tail), not merely wrong
   family. Skewed noise is worst (10x): a symmetric fit cannot represent an
   asymmetric tail at any sigma.
2. **CERT is calibrated everywhere (0.3-1.3x), including under noise that
   violates its own A3** — the conformal edge guarantee is distribution-free
   as claimed. Drift masks the Gaussian failure (0.7x) because rho*a
   widening dominates the quantile; the parametric flaw is hidden, not fixed.
3. **Path-level coverage stays 1.000 for both planners** — Bonferroni slack
   masks the broken building block. The slack-vs-soundness chain: spend the
   slack for tightness (T4 sum-aware) and only the calibrated building block
   survives.
4. **Selected-edge miss rates run below audited rates for CERT and above
   claim for Gaussian** — reported as a diagnostic; the guarantee is for
   fixed edges.

## Root-cause findings

Two candidate explanations are ruled out, and the true cause was isolated:
- **Selection bias** (optimistic-path membership selects low-c_hat edges) is
  real as a phenomenon (T4's freshness gate exists because of it) but ruled
  out here by the independent audit.
- **One-dependence of chained scores** is real (thinned mode exists for it)
  but not the cause — thinned runs break identically.
- **Clip semantics (the actual cause):** the cost-floor clip on ell is sound
  for LATENT costs (c > 0) but invalid for OBSERVABLE coverage events —
  y = c + eta can be negative under heavy left tails. Testing observables
  against clipped intervals produces a spurious 3.7x "CERT break" with the
  left-tail fingerprint (Student-t broken, right-skewed fine, Gaussian
  control fine). Fix: coverage events (ACI errs, audits) test unclipped
  intervals; the clip lives only in the search metrics (recorded in
  theory.tex honest-accounting).
- **Design rule:** the weighted buffer's effective sample size ~1/(1-rho_w)
  must exceed 1/alpha_edge - 1, else warm-up never ends (rho_w=0.99 cannot
  support alpha_edge=0.0056; 0.999 can).
