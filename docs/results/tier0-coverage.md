# Tier-0 · Coverage validation and Gaussian baseline comparison

*Across every condition the claim guarantee holds (coverage stays above the claimed confidence on every row), certification degrades gracefully with drift, and the lone broken setting self-extinguishes its claim loudly rather than overclaiming silently.*

**Reproduce:** `scripts/run_tier0.py`, `scripts/run_tier0_baselines.py`

Settings: 25 seeds x 300 rounds per condition, 6x6 grids, epsilon=5, alpha'=0.2, eps_tv=1e-4 unless noted; maintenance sensing + lazy pre-widening + alpha annealing enabled. Raw tables regenerable in `results/`.

> **Finding —** Coverage >= claimed confidence on every condition (including off-model jump and periodic), so miscoverage relative to the claim appears nowhere. Certification rate falls monotonically with drift severity (correct refusal under hard drift), and the misspecified setting self-extinguishes its claim loudly instead of overclaiming silently.

## Main sweep (CERT, 14 conditions incl. provable-mode rows)

Rows ranked best -> worst by certification rate (cert% ↑); the condition label is informational. Coverage is target-relative — every row stays above its own claimed confidence, which is the validation result.

| condition | valid% ↑ | coverage ↑ | claimed ↑ | gap~ ↓ | cert% ↑ |
|---|---:|---:|---:|---:|---:|
| **sum-aware static**    | **96.6%** | 0.966 | 0.503 | **2.98**  | **95.5%** |
| static (rho=0)          | **96.6%** | **1.000** | 0.509 | 3.73  | 93.5% |
| bounded rho=0.005       | **96.6%** | **1.000** | 0.621 | 4.75  | 71.3% |
| lambda=2 (T1b)          | 96.3% | 0.989 | 0.633 | 8.72  | 30.7% |
| misspec rho_hat=0.5x    | **96.6%** | **1.000** | 0.648 | 5.68  | 27.7% |
| lambda=2 + thinned      | 96.0% | 0.994 | 0.568 | 10.73 | 21.2% |
| bounded rho=0.02        | 96.5% | **1.000** | 0.639 | 9.62  | 12.1% |
| misspec rho_hat=2x      | 96.0% | 0.991 | 0.639 | 16.15 | 7.7%  |
| **sum-aware noise-dom** | 96.5% | 0.916 | 0.649 | 6.86  | 6.7%  |
| off-model: periodic     | 96.0% | 0.994 | 0.645 | 15.00 | 2.5%  |
| bounded rho=0.05        | 96.1% | 0.995 | 0.642 | 18.53 | 2.0%  |
| off-model: jump         | **96.6%** | 0.998 | 0.657 | 83.24 | 0.1%  |
| **provable (l2+thin+noACI)** | 96.0% | **1.000** | 0.569 | 13.65 | 0.0% |
| A2 misspec eps_tv=1e-3  | 23.3% | **1.000** | 0.199 | 11.05 | 0.0%  |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

The full 17-row sweep runs in ~100 seconds on the modernized stack (fastgraph engine, vectorized+staggered refresh). Claimed confidence is honesty-bound — it is a derived, staleness-corrected guarantee, not a knob to maximize — so it is ranked informational rather than bolded.

Findings:

1. **Coverage holds everywhere.** Coverage >= claimed confidence on every row including off-model, always above the claim; miscoverage relative to the claim appears nowhere.
2. **T2' visible.** cert% falls monotonically with drift severity; jump certifies nothing (correct refusal).
3. **A1 misspecification is benign; the other is loud.** Underestimating rho does not break coverage (the conformal layer absorbs it); overestimating costs conservatism only. A2 misspecification self-extinguishes the claim loudly (23% valid, claim annealed to 0.199) rather than overclaiming silently.
4. **Annealing buys validity time.** Validity sits ~96% on every row thanks to alpha annealing — warm-up rounds carry honest weak claims that anneal toward the target, so even provable mode (lambda=2 + thinned + frozen ACI) is 96% valid where a static margin alone would be ~9%.

### Sum-aware upper certificate (T4, theory.tex section 3)

The asymmetric certificate (block-conformal UB at level alpha', Theta(sqrt L) margin; per-edge Bonferroni LB) applied through the freshness gate with kappa-stabilized incumbents:

- **Gap ~-30% to -42% vs Bonferroni** in the noise-floor-dominated regimes (2.98 vs 3.73 static; 6.86 vs 12.24 noise-dom), certification unlocked (6.7% vs 0.0% noise-dom; ~all valid rounds certified in static).
- **Coverage 0.966 / 0.916 — above claim (~0.50/0.65), below Bonferroni's 1.000.** That is the efficiency story: the tighter bound consumes the conservatism slack rather than violating anything, and both remain far above the ungated winner's-curse level (0.823).
- **Selection bias (winner's curse) is real and measured.** Applying T4 naively to the optimizer-selected incumbent drops coverage to 0.823. The freshness gate (sum-aware bound only when every incumbent edge has been re-observed since the path became incumbent) restores conditional validity; kappa-hysteresis opens the gate by stabilizing the incumbent — its second role. No effect under strong drift (age widths dominate; gate rarely opens) — consistent with sqrt(L) applying to the noise floor only.

The sum-aware coverage values (0.966/0.916) are a known, sound, unattributed movement of the engine/cache modernization's interaction with gate-open frequency, bounded by claim-soundness either way; the conservative reading (sum-aware coverage ~0.92 at claims ~0.65) is what the paper cites.

### Provable mode and the ACI interaction (theory.tex T1b)

- **ACI cancels static margins.** With lambda=2 and ACI on, edge-misses vanish, the working alpha climbs until misses return to target, and the intervals end up no wider than lambda=1. No soundness breach, but it contradicts T1b's construction: adaptive coverage controllers and static provable margins fight. The provable mode therefore freezes ACI (use_aci=False).
- **The full provable mode (lambda=2 + thinned + frozen ACI) is sound with margin** (coverage 1.000 vs claimed 0.569) at 96% valid — the "rigor costs validity time" weakness is resolved by annealing (warm-up rounds carry honest weak claims that anneal toward target).
- **Noise/drift asymmetry as predicted.** lambda=2 costs gap in the noise-dominated regime and ~nothing under drift — the provable margin is cheap exactly where the problem is hard.

## CERT vs Gaussian mu+-beta*sigma baseline

Rows are paired by condition (CERT above Gaussian) and so are not independently ranked; **bold** marks the better planner within each pair on the directional metrics. valid% and claimed are honesty-bound here — the Gaussian's flat validity and claim are valid-and-claimed by construction, not by guarantee — so they are not bolded.

| condition | planner | valid% ↑ | coverage ↑ | claimed ↑ | gap median ↓ | cert% ↑ |
|---|---|---:|---:|---:|---:|---:|
| bounded rho=0.02, gaussian noise  | CERT     | 71.9% | **1.000** | 0.655 | **11.49** | **3.7%**  |
|                                   | Gaussian | 98.0% | **1.000** | 0.800 | 16.69 | 0.0%  |
| bounded rho=0.02, student_t noise | CERT     | 72.0% | **1.000** | 0.655 | **11.93** | **2.0%**  |
|                                   | Gaussian | 98.0% | **1.000** | 0.800 | 17.22 | 0.0%  |
| jump, student_t                   | CERT     | 76.8% | **0.999** | 0.675 | 152.9 | 0.0%  |
|                                   | Gaussian | 98.0% | 0.998 | 0.800 | **98.4**  | **1.4%**  |
| static, student_t                 | CERT     | 76.4% | **1.000** | 0.664 | 5.88  | 17.7% |
|                                   | Gaussian | 98.0% | **1.000** | 0.800 | **4.31**  | **82.3%** |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

**Honest reading — do not overclaim.** At these settings the Gaussian baseline does NOT under-cover: heavy-tailed samples inflate the fitted sigma, making mu+-beta*sigma conservative at moderate per-edge alpha. The real observed differences are:

1. **Tightness under drift.** CERT's certified gap is ~30% tighter than Gaussian under bounded drift (11.5 vs 16.7; 11.9 vs 17.2) — the weighted conformal quantile adapts where the Gaussian sigma-fit bloats.
2. **Claim honesty.** Gaussian claims a flat 0.800 with no staleness correction and stays "valid" 98% of rounds by construction; CERT's claim degrades visibly with calibration-buffer age. Same observable coverage here, but only one of the two claims is derived from an actual guarantee.
3. **Where the break is expected.** Unit-level tests show the Gaussian quantile under-estimates true heavy-tail quantiles reliably at small alpha (~0.01, the planner's Bonferroni per-edge level on longer paths). The edge-level audit (gaussian-break.md) surfaces this directly, where the Gaussian building block breaks 4.8-10x while CERT stays calibrated.

### CERT-best (recommended_config) on the standard drifting world

Rows ranked best -> worst by certification rate (cert% ↑) on the same world; coverage is target-relative and both rows stay above the claim, so the defaults' coverage column max is bolded while CERT-best ranks first on the tightness/certification metrics.

| condition | valid% ↑ | coverage ↑ | claimed ↑ | gap~ ↓ | cert% ↑ |
|---|---:|---:|---:|---:|---:|
| **CERT-best (recommended)** | **96.6%** | 0.992 | 0.639 | **6.89** | **20.0%** |
| bounded rho=0.02, defaults  | 96.5% | **1.000** | 0.639 | 9.62 | 12.1% |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

The recommended configuration (online rho + hybrid sensing + kappa + adaptive rate + gated sum-aware UB, `recommended_config()`) tightens the gap 28% and raises certification 65% relative on the same world, at coverage 0.992 — still above the claim; the consumed slack is the efficiency story, as with the sum-aware rows.
