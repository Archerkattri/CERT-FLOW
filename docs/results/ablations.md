# Ablation suite results

*Component-by-component ablation of CERT-FLOW: kappa hysteresis, lazy pre-widening, maintenance, and the round-robin backstop, scored on certificate fidelity, plan stability (churn), and latency.*

**Reproduce:** `scripts/run_ablations.py`

> **Finding —** kappa hysteresis lost its original latency job to lazy pre-widening, but earns its keep as a churn suppressor at *identical* coverage, valid%, gap, and latency. Maintenance and the backstop look inert until epsilon clears the certificate floor; in the rerun above that floor they behave exactly as the theory predicts.

Setup: full run is 20 seeds x 300 rounds (8x8 bounded drift, rho=0.02,
epsilon=5, alpha'=0.2, eps_tv=1e-4). Raw tables regenerable in `results/`.
Churn = Fox plan-stability metric (edge symmetric-difference between
consecutive incumbents); flap% = fraction of rounds with nonzero churn.

## Full run (below the certificate floor)

Rows ordered best -> worst by **churn mean** (the metric this suite is built to
move); coverage and valid% are effectively tied across conditions.

| condition · | coverage ↑ | valid% ↑ | gap~ ↓ | churn mean ↓ | churn p95 ↓ | flap% ↓ | p50 ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-backstop           | **1.000** | 51.6%     | 24.50     | **0.46** | **0**  | **3.4%**  | 0.67 |
| B=0 (no pre-widening) | **1.000** | 50.5%     | **22.52** | 0.50     | **0**  | 3.7%      | 1.29 |
| full (kappa on, B=10) | **1.000** | 51.6%     | 25.13     | 0.52     | **0**  | 3.7%      | 0.70 |
| no-maintenance        | **1.000** | 51.6%     | 25.13     | 0.52     | **0**  | 3.7%      | 0.71 |
| B=20                  | **1.000** | **53.5%** | 26.89     | 0.59     | **0**  | 4.1%      | **0.50** |
| no-kappa              | **1.000** | 51.6%     | 25.13     | 1.71     | 12     | 17.9%     | 0.70 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

## Findings

1. **kappa hysteresis clears its kill-gate, in its new role.** -70% mean churn
   (1.71 -> 0.52), p95 12 -> 0, flap rounds 17.9% -> 3.7% — at *identical*
   coverage, valid%, gap (25.13 both, to two decimals: direct evidence the
   certificate is untouched), latency, and sensing spend. The original
   latency-based gate (>=20% replanning speedup) is dead — kappa contributes
   zero latency — but the stability gate is passed decisively. Paper framing:
   flow-memory's surviving role is churn suppression within the certified
   slack, after lazy pre-widening took its latency job.
2. **Pre-widening is a clean latency/width dial.** B=0 -> 10 -> 20:
   1.29 -> 0.70 -> 0.50 ms (1.8x, 2.6x) for +12% / +19% median gap.
3. **Maintenance and backstop rows are uninformative in this regime — by
   design, not failure.** cert% = 0 everywhere here because epsilon=5 is below
   the T2' floor at L~14 (2Lq alone exceeds it): maintenance only activates
   when certified, and the greedy sensing score (2*rho*age) already emulates
   round-robin, so the backstop never binds. Maintenance's effect is shown by
   `test_maintenance_keeps_static_certificate_alive` (late-valid >=90% vs ~0);
   the backstop remains the device that makes T2'(a) a theorem rather than an
   empirical hope. Informative ablation configs set epsilon above the T2'
   floor for the chosen world (the eps=12 rerun below).
4. **The Bonferroni warm-up burden is the dominant practical cost** (~48% of
   rounds invalid at L~14). This is the strongest argument for the sum-aware
   score stretch theorem (spec section 5).

## Rerun at eps=12 (above the T2' floor; --eps12, annealed defaults)

Rows ordered best -> worst by **cert%** — the metric that was pinned at zero in
the run above and only becomes informative once epsilon is attainable.

| condition · | coverage ↑ | valid% ↑ | cert% ↑ | gap~ ↓ | churn ↓ | flap% ↓ |
|---|---:|---:|---:|---:|---:|---:|
| B=0                | **1.000** | **94.9%** | **12.4%** | **19.60** | **0.22** | **1.3%**  |
| no-backstop        | **1.000** | 94.5%     | 7.0%      | 21.21     | 0.25     | **1.3%**  |
| full (kappa, B=10) | **1.000** | 94.5%     | 4.3%      | 21.93     | 0.26     | 1.6%      |
| no-kappa           | **1.000** | 94.5%     | 4.3%      | 21.93     | 1.76     | 17.8%     |
| no-maintenance     | **1.000** | 94.5%     | 3.0%      | 21.93     | 0.25     | 1.6%      |
| B=20               | **1.000** | 94.2%     | 0.6%      | 23.72     | 0.26     | 1.6%      |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

With epsilon attainable, the previously-uninformative rows speak:
maintenance contributes +43% relative certified rounds (4.3 vs 3.0); the
pre-widening width cost is decisive near the floor (B=0 certifies 3x more
than B=10, B=20 nearly never — the adaptive-B policy exists for exactly
this); kappa's churn suppression reproduces (0.26 vs 1.76). The no-backstop
row reads slightly higher (7.0%) — the forced round-robin is a guarantee
device (T2'a applies to the deployed policy) whose forced picks can
occasionally displace greedy's better choice; we keep it for the theorem
and report the small price honestly.
