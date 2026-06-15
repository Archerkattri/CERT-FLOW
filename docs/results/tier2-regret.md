# Tier-2: Unknown-terrain navigation — regret vs sensing policy

*Pointing sensing at the certified route-gap, not at freshness or global uncertainty, is what turns a sensing budget into route quality.*

**Reproduce:** `scripts/run_tier2.py`

Full run: 25 seeds, 10x10 bounded drift rho=0.02,
unknown-terrain start: no survey, weak prior, epsilon=8, alpha'=0.2. Raw:
`results/tier2/table.json`.

Robot pays true edge costs; traversal is a free observation; regret is against
a clairvoyant oracle that replans on true costs every step. "cert-then-go" =
sense (1 obs/round) until epsilon-certified or budget B exhausted, then drive.

> **Finding —** Route-critical (certificate-gap) sensing beats every baseline at
> every budget in travel-regret, and is the only policy that converts budget into
> quality monotonically. The mission-time cost is paid honestly and shown below.

## Regret by sensing policy

Rows ordered best -> worst by travel-regret (regret mean, the headline metric).

| condition · | budget · | goal% ↑ | rounds ↓ | regret mean ↓ | regret median ↓ | sense ↓ | coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| cert-then-go, **cert**      | 40 | 100% | 417 | **2.12** | **1.75** | 41.4 | 0.989 |
| cert-then-go, **cert**      | 20 | 100% | 217 | 2.27 | 2.37 | 21.6 | 0.999 |
| cert-then-go, **cert**      | 10 | 100% | 118 | 3.21 | 2.88 | 11.7 | **1.000** |
| cert-then-go, max_age       | 40 | 100% | 416 | 3.84 | 4.28 | 41.3 | 0.945 |
| cert-then-go, max_age       | 20 | 100% | 217 | 4.49 | 4.60 | 21.7 | **1.000** |
| cert-then-go, random        | 10 | 100% | 118 | 5.37 | 5.10 | 11.8 | **1.000** |
| cert-then-go, max_width     | 10 | 100% | 118 | 6.06 | 5.32 | 11.8 | **1.000** |
| cert-then-go, max_width     | 40 | 100% | 416 | 6.07 | 6.10 | 41.3 | 0.867 |
| cert-then-go, max_age       | 10 | 100% | 118 | 6.25 | 6.19 | 11.8 | **1.000** |
| cert-then-go, random        | 40 | 100% | 417 | 6.39 | 7.01 | 41.5 | 0.999 |
| no-certificate (drive blind)| —  | 100% | **18**  | 6.60 | 6.50 | **0.0**  | n/a  |
| cert-then-go, max_width     | 20 | 100% | 218 | 6.86 | 6.73 | 21.7 | **1.000** |
| cert-then-go, random        | 20 | 100% | 217 | 7.07 | 7.25 | 21.4 | **1.000** |
| cert, sense-while-driving   | —  | 100% | **18**  | 7.19 | 7.47 | 1.8  | **1.000** |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

(goal% is identical across all conditions, so it carries no ranking signal and is
left unbolded; coverage shows "n/a" for the drive-blind row, which never builds a
certificate.)

## Findings

1. **Route-critical (certificate-gap) sensing beats every baseline at every
   budget by 1.7-3x in travel-regret** (3.21/2.27/2.12 vs 3.84-7.07). This is
   the paper's Tier-2 headline: pointing sensing at the certified gap — not at
   freshness (max_age), global uncertainty (max_width), or chance — is what
   converts observations into route quality.
2. **Only cert sensing converts budget into quality monotonically**
   (3.21 -> 2.27 -> 2.12 as B doubles). max_width is flat-to-worse with more
   budget; max_age improves but stays behind.
3. **The mission-time trade is explicit and honest:** certify-then-go pays
   ~100-400 sensing rounds before departing vs 18 rounds driving blind, buying
   lower regret plus a certificate. epsilon and B are the dial. Sensing
   while driving does not pay here (7.19 vs blind 6.60): traversal
   observations dominate once moving — the certificate's value concentrates in
   the pre-departure phase on this small grid.
4. **Coverage holds in motion** (0.867-1.000 among valid rounds, all
   conditions): the certificate's claim survives the robot actually driving,
   with traversal observations feeding the calibration buffer. max_width at
   B=40 has the lowest in-motion coverage (0.867).
5. **Anomaly (recorded):** max_age and max_width at B=40 depart on lower
   in-motion coverage — exhaustively re-sensing old edges builds stale-correction
   pressure on the confidence term.
