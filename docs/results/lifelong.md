# Tier-L: Lifelong operation — resolving objective O4/H1

*Across repeated missions, carried memory re-certifies markedly faster and
senses far less than a memoryless restart — the objective's system claim,
vindicated.*

**Reproduce:** `scripts/run_lifelong.py`

Full run (16 seeds x 8 missions x 5 memory variants; 6x6 bounded drift
rho=0.01, eps=5, unknown-terrain start, 50 idle drift rounds between missions;
first mission excluded — nothing can warm-start it). Raw:
`results/lifelong/table.json`.

Within-mission latency was H1's original claim and it failed honestly (D* Lite
reuse + pre-widening leave no room; that verdict stands). This experiment tests
the objective in its intended setting: repeated missions, where a memoryless
planner re-pays warm-up, re-learns every edge, and re-discovers corridors each
time.

> **Finding —** Carried memory re-certifies an epsilon-good route far faster
> and with far fewer sensing actions than a cold restart, and turns in valid
> claims from the first round. The trade is honest: stale beliefs certify a
> slightly worse route (higher regret), but always one provably within eps of
> optimal.

Rows ordered best -> worst on re-certification speed (`rounds->cert`, the
headline objective metric).

| variant · | cert-rate ↑ | rounds->valid ↓ | rounds->cert ↓ | sense->cert ↓ | regret~ ↓ |
|---|---:|---:|---:|---:|---:|
| full memory       | **100%** | **0.0** | **23.5** | **2.5** | 0.562 |
| full minus kappa  | **100%** | **0.0** | **23.5** | **2.5** | 0.371 |
| beliefs only      | **100%** | 10.0    | 51.0    | 5.2     | 1.082 |
| calibration only  | **100%** | **0.0** | 82.0    | 8.5     | **0.056** |
| memoryless        | **100%** | 22.0    | 101.0   | 10.5    | 0.172 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

## Findings

1. **O4 achieved in the lifelong setting: 4.3x faster re-certification
   (101 -> 23.5 rounds) and 4.2x cheaper sensing (10.5 -> 2.5) than a
   memoryless restart**, with instant claim validity (round 0 vs 22) from
   the carried calibration buffer.
2. **The ablation decomposes the memory:** calibration carryover buys
   instant validity (annealed claims from round 0) but not route knowledge
   (82 rounds to certify); beliefs buy the route knowledge (51 rounds) but
   not validity (10 rounds to first claim); together they compose fully.
3. **kappa's marginal speed contribution is zero** (full == full-minus-kappa
   on every speed/sensing column) — consistent with its established role:
   churn suppression and freshness-gate opening, not speed. H1's
   conductivity-specific latency claim remains failed; the memory SYSTEM
   claim (O4) is what the lifelong setting vindicates.
4. **Honest trade:** memory-carried incumbents certify at points slightly
   further from optimal (regret ~0.4-1.1 vs 0.17, all far inside eps=5) —
   stale beliefs certify the first epsilon-good route they can prove rather
   than re-exploring for the best one. The certificate's promise (within
   eps of optimal at the claimed confidence) holds for all variants.
