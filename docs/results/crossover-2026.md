# When do you need a certificate? — no quality crossover vs a fast uncertified replanner

*On static maps a from-scratch fast replanner wins outright on latency — that is
conceded (the scoreboard **speed** row is a FAIL by design). This experiment asks
the sharper question: is there any drift level at which you would accept the
uncertified planner's route/promises to buy that speed? The answer is no.*

**Reproduce:** `scripts/run_crossover_regret.py` (`--quick` for a 4-seed × 3-ρ
smoke). Output: `scripts/out/crossover_regret.json`; figure
`assets/crossover_regret.png`.

**Setup.** Two planners on identical ground-truth grid worlds per seed (12×12,
15 seeds, 220 rounds, 40 warm-up discarded), each sensing one edge/round:
**FAST** = last-observation point beliefs, max-age freshness sensing, from-scratch
Dijkstra, promising its believed cost; **CERT** = `CertPlanner` with
`recommended_config()`, promising `ub` when valid, else abstaining. Scored per
round: `regret = truecost(route) − OPT`; `overrun = max(0, truecost − promise)`
(the broken-promise magnitude); composite `J = regret + overrun`.

> **Finding —** There is no crossover to wait for. The certified planner's regret
> is **≤ FAST's at every drift level, including the static map** (ρ=0: 0.020 vs
> 0.036 — the fast planner's point estimates are optimistically biased even with
> no drift). FAST's point-estimate promise is exceeded on **62–97%** of rounds
> with the overrun magnitude growing 0.15 → 6.6 cost units as drift rises, while
> CERT's certified upper bound is **never** exceeded (overrun ≡ 0, 0 violations).
> The fast planner's entire advantage is **latency** (~2–4× faster per round,
> µs–ms class) — you pay ~ms/round for promises that actually hold.

## Regret and broken-promise overrun vs drift (15 seeds, pooled per ρ)

| ρ (drift/round) | FAST regret | CERT regret | FAST overrun | FAST promise-break rate | CERT overrun |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.036 | **0.020** | 0.154 | 79% | **0** |
| 0.005 | 0.055 | **0.053** | 0.219 | 62% | **0** |
| 0.01 | 0.233 | **0.194** | 0.310 | 64% | **0** |
| 0.02 | 0.775 | **0.478** | 0.581 | 68% | **0** |
| 0.05 | 2.974 | **1.502** | 2.054 | 84% | **0** |
| 0.10 | 6.378 | **2.523** | 6.598 | 97% | **0** |

*Composite J = regret + overrun: CERT wins at every ρ (0.020 vs 0.191 at ρ=0;
2.52 vs 12.98 at ρ=0.10). Crossover analysis: `rho_star = 0` (no interior
crossover — CERT ≤ FAST across the whole sweep).*

## Reading

- **The speed FAIL is unchanged and honest:** on raw per-round latency the
  uncertified planner is ~2–4× faster (µs–ms class); if all you need is a route
  and you never make a promise, use it.
- **But there is nothing to "cross over" to.** Even at ρ=0 the uncertified
  point-estimate promise is broken on most rounds (the winner's-curse optimism of
  a `min` over noisy estimates), and its regret is never lower than the certified
  planner's. The instant drift is measurable, the gap widens.
- **What you buy for the ~ms:** promises that hold (certified UB never exceeded,
  overrun ≡ 0) and honest abstention when the map is too stale to bound.
