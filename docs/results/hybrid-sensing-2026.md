# Objective-matched hybrid sensing on real METR-LA

*Does the hybrid sensing policy beat pure gap-directed sensing where it matters —
on the real cost process, on route quality, at equal validity? Promotion check
for the scoreboard **Sensing** row.*

**Reproduce:** `scripts/run_hybrid_sensing.py` (`--quick` for a 3-seed smoke; the
full run needs the METR-LA replay in `data/`). Output:
`scripts/out/hybrid_real_metrla.json`.

> **Finding —** On real METR-LA traffic (10 seeds × 288 rounds, one replay day
> each, warm-up excluded) the objective-matched **hybrid** policy cuts median
> route regret **−41%** vs the shipped pure gap-directed default (38.3 s vs
> 65.1 s) at identical validity (0 violations) and identical per-round cost
> (~1 ms), and dominates the max-age / random baselines on both regret and
> width. Hybrid is the recommended sensing configuration; pure gap-directed
> stays the reproducible default for one release and remains documented as the
> policy that is beaten in the never-attainable-ε regime.

## Real METR-LA (10 seeds × 288 rounds, warm-up 12; oracle = exact Dijkstra on the recording)

| policy | regret mean (s) ↓ | median route regret (s) ↓ | valid % | viol % | gap median (s) | ms/rd |
|---|---:|---:|---:|---:|---:|---:|
| **hybrid (objective-matched)** | **114.4** | **38.3** | 98.9 | 0.00 | 10 247 | 1.0 |
| cert (pure gap-directed, shipped default) | 132.2 | 65.1 | 98.9 | 0.00 | 8 468 | 1.2 |
| max_age (freshness) | 131.6 | 80.8 | 98.9 | 0.00 | 55 170 | 0.5 |
| random | 182.3 | 116.3 | 98.9 | 0.00 | 44 325 | 0.4 |

*↓ lower is better · **bold** = best · all policies hold coverage at 0.0000 violations.*

## Reading

- **Where the win lives — the never-attainable-ε regime.** On the real cost
  process ε = 120 s is unattainable all day (certified fraction 0.0 for every
  policy; real-traffic gaps run ~8–10 k s). This is exactly the regime the
  hybrid was built for: when the certifiability threshold (T2′) says ε cannot
  close, gap-directed sensing spends its one-per-round budget on
  certificate-relevant but route-marginal edges, while hybrid redirects it
  toward the expected-best route. Measured price of that misallocation: pure
  gap's median regret 65.1 s vs hybrid's 38.3 s (**−41%**; mean −14%).
- **The honest cost, kept.** Hybrid's certified gap is **+21%** vs pure gap
  (10 247 vs 8 468 s). In the regime where that width difference exists, *no*
  width certifies at ε, so the tighter gap is not decision-relevant there while
  departure quality is. Where ε *is* attainable, hybrid switches back to
  gap-directed sensing by construction and converges to the default's behavior
  (the synthetic mixed-regime benchmark confirms: hybrid regret −0.12 ≈
  clairvoyant oracle — see [extern-baselines §B](extern-baselines.md)).
- **Certificate-compatible.** Hybrid never trades the certificate away: it is
  gap-directed whenever certification is achievable and redirects the budget
  only where the certificate provably cannot close. Unlike the killed TEAM-CERT
  decision-focused sensing (which changed *what* was certified), hybrid changes
  only *when* gap-sensing is pointless — so `(lb, ub, confidence)` is unaffected.
- **Config.** Enable via `PlannerConfig(hybrid_sensing=True)`, or use
  `recommended_config()` (which turns it on with online ρ + κ hysteresis +
  gated sum-aware UB). The `PlannerConfig` default stays `hybrid_sensing=False`
  for one release (reproducibility of the published numbers); the default flip
  is announced for the next minor.
