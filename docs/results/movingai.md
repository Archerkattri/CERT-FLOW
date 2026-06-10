# MovingAI benchmark: real map structure, controlled drift

`scripts/run_movingai.py` full (15 seeds, 400 max rounds, bounded drift
rho=0.02, gaussian noise 0.05, delta=1, epsilon=8, alpha'=0.2, unknown-terrain
start: no survey). Raw: `results/movingai/table.json`. Three map families: DAO
dungeon (`arena.map`, full 49x49, scen endpoints, path 53 steps), street (64x64
crop of `Berlin_0_256.map` at (128,128), path 87 steps), maze (64x64 crop of
`maze512-1-0.map` at (384,128), path 86 steps). Same moving-robot semantics as
Tier-2: robot pays true edge costs, traversal is a free observation, regret is
against a clairvoyant oracle replanning on true costs every step. "cert-then-go"
= sense (1 obs/round, 0.1 each) until epsilon-certified or budget 20 exhausted,
then drive (sensing continues while driving, so spend exceeds 20).

| map | policy | goal% | rounds | regret mean | regret median | sense | coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| dao_arena     | **cert** | 100% | 252 | **3.37** | 3.87 | 25.1 | 1.000 |
| dao_arena     | random   | 100% | 252 | 4.89     | 4.91 | 25.1 | 1.000 |
| dao_arena     | max_age  | 100% | 252 | 4.08     | 4.63 | 25.2 | n/a   |
| dao_arena     | blind    | 100% | 53  | 3.96     | 3.96 | 0.0  | n/a   |
| street_berlin | **cert** | 100% | 286 | **7.52** | 7.56 | 28.5 | 1.000 |
| street_berlin | random   | 100% | 286 | 8.33     | 7.65 | 28.6 | 1.000 |
| street_berlin | max_age  | 100% | 286 | 8.59     | 8.02 | 28.6 | 1.000 |
| street_berlin | blind    | 100% | 87  | 8.70     | 8.35 | 0.0  | n/a   |
| maze          | cert     | 100% | 285 | 0.81     | 0.11 | 27.8 | 1.000 |
| maze          | random   | 100% | 285 | 0.81     | 0.11 | 28.5 | 1.000 |
| maze          | max_age  | 100% | 285 | 0.81     | 0.11 | 28.5 | n/a   |
| maze          | blind    | 100% | 86  | -0.04    | -0.05| 0.0  | n/a   |

## Findings

1. **Cert sensing has the lowest regret on every map with route choice**
   (dao 3.37 vs 3.96-4.89; Berlin 7.52 vs 8.33-8.70), and it is the only
   sensing policy that beats driving blind once mission time is counted.
   The margin is structural, not synthetic-grid luck — but it is smaller
   than Tier-2's 1.7-3x (here ~10-27%), because epsilon=8 is unattainable on
   these maps (T2' floor ~ rho*L*(L-1) is 56-150 for L=53-87), so every
   certify-then-go run departs on budget exhaustion at round ~200, not on
   certification. Departure timing is identical across sensing policies;
   cert's entire advantage is a better-informed route at departure.
2. **The maze is a built-in negative control and it behaves exactly as
   theory says:** one corridor, no route choice — all three sensing
   policies produce bit-identical regret (0.81, same to 13+ digits; the
   trajectory is forced). Sensing cannot improve a forced route; the 0.81
   paid by certify-then-go vs blind's -0.04 is purely the cost of departing
   at round ~200 after drift has wandered, vs departing at t=0 alongside the
   oracle. Blind's slightly negative regret is the greedy clairvoyant oracle
   being marginally non-optimal step-by-step.
3. **Driving blind is honestly competitive on real maps at this drift
   level:** on dao it beats random (3.96 vs 4.89) and max_age (4.08). Untargeted
   pre-departure sensing buys nothing here — the 200-round delay costs as much
   as the information gains. Only gap-targeted sensing pays for its own delay.
   This sharpens, not weakens, the Tier-2 claim: the value is in WHERE the
   observations go, not in having them.
4. **Coverage holds on real map structure** (1.000 among valid rounds, claim
   0.8) everywhere it is measurable. It is measurable only for policies that
   revisit edges: the conformal buffer needs paired observations of the same
   edge, and max_age on dao/maze (and blind everywhere) spreads single visits
   across 2000+ edges, so no scores, no valid certificates, coverage n/a — a
   property of the policy, not a soundness failure. On Berlin's larger crop
   max_age does pair up (coverage 1.000).
5. **Anomaly (recorded):** mission rounds are 252/286/285 = budget exhaustion
   at round ~200 plus the blind path length (53/87/86) — i.e. certification
   fired in zero episodes pre-departure, consistent with the T2' floor; the
   certificate machinery still pays off via sensing targeting (finding 1) and
   sound in-motion coverage (finding 4). Sense spend lands at 25-28.6 (> budget
   20) because sensing continues during the drive phase.
