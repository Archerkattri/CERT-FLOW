# Tier-1: Incremental repair latency (T3 evidence)

`scripts/run_tier1_latency.py`, full (200 rounds/cell). Raw:
`results/tier1/table.json` (regenerable, not committed).

D* Lite incremental repair vs from-scratch Dijkstra after local cost
perturbations (multiply edges within Chebyshev radius r of a random locus by
U(0.5,2)). "moving" = start walks along the current path with r=2
perturbations between steps (the robot execution regime).

| scenario | size  | r  | inc p50 (ms) | scr p50 (ms) | speedup | mean pops | nodes |
|----------|-------|----|--------------:|--------------:|--------:|----------:|------:|
| static   | 20x20 | 1  | 0.20 | 0.27 | 1.35 | 108 | 400 |
| static   | 20x20 | 10 | 2.09 | 0.22 | 0.10 | 359 | 400 |
| moving   | 20x20 | 2  | 0.12 | 0.23 | 1.92 | 28  | 400 |
| static   | 40x40 | 1  | 0.29 | 1.21 | 4.16 | 186 | 1600 |
| static   | 40x40 | 10 | 4.43 | 1.29 | 0.29 | 972 | 1600 |
| moving   | 40x40 | 2  | 0.18 | 1.12 | 6.31 | 44  | 1600 |
| static   | 80x80 | 1  | 0.56 | 5.40 | 9.70 | 480 | 6400 |
| static   | 80x80 | 10 | 9.41 | 6.99 | 0.74 | 2922 | 6400 |
| moving   | 80x80 | 2  | 0.23 | 5.48 | 23.84 | 90 | 6400 |

(Full grid including r in {2,5} in table.json; abbreviated here.)

Findings:

1. **T3 confirmed:** repair cost (pops) tracks the perturbed region, not graph
   size. At fixed locality the speedup grows with |V|: 1.35x -> 4.16x -> 9.70x
   (static r=1), 1.92x -> 6.31x -> 23.84x (moving r=2).
2. **Honest boundary:** when the changed region approaches graph scale
   (r=10 on 20x20), incremental repair loses to scratch (0.10x). Incremental
   search pays off iff changes are local relative to the graph — this is the
   regime lazy pre-widening (PlannerConfig.prewiden_rounds) restores for CERT,
   since age-widening otherwise touches every edge every round.
3. **Correctness under sustained use:** incremental cost == scratch cost on
   every round of every cell (3000+ rounds, zero mismatches).

## External speed anchor (networkx)

`scripts/run_repeated_queries.py`, 3 seeds. The Tier-1
speedups above are stated against our own `dijkstra()`; this anchors that
reference to a third-party implementation so "fast" is not a self-comparison.
We time `networkx.shortest_path_length(G, s, t, weight="weight")` (Dijkstra)
on the *same* updated graph snapshots used for the repeated-query comparison,
same perf_counter convention (wrapped around the single call, ms, p50/p95).

| size  | engine dijkstra p50 (ms) | networkx dijkstra p50 (ms) | networkx p95 (ms) | engine/networkx |
|-------|--------------------------:|----------------------------:|-------------------:|----------------:|
| 20x20 | 0.18 | 0.25 | 0.38 | 0.72x |
| 40x40 | 0.86 | 1.17 | 1.82 | 0.73x |

(p50 over 3 seeds x 50 queries x 2 cost regimes; full table and the
E-Graphs / D* Lite repeated-query comparison in extern-baselines.md Part D.)

Finding: our `dijkstra()` is ~1.37x faster than networkx's mature Dijkstra
at both sizes (engine/networkx ratio ~0.72-0.73, i.e. networkx takes ~1.37x
longer). The absolute numbers are the *same order* as networkx, so the Tier-1
incremental-repair speedups (9.7-23.8x over this scratch baseline) are not an
artifact of a slow home-grown scratch planner — they hold against, and the
scratch baseline is even slightly faster than, a widely-used external
reference. Note this is the from-scratch row only; networkx has no incremental
update API, which is the entire point of D* Lite and is not a like-for-like
comparison axis.
