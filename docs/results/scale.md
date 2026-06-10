# Full-loop CERT planner: scale benchmark

`scripts/run_scale.py`, full (3 seeds × 150 rounds/cell). Raw:
`results/scale/table.json` (regenerable, not committed).

World: `BoundedDriftWorld`, 4-connected directed grid, rho=0.02, noise_scale=0.05,
initial survey, start=(0,0), goal=(rows-1,cols-1), delta=1.0.

Configs measured:

| id | description |
|----|-------------|
| A | defaults: B=10, k_alternatives=3, rho_mode="given" |
| B | exact: B=0, k_alternatives=0 (no cache, no alternatives Dijkstras) |
| C | `recommended_config()`: online rho, hybrid sensing, kappa, adaptive_rate, sum_aware_ub |
| D | defaults + k_alternatives=0 (A minus alternatives; isolates the Dijkstra overhead) |

Per-round wall-clock p50/p95 measured with `time.perf_counter` around `planner.round()`.
Peak RSS from `resource.getrusage(RUSAGE_SELF).ru_maxrss` (Linux, kB; cumulative peak).

## Latency table

| size  | config | rounds | p50 (ms) | p95 (ms) | RSS (MB) | valid% | cert% | gap p50 |
|-------|--------|-------:|---------:|---------:|---------:|-------:|------:|--------:|
| 10x10 | A defaults B=10    | 450 |  0.67 |  1.86 | 103 |  83.6 | 0.0 | 34.66 |
| 10x10 | B exact B=0 k=0    | 450 |  1.84 |  2.31 | 104 |  86.4 | 0.0 | 33.04 |
| 10x10 | C recommended      | 450 |  1.64 |  2.23 | 104 |  85.6 | 0.0 | 16.01 |
| 10x10 | D defaults k=0     | 450 |  0.49 |  1.70 | 104 |  83.6 | 0.0 | 34.66 |
| 20x20 | A defaults B=10    | 450 |  1.46 |  7.06 | 116 |  61.1 | 0.0 | 91.89 |
| 20x20 | B exact B=0 k=0    | 450 |  7.21 |  9.48 | 116 |  58.2 | 0.0 | 89.60 |
| 20x20 | C recommended      | 450 |  7.71 | 13.37 | 117 |  46.4 | 0.0 | 47.03 |
| 20x20 | D defaults k=0     | 450 |  0.67 |  6.36 | 117 |  61.1 | 0.0 | 91.89 |
| 40x40 | A defaults B=10    | 450 |  2.54 | 28.92 | 168 |   0.0 | 0.0 |   inf |
| 40x40 | B exact B=0 k=0    | 450 | 31.02 | 43.83 | 168 |   0.0 | 0.0 |   inf |
| 40x40 | C recommended      | 450 | 34.85 | 45.93 | 170 |   0.0 | 0.0 |   inf |
| 40x40 | D defaults k=0     | 450 |  2.44 | 30.83 | 170 |   0.0 | 0.0 |   inf |
| 60x60 | A defaults B=10    | 450 |  4.34 | 69.60 | 250 |   0.0 | 0.0 |   inf |
| 60x60 | B exact B=0 k=0    | 450 | 64.06 |107.61 | 251 |   0.0 | 0.0 |   inf |
| 60x60 | C recommended      | 450 | 99.30 |123.59 | 257 |   0.0 | 0.0 |   inf |
| 60x60 | D defaults k=0     | 450 |  4.42 | 69.54 | 257 |   0.0 | 0.0 |   inf |

Sizes reference: 10x10 = 360 edges, 20x20 = 1520 edges, 40x40 = 6240 edges, 60x60 = 14160 edges.

## Findings

### 1. Where the time goes at 60x60

The dominant cost at 60x60 is the **full-graph interval refresh** (B=0 path),
not the sensing or conformal arithmetic.

Config A (B=10) p50 = **4.34 ms** vs Config B (B=0, k=0) p50 = **64.06 ms** — a 14.8x
gap attributable entirely to the pre-widening cache: with B=0 every round recomputes
all 14 160 edge metrics and pushes them to both D* Lite instances, while B=10 defers
those updates for 10 rounds and only pushes expired entries (~|E|/B ≈ 1 416 edges).

The k_alternatives contribution is cleanly isolated by comparing A vs D (both B=10):
A p50 = 4.34 ms, D p50 = 4.42 ms — less than 2% difference at 60x60. At 20x20 the
spread is slightly larger (A 1.46 ms, D 0.67 ms, ~2x) because the alternatives Dijkstra
actually completes before pre-widening dominates; at 60x60 the full-graph refresh cost
has grown enough to bury the k=3 Dijkstra overhead entirely.

Config C (recommended_config, B=10) costs more than Config B (B=0) at 60x60 despite
sharing the same pre-widening window: C p50 = 99.30 ms vs B p50 = 64.06 ms. The
difference (~35 ms) comes from the additional per-round work in recommended_config
(online rho estimation, sum-aware UB block-quantile, kappa scoring over all candidates,
hybrid VOI sensing) — these features are O(|E|) Python loops that are cheap on small
grids but accumulate at 60x60.

**Summary of dominant terms at 60x60, ordered:**
1. Full-graph metric refresh (B=0): ~60 ms/round, ~14x slower than B=10.
2. recommended_config feature overhead (online rho, sum-aware, kappa): ~35 ms/round additive.
3. k_alternatives Dijkstras (k=3 vs k=0): <1 ms at 60x60 — negligible.

### 2. Pre-widening locality story (connecting to tier1-latency)

Tier-1 showed that D* Lite repair cost tracks the changed region, not graph size, when
changes are local. The full-loop data confirms the same pattern from outside: Config A
(B=10) p50 grows from 0.67 ms (10x10, 360 edges) to 4.34 ms (60x60, 14 160 edges) —
a 6.5x increase for a 39x growth in |E|. The sublinear scaling is the pre-widening
cache doing its job: only ~|E|/B ≈ 1/10th of edges expire per round, so each round
touches a local patch rather than the full graph.

Without pre-widening (B=0), latency grows much closer to linearly with |E|:
10x10 → 1.84 ms, 20x20 → 7.21 ms (3.9x for 4.2x |E|), 40x40 → 31.02 ms (4.3x),
60x60 → 64.06 ms (2.1x). The slower-than-linear climb at 60x60 for B=0 is because
the D* Lite update is also incremental even when many edges change — bulk edge pushes
still batch-wake the priority queue rather than running separate Dijkstras.

### 3. Certificate health at scale (gap/validity)

The gap p50 grows with grid size and is infinite for >=40x40 because the conformal
quantile has not converged within 150 rounds: the calibration buffer effective mass
m grows as scores accumulate (one edge sensed per round), but the Bonferroni
per-edge level alpha/L shrinks with path length L which grows with grid diameter.
On a 40x40 grid the nominal path from (0,0) to (39,39) is ~78 edges, requiring
~78/alpha = 780 observations just to exit warm-up, far more than 150 rounds provide.

This is not a silent degradation: the planner correctly reports `cert.valid = False`
(confidence <= 0) for all rounds before the buffer fills. Certificate validity is the
right gating condition; the valid% column confirms the warm-up period is behaving
as expected.

The gap p50 of 34.66 (config A, 10x10) relative to epsilon=5.0 reflects the
Bonferroni UB structure: on a ~18-edge nominal path, the gap 2*L*q_eff ≈ 36 at a
moderate quantile q_eff ≈ 1 is the theoretically expected magnitude.

Config C (recommended) shows gap p50 = 16.01 at 10x10 (vs 34.66 for A), confirming
the tighter gap from online rho + hybrid sensing + sum-aware UB reported in prior ablations.
At 20x20 C gap p50 = 47.03 vs A gap p50 = 91.89 — the gap reduction holds.

### 4. RSS growth

RSS grows sublinearly with |E|: 103 MB (10x10, 360 edges) → 250 MB (60x60, 14 160 edges),
a 2.4x increase for a 39x |E| increase. Per-edge storage is O(1): EdgeBelief (~5 floats),
cache entries (2 floats + expiry), D* Lite node state. The main driver is Python object
overhead which amortizes at larger grids.

### 5. Honest boundary

At 60x60 with config A, p50 = 4.34 ms and p95 = 69.60 ms. The p95 spike — 16x larger
than p50 — occurs at the periodic full rebuild triggered when the conformal quantile
grows past the cache tolerance (5% band). A growing q forces a full |E|-edge refresh
and both D* Lite instances to absorb ~14 160 updates, which explains the long tail.
With B=0 (config B) there is no such spike structure: p95/p50 = 107.61/64.06 = 1.68,
because every round pays the same full-refresh cost.

**Practical recommendation:** for real-time use at 60x60 (14 160 edges), config A
(B=10) is recommended at 4 ms median; the 70 ms p95 tail can be further reduced by
widening the rebuild tolerance (currently 5%) or capping q growth rate. Config B
(>100 ms/round) is suitable only for offline certification.

### 6. Recommended-config overhead root cause (resolved)

The recommended-config overhead at scale was not the Python feature loops but
adaptive pre-widening collapsing to B=0 (full exact refresh + whole-graph D*
Lite repair, ~1M vertex updates per 40 rounds) in two regimes where exactness
buys nothing — epsilon unattainable (slack <= 0) and warm-up (q infinite,
certification not in play; on large graphs q stays infinite for ~2L rounds
because even annealing cannot support a per-edge level until the buffer holds
~L scores). Both now revert to the configured latency-optimal B; recommended-
config overhead vs defaults is ~1.5x p50 (was ~23x). The optimized latency
numbers are in the "Final latency" section below.

## Benchmark methodology (for all speed numbers in this repo)

Hardware: AMD Ryzen Threadripper PRO 7975WX, single process, single thread,
CPU only (no GPU anywhere in the stack; the planner is pure Python + heapq,
NumPy/SciPy only in statistics and world simulation). Timing:
time.perf_counter() wall-clock around planner.round() (full-loop numbers)
or around update_edges()+shortest_path() vs a fresh dijkstra() on the
identical graph (Tier-1 engine numbers); per-round samples -> p50/p95.
The oracle's ground-truth Dijkstra runs OUTSIDE the timed region. Memory =
getrusage peak RSS. No JIT, no warm-up trimming except where stated.

## Why CPU, not GPU — and where the parallelism actually was

D* Lite is inherently sequential (priority-queue pops with data-dependent
branching: the worst GPU workload); conformal quantiles operate on ~1e3
sample buffers (no batch dimension); the only GPU-shaped component would be
learned cost predictors (ours is a 3-coefficient ridge). GPU offers no
genuine benefit to this planner. The unexploited parallelism was the
benchmark harness running seeds sequentially on 1 of 64 cores: seeds are
embarrassingly parallel, and run_experiment(workers=N) (or
CERTFLOW_WORKERS=N) now executes them in processes — measured 13.7x sweep
speedup at 24 workers with bit-identical aggregates (asserted by test).
Per-round latency numbers are unaffected (still measured in-process).

## Flat-array engine (fastgraph)

`src/certflow/fastgraph.py` is a performance port of the planner's hot path
(`graphcore.DStarLite` + `dijkstra`) onto flat NumPy arrays: CSR forward/reverse
adjacency (`int32` indices, `float64` costs), `g`/`rhs`/keys in arrays, an
integer-id priority queue, and an optional numba `@njit` inner-loop kernel
(pure-array-Python fallback always present). Semantics are identical — verified
by `tests/test_fastgraph.py` (45 random graphs x 20 update batches + moving
start + edge cases, asserting `FastDStarLite cost == DStarLite cost == dijkstra
cost` exactly, plus `fast_metrics` equal to the `EdgeBelief.lower/upper` loop
bit-for-bit). numba 0.65.1 installs cleanly into `cert_env` and is used when
present.

Benchmark: 60x60 grid (3600 nodes / 14 160 edges), full-graph cost update +
recompute, p50/p95 over 200 rounds, `time.perf_counter` (same conventions as
above). numba kernel compiled out of the timed region (one warm round). Mean
over 5 seeds; the numba kernel is the speed source (the pure-Python flat path is
*slower* than the dict engine — per-element NumPy scalar access costs more than
dict access — so numba is required for the win, not a bonus).

Reproduce: `cert_env/bin/python -m certflow.fastgraph`.

### Two regimes, because the planner uses both

The planner repairs incrementally on local rounds (`update_edges` of the few
expired edges, B=10) but does a **scratch rebuild** on full-refresh rounds
(`_rebuild_searches` / B=0, which constructs a fresh `DStarLite` — this is the
"74 ms exact mode" / p95-spike path). They benchmark very differently:

| path | old (graphcore) p50 / p95 ms | FastDStarLite p50 / p95 ms | speedup |
|------|---:|---:|---:|
| incremental full-graph update + recompute | 34.3 / 43.1 | 6.6 / 7.7 | **4.9x** (4.8–5.1x) |
| scratch rebuild (`_rebuild_searches`, B=0) | 16.5 / 17.0 | 1.50 / 1.95 | **11.2x** (8.5–12.1x) |

`FastDijkstra` from scratch (target early-exit) lands at **5.0 ms** p50,
independent of update size — it always recomputes from the source.

### Where the >=10x is, and where it isn't

* **Scratch rebuild — 11.2x, goal met.** This is the planner's expensive path
  (the p95 spikes at 60x60 come from full refreshes triggering
  `_rebuild_searches`). A fresh `FastDStarLite` with a warm numba kernel and a
  pre-built `FlatGraph` solves 60x60 from scratch in **1.5 ms** vs **16.5 ms**
  for a fresh `graphcore.DStarLite`. It also beats the old plain `dijkstra()`
  (2.7 ms) and `FastDijkstra` (5.0 ms): D* Lite's overconsistent-only expansion
  on a goal-rooted search visits fewer vertices than a full Dijkstra sweep.

* **Incremental full-graph update — 4.9x, below 10x.** A full-graph update (all
  14 160 edges change in one batch) is the *worst case* for incremental repair:
  every vertex goes inconsistent, so the priority queue churns the whole graph
  and the per-pop bookkeeping dominates. The remaining time is exactly that —
  ~73 k `_update_vertex` re-evaluations and the heap operations behind them,
  now in the numba kernel but still O(|E| log|E|). On this input a scratch
  recompute is cheaper than incremental repair (1.5 ms < 6.6 ms), which is
  precisely why the planner already switches to `_rebuild_searches` when
  >30% of edges change — so the *operative* number for full-refresh rounds is
  the 11.2x scratch figure, not the 4.9x incremental one.

* **Local updates scale better.** At ~10% of edges changing per round (the
  B=10 steady state), the incremental `FastDStarLite` speedup rises to ~7–8x
  (repair tracks the changed region); at ~1% it is bounded by the start-side
  repair cost. The full-graph 4.9x is the conservative floor.

### Remaining time / honest boundary

The numba kernel removed per-call marshalling by holding the queue in
persistent flat arrays (`_init_search_numba`), which roughly halved the
incremental cost (12.7 ms -> 6.6 ms during development). What's left in the
incremental path is irreducible algorithmic work: a full-graph batch makes the
D* Lite queue O(|E|), and no data-layout change shrinks the number of
expansions. The pure-Python flat fallback is ~60 ms (slower than the dict
engine) and exists only for correctness when numba is absent — it is not a
speed path. `fast_metrics` is vectorized and bit-exact but its planner adoption
is left to integration (it replaces the per-edge `_metrics` loop).

## Final latency (fastgraph integrated, staggered + vectorized refresh)

End-to-end planner rounds with the flat-array engine (numba kernel), the
scratch-rebuild and headroom fixes, per-edge staggered pre-widening horizons
(expiry no longer synchronizes), and a vectorized gather/scatter due-subset
refresh:

| size | config | p50 | p95 | p99 |
|---|---|---:|---:|---:|
| 40x40 | recommended | 2.41 ms | 4.79 ms | 5.69 ms |
| 60x60 | recommended | 3.68 ms | 12.02 ms | 13.57 ms |
| 60x60 | defaults | 2.66 ms | 11.44 ms | 12.88 ms |

~23x p50 and ~9x p95 from the first scale measurement, single CPU core,
coverage gate 1.0000 after every change; medians 2-4 ms with worst rounds
~12-14 ms — inside real-time replanning budgets at every scale tested.

A cache/engine cost divergence in the integration was caught by the noise-free
Traversing-Mars degenerate test (gap frozen at 2.64 forever): the
scratch-rebuild shortcut returned early without pushing changed costs, and the
shared-flat constructor does not read costs from the adjacency, so rebuilds
resurrected stale arrays. Fixed by a vectorized cache->flat cost sync on
rebuild. The same investigation surfaced and fixed a sensing-policy hole
(unobserved edges carried near-zero gap-shrink score despite unbounded width;
they now dominate the score, as the gap decomposition demands). The degenerate
noise-free corners are the project's strongest integration tests.
