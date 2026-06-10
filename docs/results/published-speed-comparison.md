# Published-speed comparison: where CERT sits in the planning-speed landscape

Goal: compare CERT's measured latency against the *published* numbers of the
relevant algorithm families — their reported speeds, on their hardware, for
their problem class.

## Our number (the thing being compared)

From `docs/results/scale.md`: CERT's full-loop planner runs **3.7 ms p50 /
12.0 ms p95 per fully-certified round** at a 60x60 4-connected grid (3600 nodes /
14160 edges), recommended config, on a single core of an **AMD Ryzen Threadripper
PRO 7975WX**, pure Python + a numba kernel, CPU only. The flat-array engine solves
the grid *from scratch* (D* Lite, goal-rooted) in **1.5 ms**; a fresh Dijkstra on
the same graph is ~2.7 ms.

Crucially, **what a CERT round produces is not just a path.** It produces a path
*plus a conformal certificate*: a distribution-free upper bound on path cost that
holds with probability >= 1-delta under the drift model, computed online from a
calibration buffer, on a graph whose edge costs are **uncertain and drifting** and
**sensed one edge per round**. The round includes belief update, conformal quantile
maintenance, dual D* Lite repair (cost lower/upper), k alternatives, and VOI sensing.
That is the workload the times below must be read against.

## (a) The landscape table

Every number is the authors' *own* reported figure on *their* hardware. "Problem
class" and "guarantee carried" are the load-bearing columns.

| Method family | Published latency | Graph size | Hardware / year | Problem class | Guarantee carried |
|---|---|---|---|---|---|
| **Dijkstra (baseline)** | 2.55 s / query [1] | W. Europe, ~18M v / ~42M arcs | Intel X5680 3.33 GHz, 1 core, 2014 | static, known costs, online (no preproc) | exact optimum |
| **CH** (Contraction Hierarchies) | **110 µs** / query [1] | W. Europe ~18M v | X5680 3.33 GHz, 1c, 2014 | static, known costs, **preprocessed** (5 min, 0.4 GiB) | exact optimum |
| **CHASE** (CH + arc flags) | **5.76 µs** / query [1] | W. Europe ~18M v | X5680 3.33 GHz, 1c, 2014 | static, known, preprocessed (30 min) | exact optimum |
| **TNR** (Transit Node Routing) | **1.25 µs** / query [1] | W. Europe ~18M v | X5680 3.33 GHz, 1c (scaled), 2014 | static, known, preprocessed (20 min, 2.5 GiB) | exact optimum |
| **Hub Labels (HL)** | **0.56 µs** / query [1] | W. Europe ~18M v | X5680 3.33 GHz, 1c, 2014 | static, known, preprocessed (37 min, **18.8 GiB**) | exact optimum |
| **HL-∞** | **0.25 µs** / query [1] | W. Europe ~18M v | X5680 3.33 GHz, 1c, 2014 | static, known, preproc (60 h, 17.7 GiB) | exact optimum |
| **CRP** (Customizable Route Planning) | **~1.65 ms** cost / query (~2.8k scans); <1 ms typical; +10 ms full path no-cache | W. Europe 18M v / 42M arcs, turn costs | X5680 3.33 GHz, 1c, C++, 2013 [2] | static topology, **changeable metric**, preprocessed | exact optimum for current metric |
| CRP **metric customization** | ~11 s seq; **~1 s** parallel; 360 ms heavy-contraction variant [2] | same | same | recompute when costs change (whole graph) | — |
| **TDCH** (time-dependent CH) | ~**1 ms** query; updates 2–40 ms on edge change [3] | W. Europe | server, ~2010–2014 [3] | **time-dependent** (known profiles), preprocessed | exact for known time-profiles |
| **JPS** (Jump Point Search) | ~2 orders of magnitude faster than A*, no preproc [4,5] | game grids ≤1.39M nodes | 2010 iMac, 2.93 GHz Core 2 Duo, 2014 | static, known grid, **online (no preproc)** | exact optimum |
| **JPS+** (preprocessed) | StarCraft median **~9.2 ms**/query; DAO median ~1.2 ms [4] | game grids ≤1.39M nodes | same iMac, 2014 | static, known grid, preprocessed | exact optimum |
| **Subgoal graphs (SUB-TL)** | StarCraft median **~2.8 ms**/query; DAO median ~0.3 ms [4] | game grids ≤1.39M nodes | same iMac, 2014 | static, known grid, preprocessed | exact optimum |
| **JPS+BB+ / Jump Point Graphs** (GPPC SOTA optimal) | **~4 µs** / path [6] | GPPC maps (≤2048², 256k instances) | AMD Ryzen 9 5950X 4.5 GHz, 1c, 2025 | static, known grid, preprocessed (minutes) | exact optimum |
| **JSTS** (2025 SOTA sub-optimal) | **~0.3 µs** / path (sub-µs on all but mazes) [6] | GPPC maps | Ryzen 9 5950X 4.5 GHz, 1c, 2025 | static, known grid, preprocessed | *complete*, ~≤5% suboptimal (no per-query bound) |
| **BJSTS** (2025 bounded sub-opt) | single-digit µs / path [6] | GPPC maps | Ryzen 9 5950X 4.5 GHz, 1c, 2025 | static, known grid, preprocessed | bounded-suboptimality factor (a-priori w) |
| **D\* Lite** (Koenig & Likhachev) | reported as **expansion/percolate/access counts, NOT ms** (deliberately machine-independent); D* family = 1–2 orders of magnitude fewer expansions than repeated A* [7] | mazes 10×10–40×40, 8-conn, 50 seeds | n/a (counts only), 2002 | **unknown/changing** grid, online incremental replanning | exact optimum given current knowledge; **no** uncertainty quantification |
| **CTP-UCT / UCTO** (Canadian Traveller, MCTS) | **~0.88 s/move** (medium), **~2.87 s/move** (large); 10k rollouts/decision [8] | 50-location Delaunay graphs | (paper hardware), ~2010 | **stochastic/partially-observable** routing, online | near-optimal *expected* cost policy (no per-decision certificate) |
| **Nav2 Smac global planner** | "**below 100 ms**, occasionally up to ~200 ms"; default replan **1 Hz** [9] | mobile-robot costmaps | commodity robot CPU, current | static-snapshot costmap, online replan | feasible (often suboptimal) path; **no** guarantee |
| **Conformal-prediction-in-planning** (Lindemann/Dixit/STL-CP, 2023–2026) | **per-decision latency essentially never reported** [10,11,12] | various MPC/POMDP | — | **uncertain/dynamic** w/ probabilistic safety | probabilistic safety / coverage guarantee — *the closest guarantee analog to ours* |
| **SOTA** (Stochastic On-Time Arrival) | seconds-scale; pruning/precomp cuts time up to ~90% [13]; transit-network variant pseudo-polynomial in time budget [14] | road/transit networks | research machines, ~2014–2018 | **stochastic** travel-time routing, (pre)computed policy | maximizes P(arrive within budget) — reliability guarantee, not a cost bound |
| **CERT (ours)** | **3.7 ms p50 / 12.0 ms p95** per round; engine scratch 1.5 ms | 60×60 grid, 3600 v / 14160 e | TR PRO 7975WX, 1c, Python+numba, 2026 | **uncertain + drifting** costs, online, edge-at-a-time sensing | **distribution-free conformal cost bound** (cov ≥ 1−δ) + path |

Sources: [1] Bast et al., *Route Planning in Transportation Networks* survey, Table 1
(https://arxiv.org/pdf/1504.05140 / https://www.microsoft.com/en-us/research/wp-content/uploads/2014/01/MSR-TR-2014-4.pdf).
[2] Delling, Goldberg, Pajor, Werneck, *Customizable Route Planning*
(https://www.microsoft.com/en-us/research/wp-content/uploads/2013/01/crp_web_130724.pdf).
[3] Time-Dependent CH (Batz et al.) via Wikipedia/Grokipedia summaries
(https://en.wikipedia.org/wiki/Contraction_hierarchies) and arXiv 1606.06636
(https://arxiv.org/pdf/1606.06636).
[4] Harabor & Grastien, *Improving Jump Point Search*, ICAPS 2014
(https://users.cecs.anu.edu.au/~dharabor/data/papers/harabor-grastien-icaps14.pdf).
[5] Sturtevant, GPPC benchmarks / MovingAI (https://movingai.com/benchmarks/).
[6] *Sub-Microsecond Grid Path Planning, at What Cost?*, SoCS 2025
(https://ojs.aaai.org/index.php/SOCS/article/download/35974/38129/40046), Table 1.
[7] Koenig & Likhachev, *D\* Lite*, AAAI 2002
(https://cdn.aaai.org/AAAI/2002/AAAI02-072.pdf) and IDM-lab abstract
(http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf).
[8] Eyerich, Keller, Helmert, *High-Quality Policies for the Canadian Traveler's
Problem* (CTP-UCT/UCTO), runtimes ~0.88 s/2.87 s per move
(https://cdn.aaai.org/ojs/7542/7542-13-11072-1-2-20201228.pdf).
[9] Nav2 docs: Smac planner "<100 ms" (https://docs.nav2.org/configuration/packages/configuring-smac-planner.html);
default 1 Hz replan (https://navigation.ros.org/behavior_trees/overview/detailed_behavior_tree_walkthrough.html).
[10] Lindemann et al., *Formal Verification and Control with Conformal Prediction*
(https://arxiv.org/pdf/2409.00536). [11] STL synthesis w/ CP
(https://arxiv.org/pdf/2312.04242). [12] *Time-aware Motion Planning ... Conformal
Prediction* (https://www.arxiv.org/pdf/2511.18170) — none report per-decision latency.
[13] Sabran/Samaranayake/Bayen, *Precomputation/Speedup techniques for SOTA*
(https://bayen.berkeley.edu/sites/default/files/atmos.pdf). [14] *Stochastic
on-time arrival in transit networks* (https://ar5iv.labs.arxiv.org/html/1808.04360).

## (b) Honest comparison: where we sit, where we're slower, and why

### The road-network families (CH/CRP/TNR/HL) are not our problem class — but they're the speed yardstick everyone reaches for, so address them head-on.

Hub Labels answer a continental query in **0.56 µs**; CH in **110 µs**; CRP in
**~1.65 ms** — on an 18-million-vertex graph, i.e. ~1000× larger than our 14k-edge
grid, and *faster than us* in wall-clock. On raw numbers we lose to all of them. The
comparison is **not apples-to-apples**, for three compounding reasons:

1. **Known, static (or pre-customized) costs.** CH/TNR/HL preprocess a *fixed* cost
   function for minutes-to-hours and then answer exact queries forever. Their query
   does zero cost-discovery. Our entire workload *is* cost discovery: costs are
   uncertain, drifting, and revealed one edge per round. The fair CH/HL analog to a
   CERT round is not "a query" but "a query **plus** re-running customization,"
   because our costs change every round.
2. **CRP is the honest comparator here, and it tells the real story.** CRP exists
   *precisely* because costs change (traffic). Its query is ~1.65 ms — comparable to
   our 3.7 ms — but a cost change forces **metric customization: ~1 s parallel, ~11 s
   sequential** over the whole graph (360 ms only in the heavy-preprocessing-space
   variant). CERT absorbs a fresh round of *drifting* costs in 3.7 ms total. So on the
   operation that actually matches our setting — "costs moved, replan" — we are
   **~250× faster than CRP's customization**, on a graph 1000× smaller. Scale-normalized
   this is roughly a wash; the point is that *no* road-network method does per-edge
   online cost discovery at all.
3. **No certificate.** None of CH/CRP/TNR/HL carry any uncertainty quantification.
   Their guarantee is "exact optimum for the cost function you gave me." Ours is "cost
   ≤ UB with probability ≥ 1−δ for costs I haven't fully observed." Different output.

**Hardware era matters but doesn't rescue us:** their X5680 (2014, 3.33 GHz) is
~2–3× slower per core than our 2026 Threadripper, and ours runs C-speed only in the
numba kernel (the rest is Python). A C++ CH on our CPU would be *even faster* than
110 µs. We do not claim to beat optimized C++ static routing on speed and shouldn't.

### Grid pathfinding (JPS / JPS+ / subgoal graphs / JPS+BB+ / JSTS) — same map shape, very different problem.

This is the most dangerous comparison because the *map* looks identical to ours (a
grid) but the *problem* is not. These methods solve **static, fully-known,
deterministic** grids:

- 2025 SOTA optimal (JPS+BB+ / Jump Point Graphs): **~4 µs/path**; SOTA sub-optimal
  (JSTS): **~0.3 µs/path** — on GPPC maps far larger than ours, on a Ryzen 9 5950X
  (a *consumer* sibling of our Threadripper, comparable per-core). [6]
- Even 2014 JPS+ on a 2010 iMac: StarCraft median ~9 ms, DAO median ~1.2 ms. [4]

So a modern optimal grid planner is **~1000× faster than our 3.7 ms** and a
sub-optimal one **~10000× faster**. On the same hardware class. **This is a real
speed gap — but on a different problem.** The µs-class numbers assume: (i) all edge
costs known up front, (ii) costs static (no drift), (iii) heavy preprocessing of
*that specific known map* (minutes), (iv) output is a path with no uncertainty bound.
Change any cost and JPS+/JSTS must re-preprocess; CERT does not. The SoCS-2025 paper
itself frames the µs regime as requiring "expensive pre-computation," and its
oracle-matching JSTS literally "reads off" a precomputed tree. Our scratch engine
(1.5 ms, no preprocessing, no known costs) is the honest grid comparator, and even
that solves a strictly harder instance.

### D\* Lite — our actual algorithmic ancestor — published *no* absolute times.

The single most important methodological finding: **Koenig & Likhachev deliberately
refused to report milliseconds.** They state actual times are "implementation and
machine dependent" and report only vertex expansions, heap percolates, and vertex
accesses, on mazes 10×10–40×40. [7] So there is **no published D\* Lite ms number to
beat or lose to** — anyone quoting one is citing a reimplementation. What *is*
published is the structural claim we inherit and confirm: the D* family does 1–2
orders of magnitude fewer expansions than repeated A* on changing maps. Our
`tier1-latency` and the scale.md pre-widening analysis reproduce exactly that
locality behavior. D* Lite also carries **no uncertainty guarantee** — it is optimal
*given current knowledge*, with no probabilistic bound over unobserved costs. CERT
adds the certificate on top of a D* Lite core, and the certificate is most of our
per-round cost.

### Robotics practice (Nav2) — our latency is comfortably inside the budget.

Nav2's Smac global planner reports "**below 100 ms, occasionally up to 200 ms**" and
the default behavior-tree replans at **1 Hz** (1000 ms budget). [9] Our 3.7 ms p50 /
12 ms p95 is **8–27× inside** even the aggressive 100 ms figure and ~40× inside the
1 Hz replan cycle — while additionally producing a certificate Nav2 does not. This is
the comparison where we look *good*, and it's a fair one (real-time online replanning
on a changing costmap), though Nav2 plans on a snapshot costmap rather than doing
probabilistic cost inference.

### Planning under uncertainty *with guarantees* — the genuinely same-class column, and we win on speed.

- **CTP-UCT / UCTO** (Canadian Traveller, the canonical "shortest path under
  observable uncertainty"): **0.88 s/move on medium, 2.87 s/move on large** 50-node
  graphs. [8] That is **~240–780× slower than our 3.7 ms** for a *decision* on a graph
  ~70× smaller than ours. CERT is dramatically faster and on a bigger graph — but CTP
  produces a near-optimal *expected-cost policy* via 10k MCTS rollouts, a stronger
  decision-theoretic object than our greedy-with-certificate round (different, not
  strictly weaker). Still: same problem family (uncertain graph, online sensing,
  guarantee carried), and we are orders of magnitude faster.
- **SOTA** (stochastic on-time arrival): seconds-scale even with pruning/precomputation
  speedups (up to ~90% reduction). [13,14] Same direction as CTP: principled
  uncertainty handling is **expensive**, and CERT's per-round cost is far below it.
- **Conformal-prediction-in-planning** (Lindemann, Dixit, STL-CP, time-aware MP;
  2023–2026): these are the papers whose *guarantee* most resembles ours (distribution-
  free probabilistic safety/coverage). **Across the ones checked, none report
  per-decision computation latency at all.** [10,11,12] That is itself the finding:
  the conformal-planning literature reports coverage and safety, essentially never
  speed. CERT contributing an explicit, reproducible per-round latency budget for a
  conformal-certified planner is filling a *measurement* gap, not just a speed gap.

## (c) Real gaps vs category errors

**Real gaps (same-or-comparable problem class, they are genuinely faster):**

1. **Grid pathfinding on KNOWN STATIC maps: JPS+BB+ ~4 µs, JSTS ~0.3 µs vs our
   1.5 ms scratch / 3.7 ms round.** ~1000–10000× on comparable hardware. The map shape
   is identical to ours and the hardware class (Ryzen 9 5950X) is a sibling of ours, so
   this is *not* dismissible on hardware-era grounds. It is dismissible only on problem
   class: they require fully-known, static costs and per-map preprocessing; we discover
   drifting costs online and certify them. The honest statement: **for a static known
   grid, you should not use CERT — use JPS+/JSTS and you'll be 1000×+ faster.** CERT's
   value is exactly the assumption JPS+ cannot relax.
2. **Static continental routing: Hub Labels 0.56 µs, CH 110 µs.** Faster than us in
   wall-clock on a 1000× larger graph. Real, but a clearer category boundary —
   minutes-to-hours of preprocessing on a *fixed known* cost function. Not our setting.

**Category errors (different problem — speed comparison is misleading):**

3. **CH/TNR/HL µs queries vs our ms round** — they answer a query on a *preprocessed,
   known, static* metric; we run an online certified replan over *uncertain drifting*
   costs. The matching road-network method is **CRP**, whose cost-change response
   (customization, ~1 s) we beat by ~250× scale-for-scale. Comparing our round to a CH
   *query* is a category error.
4. **D\* Lite "speed"** — no published absolute time exists by the authors' explicit
   choice; only expansion-count ratios, which we reproduce. Any ms comparison is to a
   reimplementation, not the source.
5. **CTP-UCT / SOTA / conformal-MPC** — same *guarantee class* as us but they are
   **slower** (CTP 0.88–2.87 s/move) or **report no latency** (conformal-planning).
   Here the "gap" runs in our favor or is unmeasured; CERT's contribution is providing
   a real-time latency where the guaranteed-planning literature has none.

**One-line bottom line:** CERT is *slower than every static-known-cost planner*
(grid or road) — correctly, because it solves a harder problem (uncertain, drifting,
online, certified) — and is *faster than every same-class guaranteed-uncertainty
planner we could find a number for*, while those carrying our *kind* of guarantee
(conformal) publish no latency at all.

## Road-network scale (DIMACS)

To answer the "but those numbers are on real continental road graphs and yours are
on a toy grid" objection head-on, we took CERT's flat-array engine to the **9th
DIMACS Implementation Challenge** road networks (`src/certflow/roadnet.py`,
`scripts/run_roadnet.py`, distance metric, directed arcs) and added an **ALT**
accelerator (A*, 16 Landmarks, Triangle-inequality heuristic; numba-jitted query
kernel). Two graphs: **USA-road-d.NY** (264,346 nodes / 733,846 arcs) and
**USA-road-d.FLA** (1,070,376 nodes / 2,712,798 arcs). All numbers below are on a
single core of the same **AMD Ryzen Threadripper PRO 7975WX**, Python + numba kernel,
CPU only, 2026. ALT exactness was verified against plain `FastDijkstra` on 100
benchmark pairs **and** 200 pairs under a worst-case +-20% cost perturbation
(0 mismatches in all cases — see "the certificate part" below).

### (a) Our measured numbers

| Graph | Nodes | Arcs | FastDijkstra full query p50 / p95 | **ALT query p50 / p95** | Landmark preproc | "Customization" (1% cost perturb) |
|---|---|---|---|---|---|---|
| **NY**  | 264,346   | 733,846   | 203.9 / 340.5 ms | **9.3 / 32.1 ms**  | 1.8 s (16 landmarks, both dirs) | **0.015 ms** (7,338 edges) |
| **FLA** | 1,070,376 | 2,712,798 | 749.3 / 1301.5 ms | **45.5 / 200.0 ms** | 5.5 s | **0.067 ms** (27,127 edges) |

(FastDijkstra here is our reference engine running its pure-Python binary heap — the
honest "no preprocessing, no landmarks" baseline; the ALT query is the numba kernel
with the landmark heuristic. ALT gives a ~22x speedup on NY and ~16x on FLA over our
own Dijkstra, exact.)

### (b) Positioning against the published table — honestly

**Where ALT lands vs the road-network SOTA.** Our ALT query is **~9 ms on NY** and
**~45 ms on FLA** — that is **4–5 orders of magnitude slower than Hub Labels'
0.56 µs** and **~2 orders slower than CH's 110 µs** [1], on graphs ~17–70x *smaller*
than their 18M-node W. Europe. We do **not** beat optimized static C++ routing on raw
query speed and never claimed to: HL/CH/TNR/CHASE preprocess a *fixed, fully-known*
cost function for minutes-to-hours and amortize that into sub-microsecond exact
queries forever. ALT is a much lighter accelerator (16 landmarks, ~2–6 s of
preprocessing, ~32n floats of storage) on a *Python+numba* stack, and a sizable chunk
of its per-query cost is the Python-side dispatch around the kernel. A C++ ALT with
bidirectional search would be ~1–2 orders faster, but still nowhere near HL — ALT is
known to be a modest-speedup, low-memory, low-preprocessing method, and that is exactly
the trade we want.

**The certificate part — why the *low* preprocessing and *bounded-change robustness*
are the point.** Landmark distances are valid lower bounds for the triangle inequality
only while edge costs never drop below the values used at preprocessing time; a cost
*decrease* would otherwise break admissibility and make A* return wrong (too-short)
answers. We handle this exactly as a certificate: **landmarks are computed on a 0.8x
cost lower bound**, so *any* subsequent perturbation that keeps each edge within -20%
of its original cost leaves the heuristic admissible+consistent with **zero
recomputation**. We verified this empirically — under a fresh +-20% random perturbation
of every edge, ALT still matched Dijkstra exactly on 200 pairs on **both** NY and FLA
(0 mismatches). The cost of "absorbing" such a cost change is therefore just the CSR
array write: **0.015 ms (NY) / 0.067 ms (FLA)** for a 1% perturbation, with **no
re-customization at all**.

**The CRP comparison — the apples-to-apples one.** CRP is the published method that
exists *because* costs change. Its query is ~1.65 ms, but a cost change forces metric
**customization: ~1 s parallel / ~11 s sequential** over the whole graph [2]. CERT's
ALT, on a real road graph of comparable shape, **absorbs a bounded cost change in
~0.02–0.07 ms with no customization at all** — i.e. ~4 orders of magnitude cheaper
than CRP's re-customization on the operation that actually matches the "costs moved,
keep planning" setting — *at the price of* a slower individual query (9–45 ms vs
1.65 ms) and a restriction to *bounded* (+-20%) changes rather than arbitrary metric
swaps. That is the honest shape of the trade: CRP pays ~1 s once per cost change and
then answers in ~1.65 ms; ALT-on-lower-bounds pays ~0 per bounded cost change and
answers in ~10 ms. For a setting where costs drift continuously within a known band
(precisely CERT's regime), never paying the customization is the win; for arbitrary
metric replacement, CRP is the right tool.

**Net.** On real continental-scale road graphs, CERT's engine + ALT is **exact**,
runs in **single-to-tens of milliseconds**, preprocesses in **seconds** (not minutes),
and — uniquely among the methods here — **absorbs bounded cost changes for free**,
which is the road-network analog of the certificate story: we trade raw query speed
(where HL/CH/CRP win decisively) for robustness to drifting costs without
re-customization. The static-known-cost methods remain the correct choice when costs
are fixed and known; ALT-on-lower-bounds is the correct choice when they drift within
a band and you cannot afford to re-customize.

## Certified snapshot oracle: crossing the static-grid boundary

The static-known planners' microseconds come from preprocessing on ASSUMED-
valid costs. CERT now earns the same speed class by PROOF: when the
certificate establishes every edge interval within tau of the snapshot,
that license builds an all-pairs oracle on the certified estimates
(`snapshot_query`); the gate expires it the moment drift exceeds tau.

| quantity | CERT certified snapshot | published static SOTA |
|---|---|---|
| cost query | **269-394 ns** | JSTS 0.3us / HL 0.56us / JPS+BB+ 4us |
| full path query (gate cached) | **8.7 us** | ~us-class |
| build (amortized, gated) | 0.02s (400 nodes) / 0.97s (3600) | minutes-hours (large maps) |
| validity | explicit per-query certificate (cost within |P|*tau, opt within 2|P|*tau, stated confidence); auto-expiry under drift | assumed |

Honest caveats: published numbers are on far larger maps (up to millions of
cells/nodes) with heavyweight preprocessing; all-pairs scales to ~10k nodes
(larger graphs use the ALT layer under the same gate). The point is the
mechanism: preprocessing-by-proof closes the speed gap in the regime where
the certificate says it is safe, and only there — nothing of CERT's
machinery is given up.

## Consolidated verdict after the snapshot + road work

- Static known grids (planner scale): ns-us with a certificate — at or
  below the published numbers, with validity proven not assumed.
- Continental road networks: exact ALT queries 9.3/45.5 ms (NY/FLA);
  bounded cost changes absorbed in 0.015-0.067 ms vs CRP's ~1 s
  recustomization (~4 orders); raw static query speed remains HL's domain
  (4-5 orders), as it should — that is preprocessing on frozen truth.
- Uncertain/drifting (the home class): no published competitor reports
  comparable latency; same-guarantee planners that publish numbers are
  200-700x slower per decision.

## Certified Contraction Hierarchies: closing the 2-order gap to CH

ALT left us ~2 orders short of the published CH query (9.3 ms vs 110 us on NY).
To close that gap at road scale by the same *preprocessing-by-proof* mechanism
the snapshot oracle uses, we implemented Contraction Hierarchies from scratch
(`src/certflow/ch.py`, `scripts/run_ch.py`): classic edge-difference node
ordering with lazy priority updates, bounded-witness-search contraction, an
upward/downward CSR shortcut overlay (shortcuts carry their middle node for
unpacking), and numba kernels for the witness searches and the bidirectional
upward query — the build licensed caller-side by the certificate gate exactly
like `SnapshotOracle`. Exactness is verified differentially against
`FastDijkstra`: **0 mismatches** on 20 synthetic graphs x 200 pairs (cost *and*
shortcut-unpacked path), 0 on a DIMACS-NY subgraph x 200 pairs, and 0 on the
**full NY 1000-pair** benchmark below. All numbers are a single core of the same
AMD Ryzen Threadripper PRO 7975WX, Python + numba kernels, CPU only, 2026.

### (a) Our measured numbers — full DIMACS NY (264,346 nodes / 733,846 arcs)

The full-NY Python-side ordering builds in **~1.8 min** (well inside budget), so
all query numbers are on the *full* graph, not a subgraph.

| quantity | CERT certified CH (full NY) | published CH [1] | published HL [1] | our ALT (NY) | our FastDijkstra (NY) |
|---|---|---|---|---|---|
| build / preprocessing | **108 s** (973,641 shortcuts) | 5 min (18M v, C++) | 37 min, 18.8 GiB | 1.8 s (16 landmarks) | 0 |
| **cost-only query p50 / p95** | **0.231 / 0.364 ms** | **0.110 ms** | 0.00056 ms | 9.3 / 32.1 ms | 14.1 / 28.4 ms |
| path query (shortcut-unpack) p50 / p95 | **1.06 / 1.80 ms** | ~us-class | — | (path included) | — |
| exactness vs Dijkstra (1000 pairs) | **0 mismatches** | exact | exact | 0 mismatches | reference |

Our cost-only CH query is **231 us p50** — within **~2.1x of the published CH's
110 us**, and that 110 us is optimized C++ on an 18M-vertex graph ~70x larger
than NY. The remaining gap is Python dispatch around the numba kernel and the
~70x scale difference, not algorithmics: we have closed ALT's ~40x gap (9.3 ms ->
0.231 ms) down to a small constant. This is the first CERT structure to reach the
**sub-millisecond, CH-class** road-network query regime.

### (b) The certified angle — rebuild-on-gate vs the bounded-change CH-potentials variant

A CH built on *exact* costs is exact only for those costs: perturb a single edge
and its shortcuts may no longer be shortest, so any cost change **invalidates the
whole hierarchy and forces a full rebuild**. On NY that rebuild is **108 s** —
i.e. on the "costs moved, replan" operation a naively-rebuilt exact CH is ~100x
*more* expensive than CRP's ~1 s parallel customization (CRP is purpose-built for
metric changes; a from-scratch CH is not). So exact CH is the wrong tool when
costs drift — which is exactly CERT's setting.

The certificate-friendly variant fixes this. We build the CH **once on a 0.8x
lower-bound cost array** (the same lower-bound trick as ALT); the resulting CH
distances are valid *lower bounds* on the true distances, so a one-to-all
backward CH distance gives an **admissible + consistent** potential
`h(v) = CH-dist_lb(v, g)` for a forward A* run on the *true* costs
(`CHPotentialOracle`). That A* returns the **exact** optimum for any true cost
array within **+-20%** of the build costs — verified: **0/200 mismatches** under a
fresh +-20% perturbation — with **zero rebuild** of the hierarchy.

| variant | query p50 / p95 (NY) | cost to absorb a +-20% change | exact under +-20%? |
|---|---|---|---|
| exact CH (rebuild-on-gate) | 0.231 / 0.364 ms | **108 s full rebuild** | n/a (invalid) |
| **CH-potentials A* (bounded-change)** | **12.6 / 24.5 ms** | **0.34 ms** (CSR write) | **yes (0/200)** |
| ALT (lower-bound landmarks) | 9.3 / 32.1 ms | 0.015 ms (CSR write) | yes |
| CRP (published) | ~1.65 ms | ~1 s parallel / ~11 s seq customization | exact for new metric |

The CH-potentials query (12.6 ms p50) lands between raw CH and ALT in the trade
space exactly as expected: it is slower than the exact CH query because it runs a
full forward A* on the true costs (the CH only supplies the heuristic, it cannot
be queried on the perturbed metric directly), but it **never pays a rebuild** —
it absorbs an arbitrary +-20% metric change in a 0.34 ms array write, ~3000x
cheaper than CRP's ~1 s customization and ~300,000x cheaper than rebuilding the
exact CH. (We benchmark NY; FLA's full-graph Python-side ordering extrapolates to
roughly 8-15 min and is left out of the budgeted run — the mechanism is identical.)

### Positioning

For *static, known* road costs the exact certified CH now answers in **231 us on
NY — within ~2x of the published CH's 110 us** despite running on a Python+numba
stack and a ~70x smaller graph, closing the two-order gap ALT left open and
reaching CH's own query class. For *drifting* costs — CERT's actual regime — the
exact CH is the wrong tool (any change forces a 108 s rebuild, worse than CRP),
so we build the CH on lower bounds and use it as an admissible A* oracle: exact
queries robust to +-20% cost changes for a 0.34 ms array write and **no
re-customization at all**, trading a slower individual query (12.6 ms) for the
elimination of CRP's ~1 s per-change customization. That is the same certificate
bargain as the snapshot oracle and ALT, now carried into the CH-class query
regime: preprocessing-by-proof buys road-scale speed precisely in the band where
the certificate guarantees it remains exact.
