"""Road-network CH benchmark for CERT: certified Contraction Hierarchies.

Closes (as far as a Python+numba stack can) the ~2-order gap to published CH
(110 us / query) at road scale, and measures the certified angle honestly:

  * CH build time on a DIMACS graph (or a node-bounded subgraph when full-graph
    Python-side ordering exceeds the time budget);
  * CH query p50/p95 -- cost-only and with shortcut-unpacking -- over many
    random pairs, vs our FastDijkstra and the published CH 110 us;
  * REBUILD-ON-GATE: a cost perturbation INVALIDATES a CH built on exact costs,
    so the cost of absorbing a change is a full rebuild. We report it and
    compare to CRP's ~1 s customization;
  * BOUNDED-CHANGE variant (`CHPotentialOracle`): a CH built on 0.8x lower-bound
    costs as an admissible heuristic for a forward A* on the TRUE costs --
    exact queries robust to +-20% changes with ZERO rebuild. We measure its
    query p50/p95 (between raw CH and ALT) and verify exactness under a fresh
    +-20% perturbation.

Usage:
    cert_env/bin/python scripts/run_ch.py                 # NY 50k subgraph
    cert_env/bin/python scripts/run_ch.py --sub 0         # full NY (slow build)
    cert_env/bin/python scripts/run_ch.py --graph FLA --sub 50000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from certflow.fastgraph import FastDijkstra, _HAVE_NUMBA
from certflow.roadnet import _csr_from_arcs, load_dimacs
from certflow.ch import ContractionHierarchy, CHPotentialOracle

ROOT = Path(__file__).resolve().parents[1]
DIMACS = ROOT / "data" / "dimacs"
GRAPHS = {
    "NY": DIMACS / "USA-road-d.NY.gr",
    "FLA": DIMACS / "USA-road-d.FLA.gr",
}


def _pcts(samples_s):
    a = np.array(samples_s, dtype=np.float64)
    return float(np.percentile(a, 50) * 1e3), float(np.percentile(a, 95) * 1e3)


def _pairs(n, k, seed):
    rng = np.random.default_rng(seed)
    return [(int(rng.integers(0, n)), int(rng.integers(0, n))) for _ in range(k)]


def _induced_subgraph(g, n_sub):
    """Largest-degree-first induced subgraph on the first `n_sub` nodes."""
    indptr = np.asarray(g.indptr)
    indices = np.asarray(g.indices)
    cost = np.asarray(g.cost)
    tails, heads, weights = [], [], []
    for u in range(n_sub):
        for k in range(int(indptr[u]), int(indptr[u + 1])):
            v = int(indices[k])
            if v < n_sub:
                tails.append(u)
                heads.append(v)
                weights.append(float(cost[k]))
    return _csr_from_arcs(
        n_sub,
        np.array(tails, dtype=np.int64),
        np.array(heads, dtype=np.int64),
        np.array(weights, dtype=np.float64),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="NY", choices=list(GRAPHS))
    ap.add_argument("--sub", type=int, default=50000,
                    help="subgraph node cap (0 = full graph)")
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--exact", type=int, default=200)
    args = ap.parse_args()

    print(f"numba: {_HAVE_NUMBA}")
    path = GRAPHS[args.graph]
    if not path.exists():
        print(f"SKIP: {path} not found")
        return

    print(f"\n=== {args.graph} ({path.name}) ===", flush=True)
    t0 = time.perf_counter()
    g_full = load_dimacs(path)
    print(f"loaded full: n={g_full.n:,} m={g_full.indices.size:,} "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)

    if args.sub and args.sub < g_full.n:
        g = _induced_subgraph(g_full, args.sub)
        print(f"using induced subgraph: n={g.n:,} m={g.indices.size:,} "
              f"(full-graph Python ordering exceeds the time budget; see doc)",
              flush=True)
    else:
        g = g_full
        print("using FULL graph", flush=True)

    orig = g.cost.copy()

    # --- exact CH build ---
    ch = ContractionHierarchy(g)
    t0 = time.perf_counter()
    ch.build()
    ch_build_s = time.perf_counter() - t0
    ch.warmup()
    print(f"CH build (exact costs): {ch_build_s:.1f}s, "
          f"{ch.n_shortcuts:,} shortcuts, "
          f"up-edges {ch.up_indices.size:,} down-edges {ch.dn_indices.size:,}",
          flush=True)

    # --- bounded-change CH-potentials oracle build (on 0.8x lower bounds) ---
    orc = CHPotentialOracle(g, lower_bound_factor=0.8)
    t0 = time.perf_counter()
    orc.build()
    orc_build_s = time.perf_counter() - t0
    orc.warmup()
    print(f"CH-potentials oracle build (0.8x lower bounds): {orc_build_s:.1f}s",
          flush=True)

    pairs = _pairs(g.n, args.pairs, seed=42)
    # warm FastDijkstra caches
    FastDijkstra(g, pairs[0][0]).shortest_path(pairs[0][1])

    # --- query benchmarks ---
    dij_t, ch_cost_t, ch_path_t, orc_t = [], [], [], []
    dij_costs, ch_costs, orc_costs = [], [], []

    for (s, t) in pairs:
        t0 = time.perf_counter()
        _, dc = FastDijkstra(g, s).shortest_path(t)
        dij_t.append(time.perf_counter() - t0)
        dij_costs.append(dc)

    for (s, t) in pairs:
        t0 = time.perf_counter()
        c = ch.query(s, t)
        ch_cost_t.append(time.perf_counter() - t0)
        ch_costs.append(c)

    for (s, t) in pairs:
        t0 = time.perf_counter()
        ch.path(s, t)
        ch_path_t.append(time.perf_counter() - t0)

    for (s, t) in pairs:
        t0 = time.perf_counter()
        oc = orc.query(s, t)
        orc_t.append(time.perf_counter() - t0)
        orc_costs.append(oc)

    # exactness on the benchmark pairs
    ch_mism = sum(
        1 for d, c in zip(dij_costs, ch_costs)
        if not (abs(d - c) < 1e-6 or (np.isinf(d) and np.isinf(c)))
    )
    orc_mism = sum(
        1 for d, c in zip(dij_costs, orc_costs)
        if not (abs(d - c) < 1e-6 or (np.isinf(d) and np.isinf(c)))
    )

    d50, d95 = _pcts(dij_t)
    c50, c95 = _pcts(ch_cost_t)
    p50, p95 = _pcts(ch_path_t)
    o50, o95 = _pcts(orc_t)

    print(f"\nFastDijkstra full query: p50={d50:.3f}ms p95={d95:.3f}ms")
    print(f"CH query (cost-only):    p50={c50:.4f}ms p95={c95:.4f}ms")
    print(f"CH query (path unpack):  p50={p50:.4f}ms p95={p95:.4f}ms")
    print(f"CH-potentials A* query:  p50={o50:.4f}ms p95={o95:.4f}ms")
    print(f"CH exactness ({args.pairs} pairs): {ch_mism} mismatches")
    print(f"CH-potentials exactness ({args.pairs} pairs): {orc_mism} mismatches")

    # --- REBUILD-ON-GATE: a cost perturbation invalidates the exact CH ---
    # The exact CH is valid only for the costs it was built on. A change forces
    # a full rebuild (the CH analog of CRP's metric customization).
    print(f"\n--- rebuild-on-gate (exact CH invalidated by any cost change) ---")
    print(f"CH rebuild cost = build cost = {ch_build_s:.1f}s "
          f"(vs CRP customization ~1s parallel / ~11s sequential)")

    # --- bounded-change: +-20% perturbation, NO rebuild of the oracle ---
    rng = np.random.default_rng(999)
    factors = rng.uniform(0.8, 1.2, size=g.cost.size)
    g.cost[:] = orig * factors
    # cost of "absorbing" the change for the oracle = the CSR array write
    abs_t = []
    deltas = (orig * factors) - orig
    for _ in range(20):
        t0 = time.perf_counter()
        g.cost[:] = orig + deltas
        abs_t.append(time.perf_counter() - t0)
    abs50, _ = _pcts(abs_t)

    exact_pairs = _pairs(g.n, args.exact, seed=7)
    omism = 0
    for (s, t) in exact_pairs:
        _, dc = FastDijkstra(g, s).shortest_path(t)
        oc = orc.query(s, t)
        if not (abs(dc - oc) < 1e-6 or (np.isinf(dc) and np.isinf(oc))):
            omism += 1
    print(f"\n--- bounded-change CH-potentials (+-20%, NO rebuild) ---")
    print(f"absorb cost change (CSR write): p50={abs50:.4f}ms")
    print(f"exactness under +-20% perturbation: {omism}/{args.exact} mismatches")
    # re-measure oracle query under perturbed costs
    o2 = []
    for (s, t) in pairs:
        t0 = time.perf_counter()
        orc.query(s, t)
        o2.append(time.perf_counter() - t0)
    o2_50, o2_95 = _pcts(o2)
    print(f"CH-potentials query under +-20% costs: p50={o2_50:.4f}ms p95={o2_95:.4f}ms")
    g.cost[:] = orig

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"graph={args.graph} n={g.n:,} m={g.indices.size:,}")
    print(f"CH build {ch_build_s:.1f}s | CH cost q p50={c50:.4f}ms p95={c95:.4f}ms "
          f"| CH path q p50={p50:.4f}ms")
    print(f"CH-pot build {orc_build_s:.1f}s | CH-pot q p50={o50:.4f}ms p95={o95:.4f}ms "
          f"| +-20% robust q p50={o2_50:.4f}ms ({omism}/{args.exact} mism)")
    print(f"FastDijkstra p50={d50:.3f}ms | published CH 0.110ms | published HL 0.00056ms")


if __name__ == "__main__":
    main()
