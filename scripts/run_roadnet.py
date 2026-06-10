"""Road-network-scale benchmark for CERT: FastDijkstra vs ALT on DIMACS graphs.

Reports, per graph (NY 264k nodes, FLA 1.07M nodes):
  (a) FastDijkstra full-query p50/p95 over 100 random s-t pairs,
  (b) ALT query p50/p95 over the same pairs,
  (c) landmark preprocessing time (16 landmarks, both directions),
  (d) a 'certified-customization' microbenchmark: time to apply a 1% random
      edge-cost perturbation to the CSR cost array, plus a note on ALT
      admissibility after the perturbation. Landmarks are built on 0.8x cost
      lower bounds, so a +-20%-bounded perturbation keeps the heuristic
      admissible WITHOUT recomputing any landmark distance. We verify exactness
      under a +-20% perturbation against FastDijkstra on 200 fresh pairs.

Usage:
    cert_env/bin/python scripts/run_roadnet.py            # NY + FLA
    cert_env/bin/python scripts/run_roadnet.py --graphs NY
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from certflow.fastgraph import FastDijkstra, _HAVE_NUMBA
from certflow.roadnet import ALT, load_dimacs

ROOT = Path(__file__).resolve().parents[1]
DIMACS = ROOT / "data" / "dimacs"

GRAPHS = {
    "NY": DIMACS / "USA-road-d.NY.gr",
    "FLA": DIMACS / "USA-road-d.FLA.gr",
}


def _percentiles(samples_s: list[float]) -> tuple[float, float]:
    """p50/p95 of a sample list (seconds), returned in milliseconds."""
    a = np.array(samples_s, dtype=np.float64)
    return float(np.percentile(a, 50) * 1e3), float(np.percentile(a, 95) * 1e3)


def _random_pairs(n: int, k: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    return [
        (int(rng.integers(0, n)), int(rng.integers(0, n)))
        for _ in range(k)
    ]


def bench_graph(
    name: str,
    path: Path,
    n_pairs: int = 100,
    n_exact: int = 200,
    n_landmarks: int = 16,
    lower_bound_factor: float = 0.8,
    seed: int = 0,
) -> dict:
    print(f"\n=== {name}  ({path.name}) ===", flush=True)
    t0 = time.perf_counter()
    g = load_dimacs(path)
    load_s = time.perf_counter() - t0
    n_edges = int(g.indices.size)
    print(f"loaded: n={g.n:,} m={n_edges:,} in {load_s:.2f}s", flush=True)

    # --- (c) landmark preprocessing ---
    t0 = time.perf_counter()
    alt = ALT(g, n_landmarks=n_landmarks, lower_bound_factor=lower_bound_factor, seed=seed)
    prep_s = time.perf_counter() - t0
    alt.warmup()  # compile the query kernel out of the timed region
    print(f"ALT preprocessing ({n_landmarks} landmarks, both dirs): {prep_s:.2f}s", flush=True)

    pairs = _random_pairs(g.n, n_pairs, seed=seed + 100)

    # warm the Dijkstra path too (numba-free Python heap, but warm caches)
    FastDijkstra(g, pairs[0][0]).shortest_path(pairs[0][1])

    # --- (a) FastDijkstra full query ---
    dij_t, alt_t = [], []
    dij_costs, alt_costs = [], []
    for (s, t) in pairs:
        t0 = time.perf_counter()
        _, dc = FastDijkstra(g, s).shortest_path(t)
        dij_t.append(time.perf_counter() - t0)
        dij_costs.append(dc)
    # --- (b) ALT query ---
    for (s, t) in pairs:
        t0 = time.perf_counter()
        ac = alt.query(s, t)
        alt_t.append(time.perf_counter() - t0)
        alt_costs.append(ac)

    # exactness on the benchmark pairs
    mism = sum(
        1 for dc, ac in zip(dij_costs, alt_costs)
        if not (abs(dc - ac) < 1e-6 or (np.isinf(dc) and np.isinf(ac)))
    )
    dij50, dij95 = _percentiles(dij_t)
    alt50, alt95 = _percentiles(alt_t)
    print(f"FastDijkstra full query: p50={dij50:.3f}ms p95={dij95:.3f}ms", flush=True)
    print(f"ALT query:               p50={alt50:.3f}ms p95={alt95:.3f}ms", flush=True)
    print(f"ALT exactness on {n_pairs} bench pairs: {mism} mismatches", flush=True)

    # --- (d) certified-customization microbenchmark ---
    # Apply a 1% random edge-cost perturbation to the CSR cost array. Because
    # landmarks were built on 0.8x lower bounds, a perturbation that keeps each
    # edge >= 0.8x the original (we use +-1% here, well inside +-20%) leaves the
    # heuristic admissible with NO recomputation -> the customization cost is
    # exactly the array write.
    rng = np.random.default_rng(seed + 7)
    m = g.cost.size
    k = max(1, m // 100)  # 1% of edges
    sel = rng.choice(m, size=k, replace=False)
    deltas = rng.uniform(-0.01, 0.01, size=k) * g.cost[sel]
    # time the in-place perturbation (this is the whole "customization")
    custom_t = []
    for _ in range(20):
        t0 = time.perf_counter()
        g.cost[sel] += deltas
        custom_t.append(time.perf_counter() - t0)
        g.cost[sel] -= deltas  # restore for repeatability
    cust50, cust95 = _percentiles(custom_t)
    print(
        f"certified-customization (1% perturb, {k:,} edges): "
        f"p50={cust50:.4f}ms p95={cust95:.4f}ms", flush=True
    )

    # --- verify exactness under a bounded (+-20%) perturbation, NO re-customize ---
    # Worst-case admissibility test: decrease costs by up to 20% (down to the
    # 0.8x lower bound the landmarks were built on). The heuristic must remain
    # admissible+consistent, so ALT must still match Dijkstra exactly.
    orig = g.cost.copy()
    factors = rng.uniform(0.8, 1.2, size=m)
    g.cost[:] = orig * factors
    exact_pairs = _random_pairs(g.n, n_exact, seed=seed + 999)
    mism2 = 0
    for (s, t) in exact_pairs:
        _, dc = FastDijkstra(g, s).shortest_path(t)
        ac = alt.query(s, t)
        if not (abs(dc - ac) < 1e-6 or (np.isinf(dc) and np.isinf(ac))):
            mism2 += 1
    g.cost[:] = orig
    print(
        f"exactness under +-20% perturbation (NO re-customization), "
        f"{n_exact} pairs: {mism2} mismatches", flush=True
    )

    return {
        "name": name,
        "n": g.n,
        "m": n_edges,
        "load_s": load_s,
        "prep_s": prep_s,
        "n_landmarks": n_landmarks,
        "dij_p50_ms": dij50,
        "dij_p95_ms": dij95,
        "alt_p50_ms": alt50,
        "alt_p95_ms": alt95,
        "alt_mismatch": mism,
        "custom_p50_ms": cust50,
        "custom_p95_ms": cust95,
        "perturb_mismatch": mism2,
        "n_exact": n_exact,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", nargs="+", default=["NY", "FLA"], choices=list(GRAPHS))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--exact", type=int, default=200)
    ap.add_argument("--landmarks", type=int, default=16)
    args = ap.parse_args()

    print(f"numba: {_HAVE_NUMBA}")
    results = []
    for name in args.graphs:
        path = GRAPHS[name]
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        results.append(
            bench_graph(
                name, path,
                n_pairs=args.pairs, n_exact=args.exact, n_landmarks=args.landmarks,
            )
        )

    # summary table
    print("\n" + "=" * 78)
    print("ROAD-NETWORK BENCHMARK SUMMARY")
    print("=" * 78)
    hdr = (
        f"{'graph':<6}{'nodes':>10}{'edges':>11}"
        f"{'dij p50/p95 (ms)':>20}{'ALT p50/p95 (ms)':>20}"
    )
    print(hdr)
    for r in results:
        print(
            f"{r['name']:<6}{r['n']:>10,}{r['m']:>11,}"
            f"{r['dij_p50_ms']:>10.3f}/{r['dij_p95_ms']:<8.3f}"
            f"{r['alt_p50_ms']:>10.3f}/{r['alt_p95_ms']:<8.3f}"
        )
    print()
    for r in results:
        print(
            f"{r['name']}: landmark-prep {r['prep_s']:.1f}s | "
            f"customization (1% perturb) p50={r['custom_p50_ms']:.4f}ms | "
            f"ALT exact (bench {r['alt_mismatch']} mism, "
            f"+-20%-perturb {r['perturb_mismatch']}/{r['n_exact']} mism)"
        )


if __name__ == "__main__":
    main()
