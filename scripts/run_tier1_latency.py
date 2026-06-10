"""Tier-1 replanning-latency benchmark for CERT (spec §7 Tier-1, Theorem T3).

Purpose
-------
Empirical evidence for Theorem T3: incremental D* Lite repair cost scales with
the locally-changed region, not with graph size. This is the known-map dynamic-
metric tier — no planner loop, no conformal; pure search-engine benchmark.

Two scenarios per (size, radius):
  1. Static start: start and goal are fixed; 200 rounds of random locus + local
     Chebyshev-radius perturbations; compare incremental vs scratch latency.
  2. Moving start: walk start along current path 100 steps with radius-2
     perturbations between steps (uses DStarLite.set_start).

Results saved to results/tier1/table.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Allow running from repo root without install.
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

from certflow.graphcore import DStarLite, dijkstra  # noqa: E402


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def build_grid_graph(
    rows: int,
    cols: int,
    rng: np.random.Generator,
) -> dict[tuple[int, int], dict[tuple[int, int], float]]:
    """4-connected directed grid with random positive costs drawn from U(0.5, 3.0)."""
    graph: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    for r in range(rows):
        for c in range(cols):
            node = (r, c)
            graph.setdefault(node, {})
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    w = float(rng.uniform(0.5, 3.0))
                    graph[node][(nr, nc)] = w
    return graph


def chebyshev_edges_within(
    locus: tuple[int, int],
    radius: int,
    rows: int,
    cols: int,
    graph: dict,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """All edges (u, v) where u is within Chebyshev radius r of locus."""
    lr, lc = locus
    result = []
    for r in range(max(0, lr - radius), min(rows, lr + radius + 1)):
        for c in range(max(0, lc - radius), min(cols, lc + radius + 1)):
            node = (r, c)
            for nbr in graph.get(node, {}):
                result.append((node, nbr))
    return result


# ---------------------------------------------------------------------------
# Core benchmark function (importable)
# ---------------------------------------------------------------------------

def run_benchmark(
    rows: int,
    cols: int,
    radius: int,
    n_rounds: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Run one (size, radius) static-start benchmark and return a metrics dict.

    Returns
    -------
    dict with keys:
        size, radius, rounds,
        inc_latencies_ms, scratch_latencies_ms,
        inc_pops, scratch_pops (always 0 — Dijkstra has no pops counter),
        node_count,
        cost_mismatches  -- list of round indices where costs disagreed (should be empty)
    """
    rng = np.random.default_rng(seed)

    graph = build_grid_graph(rows, cols, rng)
    node_count = rows * cols

    start = (0, 0)
    goal = (rows - 1, cols - 1)

    # Sanity check: initial path must exist.
    init_path, init_cost = dijkstra(graph, start, goal)
    if init_path is None:
        raise RuntimeError(f"No initial path on {rows}x{cols} grid — unexpected.")

    dstar = DStarLite(graph, start, goal)

    inc_latencies_ms: list[float] = []
    scratch_latencies_ms: list[float] = []
    inc_pops_list: list[int] = []
    cost_mismatches: list[int] = []

    for round_idx in range(n_rounds):
        # Pick a random locus node.
        lr = int(rng.integers(0, rows))
        lc = int(rng.integers(0, cols))
        locus = (lr, lc)

        # Collect edges within Chebyshev radius.
        edges = chebyshev_edges_within(locus, radius, rows, cols, graph)
        if not edges:
            edges = list(graph.get(locus, {}).keys())
            edges = [(locus, v) for v in edges]
        if not edges:
            inc_latencies_ms.append(0.0)
            scratch_latencies_ms.append(0.0)
            inc_pops_list.append(0)
            continue

        # Perturb costs: multiply by U(0.5, 2.0), keep positive.
        new_costs: dict[tuple, float] = {}
        for u, v in edges:
            factor = float(rng.uniform(0.5, 2.0))
            new_costs[(u, v)] = max(1e-6, graph[u][v] * factor)

        # Apply to master graph (shared truth for both methods).
        for (u, v), w in new_costs.items():
            graph[u][v] = w

        # (a) Incremental: update_edges + shortest_path.
        t0 = time.perf_counter()
        dstar.update_edges(new_costs)
        inc_path, inc_cost = dstar.shortest_path()
        t1 = time.perf_counter()
        inc_latencies_ms.append((t1 - t0) * 1000.0)
        inc_pops_list.append(dstar.pops)

        # (b) Scratch: Dijkstra on the updated graph.
        t2 = time.perf_counter()
        scratch_path, scratch_cost = dijkstra(graph, start, goal)
        t3 = time.perf_counter()
        scratch_latencies_ms.append((t3 - t2) * 1000.0)

        # Correctness check: costs must agree exactly (both operate on same graph).
        if inc_cost != scratch_cost:
            # Allow tiny floating-point tolerance (< 1e-9 relative).
            if scratch_cost == 0.0 or abs(inc_cost - scratch_cost) / abs(scratch_cost) > 1e-9:
                cost_mismatches.append(round_idx)

    return {
        "size": f"{rows}x{cols}",
        "radius": radius,
        "rounds": n_rounds,
        "node_count": node_count,
        "inc_latencies_ms": inc_latencies_ms,
        "scratch_latencies_ms": scratch_latencies_ms,
        "inc_pops": inc_pops_list,
        "cost_mismatches": cost_mismatches,
    }


def run_moving_benchmark(
    rows: int,
    cols: int,
    n_steps: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Moving-start scenario: walk 100 steps with radius-2 perturbations."""
    radius = 2
    rng = np.random.default_rng(seed + 10000)

    graph = build_grid_graph(rows, cols, rng)
    node_count = rows * cols

    start = (0, 0)
    goal = (rows - 1, cols - 1)

    dstar = DStarLite(graph, start, goal)

    inc_latencies_ms: list[float] = []
    scratch_latencies_ms: list[float] = []
    inc_pops_list: list[int] = []
    cost_mismatches: list[int] = []

    current_start = start

    for step in range(n_steps):
        # Perturb edges around current_start.
        edges = chebyshev_edges_within(current_start, radius, rows, cols, graph)
        if not edges:
            edges = [(current_start, v) for v in graph.get(current_start, {})]

        new_costs: dict[tuple, float] = {}
        for u, v in edges:
            factor = float(rng.uniform(0.5, 2.0))
            new_costs[(u, v)] = max(1e-6, graph[u][v] * factor)

        for (u, v), w in new_costs.items():
            graph[u][v] = w

        # (a) Incremental.
        t0 = time.perf_counter()
        dstar.update_edges(new_costs)
        inc_path, inc_cost = dstar.shortest_path()
        t1 = time.perf_counter()
        inc_latencies_ms.append((t1 - t0) * 1000.0)
        inc_pops_list.append(dstar.pops)

        # (b) Scratch.
        t2 = time.perf_counter()
        scratch_path, scratch_cost = dijkstra(graph, current_start, goal)
        t3 = time.perf_counter()
        scratch_latencies_ms.append((t3 - t2) * 1000.0)

        # Correctness check.
        if inc_cost != scratch_cost:
            if scratch_cost == 0.0 or abs(inc_cost - scratch_cost) / abs(scratch_cost) > 1e-9:
                cost_mismatches.append(step)

        # Advance start: step to next node on current path (if path exists).
        if inc_path is not None and len(inc_path) > 1:
            next_node = inc_path[1]
            dstar.set_start(next_node)
            current_start = next_node
            # If we've reached the goal, stop early.
            if current_start == goal:
                break
        else:
            # Path gone (unreachable); stop moving.
            break

    return {
        "size": f"{rows}x{cols}",
        "radius": radius,
        "rounds": len(inc_latencies_ms),
        "node_count": node_count,
        "inc_latencies_ms": inc_latencies_ms,
        "scratch_latencies_ms": scratch_latencies_ms,
        "inc_pops": inc_pops_list,
        "cost_mismatches": cost_mismatches,
        "scenario": "moving",
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def compute_row(result: dict[str, Any], scenario: str = "static") -> dict[str, Any]:
    inc = result["inc_latencies_ms"]
    scr = result["scratch_latencies_ms"]
    pops = result["inc_pops"]

    inc_p50 = percentile(inc, 50)
    inc_p95 = percentile(inc, 95)
    scr_p50 = percentile(scr, 50)
    scr_p95 = percentile(scr, 95)
    speedup_p50 = scr_p50 / inc_p50 if inc_p50 > 0 else float("nan")
    mean_pops = float(np.mean(pops)) if pops else float("nan")
    node_count = result["node_count"]
    mismatches = len(result["cost_mismatches"])

    return {
        "scenario": scenario,
        "size": result["size"],
        "radius": result.get("radius", 2),
        "rounds": result["rounds"],
        "node_count": node_count,
        "inc_p50_ms": round(inc_p50, 4),
        "inc_p95_ms": round(inc_p95, 4),
        "scr_p50_ms": round(scr_p50, 4),
        "scr_p95_ms": round(scr_p95, 4),
        "speedup_p50": round(speedup_p50, 2),
        "mean_pops": round(mean_pops, 1),
        "cost_mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'scenario':<10} {'size':<8} {'r':>4} {'rounds':>7} "
        f"{'inc_p50':>9} {'inc_p95':>9} {'scr_p50':>9} {'scr_p95':>9} "
        f"{'speedup':>8} {'pops':>8} {'nodes':>7} {'mismatch':>9}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print(
            f"{row['scenario']:<10} {row['size']:<8} {row['radius']:>4} {row['rounds']:>7} "
            f"{row['inc_p50_ms']:>9.4f} {row['inc_p95_ms']:>9.4f} "
            f"{row['scr_p50_ms']:>9.4f} {row['scr_p95_ms']:>9.4f} "
            f"{row['speedup_p50']:>8.2f} {row['mean_pops']:>8.1f} "
            f"{row['node_count']:>7} {row['cost_mismatches']:>9}"
        )
    print(sep)
    print("(latencies in ms; speedup = scr_p50/inc_p50; mismatch = rounds where costs disagreed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier-1 replanning-latency benchmark (Theorem T3)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 20x20 only, 50 rounds per (size, radius) combination.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base random seed."
    )
    args = parser.parse_args()

    if args.quick:
        sizes = [(20, 20)]
        n_rounds = 50
        n_moving_steps = 30
    else:
        sizes = [(20, 20), (40, 40), (80, 80)]
        n_rounds = 200
        n_moving_steps = 100

    radii = [1, 2, 5, 10]

    table_rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []

    for rows, cols in sizes:
        size_label = f"{rows}x{cols}"
        print(f"\n=== Size {size_label} ===")

        for r in radii:
            print(f"  radius={r}  ({n_rounds} rounds) ...", end=" ", flush=True)
            result = run_benchmark(rows, cols, r, n_rounds, seed=args.seed)
            row = compute_row(result, scenario="static")
            table_rows.append(row)
            all_results.append({**result, "scenario": "static"})

            mismatches = result["cost_mismatches"]
            if mismatches:
                print(f"COST MISMATCH on rounds {mismatches[:5]}{'...' if len(mismatches) > 5 else ''}")
            else:
                print(f"ok  speedup={row['speedup_p50']:.2f}x  pops={row['mean_pops']:.1f}")

        # Moving-start scenario.
        print(f"  moving (radius=2, {n_moving_steps} steps) ...", end=" ", flush=True)
        mv_result = run_moving_benchmark(rows, cols, n_moving_steps, seed=args.seed)
        mv_row = compute_row(mv_result, scenario="moving")
        table_rows.append(mv_row)
        all_results.append(mv_result)
        print(f"ok  speedup={mv_row['speedup_p50']:.2f}x  pops={mv_row['mean_pops']:.1f}  "
              f"steps={mv_result['rounds']}")

    # Print summary table.
    print("\n=== Tier-1 Benchmark Summary ===")
    print_table(table_rows)

    # Save results.
    out_dir = _repo / "results" / "tier1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table.json"
    payload = {
        "meta": {
            "quick": args.quick,
            "seed": args.seed,
            "n_rounds": n_rounds,
            "n_moving_steps": n_moving_steps,
            "sizes": [f"{r}x{c}" for r, c in sizes],
            "radii": radii,
        },
        "table": table_rows,
        "raw": [
            {k: v for k, v in r.items() if k not in ("inc_latencies_ms", "scratch_latencies_ms", "inc_pops")}
            for r in all_results
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Report any anomalies loudly.
    total_mismatches = sum(r["cost_mismatches"] for r in table_rows)
    if total_mismatches > 0:
        print("\n*** ANOMALY: incremental vs scratch cost mismatches detected! ***")
        print("*** This is a potential graphcore bug — see 'cost_mismatches' in table.json ***")
        for row in table_rows:
            if row["cost_mismatches"] > 0:
                print(f"  {row['scenario']} {row['size']} r={row['radius']}: "
                      f"{row['cost_mismatches']} mismatch(es)")
    else:
        print("\nAll cost checks passed: incremental == scratch every round.")


if __name__ == "__main__":
    main()
