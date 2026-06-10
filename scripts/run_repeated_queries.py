"""Repeated-query planning comparison: E-Graphs vs scratch vs D* Lite reuse.

Setting where E-Graphs applies (Phillips et al. RSS 2012): a *sequence* of
planning queries on a *changing* map, with start/goal pairs drawn from a small
reusable pool so accumulated experience is relevant. We compare four planners
on the same drifting worlds:

  (a) scratch Dijkstra per query              -- optimal, no reuse
  (b) scratch weighted A* (w=1.2) per query   -- bounded-suboptimal, no reuse
  (c) EGraphPlanner (accumulating experience) -- bounded-suboptimal, path reuse
  (d) persistent D* Lite, one instance per (start,goal) pool entry
          -- our engine's incremental-reuse answer

Plus an external speed anchor:
  (e) networkx.shortest_path_length (dijkstra, weight) on the same updated graph
      -- third-party reference so "fast" is not a self-comparison.

Protocol: 20x20 and 40x40 grids, bounded drift rho=0.02, 50 queries per run,
start/goal drawn from a pool of POOL_SIZE pairs, world advanced 10 rounds (time
units) between queries so costs drift. Metrics per query: wall-clock p50/p95
(perf_counter, ms), solution-cost ratio vs the optimal (Dijkstra) on the same
graph snapshot, node expansions where countable (weighted A* and E-Graphs).
3 seeds, aggregated.

Timing convention follows docs/results/tier1-latency.md: time.perf_counter()
wrapped tightly around the planning call only, latencies in milliseconds, p50/p95
reported.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

import networkx as nx  # noqa: E402

from certflow.drift import grid_world  # noqa: E402
from certflow.egraph import EGraphPlanner, manhattan_heuristic  # noqa: E402
from certflow.graphcore import DStarLite, dijkstra  # noqa: E402

POOL_SIZE = 6
N_QUERIES = 50
DRIFT_STEP = 10.0  # time units advanced between queries
RHO = 0.02


def snapshot_graph(world: Any, t: float) -> dict[Any, dict[Any, float]]:
    """Materialize the world's true costs at time t as an adjacency dict."""
    graph: dict[Any, dict[Any, float]] = {}
    for (u, v) in world.edges():
        graph.setdefault(u, {})[v] = world.true_cost((u, v), t)
        graph.setdefault(v, {})
    return graph


def to_networkx(graph: dict[Any, dict[Any, float]]) -> nx.DiGraph:
    g = nx.DiGraph()
    for u, nbrs in graph.items():
        g.add_node(u)
        for v, w in nbrs.items():
            g.add_edge(u, v, weight=w)
    return g


def weighted_astar(
    graph: dict[Any, dict[Any, float]],
    start: Any,
    goal: Any,
    h,
    w: float = 1.2,
) -> tuple[list[Any] | None, float, int]:
    """Plain weighted A* (no experience). Returns (path, cost, expansions)."""
    import heapq
    import itertools

    if start == goal:
        return [start], 0.0, 0
    g_score: dict[Any, float] = {start: 0.0}
    prev: dict[Any, Any] = {}
    ctr = itertools.count(1)
    pq = [(w * h(start, goal), next(ctr), start)]
    closed: set[Any] = set()
    expansions = 0
    while pq:
        _, _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expansions += 1
        if u == goal:
            break
        for v, wt in graph.get(u, {}).items():
            if v in closed:
                continue
            nd = g_score[u] + wt
            if nd < g_score.get(v, float("inf")):
                g_score[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd + w * h(v, goal), next(ctr), v))
    if goal not in g_score:
        return None, float("inf"), expansions
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path, g_score[goal], expansions


def path_cost(graph: dict[Any, dict[Any, float]], path: list[Any]) -> float:
    return sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))


def make_pool(rng: np.random.Generator, rows: int, cols: int) -> list[tuple]:
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    pool = []
    while len(pool) < POOL_SIZE:
        s = nodes[int(rng.integers(0, len(nodes)))]
        gl = nodes[int(rng.integers(0, len(nodes)))]
        if s != gl:
            pool.append((s, gl))
    return pool


def run_one(
    rows: int, cols: int, seed: int, sigma: float = 0.5, regime: str = "spread"
) -> dict[str, Any]:
    """One run: returns per-planner latency lists, cost ratios, expansions.

    ``sigma`` controls the log-normal edge-cost spread. ``sigma=0.5`` (default,
    "spread") makes Manhattan a near-uninformative heuristic -> E-graphs has no
    valley to exploit. Small ``sigma`` ("uniform") makes Manhattan informative
    -> the E-graphs reuse mechanism activates (the regime where it is designed
    to win). We report both to avoid strawmanning the baseline.
    """
    world = grid_world(
        rows, cols, seed=seed, kind="bounded", rho=RHO, sigma=sigma
    )

    rng = np.random.default_rng(seed + 777)
    pool = make_pool(rng, rows, cols)
    query_seq = [pool[int(rng.integers(0, POOL_SIZE))] for _ in range(N_QUERIES)]

    # Admissible heuristic: Manhattan times a *true* global lower bound on every
    # edge cost over the query horizon (edges drift, so we minimize over all the
    # snapshot times actually used). This keeps h consistent throughout the run;
    # the same fair h is shared by both bounded-suboptimal planners.
    horizon_ts = [q * DRIFT_STEP for q in range(N_QUERIES)]
    min_edge = float("inf")
    for e in world.edges():
        for tt in horizon_ts:
            c = world.true_cost(e, tt)
            if c < min_edge:
                min_edge = c
    min_edge = max(1e-6, min_edge)
    h = manhattan_heuristic(min_edge)

    g0 = snapshot_graph(world, 0.0)
    egraph = EGraphPlanner(g0, heuristic=h, w=1.2)
    # One persistent D* Lite per distinct pool entry.
    dstar: dict[tuple, DStarLite] = {
        (s, gl): DStarLite(g0, s, gl) for (s, gl) in pool
    }
    dstar_t: dict[tuple, float] = {k: 0.0 for k in dstar}

    lat = {k: [] for k in ("dijkstra", "wastar", "egraph", "dstar", "networkx")}
    cost_ratio = {k: [] for k in ("wastar", "egraph", "dstar")}
    exp = {k: [] for k in ("wastar", "egraph")}
    unreachable = 0

    t = 0.0
    for q, (s, gl) in enumerate(query_seq):
        t = q * DRIFT_STEP
        graph = snapshot_graph(world, t)

        # Optimal reference (also (a) scratch Dijkstra).
        t0 = time.perf_counter()
        opt_path, opt_cost = dijkstra(graph, s, gl)
        lat["dijkstra"].append((time.perf_counter() - t0) * 1000.0)
        if opt_path is None:
            unreachable += 1
            continue

        # (b) scratch weighted A*.
        t0 = time.perf_counter()
        wa_path, wa_cost, wa_exp = weighted_astar(graph, s, gl, h, w=1.2)
        lat["wastar"].append((time.perf_counter() - t0) * 1000.0)
        if wa_path is not None and opt_cost > 0:
            cost_ratio["wastar"].append(wa_cost / opt_cost)
        exp["wastar"].append(wa_exp)

        # (c) E-Graph planner (accumulating experience).
        t0 = time.perf_counter()
        eg_path, eg_cost = egraph.plan(s, gl, graph=graph)
        lat["egraph"].append((time.perf_counter() - t0) * 1000.0)
        if eg_path is not None and opt_cost > 0:
            cost_ratio["egraph"].append(eg_cost / opt_cost)
        exp["egraph"].append(egraph.expansions)

        # (d) persistent D* Lite for this (s, gl).
        ds = dstar[(s, gl)]
        # Apply drift since this instance was last updated (only changed edges).
        prev_t = dstar_t[(s, gl)]
        if prev_t != t:
            changes = {}
            for (u, v) in world.edges():
                nc = graph[u][v]
                changes[(u, v)] = nc
            t0 = time.perf_counter()
            ds.update_edges(changes)
            ds_path, ds_cost = ds.shortest_path()
            lat["dstar"].append((time.perf_counter() - t0) * 1000.0)
            dstar_t[(s, gl)] = t
        else:
            t0 = time.perf_counter()
            ds_path, ds_cost = ds.shortest_path()
            lat["dstar"].append((time.perf_counter() - t0) * 1000.0)
        if ds_path is not None and opt_cost > 0:
            cost_ratio["dstar"].append(ds_cost / opt_cost)

        # (e) networkx external anchor.
        gnx = to_networkx(graph)
        t0 = time.perf_counter()
        try:
            nx.shortest_path_length(gnx, s, gl, weight="weight")
        except nx.NetworkXNoPath:
            pass
        lat["networkx"].append((time.perf_counter() - t0) * 1000.0)

    return {
        "size": f"{rows}x{cols}",
        "seed": seed,
        "regime": regime,
        "lat": lat,
        "cost_ratio": cost_ratio,
        "exp": exp,
        "unreachable": unreachable,
    }


def p(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def aggregate(runs: list[dict[str, Any]], size: str, regime: str) -> dict[str, Any]:
    sub = [r for r in runs if r["size"] == size and r["regime"] == regime]
    out: dict[str, Any] = {"size": size, "regime": regime}
    for planner in ("dijkstra", "wastar", "egraph", "dstar", "networkx"):
        allv = [x for r in sub for x in r["lat"][planner]]
        out[planner] = {
            "p50": p(allv, 50),
            "p95": p(allv, 95),
        }
    for planner in ("wastar", "egraph", "dstar"):
        ratios = [x for r in sub for x in r["cost_ratio"][planner]]
        out[planner]["cost_ratio_mean"] = (
            statistics.fmean(ratios) if ratios else float("nan")
        )
        out[planner]["cost_ratio_max"] = max(ratios) if ratios else float("nan")
    for planner in ("wastar", "egraph"):
        e = [x for r in sub for x in r["exp"][planner]]
        out[planner]["mean_exp"] = statistics.fmean(e) if e else float("nan")
    return out


def render_table(aggs: list[dict[str, Any]]) -> str:
    lines = []
    lines.append(
        "| regime | size | planner | p50 (ms) | p95 (ms) | cost ratio mean | cost ratio max | mean expansions |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    names = {
        "dijkstra": "Dijkstra (scratch, optimal)",
        "wastar": "weighted A* w=1.2 (scratch)",
        "egraph": "EGraphPlanner (experience)",
        "dstar": "D* Lite (persistent)",
        "networkx": "networkx dijkstra (extern)",
    }
    for a in aggs:
        for planner in ("dijkstra", "wastar", "egraph", "dstar", "networkx"):
            d = a[planner]
            cr_mean = d.get("cost_ratio_mean")
            cr_max = d.get("cost_ratio_max")
            me = d.get("mean_exp")
            cr_mean_s = f"{cr_mean:.4f}" if cr_mean is not None and cr_mean == cr_mean else "1.0000" if planner == "dijkstra" else "-"
            cr_max_s = f"{cr_max:.4f}" if cr_max is not None and cr_max == cr_max else "1.0000" if planner == "dijkstra" else "-"
            me_s = f"{me:.0f}" if me is not None and me == me else "-"
            lines.append(
                f"| {a['regime']} | {a['size']} | {names[planner]} | {d['p50']:.4f} | {d['p95']:.4f} "
                f"| {cr_mean_s} | {cr_max_s} | {me_s} |"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sizes", default="20,40")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    # Two regimes: "spread" (sigma=0.5, Manhattan uninformative -> no E-graph
    # valley) and "uniform" (sigma=0.05, Manhattan informative -> E-graph reuse
    # activates). Reporting both is the fair comparison.
    regimes = [("spread", 0.5), ("uniform", 0.05)]
    runs = []
    for regime, sigma in regimes:
        for n in sizes:
            for seed in range(args.seeds):
                print(f"  running {regime} {n}x{n} seed={seed} ...", flush=True)
                runs.append(run_one(n, n, seed, sigma=sigma, regime=regime))

    aggs = []
    for regime, _ in regimes:
        for n in sizes:
            aggs.append(aggregate(runs, f"{n}x{n}", regime))
    print()
    print(render_table(aggs))
    print()
    tot_unreach = sum(r["unreachable"] for r in runs)
    print(f"(total unreachable queries skipped: {tot_unreach})")


if __name__ == "__main__":
    main()
