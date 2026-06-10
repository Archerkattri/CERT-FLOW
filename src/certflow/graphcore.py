"""Incremental shortest-path core for CERT (`graphcore` component, spec §6).

Implements D* Lite (Koenig & Likhachev, AAAI 2002 — the "final" optimized
version with the `km` key modifier) plus a plain Dijkstra reference used by the
oracle and the differential tests.

The graph is a goal-rooted adjacency map `graph[u][v] = c(u, v)`; D* Lite
searches *backwards* from the goal, so it consumes predecessors of a node by
treating the same adjacency as `pred(v) = {u : v in graph[u]}` with the edge
cost `c(u, v)`. Costs are strictly positive floats and may drift over time via
`update_edges`; only edges present in the *initial* graph may be updated (no
edge insertion/deletion of topology).

Heuristic contract: the default heuristic is the zero heuristic, which is
admissible and consistent for any graph and any positive costs. A supplied
heuristic `h(a, b)` must be consistent with respect to a *lower bound* on every
edge cost that can ever occur (i.e. `h(u, goal) <= c_lb(u, v) + h(v, goal)` for
the smallest cost edge `(u, v)` may ever take). Because edge costs change over
time, a heuristic tuned to current costs can become inadmissible after an
update and silently break optimality — hence the default is zero.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Callable

from certflow.types import Edge, Node

INF = float("inf")


def dijkstra(
    graph: dict[Node, dict[Node, float]], source: Node, target: Node
) -> tuple[list[Node] | None, float]:
    """Plain forward Dijkstra. Independent reference implementation.

    Returns `(path, cost)` where `path` is the node list from `source` to
    `target` inclusive, or `(None, inf)` if `target` is unreachable.
    """
    if source == target:
        return [source], 0.0
    dist: dict[Node, float] = {source: 0.0}
    prev: dict[Node, Node] = {}
    pq: list[tuple[float, int, Node]] = [(0.0, 0, source)]
    counter = itertools.count(1)
    visited: set[Node] = set()
    while pq:
        d, _, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, next(counter), v))
    if target not in dist:
        return None, INF
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[target]


class DStarLite:
    """Incremental shortest path via D* Lite (goal-rooted g/rhs values).

    One instance maintains the shortest path from a (movable) `start` to a fixed
    `goal`. `update_edges` applies a batch of edge-cost changes and `set_start`
    moves the agent using the `km` heuristic offset so the priority queue need
    not be rebuilt. `shortest_path` runs `ComputeShortestPath` to repair only
    the locally affected region and then extracts the path.

    The `pops` attribute counts priority-queue pops performed by the most recent
    `ComputeShortestPath` call (vertex expansions, for locality benchmarks).
    """

    def __init__(
        self,
        graph: dict[Node, dict[Node, float]],
        start: Node,
        goal: Node,
        heuristic: Callable[[Node, Node], float] | None = None,
    ) -> None:
        # Forward adjacency (successors) as supplied.
        self._succ: dict[Node, dict[Node, float]] = {
            u: dict(nbrs) for u, nbrs in graph.items()
        }
        # Predecessor map for the backward search; ensure every node appears.
        self._pred: dict[Node, dict[Node, float]] = {}
        for u, nbrs in self._succ.items():
            self._pred.setdefault(u, {})
            for v, w in nbrs.items():
                self._succ.setdefault(v, {})
                self._pred.setdefault(v, {})[u] = w
        self._succ.setdefault(start, {})
        self._succ.setdefault(goal, {})
        self._pred.setdefault(start, {})
        self._pred.setdefault(goal, {})

        self._start: Node = start
        self._goal: Node = goal
        self._h: Callable[[Node, Node], float] = (
            heuristic if heuristic is not None else (lambda a, b: 0.0)
        )

        self.pops: int = 0
        self._init_search()

    # -- internal search state -------------------------------------------------

    def _init_search(self) -> None:
        self._g: dict[Node, float] = {n: INF for n in self._succ}
        self._rhs: dict[Node, float] = {n: INF for n in self._succ}
        self._km: float = 0.0
        self._rhs[self._goal] = 0.0
        self._counter = itertools.count(1)
        # Lazy-deletion priority queue; entries are (key, tie, node).
        self._pq: list[tuple[tuple[float, float], int, Node]] = []
        # Best (live) key per node currently in the queue.
        self._in_queue: dict[Node, tuple[float, float]] = {}
        self._push(self._goal)

    def _key(self, n: Node) -> tuple[float, float]:
        m = min(self._g[n], self._rhs[n])
        if m == INF:
            return (INF, INF)
        return (m + self._h(self._start, n) + self._km, m)

    def _push(self, n: Node) -> None:
        k = self._key(n)
        self._in_queue[n] = k
        heapq.heappush(self._pq, (k, next(self._counter), n))

    def _top(self) -> tuple[tuple[float, float], Node] | None:
        # Pop stale (lazily removed / outdated-key) entries.
        while self._pq:
            k, _, n = self._pq[0]
            cur = self._in_queue.get(n)
            if cur is None or cur != k:
                heapq.heappop(self._pq)
                continue
            return k, n
        return None

    def _update_vertex(self, u: Node) -> None:
        if u != self._goal:
            best = INF
            for v, w in self._succ.get(u, {}).items():
                cand = w + self._g[v]
                if cand < best:
                    best = cand
            self._rhs[u] = best
        consistent = self._g[u] == self._rhs[u]
        if consistent:
            # Remove from queue if present.
            if u in self._in_queue:
                del self._in_queue[u]
        else:
            self._push(u)

    def _compute_shortest_path(self) -> None:
        self.pops = 0
        while True:
            top = self._top()
            start_key = self._key(self._start)
            if top is None:
                break
            k_old, u = top
            # Stop once start is locally consistent and no smaller key remains.
            if not (k_old < start_key or self._rhs[self._start] != self._g[self._start]):
                break
            self.pops += 1
            k_new = self._key(u)
            if k_old < k_new:
                # Key was outdated (rose); reinsert with the new key.
                self._push(u)
                continue
            # Pop u from the queue.
            heapq.heappop(self._pq)
            if self._in_queue.get(u) == k_old:
                del self._in_queue[u]
            if self._g[u] > self._rhs[u]:
                # Overconsistent -> make consistent.
                self._g[u] = self._rhs[u]
                for p in self._pred.get(u, {}):
                    self._update_vertex(p)
            else:
                # Underconsistent -> set to INF and re-evaluate u and preds.
                self._g[u] = INF
                self._update_vertex(u)
                for p in self._pred.get(u, {}):
                    self._update_vertex(p)

    # -- public API (IncrementalSP protocol) -----------------------------------

    def update_edges(self, costs: dict[Edge, float]) -> None:
        """Apply a batch of edge-cost changes `{(u, v): new_cost}`.

        Raises `ValueError` on a non-positive cost or on an edge absent from the
        initial graph (topology is fixed). All changes are applied before any
        affected vertices are re-evaluated, matching the batch semantics of the
        D* Lite main loop (the agent senses several edges, then repairs once).
        """
        # Validate first so a bad batch leaves state untouched.
        for (u, v), w in costs.items():
            if w <= 0:
                raise ValueError(f"edge cost must be > 0, got {w} for {(u, v)}")
            if u not in self._succ or v not in self._succ[u]:
                raise ValueError(f"edge {(u, v)} not in initial graph; cannot add edges")
        for (u, v), w in costs.items():
            self._succ[u][v] = w
            self._pred[v][u] = w
        # Re-evaluate the tail of each changed edge (and the head if it is goal-
        # rooted rhs source). Updating `u` covers c(u, v) + g(v).
        touched: set[Node] = set()
        for (u, v) in costs:
            touched.add(u)
        for u in touched:
            self._update_vertex(u)

    def set_start(self, node: Node) -> None:
        """Move the agent's start to `node` (Koenig & Likhachev `km` technique).

        Increments `km` by `h(old_start, new_start)` so previously computed keys
        remain valid lower bounds, avoiding a queue rebuild.
        """
        if node not in self._succ:
            raise ValueError(f"start node {node!r} not in graph")
        if node == self._start:
            return
        self._km += self._h(self._start, node)
        self._start = node

    def shortest_path(self) -> tuple[list[Node] | None, float]:
        """Repair and return `(path, cost)` from current start to goal.

        Returns `(None, inf)` if the goal is unreachable. The path is extracted
        by greedy descent `argmin_v c(u, v) + g(v)`, guarded against cycles.
        """
        self._compute_shortest_path()
        cost = self._rhs[self._start] if self._start != self._goal else self._g[self._goal]
        # rhs(start) is the optimal cost; g(start) equals it once consistent.
        cost = min(self._g[self._start], self._rhs[self._start])
        if self._start == self._goal:
            return [self._start], 0.0
        if cost == INF:
            return None, INF
        path = [self._start]
        u = self._start
        seen = {u}
        while u != self._goal:
            best_v = None
            best = INF
            for v, w in self._succ.get(u, {}).items():
                cand = w + self._g[v]
                if cand < best:
                    best = cand
                    best_v = v
            if best_v is None or best == INF or best_v in seen:
                # No descent possible: unreachable or numerical cycle guard.
                return None, INF
            u = best_v
            seen.add(u)
            path.append(u)
        return path, cost
