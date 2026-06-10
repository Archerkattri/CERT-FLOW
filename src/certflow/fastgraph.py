"""Flat-array shortest-path engine for CERT (performance port of `graphcore`).

This module reimplements the planner's hot path — Dijkstra and D* Lite — on
flat NumPy arrays (CSR adjacency, int32 indices, float64 costs) instead of the
dict-of-dict adjacency and dict-keyed g/rhs/key state used by
`certflow.graphcore`. The algorithms and their SEMANTICS are identical; only the
data layout changes:

* nodes are integers 0..n-1 (the public API maps arbitrary hashable nodes to
  indices and back, so callers keep their `(row, col)` tuples);
* successors live in a forward CSR (`indptr`, `indices`, `cost`) and
  predecessors in a reverse CSR (`r_indptr`, `r_indices`, plus a map from each
  reverse slot to the forward slot so an edge cost is stored exactly once);
* `g`, `rhs`, and the two key components live in float64 arrays;
* the D* Lite priority queue holds integer node ids.

`FastDStarLite` matches `graphcore.DStarLite` bit-for-bit on cost (verified by
the differential tests in `tests/test_fastgraph.py`): goal-rooted g/rhs, the
`km` key modifier on `set_start`, batch `update_edges({(u, v): cost})`,
`shortest_path() -> (path, cost)`, and the `pops` expansion counter.

A numba `@njit` kernel accelerates the inner `_compute_shortest_path` loop when
numba imports cleanly; a pure-array-Python fallback is always present and
selected automatically, so numba is an optional dependency.

`fast_metrics` is a vectorized version of the per-edge `EdgeBelief.lower/upper`
loop the planner runs every round; it returns the same numbers the dict loop
does (verified bit-for-bit).
"""
from __future__ import annotations

import heapq
import itertools
from typing import Iterable

import numpy as np

from certflow.types import Edge, Node

INF = float("inf")

# Optional numba kernel. Pure-array-Python fallback is always available.
try:  # pragma: no cover - import guard
    from numba import njit  # type: ignore

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is optional
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        """No-op decorator standing in for numba when it is unavailable."""
        def wrap(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


class FlatGraph:
    """CSR-style flat representation of a goal-rooted adjacency map.

    Built once from `graph[u][v] = c(u, v)`. Nodes are assigned integer indices
    in a deterministic order (sorted-if-possible, else first-seen) and the maps
    `node_of`/`index_of` translate between caller node objects and indices.

    Forward CSR (successors of u):  edges of u are
        ``indices[indptr[u]:indptr[u+1]]`` with costs ``cost[...]``.
    Reverse CSR (predecessors of v): preds of v are
        ``r_indices[r_indptr[v]:r_indptr[v+1]]``; ``r_to_fwd[...]`` maps each
        reverse slot to its forward-CSR slot, so ``cost[r_to_fwd[k]]`` is the
        cost of edge ``(pred, v)`` — the cost array is the single source of
        truth, shared by both directions.
    """

    __slots__ = (
        "n",
        "nodes",
        "index_of",
        "indptr",
        "indices",
        "cost",
        "r_indptr",
        "r_indices",
        "r_to_fwd",
        "_edge_slot",
    )

    def __init__(self, graph: dict[Node, dict[Node, float]], extra_nodes: Iterable[Node] = ()):
        # Deterministic node ordering. Try a total order; fall back to first-seen.
        node_set: set[Node] = set(graph)
        for u, nbrs in graph.items():
            node_set.update(nbrs)
        node_set.update(extra_nodes)
        try:
            nodes = sorted(node_set)
        except TypeError:
            seen: dict[Node, None] = {}
            for u, nbrs in graph.items():
                seen.setdefault(u, None)
                for v in nbrs:
                    seen.setdefault(v, None)
            for x in extra_nodes:
                seen.setdefault(x, None)
            nodes = list(seen)

        self.nodes: list[Node] = nodes
        self.n: int = len(nodes)
        self.index_of: dict[Node, int] = {node: i for i, node in enumerate(nodes)}

        # Forward CSR.
        idx = self.index_of
        out_counts = np.zeros(self.n + 1, dtype=np.int64)
        for u, nbrs in graph.items():
            out_counts[idx[u]] = len(nbrs)
        indptr = np.zeros(self.n + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(out_counts[:-1])  # prefix sums into slots 1..n
        m = int(indptr[self.n])

        indices = np.empty(m, dtype=np.int32)
        cost = np.empty(m, dtype=np.float64)
        # `_edge_slot[(u_idx, v_idx)]` -> forward slot, for cost updates.
        edge_slot: dict[tuple[int, int], int] = {}
        cursor = indptr[:-1].astype(np.int64).copy()
        for u, nbrs in graph.items():
            ui = idx[u]
            for v, w in nbrs.items():
                vi = idx[v]
                s = int(cursor[ui])
                indices[s] = vi
                cost[s] = w
                edge_slot[(ui, vi)] = s
                cursor[ui] += 1

        self.indptr = indptr
        self.indices = indices
        self.cost = cost
        self._edge_slot = edge_slot

        # Reverse CSR (predecessors). Built from the forward edges.
        in_counts = np.zeros(self.n + 1, dtype=np.int64)
        for vi in indices:
            in_counts[vi] += 1
        r_indptr = np.zeros(self.n + 1, dtype=np.int32)
        r_indptr[1:] = np.cumsum(in_counts[:-1])
        r_indices = np.empty(m, dtype=np.int32)
        r_to_fwd = np.empty(m, dtype=np.int32)
        r_cursor = r_indptr[:-1].astype(np.int64).copy()
        for u in range(self.n):
            for s in range(int(indptr[u]), int(indptr[u + 1])):
                vi = int(indices[s])
                t = int(r_cursor[vi])
                r_indices[t] = u
                r_to_fwd[t] = s
                r_cursor[vi] += 1
        self.r_indptr = r_indptr
        self.r_indices = r_indices
        self.r_to_fwd = r_to_fwd

    def node_of(self, i: int) -> Node:
        return self.nodes[i]

    def slot_of(self, u_idx: int, v_idx: int) -> int:
        """Forward-CSR slot of edge (u_idx, v_idx), or -1 if absent."""
        return self._edge_slot.get((u_idx, v_idx), -1)

    def set_cost(self, slot: int, w: float) -> None:
        self.cost[slot] = w


# --------------------------------------------------------------------------- #
# Dijkstra
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _dijkstra_kernel(indptr, indices, cost, source, target, dist, prev):
    """Early-exit Dijkstra over CSR arrays (audit flag closed: FastDijkstra
    previously had no numba kernel, blocking hot-path use). Array binary
    heap with insertion-counter tie-break matching the Python path."""
    n = dist.shape[0]
    for i in range(n):
        dist[i] = np.inf
        prev[i] = -1
    dist[source] = 0.0
    cap = 4 * n
    hk = np.empty(cap, dtype=np.float64)
    hc = np.empty(cap, dtype=np.int64)
    hn = np.empty(cap, dtype=np.int64)
    hk[0] = 0.0
    hc[0] = 0
    hn[0] = source
    size = 1
    counter = 1
    while size > 0:
        kd = hk[0]
        ku = hn[0]
        size -= 1
        hk[0] = hk[size]
        hc[0] = hc[size]
        hn[0] = hn[size]
        i = 0
        while True:
            l, r = 2 * i + 1, 2 * i + 2
            m = i
            if l < size and (hk[l] < hk[m] or (hk[l] == hk[m] and hc[l] < hc[m])):
                m = l
            if r < size and (hk[r] < hk[m] or (hk[r] == hk[m] and hc[r] < hc[m])):
                m = r
            if m == i:
                break
            hk[i], hk[m] = hk[m], hk[i]
            hc[i], hc[m] = hc[m], hc[i]
            hn[i], hn[m] = hn[m], hn[i]
            i = m
        if kd > dist[ku]:
            continue
        if ku == target:
            return
        for k in range(indptr[ku], indptr[ku + 1]):
            v = indices[k]
            nd = kd + cost[k]
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = ku
                if size >= cap:
                    return  # heap overflow guard (cannot occur with cap=4n)
                j = size
                hk[j] = nd
                hc[j] = counter
                hn[j] = v
                counter += 1
                size += 1
                while j > 0:
                    pj = (j - 1) // 2
                    if hk[pj] < hk[j] or (hk[pj] == hk[j] and hc[pj] <= hc[j]):
                        break
                    hk[pj], hk[j] = hk[j], hk[pj]
                    hc[pj], hc[j] = hc[j], hc[pj]
                    hn[pj], hn[j] = hn[j], hn[pj]
                    j = pj


class FastDijkstra:
    """Array-based forward Dijkstra over a `FlatGraph`.

    Matches `graphcore.dijkstra` semantics: positive costs, insertion-counter
    tie-break, optional target early-exit. Distances live in a float64 array.
    Uses a numba kernel when available (audit flag: previously Python-only).
    """

    def __init__(self, flat: FlatGraph, source_idx: int):
        self.flat = flat
        self.source = source_idx

    def shortest_path(self, target_idx: int) -> tuple[list[int] | None, float]:
        """Return `(path_indices, cost)` from source to target, or `(None, inf)`.

        Early-exits when the target is popped. `path_indices` is a node-index
        list; map through `flat.node_of` for caller node objects.
        """
        flat = self.flat
        source = self.source
        if source == target_idx:
            return [source], 0.0
        if _HAVE_NUMBA:
            n = flat.n
            dist = np.empty(n, dtype=np.float64)
            prev = np.empty(n, dtype=np.int64)
            _dijkstra_kernel(flat.indptr, flat.indices, flat.cost,
                             source, target_idx, dist, prev)
            if not np.isfinite(dist[target_idx]):
                return None, INF
            path = [target_idx]
            while path[-1] != source:
                path.append(int(prev[path[-1]]))
            path.reverse()
            return path, float(dist[target_idx])
        n = flat.n
        indptr = flat.indptr
        indices = flat.indices
        cost = flat.cost
        dist = np.full(n, INF, dtype=np.float64)
        prev = np.full(n, -1, dtype=np.int64)
        visited = np.zeros(n, dtype=bool)
        dist[source] = 0.0
        counter = itertools.count(1)
        pq: list[tuple[float, int, int]] = [(0.0, 0, source)]
        while pq:
            d, _, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            if u == target_idx:
                break
            for s in range(int(indptr[u]), int(indptr[u + 1])):
                v = int(indices[s])
                nd = d + cost[s]
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, next(counter), v))
        if not np.isfinite(dist[target_idx]):
            return None, INF
        path = [target_idx]
        while path[-1] != source:
            path.append(int(prev[path[-1]]))
        path.reverse()
        return path, float(dist[target_idx])


# --------------------------------------------------------------------------- #
# D* Lite — numba inner-loop kernel (optional) and pure-Python fallback
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _compute_kernel(
    g,
    rhs,
    indptr,
    indices,
    cost,
    r_indptr,
    r_indices,
    r_to_fwd,
    heap_key0,
    heap_key1,
    heap_node,
    heap_tie,
    heap_size,
    inq_key0,
    inq_key1,
    inq_live,
    counter0,
    km,
    start,
    goal,
):
    """numba kernel: D* Lite ComputeShortestPath on flat arrays.

    Mirrors `graphcore.DStarLite._compute_shortest_path` exactly, including the
    lazy-deletion queue (stale entries skipped by comparing each popped entry's
    key against the live key in `inq_*`) and the outdated-key reinsert branch.
    Heuristic is zero (the planner's default), so the first key component is
    just `min(g,rhs) + km`. Returns `(pops, counter)`.
    """
    INF_ = np.inf
    pops = 0
    counter = counter0

    # local binary-heap helpers operate on the parallel heap_* arrays.
    while True:
        # --- find live top (skip stale) ---
        top_node = -1
        while heap_size[0] > 0:
            k0 = heap_key0[0]
            k1 = heap_key1[0]
            node = heap_node[0]
            if (not inq_live[node]) or inq_key0[node] != k0 or inq_key1[node] != k1:
                # pop root
                heap_size[0] -= 1
                last = heap_size[0]
                heap_key0[0] = heap_key0[last]
                heap_key1[0] = heap_key1[last]
                heap_node[0] = heap_node[last]
                heap_tie[0] = heap_tie[last]
                # sift down
                i = 0
                while True:
                    l = 2 * i + 1
                    r = 2 * i + 2
                    sm = i
                    if l < heap_size[0] and _lt(
                        heap_key0[l], heap_key1[l], heap_tie[l],
                        heap_key0[sm], heap_key1[sm], heap_tie[sm],
                    ):
                        sm = l
                    if r < heap_size[0] and _lt(
                        heap_key0[r], heap_key1[r], heap_tie[r],
                        heap_key0[sm], heap_key1[sm], heap_tie[sm],
                    ):
                        sm = r
                    if sm == i:
                        break
                    _swap(heap_key0, heap_key1, heap_node, heap_tie, i, sm)
                    i = sm
                continue
            top_node = node
            break
        if top_node == -1:
            break

        # start key
        m_s = g[start] if g[start] < rhs[start] else rhs[start]
        if m_s == INF_:
            sk0 = INF_
            sk1 = INF_
        else:
            sk0 = m_s + km
            sk1 = m_s

        k_old0 = heap_key0[0]
        k_old1 = heap_key1[0]
        # stop condition: not (k_old < start_key or rhs[start] != g[start])
        cond = _lt(k_old0, k_old1, 0.0, sk0, sk1, 0.0) or (rhs[start] != g[start])
        if not cond:
            break

        pops += 1
        u = top_node
        # new key of u
        m_u = g[u] if g[u] < rhs[u] else rhs[u]
        if m_u == INF_:
            kn0 = INF_
            kn1 = INF_
        else:
            kn0 = m_u + km
            kn1 = m_u

        if _lt(k_old0, k_old1, 0.0, kn0, kn1, 0.0):
            # key rose: reinsert with new key
            inq_key0[u] = kn0
            inq_key1[u] = kn1
            inq_live[u] = True
            _push(heap_key0, heap_key1, heap_node, heap_tie, heap_size, kn0, kn1, u, counter)
            counter += 1
            continue

        # pop u
        heap_size[0] -= 1
        last = heap_size[0]
        heap_key0[0] = heap_key0[last]
        heap_key1[0] = heap_key1[last]
        heap_node[0] = heap_node[last]
        heap_tie[0] = heap_tie[last]
        i = 0
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            sm = i
            if l < heap_size[0] and _lt(
                heap_key0[l], heap_key1[l], heap_tie[l],
                heap_key0[sm], heap_key1[sm], heap_tie[sm],
            ):
                sm = l
            if r < heap_size[0] and _lt(
                heap_key0[r], heap_key1[r], heap_tie[r],
                heap_key0[sm], heap_key1[sm], heap_tie[sm],
            ):
                sm = r
            if sm == i:
                break
            _swap(heap_key0, heap_key1, heap_node, heap_tie, i, sm)
            i = sm
        if inq_key0[u] == k_old0 and inq_key1[u] == k_old1 and inq_live[u]:
            inq_live[u] = False

        if g[u] > rhs[u]:
            g[u] = rhs[u]
            # update predecessors
            for t in range(r_indptr[u], r_indptr[u + 1]):
                p = r_indices[t]
                counter = _update_vertex(
                    p, g, rhs, indptr, indices, cost,
                    inq_key0, inq_key1, inq_live,
                    heap_key0, heap_key1, heap_node, heap_tie, heap_size,
                    km, goal, counter,
                )
        else:
            g[u] = INF_
            counter = _update_vertex(
                u, g, rhs, indptr, indices, cost,
                inq_key0, inq_key1, inq_live,
                heap_key0, heap_key1, heap_node, heap_tie, heap_size,
                km, goal, counter,
            )
            for t in range(r_indptr[u], r_indptr[u + 1]):
                p = r_indices[t]
                counter = _update_vertex(
                    p, g, rhs, indptr, indices, cost,
                    inq_key0, inq_key1, inq_live,
                    heap_key0, heap_key1, heap_node, heap_tie, heap_size,
                    km, goal, counter,
                )

    return pops, counter


@njit(cache=True)
def _lt(a0, a1, atie, b0, b1, btie):
    """Lexicographic (key0, key1, tie) less-than for heap ordering."""
    if a0 < b0:
        return True
    if a0 > b0:
        return False
    if a1 < b1:
        return True
    if a1 > b1:
        return False
    return atie < btie


@njit(cache=True)
def _swap(k0, k1, node, tie, i, j):
    k0[i], k0[j] = k0[j], k0[i]
    k1[i], k1[j] = k1[j], k1[i]
    node[i], node[j] = node[j], node[i]
    tie[i], tie[j] = tie[j], tie[i]


@njit(cache=True)
def _push(heap_key0, heap_key1, heap_node, heap_tie, heap_size, k0, k1, node, tie):
    i = heap_size[0]
    heap_key0[i] = k0
    heap_key1[i] = k1
    heap_node[i] = node
    heap_tie[i] = tie
    heap_size[0] += 1
    # sift up
    while i > 0:
        parent = (i - 1) // 2
        if _lt(
            heap_key0[i], heap_key1[i], heap_tie[i],
            heap_key0[parent], heap_key1[parent], heap_tie[parent],
        ):
            _swap(heap_key0, heap_key1, heap_node, heap_tie, i, parent)
            i = parent
        else:
            break


@njit(cache=True)
def _update_vertex(
    u, g, rhs, indptr, indices, cost,
    inq_key0, inq_key1, inq_live,
    heap_key0, heap_key1, heap_node, heap_tie, heap_size,
    km, goal, counter,
):
    """Recompute rhs[u] (if u != goal) and (re)queue u if inconsistent."""
    INF_ = np.inf
    if u != goal:
        best = INF_
        for s in range(indptr[u], indptr[u + 1]):
            v = indices[s]
            cand = cost[s] + g[v]
            if cand < best:
                best = cand
        rhs[u] = best
    if g[u] == rhs[u]:
        inq_live[u] = False
    else:
        m = g[u] if g[u] < rhs[u] else rhs[u]
        if m == INF_:
            k0 = INF_
            k1 = INF_
        else:
            k0 = m + km
            k1 = m
        inq_key0[u] = k0
        inq_key1[u] = k1
        inq_live[u] = True
        _push(heap_key0, heap_key1, heap_node, heap_tie, heap_size, k0, k1, u, counter)
        counter += 1
    return counter


@njit(cache=True)
def _update_tails_kernel(
    tails,
    g, rhs, indptr, indices, cost,
    inq_key0, inq_key1, inq_live,
    heap_key0, heap_key1, heap_node, heap_tie, heap_size,
    km, goal, counter0,
):
    """Re-evaluate each tail vertex after a batch cost update (numba path).

    Mirrors the loop in `update_edges` that calls `_update_vertex` on every
    changed edge's tail, on persistent flat queue arrays. Returns the new
    counter value.
    """
    counter = counter0
    for i in range(tails.shape[0]):
        counter = _update_vertex(
            tails[i], g, rhs, indptr, indices, cost,
            inq_key0, inq_key1, inq_live,
            heap_key0, heap_key1, heap_node, heap_tie, heap_size,
            km, goal, counter,
        )
    return counter


class FastDStarLite:
    """D* Lite on flat NumPy arrays — same algorithm/semantics as
    `graphcore.DStarLite` (goal-rooted g/rhs, `km` offset, batch
    `update_edges`, `shortest_path() -> (path, cost)`, `pops` counter).

    The public API takes and returns caller node objects; internally everything
    is integer-indexed. `flat` may be shared/rebuilt by the caller; edge-cost
    updates write straight into `flat.cost`.

    Heuristic: zero (admissible/consistent for all positive costs), matching the
    planner's default. A custom heuristic is intentionally not supported here —
    the planner never supplies one.
    """

    def __init__(
        self,
        graph: dict[Node, dict[Node, float]],
        start: Node,
        goal: Node,
        flat: FlatGraph | None = None,
        use_numba: bool | None = None,
    ) -> None:
        self.flat = flat if flat is not None else FlatGraph(graph, extra_nodes=(start, goal))
        if start not in self.flat.index_of or goal not in self.flat.index_of:
            # caller-provided flat without these nodes: rebuild including them
            self.flat = FlatGraph(graph, extra_nodes=(start, goal))
        self._start_node = start
        self._goal_node = goal
        self._start = self.flat.index_of[start]
        self._goal = self.flat.index_of[goal]
        self.pops = 0
        self._use_numba = _HAVE_NUMBA if use_numba is None else (use_numba and _HAVE_NUMBA)
        self._init_search()

    # -- internal state --------------------------------------------------------

    def _init_search(self) -> None:
        n = self.flat.n
        self._g = np.full(n, INF, dtype=np.float64)
        self._rhs = np.full(n, INF, dtype=np.float64)
        self._km = 0.0
        self._rhs[self._goal] = 0.0
        if self._use_numba:
            self._init_search_numba()
        else:
            self._counter = itertools.count(1)
            self._heap: list[tuple[tuple[float, float], int, int]] = []
            self._inq_key: dict[int, tuple[float, float]] = {}
            self._push(self._goal)

    def _init_search_numba(self) -> None:
        """Persistent flat queue state for the numba backend (no per-call
        marshalling). The heap arrays grow on demand via `_ensure_heap_cap`."""
        n = self.flat.n
        cap = max(64, 2 * n)
        self._heap_key0 = np.empty(cap, dtype=np.float64)
        self._heap_key1 = np.empty(cap, dtype=np.float64)
        self._heap_node = np.empty(cap, dtype=np.int64)
        self._heap_tie = np.empty(cap, dtype=np.float64)
        self._heap_size = np.zeros(1, dtype=np.int64)
        self._inq_key0 = np.full(n, INF, dtype=np.float64)
        self._inq_key1 = np.full(n, INF, dtype=np.float64)
        self._inq_live = np.zeros(n, dtype=np.bool_)
        self._counter_val = 1
        # seed the queue with the goal (rhs[goal]=0 -> key (km, 0))
        g0 = self._goal
        k0 = self._km
        k1 = 0.0
        self._inq_key0[g0] = k0
        self._inq_key1[g0] = k1
        self._inq_live[g0] = True
        self._heap_key0[0] = k0
        self._heap_key1[0] = k1
        self._heap_node[0] = g0
        self._heap_tie[0] = float(self._counter_val)
        self._heap_size[0] = 1
        self._counter_val += 1

    def _ensure_heap_cap(self, extra: int) -> None:
        """Grow the flat heap arrays so `extra` more pushes fit."""
        need = int(self._heap_size[0]) + extra
        cap = self._heap_key0.shape[0]
        if need <= cap:
            return
        new_cap = max(need, 2 * cap)
        self._heap_key0 = np.resize(self._heap_key0, new_cap)
        self._heap_key1 = np.resize(self._heap_key1, new_cap)
        self._heap_node = np.resize(self._heap_node, new_cap)
        self._heap_tie = np.resize(self._heap_tie, new_cap)

    def _key(self, n: int) -> tuple[float, float]:
        g = self._g[n]
        rhs = self._rhs[n]
        m = g if g < rhs else rhs
        if m == INF:
            return (INF, INF)
        return (m + self._km, m)

    def _push(self, n: int) -> None:
        k = self._key(n)
        self._inq_key[n] = k
        heapq.heappush(self._heap, (k, next(self._counter), n))

    def _top(self) -> tuple[tuple[float, float], int] | None:
        while self._heap:
            k, _, n = self._heap[0]
            cur = self._inq_key.get(n)
            if cur is None or cur != k:
                heapq.heappop(self._heap)
                continue
            return k, n
        return None

    def _update_vertex(self, u: int) -> None:
        flat = self.flat
        if u != self._goal:
            best = INF
            for s in range(int(flat.indptr[u]), int(flat.indptr[u + 1])):
                v = int(flat.indices[s])
                cand = flat.cost[s] + self._g[v]
                if cand < best:
                    best = cand
            self._rhs[u] = best
        if self._g[u] == self._rhs[u]:
            if u in self._inq_key:
                del self._inq_key[u]
        else:
            self._push(u)

    def _compute_shortest_path_py(self) -> None:
        self.pops = 0
        flat = self.flat
        g = self._g
        rhs = self._rhs
        start = self._start
        while True:
            top = self._top()
            # start key
            ms = g[start]
            if rhs[start] < ms:
                ms = rhs[start]
            start_key = (INF, INF) if ms == INF else (ms + self._km, ms)
            if top is None:
                break
            k_old, u = top
            if not (k_old < start_key or rhs[start] != g[start]):
                break
            self.pops += 1
            k_new = self._key(u)
            if k_old < k_new:
                self._push(u)
                continue
            heapq.heappop(self._heap)
            if self._inq_key.get(u) == k_old:
                del self._inq_key[u]
            if g[u] > rhs[u]:
                g[u] = rhs[u]
                for t in range(int(flat.r_indptr[u]), int(flat.r_indptr[u + 1])):
                    self._update_vertex(int(flat.r_indices[t]))
            else:
                g[u] = INF
                self._update_vertex(u)
                for t in range(int(flat.r_indptr[u]), int(flat.r_indptr[u + 1])):
                    self._update_vertex(int(flat.r_indices[t]))

    def _compute_shortest_path_numba(self) -> None:
        """Run the compute kernel on persistent flat queue state (no
        marshalling). Bit-identical results to the Python path."""
        flat = self.flat
        # Each iteration may push a few entries; pre-grow generously so the
        # numba kernel never overruns the heap arrays (no in-kernel realloc).
        self._ensure_heap_cap(4 * flat.n + 2 * flat.indices.size + 16)
        pops, counter_end = _compute_kernel(
            self._g, self._rhs,
            flat.indptr, flat.indices, flat.cost,
            flat.r_indptr, flat.r_indices, flat.r_to_fwd,
            self._heap_key0, self._heap_key1, self._heap_node, self._heap_tie,
            self._heap_size,
            self._inq_key0, self._inq_key1, self._inq_live,
            self._counter_val, self._km, self._start, self._goal,
        )
        self.pops = int(pops)
        self._counter_val = int(counter_end)

    def _compute_shortest_path(self) -> None:
        if self._use_numba:
            self._compute_shortest_path_numba()
        else:
            self._compute_shortest_path_py()

    # -- public API ------------------------------------------------------------

    def update_edges(self, costs: dict[Edge, float]) -> None:
        """Apply a batch of edge-cost changes `{(u, v): new_cost}`.

        Same validation/semantics as `graphcore.DStarLite.update_edges`: raises
        `ValueError` on non-positive cost or on an edge absent from the initial
        graph; all costs are written before any affected vertex is re-evaluated;
        only the tail `u` of each changed edge is touched.
        """
        flat = self.flat
        idx = flat.index_of
        slots: list[tuple[int, float]] = []
        tails: set[int] = set()
        for (u, v), w in costs.items():
            if w <= 0:
                raise ValueError(f"edge cost must be > 0, got {w} for {(u, v)}")
            ui = idx.get(u, -1)
            vi = idx.get(v, -1)
            slot = flat.slot_of(ui, vi) if (ui >= 0 and vi >= 0) else -1
            if slot < 0:
                raise ValueError(f"edge {(u, v)} not in initial graph; cannot add edges")
            slots.append((slot, w))
            tails.add(ui)
        for slot, w in slots:
            flat.cost[slot] = w
        if self._use_numba:
            tail_arr = np.fromiter(tails, dtype=np.int64, count=len(tails))
            self._ensure_heap_cap(len(tails))
            self._counter_val = int(_update_tails_kernel(
                tail_arr, self._g, self._rhs,
                flat.indptr, flat.indices, flat.cost,
                self._inq_key0, self._inq_key1, self._inq_live,
                self._heap_key0, self._heap_key1, self._heap_node, self._heap_tie,
                self._heap_size, self._km, self._goal, self._counter_val,
            ))
        else:
            for ui in tails:
                self._update_vertex(ui)

    def set_start(self, node: Node) -> None:
        """Move the agent's start to `node`. Heuristic is zero, so `km` is
        unchanged (matching `graphcore.DStarLite` with the zero heuristic)."""
        if node not in self.flat.index_of:
            raise ValueError(f"start node {node!r} not in graph")
        ni = self.flat.index_of[node]
        if ni == self._start:
            return
        # km += h(old_start, new_start); zero heuristic => += 0.
        self._start_node = node
        self._start = ni

    def shortest_path(self) -> tuple[list[Node] | None, float]:
        """Repair and return `(path, cost)` from current start to goal in
        caller node objects, or `(None, inf)` if the goal is unreachable."""
        self._compute_shortest_path()
        flat = self.flat
        g = self._g
        start = self._start
        goal = self._goal
        cost = min(g[start], self._rhs[start])
        if start == goal:
            return [self._goal_node], 0.0
        if cost == INF:
            return None, INF
        path_idx = [start]
        u = start
        seen = {u}
        while u != goal:
            best_v = -1
            best = INF
            for s in range(int(flat.indptr[u]), int(flat.indptr[u + 1])):
                v = int(flat.indices[s])
                cand = flat.cost[s] + g[v]
                if cand < best:
                    best = cand
                    best_v = v
            if best_v < 0 or best == INF or best_v in seen:
                return None, INF
            u = best_v
            seen.add(u)
            path_idx.append(u)
        return [flat.node_of(i) for i in path_idx], float(cost)


# --------------------------------------------------------------------------- #
# Vectorized edge-interval metrics
# --------------------------------------------------------------------------- #


def fast_metrics(
    c_hat: np.ndarray,
    t_obs: np.ndarray,
    rho: np.ndarray,
    t: float,
    q: float,
    cost_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-edge lower/upper interval metrics.

    Equivalent to running, over edges,
        lo = max(cost_floor, c_hat - q - rho*(t - t_obs))
        up = max(cost_floor, c_hat + q + rho*(t - t_obs))
    i.e. `EdgeBelief.lower`/`EdgeBelief.upper` (`certflow.types`). Inputs are
    parallel float arrays; returns `(lo, up)` float64 arrays.

    `q` may be `inf` (warm-up): then `age*rho` is `inf` for rho>0 and the
    clip pins `lo` at the floor while `up` is `inf` — matching the scalar
    `max(cost_floor, c_hat - inf)` / `max(cost_floor, c_hat + inf)`.
    """
    c_hat = np.asarray(c_hat, dtype=np.float64)
    t_obs = np.asarray(t_obs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    age = t - t_obs
    drift = rho * age
    # Match EdgeBelief.lower/upper's left-to-right associativity bit-for-bit:
    #   lower = (c_hat - q) - rho*age,  upper = (c_hat + q) + rho*age.
    # Folding q and drift into one `half` term first changes the rounding.
    lo = np.maximum(cost_floor, (c_hat - q) - drift)
    up = np.maximum(cost_floor, (c_hat + q) + drift)
    return lo, up


# --------------------------------------------------------------------------- #
# Benchmark (python -m certflow.fastgraph or pytest -k benchmark)
# --------------------------------------------------------------------------- #


def _grid_graph(rows: int, cols: int, rng) -> dict:
    """4-connected directed grid with random positive edge costs."""
    graph: dict = {}
    for r in range(rows):
        for c in range(cols):
            graph[(r, c)] = {}
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    graph[(r, c)][(nr, nc)] = round(rng.uniform(1.0, 10.0), 3)
    return graph


def run_benchmark(rows: int = 60, cols: int = 60, rounds: int = 100, seed: int = 0):
    """60x60 full-update + recompute round: old engine vs FastDStarLite vs
    FastDijkstra-from-scratch. Returns a dict of p50/p95 (ms) per engine."""
    import random
    from time import perf_counter

    from certflow.graphcore import DStarLite

    rng = random.Random(seed)
    graph = _grid_graph(rows, cols, rng)
    start, goal = (0, 0), (rows - 1, cols - 1)
    n_edges = sum(len(v) for v in graph.values())
    edges = [(u, v) for u in graph for v in graph[u]]

    # Build engines.
    flat = FlatGraph(graph, extra_nodes=(start, goal))
    fast = FastDStarLite(graph, start, goal, flat=flat)
    old = DStarLite(graph, start, goal)
    fast.shortest_path()
    old.shortest_path()
    # warm the numba kernel (compile cost out of the timed region)
    if fast._use_numba:
        fast.update_edges({edges[0]: graph[edges[0][0]][edges[0][1]]})
        fast.shortest_path()

    # Precompute the per-round full-graph cost updates (same across engines).
    batches = []
    for _ in range(rounds):
        batch = {}
        for (u, v) in edges:
            batch[(u, v)] = round(rng.uniform(1.0, 10.0), 3)
        batches.append(batch)

    def percentiles(samples):
        s = sorted(samples)
        p50 = s[len(s) // 2]
        p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
        return p50 * 1e3, p95 * 1e3

    old_t, fast_t, dij_t = [], [], []
    # Scratch-rebuild path (what the planner's _rebuild_searches / B=0 exact
    # mode actually does: a fresh search instance every full-refresh round).
    old_scratch_t, fast_scratch_t = [], []
    idx = flat.index_of
    src_i, dst_i = idx[start], idx[goal]
    for batch in batches:
        # incremental update + recompute (full-graph update = worst case for
        # incremental repair).
        t0 = perf_counter()
        old.update_edges(batch)
        old.shortest_path()
        old_t.append(perf_counter() - t0)
        t0 = perf_counter()
        fast.update_edges(batch)
        fast.shortest_path()
        fast_t.append(perf_counter() - t0)
        # costs are now applied in `flat`; mirror them into `graph` for the
        # dict-based scratch rebuilds.
        for (u, v), w in batch.items():
            graph[u][v] = w
        # FastDijkstra from scratch.
        t0 = perf_counter()
        FastDijkstra(flat, src_i).shortest_path(dst_i)
        dij_t.append(perf_counter() - t0)
        # old DStarLite from scratch (the planner's actual full-refresh cost).
        t0 = perf_counter()
        DStarLite(graph, start, goal).shortest_path()
        old_scratch_t.append(perf_counter() - t0)
        # FastDStarLite from scratch (reuses warm numba kernel + flat arrays).
        t0 = perf_counter()
        FastDStarLite(graph, start, goal, flat=flat).shortest_path()
        fast_scratch_t.append(perf_counter() - t0)

    out = {
        "rows": rows,
        "cols": cols,
        "n_edges": n_edges,
        "rounds": rounds,
        "numba": fast._use_numba,
        "old": percentiles(old_t),
        "fast": percentiles(fast_t),
        "dijkstra": percentiles(dij_t),
        "old_scratch": percentiles(old_scratch_t),
        "fast_scratch": percentiles(fast_scratch_t),
    }
    return out


if __name__ == "__main__":  # pragma: no cover
    res = run_benchmark()
    o50, o95 = res["old"]
    f50, f95 = res["fast"]
    d50, d95 = res["dijkstra"]
    print(f"grid {res['rows']}x{res['cols']}  edges={res['n_edges']}  "
          f"rounds={res['rounds']}  numba={res['numba']}")
    os50, os95 = res["old_scratch"]
    fs50, fs95 = res["fast_scratch"]
    print(f"{'engine':<32}{'p50 (ms)':>12}{'p95 (ms)':>12}")
    print("-- incremental full-graph update + recompute --")
    print(f"{'graphcore.DStarLite (old)':<32}{o50:>12.3f}{o95:>12.3f}")
    print(f"{'FastDStarLite':<32}{f50:>12.3f}{f95:>12.3f}")
    print(f"{'FastDijkstra (scratch)':<32}{d50:>12.3f}{d95:>12.3f}")
    print("-- scratch rebuild (planner _rebuild_searches / B=0) --")
    print(f"{'graphcore.DStarLite scratch':<32}{os50:>12.3f}{os95:>12.3f}")
    print(f"{'FastDStarLite scratch':<32}{fs50:>12.3f}{fs95:>12.3f}")
    print(f"incremental speedup (old/fast p50):  {o50 / f50:.1f}x")
    print(f"scratch speedup (old/fast p50):      {os50 / fs50:.1f}x")
