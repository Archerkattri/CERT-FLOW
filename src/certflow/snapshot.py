"""Certified snapshot oracle: certificate-gated preprocessing.

Static-known-grid planners answer in microseconds by ASSUMING the costs are
valid; CERT can answer at lookup speed by PROVING it. When the planner's
certificate establishes the map is currently tight (every edge interval
width <= tau), that proof licenses building an all-pairs structure on the
certified point estimates; queries are then O(1) array lookups carrying an
explicit per-query certificate (true cost within +/- sum of path
half-widths <= L*tau/2, at the certificate's confidence). The same
certificate tells us exactly when the oracle EXPIRES: any width exceeding
tau invalidates the snapshot, and the planner falls back to its online
machinery. Preprocessing-by-assumption becomes preprocessing-by-proof.

Scope: all-pairs is feasible only while its configured memory budget permits
it (n Dijkstras to build, O(n^2) memory). Larger graphs use the ALT layer
(roadnet.py) under the same gate. The budget is checked before allocating the
quadratic tables; callers receive an explicit resource-limit diagnostic rather
than a partially certified or silently downgraded snapshot.
"""
from __future__ import annotations

import time

import numpy as np

from certflow.fastgraph import FlatGraph, _HAVE_NUMBA


DEFAULT_MAX_BYTES = 512 * 1024 * 1024


class SnapshotResourceError(MemoryError):
    """Raised before an all-pairs snapshot would exceed its memory budget."""

    def __init__(self, *, n_nodes: int, estimated_bytes: int, max_bytes: int) -> None:
        self.n_nodes = int(n_nodes)
        self.estimated_bytes = int(estimated_bytes)
        self.max_bytes = int(max_bytes)
        super().__init__(
            "snapshot resource limit: "
            f"n_nodes={self.n_nodes}, estimated_bytes={self.estimated_bytes}, "
            f"max_bytes={self.max_bytes}; use a bounded/ALT planner path"
        )


def estimate_snapshot_bytes(n_nodes: int) -> int:
    """Estimate tables plus linear scratch space before an all-pairs build."""
    n = int(n_nodes)
    if n < 0:
        raise ValueError("n_nodes must be non-negative")
    # dist/parent tables plus dijkstra scratch. The numba heap is 4*n entries
    # of float64 + int64; the fallback uses comparable linear arrays. This is a
    # lower-bound estimate for Python allocator overhead, deliberately rounded
    # conservatively upward by one linear scratch block.
    table_bytes = n * n * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
    scratch_bytes = n * (8 + 8 + (4 * 8) + (4 * 8))
    return int(table_bytes + scratch_bytes)

if _HAVE_NUMBA:
    import numba

    @numba.njit(cache=True)
    def _sssp_kernel(indptr, indices, cost, src, dist, parent):
        n = dist.shape[0]
        for i in range(n):
            dist[i] = np.inf
            parent[i] = -1
        dist[src] = 0.0
        # binary heap in arrays
        heap_key = np.empty(4 * n, dtype=np.float64)
        heap_node = np.empty(4 * n, dtype=np.int64)
        size = 0
        heap_key[0] = 0.0
        heap_node[0] = src
        size = 1
        while size > 0:
            kd = heap_key[0]
            u = heap_node[0]
            size -= 1
            heap_key[0] = heap_key[size]
            heap_node[0] = heap_node[size]
            i = 0
            while True:
                l, r = 2 * i + 1, 2 * i + 2
                m = i
                if l < size and heap_key[l] < heap_key[m]:
                    m = l
                if r < size and heap_key[r] < heap_key[m]:
                    m = r
                if m == i:
                    break
                heap_key[i], heap_key[m] = heap_key[m], heap_key[i]
                heap_node[i], heap_node[m] = heap_node[m], heap_node[i]
                i = m
            if kd > dist[u]:
                continue
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                nd = kd + cost[k]
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    j = size
                    heap_key[j] = nd
                    heap_node[j] = v
                    size += 1
                    while j > 0:
                        p = (j - 1) // 2
                        if heap_key[p] <= heap_key[j]:
                            break
                        heap_key[p], heap_key[j] = heap_key[j], heap_key[p]
                        heap_node[p], heap_node[j] = heap_node[j], heap_node[p]
                        j = p
        return dist, parent


class SnapshotOracle:
    """All-pairs oracle over a FlatGraph snapshot, with certificate gating
    handled by the caller (see CertPlanner.snapshot_query)."""

    def __init__(self, flat: FlatGraph, *, max_bytes: int = DEFAULT_MAX_BYTES):
        self.flat = flat
        self.n = flat.n
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.estimated_bytes = estimate_snapshot_bytes(self.n)
        self._dist: np.ndarray | None = None     # (n, n) float32
        self._parent: np.ndarray | None = None   # (n, n) int32
        self.built_at: float | None = None
        self.build_seconds: float | None = None

    @property
    def ready(self) -> bool:
        """True once build() has populated the all-pairs tables."""
        return self._dist is not None

    def build(self, t: float) -> float:
        """All-pairs by n single-source runs on the current flat costs."""
        t0 = time.perf_counter()
        n = self.n
        # This check must remain before either quadratic np.empty call. A
        # caller may opt into a larger budget explicitly, but the default is
        # bounded and an over-budget result is never exposed as a certificate.
        if self.estimated_bytes > self.max_bytes:
            raise SnapshotResourceError(
                n_nodes=n,
                estimated_bytes=self.estimated_bytes,
                max_bytes=self.max_bytes,
            )
        dist = np.empty((n, n), dtype=np.float32)
        parent = np.empty((n, n), dtype=np.int32)
        d = np.empty(n, dtype=np.float64)
        p = np.empty(n, dtype=np.int64)
        for s in range(n):
            if _HAVE_NUMBA:
                _sssp_kernel(self.flat.indptr, self.flat.indices,
                             self.flat.cost, s, d, p)
                dist[s] = d
                parent[s] = p
            else:  # pure-python fallback: heapq dijkstra
                import heapq
                dd = np.full(n, np.inf)
                pp = np.full(n, -1, dtype=np.int64)
                dd[s] = 0.0
                pq = [(0.0, s)]
                while pq:
                    kd, u = heapq.heappop(pq)
                    if kd > dd[u]:
                        continue
                    for k in range(self.flat.indptr[u], self.flat.indptr[u + 1]):
                        v = self.flat.indices[k]
                        nd = kd + self.flat.cost[k]
                        if nd < dd[v]:
                            dd[v] = nd
                            pp[v] = u
                            heapq.heappush(pq, (nd, v))
                dist[s] = dd
                parent[s] = pp
        self._dist = dist
        self._parent = parent
        self.built_at = t
        self.build_seconds = time.perf_counter() - t0
        return self.build_seconds

    def invalidate(self) -> None:
        """Drop the built all-pairs tables (the snapshot has expired)."""
        self._dist = None
        self._parent = None
        self.built_at = None

    def cost(self, s_idx: int, g_idx: int) -> float:
        """Snapshot shortest-path cost s->g (inf if unreachable)."""
        if self._dist is None:
            raise RuntimeError("SnapshotOracle.cost before build(); call build() first")
        return float(self._dist[s_idx, g_idx])

    def path(self, s_idx: int, g_idx: int) -> list[int] | None:
        """Snapshot shortest path s->g as a node-index list, or None if
        unreachable. Requires a prior build()."""
        if self._dist is None or self._parent is None:
            raise RuntimeError("SnapshotOracle.path before build(); call build() first")
        if not np.isfinite(self._dist[s_idx, g_idx]):
            return None
        out = [g_idx]
        row = self._parent[s_idx]
        v = g_idx
        while v != s_idx:
            v = int(row[v])
            if v < 0:
                return None
            out.append(v)
        out.reverse()
        return out
