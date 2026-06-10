"""Real road-network scale loader + ALT (A*, Landmarks, Triangle inequality).

This module takes CERT's flat-array engine (`certflow.fastgraph`) to real
road-network scale using the 9th DIMACS Implementation Challenge road graphs
(USA-road-d.*.gr — distance metric, directed arcs, 1-indexed). It adds an ALT
accelerator: A* whose heuristic is the max over a small set of *landmarks* of a
triangle-inequality lower bound on the source-target distance.

What lives here (and why none of it touches `fastgraph.py`):

* `load_dimacs(path)` -> `RoadGraph`: streams the `.gr` file into CSR arrays
  directly (forward + reverse), skipping the dict-of-dict adjacency entirely —
  building a 1M-node dict is prohibitively slow and memory-hungry. `RoadGraph`
  exposes the exact attribute surface `FastDijkstra` reads (`n`, `indptr`,
  `indices`, `cost`, plus reverse CSR and a 1:1 integer node identity), so a
  `FastDijkstra(road, src)` runs unmodified.

* `ALT`: picks `n_landmarks` landmarks by a farthest-point heuristic (full
  `FastDijkstra` runs), stores forward and reverse landmark distance arrays
  (the graph is directed), and answers point-to-point queries with a
  numba-jitted A* kernel using the max-over-landmarks heuristic. The heuristic
  is admissible and consistent, so A* is exact.

* **Certified customization.** Landmark distances are valid *lower bounds* on
  the true distance only while edge costs never drop below the values used at
  landmark-preprocessing time. A cost *decrease* would break admissibility. We
  therefore preprocess landmarks on a *lower-bound cost array* (default 0.8x the
  current costs); any subsequent perturbation that keeps costs >= those lower
  bounds (i.e. within -20% of the originals) leaves the heuristic admissible
  *without recomputing a single landmark distance*. That is the certificate
  story: bounded cost changes are absorbed in the time it takes to write the CSR
  cost array, with no re-customization.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from certflow.fastgraph import njit

INF = float("inf")


# --------------------------------------------------------------------------- #
# RoadGraph: a FlatGraph-compatible CSR container built straight from arrays
# --------------------------------------------------------------------------- #


class RoadGraph:
    """CSR road graph with the attribute surface `FastDijkstra` reads.

    Built directly from flat arrays (no dict-of-dict path), so it scales to 1M+
    nodes. Nodes are integers 0..n-1 and the identity maps (`node_of`,
    `index_of`) are trivial — DIMACS node ids minus one. This is deliberately a
    *duck-typed* stand-in for `fastgraph.FlatGraph`: it provides `n`, the
    forward CSR (`indptr`, `indices`, `cost`), the reverse CSR (`r_indptr`,
    `r_indices`, `r_to_fwd`), and `node_of` — everything `FastDijkstra` touches —
    without importing or modifying `FlatGraph`.
    """

    __slots__ = (
        "n",
        "indptr",
        "indices",
        "cost",
        "r_indptr",
        "r_indices",
        "r_to_fwd",
        "index_of",
    )

    def __init__(
        self,
        n: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        cost: np.ndarray,
        r_indptr: np.ndarray,
        r_indices: np.ndarray,
        r_to_fwd: np.ndarray,
    ) -> None:
        self.n = int(n)
        self.indptr = indptr
        self.indices = indices
        self.cost = cost
        self.r_indptr = r_indptr
        self.r_indices = r_indices
        self.r_to_fwd = r_to_fwd
        # Identity node<->index map (DIMACS ids are 1..n, internal are 0..n-1).
        self.index_of = _IdentityMap(self.n)

    def node_of(self, i: int) -> int:
        return int(i)


class _IdentityMap:
    """Minimal mapping that returns the key for any 0 <= key < n.

    Lets `RoadGraph` satisfy code that does `flat.index_of[node]` without
    materializing an n-entry dict for a million nodes.
    """

    __slots__ = ("n",)

    def __init__(self, n: int) -> None:
        self.n = n

    def __getitem__(self, k: int) -> int:
        if 0 <= k < self.n:
            return int(k)
        raise KeyError(k)

    def __contains__(self, k: object) -> bool:
        return isinstance(k, (int, np.integer)) and 0 <= int(k) < self.n

    def get(self, k: int, default: int = -1) -> int:
        if 0 <= k < self.n:
            return int(k)
        return default


# --------------------------------------------------------------------------- #
# DIMACS .gr parser
# --------------------------------------------------------------------------- #


def load_dimacs(path: str | Path) -> RoadGraph:
    """Load a DIMACS 9th-challenge ``.gr`` graph into a `RoadGraph` (CSR).

    Format (1-indexed, directed): comment lines start with ``c``; the problem
    line ``p sp <n> <m>`` declares node and arc counts; each arc line is
    ``a <u> <v> <w>`` (tail, head, weight). We build forward and reverse CSR
    arrays directly — no intermediate dict adjacency — so this scales to the
    1.07M-node FLA graph without trouble.

    Parameters
    ----------
    path:
        Path to the (decompressed) ``.gr`` file.

    Returns
    -------
    RoadGraph
        Internal node ids are the DIMACS ids minus one (0..n-1).
    """
    path = Path(path)
    n = 0
    m = 0
    # First pass: read header to size arrays.
    with path.open("rb") as fh:
        for raw in fh:
            if raw[:1] == b"p":
                parts = raw.split()
                # p sp <n> <m>
                n = int(parts[2])
                m = int(parts[3])
                break
            if raw[:1] not in (b"c", b"\n", b""):
                # Some files put 'p' after comments only; keep scanning on 'c'.
                continue
    if n == 0:
        raise ValueError(f"No problem line ('p sp n m') found in {path}")

    # Read all arcs in a vectorized pass.
    tails = np.empty(m, dtype=np.int64)
    heads = np.empty(m, dtype=np.int64)
    weights = np.empty(m, dtype=np.float64)
    k = 0
    with path.open("rb") as fh:
        for raw in fh:
            if raw[:1] != b"a":
                continue
            parts = raw.split()
            tails[k] = int(parts[1]) - 1
            heads[k] = int(parts[2]) - 1
            weights[k] = float(parts[3])
            k += 1
    if k != m:
        # Some mirrors miscount; trim to actual.
        tails = tails[:k]
        heads = heads[:k]
        weights = weights[:k]
        m = k

    return _csr_from_arcs(n, tails, heads, weights)


def _csr_from_arcs(
    n: int,
    tails: np.ndarray,
    heads: np.ndarray,
    weights: np.ndarray,
) -> RoadGraph:
    """Build forward+reverse CSR from parallel (tail, head, weight) arc arrays.

    Forward edges are grouped by tail (stable sort), reverse by head, and
    ``r_to_fwd`` maps each reverse slot to its forward slot so the cost array is
    the single source of truth (matching `FlatGraph`'s reverse-CSR contract).
    """
    indptr = np.zeros(n + 1, dtype=np.int32)
    counts = np.bincount(tails, minlength=n)
    indptr[1:] = np.cumsum(counts).astype(np.int32)

    # Stable sort arcs by tail so each tail's edges are contiguous.
    order = np.argsort(tails, kind="stable")
    indices = heads[order].astype(np.int32)
    cost = weights[order].astype(np.float64)
    fwd_tail = tails[order]  # = sorted tails, parallel to indices/cost

    # Reverse CSR: group forward slots by head.
    r_indptr = np.zeros(n + 1, dtype=np.int32)
    r_counts = np.bincount(indices.astype(np.int64), minlength=n)
    r_indptr[1:] = np.cumsum(r_counts).astype(np.int32)
    r_order = np.argsort(indices, kind="stable")  # forward-slot order by head
    r_indices = fwd_tail[r_order].astype(np.int32)  # predecessor = the tail
    r_to_fwd = r_order.astype(np.int32)  # reverse slot -> forward slot

    return RoadGraph(n, indptr, indices, cost, r_indptr, r_indices, r_to_fwd)


# --------------------------------------------------------------------------- #
# Numba A* kernel (max-over-landmarks ALT heuristic)
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _alt_heuristic(node, target, lm_from, lm_to):
    """Admissible+consistent ALT lower bound on dist(node, target).

    For each landmark L, the triangle inequality on a directed graph gives two
    lower bounds:
        dist(node, target) >= dist(L, target) - dist(L, node)   [L as source]
        dist(node, target) >= dist(node, L)  - dist(target, L)  [L as sink]
    where ``lm_to[L, x] = dist(x -> L)`` (distance *to* the landmark, from the
    reverse search) and ``lm_from[L, x] = dist(L -> x)`` (distance *from* the
    landmark, forward search). We take the max over both forms and all
    landmarks. Unreachable (inf) terms are skipped.
    """
    n_lm = lm_from.shape[0]
    h = 0.0
    for li in range(n_lm):
        d_from_t = lm_to[li, target]   # dist(target -> L)
        d_from_n = lm_to[li, node]     # dist(node -> L)
        # form A: dist(node,target) >= dist(node->L) - dist(target->L)
        if d_from_n != np.inf and d_from_t != np.inf:
            b = d_from_n - d_from_t
            if b > h:
                h = b
        d_to_t = lm_from[li, target]   # dist(L -> target)
        d_to_n = lm_from[li, node]     # dist(L -> node)
        # form B: dist(node,target) >= dist(L->target) - dist(L->node)
        if d_to_t != np.inf and d_to_n != np.inf:
            b = d_to_t - d_to_n
            if b > h:
                h = b
    return h


@njit(cache=True)
def _astar_query(
    source,
    target,
    indptr,
    indices,
    cost,
    lm_from,
    lm_to,
):
    """A* shortest-path cost from source to target using the ALT heuristic.

    Binary min-heap keyed on f = g + h, with a parallel insertion counter for a
    deterministic tie-break (matching the Dijkstra tie-break semantics). Returns
    the path cost, or inf if the target is unreachable. The heuristic is
    admissible and consistent, so the first time ``target`` is popped its g-value
    is optimal — identical to plain Dijkstra's distance.
    """
    n = indptr.shape[0] - 1
    g = np.full(n, np.inf, dtype=np.float64)
    closed = np.zeros(n, dtype=np.bool_)

    # Flat binary heap of (f, tie, node).
    cap = 1024
    hf = np.empty(cap, dtype=np.float64)
    htie = np.empty(cap, dtype=np.int64)
    hnode = np.empty(cap, dtype=np.int64)
    hsize = 0
    counter = 0

    g[source] = 0.0
    h0 = _alt_heuristic(source, target, lm_from, lm_to)
    hf[0] = h0
    htie[0] = counter
    hnode[0] = source
    hsize = 1
    counter += 1

    while hsize > 0:
        # pop root
        u = hnode[0]
        hsize -= 1
        if hsize > 0:
            hf[0] = hf[hsize]
            htie[0] = htie[hsize]
            hnode[0] = hnode[hsize]
            # sift down
            i = 0
            while True:
                l = 2 * i + 1
                r = 2 * i + 2
                sm = i
                if l < hsize and (hf[l] < hf[sm] or (hf[l] == hf[sm] and htie[l] < htie[sm])):
                    sm = l
                if r < hsize and (hf[r] < hf[sm] or (hf[r] == hf[sm] and htie[r] < htie[sm])):
                    sm = r
                if sm == i:
                    break
                hf[i], hf[sm] = hf[sm], hf[i]
                htie[i], htie[sm] = htie[sm], htie[i]
                hnode[i], hnode[sm] = hnode[sm], hnode[i]
                i = sm

        if closed[u]:
            continue
        closed[u] = True
        if u == target:
            return g[u]
        du = g[u]
        for s in range(indptr[u], indptr[u + 1]):
            v = indices[s]
            if closed[v]:
                continue
            nd = du + cost[s]
            if nd < g[v]:
                g[v] = nd
                hv = _alt_heuristic(v, target, lm_from, lm_to)
                f = nd + hv
                # grow heap if needed
                if hsize >= hf.shape[0]:
                    newcap = 2 * hf.shape[0]
                    nhf = np.empty(newcap, dtype=np.float64)
                    ntie = np.empty(newcap, dtype=np.int64)
                    nnode = np.empty(newcap, dtype=np.int64)
                    for j in range(hsize):
                        nhf[j] = hf[j]
                        ntie[j] = htie[j]
                        nnode[j] = hnode[j]
                    hf = nhf
                    htie = ntie
                    hnode = nnode
                # push (f, counter, v)
                i = hsize
                hf[i] = f
                htie[i] = counter
                hnode[i] = v
                hsize += 1
                counter += 1
                while i > 0:
                    parent = (i - 1) // 2
                    if hf[i] < hf[parent] or (hf[i] == hf[parent] and htie[i] < htie[parent]):
                        hf[i], hf[parent] = hf[parent], hf[i]
                        htie[i], htie[parent] = htie[parent], htie[i]
                        hnode[i], hnode[parent] = hnode[parent], hnode[i]
                        i = parent
                    else:
                        break
    return np.inf


# --------------------------------------------------------------------------- #
# ALT accelerator
# --------------------------------------------------------------------------- #


class ALT:
    """A*, Landmarks, Triangle-inequality accelerator over a `RoadGraph`.

    Landmarks are picked greedily by a farthest-point heuristic; for each, two
    distance arrays are stored (forward: dist(L -> x); reverse: dist(x -> L)),
    both via full `FastDijkstra` runs on a chosen *lower-bound* cost array. The
    A* query uses the max-over-landmarks heuristic and is exact whenever the
    actual edge costs are >= the lower-bound costs used for preprocessing.

    Parameters
    ----------
    graph:
        A `RoadGraph` (or anything with the same CSR surface).
    n_landmarks:
        Number of landmarks (default 16).
    lower_bound_factor:
        Landmark distances are computed on ``factor * cost``. With the default
        0.8, any later perturbation that keeps each edge >= 0.8x its original
        (i.e. within -20%) leaves the heuristic admissible with no
        recomputation. Use 1.0 to preprocess on the current costs exactly (only
        valid if costs never decrease).
    seed:
        RNG seed for the first landmark pick.
    """

    def __init__(
        self,
        graph: RoadGraph,
        n_landmarks: int = 16,
        lower_bound_factor: float = 0.8,
        seed: int = 0,
    ) -> None:
        self.graph = graph
        self.n_landmarks = int(n_landmarks)
        self.lower_bound_factor = float(lower_bound_factor)
        self.seed = int(seed)
        self.landmarks: list[int] = []
        # lm_from[i] = dist(landmark_i -> x); lm_to[i] = dist(x -> landmark_i)
        self.lm_from = np.empty((0, graph.n), dtype=np.float64)
        self.lm_to = np.empty((0, graph.n), dtype=np.float64)
        self._build()

    # -- preprocessing ---------------------------------------------------------

    def _build(self) -> None:
        """Select landmarks (farthest-point) and fill the distance tables.

        Distances are computed on a lower-bound cost array so the heuristic
        survives bounded cost perturbations. To get full single-source distance
        arrays cheaply we temporarily swap the graph's cost array for the
        lower-bound one and run an *unbounded* Dijkstra (no early exit) per
        landmark, in both forward and reverse directions.
        """
        g = self.graph
        n = g.n
        lb_cost = g.cost * self.lower_bound_factor

        lm_from = np.empty((self.n_landmarks, n), dtype=np.float64)
        lm_to = np.empty((self.n_landmarks, n), dtype=np.float64)
        landmarks: list[int] = []

        rng = np.random.default_rng(self.seed)
        # First landmark: a random reachable node, then push to the farthest.
        cur = int(rng.integers(0, n))
        # Seed farthest-point with the node farthest from a random start.
        first_dist = _dijkstra_full(g.indptr, g.indices, lb_cost, cur)
        cur = int(_argmax_finite(first_dist))

        # min distance (forward) from the current landmark set to every node,
        # used by the farthest-point heuristic.
        min_dist = np.full(n, INF, dtype=np.float64)
        for li in range(self.n_landmarks):
            d_from = _dijkstra_full(g.indptr, g.indices, lb_cost, cur)
            d_to = _dijkstra_full(g.r_indptr, g.r_indices, lb_cost[g.r_to_fwd], cur)
            lm_from[li] = d_from
            lm_to[li] = d_to
            landmarks.append(cur)
            # update min over landmarks (use forward dist as the spread metric)
            np.minimum(min_dist, d_from, out=min_dist)
            if li + 1 < self.n_landmarks:
                # next landmark = node maximizing distance to current set
                masked = np.where(np.isfinite(min_dist), min_dist, -1.0)
                masked[landmarks] = -1.0
                cur = int(np.argmax(masked))

        self.landmarks = landmarks
        self.lm_from = lm_from
        self.lm_to = lm_to

    # -- query -----------------------------------------------------------------

    def query(self, source: int, target: int) -> float:
        """Exact shortest-path cost from source to target via ALT A*."""
        if source == target:
            return 0.0
        return float(
            _astar_query(
                int(source),
                int(target),
                self.graph.indptr,
                self.graph.indices,
                self.graph.cost,
                self.lm_from,
                self.lm_to,
            )
        )

    def warmup(self) -> None:
        """Trigger numba compilation of the query kernel (one tiny query)."""
        n = self.graph.n
        if n >= 2:
            self.query(0, 1)


# --------------------------------------------------------------------------- #
# Full single-source Dijkstra (numba) — used for landmark distance tables
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _dijkstra_full(indptr, indices, cost, source):
    """Full single-source Dijkstra over a CSR graph; returns the distance array.

    No target / early exit — computes distances to every reachable node. Used to
    fill the landmark distance tables. Binary heap with insertion-counter
    tie-break.
    """
    n = indptr.shape[0] - 1
    dist = np.full(n, np.inf, dtype=np.float64)
    visited = np.zeros(n, dtype=np.bool_)

    cap = 1024
    hd = np.empty(cap, dtype=np.float64)
    htie = np.empty(cap, dtype=np.int64)
    hnode = np.empty(cap, dtype=np.int64)
    hsize = 0
    counter = 0

    dist[source] = 0.0
    hd[0] = 0.0
    htie[0] = counter
    hnode[0] = source
    hsize = 1
    counter += 1

    while hsize > 0:
        d_top = hd[0]
        u = hnode[0]
        hsize -= 1
        if hsize > 0:
            hd[0] = hd[hsize]
            htie[0] = htie[hsize]
            hnode[0] = hnode[hsize]
            i = 0
            while True:
                l = 2 * i + 1
                r = 2 * i + 2
                sm = i
                if l < hsize and (hd[l] < hd[sm] or (hd[l] == hd[sm] and htie[l] < htie[sm])):
                    sm = l
                if r < hsize and (hd[r] < hd[sm] or (hd[r] == hd[sm] and htie[r] < htie[sm])):
                    sm = r
                if sm == i:
                    break
                hd[i], hd[sm] = hd[sm], hd[i]
                htie[i], htie[sm] = htie[sm], htie[i]
                hnode[i], hnode[sm] = hnode[sm], hnode[i]
                i = sm

        if visited[u]:
            continue
        visited[u] = True
        for s in range(indptr[u], indptr[u + 1]):
            v = indices[s]
            nd = d_top + cost[s]
            if nd < dist[v]:
                dist[v] = nd
                if hsize >= hd.shape[0]:
                    newcap = 2 * hd.shape[0]
                    nhd = np.empty(newcap, dtype=np.float64)
                    ntie = np.empty(newcap, dtype=np.int64)
                    nnode = np.empty(newcap, dtype=np.int64)
                    for j in range(hsize):
                        nhd[j] = hd[j]
                        ntie[j] = htie[j]
                        nnode[j] = hnode[j]
                    hd = nhd
                    htie = ntie
                    hnode = nnode
                i = hsize
                hd[i] = nd
                htie[i] = counter
                hnode[i] = v
                hsize += 1
                counter += 1
                while i > 0:
                    parent = (i - 1) // 2
                    if hd[i] < hd[parent] or (hd[i] == hd[parent] and htie[i] < htie[parent]):
                        hd[i], hd[parent] = hd[parent], hd[i]
                        htie[i], htie[parent] = htie[parent], htie[i]
                        hnode[i], hnode[parent] = hnode[parent], hnode[i]
                        i = parent
                    else:
                        break
    return dist


def _argmax_finite(arr: np.ndarray) -> int:
    """Index of the max finite entry (treats inf as -inf)."""
    masked = np.where(np.isfinite(arr), arr, -1.0)
    return int(np.argmax(masked))
