"""Certified Contraction Hierarchies for CERT: preprocessing-by-proof at road scale.

This module is the road-network analog of `snapshot.py`'s certificate-gated
all-pairs oracle. Where the snapshot oracle proves "the map is tight" and that
proof licenses an O(n^2) all-pairs build, a Contraction Hierarchy (CH) is the
structure that scales the same idea to hundreds of thousands of nodes: a one-time
preprocessing pass produces an overlay of *shortcut* edges so that an exact
point-to-point query becomes a tiny bidirectional search that only ever moves
"upward" in a node ordering. The published CH answers a continental query in
~110 us after a few minutes of preprocessing; our goal is to reproduce that query
class on the DIMACS road graphs from scratch, and to wire the BUILD behind the
same certificate gate the caller already drives for the snapshot oracle.

What lives here (and why none of it touches `fastgraph`/`roadnet`/`snapshot`):

* `ContractionHierarchy(road_graph_or_flat)`: classic CH. Node ordering by a
  lazily-updated priority (edge difference + contracted-neighbours, the standard
  simulation-based heuristic); contraction with *bounded* local-Dijkstra witness
  searches; an upward CSR adjacency where every edge points from a
  lower-rank to a higher-rank node, and shortcuts store their *middle* node so a
  path can be unpacked to original edges.

* Numba kernels (array heaps, in the style of `snapshot._sssp_kernel`):
  `_witness_search` (bounded local Dijkstra used during contraction) and
  `_bidir_upward` (the bidirectional upward Dijkstra that answers a query). A
  pure-Python fallback is present but numba is the point.

* `build(costs)` returns the build wall-time; `query(s, g)` returns the exact
  cost; `path(s, g)` unpacks shortcuts to the original-edge node sequence.

The certificate story (see the doc append): a CH built on *exact* costs is exact
only for those costs — any cost change INVALIDATES it and forces a rebuild
(the CH analog of CRP's metric customization). But a CH built on a *lower-bound*
cost array (default 0.8x, exactly the ALT trick) yields an admissible distance
oracle: `dist_lb(v, g)` from a CH built on lower bounds is a valid lower bound on
the true `dist(v, g)`, hence an admissible+consistent heuristic for a forward A*
on the *true* costs. That A* answers EXACT queries on any costs within +-20% of
the build costs with ZERO rebuild — the bounded-change certified variant.
"""
from __future__ import annotations

import heapq
import time
from typing import Optional

import numpy as np

from certflow.fastgraph import njit

INF = float("inf")


# --------------------------------------------------------------------------- #
# Numba kernels (array heaps, snapshot._sssp_kernel style)
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _witness_search(
    indptr,
    indices,
    cost,
    contracted,
    source,
    max_dist,
    max_settled,
    targets_mask,
    dist,
    touched,
):
    """Bounded local Dijkstra from `source` over the *not-yet-contracted* graph.

    Used during contraction to decide whether a shortcut is necessary: when
    contracting node ``x`` we ask, for each predecessor ``u`` and successor
    ``w``, whether there is a path ``u -> ... -> w`` that avoids ``x`` and is no
    longer than ``cost(u,x)+cost(x,w)``. If so, the shortcut is *witnessed* and
    can be skipped. We run one search per ``u``, stopping early once every
    relevant ``w`` is settled or the distance/settled budget is exhausted.

    `contracted[v]` marks nodes already removed from the residual graph (skipped).
    `targets_mask[v] != 0` marks the successors ``w`` we still need to settle;
    the search exits as soon as the outstanding-target count hits zero. `dist`
    is a scratch float64 array (size n) that the caller resets via `touched`:
    every node whose `dist` we wrote is appended to `touched` so the caller can
    reset exactly those entries in O(touched) instead of O(n). Returns the number
    of touched entries.
    """
    # array binary heap of (key, node)
    cap = 1024
    hk = np.empty(cap, dtype=np.float64)
    hn = np.empty(cap, dtype=np.int64)
    hsize = 0

    n_touched = 0
    dist[source] = 0.0
    touched[n_touched] = source
    n_touched += 1
    hk[0] = 0.0
    hn[0] = source
    hsize = 1
    settled = 0

    # outstanding targets we still need to settle (for early exit)
    remaining = 0
    for i in range(indptr.shape[0] - 1):
        if targets_mask[i] != 0:
            remaining += 1

    while hsize > 0:
        kd = hk[0]
        u = hn[0]
        hsize -= 1
        if hsize > 0:
            hk[0] = hk[hsize]
            hn[0] = hn[hsize]
            i = 0
            while True:
                l = 2 * i + 1
                r = 2 * i + 2
                m = i
                if l < hsize and hk[l] < hk[m]:
                    m = l
                if r < hsize and hk[r] < hk[m]:
                    m = r
                if m == i:
                    break
                hk[i], hk[m] = hk[m], hk[i]
                hn[i], hn[m] = hn[m], hn[i]
                i = m

        if kd > dist[u]:
            continue
        # stop conditions: distance budget exceeded -> nothing useful left
        if kd > max_dist:
            break
        settled += 1
        if targets_mask[u] != 0:
            remaining -= 1
            if remaining <= 0:
                break
        if settled > max_settled:
            break

        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if contracted[v]:
                continue
            nd = kd + cost[k]
            if nd > max_dist:
                continue
            if nd < dist[v]:
                if dist[v] == np.inf:
                    touched[n_touched] = v
                    n_touched += 1
                dist[v] = nd
                # push
                if hsize >= hk.shape[0]:
                    newcap = 2 * hk.shape[0]
                    nhk = np.empty(newcap, dtype=np.float64)
                    nhn = np.empty(newcap, dtype=np.int64)
                    for j in range(hsize):
                        nhk[j] = hk[j]
                        nhn[j] = hn[j]
                    hk = nhk
                    hn = nhn
                j = hsize
                hk[j] = nd
                hn[j] = v
                hsize += 1
                while j > 0:
                    p = (j - 1) // 2
                    if hk[p] <= hk[j]:
                        break
                    hk[p], hk[j] = hk[j], hk[p]
                    hn[p], hn[j] = hn[j], hn[p]
                    j = p
    return n_touched


@njit(cache=True)
def _bidir_upward(
    source,
    target,
    up_indptr,
    up_indices,
    up_cost,
    dn_indptr,
    dn_indices,
    dn_cost,
    df,
    db,
    seenf,
    seenb,
):
    """Bidirectional upward Dijkstra answering an exact point-to-point query.

    Forward search uses the *upward* graph (edges low-rank -> high-rank) from
    `source`; backward search uses the *downward* graph (the reverse upward
    edges, i.e. high-rank -> low-rank stored as a forward CSR over heads) from
    `target`. Both only ever relax edges going to strictly higher rank, so each
    is a small DAG search. The meeting point with the minimum `df[v] + db[v]`
    over settled nodes is the answer.

    `df`/`db` are scratch distance arrays (caller-allocated, reset to inf
    between queries via the touched lists `seenf`/`seenb`, whose lengths are
    returned). Standard array binary heaps, snapshot kernel style.
    """
    best = np.inf

    # two heaps
    capf = 256
    hkf = np.empty(capf, dtype=np.float64)
    hnf = np.empty(capf, dtype=np.int64)
    hsf = 0
    hkb = np.empty(capf, dtype=np.float64)
    hnb = np.empty(capf, dtype=np.int64)
    hsb = 0

    nf = 0
    nb = 0

    df[source] = 0.0
    seenf[nf] = source
    nf += 1
    hkf[0] = 0.0
    hnf[0] = source
    hsf = 1

    db[target] = 0.0
    seenb[nb] = target
    nb += 1
    hkb[0] = 0.0
    hnb[0] = target
    hsb = 1

    # Alternate the two searches. Each can stop when its min key exceeds best.
    while hsf > 0 or hsb > 0:
        # ---- forward step ----
        if hsf > 0:
            kd = hkf[0]
            u = hnf[0]
            if kd > best:
                hsf = 0  # prune: no forward node can improve best
            else:
                hsf -= 1
                if hsf > 0:
                    hkf[0] = hkf[hsf]
                    hnf[0] = hnf[hsf]
                    i = 0
                    while True:
                        l = 2 * i + 1
                        r = 2 * i + 2
                        m = i
                        if l < hsf and hkf[l] < hkf[m]:
                            m = l
                        if r < hsf and hkf[r] < hkf[m]:
                            m = r
                        if m == i:
                            break
                        hkf[i], hkf[m] = hkf[m], hkf[i]
                        hnf[i], hnf[m] = hnf[m], hnf[i]
                        i = m
                if kd <= df[u]:
                    if db[u] != np.inf:
                        tot = df[u] + db[u]
                        if tot < best:
                            best = tot
                    for k in range(up_indptr[u], up_indptr[u + 1]):
                        v = up_indices[k]
                        nd = kd + up_cost[k]
                        if nd < df[v]:
                            if df[v] == np.inf:
                                seenf[nf] = v
                                nf += 1
                            df[v] = nd
                            if hsf >= hkf.shape[0]:
                                newcap = 2 * hkf.shape[0]
                                a = np.empty(newcap, dtype=np.float64)
                                b = np.empty(newcap, dtype=np.int64)
                                for j in range(hsf):
                                    a[j] = hkf[j]
                                    b[j] = hnf[j]
                                hkf = a
                                hnf = b
                            j = hsf
                            hkf[j] = nd
                            hnf[j] = v
                            hsf += 1
                            while j > 0:
                                p = (j - 1) // 2
                                if hkf[p] <= hkf[j]:
                                    break
                                hkf[p], hkf[j] = hkf[j], hkf[p]
                                hnf[p], hnf[j] = hnf[j], hnf[p]
                                j = p
        # ---- backward step ----
        if hsb > 0:
            kd = hkb[0]
            u = hnb[0]
            if kd > best:
                hsb = 0
            else:
                hsb -= 1
                if hsb > 0:
                    hkb[0] = hkb[hsb]
                    hnb[0] = hnb[hsb]
                    i = 0
                    while True:
                        l = 2 * i + 1
                        r = 2 * i + 2
                        m = i
                        if l < hsb and hkb[l] < hkb[m]:
                            m = l
                        if r < hsb and hkb[r] < hkb[m]:
                            m = r
                        if m == i:
                            break
                        hkb[i], hkb[m] = hkb[m], hkb[i]
                        hnb[i], hnb[m] = hnb[m], hnb[i]
                        i = m
                if kd <= db[u]:
                    if df[u] != np.inf:
                        tot = df[u] + db[u]
                        if tot < best:
                            best = tot
                    for k in range(dn_indptr[u], dn_indptr[u + 1]):
                        v = dn_indices[k]
                        nd = kd + dn_cost[k]
                        if nd < db[v]:
                            if db[v] == np.inf:
                                seenb[nb] = v
                                nb += 1
                            db[v] = nd
                            if hsb >= hkb.shape[0]:
                                newcap = 2 * hkb.shape[0]
                                a = np.empty(newcap, dtype=np.float64)
                                b = np.empty(newcap, dtype=np.int64)
                                for j in range(hsb):
                                    a[j] = hkb[j]
                                    b[j] = hnb[j]
                                hkb = a
                                hnb = b
                            j = hsb
                            hkb[j] = nd
                            hnb[j] = v
                            hsb += 1
                            while j > 0:
                                p = (j - 1) // 2
                                if hkb[p] <= hkb[j]:
                                    break
                                hkb[p], hkb[j] = hkb[j], hkb[p]
                                hnb[p], hnb[j] = hnb[j], hnb[p]
                                j = p
    return best, nf, nb


@njit(cache=True)
def _bidir_upward_meet(
    source,
    target,
    up_indptr,
    up_indices,
    up_cost,
    dn_indptr,
    dn_indices,
    dn_cost,
    df,
    db,
    pf,
    pb,
    seenf,
    seenb,
):
    """Like `_bidir_upward` but also records parents and the meeting node.

    Returns ``(best, meet, nf, nb)``. `pf[v]`/`pb[v]` hold the predecessor
    (in upward / downward CSR node ids) used to reach `v` in the forward /
    backward search, so the caller can stitch the two halves and then unpack
    shortcuts. `meet` is the node minimizing ``df+db``; -1 if unreachable.
    """
    best = np.inf
    meet = -1

    capf = 256
    hkf = np.empty(capf, dtype=np.float64)
    hnf = np.empty(capf, dtype=np.int64)
    hsf = 0
    hkb = np.empty(capf, dtype=np.float64)
    hnb = np.empty(capf, dtype=np.int64)
    hsb = 0

    nf = 0
    nb = 0

    df[source] = 0.0
    pf[source] = -1
    seenf[nf] = source
    nf += 1
    hkf[0] = 0.0
    hnf[0] = source
    hsf = 1

    db[target] = 0.0
    pb[target] = -1
    seenb[nb] = target
    nb += 1
    hkb[0] = 0.0
    hnb[0] = target
    hsb = 1

    while hsf > 0 or hsb > 0:
        if hsf > 0:
            kd = hkf[0]
            u = hnf[0]
            if kd > best:
                hsf = 0
            else:
                hsf -= 1
                if hsf > 0:
                    hkf[0] = hkf[hsf]
                    hnf[0] = hnf[hsf]
                    i = 0
                    while True:
                        l = 2 * i + 1
                        r = 2 * i + 2
                        m = i
                        if l < hsf and hkf[l] < hkf[m]:
                            m = l
                        if r < hsf and hkf[r] < hkf[m]:
                            m = r
                        if m == i:
                            break
                        hkf[i], hkf[m] = hkf[m], hkf[i]
                        hnf[i], hnf[m] = hnf[m], hnf[i]
                        i = m
                if kd <= df[u]:
                    if db[u] != np.inf:
                        tot = df[u] + db[u]
                        if tot < best:
                            best = tot
                            meet = u
                    for k in range(up_indptr[u], up_indptr[u + 1]):
                        v = up_indices[k]
                        nd = kd + up_cost[k]
                        if nd < df[v]:
                            if df[v] == np.inf:
                                seenf[nf] = v
                                nf += 1
                            df[v] = nd
                            pf[v] = u
                            if hsf >= hkf.shape[0]:
                                newcap = 2 * hkf.shape[0]
                                a = np.empty(newcap, dtype=np.float64)
                                b = np.empty(newcap, dtype=np.int64)
                                for j in range(hsf):
                                    a[j] = hkf[j]
                                    b[j] = hnf[j]
                                hkf = a
                                hnf = b
                            j = hsf
                            hkf[j] = nd
                            hnf[j] = v
                            hsf += 1
                            while j > 0:
                                p = (j - 1) // 2
                                if hkf[p] <= hkf[j]:
                                    break
                                hkf[p], hkf[j] = hkf[j], hkf[p]
                                hnf[p], hnf[j] = hnf[j], hnf[p]
                                j = p
        if hsb > 0:
            kd = hkb[0]
            u = hnb[0]
            if kd > best:
                hsb = 0
            else:
                hsb -= 1
                if hsb > 0:
                    hkb[0] = hkb[hsb]
                    hnb[0] = hnb[hsb]
                    i = 0
                    while True:
                        l = 2 * i + 1
                        r = 2 * i + 2
                        m = i
                        if l < hsb and hkb[l] < hkb[m]:
                            m = l
                        if r < hsb and hkb[r] < hkb[m]:
                            m = r
                        if m == i:
                            break
                        hkb[i], hkb[m] = hkb[m], hkb[i]
                        hnb[i], hnb[m] = hnb[m], hnb[i]
                        i = m
                if kd <= db[u]:
                    if df[u] != np.inf:
                        tot = df[u] + db[u]
                        if tot < best:
                            best = tot
                            meet = u
                    for k in range(dn_indptr[u], dn_indptr[u + 1]):
                        v = dn_indices[k]
                        nd = kd + dn_cost[k]
                        if nd < db[v]:
                            if db[v] == np.inf:
                                seenb[nb] = v
                                nb += 1
                            db[v] = nd
                            pb[v] = u
                            if hsb >= hkb.shape[0]:
                                newcap = 2 * hkb.shape[0]
                                a = np.empty(newcap, dtype=np.float64)
                                b = np.empty(newcap, dtype=np.int64)
                                for j in range(hsb):
                                    a[j] = hkb[j]
                                    b[j] = hnb[j]
                                hkb = a
                                hnb = b
                            j = hsb
                            hkb[j] = nd
                            hnb[j] = v
                            hsb += 1
                            while j > 0:
                                p = (j - 1) // 2
                                if hkb[p] <= hkb[j]:
                                    break
                                hkb[p], hkb[j] = hkb[j], hkb[p]
                                hnb[p], hnb[j] = hnb[j], hnb[p]
                                j = p
    return best, meet, nf, nb


@njit(cache=True)
def _ch_backward_label(
    target,
    up_indptr,
    up_indices,
    up_cost,
    dn_indptr,
    dn_indices,
    dn_cost,
    rank_desc,
    h,
    seen,
):
    """Exact lower-bound distance array h[v] = dist_lb(v, target) for ALL v.

    CH one-to-all backward distance from `target`, in two phases:

      1. *Upward* search from `target` over the up-graph reversed: we relax, from
         each settled node, the down-edges INTO it... but the efficient CH way is
         the standard "backward upward search then downward sweep". Concretely we
         run a backward Dijkstra from `target` over the reverse hierarchy by
         relaxing up-edges in reverse: a settled node u with label h[u] relaxes
         every up-edge (u -> w) to give h[u] candidate from w side. To keep it a
         single forward-style pass we instead:
         (a) settle the target's upward search space exactly with a small
             Dijkstra over the up-graph in REVERSE (predecessors), then
         (b) sweep all nodes in DESCENDING rank order, relaxing each node's
             up-edges (v -> w, rank[w] > rank[v]) as h[v] = min(h[v],
             cost(v,w) + h[w]); because w has strictly higher rank it is already
             finalized when v is processed, so one linear sweep finalizes every
             node. This is the textbook CH one-to-all and yields the EXACT
             lower-bound distance to target for every node.

    `rank_desc` lists node ids by descending rank. `h` is an n-array
    (inf-initialised) written in place; `seen` collects touched indices for the
    caller's reset. Returns the touched count.
    """
    ns = 0
    # phase (a): backward Dijkstra from target over the down-graph gives exact
    # labels for target's own hierarchy (the meeting region). We then (b) sweep.
    cap = 1024
    hk = np.empty(cap, dtype=np.float64)
    hn = np.empty(cap, dtype=np.int64)
    hsize = 0
    h[target] = 0.0
    seen[ns] = target
    ns += 1
    hk[0] = 0.0
    hn[0] = target
    hsize = 1
    while hsize > 0:
        kd = hk[0]
        u = hn[0]
        hsize -= 1
        if hsize > 0:
            hk[0] = hk[hsize]
            hn[0] = hn[hsize]
            i = 0
            while True:
                l = 2 * i + 1
                r = 2 * i + 2
                m = i
                if l < hsize and hk[l] < hk[m]:
                    m = l
                if r < hsize and hk[r] < hk[m]:
                    m = r
                if m == i:
                    break
                hk[i], hk[m] = hk[m], hk[i]
                hn[i], hn[m] = hn[m], hn[i]
                i = m
        if kd > h[u]:
            continue
        for k in range(dn_indptr[u], dn_indptr[u + 1]):
            w = dn_indices[k]
            nd = kd + dn_cost[k]
            if nd < h[w]:
                if h[w] == np.inf:
                    seen[ns] = w
                    ns += 1
                h[w] = nd
                if hsize >= hk.shape[0]:
                    nc = 2 * hk.shape[0]
                    a = np.empty(nc, dtype=np.float64)
                    b = np.empty(nc, dtype=np.int64)
                    for j in range(hsize):
                        a[j] = hk[j]
                        b[j] = hn[j]
                    hk = a
                    hn = b
                j = hsize
                hk[j] = nd
                hn[j] = w
                hsize += 1
                while j > 0:
                    pp = (j - 1) // 2
                    if hk[pp] <= hk[j]:
                        break
                    hk[pp], hk[j] = hk[j], hk[pp]
                    hn[pp], hn[j] = hn[j], hn[pp]
                    j = pp
    # phase (b): descending-rank sweep over up-edges to finalize every node.
    # For v processed in descending rank, every up-edge (v -> w) has rank[w] >
    # rank[v], so w was processed earlier (higher rank) and h[w] is final.
    for idx in range(rank_desc.shape[0]):
        v = rank_desc[idx]
        hv = h[v]
        for k in range(up_indptr[v], up_indptr[v + 1]):
            w = up_indices[k]
            cw = h[w]
            if cw != np.inf:
                cand = up_cost[k] + cw
                if cand < hv:
                    hv = cand
        if hv != h[v]:
            if h[v] == np.inf:
                seen[ns] = v
                ns += 1
            h[v] = hv
    return ns


@njit(cache=True)
def _ch_potential_astar(
    source,
    target,
    indptr,
    indices,
    cost,
    up_indptr,
    up_indices,
    up_cost,
    dn_indptr,
    dn_indices,
    dn_cost,
    rank_desc,
    h,
    seen_h,
    g,
    closed,
    seen_g,
):
    """Forward A* on true `cost` with an EXACT CH lower-bound potential array.

    Step 1 (`_ch_backward_label`) builds h[v] = dist_lb(v, target) for all v from
    the CH built on lower-bound costs -- a one-to-all backward CH distance, hence
    admissible+consistent for any true costs >= those lower bounds. Step 2 is a
    forward A* on the *true* `cost` using that h. Exact: first pop of `target` is
    optimal. All scratch (`h`, `g`, `closed`) is caller-allocated and reset via
    touched-lists so the query stays O(search space + sweep), not O(n) of resets.
    """
    nh = _ch_backward_label(
        target, up_indptr, up_indices, up_cost,
        dn_indptr, dn_indices, dn_cost, rank_desc, h, seen_h,
    )

    result = np.inf
    ng = 0
    if h[source] != np.inf:
        cap2 = 1024
        fk = np.empty(cap2, dtype=np.float64)
        fn = np.empty(cap2, dtype=np.int64)
        fsize = 0
        g[source] = 0.0
        seen_g[ng] = source
        ng += 1
        fk[0] = h[source]
        fn[0] = source
        fsize = 1
        while fsize > 0:
            u = fn[0]
            fsize -= 1
            if fsize > 0:
                fk[0] = fk[fsize]
                fn[0] = fn[fsize]
                i = 0
                while True:
                    l = 2 * i + 1
                    r = 2 * i + 2
                    m = i
                    if l < fsize and fk[l] < fk[m]:
                        m = l
                    if r < fsize and fk[r] < fk[m]:
                        m = r
                    if m == i:
                        break
                    fk[i], fk[m] = fk[m], fk[i]
                    fn[i], fn[m] = fn[m], fn[i]
                    i = m
            if closed[u]:
                continue
            closed[u] = True
            if u == target:
                result = g[u]
                break
            du = g[u]
            for k in range(indptr[u], indptr[u + 1]):
                v = indices[k]
                if closed[v]:
                    continue
                nd = du + cost[k]
                if nd < g[v]:
                    if g[v] == np.inf:
                        seen_g[ng] = v
                        ng += 1
                    g[v] = nd
                    hv = h[v]
                    if hv == np.inf:
                        continue
                    f = nd + hv
                    if fsize >= fk.shape[0]:
                        nc = 2 * fk.shape[0]
                        a = np.empty(nc, dtype=np.float64)
                        b = np.empty(nc, dtype=np.int64)
                        for j in range(fsize):
                            a[j] = fk[j]
                            b[j] = fn[j]
                        fk = a
                        fn = b
                    j = fsize
                    fk[j] = f
                    fn[j] = v
                    fsize += 1
                    while j > 0:
                        pp = (j - 1) // 2
                        if fk[pp] <= fk[j]:
                            break
                        fk[pp], fk[j] = fk[j], fk[pp]
                        fn[pp], fn[j] = fn[j], fn[pp]
                        j = pp

    for i in range(nh):
        h[seen_h[i]] = np.inf
    for i in range(ng):
        node = seen_g[i]
        g[node] = np.inf
        closed[node] = False
    return result


# --------------------------------------------------------------------------- #
# ContractionHierarchy
# --------------------------------------------------------------------------- #


class ContractionHierarchy:
    """Classic Contraction Hierarchy over a `RoadGraph`/`FlatGraph` CSR surface.

    Build is a node-ordering + contraction pass; the result is an *upward* CSR
    (each edge from a lower-rank to a higher-rank node) plus its reverse
    (*downward*). Shortcuts store their middle node so a path can be unpacked to
    original edges. Queries are bidirectional upward Dijkstra; they are exact for
    the cost array the CH was built on.

    Parameters
    ----------
    graph:
        Anything exposing the CSR surface ``n, indptr, indices, cost`` (forward).
        We only read the forward adjacency; the hierarchy is symmetric in the
        sense that we build both an up- and a down-CSR from the shortcut overlay.
    edge_quotient_limit:
        Hop limit for witness searches (max settled nodes). Smaller = faster
        build, possibly more (unnecessary) shortcuts; correctness is unaffected.
    """

    def __init__(
        self,
        graph,
        max_settled: int = 1000,
    ) -> None:
        self.graph = graph
        self.n = int(graph.n)
        self.max_settled = int(max_settled)
        # filled by build()
        self.order: Optional[np.ndarray] = None       # rank[node] = position
        self.up_indptr: Optional[np.ndarray] = None
        self.up_indices: Optional[np.ndarray] = None
        self.up_cost: Optional[np.ndarray] = None
        self.up_mid: Optional[np.ndarray] = None       # middle node or -1
        self.dn_indptr: Optional[np.ndarray] = None
        self.dn_indices: Optional[np.ndarray] = None
        self.dn_cost: Optional[np.ndarray] = None
        self.dn_mid: Optional[np.ndarray] = None
        self.build_seconds: Optional[float] = None
        self.n_shortcuts: int = 0
        # scratch for queries (allocated lazily in build)
        self._df = None
        self._db = None
        self._pf = None
        self._pb = None
        self._seenf = None
        self._seenb = None

    # -- build ---------------------------------------------------------------

    def build(self, costs: Optional[np.ndarray] = None) -> float:
        """Build the hierarchy on `costs` (defaults to the graph's cost array).

        Returns the build wall-time in seconds. Uses an adjacency-list residual
        graph and a lazy priority queue for the node ordering; contraction calls
        the numba witness-search kernel to avoid inserting unnecessary shortcuts.
        """
        t0 = time.perf_counter()
        g = self.graph
        n = self.n
        if costs is None:
            costs = np.asarray(g.cost, dtype=np.float64)
        else:
            costs = np.asarray(costs, dtype=np.float64)

        # --- residual graph as dict-of-dict (out and in), keeping min parallel edge ---
        out_adj: list[dict] = [dict() for _ in range(n)]
        in_adj: list[dict] = [dict() for _ in range(n)]
        indptr = np.asarray(g.indptr)
        indices = np.asarray(g.indices)
        for u in range(n):
            for k in range(int(indptr[u]), int(indptr[u + 1])):
                v = int(indices[k])
                c = float(costs[k])
                if u == v:
                    continue
                if v not in out_adj[u] or c < out_adj[u][v][0]:
                    out_adj[u][v] = (c, -1)  # (cost, middle)
                    in_adj[v][u] = (c, -1)

        contracted = np.zeros(n, dtype=np.bool_)
        rank = np.full(n, -1, dtype=np.int64)
        contracted_neighbors = np.zeros(n, dtype=np.int64)
        # Every shortcut we add, as (u, w, cost, middle), for re-assembly.
        self._shortcut_list: list[tuple[int, int, float, int]] = []

        # scratch arrays for the witness kernel (residual graph rebuilt per call
        # is too slow; we run witness search over a *static snapshot* CSR of the
        # residual at simulation time is also slow — instead we do witness search
        # directly on the python adjacency for small local searches).
        # For speed at road scale we use a numba kernel over a residual CSR that
        # we rebuild lazily only when contracting (local, bounded). To keep this
        # tractable we run the local Dijkstra in pure python over the dicts: the
        # search is bounded (max_settled) so it stays small.

        import heapq as _hq

        def _edge_diff(x: int) -> tuple[int, int]:
            """(edge difference, n_shortcuts) for contracting x right now."""
            preds = [u for u in in_adj[x] if not contracted[u]]
            succs = [w for w in out_adj[x] if not contracted[w]]
            removed = len(preds) + len(succs)
            added = 0
            for u in preds:
                cux = in_adj[x][u][0]
                # witness search from u, bounded, over residual minus x
                # max needed distance = cux + max over succ of cost(x,w)
                # we settle succs; if dist(u,w) <= cux+cxw, shortcut not needed
                max_cxw = 0.0
                for w in succs:
                    if w == u:
                        continue
                    cxw = out_adj[x][w][0]
                    if cux + cxw > max_cxw:
                        max_cxw = cux + cxw
                dists = _local_dijkstra(
                    u, x, out_adj, contracted, max_cxw, self.max_settled, succs
                )
                for w in succs:
                    if w == u:
                        continue
                    cxw = out_adj[x][w][0]
                    via = cux + cxw
                    dw = dists.get(w, INF)
                    if dw > via + 1e-9:
                        added += 1
            return added - removed, added

        # initial priority queue (lazy): (priority, x)
        pq: list[tuple[float, int]] = []
        for x in range(n):
            ed, _ = _edge_diff(x)
            prio = ed  # + contracted_neighbors[x] (0 initially)
            _hq.heappush(pq, (prio, x))

        shortcuts = 0
        next_rank = 0
        while pq:
            prio, x = _hq.heappop(pq)
            if contracted[x]:
                continue
            # lazy update: recompute priority; if worse than current top, requeue
            ed, _ = _edge_diff(x)
            new_prio = ed + contracted_neighbors[x]
            if pq and new_prio > pq[0][0]:
                _hq.heappush(pq, (new_prio, x))
                continue

            # contract x: add witnessed shortcuts among its non-contracted nbrs
            preds = [u for u in in_adj[x] if not contracted[u]]
            succs = [w for w in out_adj[x] if not contracted[w]]
            for u in preds:
                cux = in_adj[x][u][0]
                max_cxw = 0.0
                for w in succs:
                    if w == u:
                        continue
                    cxw = out_adj[x][w][0]
                    if cux + cxw > max_cxw:
                        max_cxw = cux + cxw
                dists = _local_dijkstra(
                    u, x, out_adj, contracted, max_cxw, self.max_settled, succs
                )
                for w in succs:
                    if w == u:
                        continue
                    cxw = out_adj[x][w][0]
                    via = cux + cxw
                    dw = dists.get(w, INF)
                    if dw > via + 1e-9:
                        # add/relax shortcut u -> w with middle x
                        cur = out_adj[u].get(w)
                        if cur is None or via < cur[0]:
                            out_adj[u][w] = (via, x)
                            in_adj[w][u] = (via, x)
                            # record for re-assembly (min-cost version wins there)
                            self._shortcut_list.append((u, w, via, x))
                            if cur is None:
                                shortcuts += 1
            # finalize contraction
            rank[x] = next_rank
            next_rank += 1
            contracted[x] = True
            for u in preds:
                contracted_neighbors[u] += 1
                del out_adj[u][x]
            for w in succs:
                contracted_neighbors[w] += 1
                del in_adj[w][x]
            # the edges incident to x in the *other* direction (loops handled above)
            # clear x's own adjacency
            out_adj[x] = {}
            in_adj[x] = {}

        self.n_shortcuts = shortcuts
        self.order = rank

        # --- assemble upward CSR from the FULL edge set (originals + shortcuts) ---
        # We must re-derive every edge that survived; rebuild from a fresh pass:
        # collect all (u, w, cost, mid) where rank[u] < rank[w] as "up", and the
        # reverse as "down". To recover the full edge set including originals we
        # re-run over the original graph PLUS the shortcuts we recorded. But we
        # deleted residual edges as we contracted. So we instead recorded shortcuts
        # implicitly; rebuild by replaying original edges + collected shortcuts.
        self._assemble(costs, rank)

        self.build_seconds = time.perf_counter() - t0
        self._alloc_scratch()
        return self.build_seconds

    def _assemble(self, costs: np.ndarray, rank: np.ndarray) -> None:
        """Build up/down CSR from original edges + recorded shortcuts.

        Shortcuts were recorded during contraction in `self._shortcut_list`.
        Every edge (u, w) with rank[u] < rank[w] becomes an UP edge; the reverse
        orientation becomes a DOWN edge. Parallel edges keep the minimum cost.
        """
        g = self.graph
        n = self.n
        indptr = np.asarray(g.indptr)
        indices = np.asarray(g.indices)

        up: list[dict] = [dict() for _ in range(n)]   # u -> {w: (cost, mid)}
        dn: list[dict] = [dict() for _ in range(n)]   # w -> {u: (cost, mid)}

        def add_edge(u: int, w: int, c: float, mid: int) -> None:
            if u == w:
                return
            if rank[u] < rank[w]:
                d = up[u]
                if w not in d or c < d[w][0]:
                    d[w] = (c, mid)
            else:
                d = dn[w]
                if u not in d or c < d[u][0]:
                    d[u] = (c, mid)

        # original edges
        for u in range(n):
            for k in range(int(indptr[u]), int(indptr[u + 1])):
                w = int(indices[k])
                add_edge(u, w, float(costs[k]), -1)
        # shortcuts
        for (u, w, c, mid) in self._shortcut_list:
            add_edge(u, w, c, mid)

        # flatten up: edges low->high stored at the LOW node
        up_indptr = np.zeros(n + 1, dtype=np.int64)
        for u in range(n):
            up_indptr[u + 1] = up_indptr[u] + len(up[u])
        m_up = int(up_indptr[n])
        up_indices = np.empty(m_up, dtype=np.int64)
        up_cost = np.empty(m_up, dtype=np.float64)
        up_mid = np.empty(m_up, dtype=np.int64)
        idx = 0
        for u in range(n):
            for w, (c, mid) in up[u].items():
                up_indices[idx] = w
                up_cost[idx] = c
                up_mid[idx] = mid
                idx += 1

        # down: edges stored at the HIGH node (w), pointing to lower u; the
        # backward search starts at target and walks DOWN in rank.
        dn_indptr = np.zeros(n + 1, dtype=np.int64)
        for w in range(n):
            dn_indptr[w + 1] = dn_indptr[w] + len(dn[w])
        m_dn = int(dn_indptr[n])
        dn_indices = np.empty(m_dn, dtype=np.int64)
        dn_cost = np.empty(m_dn, dtype=np.float64)
        dn_mid = np.empty(m_dn, dtype=np.int64)
        idx = 0
        for w in range(n):
            for u, (c, mid) in dn[w].items():
                dn_indices[idx] = u
                dn_cost[idx] = c
                dn_mid[idx] = mid
                idx += 1

        self.up_indptr = up_indptr
        self.up_indices = up_indices
        self.up_cost = up_cost
        self.up_mid = up_mid
        self.dn_indptr = dn_indptr
        self.dn_indices = dn_indices
        self.dn_cost = dn_cost
        self.dn_mid = dn_mid

    def _alloc_scratch(self) -> None:
        n = self.n
        self._df = np.full(n, INF, dtype=np.float64)
        self._db = np.full(n, INF, dtype=np.float64)
        self._pf = np.full(n, -1, dtype=np.int64)
        self._pb = np.full(n, -1, dtype=np.int64)
        self._seenf = np.empty(n, dtype=np.int64)
        self._seenb = np.empty(n, dtype=np.int64)

    # -- query ---------------------------------------------------------------

    def query(self, s_idx: int, g_idx: int) -> float:
        """Exact shortest-path cost from `s_idx` to `g_idx` (cost-only)."""
        if self.up_indptr is None:
            raise RuntimeError("ContractionHierarchy.query before build(); call build() first")
        if s_idx == g_idx:
            return 0.0
        best, nf, nb = _bidir_upward(
            int(s_idx), int(g_idx),
            self.up_indptr, self.up_indices, self.up_cost,
            self.dn_indptr, self.dn_indices, self.dn_cost,
            self._df, self._db, self._seenf, self._seenb,
        )
        # reset touched scratch
        for i in range(nf):
            self._df[self._seenf[i]] = INF
        for i in range(nb):
            self._db[self._seenb[i]] = INF
        return float(best)

    def path(self, s_idx: int, g_idx: int) -> Optional[list[int]]:
        """Exact path (original node sequence) by unpacking shortcuts."""
        if self.up_indptr is None:
            raise RuntimeError("ContractionHierarchy.path before build(); call build() first")
        if s_idx == g_idx:
            return [int(s_idx)]
        best, meet, nf, nb = _bidir_upward_meet(
            int(s_idx), int(g_idx),
            self.up_indptr, self.up_indices, self.up_cost,
            self.dn_indptr, self.dn_indices, self.dn_cost,
            self._df, self._db, self._pf, self._pb, self._seenf, self._seenb,
        )
        if meet < 0 or not np.isfinite(best):
            self._reset(nf, nb)
            return None
        # forward half: source ... meet (each hop is an UP edge u->v)
        fwd_nodes = []
        v = int(meet)
        while v != -1:
            fwd_nodes.append(v)
            v = int(self._pf[v])
        fwd_nodes.reverse()  # source ... meet
        # backward half: meet ... target (each hop is a DOWN edge stored at high)
        bwd_nodes = []
        v = int(self._pb[int(meet)])
        while v != -1:
            bwd_nodes.append(v)
            v = int(self._pb[v])
        # bwd_nodes is meet's-next ... target

        # Build the contracted-edge path: list of (u, v) hops, each either an
        # up-edge or a down-edge; unpack every shortcut to original edges.
        contracted_path = list(fwd_nodes)  # nodes
        for nb_node in bwd_nodes:
            contracted_path.append(nb_node)

        # Now unpack: walk consecutive node pairs; for each pair find the edge
        # used (up if rank increases, down otherwise) and recursively unpack its
        # middle node.
        self._reset(nf, nb)
        out: list[int] = [contracted_path[0]]
        for i in range(len(contracted_path) - 1):
            u = contracted_path[i]
            w = contracted_path[i + 1]
            seg = self._unpack(u, w)
            out.extend(seg[1:])
        return out

    def _reset(self, nf: int, nb: int) -> None:
        for i in range(nf):
            node = int(self._seenf[i])
            self._df[node] = INF
            self._pf[node] = -1
        for i in range(nb):
            node = int(self._seenb[i])
            self._db[node] = INF
            self._pb[node] = -1

    def _unpack(self, u: int, w: int) -> list[int]:
        """Unpack the edge (u, w) into the original-node subpath [u, ..., w]."""
        mid = self._edge_mid(u, w)
        if mid < 0:
            return [u, w]
        left = self._unpack(u, mid)
        right = self._unpack(mid, w)
        return left + right[1:]

    def _edge_mid(self, u: int, w: int) -> int:
        """Find the middle node of edge (u, w) in the up/down CSR (-1 = original)."""
        rank = self.order
        if rank[u] < rank[w]:
            # up edge stored at u
            s, e = int(self.up_indptr[u]), int(self.up_indptr[u + 1])
            best_mid = -2
            best_c = INF
            for k in range(s, e):
                if int(self.up_indices[k]) == w:
                    if self.up_cost[k] < best_c:
                        best_c = self.up_cost[k]
                        best_mid = int(self.up_mid[k])
            if best_mid != -2:
                return best_mid
        else:
            # down edge stored at w
            s, e = int(self.dn_indptr[w]), int(self.dn_indptr[w + 1])
            best_mid = -2
            best_c = INF
            for k in range(s, e):
                if int(self.dn_indices[k]) == u:
                    if self.dn_cost[k] < best_c:
                        best_c = self.dn_cost[k]
                        best_mid = int(self.dn_mid[k])
            if best_mid != -2:
                return best_mid
        return -1  # treat as original if not found (shouldn't happen)

    def warmup(self) -> None:
        """Trigger numba compilation of the query kernels with tiny queries."""
        if self.n >= 2:
            self.query(0, 1)
            self.path(0, 1)


class CHPotentialOracle:
    """Bounded-change certified CH: an admissible CH-potential A* on true costs.

    A `ContractionHierarchy` built on *exact* costs is exact only for those costs
    — perturb a single edge and the hierarchy is invalid (its shortcuts may no
    longer be shortest). The certificate-friendly variant builds the CH on a
    *lower-bound* cost array (default 0.8x, exactly the ALT trick): the resulting
    CH distances are valid lower bounds on the true distances, so
    ``h(v) = CH-dist_lb(v, target)`` is an admissible+consistent heuristic for a
    forward A* run on the *true* costs. That A* returns the exact optimum for
    ANY true cost array whose every edge is >= the lower-bound costs — i.e. any
    perturbation within +-20% of the build costs (which never drops an edge below
    0.8x its original) — with ZERO rebuild of the hierarchy.

    The CH is built once on 0.8x costs; subsequent cost changes within the band
    are absorbed by writing the true CSR cost array (the same ~0.02 ms array
    write as ALT), and the query runs `_ch_potential_astar` against it.

    Parameters
    ----------
    graph:
        CSR surface (`n, indptr, indices, cost`).
    lower_bound_factor:
        The CH is built on ``factor * cost``. With 0.8, any later cost within
        -20% stays admissible. 1.0 reduces to an exact CH usable only when costs
        never decrease.
    """

    def __init__(
        self,
        graph,
        lower_bound_factor: float = 0.8,
        max_settled: int = 1000,
    ) -> None:
        self.graph = graph
        self.n = int(graph.n)
        self.lower_bound_factor = float(lower_bound_factor)
        self.ch = ContractionHierarchy(graph, max_settled=max_settled)
        self.build_seconds: Optional[float] = None
        # query scratch (n-arrays reset via touched-lists, so query is
        # O(search space + sweep) not O(n) of resets); allocated in build().
        self._rank_desc = None
        self._h = None
        self._seen_h = None
        self._g = None
        self._closed = None
        self._seen_g = None

    def build(self) -> float:
        """Build the underlying CH on the lower-bound cost array."""
        lb = np.asarray(self.graph.cost, dtype=np.float64) * self.lower_bound_factor
        self.build_seconds = self.ch.build(lb)
        n = self.n
        # nodes in DESCENDING rank order, for the CH one-to-all downward sweep.
        self._rank_desc = np.argsort(-np.asarray(self.ch.order)).astype(np.int64)
        self._h = np.full(n, INF, dtype=np.float64)
        self._seen_h = np.empty(n, dtype=np.int64)
        self._g = np.full(n, INF, dtype=np.float64)
        self._closed = np.zeros(n, dtype=np.bool_)
        self._seen_g = np.empty(n, dtype=np.int64)
        return self.build_seconds

    def query(self, s_idx: int, g_idx: int, costs: Optional[np.ndarray] = None) -> float:
        """Exact shortest-path cost on `costs` (defaults to the graph's true costs).

        The CH supplies the admissible potential; A* runs on the supplied (true)
        costs. Exact whenever every edge of `costs` is >= the lower-bound costs
        the CH was built on.
        """
        if self._rank_desc is None:
            raise RuntimeError("CHPotentialOracle.query before build(); call build() first")
        if s_idx == g_idx:
            return 0.0
        if costs is None:
            costs = np.asarray(self.graph.cost, dtype=np.float64)
        else:
            costs = np.asarray(costs, dtype=np.float64)
        ch = self.ch
        return float(
            _ch_potential_astar(
                int(s_idx), int(g_idx),
                np.asarray(self.graph.indptr), np.asarray(self.graph.indices), costs,
                ch.up_indptr, ch.up_indices, ch.up_cost,
                ch.dn_indptr, ch.dn_indices, ch.dn_cost,
                self._rank_desc,
                self._h, self._seen_h,
                self._g, self._closed, self._seen_g,
            )
        )

    def warmup(self) -> None:
        """Trigger numba compilation of the potential-A* kernel (one query)."""
        if self.n >= 2:
            self.query(0, 1)


# --------------------------------------------------------------------------- #
# Shortcut bookkeeping: record shortcuts during contraction
# --------------------------------------------------------------------------- #
# (Implemented as an attribute initialized in build via monkey-free approach.)


def _local_dijkstra(
    source: int,
    avoid: int,
    out_adj: list,
    contracted: np.ndarray,
    max_dist: float,
    max_settled: int,
    targets: list,
) -> dict:
    """Bounded local Dijkstra over the residual `out_adj`, avoiding `avoid`.

    Pure-Python (the witness searches are intentionally small: bounded by
    `max_dist` and `max_settled`, and they skip the node being contracted). Stops
    once all `targets` are settled or budgets are hit. Returns {node: dist} for
    settled nodes that matter.
    """
    if not targets:
        return {}
    target_set = set(t for t in targets if t != source)
    if not target_set:
        return {}
    dist = {source: 0.0}
    pq = [(0.0, source)]
    settled = 0
    remaining = set(target_set)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        if d > max_dist:
            break
        settled += 1
        if u in remaining:
            remaining.discard(u)
            if not remaining:
                break
        if settled > max_settled:
            break
        for v, (c, _mid) in out_adj[u].items():
            if v == avoid or contracted[v]:
                continue
            nd = d + c
            if nd > max_dist:
                continue
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
