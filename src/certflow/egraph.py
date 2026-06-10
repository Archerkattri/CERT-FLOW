"""E-Graphs-style baseline planner for CERT's repeated-query comparison.

Implements ``EGraphPlanner``: weighted A* (w = 1.2) whose heuristic is the
Experience-Graph heuristic of Phillips, Cohen, Chitta & Likhachev,
"E-Graphs: Bootstrapping Planning with Experience Graphs" (RSS 2012):

    h_E(s) = min( h(s),  min over experience-graph vertices v of
                  [ eps_E * c_exp(s, v) + h_exp(v) ] )

where ``h`` is the (consistent) base heuristic to the goal, ``c_exp(s, v)`` is a
cheapest off-experience cost to reach an experience vertex ``v`` (approximated
below), and ``h_exp(v)`` is the cost-to-goal *along experience edges only*,
computed by a Dijkstra over the experience subgraph rooted at the goal. The
heuristic biases the search to "snap onto" and reuse previously returned paths,
which is exactly the E-Graphs mechanism for accelerating repeated planning.

Faithful simplifications (documented honestly, per task brief)
--------------------------------------------------------------
1. **Off-experience reach term ``c_exp(s, v)``.** The RSS formulation defines
   ``c_exp`` via an inflated *off-E-graph* edge cost ``eps_E * c(s, s')`` so
   that h_E remains ``eps_E``-consistent. Computing the true minimum-cost
   off-experience reach to every experience vertex is itself a search; we
   approximate it with the *raw* base heuristic ``h(s, v)`` (an admissible
   lower bound on the real off-experience cost since ``h`` is consistent). We
   deliberately do NOT inflate this surrogate: the raw term creates a valley of
   lower f-values along the experience graph, which is what focuses search onto
   reused paths. Admissibility (and hence the weighted-A* ``cost <= w * optimal``
   bound) is preserved by the outer ``min`` with the base heuristic, which keeps
   ``h_E <= h(s, goal) <= h*``.
2. **Single inflation knob ``w``.** Rather than carrying a separate experience
   inflation ``eps_E`` and a search weight, we use one weighted-A* inflation
   ``w = 1.2``. The reported suboptimality guarantee is therefore the standard
   weighted-A* one (``cost <= w * optimal``), verified empirically in the tests.
3. **Grid base heuristic.** ``h`` is Manhattan distance scaled by a global
   lower bound on edge cost (``min_edge_cost``), which is consistent for the
   4-connected positive-cost grids used in the comparison. If no lower bound is
   supplied we fall back to the zero heuristic (always consistent).
4. **Experience graph = union of returned paths.** Every node pair on a returned
   path becomes an undirected experience edge with its *current* traversal cost;
   we recompute experience-edge costs from the live graph at plan time so a
   drifted world does not reuse stale experience costs (the search still expands
   real graph edges, so returned paths are always feasible on the live graph).
5. **h_exp via lazy goal-rooted Dijkstra over experience edges**, recomputed
   when the experience graph or goal changes. Bounded in size by the experience
   subgraph, not the full grid.

The planner expands *real* graph edges (so any returned path is valid and its
cost is measured on the live graph); only the heuristic consults experience.
Therefore correctness/feasibility never depends on experience quality — only
speed does. The ``expansions`` attribute counts states popped from OPEN by the
most recent ``plan`` call (for node-expansion benchmarks).
"""
from __future__ import annotations

import heapq
import itertools
from typing import Callable, Iterable

from certflow.types import Node

INF = float("inf")


def path_cost_on(
    graph: dict[Node, dict[Node, float]], path: list[Node]
) -> float:
    """Sum live edge costs along ``path``; INF if any edge is missing."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        w = graph.get(a, {}).get(b)
        if w is None:
            return INF
        total += w
    return total


def manhattan_heuristic(
    min_edge_cost: float,
) -> Callable[[Node, Node], float]:
    """Consistent grid heuristic: Manhattan distance * a lower bound on edge cost.

    Nodes must be ``(row, col)`` tuples. ``min_edge_cost`` is a global lower
    bound on every edge cost that can occur over the run; using it keeps the
    heuristic admissible/consistent even as costs drift upward.
    """

    def h(a: Node, b: Node) -> float:
        (ar, ac), (br, bc) = a, b  # type: ignore[misc]
        return (abs(ar - br) + abs(ac - bc)) * min_edge_cost

    return h


class EGraphPlanner:
    """Weighted A* (w) with an Experience-Graph heuristic (Phillips et al. 2012).

    One instance plans repeated queries on a (possibly drifting) graph, reusing
    previously returned paths as experience. Call ``plan(start, goal)`` per
    query; the returned path is added to the experience graph automatically.
    Use ``update_graph`` (or pass a fresh ``graph`` to ``plan``) to apply drift.

    Attributes
    ----------
    expansions : int
        States popped from OPEN by the most recent ``plan`` call.
    """

    def __init__(
        self,
        graph: dict[Node, dict[Node, float]],
        heuristic: Callable[[Node, Node], float] | None = None,
        w: float = 1.2,
    ) -> None:
        self._graph: dict[Node, dict[Node, float]] = graph
        self._h: Callable[[Node, Node], float] = (
            heuristic if heuristic is not None else (lambda a, b: 0.0)
        )
        self._w: float = w
        # Experience graph: undirected adjacency of node pairs seen on returned
        # paths. Costs are recomputed from the live graph at plan time.
        self._exp_adj: dict[Node, set[Node]] = {}
        self.expansions: int = 0
        # Cache of h_exp values keyed by goal (invalidated on experience change).
        self._hexp_cache_goal: Node | None = None
        self._hexp: dict[Node, float] = {}
        self._exp_dirty: bool = True
        # Per-plan memo for h_E (cleared at the start of every plan call).
        self._he_cache: dict[Node, float] = {}

    # -- experience management -------------------------------------------------

    def add_experience(self, path: Iterable[Node]) -> None:
        """Register a path as reusable experience (consecutive pairs as edges)."""
        path = list(path)
        changed = False
        for a, b in zip(path, path[1:]):
            if b not in self._exp_adj.get(a, ()):  # new undirected edge
                self._exp_adj.setdefault(a, set()).add(b)
                self._exp_adj.setdefault(b, set()).add(a)
                changed = True
        if changed:
            self._exp_dirty = True

    def update_graph(self, graph: dict[Node, dict[Node, float]]) -> None:
        """Replace the live graph (e.g. after world drift). Experience kept."""
        self._graph = graph
        # Experience-edge costs are read live, so h_exp depends on the graph;
        # invalidate the cache.
        self._exp_dirty = True

    # -- experience-graph heuristic -------------------------------------------

    def _exp_edge_cost(self, a: Node, b: Node) -> float:
        """Live traversal cost along experience edge a-b (min of both dirs)."""
        c = INF
        ca = self._graph.get(a, {}).get(b)
        if ca is not None:
            c = ca
        cb = self._graph.get(b, {}).get(a)
        if cb is not None and cb < c:
            c = cb
        return c

    def _rebuild_hexp(self, goal: Node) -> None:
        """Goal-rooted Dijkstra over experience edges only -> h_exp(v)."""
        self._hexp = {}
        if goal not in self._exp_adj:
            self._hexp_cache_goal = goal
            self._exp_dirty = False
            return
        dist: dict[Node, float] = {goal: 0.0}
        pq: list[tuple[float, int, Node]] = [(0.0, 0, goal)]
        ctr = itertools.count(1)
        while pq:
            d, _, u = heapq.heappop(pq)
            if d > dist.get(u, INF):
                continue
            for v in self._exp_adj.get(u, ()):  # experience neighbors
                w = self._exp_edge_cost(u, v)
                if w == INF:
                    continue
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, next(ctr), v))
        self._hexp = dist
        self._hexp_cache_goal = goal
        self._exp_dirty = False

    def _h_e(self, s: Node, goal: Node) -> float:
        """E-Graphs heuristic h_E(s) (simplification #1: c_exp ~= w*h).

        Memoized per plan via ``self._he_cache`` (cleared at the start of each
        ``plan``); the inner ``min`` scans the experience vertex set, so caching
        avoids re-scanning when a node is re-pushed during search.
        """
        cached = self._he_cache.get(s)
        if cached is not None:
            return cached
        base = self._h(s, goal)
        if not self._hexp:
            self._he_cache[s] = base
            return base
        best = base
        # min over experience vertices v of [ h(s, v) + h_exp(v) ].
        # Note: we use the *raw* base heuristic h(s, v) for the off-experience
        # reach surrogate rather than the RSS inflation eps_E * c(s, s'). Raw
        # h(s, v) is an admissible lower bound on the true off-experience cost,
        # so this term creates a "valley" of lower f-values along the
        # experience graph (focusing the search to snap onto and follow reused
        # paths) while keeping h_E <= base <= h* (the outer min guarantees
        # admissibility, hence the weighted-A* cost <= w * optimal bound).
        for v, hexp_v in self._hexp.items():
            cand = self._h(s, v) + hexp_v
            if cand < best:
                best = cand
        self._he_cache[s] = best
        return best

    # -- planning --------------------------------------------------------------

    def plan(
        self,
        start: Node,
        goal: Node,
        graph: dict[Node, dict[Node, float]] | None = None,
    ) -> tuple[list[Node] | None, float]:
        """Weighted-A* plan from ``start`` to ``goal`` using the E-graph heuristic.

        Returns ``(path, cost)`` measured on the live graph, or ``(None, inf)``
        if unreachable. The returned path is added to the experience graph.
        ``expansions`` is set to the number of OPEN pops.
        """
        if graph is not None:
            self.update_graph(graph)
        if self._exp_dirty or self._hexp_cache_goal != goal:
            self._rebuild_hexp(goal)
        self._he_cache = {}  # h_E memo is graph/goal-specific; reset per plan

        self.expansions = 0
        if start == goal:
            self.add_experience([start])
            return [start], 0.0

        g: dict[Node, float] = {start: 0.0}
        prev: dict[Node, Node] = {}
        ctr = itertools.count(1)
        f0 = self._w * self._h_e(start, goal)
        open_pq: list[tuple[float, int, Node]] = [(f0, next(ctr), start)]
        closed: set[Node] = set()

        # Experience incumbent (anytime / ARA*-style upper bound). Whenever the
        # search reaches a vertex that lies on the experience graph, we attempt
        # to *complete* the route by snapping onto the cheapest experience path
        # to the goal (cost read from the live graph). That yields a feasible
        # path whose cost prunes OPEN: any node with f >= incumbent cannot lead
        # to a strictly better solution. This is the E-Graphs "reuse" mechanism
        # and is the source of the expansion savings on repeated queries; the
        # weighted-A* termination rule (stop when min f on OPEN >= incumbent)
        # keeps the returned cost <= w * optimal.
        incumbent_cost: float = INF
        incumbent_path: list[Node] | None = None

        def record_incumbent(cost: float, path: list[Node]) -> None:
            nonlocal incumbent_cost, incumbent_path
            if cost < incumbent_cost:
                incumbent_cost = cost
                incumbent_path = path

        def try_snap(node: Node, g_node: float) -> None:
            nonlocal incumbent_cost, incumbent_path
            if node not in self._exp_adj:
                return
            tail = self._experience_completion(node, goal)
            if tail is None:
                return
            tail_cost = path_cost_on(self._graph, tail)
            if tail_cost == INF:
                return
            total = g_node + tail_cost
            if total < incumbent_cost:
                # Reconstruct head start..node, append experience tail.
                head = [node]
                cur = node
                while cur != start:
                    cur = prev[cur]
                    head.append(cur)
                head.reverse()
                incumbent_cost = total
                incumbent_path = head + tail[1:]

        # Upfront snap: if the start itself lies on the experience graph (the
        # common case for a repeated query from a pooled start), seed a tight
        # incumbent immediately so OPEN pruning bites from the first expansion.
        if start in self._exp_adj:
            tail = self._experience_completion(start, goal)
            if tail is not None:
                tc = path_cost_on(self._graph, tail)
                if tc < INF:
                    record_incumbent(tc, tail)

        while open_pq:
            f_u, _, u = heapq.heappop(open_pq)
            if u in closed:
                continue
            # Termination: nothing on OPEN can beat the incumbent.
            if f_u >= incumbent_cost:
                break
            closed.add(u)
            self.expansions += 1
            if u == goal:
                if g[u] < incumbent_cost:
                    path = [goal]
                    while path[-1] != start:
                        path.append(prev[path[-1]])
                    path.reverse()
                    incumbent_cost = g[u]
                    incumbent_path = path
                break
            gu = g[u]
            try_snap(u, gu)
            for v, w in self._graph.get(u, {}).items():
                if v in closed:
                    continue
                nd = gu + w
                if nd < g.get(v, INF):
                    g[v] = nd
                    prev[v] = u
                    f = nd + self._w * self._h_e(v, goal)
                    if f < incumbent_cost:
                        heapq.heappush(open_pq, (f, next(ctr), v))

        if incumbent_path is None:
            return None, INF
        self.add_experience(incumbent_path)
        return incumbent_path, incumbent_cost

    def _experience_completion(self, src: Node, goal: Node) -> list[Node] | None:
        """Cheapest experience-only path src -> goal (live costs), or None.

        Uses the goal-rooted h_exp Dijkstra tree's structure by re-deriving a
        path via greedy descent on h_exp (the experience-edge distance-to-goal).
        """
        if src not in self._hexp or self._hexp.get(goal, INF) != 0.0:
            return None
        if self._hexp.get(src, INF) == INF:
            return None
        path = [src]
        cur = src
        guard = 0
        max_steps = len(self._exp_adj) + 1
        while cur != goal:
            guard += 1
            if guard > max_steps:
                return None  # cycle guard
            best_v = None
            best_h = self._hexp.get(cur, INF)
            for v in self._exp_adj.get(cur, ()):  # follow decreasing h_exp
                hv = self._hexp.get(v, INF)
                w = self._exp_edge_cost(cur, v)
                if w == INF:
                    continue
                # Descent step consistent with the Dijkstra tree.
                if hv + w <= self._hexp.get(cur, INF) + 1e-9 and hv < best_h:
                    best_h = hv
                    best_v = v
            if best_v is None:
                return None
            cur = best_v
            path.append(cur)
        return path
