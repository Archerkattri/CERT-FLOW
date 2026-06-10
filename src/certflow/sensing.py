"""Route-critical sensing selection (spec section 4.3 step 6).

Greedy expected-gap-shrink per unit sensing cost over the route-critical edge
set, backstopped by age-triggered forced re-sensing of the optimistic path's
edges. The backstop is what makes Theorem T2'(a) hold for the deployed policy
(theory note section 1): greedy alone can starve edges.
"""
from __future__ import annotations

from certflow.graphcore import dijkstra
from certflow.types import Edge, EdgeBelief, Node


def path_edges(path: list[Node] | None) -> list[Edge]:
    if not path or len(path) < 2:
        return []
    return list(zip(path[:-1], path[1:]))


def near_optimal_alternatives(
    graph_lower: dict[Node, dict[Node, float]],
    start: Node,
    goal: Node,
    lb: float,
    k: int = 3,
    delta_subopt: float = 0.1,
    penalty: float = 3.0,
) -> set[Edge]:
    """Edges of up to k alternative ell-shortest paths within (1+delta)*LB.

    Iterative penalty method: re-run Dijkstra with previously-found paths'
    edges inflated, keep alternatives whose cost under the ORIGINAL lower
    metric stays within the suboptimality band. Approximate by design (this
    builds a sensing candidate set, not a certificate).
    """
    crit: set[Edge] = set()
    work = {u: dict(nbrs) for u, nbrs in graph_lower.items()}
    for _ in range(k):
        path, _ = dijkstra(work, start, goal)
        if path is None:
            break
        edges = path_edges(path)
        true_lcost = sum(graph_lower[u][v] for u, v in edges)
        if true_lcost <= (1.0 + delta_subopt) * lb:
            crit.update(edges)
        for u, v in edges:
            work[u][v] *= penalty
    return crit


def baseline_select(
    policy: str,
    beliefs: dict[Edge, EdgeBelief],
    t: float,
    rng,
    mean_graph: dict[Node, dict[Node, float]] | None = None,
    start: Node | None = None,
    goal: Node | None = None,
) -> Edge | None:
    """Baseline sensing policies. None are certificate-critical: random,
    max_age (freshness / persistent-monitoring revisit), max_width (global
    info-gain proxy), none, and voi (CTP-with-remote-sensing style: sense
    where uncertainty x EXPECTED-cost-route relevance is highest — improves
    the expected route, not the certified gap)."""
    edges = list(beliefs)
    if policy == "none" or not edges:
        return None
    if policy == "random":
        return edges[rng.randrange(len(edges))]
    if policy == "max_age":
        return max(edges, key=lambda e: beliefs[e].age(t))
    if policy == "max_width":
        return max(edges, key=lambda e: beliefs[e].rho * beliefs[e].age(t))
    if policy == "voi":
        if mean_graph is None or start is None or goal is None:
            raise ValueError("voi policy needs mean_graph/start/goal")
        path, _ = dijkstra(mean_graph, start, goal)
        on_path = set(path_edges(path)) if path else set()
        cand = [e for e in on_path if e in beliefs] or edges
        return max(cand, key=lambda e: beliefs[e].rho * beliefs[e].age(t))
    raise ValueError(f"unknown sensing policy: {policy}")


def select_observation(
    beliefs: dict[Edge, EdgeBelief],
    p_lb_edges: list[Edge],
    p_ub_edges: list[Edge],
    alt_edges: set[Edge],
    q: float,
    t: float,
    backstop_age: float,
) -> Edge | None:
    """Pick the next edge to sense, or None if nothing is worth sensing.

    Backstop first: the oldest optimistic-path edge whose age exceeds
    backstop_age is forced (round-robin guarantee). Otherwise greedy:
    score = expected width recovered by re-observing / sensing cost, where
    re-observation resets the interval width from 2(q + rho*age) to 2q, so
    the recoverable width is 2*rho*age. Off-path alternatives count half
    (they shrink the gap only if they change path selection).
    """
    on_lb = set(p_lb_edges)
    stale = [e for e in p_lb_edges if beliefs[e].age(t) > backstop_age]
    if stale:
        return max(stale, key=lambda e: beliefs[e].age(t))

    on_ub = set(p_ub_edges)
    primary = on_lb | on_ub  # full-weight edges (on either certifying path)
    best_edge: Edge | None = None
    best_score = 0.0
    for e in primary | alt_edges:
        b = beliefs[e]
        # an unobserved edge carries effectively unbounded width (soundness
        # fix: prior is not an observation) — observing it collapses the
        # largest interval in the system, so it dominates the gap-shrink score
        recoverable = 1e9 if not b.observed else 2.0 * b.rho * b.age(t)
        weight = 1.0 if e in primary else 0.5
        score = weight * recoverable / max(b.sense_cost, 1e-9)
        if score > best_score:
            best_score, best_edge = score, e
    return best_edge
