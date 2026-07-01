"""Additive multi-agent certificate (TEAM-CERT survivor).

For ``N`` agents that share a drifting-cost graph, the per-agent CERT-FLOW
certificates add directly::

    LB_team = sum_i LB_i        UB_team = sum_i UB_i

Soundness requirement (copied verbatim from the TEAM-CERT design; see
``docs/results/multiagent.md``): **all agents must share ONE planner / one
conformal edge-price store**, so the observation age ``a_e`` of every edge is
*global* -- any agent's observation refreshes that edge for every other agent.
Only then is each per-agent certificate sound over the shared store (the same
per-edge soundness as the single-agent claim), and, because the team optimum
separates,

    OPT_team = sum_i OPT_i,

so summing the sound per-agent bounds gives

    LB_team = sum_i LB_i <= sum_i OPT_i = OPT_team <= sum_i UB_i = UB_team.

The additive extension is sound and *exact* for independent agents. The stronger
congestion-coupled joint certificate (edge cost rising with team load) and its
decision-focused shared-sensing allocator were measured to be *looser* than this
additive bound on the real METR-LA road network (route diversity lets each agent
avoid the few congested edges) and are deliberately **not** ported here; see the
post-mortem in ``docs/results/multiagent.md``.

Team confidence combines the per-agent confidences by a union bound over the
agents' miscoverage events::

    conf_team = 1 - sum_i (1 - conf_i),   floored at 0.

An invalid (``confidence <= 0``) member drags the team confidence down exactly
as its own miscoverage budget would; the floor keeps the reported number a valid
probability.
"""
from __future__ import annotations

from dataclasses import dataclass

from certflow.types import Certificate


@dataclass
class TeamCertificate:
    """Additive team certificate over a shared conformal edge-price store.

    ``lb`` / ``ub`` bracket the team objective ``OPT_team = sum_i OPT_i``;
    ``confidence`` is the union-bound team confidence; ``per_agent`` keeps the
    individual certificates for inspection.
    """

    lb: float
    ub: float
    confidence: float
    per_agent: list[Certificate]

    @property
    def gap(self) -> float:
        return self.ub - self.lb

    @property
    def valid(self) -> bool:
        return self.confidence > 0.0


def additive_certificate(certs: list[Certificate]) -> TeamCertificate:
    """Sum the per-agent certificates into one additive team certificate.

    Parameters
    ----------
    certs:
        One :class:`~certflow.types.Certificate` per agent, **all produced from
        the same shared planner / conformal store** (see the module docstring --
        this is what makes the ages global and each summand sound).

    Returns
    -------
    A :class:`TeamCertificate` with ``lb = sum LB_i``, ``ub = sum UB_i``,
    ``confidence = max(0, 1 - sum_i (1 - conf_i))``.
    """
    if not certs:
        raise ValueError("additive_certificate requires at least one certificate")
    lb = float(sum(c.lb for c in certs))
    ub = float(sum(c.ub for c in certs))
    # Union bound over the per-agent miscoverage events: the chance that ANY
    # agent's certificate is violated is at most the sum of the per-agent
    # miscoverage budgets (1 - conf_i). Floor at 0 so the result stays a
    # probability.
    conf = 1.0 - sum(1.0 - c.confidence for c in certs)
    conf = max(0.0, conf)
    return TeamCertificate(lb=lb, ub=ub, confidence=conf, per_agent=list(certs))
