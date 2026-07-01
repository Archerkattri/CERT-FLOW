# Multi-agent extension: the additive team certificate

`certflow.team` — `TeamCertificate` + `additive_certificate(certs)`.

## Result

For `N` agents that share a drifting-cost graph, the per-agent CERT-FLOW
certificates add directly:

    LB_team = sum_i LB_i  <=  sum_i OPT_i = OPT_team  <=  sum_i UB_i = UB_team,

with team confidence combined by a union bound over the agents' miscoverage
events, `conf_team = max(0, 1 - sum_i (1 - conf_i))`. The additive bound is
**sound and exact for independent agents**: each per-agent certificate is sound
over the shared store (the same per-edge soundness as the single-agent claim),
and the team optimum separates (`OPT_team = sum_i OPT_i`), so summing the sound
per-agent bounds is exact, not conservative, on the LB side and sound on the UB
side.

### Soundness requirement (load-bearing)

All agents **must share one planner / one conformal edge-price store**, so the
observation age `a_e` of every edge is *global* — any agent's observation
refreshes that edge for every other agent. Only then is each summand sound over
the shared store. `additive_certificate` therefore documents that its inputs must
be produced from the same shared store; the tests
(`tests/test_team.py::test_soundness_and_coverage`) build exactly that: one
shared `BoundedDriftWorld` observed through a single global RNG, N planners
driven in lockstep, and verify `sum LB <= sum OPT <= sum UB` at coverage
`>= 1 - alpha` for N ∈ {2, 3, 4}.

## Why only the additive bound (post-mortem)

A stronger congestion-coupled *joint* certificate — edge cost rising with team
load, `c_e(m) = phi_e(1 + beta_e m)`, priced over unique `(e, m)` cells —
together with a decision-focused shared-sensing allocator was tested during the
TEAM-CERT phase. On synthetic forced-bottleneck graphs the joint certificate is
strictly tighter and its advantage grows with `N`; but on the **real METR-LA
Los-Angeles road network** — where route diversity lets each agent avoid the few
congested edges — the additive bound is consistently **≈10% tighter** than the
joint object at every `N`, and the sensing allocator showed no deployable win
region against independent certification and standard predict-then-optimize
baselines.

Certified-gap ratio, additive / joint ( >1 = joint tighter ):

| Instance | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|
| Synthetic forced bottleneck | 1.23 | 1.47 | 1.78 | — |
| Synthetic load-spreading | 1.12 | 1.14 | 1.10 | — |
| **METR-LA (real road network)** | **0.91** | **0.90** | **0.91** | **0.91** |

All certificates were sound (0 violations) and covered (`>= 1 - alpha`, empirical
≈0.98) on every instance; the table reports tightness, not validity.

**Source artifact:** `certified-planning/experiments/results/congestion_cert.json`
(model, additive baseline, allocator, and per-instance configs recorded in its
`notes`/`params`; `alpha = 0.1`, target coverage 0.9). The additive-vs-joint
readout on real METR-LA is the reason **only** the additive extension is ported
here.

## What is deliberately NOT ported

From the TEAM-CERT prototype (`certified-planning/experiments/certplan/`):
`congestion.py`, the `joint_ub` block-quantile UB branch, all sensing allocators
(`sensing.py`), and `mpc_field.py`. Those were falsified or non-deployable on
real data; the survivor is the additive certificate above.
