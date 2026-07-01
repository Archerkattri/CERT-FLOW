"""CERT-FLOW: certified route planning under drifting costs.

Every replanning round emits a high-probability certificate LB <= OPT <= UB
on the optimal route cost, built from age-weighted non-exchangeable conformal
prediction over drift-adjusted observation residuals, and directs paid sensing
at the edges that shrink the certified gap fastest.

Quickstart::

    from certflow import CertPlanner, PlannerConfig
    from certflow.drift import grid_world

    world = grid_world(6, 6, seed=0, kind="bounded", rho=0.02, noise_scale=0.05)
    planner = CertPlanner(world, (0, 0), (5, 5),
                          PlannerConfig(epsilon=5.0, alpha_prime=0.2))
    for _ in range(150):
        cert, sensed = planner.round()
    print(cert.lb, cert.ub, cert.confidence)

Submodules: conformal (quantile machinery), cert (the planner loop), sensing
(observation selection), fastgraph (flat-array engine), snapshot / ch
(certificate-gated preprocessing), drift / realworld / movingai / roadnet
(worlds and graphs), harness / episodes / oracle (experiment infrastructure).
"""
from certflow.cert import CertPlanner, PlannerConfig
from certflow.conformal import (
    ACITracker,
    CIACalibrator,
    CIAResult,
    ConformalScorer,
    ConformalTestMartingale,
    PASCCalibrator,
    PASCResult,
    conformal_e_value,
    conformal_p_value,
    effective_sample_size,
    merge_e_values,
    residual_drift_score,
    score_ratio_e_value,
)
from certflow.team import TeamCertificate, additive_certificate
from certflow.types import Certificate, EdgeBelief, World

__version__ = "1.0.2"

__all__ = [
    "ACITracker",
    "CIACalibrator",
    "CIAResult",
    "CertPlanner",
    "Certificate",
    "ConformalScorer",
    "ConformalTestMartingale",
    "EdgeBelief",
    "PASCCalibrator",
    "PASCResult",
    "PlannerConfig",
    "TeamCertificate",
    "World",
    "additive_certificate",
    "conformal_e_value",
    "conformal_p_value",
    "effective_sample_size",
    "merge_e_values",
    "residual_drift_score",
    "score_ratio_e_value",
    "__version__",
]
