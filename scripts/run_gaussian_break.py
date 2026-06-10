"""Gaussian-break experiment: where does mu±beta*sigma actually under-cover?

Tier-0 showed the Gaussian baseline sound-but-bloated at PATH level — the
Bonferroni union-bound slack masks its building block. The honest test is at
the EDGE level: with ACI frozen, each sensed observation either lands in the
edge's interval (built at alpha_edge = alpha'/L) or not. A calibrated method
has miss-rate <= alpha_edge; a parametric one can exceed it arbitrarily.

Conditions: 10x10 grids (L~18, alpha_edge ~ 0.0056), noise in
{gaussian (control), student_t, skewed}; CERT (conformal, lambda=1,
ACI frozen) vs Gaussian (mu + z*sigma fit, same loop). Skewed noise violates
CERT's A3 as well — T1a's distribution-free edge guarantee should still hold;
that asymmetry is the experiment's point.

Run: cert_env/bin/python scripts/run_gaussian_break.py [--quick]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from scipy.stats import beta as beta_dist

from certflow.baselines import GaussianCertPlanner
from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.oracle import opt

QUICK = "--quick" in sys.argv
SEEDS = 5 if QUICK else 25
ROUNDS = 150 if QUICK else 400
ROWS = COLS = 10
ALPHA_PRIME = 0.1

CONDITIONS = [
    ("gaussian (control)", dict(kind="static", noise_family="gaussian", noise_scale=0.2)),
    ("student_t",          dict(kind="static", noise_family="student_t", noise_scale=0.2)),
    ("skewed",             dict(kind="static", noise_family="skewed", noise_scale=0.2)),
    ("drift .02 + skewed", dict(kind="bounded", rho=0.02, noise_family="skewed", noise_scale=0.2)),
]
PLANNERS = [("CERT", CertPlanner), ("Gaussian", GaussianCertPlanner)]


def cp_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


def run_cell(cls, world_kw: dict) -> dict:
    # rho_w note: the weighted buffer's effective sample size is ~1/(1-rho_w);
    # supporting alpha_edge needs ESS >= 1/alpha_edge - 1. At alpha_edge=0.0056
    # that means rho_w >= ~0.995 — with 0.99 CERT would stay in warm-up forever
    # (a real design constraint, documented in the results).
    cfg = PlannerConfig(
        epsilon=8.0, alpha_prime=ALPHA_PRIME, rho_w=0.999, eps_tv=1e-4,
        use_aci=False,  # frozen: miss rates compare against the constructed level
    )
    edge_miss = edge_n = 0          # planner-SELECTED edges (selection diagnostic)
    audit_miss = audit_n = 0        # AUDITED uniformly-random edges (the guarantee test:
    # T1a covers a FIXED edge; planner-selected edges suffer a winner's curse —
    # the optimistic path prefers low-c_hat edges, so their next observation is
    # biased outward, inflating selected-edge miss rates for ANY method)
    covered = valid = 0
    alpha_edges = []
    import random as _random
    for seed in range(SEEDS):
        kw = dict(world_kw)
        kind = kw.pop("kind")
        w = grid_world(ROWS, COLS, seed=seed, kind=kind, **kw)
        p = cls(w, (0, 0), (ROWS - 1, COLS - 1), cfg)
        audit_rng = _random.Random(seed + 7777)
        all_edges = list(p.beliefs)
        for _ in range(ROUNDS):
            t_round = p.t
            errs0, n0 = p.aci._errs, p.aci._t
            cert, _ = p.round()
            edge_miss += p.aci._errs - errs0
            edge_n += p.aci._t - n0
            if cert.valid:
                valid += 1
                _, o = opt(w, t_round, (0, 0), (ROWS - 1, COLS - 1))
                covered += cert.lb - 1e-9 <= o <= cert.ub + 1e-9
                alpha_edges.append(ALPHA_PRIME / max(len(cert.path) - 1, 1))
                # independent audit: fresh observation of a random edge,
                # never fed back to the planner. Tested against the UNCLIPPED
                # nominal interval (T1a observable semantics): the cost-floor
                # clip is sound only for latent costs, and observables can be
                # negative under heavy-tailed noise.
                e = all_edges[audit_rng.randrange(len(all_edges))]
                b = p.beliefs[e]
                q_now = p.scorer.quantile(
                    ALPHA_PRIME / max(len(cert.path) - 1, 1), p.t
                )
                y = w.observe(e, p.t)
                if not math.isfinite(q_now):
                    continue
                half = cfg.latent_margin * q_now + b.rho * b.age(p.t)
                audit_n += 1
                audit_miss += not (
                    b.c_hat - half - 1e-12 <= y <= b.c_hat + half + 1e-12
                )
    alpha_edge = sum(alpha_edges) / len(alpha_edges) if alpha_edges else ALPHA_PRIME / 18
    alo, ahi = cp_ci(audit_miss, audit_n)
    return dict(
        audit_n=audit_n,
        audit_miss=audit_miss / audit_n if audit_n else float("nan"),
        audit_ci_lo=alo, audit_ci_hi=ahi,
        alpha_edge=alpha_edge,
        audit_ratio=(audit_miss / audit_n) / alpha_edge if audit_n else float("nan"),
        broken=bool(audit_n and alo > alpha_edge),  # CI floor above claimed level
        sel_n=edge_n,
        sel_miss=edge_miss / edge_n if edge_n else float("nan"),
        path_valid=valid,
        path_coverage=covered / valid if valid else float("nan"),
    )


def main() -> None:
    table = []
    for label, world_kw in CONDITIONS:
        for pname, cls in PLANNERS:
            m = run_cell(cls, world_kw)
            m.update(condition=label, planner=pname)
            table.append(m)
            print(f"done: {label} / {pname}", flush=True)

    outdir = Path("results/gaussian_break")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "table.json").write_text(json.dumps(table, indent=2))

    hdr = (f"{'condition':20} {'planner':9} {'audit-n':>7} {'audit-miss':>10} "
           f"{'95% CI':>16} {'alpha_e':>8} {'ratio':>6} {'sel-miss':>9} {'pathcov':>8}  verdict")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in table:
        ci = f"[{r['audit_ci_lo']:.4f},{r['audit_ci_hi']:.4f}]"
        print(f"{r['condition']:20} {r['planner']:9} {r['audit_n']:>7} "
              f"{r['audit_miss']:>10.4f} {ci:>16} {r['alpha_edge']:>8.4f} "
              f"{r['audit_ratio']:>6.1f} {r['sel_miss']:>9.4f} {r['path_coverage']:>8.3f}  "
              f"{'BROKEN' if r['broken'] else 'ok'}")


if __name__ == "__main__":
    main()
