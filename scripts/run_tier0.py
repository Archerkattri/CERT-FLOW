"""Tier-0 coverage validation sweep (spec section 7).

Rows of the headline table:
  on-model:  bounded drift at rho in {0.005, 0.02, 0.05} + static
  misspec:   rho_hat/rho in {0.5, 2.0} at rho=0.02
  off-model: jump, periodic (A1 violated / stressed)

Reports empirical coverage among VALID rounds with Clopper-Pearson 95% CI
against the claimed confidence, plus gap / certification / sensing stats.
Run:  cert_env/bin/python scripts/run_tier0.py [--quick]
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from scipy.stats import beta as beta_dist

from certflow.episodes import tier0_episode
from certflow.harness import ExperimentConfig, run_experiment, save_results

QUICK = "--quick" in sys.argv

BASE = ExperimentConfig(
    rows=6, cols=6, kind="bounded", rho=0.02,
    noise_family="gaussian", noise_scale=0.05,
    # eps_tv: drift-adjusted scores are near-stationary under A1 + stationary
    # observation noise, so the honest A2 rate is small. 1e-3 makes the
    # L*Delta_stale debt extinguish the certificate mid-episode (~3% valid).
    epsilon=5.0, alpha_prime=0.2, rho_w=0.99, eps_tv=1e-4,
    gamma_aci=0.01, delta=1.0, rho_hat_over_rho=1.0,
    n_seeds=5 if QUICK else 25, max_rounds=100 if QUICK else 300, base_seed=2026,
)

ROWS: list[tuple[str, dict]] = [
    ("static (rho=0)",          dict(kind="static")),
    ("bounded rho=0.005",       dict(rho=0.005)),
    ("bounded rho=0.02",        dict(rho=0.02)),
    ("bounded rho=0.05",        dict(rho=0.05)),
    ("misspec rho_hat=0.5*rho", dict(rho=0.02, rho_hat_over_rho=0.5)),
    ("misspec rho_hat=2*rho",   dict(rho=0.02, rho_hat_over_rho=2.0)),
    ("off-model: jump",         dict(kind="jump")),
    ("off-model: periodic",     dict(kind="periodic")),
    # provable-mode rows (theory.tex T1b + honest-accounting item 1)
    ("lambda=2 (T1b)",          dict(latent_margin=2.0)),
    ("lambda=2 + thinned",      dict(latent_margin=2.0, thinned_scores=True)),
    # the full provable T1b mode: raw conformal quantile (ACI frozen)
    ("provable (l2+thin+noACI)", dict(latent_margin=2.0, thinned_scores=True, use_aci=False)),
    # T4 sum-aware UB rows (freshness-gated, kappa for incumbent stability):
    # informative where the noise floor dominates; ~no-op under drift
    ("sum-aware static",        dict(sum_aware_ub=True, use_kappa=True, kind="static")),
    ("sum-aware noise-dom",     dict(sum_aware_ub=True, use_kappa=True, kind="static", noise_scale=0.2)),
    # noise-dominated regime: lambda=2 should cost ~2x the noise floor here
    ("noise-dom static lam=1",  dict(kind="static", noise_scale=0.2)),
    ("noise-dom static lam=2",  dict(kind="static", noise_scale=0.2, latent_margin=2.0)),
    # A2 misspecification (over-assumed score drift; A1 misspec rows above)
    ("A2 misspec eps_tv=1e-3",  dict(eps_tv=1e-3)),
    # the recommended configuration (online rho + hybrid + kappa + adaptive
    # + gated sum-aware UB) on the standard drifting world
    ("CERT-best (recommended)", dict(rho_mode="online", hybrid_sensing=True,
                                     use_kappa=True, adaptive_rate=True,
                                     sum_aware_ub=True)),
]


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


def main() -> None:
    outdir = Path("results/tier0")
    outdir.mkdir(parents=True, exist_ok=True)
    table = []
    for label, overrides in ROWS:
        cfg = dataclasses.replace(BASE, **overrides)
        result = run_experiment(tier0_episode, cfg)

        covered = valid = 0
        claimed_sum = 0.0
        gaps = []
        for ep in result.episodes:
            if not hasattr(ep, "rounds"):
                continue  # failure record
            for r in ep.rounds:
                if r.confidence > 0.0:
                    valid += 1
                    covered += r.covered
                    claimed_sum += r.confidence
                    gaps.append(r.ub - r.lb)
        agg = result.aggregate()
        lo, hi = clopper_pearson(covered, valid)
        row = {
            "label": label,
            "valid_rounds": valid,
            "valid_fraction": valid / agg["n_rounds_total"] if agg["n_rounds_total"] else 0.0,
            "coverage": covered / valid if valid else float("nan"),
            "cov_ci_lo": lo,
            "cov_ci_hi": hi,
            "claimed_mean": claimed_sum / valid if valid else float("nan"),
            "gap_median": sorted(gaps)[len(gaps) // 2] if gaps else float("nan"),
            "certified_fraction": agg["certified_fraction"],
            "sense_spend_total": agg["sense_spend_total"],
            "replan_p50_ms": 1e3 * agg["replan_latency_p50"],
        }
        table.append(row)
        save_results(result, str(outdir))
        print(f"done: {label}", flush=True)

    (outdir / "table.json").write_text(json.dumps(table, indent=2))
    hdr = (
        f"{'condition':26} {'valid%':>6} {'coverage':>9} {'95% CI':>16} "
        f"{'claimed':>8} {'gap~':>6} {'cert%':>6} {'p50ms':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in table:
        ci = f"[{r['cov_ci_lo']:.3f},{r['cov_ci_hi']:.3f}]"
        print(
            f"{r['label']:26} {100 * r['valid_fraction']:>5.1f}% {r['coverage']:>9.3f} {ci:>16} "
            f"{r['claimed_mean']:>8.3f} {r['gap_median']:>6.2f} "
            f"{100 * r['certified_fraction']:>5.1f}% {r['replan_p50_ms']:>6.1f}"
        )


if __name__ == "__main__":
    main()
