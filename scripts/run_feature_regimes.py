"""Feature-regime validation: spatial predictor (Exp A) and decision-uniform (Exp B).

Experiment A — spatial predictor in its designed dense-sensing regime (METR-LA).
  Hypothesis: with many observations per round (reporting sensor network),
  neighborhoods stay fresh and the P2 predictor pays over LOCF.
  Conditions: k in {1, 4, 8} x predictor {off, on}; 6 seeds x 200 rounds.
  Extra observations simulated by calling planner.ingest_observation(e) for
  k-1 additional edges chosen by max-age among ALL edges (reporting network).

Experiment B — decision-uniform mode on bounded-drift synthetic world.
  6x6 grid, rho=0.01, epsilon=5, alpha'=0.2, eps_tv=1e-4; 12 seeds x 300 rounds.
  decision_uniform {off, on} (max_decisions=5).
  Reports cert%, valid%, coverage, gap, claimed confidence, AND episode-level
  fraction where every acted-on certificate was valid (T6 quantity).

Run: cert_env/bin/python scripts/run_feature_regimes.py [--quick]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from certflow.cert import CertPlanner
from certflow.drift import BoundedDriftWorld, _make_rng
from certflow.graphcore import dijkstra
from certflow.realworld import (
    TrafficWorld,
    far_endpoints,
    fit_spatial_predictor,
    traffic_planner_config,
)
from certflow.sensing import path_edges

QUICK = "--quick" in sys.argv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cp_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a, k + 1, n - k))
    return lo, hi


def traffic_true_opt(world: TrafficWorld, t: float, s, g) -> float:
    snap = {u: {v: world.true_cost((u, v), t) for v in nbrs}
            for u, nbrs in world.graph.items()}
    _, cost = dijkstra(snap, s, g)
    return cost


# ---------------------------------------------------------------------------
# Experiment A
# ---------------------------------------------------------------------------

def run_experiment_a() -> list[dict]:
    SEEDS = 3 if QUICK else 6
    ROUNDS = 50 if QUICK else 200
    K_VALUES = [1, 4, 8]

    print("=== Experiment A: spatial predictor (METR-LA, dense-sensing regime) ===",
          flush=True)

    # Fit predictor once (train on first 18000 bins; val windows offset past that)
    print("  Fitting spatial predictor...", flush=True)
    predictor = fit_spatial_predictor("metr-la", train_bins=18000,
                                      fresh_age=6 * 300.0)
    print("  Predictor fit done.", flush=True)

    rows = []
    for k in K_VALUES:
        for pred_on in [False, True]:
            cov_covered = cov_valid = 0
            pred_used_total = 0
            gaps: list[float] = []
            conf_sum = 0.0
            n_valid = 0

            for seed in range(SEEDS):
                # offset_base_bins=20000 keeps val window past the predictor's training
                w = TrafficWorld("metr-la", seed=seed, n_bins=ROUNDS,
                                 offset_base_bins=20000)
                s, g = far_endpoints(w)
                cfg = traffic_planner_config(rho_mode="online",
                                             max_sense_per_round=k,
                                             adaptive_rate=True)
                p = CertPlanner(w, s, g, cfg,
                                predictor=predictor if pred_on else None)

                # Sorted edge list for max-age lookups (stable across rounds)
                all_edges = list(w.edges())

                for _ in range(ROUNDS):
                    t_round = p.t

                    # One standard round (includes 1 observation via ingest inside)
                    cert, sensed = p.round()

                    # Additional k-1 observations: max-age edges from ALL edges
                    # (simulates a dense reporting sensor network)
                    if k > 1:
                        # Sort by age descending at current time (post-round t)
                        ages = [(p.t - p.beliefs[e].t_obs, e) for e in all_edges]
                        ages.sort(reverse=True)
                        extra_count = 0
                        seen_this_round = {sensed} if sensed is not None else set()
                        for _, e in ages:
                            if extra_count >= k - 1:
                                break
                            if e in seen_this_round:
                                continue
                            p.ingest_observation(e)
                            seen_this_round.add(e)
                            extra_count += 1

                    if cert.valid:
                        n_valid += 1
                        o = traffic_true_opt(w, t_round, s, g)
                        if cert.lb - 1e-9 <= o <= cert.ub + 1e-9:
                            cov_covered += 1
                        cov_valid += 1
                        conf_sum += cert.confidence
                        gaps.append(cert.gap)

                pred_used_total += p.pred_used_rounds

            label = f"k={k}, pred={'on' if pred_on else 'off'}"
            coverage = cov_covered / cov_valid if cov_valid else float("nan")
            gap_med = float(np.median(gaps)) if gaps else float("nan")
            mean_conf = conf_sum / n_valid if n_valid else float("nan")
            avg_pred = pred_used_total / SEEDS

            rows.append(dict(
                label=label,
                k=k,
                pred_on=pred_on,
                n_valid=n_valid,
                coverage=coverage,
                gap_median=gap_med,
                mean_confidence=mean_conf,
                pred_used_rounds=avg_pred,
            ))
            print(f"  {label}: valid={n_valid}, cov={coverage:.3f}, "
                  f"gap~={gap_med:.1f}s, conf={mean_conf:.3f}, "
                  f"pred_rounds={avg_pred:.1f}", flush=True)

    return rows


# ---------------------------------------------------------------------------
# Experiment B
# ---------------------------------------------------------------------------

def run_experiment_b() -> list[dict]:
    SEEDS = 6 if QUICK else 12
    ROUNDS = 100 if QUICK else 300
    ROWS_B, COLS_B = 6, 6

    print("\n=== Experiment B: decision-uniform mode (6x6 bounded drift) ===",
          flush=True)

    rows = []
    for du_on in [False, True]:
        # episode-level metric: fraction where EVERY acted-on cert was valid
        all_episode_all_valid: list[bool] = []

        cov_covered = cov_valid = cert_count = rounds_total = 0
        gaps: list[float] = []
        conf_sum = 0.0

        for seed in range(SEEDS):
            rng = _make_rng(seed + 9000)
            w = BoundedDriftWorld(ROWS_B, COLS_B, rng,
                                  rho=0.01, noise_scale=0.05)

            # start=(0,0), goal=(5,5) for a 6x6 grid
            start = (0, 0)
            goal = (ROWS_B - 1, COLS_B - 1)

            from certflow.cert import PlannerConfig
            cfg = PlannerConfig(
                epsilon=5.0,
                alpha_prime=0.2,
                eps_tv=1e-4,
                rho_w=0.99,
                delta=1.0,
                decision_uniform=du_on,
                max_decisions=5,
            )
            p = CertPlanner(w, start, goal, cfg)

            episode_acted_certs: list[bool] = []  # per acted-on cert: was it valid?

            for _ in range(ROUNDS):
                t_round = p.t
                cert, _ = p.round()
                rounds_total += 1

                # "acted on" = certified: gap <= epsilon AND confidence >= min_certify_confidence
                is_acted_on = (
                    cert.valid
                    and cert.gap <= cfg.epsilon
                    and cert.confidence >= cfg.min_certify_confidence
                )

                if cert.valid:
                    cov_valid += 1
                    snap = {u: {v: w.true_cost((u, v), t_round) for v in nbrs}
                            for u, nbrs in w.graph.items()}
                    _, opt = dijkstra(snap, start, goal)
                    covered = cert.lb - 1e-9 <= opt <= cert.ub + 1e-9
                    if covered:
                        cov_covered += 1
                    conf_sum += cert.confidence
                    gaps.append(cert.gap)

                    if is_acted_on:
                        cert_count += 1
                        episode_acted_certs.append(covered)

            # episode-level: all acted-on certs valid?
            if episode_acted_certs:
                all_episode_all_valid.append(all(episode_acted_certs))
            else:
                # no certifications at all: vacuously "all valid"
                all_episode_all_valid.append(True)

        coverage = cov_covered / cov_valid if cov_valid else float("nan")
        valid_pct = cov_valid / rounds_total if rounds_total else 0.0
        cert_pct = cert_count / rounds_total if rounds_total else 0.0
        gap_med = float(np.median(gaps)) if gaps else float("nan")
        mean_conf = conf_sum / cov_valid if cov_valid else float("nan")
        ep_all_valid = (sum(all_episode_all_valid) / len(all_episode_all_valid)
                        if all_episode_all_valid else float("nan"))

        label = f"decision_uniform={'on' if du_on else 'off'}"
        rows.append(dict(
            label=label,
            du_on=du_on,
            valid_pct=valid_pct,
            coverage=coverage,
            cert_pct=cert_pct,
            gap_median=gap_med,
            mean_confidence=mean_conf,
            episode_all_valid_frac=ep_all_valid,
        ))
        print(f"  {label}: valid%={100*valid_pct:.1f}%, cov={coverage:.3f}, "
              f"cert%={100*cert_pct:.1f}%, gap~={gap_med:.2f}, "
              f"conf={mean_conf:.3f}, ep_all_valid={ep_all_valid:.3f}", flush=True)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows_a = run_experiment_a()
    rows_b = run_experiment_b()

    # Print summary tables
    print("\n\n=== TABLE A: Spatial predictor — dense-sensing regime (METR-LA) ===")
    hdr = f"{'condition':28} {'valid':>6} {'coverage':>9} {'gap~ (s)':>9} {'conf':>6} {'pred_rounds':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows_a:
        print(f"{r['label']:28} {r['n_valid']:>6} {r['coverage']:>9.3f} "
              f"{r['gap_median']:>9.1f} {r['mean_confidence']:>6.3f} "
              f"{r['pred_used_rounds']:>12.1f}")

    print("\n=== TABLE B: Decision-uniform mode (6x6 bounded drift, rho=0.01) ===")
    hdr = f"{'condition':30} {'valid%':>7} {'coverage':>9} {'cert%':>6} {'gap~':>7} {'conf':>6} {'ep_all_valid':>13}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows_b:
        print(f"{r['label']:30} {100*r['valid_pct']:>6.1f}% {r['coverage']:>9.3f} "
              f"{100*r['cert_pct']:>5.1f}% {r['gap_median']:>7.2f} "
              f"{r['mean_confidence']:>6.3f} {r['episode_all_valid_frac']:>13.3f}")

    # Persist results
    import json
    outdir = Path("results/feature_regimes")
    outdir.mkdir(parents=True, exist_ok=True)

    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def _clean_row(row):
        return {k: _clean(v) for k, v in row.items()}

    (outdir / "exp_a.json").write_text(
        json.dumps({"rows": [_clean_row(r) for r in rows_a]}, indent=2))
    (outdir / "exp_b.json").write_text(
        json.dumps({"rows": [_clean_row(r) for r in rows_b]}, indent=2))
    print(f"\nResults saved to {outdir}/", flush=True)


if __name__ == "__main__":
    main()
