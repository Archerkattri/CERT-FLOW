"""Age-matched predictor retraining: does training on the deployment
distribution recover the spatial gain the fresh-at-t training threw away?

Two parts:

  OFFLINE MECHANISM CHECK
    Train the fresh-trained P2 (neighbor mean AT t) and the age-matched P2
    (neighbor mean at t-b, with b as a feature) on the first train_bins of
    METR-LA, then evaluate residual-P90 on the held-out region with neighbors
    observed at deployment-realistic ages b in {6,12} bins. If age-matching is
    the right mechanism, its residual P90 at b>0 must beat the fresh-trained
    model's (which has never seen a stale neighbor). Confirmed BEFORE the
    planner runs so a null planner result can be diagnosed.

  PLANNER TABLE (feature-regimes Experiment A, replicated exactly)
    k in {1,4,8} x predictor {off, fresh-trained, age-matched}; 6 seeds x 200
    rounds; same TrafficWorld windows (offset_base_bins=20000); same metrics
    (coverage among valid, gap median, mean confidence, pred_used_rounds).
    Extra k-1 observations per round = max-age edges over ALL edges (reporting
    sensor network), identical to scripts/run_feature_regimes.py.

Run: cert_env/bin/python scripts/run_predictor_retrain.py [--quick]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from certflow.cert import CertPlanner
from certflow.graphcore import dijkstra
from certflow.realworld import (
    BIN_SECONDS,
    MPH_TO_MPS,
    TrafficWorld,
    _load_traffic,
    far_endpoints,
    fit_spatial_predictor,
    traffic_planner_config,
)

QUICK = "--quick" in sys.argv

DATASET = "metr-la"
TRAIN_BINS = 18000
MAX_DIST_M = 3000.0
RIDGE = 10.0
AGES = (12, 24, 48)              # target staleness, bins
NBR_AGE_BINS = (0, 6, 12)        # neighbor-age training ladder, bins
B_DEPLOY = (6, 12)               # deployment-realistic neighbor ages to test


def traffic_true_opt(world: TrafficWorld, t: float, s, g) -> float:
    snap = {u: {v: world.true_cost((u, v), t) for v in nbrs}
            for u, nbrs in world.graph.items()}
    _, cost = dijkstra(snap, s, g)
    return cost


# ---------------------------------------------------------------------------
# Offline mechanism check (residual-P90 in SPEED units, held-out region)
# ---------------------------------------------------------------------------

def run_offline_mechanism() -> list[dict]:
    """Fit fresh-trained and age-matched per-sensor ridges on [:TRAIN_BINS],
    evaluate residual P90 on the held-out region with neighbors at age b."""
    print("=== OFFLINE mechanism check: residual-P90 at stale neighbor ages ===",
          flush=True)
    ids, speeds, dist = _load_traffic(DATASET)
    idx = {s: i for i, s in enumerate(ids)}
    nbrs: dict[str, list[int]] = {s: [] for s in ids}
    for (u, v), m in dist.items():
        if 0.0 < m <= MAX_DIST_M and u in idx and v in idx:
            nbrs[u].append(idx[v])

    tr = speeds[:TRAIN_BINS]
    ev = speeds[TRAIN_BINS:]            # held-out region (no leakage)
    rng = np.random.default_rng(0)

    # Fit both predictors per sensor.
    fresh_w: dict[str, np.ndarray] = {}     # [own, nbr@t, intercept]
    am_w: dict[str, np.ndarray] = {}        # [own, nbr@t-b, b, intercept]
    for s_id in ids:
        js = nbrs[s_id]
        if not js:
            continue
        i = idx[s_id]
        # fresh-trained: neighbor mean AT t
        ts = rng.integers(max(AGES), TRAIN_BINS - 1, size=400)
        Xf, yf = [], []
        for t in ts:
            a = int(rng.choice(AGES))
            Xf.append([tr[t - a, i], tr[t, js].mean()])
            yf.append(tr[t, i])
        Xf = np.hstack([np.asarray(Xf), np.ones((len(Xf), 1))])
        fresh_w[s_id] = np.linalg.solve(
            Xf.T @ Xf + RIDGE * np.eye(3), Xf.T @ np.asarray(yf))
        # age-matched: neighbor mean at t-b with b as a feature
        ts2 = rng.integers(max(max(AGES), max(NBR_AGE_BINS)), TRAIN_BINS - 1,
                           size=400)
        Xa, ya = [], []
        for t in ts2:
            a = int(rng.choice(AGES))
            b = int(rng.choice(NBR_AGE_BINS))
            Xa.append([tr[t - a, i], tr[t - b, js].mean(), float(b)])
            ya.append(tr[t, i])
        Xa = np.hstack([np.asarray(Xa), np.ones((len(Xa), 1))])
        am_w[s_id] = np.linalg.solve(
            Xa.T @ Xa + RIDGE * np.eye(4), Xa.T @ np.asarray(ya))

    # Evaluate on held-out region: for each deployment neighbor age b, the
    # neighbor mean is read at age b (semi-stale, as deployed). Stale own value
    # at age a (averaged over the operational band). Residual = |pred - truth|.
    rows = []
    eval_ts = np.arange(max(max(AGES), max(NBR_AGE_BINS)), len(ev) - 1,
                        4)  # every 4th bin, matching the offline study stride
    for b in B_DEPLOY:
        fresh_res, am_res = [], []
        for s_id in ids:
            js = nbrs[s_id]
            if not js:
                continue
            i = idx[s_id]
            fw = fresh_w[s_id]
            aw = am_w[s_id]
            for a in AGES:
                tt = eval_ts[eval_ts >= max(a, b)]
                own = ev[tt - a, i]
                nbr = ev[tt - b][:, js].mean(axis=1)   # neighbor mean AT t-b
                truth = ev[tt, i]
                # fresh-trained model fed a stale neighbor mean (the mismatch)
                pf = fw[0] * own + fw[1] * nbr + fw[2]
                pf = np.clip(pf, 3.0, 75.0)
                # age-matched model: same stale neighbor, told its age b
                pa = aw[0] * own + aw[1] * nbr + aw[2] * float(b) + aw[3]
                pa = np.clip(pa, 3.0, 75.0)
                fresh_res.append(np.abs(pf - truth))
                am_res.append(np.abs(pa - truth))
        fresh_res = np.concatenate(fresh_res)
        am_res = np.concatenate(am_res)
        fp90 = float(np.quantile(fresh_res, 0.90))
        ap90 = float(np.quantile(am_res, 0.90))
        ratio = ap90 / fp90 if fp90 else float("nan")
        rows.append(dict(b=b, fresh_p90=fp90, am_p90=ap90, ratio=ratio,
                         n=int(len(fresh_res))))
        print(f"  b={b:>2} bins: fresh-trained P90={fp90:.3f} mph  "
              f"age-matched P90={ap90:.3f} mph  am/fresh={ratio:.3f}",
              flush=True)
    return rows


# ---------------------------------------------------------------------------
# Planner table (feature-regimes Experiment A, replicated exactly)
# ---------------------------------------------------------------------------

def run_planner_table() -> list[dict]:
    SEEDS = 3 if QUICK else 6
    ROUNDS = 50 if QUICK else 200
    K_VALUES = [1, 4, 8]

    print("\n=== PLANNER: feature-regimes Exp A "
          "(off / fresh-trained / age-matched) ===", flush=True)
    print("  Fitting predictors...", flush=True)
    pred_fresh = fit_spatial_predictor(DATASET, train_bins=TRAIN_BINS,
                                       fresh_age=6 * BIN_SECONDS)
    pred_am = fit_spatial_predictor(DATASET, train_bins=TRAIN_BINS,
                                    fresh_age=6 * BIN_SECONDS,
                                    age_matched=True)
    print("  Predictor fits done.", flush=True)

    modes = [("off", None), ("fresh", pred_fresh), ("age-matched", pred_am)]

    rows = []
    for k in K_VALUES:
        for mode_name, predictor in modes:
            cov_covered = cov_valid = 0
            pred_used_total = 0
            gaps: list[float] = []
            conf_sum = 0.0
            n_valid = 0

            for seed in range(SEEDS):
                w = TrafficWorld(DATASET, seed=seed, n_bins=ROUNDS,
                                 offset_base_bins=20000)
                s, g = far_endpoints(w)
                cfg = traffic_planner_config(rho_mode="online",
                                             max_sense_per_round=k,
                                             adaptive_rate=True)
                p = CertPlanner(w, s, g, cfg, predictor=predictor)

                all_edges = list(w.edges())

                for _ in range(ROUNDS):
                    t_round = p.t
                    cert, sensed = p.round()

                    if k > 1:
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

            label = f"k={k}, pred={mode_name}"
            coverage = cov_covered / cov_valid if cov_valid else float("nan")
            gap_med = float(np.median(gaps)) if gaps else float("nan")
            mean_conf = conf_sum / n_valid if n_valid else float("nan")
            avg_pred = pred_used_total / SEEDS

            rows.append(dict(
                label=label,
                k=k,
                pred=mode_name,
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


def main() -> None:
    off_rows = run_offline_mechanism()
    plan_rows = run_planner_table()

    print("\n\n=== OFFLINE mechanism table (residual-P90, held-out, mph) ===")
    hdr = f"{'nbr age b (bins)':16} {'fresh-trained P90':>18} {'age-matched P90':>17} {'am/fresh':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in off_rows:
        print(f"{r['b']:>16} {r['fresh_p90']:>18.3f} {r['am_p90']:>17.3f} "
              f"{r['ratio']:>9.3f}")

    print("\n=== PLANNER table A (off / fresh-trained / age-matched, METR-LA) ===")
    hdr = f"{'condition':22} {'valid':>6} {'coverage':>9} {'gap~ (s)':>9} {'conf':>6} {'pred_rounds':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in plan_rows:
        print(f"{r['label']:22} {r['n_valid']:>6} {r['coverage']:>9.3f} "
              f"{r['gap_median']:>9.1f} {r['mean_confidence']:>6.3f} "
              f"{r['pred_used_rounds']:>12.1f}")

    outdir = Path("results/predictor_retrain")
    outdir.mkdir(parents=True, exist_ok=True)

    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def _clean_row(row):
        return {k: _clean(v) for k, v in row.items()}

    (outdir / "offline.json").write_text(
        json.dumps({"rows": [_clean_row(r) for r in off_rows]}, indent=2))
    (outdir / "planner.json").write_text(
        json.dumps({"rows": [_clean_row(r) for r in plan_rows]}, indent=2))
    print(f"\nResults saved to {outdir}/", flush=True)


if __name__ == "__main__":
    main()
