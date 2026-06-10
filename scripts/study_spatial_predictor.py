"""Offline study: how much does SPATIALLY CORRELATED prediction tighten
conformal intervals on METR-LA / PEMS-BAY speeds?

CERT predicts each edge's current cost as last-observation-carried-forward
(LOCF). The conformal layer calibrates residuals of *that* predictor, so a
better point predictor directly tightens intervals at unchanged coverage.
This script quantifies the gain OFFLINE, with no planner involved.

We predict a stale sensor's CURRENT speed at age `a` bins, comparing:
  P0 last-value:        speed_i(t-a)                      (what CERT uses)
  P1 neighbor-delta:    speed_i(t-a) + mean_j[speed_j(t)-speed_j(t-a)]
  P2 neighbor-regress:  ridge( [speed_i(t-a), mean_j speed_j(t)] -> speed_i(t) )
Fresh neighbors are assumed observed AT time t (the planner's sensing makes
some edges fresh while others are stale) -- the realistic upper bound on the
spatial gain. A half-fresh sensitivity (only half the neighbors fresh) is
also reported.

Metric that matters: residual P90 (drives the conformal quantile at our
levels). Headline = interval-width ratio P1/P0 and P2/P0 at P90.

Run: cert_env/bin/python scripts/study_spatial_predictor.py
Pure numpy/pandas; sampled every 4th time bin for tractability.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from certflow.realworld import _load_traffic

AGES = [1, 3, 6, 12, 24, 48]          # bins of staleness (5 min each)
MAX_DIST_M = 3000.0
TIME_STRIDE = 4                       # evaluate every 4th time bin
TRAIN_FRAC = 0.60                     # first 60% fits P2; last 40% evaluated
RIDGE_LAMBDA = 1.0
P90 = 0.90
RNG = np.random.default_rng(0)


def build_neighbors(ids, dist):
    """Undirected adjacency: j is a neighbor of i if road dist (either
    direction) <= MAX_DIST_M. Returns list-of-arrays of column indices."""
    idx = {s: i for i, s in enumerate(ids)}
    nb = [set() for _ in ids]
    for (u, v), m in dist.items():
        if 0.0 < m <= MAX_DIST_M and u in idx and v in idx:
            iu, iv = idx[u], idx[v]
            nb[iu].add(iv)
            nb[iv].add(iu)
    return [np.array(sorted(s), dtype=int) for s in nb]


def ridge_fit(X, y, lam):
    """Closed-form ridge with intercept; X is (n, k). Returns (k+1,) weights."""
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    k = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(k)
    A[0, 0] -= lam  # do not penalize intercept
    return np.linalg.solve(A, Xb.T @ y)


def neighbor_fresh_mean(speeds, neighbors, t_idx, fresh_mask=None):
    """For every sensor, mean of fresh neighbors' speeds at the given time
    indices. Returns (n_times, n_sensors); NaN where a sensor has no fresh
    neighbor. fresh_mask (n_sensors_bool over neighbor *membership*) selects a
    fixed random subset of each sensor's neighbors as 'fresh'."""
    n_t = len(t_idx)
    n_s = speeds.shape[1]
    out = np.full((n_t, n_s), np.nan)
    snap = speeds[t_idx]  # (n_t, n_s)
    for s in range(n_s):
        nb = neighbors[s]
        if fresh_mask is not None:
            nb = nb[fresh_mask[s]]
        if len(nb) == 0:
            continue
        out[:, s] = snap[:, nb].mean(axis=1)
    return out


def half_fresh_masks(neighbors, rng):
    masks = []
    for nb in neighbors:
        if len(nb) == 0:
            masks.append(np.zeros(0, dtype=bool))
            continue
        m = np.zeros(len(nb), dtype=bool)
        keep = max(1, len(nb) // 2) if len(nb) >= 2 else 0
        if keep:
            m[rng.choice(len(nb), size=keep, replace=False)] = True
        masks.append(m)
    return masks


def evaluate(speeds, neighbors, fresh_mask=None):
    """Returns dict[age] -> dict of per-predictor residual arrays, split into
    'connected' (sensor has >=1 fresh neighbor) and 'isolated'."""
    n_bins, n_s = speeds.shape
    max_age = max(AGES)
    # train/eval split over time
    split = int(n_bins * TRAIN_FRAC)
    results = {}

    for a in AGES:
        # valid current-time indices: need t-a >= 0 and t < n_bins
        # eval set: t in [max(split, a), n_bins), sampled
        t_all = np.arange(max(split, a), n_bins, TIME_STRIDE)
        # P2 train set: t in [a, split), sampled
        t_train = np.arange(a, split, TIME_STRIDE)

        y_eval = speeds[t_all]                      # (T, S) current speed
        x0_eval = speeds[t_all - a]                 # last value
        nbfresh_eval = neighbor_fresh_mean(speeds, neighbors, t_all, fresh_mask)

        y_train = speeds[t_train]
        x0_train = speeds[t_train - a]
        nbfresh_train = neighbor_fresh_mean(speeds, neighbors, t_train, fresh_mask)

        # P1 neighbor-delta: speed_i(t-a) + mean_j[speed_j(t) - speed_j(t-a)]
        nbpast_eval = neighbor_fresh_mean(speeds, neighbors, t_all - a, fresh_mask)
        delta = nbfresh_eval - nbpast_eval          # (T, S)

        # which (sensor) columns are connected (have any fresh neighbor)
        has_nb = np.array([len(neighbors[s]) > 0 if fresh_mask is None
                           else fresh_mask[s].any() for s in range(n_s)])

        res_p0 = []
        res_p1 = []
        res_p2 = []
        conn_flag = []  # per residual: 1 if from a connected sensor
        for s in range(n_s):
            ye = y_eval[:, s]
            p0 = x0_eval[:, s]
            r0 = ye - p0
            res_p0.append(r0)
            if has_nb[s]:
                p1 = p0 + delta[:, s]
                # fall back to P0 where a delta is NaN (no fresh neighbor that step)
                p1 = np.where(np.isfinite(p1), p1, p0)
                res_p1.append(ye - p1)
                # P2 ridge: fit on train, predict on eval
                nf_tr = nbfresh_train[:, s]
                nf_ev = nbfresh_eval[:, s]
                tr_ok = np.isfinite(nf_tr)
                if tr_ok.sum() >= 10:
                    Xtr = np.column_stack([x0_train[tr_ok, s], nf_tr[tr_ok]])
                    w = ridge_fit(Xtr, y_train[tr_ok, s], RIDGE_LAMBDA)
                    nf_ev_f = np.where(np.isfinite(nf_ev), nf_ev, p0)
                    Xev = np.column_stack([p0, nf_ev_f])
                    p2 = w[0] + Xev @ w[1:]
                    res_p2.append(ye - p2)
                else:
                    res_p2.append(ye - p0)
                conn_flag.append(np.ones_like(r0))
            else:
                conn_flag.append(np.zeros_like(r0))

        res_p0 = np.concatenate(res_p0)
        conn_flag = np.concatenate(conn_flag).astype(bool)
        res_p1 = np.concatenate(res_p1) if res_p1 else np.zeros(0)
        res_p2 = np.concatenate(res_p2) if res_p2 else np.zeros(0)

        results[a] = dict(
            p0_all=res_p0,
            p0_conn=res_p0[conn_flag],
            p0_iso=res_p0[~conn_flag],
            p1_conn=res_p1,
            p2_conn=res_p2,
        )
    return results


def stats(res):
    """MAE and P90 of |residual| (P90 is the symmetric conformal quantile)."""
    if len(res) == 0:
        return dict(mae=float("nan"), p90=float("nan"), n=0)
    a = np.abs(res)
    return dict(mae=float(a.mean()), p90=float(np.quantile(a, P90)), n=int(len(res)))


def run_dataset(ds, half=False):
    ids, speeds, dist = _load_traffic(ds)
    neighbors = build_neighbors(ids, dist)
    n_s = len(ids)
    deg = np.array([len(nb) for nb in neighbors])
    frac_conn = float((deg >= 1).mean())

    fresh_mask = half_fresh_masks(neighbors, RNG) if half else None
    res = evaluate(speeds, neighbors, fresh_mask)

    table = {}
    for a in AGES:
        r = res[a]
        s0c = stats(r["p0_conn"])
        s1 = stats(r["p1_conn"])
        s2 = stats(r["p2_conn"])
        s0i = stats(r["p0_iso"])
        table[a] = dict(
            p0_conn=s0c, p1_conn=s1, p2_conn=s2, p0_iso=s0i,
            ratio_p1_p90=(s1["p90"] / s0c["p90"]) if s0c["p90"] else float("nan"),
            ratio_p2_p90=(s2["p90"] / s0c["p90"]) if s0c["p90"] else float("nan"),
            improve_p1=(1 - s1["p90"] / s0c["p90"]) if s0c["p90"] else float("nan"),
            improve_p2=(1 - s2["p90"] / s0c["p90"]) if s0c["p90"] else float("nan"),
        )
    return dict(
        dataset=ds, n_sensors=n_s, frac_connected=frac_conn,
        n_isolated=int((deg == 0).sum()), mean_degree=float(deg.mean()),
        table=table,
    )


def fmt_table(out):
    lines = []
    hdr = (f"{'age':>4} {'P0_p90':>8} {'P1_p90':>8} {'P2_p90':>8} "
           f"{'P1/P0':>7} {'P2/P0':>7} {'P0_MAE':>7} {'P1_MAE':>7} {'P2_MAE':>7} "
           f"{'iso_p90':>8} {'iso_MAE':>8}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for a in AGES:
        r = out["table"][a]
        lines.append(
            f"{a:>4} {r['p0_conn']['p90']:>8.3f} {r['p1_conn']['p90']:>8.3f} "
            f"{r['p2_conn']['p90']:>8.3f} {r['ratio_p1_p90']:>7.3f} "
            f"{r['ratio_p2_p90']:>7.3f} {r['p0_conn']['mae']:>7.3f} "
            f"{r['p1_conn']['mae']:>7.3f} {r['p2_conn']['mae']:>7.3f} "
            f"{r['p0_iso']['p90']:>8.3f} {r['p0_iso']['mae']:>8.3f}"
        )
    return "\n".join(lines)


def main():
    datasets = ["metr-la", "pems-bay"]
    full = {}
    for ds in datasets:
        full[ds] = dict(fresh=run_dataset(ds, half=False),
                        half=run_dataset(ds, half=True))
        o = full[ds]["fresh"]
        print(f"\n=== {ds} (all fresh neighbors) ===")
        print(f"sensors {o['n_sensors']}  frac connected {o['frac_connected']:.3f}  "
              f"isolated {o['n_isolated']}  mean degree {o['mean_degree']:.1f}")
        print(fmt_table(o))
        print(f"\n--- {ds} (HALF fresh neighbors) ---")
        print(fmt_table(full[ds]["half"]))

    outdir = Path("results/spatial_predictor")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "study.json").write_text(json.dumps(full, indent=2))
    print(f"\nwrote {outdir/'study.json'}")
    return full


if __name__ == "__main__":
    main()
