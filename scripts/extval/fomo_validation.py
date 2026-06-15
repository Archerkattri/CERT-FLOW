"""FoMo off-road validation: does CERT's certificate hold on REAL seasonal
drifting traversal cost?

Foret Montmorency (norlab), 6 route-colors re-traversed across up to 12
deployments over a year (winter snow -> summer vegetation). For each route we
derive a per-segment traversal COST from the ground-truth GNSS trace (gt.txt,
TUM, projected metres) and the battery power log (pack_voltage * pack_current),
then treat each deployment as an observation of that segment's cost. The
seasonal change IS the drift. We run CERT's exact conformal machinery
(ConformalScorer + the drift-adjusted score, the same code the paper uses) and
measure coverage of the per-segment interval and of the Bonferroni path bound
on held-out (staler) deployments.

Honest disclosures (printed in the report):
- Segments are aligned by NORMALISED ARC-LENGTH along each deployment's own
  trace; the colours are re-traversals of the same physical trail, so
  fraction-of-route approximates the same location across deployments. This is
  a modelling choice, stated, not a ground-truth segment correspondence.
- Cost = traversal TIME per segment (primary) and ENERGY = integral of
  |pack_voltage * pack_current| dt (secondary), both from real logs.
- No perception, no hardware, no robot. Replay only.

Run: cert_env/bin/python scripts/extval/fomo_validation.py
"""
from __future__ import annotations

import collections
import math
import pathlib

import numpy as np

from certflow.conformal import ConformalScorer, path_alpha_edge

FOMO = pathlib.Path("data/fomo")
N_SEG = 12            # segments (edges) per route
ALPHA = 0.1           # target miscoverage (90% intervals)
MIN_DEPLOY = 4        # need at least this many deployments of a colour


def _load_poses(gt_path: pathlib.Path):
    """TUM gt.txt -> (t[s], x[m], y[m]); empty on failure."""
    try:
        a = np.loadtxt(gt_path)
        if a.ndim != 2 or a.shape[0] < 5:
            return None
        return a[:, 0], a[:, 1], a[:, 2]
    except Exception:
        return None


def _load_power(meta_dir: pathlib.Path):
    """battery_logs.csv -> (t[s], power[W]); |V*I| (discharge magnitude)."""
    p = meta_dir / "battery_logs.csv"
    if not p.exists():
        return None
    try:
        import csv
        ts, pw = [], []
        with open(p) as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    t = float(row["timestamp"]) / 1e6   # us -> s
                    v = float(row["pack_voltage"]); i = float(row["pack_current"])
                    ts.append(t); pw.append(abs(v * i))
                except Exception:
                    continue
        if len(ts) < 5:
            return None
        return np.array(ts), np.array(pw)
    except Exception:
        return None


def _segment_costs(t, x, y, power):
    """Per-segment traversal time and energy, binned by normalised arc-length."""
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    total = d[-1]
    if total < 1.0:                       # degenerate / stationary trace
        return None
    frac = d / total
    edges = np.linspace(0.0, 1.0, N_SEG + 1)
    seg = np.clip(np.searchsorted(edges, frac, side="right") - 1, 0, N_SEG - 1)
    time_cost = np.zeros(N_SEG)
    energy_cost = np.zeros(N_SEG)
    # integrate per pose-interval into its segment
    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        if dt <= 0 or dt > 30:            # skip gaps/outliers
            continue
        s = seg[k]
        time_cost[s] += dt
        if power is not None:
            # mean power over [t_k, t_k+1] * dt  (nearest-sample)
            j = np.searchsorted(power[0], t[k])
            j = min(max(j, 0), len(power[1]) - 1)
            energy_cost[s] += power[1][j] * dt
    if (time_cost <= 0).any():
        return None                        # incomplete segment coverage
    return time_cost, energy_cost


def build_routes():
    """colour -> list of (date, time_cost[N_SEG], energy_cost[N_SEG]) sorted by date."""
    routes: dict[str, list] = collections.defaultdict(list)
    if not FOMO.exists():
        return routes
    for date_dir in sorted(p for p in FOMO.iterdir() if p.is_dir()):
        for traj in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            colour = traj.name.split("_")[0]
            poses = _load_poses(traj / "gt.txt")
            if poses is None:
                continue
            power = _load_power(traj / "metadata")
            sc = _segment_costs(*poses, power)
            if sc is None:
                continue
            routes[colour].append((date_dir.name, sc[0], sc[1]))
    return {c: v for c, v in routes.items() if len(v) >= MIN_DEPLOY}


def coverage_experiment(routes, cost_idx=0, label="time"):
    """CERT conformal coverage on real seasonal drift.

    For each route, walk deployments in date order as repeated observations of
    each segment. The drift-adjusted score is |obs - prev_obs| - rho*age, where
    age is deployments-elapsed (seasonal steps). Calibrate the weighted quantile
    over all segment scores seen so far; for each new deployment predict
    interval c_hat_prev +/- (q + rho*age) per segment and check coverage of the
    realised cost. Also check the Bonferroni PATH bound over the whole route.
    """
    # estimate a global per-step drift bound rho from realised |delta|/age
    deltas = []
    for v in routes.values():
        arr = np.array([d[1 + cost_idx] for d in v])   # (D, N_SEG)
        deltas.append(np.abs(np.diff(arr, axis=0)))
    rho = float(np.quantile(np.concatenate([d.ravel() for d in deltas]), 0.75))

    edge_cov = edge_tot = 0
    path_cov = path_tot = 0
    a1_viol = a1_tot = 0
    widths = []
    scorer = ConformalScorer(rho_w=0.97, eps_tv=0.0)     # ESS ceiling ~33
    gt = 0.0                                              # monotonic deployment-step time
    for v in routes.values():
        arr = np.array([d[1 + cost_idx] for d in v])    # (D, N_SEG) date-ordered
        D = arr.shape[0]
        for d in range(1, D):
            gt += 1.0
            age = 1.0                                     # one seasonal step
            # EDGE intervals at the marginal level alpha (90%); PATH bound at
            # the Bonferroni per-edge level alpha/L (needs much more mass).
            q_edge = scorer.quantile(ALPHA, gt) if len(scorer) > 10 else math.inf
            q_path = scorer.quantile(path_alpha_edge(ALPHA, N_SEG), gt) \
                if len(scorer) > N_SEG else math.inf
            lb_sum = ub_sum = 0.0
            for s in range(N_SEG):
                pred = arr[d - 1, s]; true = arr[d, s]
                if math.isfinite(q_edge):
                    half = q_edge + rho * age
                    lo, hi = max(pred - half, 0.0), pred + half
                    edge_tot += 1
                    edge_cov += (lo - 1e-9 <= true <= hi + 1e-9)
                    widths.append(hi - lo)
                if math.isfinite(q_path):
                    hp = q_path + rho * age
                    lb_sum += max(pred - hp, 0.0); ub_sum += pred + hp
                # A1 check: realised drift vs the assumed rho bound
                a1_tot += 1
                a1_viol += (abs(true - pred) > rho * age + 1e-9)
                scorer.push(abs(true - pred) - rho * age, gt)
            if math.isfinite(q_path):
                true_path = float(arr[d].sum())
                path_tot += 1
                path_cov += (lb_sum - 1e-9 <= true_path <= ub_sum + 1e-9)
    return dict(
        label=label, rho=rho, n_routes=len(routes),
        edge_coverage=edge_cov / max(edge_tot, 1), edge_n=edge_tot,
        path_coverage=path_cov / max(path_tot, 1), path_n=path_tot,
        a1_violation=a1_viol / max(a1_tot, 1),
        median_width=float(np.median(widths)) if widths else float("nan"))


def main():
    routes = build_routes()
    if not routes:
        print("No FoMo routes found under data/fomo/ (download incomplete?).")
        return
    print(f"FoMo routes (>= {MIN_DEPLOY} deployments): "
          + ", ".join(f"{c}({len(v)})" for c, v in routes.items()))
    for idx, lab in [(0, "traversal-time"), (1, "energy")]:
        r = coverage_experiment(routes, idx, lab)
        print(f"\n[{lab}] rho(p75 per-step)={r['rho']:.3g} over {r['n_routes']} routes")
        print(f"  edge coverage : {r['edge_coverage']:.3f}  (n={r['edge_n']}, "
              f"target {1-ALPHA:.2f})")
        print(f"  path coverage : {r['path_coverage']:.3f}  (n={r['path_n']})")
        print(f"  A1-violation  : {r['a1_violation']:.3f}  "
              f"(realised drift exceeding the rho bound)")
        print(f"  median width  : {r['median_width']:.3g}")


if __name__ == "__main__":
    main()
