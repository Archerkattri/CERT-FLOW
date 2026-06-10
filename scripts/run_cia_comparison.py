"""Head-to-head: Luo & Zhou CIA (Conformalized Interval Arithmetic) vs CERT
on METR-LA travel-time path-cost intervals, as the calibration->test TIME GAP
grows.

WHY THIS EXPERIMENT
-------------------
CIA (Luo & Zhou, "Conformalized Interval Arithmetic with Symmetric
Calibration", arXiv 2408.10939, AAAI 2025; code github.com/luo-lorry/CIA) is
CERT's closest conformal neighbour: it builds conformal prediction intervals
for SUMS of per-element labels (path cost = sum of edge costs) under the
EXCHANGEABILITY of the per-element nonconformity scores. CERT replaces that
exchangeability with non-exchangeable age weights + an explicit drift term
(c_hat +/- (q + rho*age)). This script measures the exchangeability-fragility
CIA was never built to survive: coverage of the true path sum at T_cal+gap
when the interval was calibrated at T_cal, as the gap grows.

ATTRIBUTION / LICENSE
---------------------
The CIA construction implemented in `cia_interval` below is a FAITHFUL
EXTRACTION of the symmetric-calibration sum interval from CIA's own
`main.py::group_by_dimensions` (the "stratified CIA" / "nonstratified CIA"
branches): per-edge signed scores, a random two-way split of the labelled
edge population (symmetric calibration), path-sum scores stratified by path
length k, and the (1-alpha)(n+1)/n calibration percentile, with the interval
centred at the predicted path sum and half-width = that percentile (their
efficiency is reported as 2*percentile, i.e. the full width). We run THEIR
construction, not their CLI, because their CLI consumes static Anaheim/Chicago
flow `.tntp` data and we need a temporal METR-LA gap sweep. Deviations from
their original construction are listed in docs/results/cia-comparison.md.

The CIA repository has NO LICENSE file. We run an extracted re-implementation
of their published method for a research comparison under academic-use norms,
and say so prominently in the results doc. No CIA source code is copied or
vendored into this repository.

Run: cert_env/bin/python scripts/run_cia_comparison.py [--quick]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from certflow.conformal import ConformalScorer, path_alpha_edge
from certflow.graphcore import dijkstra
from certflow.realworld import BIN_SECONDS, TrafficWorld

QUICK = "--quick" in sys.argv

# ---- experiment constants -------------------------------------------------
ALPHA = 0.10                     # target 90% coverage (1 - alpha)
N_PATHS = 12 if QUICK else 50    # fixed set of random s-g paths
MIN_EDGES, MAX_EDGES = 6, 15
N_REPS = 6 if QUICK else 20      # random (T_cal, path) repetitions per gap
GAP_BINS = [0, 12, 36, 72, 144, 288]   # 0, 1h, 3h, 6h, 12h, 24h at 5-min bins
GAP_LABELS = ["0", "1h", "3h", "6h", "12h", "24h"]
# one long window so T_cal + max(gap) always fits; topology is seed-independent
WINDOW_BINS = 600 if QUICK else 3200
CAL_OBS_PER_EDGE = 1             # one observation per edge per calibration slice
RHO_QUANTILE = 0.75             # CERT drift dial: p75, the width-optimal point
                                # established in docs/results/metr-la.md
MASTER_SEED = 0


# ---------------------------------------------------------------------------
# CIA construction (faithful extraction of luo-lorry/CIA main.py).
# ---------------------------------------------------------------------------
def cia_calibrate(per_edge_scores: np.ndarray, rng: np.random.Generator):
    """CIA symmetric calibration. Mirrors group_by_dimensions: randomly split
    the labelled edge population into two halves; calibration draws path-sum
    scores from one half. Here we return the (shuffled) calibration-half score
    pool from which path-k sums are sampled, exactly as CIA samples k edges
    from the calibration partition to form a length-k path-sum score
    (cf. the 'Group sampling' branch np.random.choice(..., size=(k,))).
    """
    idx = np.arange(len(per_edge_scores))
    rng.shuffle(idx)
    half = len(idx) // 2
    cal_pool = per_edge_scores[idx[:half]]      # symmetric-calibration half
    return cal_pool


def cia_threshold(cal_pool: np.ndarray, k: int, alpha: float,
                  rng: np.random.Generator, n_samples: int = 200) -> float:
    """CIA path-k threshold via symmetric calibration, stratified by length k.

    Exactly CIA's 'Group sampling' construction (main.py lines ~220-234):
    draw n_samples calibration path-sum scores, each = |sum of k edge scores
    sampled WITHOUT replacement from the calibration half|, then take the
    (1-alpha)(1+n)/n percentile. Returns the half-width (CIA's interval is
    predicted_sum +/- threshold; its reported efficiency is 2*threshold).
    """
    m = len(cal_pool)
    k = min(k, m)
    sums = np.empty(n_samples)
    for i in range(n_samples):
        pick = rng.choice(m, size=k, replace=False)
        sums[i] = np.abs(cal_pool[pick].sum())
    pct = min(100.0, (100.0 - 100.0 * alpha) * (1 + n_samples) / n_samples)
    return float(np.percentile(sums, pct))


# ---------------------------------------------------------------------------
# Path construction on the real graph.
# ---------------------------------------------------------------------------
def build_paths(world: TrafficWorld, n_paths: int, lo: int, hi: int,
                rng: np.random.Generator):
    """A fixed set of simple s-g paths with edge counts in [lo, hi], from the
    real graph topology (shortest path on cost-at-t0, trimmed to length)."""
    snap = {u: {v: world.true_cost((u, v), 0.0) for v in nbrs}
            for u, nbrs in world.graph.items()}
    nodes = sorted(world.graph)
    paths: list[list] = []
    tries = 0
    while len(paths) < n_paths and tries < n_paths * 200:
        tries += 1
        s = nodes[rng.integers(len(nodes))]
        g = nodes[rng.integers(len(nodes))]
        if s == g:
            continue
        node_path, cost = dijkstra(snap, s, g)
        if node_path is None or not (lo + 1 <= len(node_path) <= hi + 1):
            continue
        edges = list(zip(node_path[:-1], node_path[1:]))
        if not (lo <= len(edges) <= hi):
            continue
        if edges in paths:
            continue
        paths.append(edges)
    if len(paths) < n_paths:
        raise RuntimeError(f"only found {len(paths)}/{n_paths} paths")
    return paths


def true_path_sum(world: TrafficWorld, edges, t: float) -> float:
    return sum(world.true_cost(e, t) for e in edges)


def cp_ci(k: int, n: int):
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------------------
# Main sweep.
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(MASTER_SEED)
    world = TrafficWorld(dataset="metr-la", seed=0, n_bins=WINDOW_BINS,
                         rho_quantile=RHO_QUANTILE)
    all_edges = list(world.edges())
    paths = build_paths(world, N_PATHS, MIN_EDGES, MAX_EDGES, rng)
    plen = [len(p) for p in paths]
    print(f"graph: {len(world.graph)} nodes, {len(all_edges)} edges; "
          f"{len(paths)} paths, edge counts {min(plen)}-{max(plen)} "
          f"(median {int(np.median(plen))})", flush=True)

    # accumulators[gap] = dict of running counts/lists
    cia = {g: {"cov": 0, "n": 0, "w": []} for g in GAP_BINS}
    cert = {g: {"cov": 0, "n": 0, "w": []} for g in GAP_BINS}

    max_gap = max(GAP_BINS)
    # valid calibration times: leave room for the largest gap and a settle margin
    t_cal_lo, t_cal_hi = 2, WINDOW_BINS - max_gap - 2

    for rep in range(N_REPS):
        for gap in GAP_BINS:
            tcal_bin = int(rng.integers(t_cal_lo, t_cal_hi))
            t_cal = tcal_bin * BIN_SECONDS
            t_test = (tcal_bin + gap) * BIN_SECONDS
            pidx = int(rng.integers(len(paths)))
            edges = paths[pidx]
            k = len(edges)

            # ---- shared calibration data: observed edge costs at T_cal ----
            # CIA per-edge score = signed residual (obs - true) at T_cal, the
            # observation-noise residual that IS exchangeable within the slice
            # (this is CIA's 'yhat - y'; here yhat = obs, y = true cost). The
            # SAME observation stream feeds CERT below.
            obs_at_cal = {e: max(world.observe(e, t_cal), 1.0)
                          for e in all_edges}
            cia_scores = np.array(
                [obs_at_cal[e] - world.true_cost(e, t_cal) for e in all_edges])

            # ===== CIA: symmetric-calibration sum interval =====
            cal_pool = cia_calibrate(cia_scores, rng)
            half_w = cia_threshold(cal_pool, k, ALPHA, rng)
            pred_sum = sum(obs_at_cal[e] for e in edges)
            cia_lb, cia_ub = pred_sum - half_w, pred_sum + half_w
            truth = true_path_sum(world, edges, t_test)
            cia[gap]["cov"] += int(cia_lb - 1e-9 <= truth <= cia_ub + 1e-9)
            cia[gap]["n"] += 1
            cia[gap]["w"].append(cia_ub - cia_lb)

            # ===== CERT: per-edge c_hat +/- (q + rho*gap) summed (Bonferroni) =====
            # q from a ConformalScorer calibrated on the SAME calibration slice's
            # drift-adjusted residuals, at the Bonferroni per-edge level
            # alpha/k; rho = the world's p75 per-edge drift rate (as in our
            # traffic runs); the staleness term is rho * gap_seconds.
            scorer = ConformalScorer(rho_w=1.0, eps_tv=0.0)   # gap is single-step:
            # one calibration slice, no age weighting needed within the slice
            for e in all_edges:
                # drift-adjusted score at calibration (age ~ 0 here, so this is
                # |obs - c_hat|); c_hat is the prior/center == obs at T_cal, so
                # the residual is the noise magnitude, matching CERT's push.
                scorer.push(abs(obs_at_cal[e] - world.true_cost(e, t_cal)), t_cal)
            alpha_edge = path_alpha_edge(ALPHA, k)
            q = scorer.quantile(alpha_edge, t_cal)
            if not np.isfinite(q):
                q = max(cia_scores.max() if len(cia_scores) else 0.0, 0.0)
            gap_seconds = gap * BIN_SECONDS
            cert_lb = cert_ub = 0.0
            for e in edges:
                rho_e = world.rho_true(e)
                half = q + rho_e * gap_seconds
                c_hat = obs_at_cal[e]
                cert_lb += max(1.0, c_hat - half)
                cert_ub += c_hat + half
            cert[gap]["cov"] += int(cert_lb - 1e-9 <= truth <= cert_ub + 1e-9)
            cert[gap]["n"] += 1
            cert[gap]["w"].append(cert_ub - cert_lb)
        print(f"  rep {rep + 1}/{N_REPS} done", flush=True)

    # ---- assemble table ----
    def summarize(acc):
        out = []
        for g in GAP_BINS:
            a = acc[g]
            lo, hi = cp_ci(a["cov"], a["n"])
            out.append(dict(
                gap_bins=g,
                coverage=a["cov"] / a["n"] if a["n"] else float("nan"),
                ci_lo=lo, ci_hi=hi,
                median_width=float(np.median(a["w"])) if a["w"] else float("nan"),
                n=a["n"],
            ))
        return out

    result = dict(
        alpha=ALPHA, target_coverage=1 - ALPHA,
        n_paths=len(paths), n_reps=N_REPS,
        gap_bins=GAP_BINS, gap_labels=GAP_LABELS,
        rho_quantile=RHO_QUANTILE,
        a1_violation_rate=world.a1_violation_rate,
        cia=summarize(cia), cert=summarize(cert),
    )
    outdir = Path("results/cia_comparison")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "table.json").write_text(json.dumps(result, indent=2))

    # ---- print ----
    print(f"\nTarget coverage {1 - ALPHA:.2f} (alpha={ALPHA}); "
          f"{len(paths)} paths x {N_REPS} reps/gap; rho p{int(RHO_QUANTILE*100)}; "
          f"A1-violation rate {world.a1_violation_rate:.3f}")
    hdr = f"{'gap':>5} {'CIA cov':>8} {'CIA 95%CI':>16} {'CIA medW':>9}  " \
          f"{'CERT cov':>8} {'CERT 95%CI':>16} {'CERT medW':>9}"
    print(hdr)
    print("-" * len(hdr))
    for i, g in enumerate(GAP_BINS):
        c, k = result["cia"][i], result["cert"][i]
        print(f"{GAP_LABELS[i]:>5} "
              f"{c['coverage']:>8.3f} [{c['ci_lo']:.3f},{c['ci_hi']:.3f}]"
              f"{'':>2} {c['median_width']:>9.0f}  "
              f"{k['coverage']:>8.3f} [{k['ci_lo']:.3f},{k['ci_hi']:.3f}]"
              f"{'':>2} {k['median_width']:>9.0f}")


if __name__ == "__main__":
    main()
