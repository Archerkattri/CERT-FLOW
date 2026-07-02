"""Width head-to-head: attack CERT-FLOW's WEAK row (interval width) with five
UB/pricing strategies on the SAME worlds/seeds, measuring coverage vs true OPT.

Modes (all sharing one LB construction unless noted):
  (i)   default        -- Bonferroni per-edge pricing (the shipped certificate).
  (ii)  sum_aware      -- sum_aware_ub=True (T4 block-quantile UB, existing flag).
  (iii) pasc           -- path_calibration="pasc" (joint per-edge radius; the
                          known +25.1% regression, kept for completeness).
  (iv)  cia-ub         -- default LB, UB replaced per round by the upper end of
                          cia_path_certificate() on the incumbent when finite
                          (a valid UB on OPT: the incumbent is a feasible path),
                          else fall back to the default UB (fallback fraction
                          recorded).
  (v)   shrink         -- default certificate UNCHANGED (shrink_license=True is
                          purely observational); ADDITIONALLY records the Tier-2
                          licensed shrunk gap (diagnostics shrunk_gap + licensed_k)
                          and the SHADOW VIOLATION RATE: per round, does the
                          shrunk interval [shadow_lb, shadow_ub] (incumbent radii
                          * licensed_k) still contain the true OPT? That measured
                          coverage at measured width is the headline.

Benchmarks: (A) real METR-LA replay, (B) a bounded-drift grid. Seeds are run in
parallel (multiprocessing); true OPT per round is computed ONCE per seed and
shared across modes (it depends only on world/time/endpoints).

Run: PYTHONPATH=src python scripts/run_width_attack.py [--quick]
  --quick = 3 seeds x 100 rounds smoke test (run this FIRST).
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from multiprocessing import Pool
from pathlib import Path

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.graphcore import dijkstra
from certflow.realworld import TrafficWorld, far_endpoints, traffic_planner_config

QUICK = "--quick" in sys.argv
SEEDS = 3 if QUICK else 10
METR_ROUNDS = 100 if QUICK else 288
GRID_ROUNDS = 100 if QUICK else 150
TOL = 1e-6

# (name, config-override, kind)
MODES = [
    ("default", dict(), "plain"),
    ("sum_aware", dict(sum_aware_ub=True), "plain"),
    ("pasc", dict(path_calibration="pasc"), "plain"),
    ("cia-ub", dict(), "cia_ub"),
    ("shrink", dict(shrink_license=True), "shrink"),
]


def true_opt(world, t, s, g) -> float:
    snap = {u: {v: world.true_cost((u, v), t) for v in nbrs}
            for u, nbrs in world.graph.items()}
    _, cost = dijkstra(snap, s, g)
    return cost


def _blank(kind):
    d = dict(rounds_total=0, valid=0, violations=0, gaps=[], cert_gaps=[])
    if kind == "cia_ub":
        d["cia_fallback"] = 0
    if kind == "shrink":
        d.update(licensed_ks=[], shadow_gaps=[], shadow_n=0, shadow_violations=0)
    return d


def _build(bench, seed, rounds):
    """(world, start, goal, config-factory, delta) for a benchmark+seed."""
    if bench == "metr-la":
        w = TrafficWorld(dataset="metr-la", seed=seed, n_bins=rounds)
        s, g = far_endpoints(w)
        return w, s, g, (lambda **o: traffic_planner_config(**o)), traffic_planner_config().delta
    else:  # drift grid
        w = grid_world(10, 10, seed=seed, kind="bounded", rho=0.02, noise_scale=0.05)
        s, g = (0, 0), (9, 9)
        base = dict(epsilon=5.0, alpha_prime=0.2)
        return w, s, g, (lambda **o: PlannerConfig(**{**base, **o})), 1.0


def run_seed(task):
    bench, seed, rounds = task
    w, s, g, cfg_of, delta = _build(bench, seed, rounds)
    opt = [true_opt(w, i * delta, s, g) for i in range(rounds)]  # shared

    out = {}
    for mname, over, kind in MODES:
        cfg = cfg_of(**over)
        eps, min_conf = cfg.epsilon, cfg.min_certify_confidence
        p = CertPlanner(w, s, g, cfg)
        stats = _blank(kind)
        for i in range(rounds):
            cert, _ = p.round()
            stats["rounds_total"] += 1
            if not cert.valid:
                continue
            lb, ub = cert.lb, cert.ub
            if kind == "cia_ub":
                cia = p.cia_path_certificate(cert.path)
                if cia is not None and math.isfinite(cia.ub):
                    ub = cia.ub
                else:
                    stats["cia_fallback"] += 1
            o = opt[i]
            stats["valid"] += 1
            stats["violations"] += not (lb - TOL <= o <= ub + TOL)
            gap = ub - lb
            stats["gaps"].append(gap)
            if gap <= eps and cert.confidence >= min_conf:
                stats["cert_gaps"].append(gap)
            if kind == "shrink":
                d = p.diagnostics()
                slb, sub = d["shrunk_lb"], d["shrunk_ub"]
                stats["licensed_ks"].append(d["licensed_k"])
                if slb == slb and sub == sub:  # finite (not nan)
                    stats["shadow_n"] += 1
                    stats["shadow_gaps"].append(sub - slb)
                    stats["shadow_violations"] += not (slb - TOL <= o <= sub + TOL)
        out[mname] = stats
    return bench, out


def _merge(dicts):
    """Merge per-seed stat dicts (list of {mode: stats}) into {mode: stats}."""
    merged = {}
    for _, per_mode in dicts:
        for mname, s in per_mode.items():
            m = merged.setdefault(mname, None)
            if m is None:
                merged[mname] = {k: (list(v) if isinstance(v, list) else v)
                                 for k, v in s.items()}
            else:
                for k, v in s.items():
                    if isinstance(v, list):
                        m[k].extend(v)
                    else:
                        m[k] += v
    return merged


def _row(mname, kind, s, default_gap_med):
    gaps, cert_gaps = s["gaps"], s["cert_gaps"]
    gap_med = st.median(gaps) if gaps else float("nan")
    row = dict(
        mode=mname,
        valid_pct=s["valid"] / s["rounds_total"] if s["rounds_total"] else float("nan"),
        violation_rate=s["violations"] / s["valid"] if s["valid"] else float("nan"),
        n_valid=s["valid"],
        gap_mean=st.mean(gaps) if gaps else float("nan"),
        gap_median=gap_med,
        gap_ratio=(gap_med / default_gap_med
                   if gaps and default_gap_med else float("nan")),
        cert_gap_median=st.median(cert_gaps) if cert_gaps else float("nan"),
    )
    if kind == "cia_ub":
        row["cia_fallback_frac"] = s["cia_fallback"] / s["valid"] if s["valid"] else float("nan")
    if kind == "shrink":
        lks = s["licensed_ks"]
        sgaps = s["shadow_gaps"]
        sgap_med = st.median(sgaps) if sgaps else float("nan")
        row.update(
            licensed_k_median=st.median(lks) if lks else float("nan"),
            licensed_k_min=min(lks) if lks else float("nan"),
            licensed_k_lt1_frac=(sum(k < 1.0 for k in lks) / len(lks)
                                 if lks else float("nan")),
            shrunk_gap_median=sgap_med,
            shrunk_gap_ratio=(sgap_med / default_gap_med
                              if sgaps and default_gap_med else float("nan")),
            shadow_violation_rate=(s["shadow_violations"] / s["shadow_n"]
                                   if s["shadow_n"] else float("nan")),
            shadow_n=s["shadow_n"],
        )
    return row


def run_bench(bench, rounds):
    tasks = [(bench, seed, rounds) for seed in range(SEEDS)]
    with Pool(processes=min(SEEDS, 16)) as pool:
        results = pool.map(run_seed, tasks)
    merged = _merge(results)
    default_gap_med = st.median(merged["default"]["gaps"]) if merged["default"]["gaps"] else float("nan")
    kinds = {m: k for m, _, k in MODES}
    return [_row(m, kinds[m], merged[m], default_gap_med) for m, _, _ in MODES]


def _print_table(title, rows):
    print(f"\n=== {title} ===")
    hdr = (f"{'mode':10} {'valid%':>7} {'viol':>7} {'gap~med':>9} "
           f"{'gap~mean':>9} {'ratio':>6} {'cert~gap':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['mode']:10} {100*r['valid_pct']:>6.1f}% {r['violation_rate']:>7.4f} "
              f"{r['gap_median']:>9.3f} {r['gap_mean']:>9.3f} "
              f"{r['gap_ratio']:>6.3f} {r['cert_gap_median']:>9.3f}")
    for r in rows:
        if "cia_fallback_frac" in r:
            print(f"  cia-ub fallback fraction: {r['cia_fallback_frac']:.3f}")
        if "shadow_violation_rate" in r:
            print(f"  shrink Tier-2: licensed_k median={r['licensed_k_median']:.2f} "
                  f"min={r['licensed_k_min']:.2f} (<1 in {100*r['licensed_k_lt1_frac']:.0f}% of rounds); "
                  f"shrunk gap ratio={r['shrunk_gap_ratio']:.3f}; "
                  f"SHADOW violation rate={r['shadow_violation_rate']:.4f} "
                  f"(n={r['shadow_n']})")


def main():
    print(f"width_attack: seeds={SEEDS} metr_rounds={METR_ROUNDS} "
          f"grid_rounds={GRID_ROUNDS} quick={QUICK}", flush=True)
    metr = run_bench("metr-la", METR_ROUNDS)
    print("done METR-LA", flush=True)
    grid = run_bench("grid", GRID_ROUNDS)
    print("done grid", flush=True)

    outdir = Path("scripts/out")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "width_attack.json").write_text(json.dumps(dict(
        seeds=SEEDS, metr_rounds=METR_ROUNDS, grid_rounds=GRID_ROUNDS,
        quick=QUICK, metr_la=metr, grid=grid,
    ), indent=2))

    _print_table(f"METR-LA ({SEEDS} seeds x {METR_ROUNDS} rounds)", metr)
    _print_table(f"drift grid 10x10 ({SEEDS} seeds x {GRID_ROUNDS} rounds)", grid)
    print(f"\nwrote {outdir/'width_attack.json'}")


if __name__ == "__main__":
    main()
