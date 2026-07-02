"""Hybrid-sensing promotion check on real METR-LA traffic.

Scoreboard row under attack: "Sensing — GOOD (hybrid) / FAIL (pure gap-directed)".
The hybrid (objective-matched) policy beat pure gap-directed sensing 5x and the
CTP-RS-style VOI baseline on the synthetic unknown-terrain benchmark
(extern-baselines Part B), but no REAL-data comparison exists. Promotion to
default requires evidence that hybrid >= pure policies where it matters: on the
real cost process, on both axes at once (route regret AND certificate quality).

Design: per-round route-selection comparison on TrafficWorld (METR-LA replay,
one day = 288 five-minute bins). Each policy runs the SAME planner class on the
same replay (identical true costs; observation noise per instance) and differs
only in the sensing configuration:

  cert    - sensing_policy="cert", hybrid_sensing=False   (shipped default)
  hybrid  - sensing_policy="cert", hybrid_sensing=True    (promotion candidate)
  max_age - freshness round-robin baseline
  random  - random-edge baseline

Per round (after warm-up): regret = truecost(incumbent, t) - truecost(opt, t);
certificate validity/violations vs true OPT; certified-round fraction; gap.

Run: PYTHONPATH=<certflow src> python hybrid_real_metrla.py [--quick]
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from certflow.cert import CertPlanner
from certflow.graphcore import dijkstra
from certflow.realworld import TrafficWorld, far_endpoints, traffic_planner_config

QUICK = "--quick" in sys.argv
SEEDS = 3 if QUICK else 10
ROUNDS = 60 if QUICK else 288
WARMUP = 12  # rounds excluded from regret scoring (calibration warm-up)

POLICIES = {
    "cert (pure gap, default)": dict(sensing_policy="cert", hybrid_sensing=False),
    "hybrid (objective-matched)": dict(sensing_policy="cert", hybrid_sensing=True),
    "max_age (freshness)": dict(sensing_policy="max_age", hybrid_sensing=False),
    "random": dict(sensing_policy="random", hybrid_sensing=False),
}


def true_snapshot(world: TrafficWorld, t: float) -> dict:
    return {u: {v: world.true_cost((u, v), t) for v in nbrs}
            for u, nbrs in world.graph.items()}


def path_true_cost(snap: dict, path: list) -> float:
    return sum(snap[u][v] for u, v in zip(path, path[1:]))


def run_one(args: tuple) -> dict:
    policy_name, overrides, seed = args
    world = TrafficWorld(dataset="metr-la", seed=seed, n_bins=ROUNDS)
    s, g = far_endpoints(world)
    cfg = traffic_planner_config(**overrides)
    planner = CertPlanner(world, s, g, cfg)

    regrets, gaps = [], []
    violations = valid_rounds = certified = scored = 0
    wall = 0.0
    for i in range(ROUNDS):
        t_round = planner.t
        t0 = time.perf_counter()
        cert, _ = planner.round()
        wall += time.perf_counter() - t0
        if i < WARMUP:
            continue
        snap = true_snapshot(world, t_round)
        _, opt = dijkstra(snap, s, g)
        scored += 1
        if cert.valid:
            valid_rounds += 1
            gaps.append(cert.gap)
            if not (cert.lb <= opt <= cert.ub):
                violations += 1
            if cert.gap <= cfg.epsilon:
                certified += 1
        if cert.path:
            regrets.append(path_true_cost(snap, cert.path) - opt)
    return dict(
        policy=policy_name, seed=seed,
        regret_mean=st.mean(regrets) if regrets else float("nan"),
        regret_median=st.median(regrets) if regrets else float("nan"),
        valid_frac=valid_rounds / scored,
        violation_frac=violations / max(1, valid_rounds),
        certified_frac=certified / scored,
        gap_median=st.median(gaps) if gaps else float("nan"),
        wall_per_round_ms=1e3 * wall / ROUNDS,
    )


def main() -> None:
    jobs = [(name, over, seed)
            for name, over in POLICIES.items() for seed in range(SEEDS)]
    with Pool(min(8, len(jobs))) as pool:
        rows = pool.map(run_one, jobs)

    agg = {}
    for name in POLICIES:
        rs = [r for r in rows if r["policy"] == name]
        agg[name] = dict(
            seeds=len(rs),
            regret_mean=st.mean(r["regret_mean"] for r in rs),
            regret_median=st.median(r["regret_median"] for r in rs),
            valid_frac=st.mean(r["valid_frac"] for r in rs),
            violation_frac=st.mean(r["violation_frac"] for r in rs),
            certified_frac=st.mean(r["certified_frac"] for r in rs),
            gap_median=st.median(r["gap_median"] for r in rs),
            wall_per_round_ms=st.mean(r["wall_per_round_ms"] for r in rs),
        )

    out = Path(__file__).parent / "out" / "hybrid_real_metrla.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(
        config=dict(seeds=SEEDS, rounds=ROUNDS, warmup=WARMUP, quick=QUICK),
        per_seed=rows, aggregate=agg), indent=2))

    hdr = (f"{'policy':<28} {'regret mean':>11} {'regret med':>10} "
           f"{'valid%':>7} {'viol%':>6} {'cert%':>6} {'gap med':>9} {'ms/rd':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, a in agg.items():
        print(f"{name:<28} {a['regret_mean']:>11.1f} {a['regret_median']:>10.1f} "
              f"{100*a['valid_frac']:>6.1f}% {100*a['violation_frac']:>5.2f}% "
              f"{100*a['certified_frac']:>5.1f}% {a['gap_median']:>9.0f} "
              f"{a['wall_per_round_ms']:>6.1f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
