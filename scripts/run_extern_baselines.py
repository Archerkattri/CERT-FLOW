"""External-algorithm comparisons: AD*-semantics bounds and CTP-RS-style VOI.

Part A — bound semantics on a SHARED observation stream: one planner runs
with neutral freshness sensing (max_age); every valid round, three bound
constructions are evaluated from the SAME belief state against the true
optimum: CERT's conformal certificate, and AD*/ARA*-style w-suboptimality
intervals [c(P-hat)/w, c(P-hat)] on point estimates (w in {1.2, 1.5, 2.0}).
The AD* claim is sound on its own map; the table measures what happens when
that semantics meets drift and noise. Run on synthetic drift and METR-LA.

Part B — sensing-policy regret: certify-then-go Tier-2 with the CTP-RS-style
"voi" policy (sense where uncertainty x expected-cost-route relevance is
highest) vs CERT's certificate-gap policy.

Part C — TASP-degenerate: computation-only tightening (never sense) in a
drifting world: the certificate can never become valid (nothing to
calibrate with); reported as a row, not a table.

Run: cert_env/bin/python scripts/run_extern_baselines.py [--quick]
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from certflow.baselines import adstar_bound
from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.episodes import tier2_episode
from certflow.graphcore import dijkstra
from certflow.harness import ExperimentConfig, run_experiment
from certflow.oracle import opt

QUICK = "--quick" in sys.argv
SEEDS_A = 4 if QUICK else 15
ROUNDS_A = 100 if QUICK else 300
SEEDS_B = 4 if QUICK else 15
W_LIST = (1.2, 1.5, 2.0)


def part_a_world(kind: str, seed: int):
    if kind == "synthetic":
        w = grid_world(6, 6, seed=seed, kind="bounded", rho=0.02, noise_scale=0.05)
        cfg = PlannerConfig(epsilon=5.0, alpha_prime=0.2, eps_tv=1e-4,
                            sensing_policy="max_age")
        return w, (0, 0), (5, 5), cfg, lambda t, s, g: opt(w, t, s, g)[1]
    from certflow.realworld import TrafficWorld, far_endpoints, traffic_planner_config

    w = TrafficWorld(seed=seed, n_bins=ROUNDS_A)
    s, g = far_endpoints(w)
    cfg = traffic_planner_config(sensing_policy="max_age")

    def true_opt(t, s_, g_):
        snap = {u: {v: w.true_cost((u, v), t) for v in nb}
                for u, nb in w.graph.items()}
        return dijkstra(snap, s_, g_)[1]

    return w, s, g, cfg, true_opt


def part_a(kind: str) -> list[dict]:
    stats = {"CERT": [0, 0, []]}
    for wgt in W_LIST:
        stats[f"AD* w={wgt}"] = [0, 0, []]
    for seed in range(SEEDS_A):
        w, s, g, cfg, true_opt = part_a_world(kind, seed)
        p = CertPlanner(w, s, g, cfg)
        struct = {u: dict(nbrs) for u, nbrs in p._graph_lower_cache.items()}
        for _ in range(ROUNDS_A):
            t0 = p.t
            cert, _ = p.round()
            if not cert.valid:
                continue
            o = true_opt(t0, s, g)
            rec = stats["CERT"]
            rec[0] += cert.lb - 1e-9 <= o <= cert.ub + 1e-9
            rec[1] += 1
            rec[2].append(cert.gap)
            for wgt in W_LIST:
                lo, hi = adstar_bound(p.beliefs, struct, s, g, w=wgt,
                                      cost_floor=cfg.cost_floor)
                rec = stats[f"AD* w={wgt}"]
                rec[0] += lo - 1e-9 <= o <= hi + 1e-9
                rec[1] += 1
                rec[2].append(hi - lo)
    rows = []
    for name, (cov, n, widths) in stats.items():
        rows.append(dict(
            world=kind, bound=name, n=n,
            validity=cov / n if n else float("nan"),
            width_median=sorted(widths)[len(widths) // 2] if widths else float("nan"),
        ))
    return rows


def part_b() -> list[dict]:
    base = ExperimentConfig(
        rows=10, cols=10, kind="bounded", rho=0.02,
        noise_family="gaussian", noise_scale=0.05,
        epsilon=8.0, alpha_prime=0.2, rho_w=0.99, eps_tv=1e-4, gamma_aci=0.01,
        use_kappa=True, initial_survey=False,
        move_policy="when_certified", sense_budget=20.0,
        n_seeds=SEEDS_B, max_rounds=200 if QUICK else 600, base_seed=2026,
    )
    rows = []
    variants = [
        ("cert", dict(sensing_policy="cert")),
        ("voi", dict(sensing_policy="voi")),
        ("hybrid", dict(sensing_policy="cert", hybrid_sensing=True)),
    ]
    for policy, over in variants:
        cfg = dataclasses.replace(base, **over)
        res = run_experiment(tier2_episode, cfg)
        regrets, goals = [], 0
        for ep in res.episodes:
            if getattr(ep, "reached_goal", False):
                goals += 1
                regrets.append(ep.travel_cost - ep.oracle_cost)
        rows.append(dict(
            policy=policy, goal=goals / max(len(res.episodes), 1),
            regret_mean=sum(regrets) / len(regrets) if regrets else float("nan"),
            regret_median=sorted(regrets)[len(regrets) // 2] if regrets else float("nan"),
        ))
        print(f"done: part B {policy}", flush=True)
    return rows


def part_c() -> dict:
    w = grid_world(6, 6, seed=0, kind="bounded", rho=0.02, noise_scale=0.05)
    p = CertPlanner(w, (0, 0), (5, 5),
                    PlannerConfig(epsilon=5.0, alpha_prime=0.2, eps_tv=1e-4,
                                  sensing_policy="none"))
    valid = sum(p.round()[0].valid for _ in range(ROUNDS_A))
    return dict(valid_rounds=valid, rounds=ROUNDS_A)


def main() -> None:
    out = {"part_a": [], "part_b": None, "part_c": None}
    for kind in ("synthetic", "metr-la"):
        out["part_a"].extend(part_a(kind))
        print(f"done: part A {kind}", flush=True)
    out["part_b"] = part_b()
    out["part_c"] = part_c()

    outdir = Path("results/extern_baselines")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "table.json").write_text(json.dumps(out, indent=2))

    print("\nPart A — bound semantics on a shared observation stream:")
    hdr = f"{'world':10} {'bound':12} {'n':>6} {'validity':>9} {'width~':>9}"
    print(hdr); print("-" * len(hdr))
    for r in out["part_a"]:
        print(f"{r['world']:10} {r['bound']:12} {r['n']:>6} "
              f"{r['validity']:>9.3f} {r['width_median']:>9.2f}")
    print("\nPart B — certify-then-go regret (budget 20, unknown terrain):")
    for r in out["part_b"]:
        print(f"  {r['policy']:5}: goal={r['goal']:.0%} regret mean={r['regret_mean']:.2f} "
              f"median={r['regret_median']:.2f}")
    c = out["part_c"]
    print(f"\nPart C — TASP-degenerate (never sense, drifting world): "
          f"{c['valid_rounds']}/{c['rounds']} valid rounds "
          f"(computation cannot substitute for observation)")


if __name__ == "__main__":
    main()
