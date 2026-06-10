"""Tier-L: lifelong operation — the honest home of objective O4/H1.

Within one mission, memory cannot cut replanning latency (D* Lite reuses
search state; pre-widening restores locality — that verdict stands). Across
MISSIONS in the same drifting environment, a memoryless planner re-pays
warm-up (calibration n0), re-learns every edge, and re-discovers corridors.
This experiment measures what each persistent component buys:

  memoryless   fresh planner per mission (unknown-terrain start)
  beliefs      carry edge beliefs only (fresh calibration, fresh kappa)
  calibration  carry the score buffer + ACI only
  full-nok     carry everything except kappa
  full         carry everything (retarget())

Protocol: 6x6 bounded drift rho=0.01, eps=5, alpha'=0.2; 8 missions per
seed from a fixed endpoint pool; 50 idle rounds of drift between missions;
each mission runs to certification + 5 maintenance rounds or 250 rounds.
Metrics per mission (excluding the first, which no variant can warm-start):
rounds to first valid certificate, rounds to certification, sensing spend
to certification, median round latency, certified-incumbent regret.

Run: cert_env/bin/python scripts/run_lifelong.py [--quick]
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.graphcore import dijkstra
from certflow.sensing import path_edges

QUICK = "--quick" in sys.argv
SEEDS = 4 if QUICK else 16
MISSIONS = 4 if QUICK else 8
MAX_ROUNDS = 250
IDLE = 50.0
POOL = [((0, 0), (5, 5)), ((0, 5), (5, 0)), ((2, 0), (5, 5)), ((0, 0), (3, 5)),
        ((5, 0), (0, 5)), ((0, 2), (5, 3)), ((5, 5), (0, 0)), ((3, 5), (0, 0))]

CFG = dict(epsilon=5.0, alpha_prime=0.2, rho_w=0.99, eps_tv=1e-4,
           gamma_aci=0.01, initial_survey=False)


def true_opt(world, t, s, g):
    snap = {u: {v: world.true_cost((u, v), t) for v in nb}
            for u, nb in world.graph.items()}
    return dijkstra(snap, s, g)[1]


def run_mission(p, world, s, g, eps):
    """Returns (rounds_to_valid, rounds_to_cert, sense_to_cert, lat_p50,
    regret) — None fields when never reached."""
    r_valid = r_cert = None
    spend0 = p.sense_spend
    lats = []
    regret = None
    extra = 0
    for i in range(MAX_ROUNDS):
        t0 = time.perf_counter()
        cert, _ = p.round()
        lats.append(time.perf_counter() - t0)
        if r_valid is None and cert.valid:
            r_valid = i
        certified = (cert.valid and cert.gap <= eps
                     and cert.confidence >= p.cfg.min_certify_confidence)
        if certified and r_cert is None:
            r_cert = i
            opt_c = true_opt(world, p.t, s, g)
            inc = sum(world.true_cost(e, p.t) for e in path_edges(cert.path))
            regret = inc - opt_c
        if r_cert is not None:
            extra += 1
            if extra >= 5:
                break
    return (r_valid, r_cert, p.sense_spend - spend0,
            1e3 * statistics.median(lats), regret)


def make_variant(variant, world, s, g, t_now, saved):
    cfg = PlannerConfig(use_kappa=(variant != "full-nok"), **CFG)
    if variant == "full" or variant == "full-nok":
        if saved.get("planner") is not None:
            p = saved["planner"]
            p.t = t_now
            p.retarget(s, g)
            return p
    p = CertPlanner(world, s, g, cfg, t0=t_now)
    if variant == "beliefs" and saved.get("beliefs"):
        for e, b_old in saved["beliefs"].items():
            b = p.beliefs[e]
            b.c_hat, b.t_obs, b.observed = b_old
    if variant == "calibration" and saved.get("scorer") is not None:
        p.scorer = saved["scorer"]
        p.aci = saved["aci"]
    return p


def main():
    variants = ["memoryless", "beliefs", "calibration", "full-nok", "full"]
    agg = {v: {"valid": [], "cert": [], "sense": [], "lat": [], "regret": [],
               "cert_rate": []} for v in variants}
    for seed in range(SEEDS):
        for v in variants:
            world = grid_world(6, 6, seed=seed, kind="bounded", rho=0.01,
                               noise_scale=0.05)
            t_now = 0.0
            saved = {}
            for m in range(MISSIONS):
                s, g = POOL[(m + seed) % len(POOL)]
                p = make_variant(v, world, s, g, t_now, saved)
                rv, rc, sp, lat, reg = run_mission(p, world, s, g, CFG["epsilon"])
                t_now = p.t + IDLE
                saved = {
                    "planner": p,
                    "beliefs": {e: (b.c_hat, b.t_obs, b.observed)
                                for e, b in p.beliefs.items()},
                    "scorer": p.scorer, "aci": p.aci,
                }
                if m == 0:
                    continue  # first mission: nothing to warm-start
                a = agg[v]
                a["cert_rate"].append(rc is not None)
                if rv is not None:
                    a["valid"].append(rv)
                if rc is not None:
                    a["cert"].append(rc)
                    a["sense"].append(sp)
                    a["regret"].append(reg)
                a["lat"].append(lat)
        print(f"done seed {seed}", flush=True)

    rows = []
    hdr = (f"{'variant':12} {'cert-rate':>9} {'rounds->valid':>13} "
           f"{'rounds->cert':>12} {'sense->cert':>11} {'lat p50':>8} {'regret~':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    for v in variants:
        a = agg[v]
        row = dict(
            variant=v,
            cert_rate=sum(a["cert_rate"]) / max(len(a["cert_rate"]), 1),
            valid_med=med(a["valid"]), cert_med=med(a["cert"]),
            sense_med=med(a["sense"]), lat_med=med(a["lat"]),
            regret_med=med(a["regret"]),
        )
        rows.append(row)
        print(f"{v:12} {row['cert_rate']:>8.0%} {row['valid_med']:>13.1f} "
              f"{row['cert_med']:>12.1f} {row['sense_med']:>11.1f} "
              f"{row['lat_med']:>8.2f} {row['regret_med']:>8.3f}")
    outdir = Path("results/lifelong")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "table.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
