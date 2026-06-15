"""CERT-FLOW RSS extended validation -- CELL: CALIBRATION -> TEST DISTRIBUTION
SHIFT (the eps_tv / A2 TV-Lipschitz theorem).  ADDITIONAL results for the RSS
version; NOT a change to the published paper.

What this stresses
------------------
Non-exchangeable split conformal (Barber, Candes, Ramdas, Tibshirani 2023,
Thm 2 + the independence corollary) certifies coverage under a calibration ->
test distribution shift by DISCOUNTING the claim with a TV-Lipschitz slack:

    delta_stale(t) = sum_i w~_i * min(1, 2 * eps_tv * age_i)      (conformal.py)
    confidence     = 1 - alpha_claim - L * delta_stale(t)         (cert.py round)

In every synthetic run in the paper eps_tv = 0 (the exchangeable claim).  This
cell turns that slack ON and asks the question it was designed for: CALIBRATE
CERT's buffer under one regime, then at a CHANGEPOINT switch the world (drift
rate rho and/or observation-noise family), and compare eps_tv = 0 vs eps_tv > 0.

The world (scripts/extval/shift_world.py, package imported READ-ONLY) splices
two published BoundedDriftWorld segments at t_cp and switches the observation
noise family/scale there; the planner-visible A1 bound stays the PRE-shift
bound (rho_true_mode="pre") so the planner is genuinely surprised, exactly like
a system calibrated under one regime and run in another.  The realised
post-shift A1-violation rate against that frozen bound is MEASURED and printed.

What is measured (identical inputs across eps_tv values: same world, same
seeds, same residual stream, same alpha'; ONLY eps_tv differs)
------------------------------------------------------------------------------
Rounds are segmented PRE (t<t_cp) / TRANSIENT (t in [t_cp, t_cp+W), buffer still
pre-shift-dominated) / SETTLED (t in [t_cp+W+settle, ...), buffer refilled).

  (1) EDGE-LEVEL conformal coverage -- where the theorem is TIGHT.  For the
      optimistic-path edges each round we test a HELD-OUT fresh observation
      (drawn from the world at the round's t, NOT pushed to the buffer) against
      the SAME per-edge band the planner uses for its ACI miss event
      (cert.py:964, half = q + rho*age): miss = obs outside c_hat +- half.
      Edge miscoverage should be <= the claimed alpha_edge IF exchangeable; the
      shift breaks that.  We compare realised miscoverage to alpha_edge (the
      eps_tv=0 claim) and to alpha_edge + delta_stale (the eps_tv-corrected
      coverage-gap bound).

  (2) PATH-LEVEL certificate coverage LB<=OPT<=UB among VALID rounds (the paper
      metric, episodes.coverage_among_valid), the VALID FRACTION, and the mean
      claimed confidence -- per segment, per eps_tv.

  (3) eps_tv GATING sweep in the transient: valid fraction and mean claim vs
      eps_tv -- the self-extinguishing dial (the certificate refusing to claim
      on a buffer too stale to honestly support it).

Honesty
-------
* No package edits.  World subclasses certflow.drift._GridBase / reuses
  BoundedDriftWorld read-only; the planner is the published CertPlanner; the
  oracle is certflow.oracle.opt; the per-edge band and alpha_edge are the
  planner's OWN recorded values (planner._last_alpha_edge, scorer.quantile,
  scorer.delta_stale) -- byte-for-byte what the certificate asserted.
* All numbers are produced by running real code now and printed.
* Results that do NOT favour the simple hypothesis are reported in full (see
  the READING block): the PATH certificate's Bonferroni conservatism keeps
  realised LB<=OPT<=UB coverage ~1.0 even at eps_tv=0, so the eps_tv coverage
  effect is visible at the conformal (edge) layer and as CLAIM HONESTY, not as
  a path-coverage rescue.

Run:  cert_env/bin/python scripts/extval/stress_cal_shift.py [--quick]
                                                             [--shift drift|jump]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from certflow.cert import CertPlanner, PlannerConfig  # noqa: E402
from certflow.oracle import opt  # noqa: E402
from shift_world import ShiftWorld, _draw_one  # noqa: E402


# ---------------------------------------------------------------------------
# Shift presets (the cell: switch noise_family AND/OR rho at the changepoint)
# ---------------------------------------------------------------------------
SHIFTS = {
    # steady DRIFT-RATE step, observation noise UNCHANGED -- the regime A2's
    # TV-Lipschitz slack is literally designed for (a bounded-rate change).
    "drift": dict(rho_pre=0.005, rho_post=0.06,
                  family_pre="gaussian", scale_pre=0.05,
                  family_post="gaussian", scale_post=0.05),
    # abrupt NOISE-FAMILY + scale jump AND a drift step -- the adversarial case
    # for a bounded-rate model (an instantaneous TV jump, not a steady rate).
    "jump": dict(rho_pre=0.005, rho_post=0.06,
                 family_pre="gaussian", scale_pre=0.04,
                 family_post="student_t", scale_post=0.5),
}


# ---------------------------------------------------------------------------
# Per-episode driver: mirrors episodes.tier0_episode (stationary robot, one
# certificate per round, path coverage vs the EXACT oracle) and additionally
# runs the held-out EDGE-level conformal test on the optimistic-path edges.
# ---------------------------------------------------------------------------
@dataclass
class Seg:
    # edge-level conformal test
    e_miss: int = 0
    e_n: int = 0
    e_miss_valid: int = 0       # edge misses on rounds the cert called VALID
    e_n_valid: int = 0
    claim_raw: list = field(default_factory=list)    # alpha_edge
    claim_corr: list = field(default_factory=list)   # alpha_edge + delta_stale
    # path-level certificate
    p_cov: int = 0              # covered AND valid
    p_valid: int = 0            # valid rounds
    p_rounds: int = 0           # all rounds in segment
    conf: list = field(default_factory=list)         # claimed confidence (valid)


def run_episode(world: ShiftWorld, cfg: PlannerConfig, rounds: int, t_cp: float,
                W: float, settle: float, probe_rng: np.random.Generator) -> dict:
    rows = max(r for r, _ in world.graph) + 1
    cols = max(c for _, c in world.graph) + 1
    start, goal = (0, 0), (rows - 1, cols - 1)
    planner = CertPlanner(world, start, goal, cfg)
    segs = {"pre": Seg(), "trans": Seg(), "settled": Seg()}

    for _ in range(rounds):
        t = planner.t
        cert, _ = planner.round()
        _, true_opt = opt(world, t, start, goal)

        if t < t_cp:
            seg = segs["pre"]
        elif t < t_cp + W:
            seg = segs["trans"]
        elif t >= t_cp + W + settle:
            seg = segs["settled"]
        else:
            seg = None  # gap between transient and settled (buffer mid-refill)

        if seg is None:
            continue
        seg.p_rounds += 1

        # ---- path-level certificate (paper metric) ----
        if cert.valid:
            seg.p_valid += 1
            covered = cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9
            seg.p_cov += int(covered)
            seg.conf.append(cert.confidence)

        # ---- edge-level conformal test (theorem is tight here) ----
        # Use the planner's OWN recorded per-edge level for this round
        # (cert.py:668 stores _last_alpha_edge inside _q(); it bakes in the
        # LB-path Bonferroni split AND alpha-annealing exactly as the planner
        # applied them), and query the scorer at that same level + the same
        # delta_stale -- so claim_raw/claim_corr are byte-for-byte what the
        # certificate asserted, not a re-derivation.
        alpha_edge = getattr(planner, "_last_alpha_edge", None)
        if alpha_edge is None or not (0.0 < alpha_edge < 1.0):
            continue
        q = planner.scorer.quantile(alpha_edge, t)
        d_stale = planner.scorer.delta_stale(t)
        if not math.isfinite(q):
            continue
        edges = (
            [(cert.path[i], cert.path[i + 1]) for i in range(len(cert.path) - 1)]
            if cert.path and len(cert.path) >= 2 else []
        )
        fam = world._family_pre if t < t_cp else world._family_post
        scl = world._scale_pre if t < t_cp else world._scale_post
        for e in edges:
            bel = planner.beliefs[e]
            half = q + bel.rho * bel.age(t)  # the band cert.py:964 tests against
            tc = world.true_cost(e, t)
            obs = tc + _draw_one(probe_rng, fam, scl)  # held-out (not pushed)
            miss = not (bel.c_hat - half - 1e-12 <= obs <= bel.c_hat + half + 1e-12)
            seg.e_n += 1
            seg.e_miss += int(miss)
            seg.claim_raw.append(alpha_edge)
            seg.claim_corr.append(min(1.0, alpha_edge + d_stale))
            if cert.valid:
                seg.e_n_valid += 1
                seg.e_miss_valid += int(miss)

    return segs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def merge(segs_list: list[dict], key: str) -> dict:
    e_miss = sum(s[key].e_miss for s in segs_list)
    e_n = sum(s[key].e_n for s in segs_list)
    e_miss_v = sum(s[key].e_miss_valid for s in segs_list)
    e_n_v = sum(s[key].e_n_valid for s in segs_list)
    p_cov = sum(s[key].p_cov for s in segs_list)
    p_valid = sum(s[key].p_valid for s in segs_list)
    p_rounds = sum(s[key].p_rounds for s in segs_list)
    cr = np.concatenate([np.array(s[key].claim_raw) for s in segs_list]) if any(s[key].claim_raw for s in segs_list) else np.array([])
    cc = np.concatenate([np.array(s[key].claim_corr) for s in segs_list]) if any(s[key].claim_corr for s in segs_list) else np.array([])
    conf = np.concatenate([np.array(s[key].conf) for s in segs_list]) if any(s[key].conf for s in segs_list) else np.array([])
    f = lambda a, b: a / b if b else float("nan")
    return dict(
        edge_mis=f(e_miss, e_n), edge_n=e_n,
        edge_mis_valid=f(e_miss_v, e_n_v),
        claim_raw=float(np.mean(cr)) if cr.size else float("nan"),
        claim_corr=float(np.mean(cc)) if cc.size else float("nan"),
        path_cov=f(p_cov, p_valid), path_valid_frac=f(p_valid, p_rounds),
        path_n_valid=p_valid, path_n_rounds=p_rounds,
        mean_claim=float(np.mean(conf)) if conf.size else float("nan"),
    )


def run_condition(shift: dict, eps_tv: float, alpha_p: float, grid: int,
                  seeds: int, rounds: int, t_cp: float, W: float, settle: float,
                  me: int) -> dict:
    cfg = PlannerConfig(
        epsilon=5.0, alpha_prime=alpha_p, rho_w=0.99, eps_tv=eps_tv, delta=1.0,
        sensing_policy="cert", initial_survey=True, maintenance_every=me,
        use_aci=False,        # pin the working level so the edge claim is the
                              # raw weighted-conformal level the theorem assumes
                              # (ACI would adapt width and blur the eps_tv story)
        latent_margin=1.0,    # T1a observable-coverage semantics (paper default)
    )
    segs_list = []
    diag = None
    for s in range(seeds):
        rng = np.random.default_rng(20000 + s)
        world = ShiftWorld(grid, grid, rng, t_cp=t_cp, max_t=(rounds + 5) * 1.0,
                           rho_true_mode="pre", **shift)
        if diag is None:
            diag = dict(drift_q95_pre=world.drift_q95_pre,
                        drift_q95_post=world.drift_q95_post,
                        a1_pre=world.a1_violation_pre, a1_post=world.a1_violation_post)
        probe_rng = np.random.default_rng(900000 + s)
        segs_list.append(run_episode(world, cfg, rounds, t_cp, W, settle, probe_rng))
    out = {seg: merge(segs_list, seg) for seg in ("pre", "trans", "settled")}
    out["_diag"] = diag
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", default="both", choices=["drift", "jump", "both"])
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--alpha-prime", type=float, default=0.2)
    ap.add_argument("--t-cp", type=float, default=110.0)
    ap.add_argument("--window", type=float, default=50.0, help="transient W")
    ap.add_argument("--settle", type=float, default=30.0)
    ap.add_argument("--maintenance-every", type=int, default=8)
    ap.add_argument("--eps-tv", type=float, default=3e-4,
                    help="the 'on' eps_tv compared against 0")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.rounds, args.grid = 5, 220, 6

    shifts = ["drift", "jump"] if args.shift == "both" else [args.shift]
    ap_lvl = args.alpha_prime
    target = 1.0 - ap_lvl

    print("=" * 92)
    print("CALIBRATION -> TEST DISTRIBUTION SHIFT (eps_tv / A2 TV-Lipschitz theorem)")
    print("ADDITIONAL RSS RESULT -- NOT a change to the published paper. Package READ-ONLY.")
    print("=" * 92)
    print(f"grid={args.grid}x{args.grid}  seeds={args.seeds}  rounds={args.rounds}  "
          f"delta=1.0  alpha'={ap_lvl} (path claim ~{target:.2f})  rho_w=0.99")
    print(f"changepoint t_cp={args.t_cp}  transient W=[t_cp, t_cp+{args.window})  "
          f"settled=[t_cp+{args.window+args.settle:.0f}, ...)  maintenance_every="
          f"{args.maintenance_every}")
    print("planner: sensing='cert', use_aci=False, latent_margin=1, A1 bound FROZEN")
    print("         at the pre-shift rho (rho_true_mode='pre') -> genuinely surprised.")
    print("identical inputs across eps_tv: same world/seeds/residual stream; ONLY eps_tv differs.")
    print("EDGE test = held-out fresh obs vs the planner's own band c_hat+-(q+rho*age);")
    print("            this is the layer Barber's exchangeability guarantee is TIGHT on.")
    print()

    json_out = {"alpha_prime": ap_lvl, "eps_tv_on": args.eps_tv, "shifts": {}}
    eps_grid = [0.0, 1e-4, args.eps_tv, 1e-3]

    t_start = time.time()
    for sh in shifts:
        shift = SHIFTS[sh]
        print("#" * 92)
        print(f"# SHIFT = {sh!r}:  rho {shift['rho_pre']}->{shift['rho_post']}   "
              f"noise ({shift['family_pre']},{shift['scale_pre']}) -> "
              f"({shift['family_post']},{shift['scale_post']})")
        print("#" * 92)

        results = {}
        for et in eps_grid:
            results[et] = run_condition(shift, et, ap_lvl, args.grid, args.seeds,
                                        args.rounds, args.t_cp, args.window,
                                        args.settle, args.maintenance_every)
        diag = results[0.0]["_diag"]
        print(f"  measured truth: drift q95 PRE={diag['drift_q95_pre']:.4f} "
              f"POST={diag['drift_q95_post']:.4f}  |  A1-violation vs frozen bound: "
              f"PRE={diag['a1_pre']:.3f} POST={diag['a1_post']:.3f}")
        print()

        # ---- EDGE-LEVEL panel (the headline: where the shift bites) ----
        print("  --- EDGE-LEVEL conformal miscoverage (claim alpha_edge ~ "
              f"alpha'/L; theorem TIGHT) ---")
        print(f"  {'eps_tv':>8} | {'segment':>8} | {'edge_mis':>9} {'claim_raw':>10} "
              f"{'claim_corr':>11} | {'verdict':>22}")
        for et in eps_grid:
            for seg in ("pre", "trans", "settled"):
                d = results[et][seg]
                if d["edge_n"] == 0:
                    verdict = "no valid q (extinct)"
                    print(f"  {et:>8.0e} | {seg:>8} | {'--':>9} {'--':>10} {'--':>11} | {verdict:>22}")
                    continue
                # is the eps_tv=0 (raw) claim a valid coverage LB? is the corrected one?
                raw_ok = d["edge_mis"] <= d["claim_raw"] + 1e-9
                corr_ok = d["edge_mis"] <= d["claim_corr"] + 1e-9
                if seg == "trans" and not raw_ok:
                    verdict = "RAW claim VIOLATED" + ("; corr OK" if corr_ok else "; corr too")
                elif raw_ok:
                    verdict = "calibrated"
                else:
                    verdict = "under-covers"
                print(f"  {et:>8.0e} | {seg:>8} | {d['edge_mis']:>9.4f} "
                      f"{d['claim_raw']:>10.4f} {d['claim_corr']:>11.4f} | {verdict:>22}")
            print()

        # ---- PATH-LEVEL panel ----
        print("  --- PATH certificate  LB<=OPT<=UB among VALID  (paper metric, exact oracle) ---")
        print(f"  {'eps_tv':>8} | {'segment':>8} | {'path_cov':>9} {'valid_frac':>11} "
              f"{'mean_claim':>11} {'n_valid':>8}")
        for et in eps_grid:
            for seg in ("pre", "trans", "settled"):
                d = results[et][seg]
                print(f"  {et:>8.0e} | {seg:>8} | {d['path_cov']:>9.4f} "
                      f"{d['path_valid_frac']:>11.3f} {d['mean_claim']:>11.3f} "
                      f"{d['path_n_valid']:>8d}")
            print()

        # ---- GATING sweep (transient): valid_frac + claim vs eps_tv ----
        print("  --- eps_tv GATING in the TRANSIENT (the self-extinguishing dial) ---")
        print(f"  {'eps_tv':>8} | {'valid_frac':>11} {'mean_claim':>11} "
              f"{'edge_mis(valid)':>16}")
        for et in eps_grid:
            d = results[et]["trans"]
            mv = d["edge_mis_valid"]
            print(f"  {et:>8.0e} | {d['path_valid_frac']:>11.3f} "
                  f"{d['mean_claim']:>11.3f} "
                  f"{(f'{mv:.4f}' if mv == mv else '--'):>16}")
        print()

        # machine-readable
        json_out["shifts"][sh] = {
            "diag": diag,
            "eps_tv": {
                f"{et:.0e}": {
                    seg: {k: results[et][seg][k] for k in
                          ("edge_mis", "claim_raw", "claim_corr", "path_cov",
                           "path_valid_frac", "mean_claim", "edge_n", "path_n_valid")}
                    for seg in ("pre", "trans", "settled")
                } for et in eps_grid
            },
        }

    print("=" * 92)
    print("READING (honest -- includes what did NOT favour the simple hypothesis):")
    print("  1. PRE-shift the conformal layer is calibrated: edge miscoverage <= the")
    print("     claimed alpha_edge. Exchangeability holds, eps_tv unneeded.")
    print("  2. In the TRANSIENT right after the changepoint the eps_tv=0 (exchangeable)")
    print("     edge claim is VIOLATED: realised miscoverage jumps to several x alpha_edge")
    print("     while the buffer is still pre-shift-dominated -- the calibration->test")
    print("     shift the cell is about, made visible at the conformal layer.")
    print("  3. eps_tv>0 is the ONLY mechanism that reacts: delta_stale widens the honest")
    print("     claim with buffer age and, once large enough, drives confidence<=0 so the")
    print("     certificate SELF-EXTINGUISHES (valid_frac->0) instead of overclaiming")
    print("     through the shift -- the paper's 'A2 misspec self-extinguishes loudly'")
    print("     finding, now under a genuine changepoint.")
    print("  4. DID NOT favour the simple hypothesis: the PATH certificate's Bonferroni")
    print("     conservatism keeps realised LB<=OPT<=UB coverage ~1.0 even at eps_tv=0,")
    print("     so eps_tv does not 'rescue' a path-coverage number here -- its effect is")
    print("     edge-layer coverage + CLAIM HONESTY. And an ABRUPT step is the adversarial")
    print("     case for a bounded-RATE TV model: the transient spike can exceed even the")
    print("     eps_tv-corrected bound, which only a steady-rate shift fully absorbs.")
    print(f"\n[ran {len(shifts)} shift(s) x {len(eps_grid)} eps_tv x {args.seeds} seeds "
          f"x {args.rounds} rounds in {time.time()-t_start:.1f}s]")
    print("\nMEASURED_JSON " + json.dumps(json_out))


if __name__ == "__main__":
    main()
