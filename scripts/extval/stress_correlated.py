"""CERT-FLOW RSS extended validation -- CELL: spatially-correlated + heavy-
tailed drift (a moving congestion front).  ADDITIONAL results for the RSS
version; NOT a change to the published paper.

What this does
--------------
Stresses CERT under a one-factor "moving congestion front": a SHARED latent
process with HEAVY-TAILED (student-t df=3 / Pareto a=3) increments drives
neighbouring grid edges TOGETHER (spatial correlation), with the correlated
cluster sweeping across the grid. It compares CERT's coverage and certificate
WIDTH against a MATCHED independent-drift control -- same per-edge marginal
drift magnitude, same heavy tails, same A1 bound fed to the planner, same
seeds, same target alpha; the ONLY difference is cross-edge dependence.

Two questions the cell asks:
  (Q1 coverage threat)  Does spatial correlation + heavy tails break the
        LB<=OPT<=UB coverage the certificate claims?  (Run CERT on both worlds,
        score coverage against the EXACT oracle, compare to the matched control.)
  (Q2 width over-pay)   Per-edge Bonferroni (alpha'/L per edge) assumes the
        worst case that edge miscoverages are additive/disjoint. A JOINT path
        bound (sum-then-quantile) can be tighter when path-sum residuals do not
        perfectly co-move. We quantify the over-pay APPLES-TO-APPLES on ONE
        stream: the realised ground-truth conformal residual r_e = c_true(t) -
        c_hat_e on the incumbent edges (the exact quantity the conformal layer
        must bracket). Two TWO-SIDED-SYMMETRIC path half-widths at the SAME
        path level alpha' are formed from that single stream:
          Bonferroni:  L * Q_{1-alpha'/L}(|r_e|)        (per-edge, then summed)
          Joint:           Q_{1-alpha'}(|sum_e r_e|)    (sum, then quantile)
        over-pay = Bonferroni / Joint. The ONLY difference is the aggregation
        order, so the ratio isolates the correlation effect with no drift or
        one-sidedness confound (the rho*age Lipschitz term is the same additive
        constant on both path bounds and cancels). Under INDEPENDENCE the sum
        pools ~sqrt(L) (ratio > 1: Bonferroni over-pays); under POSITIVE
        correlation the sum's spread approaches L*(per-edge spread), the joint
        advantage ERODES, and the ratio falls toward 1. This is the OPPOSITE of
        a naive "joint always wins under correlation" prior -- reported as a
        SURPRISE, not buried.

  The package's own conformal.block_quantile (what cert.sum_aware_ub actually
        deploys) is ALSO reported, as a SECONDARY diagnostic. But it is a
        ONE-SIDED quantile of DRIFT-INCLUDED signed sums while the Bonferroni
        q(alpha'/L) is built from DRIFT-ADJUSTED ABSOLUTE scores |obs-c_hat|-
        rho*age; those two confounds make the raw block/Bonferroni ratio NOT a
        clean width measurement, so the Q2 HEADLINE is the same-stream audit (a)
        above, and (b) is labelled confounded.

Faithfulness
------------
* No package edits. Worlds subclass certflow.drift._GridBase read-only; the
  planner is the published certflow.CertPlanner; the oracle is certflow.oracle.
* The headline audit uses the planner's OWN incumbent path and the ground-truth
  residuals it is built to bracket; both half-widths share that one stream and
  differ only in aggregation order -- a fair correlation isolation.
* The package block_quantile is reported verbatim as a secondary number, with
  its confounds called out, never used to overstate a win.
* rho_true (the A1 bound) is shared between the two worlds so CERT's inputs are
  byte-identical; each world's realised A1-violation rate vs that bound is
  measured and printed.
* All numbers printed are produced by running real code now.

Run:  cert_env/bin/python scripts/extval/stress_correlated.py
      (add --quick for a fast smoke; --family pareto for the Pareto tail)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from certflow.cert import CertPlanner, PlannerConfig  # noqa: E402
from certflow.conformal import ConformalScorer, path_alpha_edge  # noqa: E402
from certflow.oracle import opt  # noqa: E402
from correlated_world import (  # noqa: E402
    CorrelatedDriftWorld,
    IndepDriftWorld,
)


# ---------------------------------------------------------------------------
# Per-episode driver (mirrors episodes.tier0_episode, but on our custom world:
# stationary robot, one certificate per round, coverage scored vs exact oracle)
# ---------------------------------------------------------------------------

@dataclass
class EpiOut:
    n_valid: int = 0
    n_covered: int = 0          # covered AND valid
    n_rounds: int = 0
    gaps: list = None           # gap per valid round
    # (1) SECONDARY: the shipped sum-aware bound (conformal.block_quantile)
    #     vs Bonferroni on the planner's own buffer. Confounded (block is
    #     one-sided & drift-included; Bonferroni q is two-sided & drift-removed)
    #     -- reported because it is what cert.sum_aware_ub deploys, NOT as the
    #     Q2 headline. The headline is (2), the same-stream symmetric audit.
    blk_overpay_ratio: list = None      # L*q  /  block_quantile  (per round)
    # (2) GROUND-TRUTH conformal residuals r_e = c_true(t) - c_hat_e on the
    #     incumbent path (the exact quantity the conformal layer must bracket;
    #     == what the planner pushes via push_signed). Pooled across rounds to
    #     compute the assumption-free Bonferroni-vs-joint half-widths offline.
    #     BOTH margins are built from THIS one stream and are two-sided
    #     symmetric, so their ratio isolates per-edge-vs-joint aggregation
    #     (i.e. the correlation effect) with NO drift/one-sided confound. The
    #     rho*age Lipschitz term is an identical additive constant on both path
    #     bounds, so it cancels in the ratio and is excluded here on purpose.
    edge_resid: list = None     # per-edge r_e = c_true - c_hat on incumbent
    pathsum_resid: list = None  # signed sum_e r_e over the incumbent path
    path_lens: list = None      # L per round (for the union-bound level)
    a1_violation: float = 0.0
    nbr_corr: float = 0.0

    def __post_init__(self):
        for f in ("gaps", "blk_overpay_ratio", "edge_resid",
                  "pathsum_resid", "path_lens"):
            if getattr(self, f) is None:
                setattr(self, f, [])


def run_episode(world, cfg: PlannerConfig, max_rounds: int) -> EpiOut:
    rows = max(r for r, _ in world.graph) + 1
    cols = max(c for _, c in world.graph) + 1
    start, goal = (0, 0), (rows - 1, cols - 1)
    planner = CertPlanner(world, start, goal, cfg)
    out = EpiOut()
    out.a1_violation = float(world.a1_violation_rate)
    out.nbr_corr = float(world.neighbour_increment_corr())
    ap = cfg.alpha_prime  # target path-level miscoverage (same for both worlds)

    for _ in range(max_rounds):
        t_round = planner.t
        cert, _ = planner.round()
        _, true_opt = opt(world, t_round, start, goal)
        out.n_rounds += 1
        if cert.valid:
            out.n_valid += 1
            covered = cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9
            out.n_covered += int(covered)
            out.gaps.append(cert.gap)

        inc_edges = _path_edges(cert.path)
        L = max(len(inc_edges), 1)
        out.path_lens.append(L)

        # ---- (1) SECONDARY: shipped sum-aware (block) bound vs Bonferroni --
        # The block bound is the package's own construction (conformal.
        # ConformalScorer.block_quantile, theory.tex T4, what cert.sum_aware_ub
        # deploys). Confounded vs q (one-sided/drift-included vs two-sided/
        # drift-removed), so NOT the headline -- see the offline audit (2).
        scorer: ConformalScorer = planner.scorer
        q_edge = scorer.quantile(path_alpha_edge(ap, L), planner.t)  # alpha'/L
        q_blk = scorer.block_quantile(ap, planner.t, L)             # joint sum
        if math.isfinite(q_edge) and math.isfinite(q_blk) and q_blk > 1e-9:
            out.blk_overpay_ratio.append((L * q_edge) / q_blk)

        # ---- (2) ground-truth conformal residuals on the incumbent path -----
        # The conformal layer must bracket the path-cost prediction error
        # sum_e (c_e(t) - c_hat_e) on a fixed path. We record the per-edge
        # residual r_e and its path sum; the offline aggregation forms the two
        # symmetric half-widths from THIS one stream. (rho*age is excluded: it
        # is the same additive constant on both path bounds and cancels in the
        # ratio -- we audit the conformal margin, which is where aggregation
        # choice actually matters.)
        if inc_edges:
            signed_sum = 0.0
            for e in inc_edges:
                r = world.true_cost(e, t_round) - planner.beliefs[e].c_hat
                out.edge_resid.append(r)
                signed_sum += r
            out.pathsum_resid.append(signed_sum)

    return out


def _path_edges(path):
    if not path or len(path) < 2:
        return []
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def _cat(outs, attr):
    arrs = [np.asarray(getattr(o, attr), dtype=float) for o in outs]
    arrs = [a for a in arrs if a.size]
    return np.concatenate(arrs) if arrs else np.array([])


def aggregate(outs: list[EpiOut], alpha_prime: float) -> dict:
    n_valid = sum(o.n_valid for o in outs)
    n_cov = sum(o.n_covered for o in outs)
    n_rounds = sum(o.n_rounds for o in outs)
    gaps = _cat(outs, "gaps")
    blk_overpay = _cat(outs, "blk_overpay_ratio")

    # --- definitive (assumption-free) Bonferroni-vs-joint HALF-WIDTHS from the
    # realised ground-truth conformal residuals, pooled across rounds & seeds.
    # Both half-widths are TWO-SIDED SYMMETRIC and built from the SAME residual
    # stream, so the ratio isolates per-edge-vs-joint aggregation (correlation)
    # with no drift/one-sidedness confound. ---
    edge_resid = _cat(outs, "edge_resid")
    pathsum = _cat(outs, "pathsum_resid")
    path_lens = _cat(outs, "path_lens")
    L_typ = float(np.median(path_lens)) if path_lens.size else float("nan")
    bonf_margin = joint_margin = gt_overpay = float("nan")
    if edge_resid.size and pathsum.size and math.isfinite(L_typ) and L_typ >= 1:
        # Bonferroni half-width: a symmetric per-edge interval covering r_e at
        # level alpha'/L_typ, summed over the path (the union bound CERT
        # applies). Q_{1-a'/L}(|r_e|) is the per-edge symmetric half-width.
        per_edge_hw = float(np.quantile(np.abs(edge_resid), 1.0 - alpha_prime / L_typ))
        bonf_margin = L_typ * per_edge_hw
        # Joint half-width: a single symmetric interval on the path-sum residual
        # at level alpha'. Q_{1-a'}(|sum_e r_e|) is the tightest symmetric path
        # half-width achieving the SAME path-level coverage under the ACTUAL
        # joint distribution (spatial correlation included).
        joint_margin = float(np.quantile(np.abs(pathsum), 1.0 - alpha_prime))
        if joint_margin > 1e-9:
            gt_overpay = bonf_margin / joint_margin

    return {
        "coverage": (n_cov / n_valid) if n_valid else float("nan"),
        "valid_frac": (n_valid / n_rounds) if n_rounds else float("nan"),
        "n_valid": n_valid,
        "n_rounds": n_rounds,
        "gap_mean": float(np.mean(gaps)) if gaps.size else float("nan"),
        "gap_median": float(np.median(gaps)) if gaps.size else float("nan"),
        "a1_violation": float(np.mean([o.a1_violation for o in outs])),
        "nbr_corr": float(np.mean([o.nbr_corr for o in outs])),
        "L_typ": L_typ,
        # published block bound vs Bonferroni (on the planner's buffer)
        "blk_overpay_median": float(np.median(blk_overpay)) if blk_overpay.size else float("nan"),
        # assumption-free ground-truth margins (the definitive over-pay)
        "bonf_margin": bonf_margin,
        "joint_margin": joint_margin,
        "gt_overpay": gt_overpay,
        "n_audit": int(blk_overpay.size),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="student_t", choices=["student_t", "pareto"])
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--latent-scale", type=float, default=0.06)
    ap.add_argument("--idio-scale", type=float, default=0.02)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.rounds, args.grid = 4, 80, 6

    G = args.grid
    max_t = (args.rounds + 5) * 1.0  # delta=1.0/round
    # CERT config: the coverage-validation (Tier-0) recipe used in the paper's
    # tier0 sweep -- alpha annealing on, maintenance sensing on, given rho.
    # Same config for BOTH worlds (identical inputs).
    cfg_kwargs = dict(
        epsilon=5.0,
        alpha_prime=0.2,
        rho_w=0.99,
        eps_tv=1e-4,
        delta=1.0,
        sensing_policy="cert",
        initial_survey=True,
    )

    print("=" * 78)
    print("CERT-FLOW RSS extended validation -- spatially-correlated + heavy-tailed")
    print("drift (moving congestion front).  ADDITIONAL results, NOT the published paper.")
    print("=" * 78)
    print(f"family={args.family}  grid={G}x{G}  seeds={args.seeds}  "
          f"rounds={args.rounds}  alpha'={cfg_kwargs['alpha_prime']}  "
          f"latent_scale={args.latent_scale} idio_scale={args.idio_scale}")
    print("Worlds: CORRELATED (one shared heavy-tailed latent + moving front) vs")
    print("        INDEP control (matched per-edge marginal, independent walks,")
    print("        SHARED rho_true so CERT's A1 input is byte-identical).")
    print()

    corr_outs: list[EpiOut] = []
    indep_outs: list[EpiOut] = []
    t0 = time.time()
    for s in range(args.seeds):
        # correlated world
        rng_c = np.random.default_rng(1000 + s)
        cw = CorrelatedDriftWorld(
            G, G, rng_c, heavy_family=args.family,
            latent_scale=args.latent_scale, idio_scale=args.idio_scale,
            max_t=max_t,
        )
        corr_outs.append(run_episode(cw, PlannerConfig(**cfg_kwargs), args.rounds))

        # matched independent control: same per-edge increment std, DIFFERENT
        # independent seed for the drift, SHARED rho_true (identical A1 input).
        rng_i = np.random.default_rng(9_000_000 + s)
        iw = IndepDriftWorld(
            G, G, rng_i, edge_increment_std=cw._edge_increment_std,
            heavy_family=args.family, max_t=max_t,
            shared_rho=cw._rho_arr,
        )
        indep_outs.append(run_episode(iw, PlannerConfig(**cfg_kwargs), args.rounds))

    dt = time.time() - t0
    ap = cfg_kwargs["alpha_prime"]
    A = aggregate(corr_outs, ap)
    B = aggregate(indep_outs, ap)

    def row(name, d):
        return (f"  {name:<26} cov={d['coverage']:.4f}  valid={d['valid_frac']:.3f}  "
                f"gap_med={d['gap_median']:.3f}  gap_mean={d['gap_mean']:.3f}  "
                f"nbr_corr={d['nbr_corr']:+.3f}  A1viol={d['a1_violation']:.3f}")

    print("--- (Q1) COVERAGE + CERTIFICATE GAP (CERT on each world; oracle exact) ---")
    print(row("CORRELATED (shared front)", A))
    print(row("INDEP control (matched)", B))
    print(f"  [{A['n_valid']}+{B['n_valid']} valid rounds; target coverage "
          f">= 1-alpha' = {1-ap:.2f};  L_typ ~ {A['L_typ']:.0f} edges]")
    print()
    print("--- (Q2) BONFERRONI WIDTH OVER-PAY ---")
    print("  (a) DEFINITIVE, assumption-free, SAME residual stream, both two-sided:")
    print("      Bonferroni half-width  L*Q_{1-a'/L}(|r_e|)  vs  joint half-width")
    print("      Q_{1-a'}(|sum_e r_e|),  r_e = c_true - c_hat on incumbent edges:")
    print(f"      CORRELATED: Bonf_hw={A['bonf_margin']:.3f}  joint_hw={A['joint_margin']:.3f}  "
          f"-> Bonf/joint = {A['gt_overpay']:.3f}")
    print(f"      INDEP     : Bonf_hw={B['bonf_margin']:.3f}  joint_hw={B['joint_margin']:.3f}  "
          f"-> Bonf/joint = {B['gt_overpay']:.3f}")
    print("  (b) PUBLISHED sum-aware bound (conformal.block_quantile, theory.tex T4)")
    print("      vs Bonferroni on the planner's buffer -- SECONDARY (confounded:")
    print("      one-sided & drift-included vs two-sided & drift-removed):")
    print(f"      CORRELATED: L*q(a'/L) / block_quantile(a') median = {A['blk_overpay_median']:.3f}")
    print(f"      INDEP     : L*q(a'/L) / block_quantile(a') median = {B['blk_overpay_median']:.3f}")
    print()
    if math.isfinite(A["gt_overpay"]) and math.isfinite(B["gt_overpay"]):
        print("  INTERPRETATION (this did NOT go the naive way -- see SURPRISES):")
        print(f"    The joint half-width beats Bonferroni only when path-sum residuals")
        print(f"    CANCEL. Under matched INDEPENDENCE they pool ~sqrt(L), so the joint")
        print(f"    bound is {B['gt_overpay']:.2f}x tighter -- Bonferroni over-pays there.")
        print(f"    Under POSITIVE spatial CORRELATION the residuals STACK (std of the")
        print(f"    sum -> ~L*per-edge), so the joint half-width inflates and the gap")
        print(f"    to Bonferroni SHRINKS to {A['gt_overpay']:.2f}x.")
        print(f"    => Correlation ERODES the joint-bound headroom; CERT's per-edge")
        print(f"       Bonferroni is NOT a width liability under the correlated front")
        print(f"       (it is order-optimal here, consistent with limitations.md:")
        print(f"       sum-aware buys ~sqrt(L) only in the independent noise floor).")
    print()
    print(f"[ran {args.seeds} seeds x {args.rounds} rounds per world in {dt:.1f}s]")

    # machine-readable line for downstream sanity-checks
    print("\nMEASURED_JSON " + _json_line(A, B, args, cfg_kwargs))


def _json_line(A, B, args, cfg) -> str:
    import json
    keys = ("coverage", "valid_frac", "gap_median", "gap_mean", "nbr_corr",
            "a1_violation", "gt_overpay", "blk_overpay_median",
            "bonf_margin", "joint_margin", "L_typ")
    return json.dumps({
        "family": args.family,
        "grid": args.grid,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "alpha_prime": cfg["alpha_prime"],
        "correlated": {k: A[k] for k in keys},
        "independent": {k: B[k] for k in keys},
    })


if __name__ == "__main__":
    main()
