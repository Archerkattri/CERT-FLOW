"""Certified-loop scaling curve (RSS-version EXTENDED VALIDATION).

ADDITIONAL results for the RSS version of the CERT-FLOW paper. NOT a change to
the published paper. Nothing here modifies src/certflow -- the published package
is imported READ-ONLY (certflow.cert, certflow.drift, certflow.oracle).

------------------------------------------------------------------------------
CELL: Certified-loop scaling curve
------------------------------------------------------------------------------
An honest scaling curve for the FULL certified loop (conformal + dual D* Lite +
pre-widening + sensing) on synthetic bounded-drift grids of increasing size
(20x20 .. 100x100). Two things are measured, both produced by running real code
now (never hardcoded), and both reported even where they do NOT favor CERT:

  (A) Per-round steady-state latency p50/p95 vs |E| (and path length L). This
      is the recurring cost of one planner.round() once the numba kernel is
      warm; it is the operative real-time budget question.

  (B) The Bonferroni-LB WARM-UP cost vs L / |E|: rounds-to-first-valid and the
      certificate gap at that point, plus rounds-to-target-alpha (when the
      annealed claim reaches the full alpha' target). This is the headline of
      the cell: the per-edge Bonferroni level alpha'/L shrinks with the path
      length L, and L grows with the grid diameter (~sqrt(|V|), |E| ~ |V|), so
      the calibration buffer must hold ~L/alpha' scores before the claim is at
      target -- a warm-up that grows with L.

We run the warm-up sweep under TWO calibration regimes so the L-scaling is not
confounded with the buffer's effective-sample-size (ESS) ceiling:

  * recommended  -- recommended_config(), rho_w=0.99. The age-geometric weight
    rho_w^age caps the effective sample size at ESS ~ 1/(1-rho_w) ~ 100. When
    L exceeds ESS the certificate is STRUCTURALLY unreachable (m can never
    exceed L-1). This is an honest negative: at large grids recommended CERT
    never exits warm-up within any horizon. It is the deployed config, so it is
    the primary latency config too.

  * high-ESS     -- recommended_config() but rho_w=0.9995 (ESS ~ 2000), which
    lifts the ceiling above every L tested. This ISOLATES the pure Bonferroni
    L-scaling of rounds-to-first-valid and rounds-to-target-alpha. It is a
    faithful CERT configuration (rho_w is a published knob), chosen here to
    expose the L-dependence rather than the ESS cap.

Faithfulness / fairness:
  * World, start/goal, seeds, alpha', epsilon are IDENTICAL across the regimes
    being compared at a given size; only the axis under study changes.
  * Numbers are printed, not hardcoded. Latency uses time.perf_counter around
    planner.round() with ONE warm round discarded (the published convention:
    numba kernel compiled out of the timed region; docs/results/scale.md).
  * Where a size is too slow we cap the rounds and SAY SO (see CAPS below).

NO new downloads. Synthetic grids only (DIMACS preprocessing scale is covered
separately by scripts/run_roadnet.py / run_scale.py and is not re-run here).

Usage
-----
    cert_env/bin/python scripts/extval/scaling.py            # full curve
    cert_env/bin/python scripts/extval/scaling.py --quick    # smoke
    cert_env/bin/python scripts/extval/scaling.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from certflow.cert import CertPlanner, recommended_config  # noqa: E402
from certflow.drift import grid_world  # noqa: E402
from certflow.oracle import opt  # noqa: E402

# ---------------------------------------------------------------------------
# Sweep parameters (shared, identical across regimes)
# ---------------------------------------------------------------------------
SIZES_FULL = [(20, 20), (40, 40), (60, 60), (80, 80), (100, 100)]
SIZES_QUICK = [(20, 20), (40, 40)]
SEEDS_FULL = [2026, 2027, 2028]
SEEDS_QUICK = [2026]

# Shared world / planner knobs (the recommended config under bounded drift).
RHO = 0.02
NOISE_SCALE = 0.05
EPSILON = 5.0
ALPHA_PRIME = 0.1

# Latency: steady-state rounds timed AFTER one discarded warm round (numba JIT).
N_LATENCY_ROUNDS = 120
N_WARM_ROUNDS = 1

# Warm-up horizon: how many rounds we let the loop run looking for first-valid
# and target-alpha. CAP: large grids never exit warm-up under rho_w=0.99 (ESS
# ceiling), and even the high-ESS regime needs ~L/alpha' rounds which is ~1180
# at L~118; we cap the horizon and report "not reached within cap" honestly
# rather than running unboundedly. Capping rounds is the documented tradeoff.
# 1200 rounds reaches RTTA (~L/alpha') for L<=120 (grids up to ~60x60) and
# RTFV (~L) for all sizes in the high-ESS regime; 80x80/100x100 RTTA (needs
# ~1580/1980 rounds) is reported as capped.
WARMUP_HORIZON_FULL = 1200
WARMUP_HORIZON_QUICK = 200

# ESS ceiling note: ESS ~ 1/(1-rho_w).
RHO_W_RECOMMENDED = 0.99      # ESS ~ 100  (the deployed default)
RHO_W_HIGH_ESS = 0.9995       # ESS ~ 2000 (isolates Bonferroni L-scaling)


def n_edges_of(rows: int, cols: int) -> int:
    """4-connected directed grid edge count: 2*(R*(C-1)+(R-1)*C)."""
    return 2 * (rows * (cols - 1) + (rows - 1) * cols)


# ---------------------------------------------------------------------------
# (A) Latency measurement
# ---------------------------------------------------------------------------
@dataclass
class LatencyCell:
    size: str
    n_edges: int
    L_typ: float          # median LB-path length over timed rounds
    seeds: list[int]
    n_rounds_timed: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    init_ms_median: float  # one-time setup (survey + first searches + JIT)


def measure_latency(rows: int, cols: int, seeds: list[int],
                    n_rounds: int) -> LatencyCell:
    """Steady-state per-round latency for the recommended config.

    One warm round is run and discarded (numba kernel compiles out of the timed
    region -- the published convention). The oracle is NOT in the timed region.
    """
    all_walls: list[float] = []
    all_L: list[int] = []
    inits: list[float] = []
    for seed in seeds:
        world = grid_world(rows, cols, seed=seed, kind="bounded",
                           rho=RHO, noise_scale=NOISE_SCALE)
        start, goal = (0, 0), (rows - 1, cols - 1)
        cfg = recommended_config(epsilon=EPSILON, alpha_prime=ALPHA_PRIME,
                                 rho_w=RHO_W_RECOMMENDED, prewiden_rounds=10,
                                 k_alternatives=3)
        t0 = time.perf_counter()
        planner = CertPlanner(world, start, goal, cfg)
        # warm rounds (discarded): JIT + cache fill
        for _ in range(N_WARM_ROUNDS):
            planner.round()
        inits.append((time.perf_counter() - t0) * 1000.0)
        for _ in range(n_rounds):
            w0 = time.perf_counter()
            cert, _ = planner.round()
            all_walls.append(time.perf_counter() - w0)
            if cert.path:
                all_L.append(len(cert.path) - 1)
    walls_ms = np.array(all_walls) * 1000.0
    return LatencyCell(
        size=f"{rows}x{cols}",
        n_edges=n_edges_of(rows, cols),
        L_typ=float(np.median(all_L)) if all_L else float("nan"),
        seeds=seeds,
        n_rounds_timed=len(all_walls),
        p50_ms=float(np.percentile(walls_ms, 50)),
        p95_ms=float(np.percentile(walls_ms, 95)),
        p99_ms=float(np.percentile(walls_ms, 99)),
        init_ms_median=float(np.median(inits)),
    )


# ---------------------------------------------------------------------------
# (B) Bonferroni warm-up measurement
# ---------------------------------------------------------------------------
@dataclass
class WarmupCell:
    size: str
    n_edges: int
    regime: str            # "recommended" | "high_ESS"
    rho_w: float
    ess_cap: float         # 1/(1-rho_w)
    seeds: list[int]
    horizon: int
    L_median: float        # median LB-path length (the Bonferroni denominator)
    # rounds-to-first-valid: first round with cert.valid (annealed weakest claim)
    rtfv_median: float     # nan if no seed reached it within horizon
    rtfv_reached_frac: float
    gap_at_first_valid_median: float
    conf_at_first_valid_median: float
    # rounds-to-target-alpha: claim annealed back to the full alpha' target
    rtta_median: float     # nan if not reached within horizon
    rtta_reached_frac: float
    gap_at_target_median: float
    # diagnostics
    eff_mass_max_median: float   # max effective sample size observed (the ESS plateau)
    scores_needed_first_valid: float  # ~ L (m must exceed L-1)
    scores_needed_target: float       # ~ L/alpha'
    # True when first-valid is unreachable because the ESS ceiling < L-1
    # (a structural limit, not merely a too-short horizon).
    ess_ceiling_blocks_valid: bool = False


def _run_warmup_seed(rows: int, cols: int, seed: int, rho_w: float,
                     horizon: int) -> dict:
    """One seed's warm-up trace. Returns the milestone rounds + gaps.

    first-valid: cert.valid (confidence>0) first becomes true. With annealing,
    this is the round where the buffer's effective mass m makes the supportable
    path level L/(m+1) drop below 1 (so the weakest claim becomes emittable).

    target-alpha: the annealed claim (planner._alpha_claim) reaches alpha' to
    within 1e-9, i.e. L/(m+1) <= alpha' -> m >= L/alpha' - 1. This is the round
    the certificate stops being weakened by the warm-up annealer.
    """
    world = grid_world(rows, cols, seed=seed, kind="bounded",
                       rho=RHO, noise_scale=NOISE_SCALE)
    start, goal = (0, 0), (rows - 1, cols - 1)
    cfg = recommended_config(epsilon=EPSILON, alpha_prime=ALPHA_PRIME,
                             rho_w=rho_w, prewiden_rounds=10, k_alternatives=3)
    planner = CertPlanner(world, start, goal, cfg)

    first_valid = None
    gap_fv = math.nan
    conf_fv = math.nan
    target_round = None
    gap_target = math.nan
    L_samples: list[int] = []
    eff_mass_max = 0.0
    ess_cap = 1.0 / (1.0 - rho_w)
    plateau_count = 0
    prev_m = -1.0
    capped_unreachable = False
    for rnd in range(horizon):
        cert, _ = planner.round()
        L = len(cert.path) - 1 if cert.path else 0
        if L > 0:
            L_samples.append(L)
        m = planner.scorer.effective_mass(planner.t)
        eff_mass_max = max(eff_mass_max, m)
        if cert.valid and first_valid is None:
            first_valid = rnd
            gap_fv = cert.gap
            conf_fv = cert.confidence
        # _alpha_claim is set inside _q each round; target reached when it has
        # annealed back down to the configured alpha' (no longer weakened).
        ac = getattr(planner, "_alpha_claim", ALPHA_PRIME)
        if target_round is None and ac <= ALPHA_PRIME + 1e-9 and cert.valid:
            target_round = rnd
            gap_target = cert.gap
        if first_valid is not None and target_round is not None:
            # both milestones found; L is stable -- stop early.
            break
        # Honest early-stop when first-valid is structurally UNREACHABLE: the
        # effective mass has plateaued at the ESS ceiling (m ~ 1/(1-rho_w)) and
        # that ceiling is below the L-1 the weakest annealed claim needs. Once
        # m stops growing for many rounds and m < L-1 with no valid claim yet,
        # running the rest of the horizon cannot change the verdict. We mark it
        # capped_unreachable so the caller knows this is a ceiling, not a
        # too-short horizon.
        if first_valid is None and L > 0:
            if abs(m - prev_m) < 1e-6 * max(prev_m, 1.0):
                plateau_count += 1
            else:
                plateau_count = 0
            prev_m = m
            if (plateau_count >= 40 and m < L - 1
                    and ess_cap < L - 1 and rnd > 5):
                capped_unreachable = True
                break
    L_median = float(np.median(L_samples)) if L_samples else float("nan")
    return {
        "first_valid": first_valid,
        "gap_fv": gap_fv,
        "conf_fv": conf_fv,
        "target_round": target_round,
        "gap_target": gap_target,
        "L_median": L_median,
        "eff_mass_max": eff_mass_max,
        "capped_unreachable": capped_unreachable,
    }


def measure_warmup(rows: int, cols: int, seeds: list[int], rho_w: float,
                   regime: str, horizon: int) -> WarmupCell:
    runs = [_run_warmup_seed(rows, cols, s, rho_w, horizon) for s in seeds]

    rtfv_reached = [r for r in runs if r["first_valid"] is not None]
    rtta_reached = [r for r in runs if r["target_round"] is not None]
    L_med = float(np.median([r["L_median"] for r in runs
                             if not math.isnan(r["L_median"])])) \
        if any(not math.isnan(r["L_median"]) for r in runs) else float("nan")

    return WarmupCell(
        size=f"{rows}x{cols}",
        n_edges=n_edges_of(rows, cols),
        regime=regime,
        rho_w=rho_w,
        ess_cap=1.0 / (1.0 - rho_w),
        seeds=seeds,
        horizon=horizon,
        L_median=L_med,
        rtfv_median=float(np.median([r["first_valid"] for r in rtfv_reached]))
        if rtfv_reached else float("nan"),
        rtfv_reached_frac=len(rtfv_reached) / len(runs),
        gap_at_first_valid_median=float(np.median(
            [r["gap_fv"] for r in rtfv_reached])) if rtfv_reached else float("nan"),
        conf_at_first_valid_median=float(np.median(
            [r["conf_fv"] for r in rtfv_reached])) if rtfv_reached else float("nan"),
        rtta_median=float(np.median([r["target_round"] for r in rtta_reached]))
        if rtta_reached else float("nan"),
        rtta_reached_frac=len(rtta_reached) / len(runs),
        gap_at_target_median=float(np.median(
            [r["gap_target"] for r in rtta_reached])) if rtta_reached else float("nan"),
        eff_mass_max_median=float(np.median([r["eff_mass_max"] for r in runs])),
        scores_needed_first_valid=L_med if not math.isnan(L_med) else float("nan"),
        scores_needed_target=L_med / ALPHA_PRIME if not math.isnan(L_med) else float("nan"),
        ess_ceiling_blocks_valid=any(r.get("capped_unreachable") for r in runs),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(x: float, nd: int = 2, inf_str: str = "  inf") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    if isinstance(x, float) and math.isinf(x):
        return inf_str
    return f"{x:.{nd}f}"


def print_latency_table(cells: list[LatencyCell]) -> None:
    print("\n" + "=" * 92)
    print("(A) PER-ROUND STEADY-STATE LATENCY  (recommended_config, rho_w=0.99, "
          f"{N_WARM_ROUNDS} warm round discarded)")
    print("-" * 92)
    print(f"{'size':<9}{'|E|':>8}{'L_typ':>8}{'rounds':>8}"
          f"{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'init ms':>10}")
    for c in cells:
        print(f"{c.size:<9}{c.n_edges:>8}{_fmt(c.L_typ, 0):>8}{c.n_rounds_timed:>8}"
              f"{_fmt(c.p50_ms):>10}{_fmt(c.p95_ms):>10}{_fmt(c.p99_ms):>10}"
              f"{_fmt(c.init_ms_median, 0):>10}")
    print("=" * 92)


def print_warmup_table(cells: list[WarmupCell]) -> None:
    print("\n" + "=" * 118)
    print("(B) BONFERRONI-LB WARM-UP COST vs L / |E|")
    print("    rounds-to-first-valid (RTFV, annealed weakest claim) and "
          "rounds-to-target-alpha (RTTA, claim back at alpha')")
    print("-" * 118)
    print(f"{'size':<9}{'|E|':>7}{'regime':>13}{'ESS_cap':>9}{'L':>5}"
          f"{'RTFV':>7}{'gap@FV':>9}{'RTTA':>7}{'gap@TA':>9}"
          f"{'m_max':>8}{'~L scores':>10}{'~L/a scores':>12}")
    for c in cells:
        if c.rtfv_reached_frac > 0:
            rtfv = _fmt(c.rtfv_median, 0)
        elif c.ess_ceiling_blocks_valid:
            rtfv = " CEIL"   # structurally unreachable: ESS cap < L-1
        else:
            rtfv = " >cap"   # not reached within horizon (would need more rounds)
        rtta = _fmt(c.rtta_median, 0) if c.rtta_reached_frac > 0 else " >cap"
        print(f"{c.size:<9}{c.n_edges:>7}{c.regime:>13}{_fmt(c.ess_cap, 0):>9}"
              f"{_fmt(c.L_median, 0):>5}{rtfv:>7}"
              f"{_fmt(c.gap_at_first_valid_median, 1):>9}{rtta:>7}"
              f"{_fmt(c.gap_at_target_median, 1):>9}"
              f"{_fmt(c.eff_mass_max_median, 0):>8}"
              f"{_fmt(c.scores_needed_first_valid, 0):>10}"
              f"{_fmt(c.scores_needed_target, 0):>12}")
    print("=" * 118)
    print("RTFV: 'CEIL' = structurally unreachable (ESS ceiling 1/(1-rho_w) < L-1);")
    print("      '>cap' = not reached within the horizon (needs more rounds, not blocked).")
    print("RTTA '>cap' = claim never annealed back to alpha' within horizon. gap vs epsilon=5.0.")


def main() -> None:
    ap = argparse.ArgumentParser(description="CERT certified-loop scaling curve")
    ap.add_argument("--quick", action="store_true", help="smoke: 2 sizes, 1 seed")
    ap.add_argument("--json", type=str, default=None,
                    help="write machine-readable results to this path")
    args = ap.parse_args()

    sizes = SIZES_QUICK if args.quick else SIZES_FULL
    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL
    warm_horizon = WARMUP_HORIZON_QUICK if args.quick else WARMUP_HORIZON_FULL
    n_lat = 40 if args.quick else N_LATENCY_ROUNDS

    print(f"[scaling] EXTENDED VALIDATION (RSS additional result) "
          f"mode={'quick' if args.quick else 'full'}")
    print(f"[scaling] sizes={[f'{r}x{c}' for r, c in sizes]} seeds={seeds}")
    print(f"[scaling] bounded drift rho={RHO} noise_scale={NOISE_SCALE} "
          f"epsilon={EPSILON} alpha'={ALPHA_PRIME}")
    print(f"[scaling] latency rounds/seed={n_lat} (after {N_WARM_ROUNDS} warm); "
          f"warm-up horizon={warm_horizon} rounds")

    # ---- (A) latency ----
    lat_cells: list[LatencyCell] = []
    for rows, cols in sizes:
        t0 = time.perf_counter()
        cell = measure_latency(rows, cols, seeds, n_lat)
        lat_cells.append(cell)
        print(f"  [lat] {cell.size} |E|={cell.n_edges} "
              f"p50={cell.p50_ms:.2f}ms p95={cell.p95_ms:.2f}ms "
              f"({time.perf_counter() - t0:.0f}s)")

    # ---- (B) warm-up, two regimes ----
    warm_cells: list[WarmupCell] = []
    for regime, rho_w in (("recommended", RHO_W_RECOMMENDED),
                          ("high_ESS", RHO_W_HIGH_ESS)):
        for rows, cols in sizes:
            t0 = time.perf_counter()
            cell = measure_warmup(rows, cols, seeds, rho_w, regime, warm_horizon)
            warm_cells.append(cell)
            rtfv = "none" if cell.rtfv_reached_frac == 0 else f"{cell.rtfv_median:.0f}"
            rtta = "none" if cell.rtta_reached_frac == 0 else f"{cell.rtta_median:.0f}"
            print(f"  [warm:{regime}] {cell.size} L={cell.L_median:.0f} "
                  f"RTFV={rtfv} RTTA={rtta} m_max={cell.eff_mass_max_median:.0f} "
                  f"({time.perf_counter() - t0:.0f}s)")

    print_latency_table(lat_cells)
    print_warmup_table(warm_cells)

    # ---- scaling-fit summary (printed, computed from the measured numbers) ----
    print("\n--- scaling fits (computed from the rows above) ---")
    if len(lat_cells) >= 2:
        a, b = lat_cells[0], lat_cells[-1]
        eratio = b.n_edges / a.n_edges
        p50ratio = b.p50_ms / a.p50_ms if a.p50_ms > 0 else float("nan")
        print(f"latency p50: {a.size}->{b.size} is {p50ratio:.1f}x for a "
              f"{eratio:.0f}x |E| growth (sublinear if <{eratio:.0f}x).")
    he = [c for c in warm_cells if c.regime == "high_ESS"
          and c.rtfv_reached_frac > 0]
    if len(he) >= 2:
        a, b = he[0], he[-1]
        print(f"warm-up RTFV (high-ESS, L-scaling isolated): "
              f"{a.size} L={a.L_median:.0f} -> {b.size} L={b.L_median:.0f}; "
              f"RTFV {a.rtfv_median:.0f} -> {b.rtfv_median:.0f} rounds "
              f"(ratio {b.rtfv_median / a.rtfv_median:.1f}x vs "
              f"L ratio {b.L_median / a.L_median:.1f}x).")
    rec_ceiling = [c for c in warm_cells if c.regime == "recommended"
                   and c.ess_ceiling_blocks_valid]
    if rec_ceiling:
        sizes_u = ", ".join(f"{c.size}(L={c.L_median:.0f})" for c in rec_ceiling)
        print(f"recommended (rho_w=0.99, ESS~100): first-valid STRUCTURALLY "
              f"unreachable at {sizes_u} -- L exceeds the ESS ceiling, so the "
              f"weakest annealed claim can never be supported (honest negative).")

    if args.json:
        payload = {
            "cell": "certified-loop scaling curve (RSS extended validation)",
            "params": {
                "sizes": [f"{r}x{c}" for r, c in sizes], "seeds": seeds,
                "rho": RHO, "noise_scale": NOISE_SCALE, "epsilon": EPSILON,
                "alpha_prime": ALPHA_PRIME, "latency_rounds": n_lat,
                "warmup_horizon": warm_horizon,
                "rho_w_recommended": RHO_W_RECOMMENDED,
                "rho_w_high_ess": RHO_W_HIGH_ESS,
            },
            "latency": [asdict(c) for c in lat_cells],
            "warmup": [asdict(c) for c in warm_cells],
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n[scaling] machine-readable results -> {args.json}")


if __name__ == "__main__":
    main()
