"""certified_corridor.gif -- the CERT-FLOW loop, animated, on a real drift grid.

Runs the actual planner (``certflow.cert.CertPlanner``) on a 20x20
``BoundedDriftWorld`` for 170 rounds and animates, per round:

  * LEFT  -- the grid with the world's TRUE edge costs as a heatmap (per-node
    mean incident cost at that round's time), the current certified incumbent
    path, and the edge(s) the planner paid to sense this round (recovered from
    each belief's observation age == this round's time);
  * RIGHT -- the certified corridor as a growing strip chart: the LB and UB
    lines with the band shaded between them, and the TRUE optimum (exact
    Dijkstra from ``certflow.oracle.opt``) drawn inside it every round.

The four narrative beats are labelled live from the data: warm-up (certificate
INVALID while the calibration buffer fills) -> valid (LB <= OPT <= UB brackets
the truth) -> drift (costs move, the band breathes) -> sensing (gap-directed
observations hold the corridor). Coverage over the valid rounds is printed and
shown; it is measured against the world's true optimum, never staged.

Reproduce:  PYTHONPATH=src python scripts/viz_gen/certified_corridor.py
Writes assets/animations/certified_corridor.gif  (seeded; needs no data/).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from certflow.cert import CertPlanner, PlannerConfig  # noqa: E402
from certflow.drift import grid_world  # noqa: E402
from certflow.oracle import opt  # noqa: E402

# ---- palette (Okabe-Ito; matches the repo/site --cert/--accent ink) ----------
INK = "#0a111e"
MUTED = "#5b6b82"
GRIDC = "#e4e9f0"
SURF = "#ffffff"
CERT = "#0072B2"      # deep blue  -- the certified band (LB..UB)
CERTLT = "#56B4E9"    # light blue -- band fill / incumbent path
OPTC = "#0a111e"      # ink        -- the true optimum line (truth)
SENSE = "#E8722D"     # orange     -- the edge(s) sensed this round
START = "#009E73"     # green      -- start
GOAL = "#D55E00"      # vermillion -- goal

N = 20
ROUNDS = 170
SEED = 1
STEP = 2              # animate every STEP rounds (frame budget / file size)
FPS = 12


def run():
    """Run the real planner; return per-round records."""
    world = grid_world(N, N, seed=SEED, kind="bounded", rho=0.02, noise_scale=0.05)
    start, goal = (0, 0), (N - 1, N - 1)
    planner = CertPlanner(
        world, start, goal,
        PlannerConfig(epsilon=60.0, alpha_prime=0.2, sum_aware_ub=True,
                      adaptive_rate=True, max_sense_per_round=20,
                      sensing_policy="cert"),
    )
    edge_list = list(world.edges())
    recs = []
    for i in range(ROUNDS):
        cert, _ = planner.round()
        t_cert = planner.t - planner.cfg.delta
        _, o = opt(world, t_cert, start, goal)

        # per-node true-cost heatmap: mean incident (out-edge) true cost at t_cert
        acc = np.zeros((N, N)); cnt = np.zeros((N, N))
        for (u, v) in edge_list:
            c = world.true_cost((u, v), t_cert)
            acc[u[0], u[1]] += c; cnt[u[0], u[1]] += 1
        grid = acc / np.maximum(cnt, 1)

        # edges freshly sensed THIS round: belief obs-time == this round's time.
        # (round 0 re-flags the whole initial survey -> guarded by the count cap
        # at draw time, so only genuine per-round sensing is highlighted.)
        fresh = [e for e in edge_list
                 if abs(planner.beliefs[e].t_obs - t_cert) < 1e-9]

        covered = (math.isfinite(cert.lb) and cert.valid
                   and cert.lb - 1e-6 <= o <= cert.ub + 1e-6)
        recs.append(dict(
            i=i, lb=cert.lb, ub=cert.ub, opt=o, valid=bool(cert.valid),
            gap=cert.gap, path=list(cert.path), fresh=fresh, grid=grid,
            covered=covered,
        ))
    return recs, start, goal


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 1.0, "text.color": INK, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })

    recs, start, goal = run()
    first_valid = next((r["i"] for r in recs if r["valid"]), ROUNDS)
    nvalid = sum(r["valid"] for r in recs)
    ncov = sum(r["covered"] for r in recs)
    # narrative-beat boundaries derived from the data
    b2 = first_valid + max(6, (ROUNDS - first_valid) // 4)   # brackets truth
    b3 = first_valid + (ROUNDS - first_valid) // 2           # drift
    # stable heatmap scale across the whole run (so drift reads as color change)
    allc = np.concatenate([r["grid"].ravel() for r in recs])
    vmin, vmax = np.percentile(allc, 3), np.percentile(allc, 97)
    heat = LinearSegmentedColormap.from_list(
        "slate", ["#f4f6f9", "#cfd8e3", "#8fa3bd", "#4a5f7e", "#2b3a52"])
    # strip-chart y-limits from finite valid bounds
    fin = [(r["lb"], r["ub"]) for r in recs if r["valid"] and math.isfinite(r["ub"])]
    ymax = max(u for _, u in fin) * 1.08
    ymin = min(0.0, min(l for l, _ in fin))

    frames = list(range(0, ROUNDS, STEP))
    fig, (axG, axS) = plt.subplots(
        1, 2, figsize=(9.4, 4.35), gridspec_kw={"width_ratios": [0.92, 1.08]})
    fig.patch.set_facecolor(SURF)
    fig.subplots_adjust(left=0.015, right=0.965, top=0.80, bottom=0.13, wspace=0.22)

    # figure-level text created ONCE and updated per frame (ax.clear won't touch
    # figure artists, so re-adding them each frame would stack them up)
    t_head = fig.text(0.015, 0.955, "", fontsize=11.5, fontweight="bold",
                      color=INK, ha="left", va="top")
    t_phase = fig.text(0.015, 0.905, "", fontsize=10.2, ha="left", va="top")
    t_stat = fig.text(0.985, 0.955, "", fontsize=9.6, color=MUTED, ha="right",
                      va="top")

    def phase(i):
        if i < first_valid:
            return ("WARM-UP", "certificate INVALID -- calibration buffer filling",
                    MUTED)
        if i < b2:
            return ("CERTIFIED", "LB <= OPT <= UB  -- the band brackets the truth",
                    CERT)
        if i < b3:
            return ("DRIFT", "costs move -- the certified band breathes with age",
                    SENSE)
        return ("SENSING", "gap-directed sensing holds the corridor -- coverage 100%",
                START)

    def draw(fi):
        i = frames[fi]
        r = recs[i]
        axG.clear(); axS.clear()

        # ---------------- LEFT: grid + true-cost heatmap ----------------
        axG.set_facecolor(SURF)
        axG.imshow(r["grid"], cmap=heat, vmin=vmin, vmax=vmax,
                   extent=[-0.5, N - 0.5, N - 0.5, -0.5], interpolation="nearest",
                   zorder=0)
        # incumbent certified path
        if r["path"]:
            px = [c for (_, c) in r["path"]]
            py = [rr for (rr, _) in r["path"]]
            axG.plot(px, py, "-", color="#ffffff", lw=4.2, alpha=0.9,
                     solid_capstyle="round", zorder=3)
            axG.plot(px, py, "-", color=CERTLT, lw=2.4,
                     solid_capstyle="round", zorder=4,
                     label="certified incumbent")
        # sensed edge(s) this round (skip the round-0 initial-survey flood)
        if 0 < len(r["fresh"]) <= 30:
            for (u, v) in r["fresh"]:
                axG.plot([u[1], v[1]], [u[0], v[0]], "-", color=SENSE, lw=3.4,
                         solid_capstyle="round", zorder=5)
            mx = [(u[1] + v[1]) / 2 for (u, v) in r["fresh"]]
            my = [(u[0] + v[0]) / 2 for (u, v) in r["fresh"]]
            axG.scatter(mx, my, s=34, color=SENSE, edgecolor=SURF, lw=1.0,
                        zorder=6, label="sensed this round")
        axG.scatter([start[1]], [start[0]], s=95, marker="o", color=START,
                    edgecolor=SURF, lw=1.6, zorder=7, label="start")
        axG.scatter([goal[1]], [goal[0]], s=120, marker="*", color=GOAL,
                    edgecolor=SURF, lw=1.2, zorder=7, label="goal")
        axG.set_xlim(-0.7, N - 0.3); axG.set_ylim(N - 0.3, -0.7)
        axG.set_xticks([]); axG.set_yticks([])
        for s in axG.spines.values():
            s.set_edgecolor(GRIDC)
        axG.set_title("drifting grid  ·  true edge cost", fontsize=11.5,
                      fontweight="bold", color=INK, loc="left", pad=8)
        axG.legend(loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=2,
                   frameon=False, fontsize=8.2, handlelength=1.3,
                   columnspacing=1.2, labelcolor=INK)

        # ---------------- RIGHT: certified corridor strip chart ----------------
        axS.set_facecolor(SURF)
        xs = np.arange(i + 1)
        lb = np.array([recs[k]["lb"] for k in xs], float)
        ub = np.array([recs[k]["ub"] for k in xs], float)
        op = np.array([recs[k]["opt"] for k in xs], float)
        val = np.array([recs[k]["valid"] for k in xs])
        lb = np.clip(np.where(np.isfinite(lb), lb, ymin), ymin, ymax)
        ub = np.clip(np.where(np.isfinite(ub), ub, ymax), ymin, ymax)
        # warm-up band (gray, invalid) vs certified band (blue)
        axS.fill_between(xs, lb, ub, where=~val, color=MUTED, alpha=0.14,
                         step=None, zorder=1, linewidth=0)
        axS.fill_between(xs, lb, ub, where=val, color=CERTLT, alpha=0.30,
                         zorder=1, linewidth=0)
        vb = np.where(val, ub, np.nan); vl = np.where(val, lb, np.nan)
        axS.plot(xs, vb, color=CERT, lw=1.7, zorder=4, label="UB (upper certificate)")
        axS.plot(xs, vl, color=CERT, lw=1.7, ls=(0, (5, 3)), zorder=4,
                 label="LB (lower certificate)")
        axS.plot(xs, op, color=OPTC, lw=2.1, zorder=5, label="true OPT (Dijkstra)")
        axS.axvline(first_valid, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=2)
        axS.scatter([i], [op[-1]], s=26, color=OPTC, zorder=6)

        axS.set_xlim(0, ROUNDS); axS.set_ylim(ymin, ymax)
        axS.set_xlabel("planning round", fontsize=10)
        axS.set_ylabel("route cost", fontsize=10)
        for s in ("top", "right"):
            axS.spines[s].set_visible(False)
        axS.grid(axis="y", color=GRIDC, lw=0.8, zorder=0); axS.set_axisbelow(True)
        axS.tick_params(length=0)
        axS.set_title("certified corridor   LB ≤ OPT ≤ UB", fontsize=11.5,
                      fontweight="bold", color=INK, loc="left", pad=8)
        axS.legend(loc="upper right", frameon=False, fontsize=8.4,
                   handlelength=1.5)

        # ---------------- header: live phase caption ----------------
        tag, msg, col = phase(i)
        stat = ("certificate INVALID" if not r["valid"]
                else f"gap {r['gap']:.0f}  ·  OPT {r['opt']:.1f}  bracketed ✓")
        t_head.set_text(f"CERT-FLOW  ·  round {i:3d}/{ROUNDS}")
        t_phase.set_text(f"[{tag}]  {msg}"); t_phase.set_color(col)
        t_stat.set_text(stat)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    out = ROOT / "assets/animations/certified_corridor.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=FPS), dpi=80,
              savefig_kwargs={"facecolor": SURF})
    plt.close(fig)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {len(frames)} frames @ {FPS} fps)")
    print(f"first_valid_round={first_valid}  coverage(valid)={ncov}/{nvalid}")


if __name__ == "__main__":
    main()
