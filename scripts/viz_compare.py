"""Proof-of-concept comparison animation: the certificate that holds vs the
one that breaks, on a drifting grid.

Honest by construction: every frame replays a REAL run. CERT's [LB,UB] is the
emitted certificate; the AD*-semantics band is the w-suboptimality interval
[c/w, c] on the planner's stale point estimates (exactly how extern-baselines
scores it). The running coverage fractions printed are measured, not staged.

Output: viz_out/cert_vs_adstar.gif + four key PNG frames.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.graphcore import dijkstra
from certflow.oracle import opt

BLUE, ORANGE, BLK, GRN, RED, SKY = (
    "#0072B2", "#D55E00", "#111111", "#009E73", "#CC2B1D", "#56B4E9")
R = C = 10
START, GOAL = (0, 0), (R - 1, C - 1)
ROUNDS = 60
W_ADSTAR = 1.5
OUT = pathlib.Path("viz_out")
OUT.mkdir(exist_ok=True)


def capture():
    world = grid_world(R, C, seed=4, kind="bounded", rho=0.05, noise_scale=0.08)
    # surveyed-then-drift: the realistic field scenario — you map the area
    # once, then costs drift. CERT re-senses to keep the band tight; AD*
    # keeps trusting its stale point estimates.
    cfg = PlannerConfig(epsilon=6.0, alpha_prime=0.2, eps_tv=1e-4,
                        initial_survey=True, adaptive_rate=True)
    p = CertPlanner(world, START, GOAL, cfg)
    frames = []
    cert_hits = ad_hits = valid = 0
    for i in range(ROUNDS):
        cert, sensed = p.round()
        t_eval = p.t - cfg.delta
        _, true_opt = opt(world, t_eval, START, GOAL)
        # AD* semantics: route on stale point estimates, w-suboptimality band
        snap = {u: {v: p.beliefs[(u, v)].c_hat for v in nb}
                for u, nb in world.graph.items()}
        ad_path, c_point = dijkstra(snap, START, GOAL)
        ad_lo, ad_hi = c_point / W_ADSTAR, c_point
        # drift heatmap: mean outgoing true-cost per node
        heat = np.full((R, C), np.nan)
        for (u, nb) in world.graph.items():
            vals = [world.true_cost((u, v), t_eval) for v in nb]
            if vals:
                heat[u[0], u[1]] = float(np.mean(vals))
        lb_path, _ = p.sp_lower.shortest_path()
        cert_in = cert.valid and (cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9)
        ad_in = ad_lo - 1e-9 <= true_opt <= ad_hi + 1e-9
        if cert.valid:
            valid += 1
            cert_hits += cert_in
        ad_hits += ad_in
        frames.append(dict(
            i=i, heat=heat, path=lb_path or [], sensed=sensed,
            opt=true_opt, cert=(cert.lb, cert.ub, cert.valid),
            ad=(ad_lo, ad_hi), cert_in=cert_in, ad_in=ad_in,
            cert_valid=bool(cert.valid),
            cert_cov=cert_hits / max(valid, 1),
            ad_cov=ad_hits / (i + 1)))
    return frames


def render(frames):
    fig = plt.figure(figsize=(11.5, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25])
    gx = fig.add_subplot(gs[:, 0])     # grid spans both rows
    gc = fig.add_subplot(gs[0, 1])     # CERT band timeline
    ga = fig.add_subplot(gs[1, 1])     # AD* band timeline
    opts = [f["opt"] for f in frames]
    lo = min(min(o for o in opts), 0.0)
    ymax = max(max(opts), max(f["ad"][1] for f in frames),
               max(f["cert"][1] for f in frames)) * 1.12
    ymin = max(0.0, min(opts) * 0.6)

    def band(ax, k, key, color, name, hits_key, gated):
        xs = np.arange(k + 1)
        oo = [frames[j]["opt"] for j in xs]
        # a claim exists every round for AD*; for CERT only once valid
        claim = [(frames[j]["cert_valid"] if gated else True) for j in xs]
        lo_ = np.array([frames[j][key][0] if claim[j] else np.nan for j in xs])
        hi_ = np.array([frames[j][key][1] if claim[j] else np.nan for j in xs])
        ax.fill_between(xs, lo_, hi_, color=color, alpha=0.40, lw=0,
                        label=f"{name} band")
        ax.plot(xs, oo, "-", color=BLK, lw=1.8, label="true OPT")
        # warm-up region (CERT only): no claim yet — shade, don't penalize
        if gated:
            wu = [j for j in xs if not frames[j]["cert_valid"]]
            if wu:
                ax.axvspan(min(wu), max(wu) + 1, color="0.85", alpha=0.5, lw=0)
                ax.text(min(wu) + 0.3, ymax * 0.9, "warm-up\n(no claim)",
                        fontsize=6.5, color="0.4", va="top")
        # genuine misses only: a claim WAS made and OPT fell outside it
        viol = [j for j in xs if claim[j] and not frames[j][hits_key]]
        if viol:
            ax.plot(viol, [frames[j]["opt"] for j in viol], "x", color=RED,
                    ms=6, mew=2, label="OPT outside band")
        ax.set_xlim(0, ROUNDS); ax.set_ylim(ymin, ymax)
        cov = frames[k][("cert_cov" if gated else "ad_cov")]
        ax.set_title(f"{name}: coverage {cov:.0%} of claims", fontsize=10,
                     color=(BLUE if name == "CERT" else ORANGE))
        ax.legend(loc="upper left", fontsize=6.5, framealpha=0.9, ncol=3)
        ax.tick_params(labelsize=7)

    def draw(k):
        f = frames[k]
        gx.clear(); gc.clear(); ga.clear()
        # --- left: drifting grid + CERT path ---
        gx.imshow(f["heat"], cmap="YlOrBr", origin="upper", alpha=0.55,
                  extent=[-0.5, C - 0.5, R - 0.5, -0.5])
        if len(f["path"]) > 1:
            ys = [n[0] for n in f["path"]]; xs = [n[1] for n in f["path"]]
            gx.plot(xs, ys, "-", color=BLUE, lw=3, solid_capstyle="round",
                    label="CERT optimistic path")
        if f["sensed"]:
            (u, v) = f["sensed"]
            gx.plot([u[1], v[1]], [u[0], v[0]], "-", color=RED, lw=5,
                    alpha=0.9, solid_capstyle="round")
            gx.plot((u[1] + v[1]) / 2, (u[0] + v[0]) / 2, "v", color=RED,
                    ms=9, label="sensed this round")
        gx.plot(START[1], START[0], "o", color=GRN, ms=13)
        gx.plot(GOAL[1], GOAL[0], "*", color=RED, ms=18)
        gx.set_xticks([]); gx.set_yticks([])
        gx.set_title(f"Surveyed {R}×{C} world, now drifting — "
                     f"round {f['i']+1}/{ROUNDS}\n"
                     "background = current true edge cost", fontsize=10)
        gx.legend(loc="lower right", fontsize=7, framealpha=0.9)
        # --- right: two stacked band timelines, unambiguous ---
        band(gc, k, "cert", SKY, "CERT", "cert_in", True)
        band(ga, k, "ad", ORANGE, "AD*", "ad_in", False)
        ga.set_xlabel("round", fontsize=8)
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=180)
    gif = OUT / "cert_vs_adstar.gif"
    anim.save(str(gif), writer=PillowWriter(fps=6))
    # key frames as standalone PNGs
    for tag, k in [("01_start", 2), ("02_early", 14),
                   ("03_mid", 32), ("04_end", len(frames) - 1)]:
        draw(k)
        fig.savefig(OUT / f"frame_{tag}.png", dpi=110)
    plt.close(fig)
    final = frames[-1]
    print(f"GIF: {gif}  ({gif.stat().st_size // 1024} KB)")
    print(f"FINAL measured coverage — CERT {final['cert_cov']:.1%} | "
          f"AD* {final['ad_cov']:.1%}  (over {ROUNDS} rounds)")


if __name__ == "__main__":
    render(capture())
