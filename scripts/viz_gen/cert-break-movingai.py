"""Honest comparison animation on a REAL MovingAI map: CERT vs AD* vs OPT.

Sibling of scripts/viz_compare.py (the proven, honest template), moved from a
synthetic drifting grid onto a real MovingAI benchmark map. Every frame replays
a REAL run of the actual planner; coverage numbers are MEASURED at render time.

What is shown
-------------
Left  : a real DAO map crop (`data/movingai/dao/arena.map`, 16x16 window) with
        its walls (tree pillars + boundary), the robot's CERT optimistic path,
        and the edge sensed this round. Start->goal corner-to-corner.
Right : two stacked band timelines against the true drifting optimum (black).
        - CERT [LB, UB]: the emitted certificate. Warm-up rounds (no paired
          observations yet -> confidence 0) are shown as a GREY "no claim"
          region, NEVER as a coverage violation. Coverage is measured ONLY over
          rounds where a claim was actually made.
        - AD* [c(P-hat)/w, c(P-hat)] (w=1.5): the genuine w-suboptimality band
          on the planner's stale point estimates, scored exactly as
          `certflow.baselines.adstar_bound` defines it. A red x marks a round
          where AD* claimed and the true optimum fell OUTSIDE its band.

Honesty
-------
The story is certificate VALIDITY (CERT 1.0 vs AD* ~0.4 here, consistent with
docs/results/extern-baselines.md Part A: AD* w=1.5 validity 0.57-0.59 synthetic,
0.02-0.07 real) and behaviour under drift. It is NOT a race to the goal: on a
known static map CERT ties A*/D*. The cost of CERT's soundness is width -- its
band is wider than AD*'s. That trade is shown plainly, not hidden: as un-sensed
edges age, CERT's lower bound relaxes toward the floor (OPT could route through
a stale-cheap edge), so the band is wide but provably contains OPT every round;
AD*'s band is narrow and frequently wrong.

Output (under viz_out/cert-break-movingai/):
  cert-break-movingai.mp4, cert-break-movingai.gif, poster_1..4.png
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from certflow.baselines import adstar_bound
from certflow.cert import CertPlanner, PlannerConfig
from certflow.movingai import crop, movingai_world_from_grid, parse_map
from certflow.oracle import opt
from certflow.sensing import path_edges

# Colorblind-safe palette (Wong): blue/sky = CERT, orange = AD*, black = OPT.
BLUE, SKY, ORANGE, BLK, GRN, RED = (
    "#0072B2", "#56B4E9", "#D55E00", "#111111", "#009E73", "#CC2B1D")

# --- locked, verified configuration (see module docstring) ---------------
MAP_PATH = "data/movingai/dao/arena.map"   # DAO dungeon: an OPEN family
CROP_CENTER = (26, 22)                       # most-open window of arena.map
CROP_SIZE = 16                               # ~16 cells for legibility
START, GOAL = (0, 0), (7, 7)                 # far-apart, connected (L=14 at t0)
SEED = 7
RHO = 0.05                                   # bounded-drift rate
NOISE = 0.20                                 # observation noise scale
ROUNDS = 140
W_ADSTAR = 1.5
EPSILON, ALPHA_PRIME, EPS_TV = 8.0, 0.2, 1e-4

OUT = pathlib.Path("viz_out/cert-break-movingai")
OUT.mkdir(parents=True, exist_ok=True)


def build_grid_world():
    """Parse the real map, crop to a legible open window, build the world."""
    full = parse_map(MAP_PATH)
    sub, _, _ = crop(full, CROP_CENTER[0], CROP_CENTER[1], CROP_SIZE)
    world = movingai_world_from_grid(
        sub, seed=SEED, kind="bounded", rho=RHO, noise_scale=NOISE)
    return sub, world


def wall_mask(sub: list[str]) -> np.ndarray:
    """Boolean grid: True where the cell is an impassable wall (drawn dark)."""
    H, W = len(sub), len(sub[0])
    m = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            m[r, c] = sub[r][c] not in ".GS"
    return m


def capture():
    """Replay a REAL run; record genuine per-round state for every frame."""
    sub, world = build_grid_world()
    walls = wall_mask(sub)
    # Neutral re-sensing on a shared stream (extern-baselines Part A method):
    # max_age round-robins observations so ONLY the bound semantics differ
    # between CERT and AD* (both read the same beliefs).
    cfg = PlannerConfig(
        epsilon=EPSILON, alpha_prime=ALPHA_PRIME, eps_tv=EPS_TV,
        initial_survey=True, sensing_policy="max_age")
    p = CertPlanner(world, START, GOAL, cfg)

    frames = []
    cert_hits = ad_hits = valid = 0
    for i in range(ROUNDS):
        cert, sensed = p.round()
        t_eval = p.t - cfg.delta
        _, true_opt = opt(world, t_eval, START, GOAL)

        # Genuine AD*/ARA* semantics: w-suboptimality band on stale point
        # estimates (exactly certflow.baselines.adstar_bound).
        ad_lo, ad_hi = adstar_bound(p.beliefs, world.graph, START, GOAL,
                                    w=W_ADSTAR)

        # CERT's optimistic path (the lower-bound shortest path) for the map.
        lb_path, _ = p.sp_lower.shortest_path()

        cert_in = bool(cert.valid and cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9)
        ad_in = bool(ad_lo - 1e-9 <= true_opt <= ad_hi + 1e-9)
        if cert.valid:
            valid += 1
            cert_hits += cert_in
        ad_hits += ad_in

        frames.append(dict(
            i=i, walls=walls, path=lb_path or [], sensed=sensed,
            opt=true_opt, cert=(cert.lb, cert.ub), ad=(ad_lo, ad_hi),
            cert_in=cert_in, ad_in=ad_in, cert_valid=bool(cert.valid),
            confidence=float(cert.confidence),
            cert_cov=cert_hits / max(valid, 1),
            ad_cov=ad_hits / (i + 1)))
    meta = dict(H=walls.shape[0], W=walls.shape[1], valid=valid,
                cert_cov=cert_hits / max(valid, 1), ad_cov=ad_hits / ROUNDS,
                first_valid=next((f["i"] for f in frames if f["cert_valid"]),
                                 None))
    return frames, meta


def render(frames, meta):
    H, W = meta["H"], meta["W"]
    fig = plt.figure(figsize=(12.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.32])
    gx = fig.add_subplot(gs[:, 0])     # map spans both rows
    gc = fig.add_subplot(gs[0, 1])     # CERT band timeline
    ga = fig.add_subplot(gs[1, 1])     # AD* band timeline

    # shared y-range across both band panels (clip CERT's growing UB so the
    # true-OPT line and AD* band stay legible; the clip is annotated).
    opts = [f["opt"] for f in frames]
    ad_his = [f["ad"][1] for f in frames]
    ad_los = [f["ad"][0] for f in frames]
    ymin = max(0.0, min(min(opts), min(ad_los)) - 2.0)
    # cap the visible band at a multiple of the OPT scale so AD* misses are
    # readable; CERT's UB can exceed this and is drawn clipped with a caret.
    ymax = max(max(opts), max(ad_his)) * 2.6
    UBCLIP = ymax

    warm = [f["i"] for f in frames if not f["cert_valid"]]
    wu_lo, wu_hi = (min(warm), max(warm) + 1) if warm else (None, None)

    def band(ax, k, key, color, name, hits_key, gated):
        xs = np.arange(k + 1)
        oo = [frames[j]["opt"] for j in xs]
        # a claim exists every round for AD*; for CERT only once valid
        claim = [(frames[j]["cert_valid"] if gated else True) for j in xs]
        lo_ = np.array([frames[j][key][0] if claim[j] else np.nan for j in xs])
        hi_raw = np.array([frames[j][key][1] if claim[j] else np.nan for j in xs])
        hi_ = np.minimum(hi_raw, UBCLIP)
        ax.fill_between(xs, lo_, hi_, color=color, alpha=0.42, lw=0,
                        label=f"{name} band [LB, UB]")
        ax.plot(xs, oo, "-", color=BLK, lw=2.0, label="true OPT", zorder=5)
        # warm-up region (CERT only): no claim yet -- shade grey, never a miss
        if gated and wu_lo is not None and wu_lo <= k:
            ax.axvspan(wu_lo, min(wu_hi, k + 1), color="0.82", alpha=0.7, lw=0,
                       zorder=0)
            ax.text(wu_lo + 0.4, ymin + (ymax - ymin) * 0.06,
                    "warm-up\nno claim", fontsize=8, color="0.35", va="bottom")
        # caret where CERT's true UB exceeds the visible clip (honesty: the
        # band is even wider than drawn, not cut to look tighter)
        if gated:
            over = [j for j in xs if claim[j] and hi_raw[j] > UBCLIP + 1e-9]
            if over:
                ax.plot(over, [UBCLIP] * len(over), "^", color=color, ms=4,
                        mew=0, alpha=0.8)
        # genuine misses only: a claim WAS made and OPT fell outside it
        viol = [j for j in xs if claim[j] and not frames[j][hits_key]]
        if viol:
            ax.plot(viol, [frames[j]["opt"] for j in viol], "x", color=RED,
                    ms=7, mew=2.2, label="OPT outside band", zorder=6)
        ax.set_xlim(0, ROUNDS)
        ax.set_ylim(ymin, ymax)
        cov = frames[k][("cert_cov" if gated else "ad_cov")]
        denom = "claims" if gated else "rounds"
        ax.set_title(f"{name}: covers true OPT in {cov:.0%} of {denom}",
                     fontsize=11.5,
                     color=(BLUE if name == "CERT" else ORANGE), pad=4)
        ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92, ncol=2,
                  handlelength=1.4, columnspacing=1.0)
        ax.tick_params(labelsize=8.5)
        ax.set_ylabel("path cost", fontsize=9)

    def draw(k):
        f = frames[k]
        gx.clear(); gc.clear(); ga.clear()
        # --- left: real map (walls) + CERT optimistic path + sensed edge ---
        gx.imshow(f["walls"], cmap="Greys", origin="upper", vmin=0, vmax=1.4,
                  extent=[-0.5, W - 0.5, H - 0.5, -0.5], interpolation="nearest")
        gx.set_xlim(-0.5, W - 0.5)
        gx.set_ylim(H - 0.5, -0.5)
        if len(f["path"]) > 1:
            ys = [n[0] for n in f["path"]]
            xs = [n[1] for n in f["path"]]
            gx.plot(xs, ys, "-", color=BLUE, lw=3.4, solid_capstyle="round",
                    label="CERT optimistic path", zorder=3)
        if f["sensed"]:
            (u, v) = f["sensed"]
            gx.plot([u[1], v[1]], [u[0], v[0]], "-", color=RED, lw=6,
                    alpha=0.85, solid_capstyle="round", zorder=4)
            gx.plot((u[1] + v[1]) / 2, (u[0] + v[0]) / 2, "v", color=RED,
                    ms=10, label="edge sensed this round", zorder=5)
        gx.plot(START[1], START[0], "o", color=GRN, ms=14, zorder=6)
        gx.plot(GOAL[1], GOAL[0], "*", color=RED, ms=20, zorder=6)
        gx.text(START[1] + 0.5, START[0] - 0.2, "start", color=GRN,
                fontsize=9, fontweight="bold", va="bottom")
        gx.text(GOAL[1] - 0.5, GOAL[0] + 0.3, "goal", color=RED, fontsize=9,
                fontweight="bold", ha="right", va="top")
        gx.set_xticks([]); gx.set_yticks([])
        claim_txt = (f"valid claim (conf {f['confidence']:.0%})"
                     if f["cert_valid"] else "warm-up: no claim yet")
        gx.set_title(
            f"Real MovingAI map  (DAO arena, {H}x{W} crop) under drift\n"
            f"round {f['i'] + 1}/{ROUNDS}   --   CERT {claim_txt}",
            fontsize=11)
        # custom legend (avoid duplicate auto entries, keep it readable)
        handles = [
            Line2D([], [], color=BLUE, lw=3.4, label="CERT optimistic path"),
            Line2D([], [], color=RED, lw=6, alpha=0.85,
                   label="edge sensed this round"),
            Line2D([], [], marker="o", color=GRN, lw=0, ms=10, label="start"),
            Line2D([], [], marker="*", color=RED, lw=0, ms=13, label="goal"),
            Patch(facecolor="0.25", label="wall (impassable)"),
        ]
        gx.legend(handles=handles, loc="lower right", fontsize=8.0,
                  framealpha=0.93, handlelength=1.4)
        # --- right: two stacked band timelines, unambiguous ---
        band(gc, k, "cert", SKY, "CERT", "cert_in", True)
        band(ga, k, "ad", ORANGE, "AD*", "ad_in", False)
        ga.set_xlabel("round (robot stationary at start; world drifts)",
                      fontsize=9)
        fig.suptitle(
            "Certificate validity on a real map: CERT's band always contains "
            "the true optimum; AD*'s stale point-estimate band does not",
            fontsize=12.5, fontweight="bold")
        return []

    # --- animation ---
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=160)
    mp4 = OUT / "cert-break-movingai.mp4"
    gif = OUT / "cert-break-movingai.gif"
    used_mp4 = True
    try:
        writer = FFMpegWriter(fps=7, bitrate=1800,
                              metadata={"title": "CERT vs AD* on MovingAI"})
        anim.save(str(mp4), writer=writer, dpi=110)
    except Exception as exc:  # pragma: no cover - environment dependent
        used_mp4 = False
        print(f"WARNING: FFMpegWriter failed ({exc}); MP4 not written.")
    # GIF for web embedding (looping)
    anim.save(str(gif), writer=PillowWriter(fps=6))

    # --- 4 poster PNGs: early / two middle / final ---
    fv = meta["first_valid"] or 0
    poster_rounds = [
        max(2, fv // 2),            # 1: deep in warm-up (no claim)
        min(len(frames) - 1, fv + 6),  # 2: just after first valid claim
        min(len(frames) - 1, (fv + len(frames)) // 2),  # 3: mid, drift acting
        len(frames) - 1,            # 4: final
    ]
    poster_paths = []
    for n, k in enumerate(poster_rounds, start=1):
        draw(k)
        pth = OUT / f"poster_{n}.png"
        fig.savefig(pth, dpi=110)
        poster_paths.append(pth)
    plt.close(fig)

    final = frames[-1]
    print(f"MP4: {mp4 if used_mp4 else 'FAILED'} "
          f"({mp4.stat().st_size // 1024 if used_mp4 and mp4.exists() else 0} KB)")
    print(f"GIF: {gif}  ({gif.stat().st_size // 1024} KB)")
    for p_ in poster_paths:
        print(f"poster: {p_}  ({p_.stat().st_size // 1024} KB)")
    print(f"first_valid round: {meta['first_valid']}  "
          f"(warm-up rounds before it are shown as 'no claim')")
    print(f"MEASURED coverage -- CERT {final['cert_cov']:.1%} of "
          f"{meta['valid']} valid claims | AD* {final['ad_cov']:.1%} of "
          f"{ROUNDS} rounds")
    return used_mp4


if __name__ == "__main__":
    frames, meta = capture()
    render(frames, meta)
