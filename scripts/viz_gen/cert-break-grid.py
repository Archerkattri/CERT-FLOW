"""cert-break-grid: the certificate that HOLDS vs the one that BREAKS, on a
surveyed-then-drifting grid. Band-comparison animation, CERT vs AD*.

This is a polished, render-complete descendant of scripts/viz_compare.py. The
capture stage is byte-for-byte the proven-honest logic from that template:
every frame replays ONE real round of the actual CertPlanner against a real
BoundedDriftWorld, AD*'s band uses the GENUINE bounded-suboptimal semantics
(certflow.baselines.adstar_bound: OPT in [c(P-hat)/w, c(P-hat)], w=1.5, on the
planner's own stale point estimates), and the optimum it is scored against is
the oracle's true shortest path. Coverage fractions are MEASURED at render
time. Warm-up rounds (CERT emits no certificate until its conformal buffer has
paired observations) are shown as a grey "no claim" region, NEVER as a
violation; a red x marks ONLY a round where a claim was made and the true OPT
fell outside it.

Outputs (viz_out/cert-break-grid/):
  cert-break-grid.mp4   FFMpegWriter, fps 6, bounded bitrate (< 6 MB)
  cert-break-grid.gif   looping, decimated frames (< 3 MB)
  poster_1..4.png       early / two middle / final key frames, ~110 dpi

Run:  cert_env/bin/python scripts/viz_gen/cert-break-grid.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402

# Make the in-repo package importable when run as a plain script.
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from certflow.baselines import adstar_bound  # noqa: E402
from certflow.cert import CertPlanner, PlannerConfig  # noqa: E402
from certflow.drift import grid_world  # noqa: E402
from certflow.oracle import opt  # noqa: E402

# ---- colorblind-safe palette (Okabe-Ito) ----------------------------------
BLUE = "#0072B2"   # CERT path / accents
SKY = "#56B4E9"    # CERT band fill
ORANGE = "#D55E00" # AD* (the baseline)
BLK = "#111111"    # true OPT
GRN = "#009E73"    # start
RED = "#CC2B1D"    # sensed edge / violation marker

R = C = 10
START, GOAL = (0, 0), (R - 1, C - 1)
ROUNDS = 60
W_ADSTAR = 1.5
OUT = pathlib.Path(__file__).resolve().parents[2] / "viz_out" / "cert-break-grid"
OUT.mkdir(parents=True, exist_ok=True)


def capture():
    """Replay one real CERT run; record exactly what each side claimed and
    whether the true optimum fell inside it. No numbers are staged."""
    world = grid_world(R, C, seed=4, kind="bounded", rho=0.05, noise_scale=0.08)
    # Surveyed-then-drift: the realistic field scenario — map the area once at
    # t0, then costs drift. CERT re-senses to keep [LB,UB] sound; AD* keeps
    # trusting its stale point estimates.
    cfg = PlannerConfig(epsilon=6.0, alpha_prime=0.2, eps_tv=1e-4,
                        initial_survey=True, adaptive_rate=True)
    p = CertPlanner(world, START, GOAL, cfg)
    frames = []
    cert_hits = ad_hits = valid = 0
    ad_width_sum = cert_width_sum = 0.0
    for i in range(ROUNDS):
        cert, sensed = p.round()
        # score against the optimum at the round's evaluation instant
        t_eval = p.t - cfg.delta
        _, true_opt = opt(world, t_eval, START, GOAL)
        # AD* semantics: genuine bounded-suboptimal band on the planner's
        # current stale point estimates (the canonical helper used by the
        # extern-baselines experiment, so the framing matches the paper).
        ad_lo, ad_hi = adstar_bound(
            p.beliefs, world.graph, START, GOAL, w=W_ADSTAR, cost_floor=cfg.cost_floor)
        # drift heatmap: mean outgoing true cost per node, right now
        heat = np.full((R, C), np.nan)
        for (u, nb) in world.graph.items():
            vals = [world.true_cost((u, v), t_eval) for v in nb]
            if vals:
                heat[u[0], u[1]] = float(np.mean(vals))
        lb_path, _ = p.sp_lower.shortest_path()
        cert_in = bool(cert.valid and (cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9))
        ad_in = bool(ad_lo - 1e-9 <= true_opt <= ad_hi + 1e-9)
        if cert.valid:
            valid += 1
            cert_hits += cert_in
            cert_width_sum += (cert.ub - cert.lb)
        ad_hits += ad_in
        ad_width_sum += (ad_hi - ad_lo)
        frames.append(dict(
            i=i, heat=heat, path=lb_path or [], sensed=sensed,
            opt=true_opt, cert=(cert.lb, cert.ub, bool(cert.valid)),
            ad=(ad_lo, ad_hi), cert_in=cert_in, ad_in=ad_in,
            cert_valid=bool(cert.valid),
            cert_cov=cert_hits / max(valid, 1),
            ad_cov=ad_hits / (i + 1),
            cert_mean_w=cert_width_sum / max(valid, 1),
            ad_mean_w=ad_width_sum / (i + 1)))
    return frames


def _band(ax, frames, k, key, color, name, hits_key, gated, ymin, ymax):
    """Draw one side's band timeline up to round k, with honest warm-up
    shading and genuine-violation markers."""
    xs = np.arange(k + 1)
    oo = [frames[j]["opt"] for j in xs]
    # AD* claims every round; CERT claims only once its certificate is valid.
    claim = [(frames[j]["cert_valid"] if gated else True) for j in xs]
    lo_ = np.array([frames[j][key][0] if claim[j] else np.nan for j in xs])
    hi_ = np.array([frames[j][key][1] if claim[j] else np.nan for j in xs])
    ax.fill_between(xs, lo_, hi_, color=color, alpha=0.45, lw=0,
                    label=f"{name} interval", zorder=1)
    ax.plot(xs, oo, "-", color=BLK, lw=2.2, label="true OPT", zorder=3)
    # warm-up region (CERT only): NO claim yet — shade it, never penalize it
    if gated:
        wu = [j for j in xs if not frames[j]["cert_valid"]]
        if wu:
            ax.axvspan(min(wu), max(wu) + 1, color="0.86", alpha=0.7, lw=0,
                       zorder=0)
            ax.text((min(wu) + max(wu) + 1) / 2.0, ymin + 0.07 * (ymax - ymin),
                    "warm-up\nno claim", fontsize=10, color="0.35",
                    ha="center", va="bottom", style="italic")
    # genuine miss: a claim WAS made AND the true OPT fell outside it
    viol = [j for j in xs if claim[j] and not frames[j][hits_key]]
    if viol:
        ax.plot(viol, [frames[j]["opt"] for j in viol], "x", color=RED,
                ms=7, mew=2.4, label="OPT outside interval", zorder=4)
    ax.set_xlim(0, ROUNDS)
    ax.set_ylim(ymin, ymax)
    cov = frames[k][("cert_cov" if gated else "ad_cov")]
    n_claims = (sum(1 for j in xs if frames[j]["cert_valid"]) if gated else k + 1)
    accent = BLUE if gated else ORANGE
    ax.set_title(
        f"{name}: true OPT covered on {cov:.0%} of its {n_claims} claims",
        fontsize=13, color=accent, fontweight="bold", pad=4)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92, ncol=3,
              handlelength=1.3, columnspacing=1.0, borderpad=0.3)
    ax.tick_params(labelsize=10)
    ax.set_ylabel("path cost", fontsize=10)


def _draw(fig, axes, frames, k, ymin, ymax):
    gx, gc, ga = axes
    f = frames[k]
    gx.clear(); gc.clear(); ga.clear()
    # --- left: drifting grid + CERT optimistic path + this round's sensing ---
    gx.imshow(f["heat"], cmap="YlOrBr", origin="upper", alpha=0.6,
              extent=[-0.5, C - 0.5, R - 0.5, -0.5])
    if len(f["path"]) > 1:
        ys = [n[0] for n in f["path"]]
        xs = [n[1] for n in f["path"]]
        gx.plot(xs, ys, "-", color=BLUE, lw=3.4, solid_capstyle="round",
                label="CERT optimistic path", zorder=2)
    if f["sensed"]:
        (u, v) = f["sensed"]
        gx.plot([u[1], v[1]], [u[0], v[0]], "-", color=RED, lw=5.5,
                alpha=0.9, solid_capstyle="round", zorder=3)
        gx.plot((u[1] + v[1]) / 2, (u[0] + v[0]) / 2, "v", color=RED,
                ms=11, label="sensed this round", zorder=4)
    gx.plot(START[1], START[0], "o", color=GRN, ms=15, zorder=5)
    gx.plot(GOAL[1], GOAL[0], "*", color=RED, ms=21, zorder=5)
    gx.set_xticks([]); gx.set_yticks([])
    gx.set_title(f"Surveyed {R}x{C} grid, now drifting  —  round "
                 f"{f['i'] + 1}/{ROUNDS}", fontsize=12.5, pad=8)
    gx.set_xlabel("background shade = current true edge cost", fontsize=10.5)
    gx.legend(loc="lower right", fontsize=10, framealpha=0.92)
    # --- right: two stacked, unambiguous band timelines ---
    _band(gc, frames, k, "cert", SKY, "CERT [LB, UB]", "cert_in", True,
          ymin, ymax)
    _band(ga, frames, k, "ad", ORANGE, "AD* w=1.5 band", "ad_in", False,
          ymin, ymax)
    ga.set_xlabel("replanning round", fontsize=11)
    return []


def _fig():
    fig = plt.figure(figsize=(12.6, 5.9))
    fig.suptitle(
        "Certificate validity under drift:  CERT [LB <= OPT <= UB] holds  vs  "
        "AD* w-suboptimality band breaks",
        fontsize=14.5, fontweight="bold", y=0.975)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.32],
                          left=0.045, right=0.992, top=0.855, bottom=0.085,
                          wspace=0.16, hspace=0.40)
    gx = fig.add_subplot(gs[:, 0])   # grid spans both rows
    gc = fig.add_subplot(gs[0, 1])   # CERT band timeline
    ga = fig.add_subplot(gs[1, 1])   # AD* band timeline
    return fig, (gx, gc, ga)


def render(frames):
    fig, axes = _fig()
    # shared y-range for the two timelines, computed from the real run
    opts = [f["opt"] for f in frames]
    ymax = max(max(opts),
               max(f["ad"][1] for f in frames),
               max(f["cert"][1] for f in frames)) * 1.14
    ymin = max(0.0, min(opts) * 0.55)

    def draw(k):
        return _draw(fig, axes, frames, k, ymin, ymax)

    # ---- MP4 (preferred) ----
    mp4 = OUT / "cert-break-grid.mp4"
    mp4_ok = False
    if FFMpegWriter.isAvailable():
        try:
            anim = FuncAnimation(fig, draw, frames=len(frames), interval=160)
            # ~110 dpi * (12.6x5.6) is large; cap bitrate so the file stays
            # well under 6 MB for 60 frames at fps 6.
            writer = FFMpegWriter(fps=6, bitrate=2400,
                                  metadata={"title": "CERT vs AD* under drift"})
            anim.save(str(mp4), writer=writer, dpi=110)
            mp4_ok = mp4.exists() and mp4.stat().st_size > 0
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] FFMpeg writer failed ({exc!r}); MP4 skipped, "
                  "GIF is the fallback.")
            mp4_ok = False
    else:
        print("[warn] FFMpeg writer not available; MP4 skipped, GIF fallback.")

    # ---- GIF (looping, smaller: decimate to every other frame) ----
    gif = OUT / "cert-break-grid.gif"
    gif_frames = list(range(0, len(frames), 2))
    if gif_frames[-1] != len(frames) - 1:
        gif_frames.append(len(frames) - 1)

    def draw_gif(idx):
        return draw(gif_frames[idx])

    anim_gif = FuncAnimation(fig, draw_gif, frames=len(gif_frames),
                             interval=220)
    anim_gif.save(str(gif), writer=PillowWriter(fps=5), dpi=80)

    # ---- 4 poster PNGs: early / two middle / final ----
    posters = [("poster_1", 4), ("poster_2", 24),
               ("poster_3", 38), ("poster_4", len(frames) - 1)]
    poster_paths = []
    for tag, k in posters:
        draw(k)
        path = OUT / f"{tag}.png"
        fig.savefig(path, dpi=110)
        poster_paths.append(path)
    plt.close(fig)

    # ---- measured numbers (the honest readout) ----
    final = frames[-1]
    n_cert_claims = sum(1 for f in frames if f["cert_valid"])
    n_warmup = ROUNDS - n_cert_claims
    print("=" * 64)
    print(f"MP4 : {mp4 if mp4_ok else '(skipped — GIF fallback)'}"
          + (f"  ({mp4.stat().st_size / 1e6:.2f} MB)" if mp4_ok else ""))
    print(f"GIF : {gif}  ({gif.stat().st_size / 1e6:.2f} MB)  "
          f"[{len(gif_frames)} frames, looping]")
    for p in poster_paths:
        print(f"PNG : {p}  ({p.stat().st_size // 1024} KB)")
    print("-" * 64)
    print("MEASURED over the real run "
          f"({ROUNDS} rounds, seed=4, rho=0.05, noise=0.08):")
    print(f"  CERT  coverage = {final['cert_cov']:.3f}  "
          f"of {n_cert_claims} valid claims  "
          f"({n_warmup} warm-up rounds = no claim, not counted as misses)")
    print(f"  AD*   coverage = {final['ad_cov']:.3f}  "
          f"of {ROUNDS} claims (w={W_ADSTAR})")
    print(f"  mean interval width:  CERT {final['cert_mean_w']:.2f}  vs  "
          f"AD* {final['ad_mean_w']:.2f}  "
          f"(CERT wider-but-sound: {final['cert_mean_w'] / max(final['ad_mean_w'], 1e-9):.1f}x)")
    print("=" * 64)
    return dict(
        cert_cov=final["cert_cov"], ad_cov=final["ad_cov"],
        cert_mean_w=final["cert_mean_w"], ad_mean_w=final["ad_mean_w"],
        n_cert_claims=n_cert_claims, n_warmup=n_warmup,
        mp4_ok=mp4_ok)


if __name__ == "__main__":
    render(capture())
