"""Coverage-observability figure for the live-wiring layer (README + project page).

Coverage made observable: the Shiryaev-Roberts statistic from the WATCH
testability layer, replayed faithfully from run_watch_testability.py's seed-0
streams. Under the correctly-modelled (null) stream it stays quiet below the
alarm threshold; on a stream with an injected regime shift at round 250 it
crosses the threshold ~7 rounds later. This is the same detector CERT-FLOW now
runs live inside round() (planner.sr), at zero cost to the certificate -- quiet
on 20/20 real METR-LA replay days.

(The certified-width comparison across pricing methods now lives in its own
comprehensive chart, assets/width_methods.png; scripts/viz_gen/width_methods.py.)

Reproduce:  PYTHONPATH=src python scripts/viz_gen/live_wiring_fig.py
Writes assets/live_wiring_2026.png. Needs no data/ (fully synthetic replay).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from certflow.conformal import (  # noqa: E402
    ConformalScorer,
    ShiryaevRobertsDetector,
    conformal_p_value,
)

# ---- palette (Okabe-Ito; matches the repo/site --cert/--accent/--ok ink) -----
NULL = "#009E73"   # green      (quiet = good)
SHIFT = "#D55E00"  # vermillion (alarm)
INK = "#0a111e"
MUTED = "#5b6b82"
GRID = "#e4e9f0"
SURF = "#ffffff"

# ---- experiment constants (mirror scripts/run_watch_testability.py defaults) --
ROUNDS, RHO_W, DRIFT, NOISE = 3000, 0.98, 0.05, 0.10
WARMUP, SHIFT_AT, SR_THRESHOLD = 100, 250, 10000.0


def _make_stream(rng, rounds, drift, noise, shift_at=None, shift_drift=None):
    c, last = 0.0, None
    scores = []
    for t in range(rounds):
        d = drift if (shift_at is None or t < shift_at) else shift_drift
        c += rng.uniform(-d, d)
        y = c + rng.normal(0.0, noise)
        if last is not None:
            scores.append(abs(y - last))
        last = y
    return scores


def _sr_trajectory(scores, threshold):
    """Replay one score stream through the age-weighted scorer + SR detector;
    return the per-monitoring-round SR statistic (identical construction to
    run_watch_testability.run_stream)."""
    scorer = ConformalScorer(rho_w=RHO_W, max_buffer=1000)
    sr = ShiryaevRobertsDetector(threshold=threshold, epsilon=0.5)
    traj = []
    for t, r in enumerate(scores):
        if len(scorer) >= WARMUP:
            cal = [s.residual for s in scorer._buf]
            w = scorer._weights(float(t))
            traj.append(sr.update(conformal_p_value(r, cal, w)))
        scorer.push(r, float(t))
    return np.asarray(traj), sr.alarm_round


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "svg.fonttype": "none", "axes.linewidth": 1.0,
        "text.color": INK, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })

    # ---- faithful seed-0 replay -----------------------------------------------
    rng = np.random.default_rng(0)
    null_scores = _make_stream(rng, ROUNDS, DRIFT, NOISE)
    shift_scores = _make_stream(rng, ROUNDS, DRIFT, NOISE,
                                shift_at=SHIFT_AT, shift_drift=DRIFT * 40.0)
    null_traj, _ = _sr_trajectory(null_scores, SR_THRESHOLD)
    shift_traj, alarm_step = _sr_trajectory(shift_scores, SR_THRESHOLD)
    shift_mon = SHIFT_AT - WARMUP                     # shift, in monitoring rounds
    delay = None if alarm_step is None else alarm_step - shift_mon

    fig, axB = plt.subplots(1, 1, figsize=(8.2, 4.5))
    fig.patch.set_facecolor(SURF)

    # ============================ SR MONITOR ===================================
    axB.set_facecolor(SURF)
    r_null = np.arange(len(null_traj))
    r_shift = np.arange(len(shift_traj))
    axB.plot(r_null, np.clip(null_traj, 1e-3, None), color=NULL, lw=1.8,
             label="correct model (null)", zorder=4)
    axB.plot(r_shift, np.clip(shift_traj, 1e-3, None), color=SHIFT, lw=1.8,
             label="regime shift @ round 250", zorder=5)
    axB.axhline(SR_THRESHOLD, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2)
    axB.text(len(r_shift) * 0.34, SR_THRESHOLD * 2.1,
             "alarm threshold (ARL 10⁴)", ha="center", va="bottom",
             color=MUTED, fontsize=9)
    axB.axvline(shift_mon, color=INK, lw=1.0, ls=(0, (2, 3)), zorder=2)
    axB.text(shift_mon - 14, 6e4, "injected shift", rotation=90, va="center",
             ha="right", color=INK, fontsize=8.8)

    if delay is not None:
        yx = shift_traj[alarm_step]
        axB.annotate(
            f"Shiryaev–Roberts fires\n+{delay} rounds after the shift",
            xy=(alarm_step, yx), xytext=(alarm_step + len(r_shift) * 0.14, yx * 30),
            color=INK, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))

    axB.text(len(r_null) * 0.5, null_traj.max() * 0.6,
             "plain martingale stays quiet\n(decayed over the long null)",
             color=NULL, fontsize=9, ha="center", va="top")

    axB.set_yscale("log")
    axB.set_ylim(0.2, max(shift_traj.max(), SR_THRESHOLD) * 12)
    axB.set_xlim(0, len(r_shift))
    axB.set_xlabel("monitoring round", fontsize=10.5)
    axB.set_ylabel("Shiryaev–Roberts statistic  R$_t$", fontsize=10.5)
    axB.set_title("Coverage is now observable  (WATCH monitor)",
                  fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=24)
    axB.text(0, 1.005,
             "quiet 20/20 real METR-LA seeds · no certificate change",
             transform=axB.transAxes, fontsize=9.5, color=MUTED, va="bottom")
    axB.legend(loc="upper right", frameon=False, fontsize=9.5,
               bbox_to_anchor=(1.0, 0.98))
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)
    axB.grid(axis="y", color=GRID, lw=0.9, which="major", zorder=0)
    axB.set_axisbelow(True)
    axB.tick_params(length=0)

    fig.tight_layout(pad=1.4)
    out = ROOT / "assets/live_wiring_2026.png"
    fig.savefig(out, dpi=200, facecolor=SURF, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"SR monitor: null peak {null_traj.max():.0f}  shift peak {shift_traj.max():.3g}  "
          f"alarm +{delay} rounds")


if __name__ == "__main__":
    main()
