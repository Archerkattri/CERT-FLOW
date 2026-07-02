"""Two-panel figure for the 2026 live-wiring layer (README + project page).

Panel (a) -- certified width on REAL METR-LA: median UB-LB for the default
Bonferroni path calibration vs the experimental PASC joint radius, read from the
committed benchmark table (20 seeds x 288 rounds). Both hold 0.0000 violations;
PASC is +25.1% wider -- the honest negative (opposite of the synthetic grid).

Panel (b) -- coverage made observable: the Shiryaev-Roberts statistic from the
WATCH testability layer, replayed faithfully from run_watch_testability.py's
seed-0 streams. Under the correctly-modelled (null) stream it stays quiet below
the alarm threshold; on a stream with an injected regime shift at round 250 it
crosses the threshold ~7 rounds later. Same detector CERT-FLOW now runs live
(planner.sr), at zero cost to the certificate.

Reproduce:  PYTHONPATH=src python scripts/viz_gen/live_wiring_fig.py
Writes assets/live_wiring_2026.png. Needs no data/ (synthetic panel + the
committed results/live_wiring/table.json).
"""
from __future__ import annotations

import json
import math
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
BONF = "#0072B2"   # deep blue  (default calibration)
PASC = "#E8722D"   # orange     (experimental)
NULL = "#009E73"   # green      (quiet = good)
SHIFT = "#D55E00"  # vermillion (alarm)
INK = "#0a111e"
MUTED = "#5b6b82"
GRID = "#e4e9f0"
SURF = "#ffffff"

# ---- experiment constants (mirror scripts/run_watch_testability.py defaults) --
ROUNDS, ALPHA, RHO_W, DRIFT, NOISE = 3000, 0.1, 0.98, 0.05, 0.10
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

    # ---- data: panel (a) from the committed benchmark table -------------------
    tbl = json.loads((ROOT / "results/live_wiring/table.json").read_text())
    rows = {r["mode"].split(" (")[0].split(" +")[0]: r for r in tbl["rows"]}
    b_med = rows["Bonferroni"]["gap_median"]
    p_med = rows["PASC"]["gap_median"]
    b_mean = rows["Bonferroni"]["gap_mean"]
    p_mean = rows["PASC"]["gap_mean"]
    delta_pct = (p_med / b_med - 1.0) * 100.0

    # ---- data: panel (b) faithful seed-0 replay -------------------------------
    rng = np.random.default_rng(0)
    null_scores = _make_stream(rng, ROUNDS, DRIFT, NOISE)
    shift_scores = _make_stream(rng, ROUNDS, DRIFT, NOISE,
                                shift_at=SHIFT_AT, shift_drift=DRIFT * 40.0)
    null_traj, _ = _sr_trajectory(null_scores, SR_THRESHOLD)
    shift_traj, alarm_step = _sr_trajectory(shift_scores, SR_THRESHOLD)
    shift_mon = SHIFT_AT - WARMUP                     # shift, in monitoring rounds
    delay = None if alarm_step is None else alarm_step - shift_mon

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.0, 1.28]})
    fig.patch.set_facecolor(SURF)

    # ============================ PANEL (a) ====================================
    axA.set_facecolor(SURF)
    x = [0, 1]
    meds = [b_med, p_med]
    means = [b_mean, p_mean]
    cols = [BONF, PASC]
    bars = axA.bar(x, meds, width=0.62, color=cols, edgecolor=SURF, linewidth=2,
                   zorder=3)
    # mean marker (thin tick above the median bar) -- honest second stat
    for xi, m in zip(x, means):
        axA.plot([xi - 0.2, xi + 0.2], [m, m], color=INK, lw=1.6, zorder=5)
    axA.plot([], [], color=INK, lw=1.6, label="mean")  # legend proxy
    for xi, m in zip(x, meds):
        axA.text(xi, m - max(meds) * 0.05, f"{m:,.0f}", ha="center", va="top",
                 color=SURF, fontsize=13, fontweight="bold", zorder=6)
    for xi, m in zip(x, means):
        axA.text(xi, m + max(meds) * 0.02, f"mean {m:,.0f}", ha="center",
                 va="bottom", color=MUTED, fontsize=9.5)

    # +25.1% delta bracket
    top = max(means) * 1.22
    axA.plot([0, 0, 1, 1], [means[0] * 1.06, top, top, means[1] * 1.06],
             color=MUTED, lw=1.1, zorder=2)
    axA.text(0.5, top * 1.005, f"+{delta_pct:.1f}%  width", ha="center",
             va="bottom", color=SHIFT, fontsize=11.5, fontweight="bold")

    axA.set_xticks(x)
    axA.set_xticklabels(["Bonferroni\n(default)", "PASC\n(experimental)"],
                        fontsize=11, color=INK)
    axA.set_ylabel("certified width  UB − LB   (median, s)", fontsize=10.5)
    axA.set_ylim(0, top * 1.16)
    axA.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    axA.set_title("Certified width on real METR-LA",
                  fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=10)
    axA.text(0, 1.005, "20 seeds × 288 rounds · violations 0.0000 both",
             transform=axA.transAxes, fontsize=9.5, color=MUTED, va="bottom")
    axA.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.2,
               bbox_to_anchor=(0.0, 0.94))
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axA.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    axA.set_axisbelow(True)
    axA.tick_params(length=0)

    # ============================ PANEL (b) ====================================
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
                  fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=10)
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

    fig.tight_layout(pad=1.4, w_pad=3.0)
    out = ROOT / "assets/live_wiring_2026.png"
    fig.savefig(out, dpi=200, facecolor=SURF, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"panel a: Bonf median {b_med:,.0f}s  PASC {p_med:,.0f}s  (+{delta_pct:.1f}%)")
    print(f"panel b: null peak {null_traj.max():.0f}  shift peak {shift_traj.max():.3g}  "
          f"alarm +{delay} rounds")


if __name__ == "__main__":
    main()
