"""pasc_vs_bonferroni.png -- one honest visual, both truths.

The 2026 PASC joint per-edge radius replaces the alpha/L Bonferroni union with a
single block-max quantile. Whether that TIGHTENS the certificate depends on how
long the optimistic paths are relative to the calibration buffer:

  * SYNTHETIC 4x4 grid (short paths, L=6): PASC is TIGHTER -- computed live here
    by replaying the exact tests/test_live_wiring.py pooled-seed protocol.
  * REAL METR-LA traffic (long paths, L~14-18): PASC is WIDER (+25.1%) -- read
    from the committed 20-seed x 288-round benchmark (results/live_wiring/
    table.json), the honest negative that keeps PASC an experimental flag.

Both regimes hold coverage at 0.0000 violations, so this is purely a width
story. A static diverging bar (animation adds nothing) shows the sign flip.

Reproduce:  PYTHONPATH=src python scripts/viz_gen/pasc_vs_bonferroni.py
Writes assets/animations/pasc_vs_bonferroni.png.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from certflow.cert import CertPlanner, PlannerConfig  # noqa: E402
from certflow.drift import grid_world  # noqa: E402

INK = "#0a111e"
MUTED = "#5b6b82"
GRIDC = "#e4e9f0"
SURF = "#ffffff"
WIN = "#009E73"       # green      -- PASC tighter (good)
LOSE = "#D55E00"      # vermillion -- PASC wider (regression)


def _grid_gaps(mode: str, seed: int, rounds: int = 200):
    """One seed of the tests/test_live_wiring.py grid protocol."""
    world = grid_world(4, 4, seed=seed, kind="bounded", noise_scale=0.05, rho=0.02)
    p = CertPlanner(world, (0, 0), (3, 3),
                    PlannerConfig(epsilon=3.0, alpha_prime=0.2, delta=1.0,
                                  path_calibration=mode))
    gaps = []
    for _ in range(rounds):
        cert, _ = p.round()
        if cert.valid and math.isfinite(cert.gap):
            gaps.append(cert.gap)
    return gaps


def grid_delta():
    pb, pp = [], []
    for seed in (7, 5, 11, 3, 42):
        pb += _grid_gaps("bonferroni", seed)
        pp += _grid_gaps("pasc", seed)
    mb, mp = st.median(pb), st.median(pp)
    return mb, mp, (mp / mb - 1.0) * 100.0


def metrla_delta():
    tbl = json.loads((ROOT / "results/live_wiring/table.json").read_text())
    rows = {r["mode"].split(" (")[0].split(" +")[0]: r for r in tbl["rows"]}
    mb = rows["Bonferroni"]["gap_median"]
    mp = rows["PASC"]["gap_median"]
    return mb, mp, (mp / mb - 1.0) * 100.0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 1.0, "text.color": INK, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })

    gb, gp, gd = grid_delta()
    mb, mp, md = metrla_delta()

    # rows top->bottom: grid (win), METR-LA (lose)
    rows = [
        ("synthetic 4×4 grid",
         f"short paths  L≈6   ·   Bonf {gb:.2f} → PASC {gp:.2f}",
         gd, WIN, "PASC tighter"),
        ("real METR-LA traffic",
         f"long paths  L≈14–18   ·   Bonf {mb:,.0f}s → PASC {mp:,.0f}s",
         md, LOSE, "PASC wider"),
    ]
    ys = [1, 0]

    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    fig.subplots_adjust(left=0.20, right=0.965, top=0.74, bottom=0.19)

    xlim = 33
    for y, (name, sub, d, col, tag) in zip(ys, rows):
        ax.barh(y, d, height=0.46, color=col, edgecolor=SURF, linewidth=2,
                zorder=3)
        # value + tag stacked at the bar end (relief for the WARN-contrast orange;
        # stacking avoids horizontal overflow past the axis)
        ha = "left" if d > 0 else "right"
        off = 1.1 if d > 0 else -1.1
        ax.text(d + off, y + 0.15, f"{d:+.1f}%", ha=ha, va="center",
                color=col, fontsize=15, fontweight="bold", zorder=5)
        ax.text(d + off, y - 0.16, tag, ha=ha, va="center", color=MUTED,
                fontsize=9.5, zorder=5)
        # row label + detail on the left margin
        ax.text(-xlim - 1.5, y + 0.12, name, ha="right", va="center",
                color=INK, fontsize=12, fontweight="bold")
        ax.text(-xlim - 1.5, y - 0.16, sub, ha="right", va="center",
                color=MUTED, fontsize=8.8)

    ax.axvline(0, color=INK, lw=1.6, zorder=4)
    ax.text(0, 1.62, "Bonferroni\nbaseline", ha="center", va="bottom",
            color=INK, fontsize=8.8, linespacing=1.05)
    # direction cues
    ax.text(-xlim + 1, -0.62, "◄ tighter (better)", ha="left", va="center",
            color=WIN, fontsize=9.5, fontweight="bold")
    ax.text(xlim - 1, -0.62, "wider (worse) ►", ha="right", va="center",
            color=LOSE, fontsize=9.5, fontweight="bold")

    ax.set_xlim(-xlim, xlim); ax.set_ylim(-0.8, 1.75)
    ax.set_yticks([])
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax.set_xticklabels(["−30", "−20", "−10", "0", "+10", "+20", "+30"], fontsize=9)
    ax.set_xlabel("PASC certified width vs default Bonferroni  (%)", fontsize=10.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRIDC, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0)

    fig.text(0.20, 0.955,
             "PASC joint radius: tighter on short paths, wider on long ones",
             fontsize=13.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.20, 0.885,
             "both regimes hold coverage at 0.0000 violations — this is purely a "
             "width story, and the sign flips",
             fontsize=9.6, color=MUTED, ha="left", va="top")

    out = ROOT / "assets/animations/pasc_vs_bonferroni.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB)")
    print(f"grid: Bonf {gb:.3f} PASC {gp:.3f} -> {gd:+.1f}%")
    print(f"METR-LA: Bonf {mb:,.1f} PASC {mp:,.1f} -> {md:+.1f}%")


if __name__ == "__main__":
    main()
