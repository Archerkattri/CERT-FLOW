"""sensing_regret.png -- objective-matched hybrid sensing on real METR-LA.

Median route regret (true realized cost - true optimum, seconds) for four
sensing policies run on the SAME METR-LA replay (10 seeds x 288 rounds), read
from the committed benchmark scripts/out/hybrid_real_metrla.json (produced by
scripts/run_hybrid_sensing.py). All four hold coverage at 0.0000 violations and
cost ~1 ms/round; they differ only in WHICH edge the one-per-round budget senses.

Hybrid (objective-matched: gap-directed while the target gap is attainable, VOI
toward the expected-best route when it is not) dominates the field. Pure
gap-directed -- the shipped default -- pays for spending its budget on
certificate-relevant but route-marginal edges in the never-attainable real-
traffic regime.

Reproduce:  python scripts/viz_gen/sensing_regret.py
Writes assets/sensing_regret.png.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

INK = "#0a111e"
MUTED = "#5b6b82"
GRIDC = "#e4e9f0"
SURF = "#ffffff"
WIN = "#009E73"     # green  -- hybrid (the recommended winner)
REST = "#5b6b82"    # muted  -- dominated baselines
DEFAULT = "#0072B2"  # blue  -- the current shipped default (pure gap)


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

    agg = json.loads((ROOT / "scripts/out/hybrid_real_metrla.json").read_text())["aggregate"]

    # (label, key, color, note)
    rows = [
        ("hybrid\n(objective-matched)", "hybrid (objective-matched)", WIN,
         "recommended"),
        ("pure gap-directed\n(shipped default)", "cert (pure gap, default)", DEFAULT,
         "default"),
        ("max-age\n(freshness)", "max_age (freshness)", REST, ""),
        ("random", "random", REST, ""),
    ]
    labels = [r[0] for r in rows]
    vals = [agg[r[1]]["regret_median"] for r in rows]
    cols = [r[2] for r in rows]
    notes = [r[3] for r in rows]

    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.80, bottom=0.17)

    ax.bar(x, vals, width=0.64, color=cols, edgecolor=SURF, linewidth=2, zorder=3)
    top = max(vals)
    for xi, v, note in zip(x, vals, notes):
        ax.text(xi, v + top * 0.02, f"{v:.1f} s", ha="center", va="bottom",
                color=INK, fontsize=13, fontweight="bold", zorder=5)
        if note:
            ax.text(xi, v + top * 0.09, note, ha="center", va="bottom",
                    color=MUTED, fontsize=9, style="italic", zorder=5)

    # -41% bracket from hybrid to the default
    hy, pg = vals[0], vals[1]
    ax.annotate("", xy=(1, pg), xytext=(0, hy),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0, ls=(0, (4, 3))))
    ax.text(0.06, top * 0.72, f"−{(1 - hy / pg) * 100:.0f}%\nvs default",
            ha="left", va="center", color=WIN, fontsize=12.5, fontweight="bold",
            linespacing=1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylabel("median route regret   (true cost − optimum, s)", fontsize=10.5)
    ax.set_ylim(0, top * 1.28)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDC, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    fig.text(0.12, 0.965,
             "Objective-matched sensing wins on real METR-LA route regret",
             fontsize=13.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.12, 0.905,
             "10 seeds × 288 rounds · all policies at 0.0000 coverage violations and ~1 ms/round · "
             "hybrid is the\nonly policy on the Pareto frontier (pays ≤+21% certified width only "
             "where no width certifies)",
             fontsize=9.0, color=MUTED, ha="left", va="top", linespacing=1.25)

    out = ROOT / "assets/sensing_regret.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    for lbl, v in zip([r[1] for r in rows], vals):
        print(f"  {lbl:34s} {v:8.2f} s")


if __name__ == "__main__":
    main()
