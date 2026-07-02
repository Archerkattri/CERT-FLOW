"""width_methods.png -- the certified-width story on real METR-LA, all methods.

One comprehensive chart of every width-relevant pricing option, measured on the
SAME paired real-traffic run (10 seeds x 288 rounds, identical replay per mode)
so the bars are directly comparable. Read from the committed benchmark
scripts/out/width_attack.json (produced by scripts/run_width_attack.py).

Each bar is the median certified gap (UB - LB) relative to the shipped default
(per-edge Bonferroni). Tighter (left, green) is better; wider (right,
vermillion) is worse. The four a-priori distribution-free modes all hold
coverage at 0.0000 violations. The licensed ShrinkLicense tier is a DIFFERENT
object -- an a-posteriori, anytime-valid Tier-2 radius -- so it is drawn hatched
and carries its measured shadow-miscoverage label; it never replaces the
certificate.

Reproduce:  python scripts/viz_gen/width_methods.py
Writes assets/width_methods.png.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Okabe-Ito colorblind-safe palette (matches the repo/site inks and the other
# figure generators). Validated: adjacent-pair CVD deltaE >= 17.
INK = "#0a111e"
MUTED = "#5b6b82"
GRIDC = "#e4e9f0"
SURF = "#ffffff"
TIGHTER = "#009E73"   # green      -- a-priori sum-level UB (tighter, good)
WIDER = "#D55E00"     # vermillion -- PASC (wider on long paths, the negative)
LICENSED = "#0072B2"  # blue       -- Tier-2 licensed shrink (different claim)


def load_metrla():
    data = json.loads((ROOT / "scripts/out/width_attack.json").read_text())
    rows = {r["mode"]: r for r in data["metr_la"]}
    base = rows["default"]["gap_median"]
    out = {}
    for mode in ("default", "sum_aware", "pasc", "cia-ub", "shrink"):
        r = rows[mode]
        out[mode] = r
    return base, out


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

    base, r = load_metrla()

    def pct(g):
        return (g / base - 1.0) * 100.0

    # bars top -> bottom: worst (PASC, wider) down to best (ShrinkLicense).
    # (short y-label, delta%, median_s, color, hatched, detail)
    bars = [
        ("PASC\n(block-max)", pct(r["pasc"]["gap_median"]), r["pasc"]["gap_median"],
         WIDER, False, "0 viol."),
        ("sum-aware UB\n(sum_aware_ub)", pct(r["sum_aware"]["gap_median"]),
         r["sum_aware"]["gap_median"], TIGHTER, False, "0 viol."),
        ("CIA-UB\n(cia_path_certificate)", pct(r["cia-ub"]["gap_median"]),
         r["cia-ub"]["gap_median"], TIGHTER, False, "0 viol. · 20% fallback"),
        ("ShrinkLicense T2\n(licensed)",
         pct(r["shrink"]["shrunk_gap_median"]), r["shrink"]["shrunk_gap_median"],
         LICENSED, True, "0.51% viol."),
    ]

    ys = list(range(len(bars) - 1, -1, -1))  # top row highest y

    fig, ax = plt.subplots(figsize=(10.6, 4.7))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    fig.subplots_adjust(left=0.22, right=0.965, top=0.79, bottom=0.16)

    xlo, xhi = -104, 46
    for y, (name, d, med, col, hatch, detail) in zip(ys, bars):
        ax.barh(y, d, height=0.58, color=col, edgecolor=SURF, linewidth=2,
                zorder=3, hatch=("////" if hatch else None))
        ha = "left" if d > 0 else "right"
        off = 2.0 if d > 0 else -2.0
        ax.text(d + off, y + 0.15, f"{d:+.1f}%", ha=ha, va="center",
                color=col, fontsize=15, fontweight="bold", zorder=5)
        ax.text(d + off, y - 0.21, f"median {med:,.0f} s · {detail}", ha=ha,
                va="center", color=MUTED, fontsize=8.6, zorder=5)

    ax.axvline(0, color=INK, lw=1.6, zorder=4)
    ax.text(0.4, len(bars) - 0.30,
            f"Bonferroni baseline (shipped default) · median {base:,.0f} s",
            ha="left", va="bottom", color=INK, fontsize=9.2, fontweight="bold")
    ax.text(xlo + 2, -0.74, "◄ tighter (better)", ha="left", va="center",
            color=TIGHTER, fontsize=10, fontweight="bold")
    ax.text(xhi - 2, -0.74, "wider (worse) ►", ha="right", va="center",
            color=WIDER, fontsize=10, fontweight="bold")

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.98, len(bars) + 0.05)
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=10.2, color=INK)
    ax.set_xticks([-100, -80, -60, -40, -20, 0, 20, 40])
    ax.set_xticklabels(["−100", "−80", "−60", "−40", "−20", "0", "+20", "+40"], fontsize=9)
    ax.set_xlabel("certified gap (UB − LB) vs default Bonferroni   (%)", fontsize=10.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRIDC, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    fig.text(0.02, 0.965,
             "Certified width on real METR-LA: the union-bound tax, and where it is recoverable",
             fontsize=13.2, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.905,
             "10 seeds × 288 rounds, paired replay · the three a-priori sum-level modes hold "
             "0.0000 violations;\nsum-level calibration tightens 24–27%, while block-max (PASC) "
             "starves on long paths (L≈14–18). The\nhatched licensed tier is a-posteriori "
             "(anytime-valid) — a different claim, not the distribution-free certificate.",
             fontsize=9.0, color=MUTED, ha="left", va="top", linespacing=1.25)

    out = ROOT / "assets/width_methods.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB)")
    for name, d, med, *_ in bars:
        print(f"  {name.splitlines()[0]:22s} {d:+6.1f}%   median {med:,.0f} s")


if __name__ == "__main__":
    main()
