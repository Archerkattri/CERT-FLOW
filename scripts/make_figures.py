"""make_figures.py — Publication figures for the CERT paper.

Generates four vector PDFs in paper/figures/:
  fig_gap_trajectory.pdf    — Certificate gap / band / OPT for one episode
  fig_coverage_vs_claim.pdf — Empirical coverage vs claimed level
  fig_regret_bars.pdf       — Travel regret by sensing policy
  fig_bound_validity.pdf    — AD*/CERT validity-vs-width scatter

Run from the repo root:
    cert_env/bin/python scripts/make_figures.py

All figures are deterministic (seed fixed); running twice gives identical PDFs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
# Grayscale-legible, colorblind-safe palette (Wong + extended grey)
BLK  = "#000000"
GRY1 = "#555555"   # dark grey
GRY2 = "#999999"   # mid grey
GRY3 = "#CCCCCC"   # light grey
BLUE = "#0072B2"
ORG  = "#E69F00"
GRN  = "#009E73"
SKY  = "#56B4E9"
RED  = "#D55E00"
VIO  = "#CC79A7"

SINGLE_W = 3.35   # single-column width inches (≈85 mm)
DOUBLE_W = 6.9    # double-column width inches (≈175 mm)
FONT_LABEL = 9
FONT_TICK  = 8
FONT_ANNOT = 8
FONT_LEGEND = 8

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "axes.titlesize": 9,
    "axes.labelsize": FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "figure.dpi": 150,
    "pdf.fonttype": 42,   # embed fonts (not Type 3)
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
})


# ===========================================================================
# FIGURE 1 — Certificate gap trajectory for one representative episode
# ===========================================================================

def make_fig1_gap_trajectory():
    """Re-simulate a single 6x6 bounded rho=0.02 episode (seed=3) and plot
    the certificate [LB, UB] band, the gap (UB-LB), and the true OPT over
    rounds, with warm-up / valid / certified phases shaded."""

    from certflow.cert import CertPlanner, PlannerConfig
    from certflow.drift import grid_world
    from certflow.oracle import opt as oracle_opt

    SEED = 3
    ROWS, COLS = 6, 6
    RHO  = 0.02
    EPS  = 5.0

    world = grid_world(ROWS, COLS, seed=SEED, kind="bounded", rho=RHO,
                       noise_family="gaussian", noise_scale=0.05)
    start, goal = (0, 0), (ROWS - 1, COLS - 1)

    cfg = PlannerConfig(
        epsilon=EPS,
        alpha_prime=0.2,
        rho_w=0.99,
        eps_tv=1e-4,
        gamma_aci=0.01,
        delta=1.0,
        rho_hat_over_rho=1.0,
        sensing_policy="cert",
        initial_survey=True,
        latent_margin=1.0,
        use_aci=True,
        anneal_alpha=True,
        min_certify_confidence=0.5,
    )
    planner = CertPlanner(world, start, goal, cfg)

    MAX_ROUNDS = 300
    ts, lbs, ubs, opts, valids, certs, confs = [], [], [], [], [], [], []

    for _ in range(MAX_ROUNDS):
        t_now = planner.t
        cert, _ = planner.round()
        _, true_opt = oracle_opt(world, t_now, start, goal)

        ts.append(t_now)
        lbs.append(cert.lb)
        ubs.append(cert.ub)
        opts.append(true_opt)
        valids.append(cert.valid)
        certs.append(cert.valid and cert.gap <= EPS)
        confs.append(cert.confidence)

    ts   = np.array(ts,   dtype=float)
    lbs  = np.array(lbs,  dtype=float)
    ubs  = np.array(ubs,  dtype=float)
    opts = np.array(opts, dtype=float)
    gaps = ubs - lbs
    valids = np.array(valids, dtype=bool)
    certs  = np.array(certs,  dtype=bool)

    # Phase boundaries
    warmup_end = None
    for i, v in enumerate(valids):
        if v:
            warmup_end = i
            break
    cert_start = None
    for i, c in enumerate(certs):
        if c:
            cert_start = i
            break

    # --- Plot ---
    fig, (ax_band, ax_gap) = plt.subplots(
        2, 1, figsize=(SINGLE_W, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
        layout="constrained",
    )

    r = np.arange(MAX_ROUNDS)

    # Phase shading (same on both axes)
    def shade_phases(ax):
        if warmup_end is not None and warmup_end > 0:
            ax.axvspan(0, warmup_end, color=GRY3, alpha=0.40, linewidth=0,
                       label="warm-up")
        v_end = cert_start if cert_start is not None else MAX_ROUNDS
        ax.axvspan(warmup_end if warmup_end is not None else 0,
                   v_end, color=GRY2, alpha=0.18, linewidth=0,
                   label="valid (uncertified)")
        if cert_start is not None:
            ax.axvspan(cert_start, MAX_ROUNDS, color=SKY, alpha=0.20,
                       linewidth=0, label="certified")

    shade_phases(ax_band)
    shade_phases(ax_gap)

    # [LB, UB] band
    ax_band.fill_between(r, lbs, ubs, color=SKY, alpha=0.35, linewidth=0,
                         label="[LB, UB] band")
    ax_band.plot(r, lbs, color=BLUE, linewidth=0.9, linestyle="--",
                 label="LB")
    ax_band.plot(r, ubs, color=BLUE, linewidth=0.9, linestyle="-",
                 label="UB")
    ax_band.plot(r, opts, color=BLK, linewidth=1.2, linestyle="-",
                 label="true OPT")

    ax_band.set_ylabel("Route cost", fontsize=FONT_LABEL)

    # Gap trajectory
    ax_gap.plot(r, gaps, color=RED, linewidth=1.2, label="gap (UB−LB)")
    ax_gap.axhline(EPS, color=GRY1, linewidth=0.8, linestyle=":",
                   label=f"$\\varepsilon={EPS:.0f}$")

    ax_gap.set_xlabel("Round", fontsize=FONT_LABEL)
    ax_gap.set_ylabel("Gap", fontsize=FONT_LABEL)

    # Legend on upper panel only
    # Build custom legend entries
    legend_handles = [
        mpatches.Patch(color=GRY3, alpha=0.60, label="warm-up"),
        mpatches.Patch(color=GRY2, alpha=0.50, label="valid"),
        mpatches.Patch(color=SKY,  alpha=0.50, label="certified"),
        Line2D([0], [0], color=BLK,  linewidth=1.2, label="true OPT"),
        mpatches.Patch(color=SKY,  alpha=0.55, label="[LB, UB]"),
        Line2D([0], [0], color=BLUE, linewidth=0.9, linestyle="--",
               label="LB"),
        Line2D([0], [0], color=BLUE, linewidth=0.9, linestyle="-",
               label="UB"),
    ]
    ax_band.legend(handles=legend_handles, fontsize=7, ncol=4,
                   loc="lower center", bbox_to_anchor=(0.5, 1.02),
                   frameon=False)

    ax_gap.legend(fontsize=FONT_LEGEND, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, 1.02), frameon=False)

    out = FIG_DIR / "fig_gap_trajectory.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  ({out.stat().st_size} bytes)")
    return out


# ===========================================================================
# FIGURE 2 — Empirical coverage vs claimed level
# ===========================================================================

def make_fig2_coverage_vs_claim():
    """Horizontal dot plot: empirical coverage with CP error bars vs claimed
    level for headline conditions.  Data transcribed exactly from
    docs/results tables (post-annealing rerun and real-data sections)."""

    # -----------------------------------------------------------------------
    # Data — transcribed verbatim from docs/results/tier0-coverage.md
    # (post-annealing rows) and metr-la.md / pems-bay.
    # Coverage and CI are empirical (Clopper-Pearson, reported in the paper).
    # Claimed is the mean annealed level from table.json.
    # -----------------------------------------------------------------------

    # Load from table.json for the synthetic conditions
    tier0_path = REPO / "results" / "tier0" / "table.json"
    with open(tier0_path) as f:
        tier0 = json.load(f)

    # Build lookup by label
    t0 = {d["label"]: d for d in tier0}

    # Synthetic conditions we want (post-annealing table)
    synth_rows = [
        ("static ($\\rho=0$)",              "static (rho=0)"),
        ("bounded $\\rho=0.005$",           "bounded rho=0.005"),
        ("bounded $\\rho=0.02$",            "bounded rho=0.02"),
        ("bounded $\\rho=0.05$",            "bounded rho=0.05"),
        ("off-model: jump",                 "off-model: jump"),
        ("off-model: periodic",             "off-model: periodic"),
        ("misspec $\\hat\\rho=0.5\\times$", "misspec rho_hat=0.5*rho"),
        ("misspec $\\hat\\rho=2\\times$",   "misspec rho_hat=2*rho"),
    ]

    # Real-data rows — from docs/results/metr-la.md (manually checked)
    real_rows = [
        # label, coverage, ci_lo, ci_hi, claimed
        ("METR-LA ($\\rho=p95$)", 1.000, 0.999, 1.000, 0.588),
        ("METR-LA ($\\rho=p75$)", 1.000, 0.999, 1.000, 0.572),
        ("PEMS-BAY ($\\rho=p95$)", 1.000, 0.999, 1.000, 0.683),
        ("PEMS-BAY p75+adapt.",   0.982, 0.978, 0.985, 0.637),
    ]

    rows = []
    for label, key in synth_rows:
        if key not in t0:
            print(f"  WARNING: missing key '{key}' in tier0/table.json")
            continue
        d = t0[key]
        rows.append((label,
                     d["coverage"],
                     d["cov_ci_lo"],
                     d["cov_ci_hi"],
                     d["claimed_mean"]))

    for r in real_rows:
        rows.append(r)

    labels   = [r[0] for r in rows]
    coverage = np.array([r[1] for r in rows])
    ci_lo    = np.array([r[2] for r in rows])
    ci_hi    = np.array([r[3] for r in rows])
    claimed  = np.array([r[4] for r in rows])

    n = len(rows)
    y = np.arange(n)[::-1]   # top to bottom

    fig, ax = plt.subplots(figsize=(SINGLE_W, n * 0.32 + 0.9))

    # Error bars for empirical coverage
    err_lo = coverage - ci_lo
    err_hi = ci_hi - coverage
    ax.errorbar(coverage, y,
                xerr=[err_lo, err_hi],
                fmt="o", color=BLUE, markersize=4.5, linewidth=1.0,
                capsize=2.5, label="empirical coverage\n(95% CP interval)")

    # Claimed level markers
    ax.scatter(claimed, y, marker="|", s=60, color=RED, linewidths=1.4,
               zorder=4, label="claimed level")

    # Reference line at 1.0
    ax.axvline(1.0, color=GRY2, linewidth=0.7, linestyle="--")

    # Diagonal guide: coverage == claimed is the ideal
    x_guide = np.linspace(0, 1, 200)
    # (don't draw diagonal; the point is coverage >= claimed, not =)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Coverage probability", fontsize=FONT_LABEL)
    ax.set_xlim(0.45, 1.04)
    ax.tick_params(axis="x", labelsize=FONT_TICK)

    # Shade real-data rows
    n_real = len(real_rows)
    ax.axhspan(-0.5, n_real - 0.5, color=GRY3, alpha=0.25, linewidth=0,
               zorder=0)
    ax.text(0.453, n_real / 2 - 0.5, "real\ntraffic", fontsize=6.5,
            color=GRY1, va="center", ha="left", rotation=90)

    ax.legend(fontsize=FONT_LEGEND, loc="lower right",
              bbox_to_anchor=(1.0, 1.01), frameon=False)
    fig.tight_layout()

    out = FIG_DIR / "fig_coverage_vs_claim.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  ({out.stat().st_size} bytes)")
    return out


# ===========================================================================
# FIGURE 3 — Travel regret bars (sensing policy comparison)
# ===========================================================================

def make_fig3_regret_bars():
    """Grouped horizontal bars: travel-regret (mean) by sensing policy
    across three map groups (Tier-2 synthetic, DAO dungeon, Berlin street).
    Data from results/tier2/table.json and results/movingai/table.json."""

    tier2_path  = REPO / "results" / "tier2"  / "table.json"
    movai_path  = REPO / "results" / "movingai" / "table.json"
    extern_path = REPO / "results" / "extern_baselines" / "table.json"

    with open(tier2_path)  as f: tier2  = json.load(f)
    with open(movai_path)  as f: movai  = json.load(f)
    with open(extern_path) as f: extern = json.load(f)

    # -----------------------------------------------------------------------
    # Tier-2 synthetic: cert/random/max_age/max_width + blind at B=20
    # The paper table cites B=20 as the representative budget row.
    # We add voi/hybrid from extern_baselines part_b (also B=20, 15 seeds).
    # -----------------------------------------------------------------------
    def tier2_mean(label_fragment, t2=tier2):
        """Extract regret_mean for the row whose label contains the fragment."""
        for d in t2:
            if label_fragment in d["label"]:
                return d["regret_mean"]
        return float("nan")

    t2_cert     = tier2_mean("cert       | B=20")
    t2_random   = tier2_mean("random     | B=20")
    t2_max_age  = tier2_mean("max_age    | B=20")
    t2_max_wid  = tier2_mean("max_width  | B=20")
    t2_blind    = tier2_mean("no-cert")

    # voi and hybrid from extern part_b
    ext_b = {d["policy"]: d["regret_mean"] for d in extern["part_b"]}
    t2_voi    = ext_b.get("voi",    float("nan"))
    t2_hybrid = ext_b.get("hybrid", float("nan"))

    # -----------------------------------------------------------------------
    # MovingAI: cert/random/max_age/blind per map
    # -----------------------------------------------------------------------
    def movai_mean(map_name, policy, mv=movai):
        for d in mv:
            if d["map"] == map_name and d["policy"] == policy:
                return d["regret_mean"]
        return float("nan")

    # -----------------------------------------------------------------------
    # Build panel data
    # -----------------------------------------------------------------------
    # Policies to plot (in bar order):
    # cert, hybrid, voi, random, max_age, max_width, blind
    # Not all policies appear in every dataset; use NaN for absent ones.

    policies = ["cert", "hybrid", "voi", "random", "max_age", "max_width", "blind"]
    pol_labels = ["cert (gap-directed)", "hybrid (obj-matched)", "VOI (CTP-RS)",
                  "random", "max-age", "max-width", "drive blind"]
    pol_colors = [BLUE, GRN, SKY, GRY2, ORG, VIO, BLK]
    pol_hatch  = [None, "//", "..", None, "xx", "oo", "\\\\"]

    # Group 1: Tier-2 synthetic, B=20 (25 seeds, 10x10)
    synth_vals = [t2_cert, t2_hybrid, t2_voi, t2_random, t2_max_age, t2_max_wid, t2_blind]

    # Group 2: DAO dungeon (15 seeds)
    dao_vals = [
        movai_mean("dao_arena", "cert"),
        float("nan"),  # hybrid not in movingai run
        float("nan"),
        movai_mean("dao_arena", "random"),
        movai_mean("dao_arena", "max_age"),
        float("nan"),
        movai_mean("dao_arena", "blind"),
    ]

    # Group 3: Berlin street (15 seeds)
    berlin_vals = [
        movai_mean("street_berlin", "cert"),
        float("nan"),
        float("nan"),
        movai_mean("street_berlin", "random"),
        movai_mean("street_berlin", "max_age"),
        float("nan"),
        movai_mean("street_berlin", "blind"),
    ]

    groups = [
        ("Synthetic 10x10\n(B=20, 25 seeds)", synth_vals),
        ("DAO dungeon\n(15 seeds)",            dao_vals),
        ("Berlin street\n(15 seeds)",          berlin_vals),
    ]

    n_pol = len(policies)
    n_grp = len(groups)
    bar_h  = 0.12      # bar height
    grp_gap = 0.18     # extra gap between groups

    fig_h = n_grp * (n_pol * bar_h + grp_gap) + 0.6
    fig, ax = plt.subplots(figsize=(DOUBLE_W * 0.72, fig_h))

    # Build y-positions: groups from top to bottom
    group_centers = []
    yticks, yticklabels = [], []

    for g_idx, (g_label, g_vals) in enumerate(groups):
        # y offset: place group g_idx below the previous
        base_y = -(g_idx * (n_pol * bar_h + grp_gap))
        group_centers.append(base_y - (n_pol - 1) * bar_h / 2)

        for p_idx, (val, col, hatch) in enumerate(zip(g_vals, pol_colors, pol_hatch)):
            y_pos = base_y - p_idx * bar_h
            yticks.append(y_pos)
            yticklabels.append(pol_labels[p_idx] if g_idx == 0 else "")
            if not np.isnan(val):
                ax.barh(y_pos, val, height=bar_h * 0.85,
                        facecolor=col, hatch=hatch,
                        edgecolor=GRY1 if hatch else "none",
                        linewidth=0.4, alpha=0.88,
                        label=pol_labels[p_idx] if g_idx == 0 else "_nolegend_")
                ax.text(val + 0.05, y_pos, f"{val:.2f}",
                        va="center", ha="left", fontsize=6.5)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=FONT_TICK)
    ax.set_xlabel("Travel regret (mean)", fontsize=FONT_LABEL)
    ax.axvline(0, color=GRY2, linewidth=0.6, linestyle="--")

    # Group labels positioned as right-margin annotations don't work well
    # before xlim is known; add them after xlim is set
    ax.relim()
    ax.autoscale_view()
    ax.set_xlim(right=ax.get_xlim()[1] * 1.10)
    xmax = ax.get_xlim()[1]

    # Group labels: outside the right axis edge (never collide with bars)
    for g_idx, (g_label, _) in enumerate(groups):
        base_y = -(g_idx * (n_pol * bar_h + grp_gap))
        center_y = base_y - (n_pol - 1) * bar_h / 2
        ax.annotate(g_label, xy=(1.015, center_y),
                    xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=7.5, color=GRY1)

    # Legend (policy colors) — only for policies that actually appear
    handles = []
    for p_idx, (pol_lbl, col, hatch) in enumerate(zip(pol_labels, pol_colors, pol_hatch)):
        # include only if any group has data for this policy
        any_data = any(not np.isnan(grp_vals[p_idx]) for _, grp_vals in groups)
        if any_data:
            handles.append(mpatches.Patch(
                facecolor=col, hatch=hatch,
                edgecolor="white" if hatch else "none",
                alpha=0.88, label=pol_lbl))
    ax.legend(handles=handles, fontsize=FONT_LEGEND, loc="lower left",
              bbox_to_anchor=(0.0, 1.01), frameon=False, ncol=4)

    fig.tight_layout()
    out = FIG_DIR / "fig_regret_bars.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  ({out.stat().st_size} bytes)")
    return out


# ===========================================================================
# FIGURE 4 — Bound validity vs width scatter (AD* vs CERT)
# ===========================================================================

def make_fig4_bound_validity():
    """Scatter: validity (y) vs median width (x, log scale) for CERT and
    AD* w-variants, on two panels: synthetic rho=0.02 and METR-LA.
    Data from results/extern_baselines/table.json (part_a)."""

    extern_path = REPO / "results" / "extern_baselines" / "table.json"
    with open(extern_path) as f:
        extern = json.load(f)

    part_a = extern["part_a"]

    # Separate by world
    synth  = [d for d in part_a if d["world"] == "synthetic"]
    metrla = [d for d in part_a if d["world"] == "metr-la"]

    def plot_panel(ax, rows, title):
        # Assign style by bound
        styles = {
            "CERT":     dict(marker="*", color=BLUE, s=90, zorder=5,
                             label="CERT"),
            "AD* w=1.2": dict(marker="o", color=RED,  s=45, zorder=4,
                              label="AD* $w=1.2$"),
            "AD* w=1.5": dict(marker="s", color=ORG,  s=45, zorder=4,
                              label="AD* $w=1.5$"),
            "AD* w=2.0": dict(marker="^", color=GRY1, s=45, zorder=4,
                              label="AD* $w=2.0$"),
        }
        for d in rows:
            bnd   = d["bound"]
            w     = d["width_median"]
            v     = d["validity"]
            n     = d["n"]
            sty   = styles.get(bnd, dict(marker="x", color=BLK, s=30))
            ax.scatter([w], [v], zorder=sty.get("zorder", 3),
                       marker=sty["marker"], color=sty["color"],
                       s=sty["s"], label=sty["label"])

        ax.axhline(1.0, color=GRY3, linewidth=0.7, linestyle="--")
        ax.set_xscale("log")
        ax.set_ylabel("Interval validity", fontsize=FONT_LABEL)
        ax.set_xlabel("Median interval width (log scale)", fontsize=FONT_LABEL)
        ax.set_ylim(-0.05, 1.12)
        ax.set_title(title, fontsize=FONT_LABEL, pad=3)

        # n annotation
        n_vals = {d["bound"]: d["n"] for d in rows}
        n_unique = list(set(n_vals.values()))
        n_str = ", ".join(str(x) for x in sorted(n_unique))
        ax.text(0.98, 0.03, f"n={n_str} intervals", ha="right",
                transform=ax.transAxes, fontsize=6.5, color=GRY1)

        handles = [
            Line2D([0], [0], marker=styles[k]["marker"], color="w",
                   markerfacecolor=styles[k]["color"],
                   markersize=6, label=k)
            for k in styles if any(d["bound"] == k for d in rows)
        ]
        ax.legend(handles=handles, fontsize=FONT_LEGEND, frameon=False,
                  loc="upper left")

    fig, (ax_s, ax_m) = plt.subplots(
        1, 2, figsize=(DOUBLE_W * 0.72, 2.8),
        layout="constrained",
    )

    plot_panel(ax_s, synth,
               "Synthetic ($\\rho=0.02$)")
    plot_panel(ax_m, metrla,
               "METR-LA (real traffic)")

    # Panel labels
    for ax, lbl in [(ax_s, "A"), (ax_m, "B")]:
        ax.text(-0.14, 1.06, lbl, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

    out = FIG_DIR / "fig_bound_validity.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  ({out.stat().st_size} bytes)")
    return out


# ===========================================================================
# LaTeX figure fragment
# ===========================================================================

FIGURES_TEX = r"""\
% figures.tex — auto-generated by scripts/make_figures.py
% Insert \input{sections/figures} where desired in main.tex.
% DO NOT EDIT BY HAND — regenerate with: cert_env/bin/python scripts/make_figures.py

% -----------------------------------------------------------------------
\begin{figure}[tbp]
  \centering
  \includegraphics{figures/fig_gap_trajectory}
  \caption{Certificate gap trajectory for one representative episode
    (6$\times$6 grid, bounded drift $\rho{=}0.02$, \texttt{seed=3},
    default \textsc{cert} planner).
    \emph{Top}: true optimal cost OPT (black) overlaid on the $[\mathrm{LB},\mathrm{UB}]$
    certificate band (blue fill).  \emph{Bottom}: certificate gap $\mathrm{UB}{-}\mathrm{LB}$
    (red) and the target $\varepsilon{=}5$ (dotted).
    Shading: grey = warm-up (annealing claim below target),
    mid-grey = valid-uncertified, blue = certified ($\mathrm{gap}{\le}\varepsilon$).
    Single episode shown; aggregate over 25 seeds $\times$ 300 rounds is reported in
    Table~\ref{tab:tier0}.}
  \label{fig:gap_trajectory}
\end{figure}

% -----------------------------------------------------------------------
\begin{figure}[tbp]
  \centering
  \includegraphics{figures/fig_coverage_vs_claim}
  \caption{Empirical coverage (filled circles, 95\% Clopper--Pearson bars) vs
    claimed confidence level (red tick marks) for eight synthetic conditions
    (white background) and four real-traffic conditions
    (grey background; METR-LA 20 days, PEMS-BAY 20 days).
    Synthetic: 25 seeds $\times$ 300 rounds, $6{\times}6$ grid, $\alpha'{=}0.2$.
    Coverage equals or exceeds the claim in every row; no row falls below
    the claimed level.}
  \label{fig:coverage_vs_claim}
\end{figure}

% -----------------------------------------------------------------------
\begin{figure}[tbp]
  \centering
  \includegraphics{figures/fig_interval_mechanism}
  \caption{The certificate's building block on one edge (bounded drift
    $\rho{=}0.02$, seed 7): the interval $\hat c \pm (\lambda q + \rho a)$
    (blue band) widens linearly with age $a$ and snaps tight when the edge
    is re-observed (red markers); the true cost (black) wanders within.
    Staleness is priced, not ignored.}
  \label{fig:interval_mechanism}
\end{figure}

% -----------------------------------------------------------------------
\begin{figure*}[tbp]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_cia_collapse}
  \caption{Exchangeable conformal path sums (CIA~\citep{luo2024conformalized},
    their construction extracted and run on METR-LA; 50 paths $\times$ 20
    repetitions per gap, target 0.90) vs CERT as the calibration-to-test gap
    grows. \emph{Left}: CIA covers at gap 0 (its home setting) and collapses
    to 0.20--0.25 at the 3--6\,h staleness common in operation; the partial
    24\,h recovery is the diurnal cycle returning the network near its
    calibration state --- the failure mode is drift, not noise. CERT holds
    0.95--1.00 at every gap. \emph{Right}: the price, paid explicitly ---
    CIA's width is frozen ($\sim$50\,s, no time-dependent term) while CERT's
    $\rho\cdot$gap widening grows. Error bars: 95\% Clopper--Pearson.}
  \label{fig:cia_collapse}
\end{figure*}

% -----------------------------------------------------------------------
\begin{figure}[tbp]
  \centering
  \includegraphics[width=0.92\textwidth]{figures/fig_regret_bars}
  \caption{Travel regret (mean, lower is better) by sensing policy across three
    map groups.
    \emph{Synthetic}: 10$\times$10 bounded drift $\rho{=}0.02$, budget $B{=}20$,
    25 seeds; includes hybrid (objective-matched, cert+VOI) and VOI (CTP-RS
    expected-route) policies from the external-baseline run (15 seeds).
    \emph{DAO dungeon / Berlin street}: MovingAI maps, 15 seeds,
    bounded drift $\rho{=}0.02$, $B{=}20$.
    Blank bars indicate conditions not run for that dataset.
    Regret is against a clairvoyant oracle replanning on true costs every step.}
  \label{fig:regret_bars}
\end{figure}

% -----------------------------------------------------------------------
\begin{figure}[tbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/fig_bound_validity}
  \caption{Interval validity (fraction of rounds the true OPT lies inside the
    reported interval; $y$-axis) vs median interval width ($x$-axis, log scale)
    for \textsc{cert} and three AD$^*$-style inflation widths ($w\in\{1.2,1.5,2.0\}$),
    evaluated on a shared stale observation stream (neutral max-age sensing).
    \textbf{A}: synthetic bounded drift $\rho{=}0.02$ (15 seeds; $n=4340$ intervals per bound).
    \textbf{B}: METR-LA replayed traffic (15 seeds; $n=4273$).
    Narrow-and-wrong vs wide-and-sound is the observed trade-off:
    AD$^*$ semantics hedge search suboptimality, not map staleness.}
  \label{fig:bound_validity}
\end{figure}
"""




# ===========================================================================
# FIGURE 5 — CIA exchangeability collapse vs CERT under staleness
# ===========================================================================

def make_fig5_cia_collapse():
    import json
    d = json.load(open(REPO / "results" / "cia_comparison" / "table.json"))
    labels = d["gap_labels"]
    x = range(len(labels))
    fig, (ax_cov, ax_w) = plt.subplots(
        1, 2, figsize=(DOUBLE_W, 2.3), constrained_layout=True)

    for key, color, name, marker in (
            ("cert", BLUE, "CERT", "o"), ("cia", RED, "CIA", "s")):
        rows = d[key]
        cov = [r["coverage"] for r in rows]
        lo = [r["coverage"] - r["ci_lo"] for r in rows]
        hi = [r["ci_hi"] - r["coverage"] for r in rows]
        ax_cov.errorbar(x, cov, yerr=[lo, hi], color=color, marker=marker,
                        markersize=4, linewidth=1.2, capsize=2, label=name)
        ax_w.plot(x, [r["median_width"] for r in rows], color=color,
                  marker=marker, markersize=4, linewidth=1.2, label=name)

    ax_cov.axhline(d["target_coverage"], color=GRY1, linewidth=0.8,
                   linestyle=":", zorder=0)
    ax_cov.annotate("target 0.90", xy=(0.13, 0.815),
                    fontsize=FONT_ANNOT, color=GRY1)
    ax_cov.set_ylim(0, 1.05)
    ax_cov.set_ylabel("coverage of true path sums")
    ax_cov.set_xlabel("calibration-to-test gap")
    ax_cov.set_xticks(list(x), labels)
    ax_cov.annotate("diurnal\nre-coverage", xy=(4.85, 0.50), xytext=(3.85, 0.04),
                    fontsize=FONT_ANNOT, color=RED,
                    arrowprops=dict(arrowstyle="-", color=RED, linewidth=0.6))
    ax_cov.legend(frameon=False, loc="center left")

    ax_w.set_yscale("log")
    ax_w.set_ylabel("median interval width (s)")
    ax_w.set_xlabel("calibration-to-test gap")
    ax_w.set_xticks(list(x), labels)
    ax_w.annotate("width frozen:\nno time term", xy=(3, 54), xytext=(2.2, 180),
                  fontsize=FONT_ANNOT, color=RED,
                  arrowprops=dict(arrowstyle="-", color=RED, linewidth=0.6))
    ax_w.annotate("pays width,\nkeeps coverage", xy=(3, 4266),
                  xytext=(0.3, 3000), fontsize=FONT_ANNOT, color=BLUE,
                  arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.6))
    out = FIG_DIR / "fig_cia_collapse.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


# ===========================================================================
# FIGURE 6 — The interval mechanism on one edge (staleness widens, sensing
# resets, drift-adjusted residuals stay calibrated)
# ===========================================================================

def make_fig6_interval_mechanism():
    from certflow.cert import CertPlanner, PlannerConfig
    from certflow.drift import grid_world

    world = grid_world(6, 6, seed=7, kind="bounded", rho=0.02,
                       noise_scale=0.05)
    p = CertPlanner(world, (0, 0), (5, 5),
                    PlannerConfig(epsilon=5.0, alpha_prime=0.2, eps_tv=1e-4))
    e = ((2, 2), (2, 3))
    ts, lows, ups, truth = [], [], [], []
    obs_t, obs_v = [], []
    for i in range(150):
        _, sensed = p.round()
        b = p.beliefs[e]
        q = p.scorer.quantile(p._last_alpha_edge, p.t)
        if not (q == q and q != float("inf")):
            continue
        half = p.cfg.latent_margin * q + b.rho * b.age(p.t)
        ts.append(p.t)
        lows.append(max(p.cfg.cost_floor, b.c_hat - half))
        ups.append(b.c_hat + half)
        truth.append(world.true_cost(e, p.t))
        if sensed == e:
            obs_t.append(p.t)
            obs_v.append(b.c_hat)

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.1), constrained_layout=True)
    ax.fill_between(ts, lows, ups, color=SKY, alpha=0.35, linewidth=0,
                    label=r"interval $\hat c \pm (\lambda q + \rho a)$")
    ax.plot(ts, truth, color=BLK, linewidth=1.1, label="true cost $c_e(t)$")
    if obs_t:
        ax.plot(obs_t, obs_v, linestyle="none", marker="v", markersize=5,
                color=RED, label="observation (resets age)")
    ax.set_xlabel("time $t$ (rounds)")
    ax.set_ylabel("edge cost")
    ax.legend(frameon=False, fontsize=FONT_LEGEND, loc="lower right")
    out = FIG_DIR / "fig_interval_mechanism.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def write_figures_tex():
    out = REPO / "paper" / "sections" / "figures.tex"
    out.write_text(FIGURES_TEX)
    print(f"  wrote {out}  ({out.stat().st_size} bytes)")
    return out


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=== make_figures.py ===")
    print(f"Output directory: {FIG_DIR}")

    print("\n[1/4] fig_gap_trajectory.pdf ...")
    f1 = make_fig1_gap_trajectory()

    print("\n[2/4] fig_coverage_vs_claim.pdf ...")
    f2 = make_fig2_coverage_vs_claim()

    print("\n[3/4] fig_regret_bars.pdf ...")
    f3 = make_fig3_regret_bars()

    print("\n[4/4] fig_bound_validity.pdf ...")
    f4 = make_fig4_bound_validity()

    print("\n[+] fig 5: CIA exchangeability collapse ...")
    print("    ->", make_fig5_cia_collapse())
    print("\n[+] fig 6: interval mechanism ...")
    print("    ->", make_fig6_interval_mechanism())

    print("\n[+] paper/sections/figures.tex ...")
    ft = write_figures_tex()

    print("\nDone.  Files:")
    for p in [f1, f2, f3, f4, ft]:
        print(f"  {p}  {p.stat().st_size} B")


if __name__ == "__main__":
    main()
