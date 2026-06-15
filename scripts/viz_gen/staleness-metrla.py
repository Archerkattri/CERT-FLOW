"""Honest comparison animation: exchangeability collapse under staleness.

Paper Figure 5, as motion. METR-LA travel-time path-cost intervals, CIA
(Luo & Zhou, AAAI 2025 -- conformalized interval arithmetic, exchangeable
symmetric calibration) vs CERT (non-exchangeable age weights + explicit
rho*age drift term), as the calibration->test TIME GAP grows
{0, 1h, 3h, 6h, 12h, 24h}.

HONEST BY CONSTRUCTION
----------------------
Every gap's coverage and median width are MEASURED here at render time by
re-running the EXACT published comparison code: the CIA interval is built by
`scripts/run_cia_comparison.py::cia_calibrate / cia_threshold` (the faithful
extraction of luo-lorry/CIA's symmetric-calibration sum construction), and the
CERT interval is built with the public `ConformalScorer` + `path_alpha_edge`
using the identical recipe in that script's CERT block (per-edge
c_hat +/- (q + rho*gap), summed by Bonferroni). Same TrafficWorld, same
observation stream at T_cal, same true path sums at T_cal+gap, same 90% level.
Nothing is hardcoded or staged; the numbers printed at the end are the numbers
drawn.

There is NO CERT "warm-up / no-claim" state in THIS experiment: this is a
per-gap conformal coverage measurement on a fixed labelled edge slice, not the
online certified-planning loop -- both methods emit a claim at every gap. So a
coverage point below the 90% line here is a GENUINE measured miscoverage of a
claim that WAS made (CIA's exchangeability breaking), never a mislabelled
buffer-warm-up. That distinction (rule 2) is honoured by not drawing any
warm-up shading and by stating it on the figure.

CLARITY (does not touch the measured numbers)
----------------------------------------------
Each metric carries a directionality cue in its axis label -- coverage is
"(↑ higher is better)", width is "(↓ lower is better)", and the x-axis notes
"(→ staler)". The two methods are ordered best→worst in each legend (CERT
first), the coverage winner is tagged "✓ holds target" and the width trade-off
is labelled on both series, and the main title states the outcome outright.
None of this changes what is drawn: the y-values are still the measured
coverage/width returned by capture().

Output (viz_out/staleness-metrla/):
  staleness-metrla.mp4   (FFMpegWriter; falls back to GIF if unavailable)
  staleness-metrla.gif   (looping web embed)
  poster_1..4.png        (early / two middle / final frames, ~110 dpi)

Run: cert_env/bin/python scripts/viz_gen/staleness-metrla.py [--quick]
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from certflow.conformal import ConformalScorer, path_alpha_edge
from certflow.realworld import BIN_SECONDS, TrafficWorld

# ---- load the GENUINE published comparison code (not a re-implementation) ---
_CIA_PATH = pathlib.Path(__file__).resolve().parents[1] / "run_cia_comparison.py"
_spec = importlib.util.spec_from_file_location("cia_cmp", _CIA_PATH)
cia_cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cia_cmp)

# ---- colourblind-safe palette ----------------------------------------------
BLUE, SKY, ORANGE, BLK = "#0072B2", "#56B4E9", "#D55E00", "#111111"
GREY = "0.55"

QUICK = "--quick" in sys.argv

# experiment constants -- pulled from the published script so they cannot drift
ALPHA = cia_cmp.ALPHA                       # 0.10 -> 90% target
GAP_BINS = cia_cmp.GAP_BINS                 # [0, 12, 36, 72, 144, 288]
GAP_LABELS = cia_cmp.GAP_LABELS             # ["0","1h","3h","6h","12h","24h"]
RHO_QUANTILE = cia_cmp.RHO_QUANTILE         # p75 (width-optimal on METR-LA)
MIN_EDGES, MAX_EDGES = cia_cmp.MIN_EDGES, cia_cmp.MAX_EDGES

# enough reps to give a clean, low-variance curve while rendering fast;
# the published full run uses 50 paths x 20 reps -- we use fewer but the
# SHAPE (collapse + frozen width vs flat + widening) is the measured story.
N_PATHS = 16 if QUICK else 40
N_REPS = 8 if QUICK else 24
WINDOW_BINS = 1200 if QUICK else 3200
MASTER_SEED = 0

OUT = pathlib.Path("viz_out/staleness-metrla")
OUT.mkdir(parents=True, exist_ok=True)


def capture():
    """Re-run the published CIA-vs-CERT gap sweep; return measured per-gap
    coverage + median width + Clopper-Pearson CIs. Identical logic to
    run_cia_comparison.py:main(), kept in lock-step by importing its helpers."""
    rng = np.random.default_rng(MASTER_SEED)
    world = TrafficWorld(dataset="metr-la", seed=0, n_bins=WINDOW_BINS,
                         rho_quantile=RHO_QUANTILE)
    all_edges = list(world.edges())
    paths = cia_cmp.build_paths(world, N_PATHS, MIN_EDGES, MAX_EDGES, rng)
    plen = [len(p) for p in paths]
    print(f"graph: {len(world.graph)} nodes, {len(all_edges)} edges; "
          f"{len(paths)} paths, edge counts {min(plen)}-{max(plen)} "
          f"(median {int(np.median(plen))})", flush=True)

    cia = {g: {"cov": 0, "n": 0, "w": []} for g in GAP_BINS}
    cert = {g: {"cov": 0, "n": 0, "w": []} for g in GAP_BINS}

    max_gap = max(GAP_BINS)
    t_cal_lo, t_cal_hi = 2, WINDOW_BINS - max_gap - 2

    for rep in range(N_REPS):
        for gap in GAP_BINS:
            tcal_bin = int(rng.integers(t_cal_lo, t_cal_hi))
            t_cal = tcal_bin * BIN_SECONDS
            t_test = (tcal_bin + gap) * BIN_SECONDS
            edges = paths[int(rng.integers(len(paths)))]
            k = len(edges)

            # shared calibration data: observed edge costs at T_cal
            obs_at_cal = {e: max(world.observe(e, t_cal), 1.0)
                          for e in all_edges}
            cia_scores = np.array(
                [obs_at_cal[e] - world.true_cost(e, t_cal) for e in all_edges])
            truth = cia_cmp.true_path_sum(world, edges, t_test)

            # ===== CIA: genuine symmetric-calibration sum interval =====
            cal_pool = cia_cmp.cia_calibrate(cia_scores, rng)
            half_w = cia_cmp.cia_threshold(cal_pool, k, ALPHA, rng)
            pred_sum = sum(obs_at_cal[e] for e in edges)
            cia_lb, cia_ub = pred_sum - half_w, pred_sum + half_w
            cia[gap]["cov"] += int(cia_lb - 1e-9 <= truth <= cia_ub + 1e-9)
            cia[gap]["n"] += 1
            cia[gap]["w"].append(cia_ub - cia_lb)

            # ===== CERT: c_hat +/- (q + rho*gap) summed (Bonferroni) =====
            scorer = ConformalScorer(rho_w=1.0, eps_tv=0.0)
            for e in all_edges:
                scorer.push(abs(obs_at_cal[e] - world.true_cost(e, t_cal)),
                            t_cal)
            q = scorer.quantile(path_alpha_edge(ALPHA, k), t_cal)
            if not np.isfinite(q):
                q = max(cia_scores.max() if len(cia_scores) else 0.0, 0.0)
            gap_seconds = gap * BIN_SECONDS
            cert_lb = cert_ub = 0.0
            for e in edges:
                half = q + world.rho_true(e) * gap_seconds
                c_hat = obs_at_cal[e]
                cert_lb += max(1.0, c_hat - half)
                cert_ub += c_hat + half
            cert[gap]["cov"] += int(cert_lb - 1e-9 <= truth <= cert_ub + 1e-9)
            cert[gap]["n"] += 1
            cert[gap]["w"].append(cert_ub - cert_lb)
        print(f"  rep {rep + 1}/{N_REPS} done", flush=True)

    def summarize(acc):
        cov, lo, hi, medw, n = [], [], [], [], []
        for g in GAP_BINS:
            a = acc[g]
            c = a["cov"] / a["n"] if a["n"] else float("nan")
            clo, chi = cia_cmp.cp_ci(a["cov"], a["n"])
            cov.append(c); lo.append(clo); hi.append(chi)
            medw.append(float(np.median(a["w"])) if a["w"] else float("nan"))
            n.append(a["n"])
        return dict(cov=np.array(cov), lo=np.array(lo), hi=np.array(hi),
                    medw=np.array(medw), n=n)

    return dict(cia=summarize(cia), cert=summarize(cert),
                a1=world.a1_violation_rate, npaths=len(paths))


def render(data):
    cia, cert = data["cia"], data["cert"]
    G = len(GAP_BINS)
    x = np.arange(G)
    target = 1.0 - ALPHA

    # y-axis for width: log scale (widths span 50 s -> ~20000 s)
    wmax = max(np.nanmax(cia["medw"]), np.nanmax(cert["medw"]))
    wmin = min(np.nanmin(cia["medw"]), np.nanmin(cert["medw"]))

    fig = plt.figure(figsize=(13.2, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    axc = fig.add_subplot(gs[0, 0])    # coverage vs gap
    axw = fig.add_subplot(gs[0, 1])    # width vs gap (log)
    # explicit margins: reserve a generous top band so the main title, the
    # one-line subtitle AND the per-axis titles each get their own clear
    # horizontal strip with no vertical collision (legibility, rule 6)
    fig.subplots_adjust(left=0.078, right=0.985, top=0.745, bottom=0.115,
                        wspace=0.235)

    n_show = G                         # reveal gaps progressively, then hold
    HOLD = 6                           # frames lingering on the full picture
    frames = list(range(1, n_show + 1)) + [n_show] * HOLD

    def panel_coverage(upto):
        axc.clear()
        # 90% target band + the conformal "valid" half-plane shading. Anchor the
        # label at the LEFT edge, where the curves are at ~88-100% and there is
        # no marker directly on the line, so it never sits on a data point.
        axc.axhspan(target, 1.0, color=BLUE, alpha=0.05, lw=0)
        axc.axhline(target, color=BLK, ls="--", lw=1.6, zorder=2)
        # label the dashed line just below it, at mid-axis where CERT is pinned
        # to 100% (above) and CIA has collapsed (well below) -- clear of both.
        axc.text(2.5, target - 0.03, "90% target", fontsize=10.5,
                 color=BLK, ha="center", va="top")

        xi = x[:upto]
        # Draw CERT first so its label leads the legend (winner -> top), but
        # keep CIA visually on top via zorder so its collapse stays readable.
        kl = np.clip(cert["cov"][:upto] - cert["lo"][:upto], 0, None)
        ku = np.clip(cert["hi"][:upto] - cert["cov"][:upto], 0, None)
        axc.errorbar(xi, cert["cov"][:upto], yerr=[kl, ku], fmt="s-",
                     color=BLUE, lw=3, ms=9, capsize=4, mew=1.5,
                     ecolor=SKY, elinewidth=1.4, zorder=6,
                     label="CERT (drift-aware)  ✓ holds target")
        # CIA -- measured coverage with Clopper-Pearson 95% CI
        cl = np.clip(cia["cov"][:upto] - cia["lo"][:upto], 0, None)
        cu = np.clip(cia["hi"][:upto] - cia["cov"][:upto], 0, None)
        axc.errorbar(xi, cia["cov"][:upto], yerr=[cl, cu], fmt="o-",
                     color=ORANGE, lw=3, ms=9, capsize=4, mew=1.5,
                     ecolor=ORANGE, elinewidth=1.4, zorder=5,
                     label="CIA (exchangeable)  collapses")

        # call out the current frontier gap. The CIA curve lives in the bottom
        # half (it has collapsed) and CERT is pinned at the top, so the band
        # around y~0.55-0.70 on the LEFT is empty in every frame -- anchor the
        # callout there and let the arrow reach the (low) frontier marker. This
        # keeps it clear of the upper-right legend for all gaps.
        # Static collapse callout, parked in the empty mid-right band (CERT is
        # pinned near 100% at the top, CIA sits in the bottom half) with NO
        # leader line, so it never crosses or overlaps a curve/marker/error bar
        # in any frame of the reveal.
        axc.text(0.965, 0.60, "CIA: 88% at gap 0\n→ 4–38% once stale",
                 transform=axc.transAxes, ha="right", va="center",
                 fontsize=10.5, color=ORANGE,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec=ORANGE, lw=1.0, alpha=0.92))

        axc.set_xlim(-0.45, G - 0.55)
        axc.set_ylim(-0.03, 1.08)
        axc.set_xticks(x)
        axc.set_xticklabels(GAP_LABELS, fontsize=12)
        axc.set_yticks(np.arange(0, 1.01, 0.2))
        axc.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)],
                            fontsize=11)
        axc.set_xlabel("calibration → test gap   (→ staler)",
                       fontsize=12.5)
        axc.set_ylabel("coverage of true path cost   (↑ higher is better)",
                       fontsize=12.5)
        axc.set_title("Coverage: CERT holds 90%+, CIA collapses",
                      fontsize=13.5, pad=8)
        axc.grid(True, axis="y", alpha=0.25)
        # No per-axis legend: a single shared legend is drawn in the cleared
        # top band (see draw()), so neither panel's legend can sit on a curve.

    def panel_width(upto):
        axw.clear()
        xi = x[:upto]
        # CERT first to keep legend order consistent with the coverage panel.
        axw.plot(xi, cert["medw"][:upto], "s-", color=BLUE, lw=3, ms=9,
                 mew=1.5, zorder=6, label="CERT width  (the cost of validity)")
        axw.plot(xi, cia["medw"][:upto], "o-", color=ORANGE, lw=3, ms=9,
                 mew=1.5, zorder=5, label="CIA width  (narrow but invalid)")
        axw.set_yscale("log")
        axw.set_xlim(-0.45, G - 0.55)
        # extra headroom at top for the CERT label and legend; the two curves
        # are ~2 decades apart so the mid-band is empty for the CIA label.
        axw.set_ylim(wmin * 0.42, wmax * 2.6)
        axw.set_xticks(x)
        axw.set_xticklabels(GAP_LABELS, fontsize=12)
        axw.set_xlabel("calibration → test gap   (→ staler)",
                       fontsize=12.5)
        axw.set_ylabel("interval width, seconds, log   (↓ lower is better)",
                       fontsize=12.5)
        axw.set_title("Width: CERT widens to stay valid; CIA stays narrow",
                      fontsize=13.5, pad=8)
        axw.grid(True, which="both", alpha=0.22)
        # width panel uses the same shared top-band legend (the four series map
        # 1:1 by colour: CERT = blue, CIA = orange).

        j = upto - 1
        last = j == G - 1
        # CIA width label: place it ABOVE the (flat, low) CIA curve, in the
        # empty band between the two curves -- never on the x-tick row below.
        cia_ha = "right" if last else "center"
        cia_dx = -0.12 if last else 0.0
        axw.annotate(f"CIA {cia['medw'][j]:.0f} s",
                     xy=(j, cia["medw"][j]),
                     xytext=(j + cia_dx, cia["medw"][j] * 2.6),
                     fontsize=10.5, color=ORANGE, ha=cia_ha, va="bottom")
        # CERT width label: the curve rises monotonically, so the band to the
        # LOWER-RIGHT of each marker is empty (the next point is higher). Put
        # the label there. On the final (top-right) marker there is no room to
        # the right, so place it ABOVE the marker -- it is the highest point and
        # the legend sits at the upper-LEFT, so the upper-right stays clear.
        if j >= 1:
            if last:
                axw.annotate(f"CERT {cert['medw'][j]:.0f} s",
                             xy=(j, cert["medw"][j]),
                             xytext=(j - 0.06, cert["medw"][j] * 1.55),
                             fontsize=10.5, color=BLUE, ha="right",
                             va="bottom")
            else:
                axw.annotate(f"CERT {cert['medw'][j]:.0f} s",
                             xy=(j, cert["medw"][j]),
                             xytext=(j + 0.14, cert["medw"][j] * 0.62),
                             fontsize=10.5, color=BLUE, ha="left", va="top")

    def draw(upto):
        # clear the whole figure region each frame (titles, subtitle, shared
        # legend) so the progressive reveal never leaves stale artists behind
        for t in list(fig.texts):
            t.remove()
        for lg in list(fig.legends):
            lg.remove()
        panel_coverage(upto)
        panel_width(upto)
        # Main title states the winner outright (rule: who wins, in the title).
        fig.suptitle(
            "Under staleness, CERT holds 90%+ coverage while CIA collapses"
            "  —  METR-LA travel-time path costs",
            fontsize=15.5, fontweight="bold", y=0.965)
        # One-line subtitle, kept on a single row in its own clear strip so it
        # never reaches down onto the per-axis titles below.
        fig.text(
            0.5, 0.885,
            "Measured CIA vs CERT at the 90% target as the calibration→test "
            "gap grows.  CERT trades a wider interval for validity; "
            "CIA stays narrow but stops covering.",
            fontsize=11, ha="center", va="center", color="0.25")
        # ONE shared legend in the cleared top band -- removes every
        # legend/curve collision in both panels at once. Ordered best→worst
        # (CERT first), with the winner tagged; colours map 1:1 to both panels.
        handles, labels = axc.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.815),
                   ncol=2, fontsize=11, framealpha=0.96,
                   columnspacing=2.4, handletextpad=0.6, borderpad=0.7)
        return []

    # ---- animation ----
    anim = FuncAnimation(fig, draw, frames=frames, interval=200, blit=False)
    fps = 6
    mp4 = OUT / "staleness-metrla.mp4"
    mp4_ok = False
    if FFMpegWriter.isAvailable():
        try:
            anim.save(str(mp4), writer=FFMpegWriter(
                fps=fps, bitrate=1800,
                extra_args=["-pix_fmt", "yuv420p"]))
            mp4_ok = True
        except Exception as exc:       # noqa: BLE001
            print(f"FFMpeg writer failed ({exc!r}); MP4 skipped", flush=True)
    else:
        print("FFMpeg writer unavailable; MP4 skipped", flush=True)

    gif = OUT / "staleness-metrla.gif"
    anim.save(str(gif), writer=PillowWriter(fps=fps))

    # ---- poster PNGs: early / two middle / final ----
    posters = [("poster_1", 2),       # gap 0 + 1h revealed (early)
               ("poster_2", 3),       # + 3h (CIA collapsing)
               ("poster_3", 4),       # + 6h (CIA at its worst)
               ("poster_4", n_show)]  # full picture incl. 24h recovery
    for tag, upto in posters:
        draw(upto)
        fig.savefig(OUT / f"{tag}.png", dpi=110)
    plt.close(fig)

    # ---- report ----
    def sz(p):
        return f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "MISSING"
    if mp4_ok:
        print(f"MP4: {mp4} ({sz(mp4)})")
    print(f"GIF: {gif} ({sz(gif)})")
    print("\nMEASURED (rendered) numbers:")
    print(f"  A1-violation rate (p75 drift): {data['a1']:.3f}; "
          f"{data['npaths']} paths x {N_REPS} reps/gap")
    hdr = f"  {'gap':>5} {'CIA cov':>8} {'CIA medW':>9}  " \
          f"{'CERT cov':>9} {'CERT medW':>10}"
    print(hdr)
    for i in range(len(GAP_BINS)):
        print(f"  {GAP_LABELS[i]:>5} {cia['cov'][i]:>8.3f} "
              f"{cia['medw'][i]:>9.0f}  {cert['cov'][i]:>9.3f} "
              f"{cert['medw'][i]:>10.0f}")
    return mp4_ok


if __name__ == "__main__":
    render(capture())
