"""watch_alarm.gif -- the 2026 observability layer catching a broken model.

Runs the real planner with ``PlannerConfig(watch_monitor=True)`` on a
bounded-drift grid whose true costs SURGE mid-run (``DriftJumpWorld``, the
regime change exercised by tests/test_live_wiring.py). The staleness model the
certificate leans on is calibrated for the slow pre-jump drift, so the jump is a
clean violation. Animates, per round:

  * LEFT  -- the certificate band (LB..UB) with the true optimum inside it.
    Under the correct model it brackets truth; when the world jumps, the band is
    priced off stale (cheap) observations and the true optimum briefly ESCAPES
    it -- rounds drawn in vermillion. That silent staleness is exactly what the
    monitor exists to surface.
  * RIGHT -- the Shiryaev-Roberts statistic R_t (planner.sr.R) on a log axis.
    It crawls flat under the correct model (median ~1.4) and, within a handful of
    rounds of the jump, explodes past its alarm threshold (ARL 5000). The
    crossing is annotated. The monitor is purely observational -- it changes no
    certificate.

Reproduce:  PYTHONPATH=src python scripts/viz_gen/watch_alarm.py
Writes assets/animations/watch_alarm.gif  (seeded; needs no data/).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from certflow.cert import CertPlanner, PlannerConfig  # noqa: E402
from certflow.drift import grid_world  # noqa: E402
from certflow.oracle import opt  # noqa: E402

INK = "#0a111e"
MUTED = "#5b6b82"
GRIDC = "#e4e9f0"
SURF = "#ffffff"
CERT = "#0072B2"      # deep blue  -- certificate band
CERTLT = "#56B4E9"    # light blue -- band fill
OPTC = "#0a111e"      # ink        -- true optimum
QUIET = "#009E73"     # green      -- monitor quiet (model holds)
ALARM = "#D55E00"     # vermillion -- alarm / model violated

N = 6
ROUNDS = 240
SEED = 0
T_JUMP = 120.0
SURGE = 5.0
SR_THRESHOLD = 5000.0
STEP = 3
FPS = 12


class DriftJumpWorld:
    """Bounded-drift grid whose true costs multiply by ``surge`` after ``t_jump``
    -- the clean staleness-model break from tests/test_live_wiring.py."""

    def __init__(self, base, t_jump: float, surge: float) -> None:
        self.base = base
        self.graph = base.graph
        self.t_jump = t_jump
        self.surge = surge

    def edges(self):
        return self.base.edges()

    def rho_true(self, e):
        return self.base.rho_true(e)

    def true_cost(self, e, t):
        c = self.base.true_cost(e, t)
        return c * self.surge if t >= self.t_jump else c

    def observe(self, e, t):
        return self.true_cost(e, t) + self.base._draw_noise()


def run():
    base = grid_world(N, N, seed=SEED, kind="bounded", noise_scale=0.05)
    world = DriftJumpWorld(base, t_jump=T_JUMP, surge=SURGE)
    start, goal = (0, 0), (N - 1, N - 1)
    p = CertPlanner(world, start, goal,
                    PlannerConfig(epsilon=3.0, alpha_prime=0.2, delta=1.0,
                                  watch_monitor=True, sr_threshold=SR_THRESHOLD))
    recs = []
    for i in range(ROUNDS):
        cert, _ = p.round()
        t_cert = p.t - p.cfg.delta
        _, o = opt(world, t_cert, start, goal)
        covered = (cert.valid and math.isfinite(cert.lb)
                   and cert.lb - 1e-6 <= o <= cert.ub + 1e-6)
        recs.append(dict(i=i, lb=cert.lb, ub=cert.ub, opt=o, valid=bool(cert.valid),
                         R=float(p.sr.R), covered=covered))
    return recs, p.sr.alarm_round


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 1.0, "text.color": INK, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })

    recs, alarm_round = run()
    jump = int(T_JUMP)
    delay = None if alarm_round is None else alarm_round - jump
    nvalid = sum(r["valid"] for r in recs)
    ncov = sum(r["covered"] for r in recs)
    nviol = nvalid - ncov

    lb = np.array([r["lb"] for r in recs], float)
    ub = np.array([r["ub"] for r in recs], float)
    op = np.array([r["opt"] for r in recs], float)
    val = np.array([r["valid"] for r in recs])
    cov = np.array([r["covered"] for r in recs])
    R = np.clip(np.array([r["R"] for r in recs], float), 0.3, None)

    finite_ub = ub[val & np.isfinite(ub)]
    yhi = max(finite_ub.max(), op.max()) * 1.10
    r_hi = max(R.max(), SR_THRESHOLD) * 6

    frames = list(range(0, ROUNDS, STEP))
    fig, (axB, axR) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    fig.patch.set_facecolor(SURF)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.14, wspace=0.26)

    t_head = fig.text(0.075, 0.955, "", fontsize=11.5, fontweight="bold",
                      color=INK, ha="left", va="top")
    t_phase = fig.text(0.075, 0.905, "", fontsize=10.2, ha="left", va="top")
    t_stat = fig.text(0.975, 0.955, "", fontsize=9.6, color=MUTED, ha="right",
                      va="top")

    def draw(fi):
        i = frames[fi]
        axB.clear(); axR.clear()
        xs = np.arange(i + 1)

        # ---------------- LEFT: certificate band ----------------
        axB.set_facecolor(SURF)
        lbc = np.clip(np.where(np.isfinite(lb[:i + 1]), lb[:i + 1], 0.0), 0.0, yhi)
        ubc = np.clip(np.where(np.isfinite(ub[:i + 1]), ub[:i + 1], yhi), 0.0, yhi)
        vv = val[:i + 1]
        axB.fill_between(xs, lbc, ubc, where=vv, color=CERTLT, alpha=0.30,
                         zorder=1, linewidth=0)
        axB.plot(xs, np.where(vv, ubc, np.nan), color=CERT, lw=1.6, zorder=3)
        axB.plot(xs, np.where(vv, lbc, np.nan), color=CERT, lw=1.6, ls=(0, (5, 3)),
                 zorder=3)
        axB.plot(xs, op[:i + 1], color=OPTC, lw=1.9, zorder=4, label="true OPT")
        # rounds where the true optimum escaped the band (staleness violation)
        vmask = vv & ~cov[:i + 1]
        if vmask.any():
            axB.scatter(xs[vmask], op[:i + 1][vmask], s=30, color=ALARM,
                        zorder=6, label="OPT escaped band")
        if i >= jump:
            axB.axvline(jump, color=ALARM, lw=1.3, ls=(0, (4, 3)), zorder=2)
            axB.text(jump - 4, yhi * 0.96, "drift jump", rotation=90, ha="right",
                     va="top", color=ALARM, fontsize=8.6)
        axB.set_xlim(0, ROUNDS); axB.set_ylim(0, yhi)
        axB.set_xlabel("planning round", fontsize=10)
        axB.set_ylabel("route cost", fontsize=10)
        for s in ("top", "right"):
            axB.spines[s].set_visible(False)
        axB.grid(axis="y", color=GRIDC, lw=0.8, zorder=0); axB.set_axisbelow(True)
        axB.tick_params(length=0)
        axB.set_title("certificate band   LB ≤ OPT ≤ UB", fontsize=11.5,
                      fontweight="bold", color=INK, loc="left", pad=8)
        if axB.get_legend_handles_labels()[0]:
            axB.legend(loc="upper left", frameon=False, fontsize=8.6,
                       handlelength=1.4)

        # ---------------- RIGHT: Shiryaev-Roberts monitor ----------------
        axR.set_facecolor(SURF)
        Rc = R[:i + 1]
        if alarm_round is not None and i >= alarm_round:
            axR.plot(xs[:alarm_round + 1], Rc[:alarm_round + 1], color=QUIET,
                     lw=1.9, zorder=4)
            axR.plot(xs[alarm_round:], Rc[alarm_round:], color=ALARM, lw=2.1,
                     zorder=5)
        else:
            axR.plot(xs, Rc, color=QUIET, lw=1.9, zorder=4)
        axR.axhline(SR_THRESHOLD, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=2)
        axR.text(6, SR_THRESHOLD * 1.5, "alarm threshold (ARL 5000)", color=MUTED,
                 fontsize=8.6, va="bottom")
        if i >= jump:
            axR.axvline(jump, color=ALARM, lw=1.3, ls=(0, (4, 3)), zorder=2)
        if alarm_round is not None and i >= alarm_round:
            axR.scatter([alarm_round], [R[alarm_round]], s=46, color=ALARM,
                        edgecolor=SURF, lw=1.1, zorder=7)
            axR.annotate(
                f"staleness model violated\nALARM  ·  +{delay} rounds",
                xy=(alarm_round, R[alarm_round]),
                xytext=(alarm_round + 8, SR_THRESHOLD * 0.06),
                color=ALARM, fontsize=9.4, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ALARM, lw=1.2))
        elif i < jump:
            axR.text(ROUNDS * 0.5, 4.0,
                     "quiet under the correct model\n(median R ≈ 1.4)",
                     color=QUIET, fontsize=9.2, ha="center", va="center")
        axR.set_yscale("log"); axR.set_ylim(0.3, r_hi); axR.set_xlim(0, ROUNDS)
        axR.set_xlabel("planning round", fontsize=10)
        axR.set_ylabel("Shiryaev–Roberts  R$_t$", fontsize=10)
        for s in ("top", "right"):
            axR.spines[s].set_visible(False)
        axR.grid(axis="y", color=GRIDC, lw=0.8, which="major", zorder=0)
        axR.set_axisbelow(True); axR.tick_params(length=0)
        axR.set_title("coverage now observable · WATCH", fontsize=11.5,
                      fontweight="bold", color=INK, loc="left", pad=8)

        # ---------------- header ----------------
        if i < jump:
            tag, msg, col = ("MODEL HOLDS",
                             "band brackets OPT · monitor quiet", QUIET)
        elif alarm_round is not None and i < alarm_round:
            tag, msg, col = ("DRIFT JUMP",
                             "costs surged · statistic climbing", ALARM)
        else:
            tag, msg, col = ("ALARM",
                             f"staleness violated · caught +{delay} rounds after the jump",
                             ALARM)
        t_head.set_text(f"CERT-FLOW · watch_monitor=True · round {i:3d}/{ROUNDS}")
        t_phase.set_text(f"[{tag}]  {msg}"); t_phase.set_color(col)
        t_stat.set_text(f"R$_t$ = {R[i]:,.0f}"
                        if R[i] >= 10 else f"R$_t$ = {R[i]:.1f}")
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    out = ROOT / "assets/animations/watch_alarm.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=FPS), dpi=80,
              savefig_kwargs={"facecolor": SURF})
    plt.close(fig)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {len(frames)} frames @ {FPS} fps)")
    print(f"alarm_round={alarm_round}  delay=+{delay}  peakR={R.max():,.0f}  "
          f"coverage(valid)={ncov}/{nvalid}  post-jump violations={nviol}")


if __name__ == "__main__":
    main()
