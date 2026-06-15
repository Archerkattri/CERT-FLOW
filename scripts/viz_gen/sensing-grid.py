"""Sensing that pays: a regret-race on synthetic unknown drifting terrain.

Honest by construction. Four navigation policies share the SAME unknown-terrain
drifting grid and the SAME fixed sensing budget B, then race a clairvoyant
oracle that replans on true costs every step:

  - CERT     : gap-directed (route-critical) sensing  -- the contribution
  - random   : sense a random edge each round
  - max-age  : sense the stalest edge (freshness / persistent monitoring)
  - drive-blind : no certificate, no sensing -- depart immediately on the prior

This is EXACTLY the Tier-2 episode of scripts/run_tier2.py / episodes.py:
each round the planner certifies and senses (per its sensing_policy) until it is
epsilon-certified or the budget is spent, then the robot traverses one edge of
the incumbent, paying the TRUE cost (a free observation), and repeats to the
goal. Running travel-regret = (true travel paid so far) - (clairvoyant oracle's
true cost to reach the robot's CURRENT node). At the goal it equals the paper's
Tier-2 regret = travel_cost - oracle_cost.

What is MEASURED at render time (nothing staged, nothing hardcoded):
  * every regret value comes from a real CertPlanner run of the actual library;
  * curves are the MEAN running regret over N independent seeds (the honest
    claim -- on a single seed CERT is not always lowest; the systematic
    advantage is a mean-over-seeds effect, matching docs/results/tier2-regret.md);
  * the grid panel replays ONE representative seed's real run for legibility.

Honesty notes wired into the frame:
  * The pre-departure SENSING PHASE (certify-then-go has paid no travel cost
    yet) is shown as a grey "sensing, not moving yet" band -- regret is 0 there
    because the robot has not moved, which is NOT a coverage claim or a
    violation. drive-blind has no such band (it departs at round 1).
  * drive-blind reaches the goal in ~18 rounds; after that its regret is final
    and flat (it stopped). The certify-then-go policies sense ~B rounds first,
    so they finish later -- the explicit, honest mission-time trade.
  * No "race to the goal on a known static map" framing: the map is UNKNOWN and
    DRIFTING; the story is which sensing policy converts a fixed observation
    budget into the lowest travel-regret.

Output (all under viz_out/sensing-grid/):
  sensing-grid.mp4, sensing-grid.gif, poster_1..4.png
"""
from __future__ import annotations

import dataclasses
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from certflow.cert import CertPlanner
from certflow.drift import grid_world
from certflow.episodes import oracle_walk_cost, planner_config
from certflow.harness import ExperimentConfig

# ---- colorblind-safe palette (Wong) -----------------------------------------
BLUE = "#0072B2"   # CERT
SKY = "#56B4E9"   # CERT fill / secondary
ORANGE = "#D55E00"   # baseline emphasis (random)
GRN = "#009E73"   # max-age
PURP = "#CC79A7"   # drive-blind
BLK = "#111111"   # oracle / true OPT
RED = "#CC2B1D"   # sensed-edge marker / start-goal accents

OUT = pathlib.Path("viz_out/sensing-grid")
OUT.mkdir(parents=True, exist_ok=True)

# ---- experiment config: Tier-2 spec, unknown drifting terrain ---------------
# Mirrors scripts/run_tier2.py BASE (10x10 bounded drift rho=0.02, no survey,
# epsilon=8, alpha'=0.2). Budget B=20 reproduces the doc's cleanest-separation
# row (tier2-regret.md: cert 2.27 vs random 7.07 / max_age 4.49 / blind 6.60).
# Each round = 1 obs at cost 0.1, so certify-then-go spends ~200 obs (rounds)
# mapping the unknown grid before it can certify, then drives the certified
# route (~18 moves) -> ~217-round episodes; drive-blind departs at round 1.
# The long sensing phase is frame-subsampled (SENSE_STRIDE) so the clip stays
# short -- this changes only which rounds become frames, never a measured value.
BASE = ExperimentConfig(
    rows=10, cols=10, kind="bounded", rho=0.02,
    noise_family="gaussian", noise_scale=0.05,
    epsilon=8.0, alpha_prime=0.2, rho_w=0.99, eps_tv=1e-4,
    gamma_aci=0.01, delta=1.0, rho_hat_over_rho=1.0, use_kappa=True,
    initial_survey=False, max_rounds=600,
)
BUDGET = 20.0
ROWS = COLS = 10
START, GOAL = (0, 0), (ROWS - 1, COLS - 1)

# Policies: (sensing_policy, move_policy, budget, label, color)
POLICIES = [
    ("cert", "when_certified", BUDGET, "CERT (gap-directed)", BLUE),
    ("random", "when_certified", BUDGET, "random", ORANGE),
    ("max_age", "when_certified", BUDGET, "max-age", GRN),
    ("none", "always", float("inf"), "drive-blind", PURP),
]

N_SEEDS = 15            # mean-over-seeds = the honest claim (see module docstring)
BASE_SEED = 2026
GRID_SEED = 2033        # representative single run for the grid panel
MAX_ROUNDS_CAP = 240    # animation horizon (episodes finish well within this)
FPS = 6

# Frame subsampling: the pre-departure SENSING PHASE is genuinely static for the
# regret curve (no travel cost paid -> regret 0) and long (~200 rounds at B=20),
# so we render only every SENSE_STRIDE-th sensing round; the DRIVING PHASE (where
# the regret race actually happens) is rendered every round. This never alters a
# measured value -- the regret curve plotted at each frame is the true per-round
# series up to that frame's round; we only choose WHICH rounds become frames.
SENSE_STRIDE = 12


# ---------------------------------------------------------------------------
# Capture: a REAL Tier-2 run, recording per-round running regret & geometry.
# ---------------------------------------------------------------------------
def run_episode(sp: str, mp: str, budget: float, seed: int, want_geom: bool):
    """Replay one Tier-2 episode (episodes.tier2_episode loop, instrumented).

    Returns running-regret series (per round) plus, if want_geom, the grid
    geometry needed to draw this run: current position, incumbent path, sensed
    edge, true-cost heatmap, and the round-by-round time stamp.
    """
    cfg = dataclasses.replace(
        BASE, sensing_policy=sp, move_policy=mp, sense_budget=budget,
        max_rounds=MAX_ROUNDS_CAP,
    )
    world = grid_world(cfg.rows, cfg.cols, seed=seed, kind=cfg.kind, rho=cfg.rho,
                       noise_family=cfg.noise_family, noise_scale=cfg.noise_scale)
    planner = CertPlanner(world, START, GOAL, planner_config(cfg))

    oracle_total = oracle_walk_cost(world, START, GOAL, 0.0, cfg.delta)

    # oracle cost-to-reach a node = clairvoyant walk START -> node (memoized).
    _o2_cache: dict = {START: 0.0}

    def oracle_to(node):
        if node not in _o2_cache:
            _o2_cache[node] = oracle_walk_cost(world, START, node, 0.0, cfg.delta)
        return _o2_cache[node]

    pos = START
    travel = 0.0
    moving = False               # has the robot departed yet?
    reached_round = None
    regret = []                  # running travel-regret per round
    moved_flag = []              # did the robot move THIS round?
    geom = [] if want_geom else None

    for r in range(cfg.max_rounds):
        cert, sensed = planner.round()
        certified_now = bool(cert.valid and cert.gap <= cfg.epsilon)
        may_move = (
            mp == "always"
            or certified_now
            or planner.sense_spend >= cfg.sense_budget
        )
        did_move = False
        if may_move and cert.path and len(cert.path) >= 2 and cert.path[0] == pos:
            moving = True
            e = (cert.path[0], cert.path[1])
            travel += world.true_cost(e, planner.t)  # MEASURED true cost
            planner.ingest_observation(e)
            pos = cert.path[1]
            planner.advance_start(pos)
            did_move = True
            if pos == GOAL and reached_round is None:
                reached_round = r

        # running regret = true travel paid so far minus the clairvoyant
        # oracle's true cost to the robot's CURRENT node (0 before departure).
        regret.append(travel - oracle_to(pos))
        moved_flag.append(did_move)

        if want_geom:
            t_eval = planner.t - cfg.delta
            heat = np.full((ROWS, COLS), np.nan)
            for (u, nb) in world.graph.items():
                vals = [world.true_cost((u, v), t_eval) for v in nb]
                if vals:
                    heat[u[0], u[1]] = float(np.mean(vals))
            geom.append(dict(
                pos=pos, path=list(cert.path) if cert.path else [],
                sensed=sensed, heat=heat, moving=moving, did_move=did_move,
                certified=certified_now, gap=cert.gap if np.isfinite(cert.gap) else np.nan,
            ))

        if pos == GOAL:
            break

    return dict(
        regret=np.array(regret, dtype=float),
        moved=np.array(moved_flag, dtype=bool),
        reached_round=reached_round,
        n_rounds=len(regret),
        final_regret=travel - oracle_total,
        oracle_total=oracle_total,
        geom=geom,
    )


def pad(series: np.ndarray, n: int) -> np.ndarray:
    """Extend a running-regret series to length n by holding its last value
    (a policy that already reached the goal keeps its final regret -- it has
    stopped; honest, not invented)."""
    if len(series) >= n:
        return series[:n]
    if len(series) == 0:
        return np.zeros(n)
    return np.concatenate([series, np.full(n - len(series), series[-1])])


def capture():
    """Run every policy over N_SEEDS; build mean running-regret curves and the
    representative-seed geometry. All numbers MEASURED here."""
    seeds = [BASE_SEED + i for i in range(N_SEEDS)]
    # ensure the representative grid seed is one of those we average over
    if GRID_SEED not in seeds:
        seeds.append(GRID_SEED)

    per_policy = {}
    horizon = 0
    print(f"Capturing {len(POLICIES)} policies x {len(seeds)} seeds "
          f"(B={BUDGET}, eps={BASE.epsilon}) ...", flush=True)
    for sp, mp, b, label, color in POLICIES:
        runs = []
        depart_rounds = []
        finals = []
        for sd in seeds:
            want_geom = (sd == GRID_SEED)
            res = run_episode(sp, mp, b, sd, want_geom)
            runs.append(res)
            finals.append(res["final_regret"])
            # departure round = first round the robot moved
            mv = np.nonzero(res["moved"])[0]
            depart_rounds.append(int(mv[0]) if len(mv) else res["n_rounds"])
            horizon = max(horizon, res["n_rounds"])
        per_policy[label] = dict(
            runs=runs, color=color, finals=np.array(finals),
            depart_med=int(np.median(depart_rounds)),
            grid_run=next(rr for rr, sd in zip(runs, seeds) if sd == GRID_SEED),
        )
        a = np.array(finals)
        print(f"  {label:22s} final-regret mean={a.mean():5.2f} "
              f"median={np.median(a):5.2f} (over {len(seeds)} seeds), "
              f"median depart=round {per_policy[label]['depart_med']}", flush=True)

    horizon = min(horizon, MAX_ROUNDS_CAP)

    # mean (and min/max spread) running-regret curve per policy over seeds
    for label, d in per_policy.items():
        M = np.vstack([pad(r["regret"], horizon) for r in d["runs"]])
        d["mean_curve"] = M.mean(axis=0)
        d["lo_curve"] = M.min(axis=0)
        d["hi_curve"] = M.max(axis=0)

    return per_policy, horizon, len(seeds)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(per_policy, horizon, n_seeds):
    from matplotlib.lines import Line2D

    labels = [lab for _, _, _, lab, _ in POLICIES]
    short = {labels[0]: "CERT", labels[1]: "random",
             labels[2]: "max-age", labels[3]: "drive-blind"}
    cert_label = labels[0]

    # y-limit for the regret plot from the real curves
    ymax = max(d["hi_curve"].max() for d in per_policy.values())
    ymax = max(ymax, max(d["mean_curve"].max() for d in per_policy.values())) * 1.12
    ymax = max(ymax, 1.0)

    # final mean regret per policy (MEASURED) -> ranking best(lowest)->worst,
    # and the winner, used to state the result in the title and mark the bars.
    final_mean = {lab: float(per_policy[lab]["finals"].mean()) for lab in labels}
    rank_order = sorted(labels, key=lambda lab: final_mean[lab])  # best -> worst
    winner = rank_order[0]
    runner_up = rank_order[1]
    win_ratio = (final_mean[runner_up] / final_mean[winner]
                 if final_mean[winner] > 1e-9 else float("nan"))

    # representative-seed grid run drives the grid panel
    grid_geom = per_policy[cert_label]["grid_run"]["geom"]
    n_grid = len(grid_geom)

    # departure round on the grid seed = first round CERT moves (budget-limited:
    # all certify-then-go policies depart together). This is where the regret
    # race actually begins; before it, robots are mapping.
    cert_moved = per_policy[cert_label]["grid_run"]["moved"]
    mv = np.nonzero(cert_moved)[0]
    depart = int(mv[0]) if len(mv) else horizon // 2

    # ---- frame -> true-round schedule ---------------------------------------
    # Subsample the long static sensing phase (regret 0, robots stationary),
    # render every driving round. The plotted curve is always the TRUE per-round
    # series up to the frame's round -- only frame selection is subsampled.
    sensing_rounds = list(range(0, depart, SENSE_STRIDE))
    if sensing_rounds and sensing_rounds[-1] != depart - 1:
        sensing_rounds.append(depart - 1)
    driving_rounds = list(range(depart, horizon))
    frame_rounds = sensing_rounds + driving_rounds
    # a short hold on the final frame so the finish reads in the loop
    frame_rounds += [horizon - 1] * 4

    fig = plt.figure(figsize=(12.8, 5.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.34], height_ratios=[1.0, 0.6])
    gx = fig.add_subplot(gs[:, 0])     # grid (spans both rows)
    gr = fig.add_subplot(gs[0, 1])     # running-regret curves
    gb = fig.add_subplot(gs[1, 1])     # live bar chart of current mean regret

    # state WHO WINS, with the measured numbers, right in the title (kept short
    # enough to never clip at this figure width).
    fig.suptitle(
        "Sensing that pays  -  same unknown drifting terrain, same budget "
        f"(B={BUDGET:.0f} obs);  metric = travel-regret, lower is better\n"
        f"WINNER: {short[winner]} -- lowest regret ({final_mean[winner]:.2f}), "
        f"{win_ratio:.1f}x better than next-best {short[runner_up]} "
        f"({final_mean[runner_up]:.2f})",
        fontsize=12.0, fontweight="bold",
    )

    def draw_regret(t):
        """t is a TRUE round index; plot the measured curves up to round t."""
        gr.clear()
        xs = np.arange(horizon)
        kk = min(t + 1, horizon)
        # honesty band: pre-departure, certify-then-go has paid NO travel cost
        # (regret 0) because it has not moved -- a SENSING phase, not a coverage
        # claim and not a violation. drive-blind has no such band (departs at r1).
        # Place the note in the EMPTY gap inside the grey band, between the
        # zero-regret lines (CERT/random/max-age) and the drive-blind plateau,
        # so it never touches a curve, the rising right-side band, or the legend.
        if depart > 1:
            gr.axvspan(0, depart, color="0.88", alpha=0.6, lw=0, zorder=0)
            # drive-blind's plateau within the sensing window = the only nonzero
            # curve there; sit comfortably below it (and above 0).
            blind_plateau = float(per_policy[labels[3]]["mean_curve"]
                                  [max(0, depart - 1)])
            note_y = min(blind_plateau * 0.5, ymax * 0.20)
            gr.text(depart * 0.5, note_y,
                    "sensing phase: certify-then-go is mapping, not moving yet\n"
                    "(regret 0 = no travel paid, NOT a coverage claim)",
                    fontsize=7.3, color="0.28", ha="center", va="center",
                    zorder=1, linespacing=1.3)
        gr.axhline(0.0, color=BLK, lw=1.6, ls="--", zorder=2,
                   label="clairvoyant oracle (regret 0)")
        for _, _, _, lab, _ in POLICIES:
            d = per_policy[lab]
            c = d["color"]
            # min-max spread (over seeds) for CERT and random: shows the claim
            # is a distribution, not one lucky seed
            if lab in (cert_label, labels[1]):
                fill_c = SKY if lab == cert_label else c
                gr.fill_between(xs[:kk], d["lo_curve"][:kk], d["hi_curve"][:kk],
                                color=fill_c, alpha=0.16, lw=0, zorder=2)
            gr.plot(xs[:kk], d["mean_curve"][:kk], "-", color=c, lw=2.6,
                    zorder=4, label=lab, solid_capstyle="round")
            if kk >= 1:
                gr.plot(kk - 1, d["mean_curve"][kk - 1], "o", color=c, ms=6,
                        zorder=5, mec="white", mew=0.8)
        gr.set_xlim(0, horizon)
        gr.set_ylim(-ymax * 0.06, ymax)
        # directionality cue lives IN the axis label (never over data):
        gr.set_ylabel("running travel-regret\n"
                      r"$\downarrow$ lower is better (closer to oracle)",
                      fontsize=9.5)
        gr.set_xlabel("replanning round", fontsize=9.5)
        gr.set_title(f"Regret accumulating over rounds  (mean of {n_seeds} seeds; "
                     "band = seed min-max)", fontsize=10.0)
        gr.legend(loc="upper left", fontsize=8.0, framealpha=0.92, ncol=2)
        gr.tick_params(labelsize=8.5)
        gr.grid(True, alpha=0.25, lw=0.6)

    def draw_bars(t):
        gb.clear()
        kk = min(t, horizon - 1)
        # rank bars best (lowest regret) -> worst by the CURRENT mean value, so
        # the ladder is always visually monotonic; the winner is tagged.
        cur = {lab: float(per_policy[lab]["mean_curve"][kk]) for lab in labels}
        order = sorted(labels, key=lambda lab: cur[lab])         # best -> worst
        vals = [cur[lab] for lab in order]
        finals = [final_mean[lab] for lab in order]
        colors = [per_policy[lab]["color"] for lab in order]
        # top row = best: place rank 1 at the TOP (highest y)
        ypos = np.arange(len(order))[::-1]
        gb.barh(ypos, vals, color=colors, alpha=0.92, height=0.62, zorder=3)
        # tick where each policy's FINAL (episode-end) mean regret lands
        for y, fv in zip(ypos, finals):
            gb.plot([fv, fv], [y - 0.32, y + 0.32], color=BLK, lw=1.5,
                    alpha=0.6, zorder=4)
        xr = ymax * 1.04                                          # x right edge
        # "✓ best at finish" tags the MEASURED episode-end winner (a stable fact),
        # so it never overclaims on a frame where everyone is still tied at 0.
        for rank, (y, v, lab) in enumerate(zip(ypos, vals, order), start=1):
            best = (lab == winner)
            tag = "   ✓ best at finish" if best else ""
            gb.text(v + ymax * 0.012, y, f"{v:.2f}{tag}", va="center",
                    ha="left", fontsize=9.0,
                    fontweight=("bold" if best else "normal"),
                    color=(per_policy[lab]["color"] if best else "0.15"))
        gb.axvline(0.0, color=BLK, lw=1.4, ls="--", zorder=2)
        gb.set_yticks(ypos)
        # rank number + name; winner label in bold colour
        yticklabels = [f"{r}. {short[lab]}"
                       for r, lab in enumerate(order, start=1)]
        gb.set_yticklabels(yticklabels, fontsize=9.0)
        for tick_lab, lab in zip(gb.get_yticklabels(), order):
            if lab == winner:
                tick_lab.set_fontweight("bold")
                tick_lab.set_color(per_policy[lab]["color"])
        gb.set_xlim(min(-ymax * 0.04, min(vals) - ymax * 0.02), xr)
        # directionality cue in the axis label (never over data):
        gb.set_xlabel(r"current mean travel-regret   $\downarrow$ lower is better"
                      "   ( | marks episode-end value )", fontsize=8.4)
        gb.set_title(f"Ranking right now: best → worst "
                     f"(winner: {short[winner]})", fontsize=10.0)
        gb.tick_params(labelsize=8.5)
        gb.grid(True, axis="x", alpha=0.25, lw=0.6)

    markers = {labels[0]: "o", labels[1]: "s", labels[2]: "D", labels[3]: "P"}
    offs = {labels[0]: (0.0, -0.16), labels[1]: (0.17, 0.0),
            labels[2]: (-0.17, 0.0), labels[3]: (0.0, 0.18)}

    def draw_grid(t):
        gx.clear()
        gi = min(t, n_grid - 1)
        g = grid_geom[gi]
        moving = g["moving"]
        gx.imshow(g["heat"], cmap="YlOrBr", origin="upper", alpha=0.6,
                  extent=[-0.5, COLS - 0.5, ROWS - 0.5, -0.5], zorder=0)

        # CERT's incumbent (optimistic) route this round -- the route it is
        # certifying and, once certified, drives. During sensing it is the
        # corridor CERT's gap-directed observations focus on (the route-critical
        # set + age backstop in certflow.sensing.select_observation).
        path = g["path"]
        if len(path) > 1:
            ys = [n[0] for n in path]
            xs = [n[1] for n in path]
            route_lbl = ("CERT certified route" if moving
                         else "CERT optimistic route (being certified)")
            gx.plot(xs, ys, "-", color=BLUE, lw=3.0, alpha=0.95, zorder=3,
                    solid_capstyle="round", label=route_lbl)
        # CERT's gap-directed sensed edge this round (any phase)
        if g["sensed"] is not None:
            (u, v) = g["sensed"]
            gx.plot([u[1], v[1]], [u[0], v[0]], "-", color=RED, lw=4.5,
                    alpha=0.8, solid_capstyle="round", zorder=4)
            gx.plot((u[1] + v[1]) / 2, (u[0] + v[0]) / 2, "v", color=RED,
                    ms=9, zorder=5, label="CERT sensed edge (this round)")

        # the four robots' current positions (offset so co-located markers show)
        for lab in labels:
            gg = per_policy[lab]["grid_run"]["geom"]
            p = gg[min(t, len(gg) - 1)]["pos"]
            dx, dy = offs[lab]
            gx.plot(p[1] + dx, p[0] + dy, markers[lab],
                    color=per_policy[lab]["color"], ms=12, zorder=6,
                    mec="white", mew=1.3)
        gx.plot(START[1], START[0], "o", color="0.2", ms=7, zorder=2,
                mec="white", mew=1.0)
        gx.plot(GOAL[1], GOAL[0], "*", color=RED, ms=20, zorder=7,
                mec="white", mew=0.8)
        gx.set_xticks([])
        gx.set_yticks([])
        gx.set_xlim(-0.7, COLS - 0.3)
        gx.set_ylim(ROWS - 0.3, -0.7)

        phase = "SENSING  (all policies mapping the unknown grid)" if not moving \
            else "DRIVING  (each robot follows its own route to goal)"
        gx.set_title(
            f"Unknown {ROWS}x{COLS} terrain, drifting  -  round {t + 1}\n"
            f"phase: {phase}\n"
            "background = current true edge cost (darker = costlier)",
            fontsize=9.6)

        # robot-position legend
        handles = [
            Line2D([0], [0], marker=markers[lab], color="w",
                   markerfacecolor=per_policy[lab]["color"], markersize=10,
                   markeredgecolor="white", label=short[lab])
            for lab in labels
        ]
        handles.append(Line2D([0], [0], marker="*", color="w",
                              markerfacecolor=RED, markersize=15, label="goal"))
        leg1 = gx.legend(handles=handles, loc="upper right", fontsize=7.3,
                         framealpha=0.93, title="robot position", ncol=2,
                         title_fontsize=7.5)
        gx.add_artist(leg1)
        # route/sensed-edge legend (lower-left)
        gx.legend(loc="lower left", fontsize=7.0, framealpha=0.92)
        if not moving:
            gx.text(0.5, -0.04,
                    "all four robots are still at START (markers stacked) - "
                    "spending the same budget B mapping before departure",
                    transform=gx.transAxes, fontsize=7.2, color="0.3",
                    ha="center", va="top")

    def draw_round(t):
        draw_grid(t)
        draw_regret(t)
        draw_bars(t)
        return []

    def draw_frame(fi):
        return draw_round(frame_rounds[fi])

    anim = FuncAnimation(fig, draw_frame, frames=len(frame_rounds),
                         interval=int(1000 / FPS))

    # ---- MP4 (preferred) -----------------------------------------------------
    mp4 = OUT / "sensing-grid.mp4"
    used_mp4 = False
    try:
        writer = FFMpegWriter(fps=FPS, bitrate=1500,
                              metadata={"title": "CERT-FLOW sensing-grid"})
        anim.save(str(mp4), writer=writer, dpi=108)
        used_mp4 = True
        print(f"MP4: {mp4} ({mp4.stat().st_size // 1024} KB)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"FFMpegWriter failed ({exc}); MP4 skipped, GIF will stand in.",
              flush=True)

    # ---- GIF (looping, web embed; lower dpi to stay small) ------------------
    gif = OUT / "sensing-grid.gif"
    anim.save(str(gif), writer=PillowWriter(fps=FPS), dpi=72)
    print(f"GIF: {gif} ({gif.stat().st_size // 1024} KB)", flush=True)

    # ---- 4 poster PNGs: deep sensing / just-departed / mid-drive / finish ---
    p_sense = max(0, depart - 1)                          # end of sensing phase
    p_early = max(1, depart // 2)                         # mid sensing
    p_mid = min(horizon - 1, depart + (horizon - depart) // 2)  # mid drive
    p_final = horizon - 1                                 # finish
    poster_rounds = [p_early, p_sense, p_mid, p_final]
    for i, t in enumerate(poster_rounds, start=1):
        draw_round(t)
        fig.savefig(OUT / f"poster_{i}.png", dpi=110)
        print(f"poster_{i}.png @ round {t + 1}", flush=True)
    plt.close(fig)

    # ---- printed MEASURED summary (sanity vs docs/results/tier2-regret.md) --
    print("\n=== MEASURED final mean travel-regret (over "
          f"{n_seeds} seeds, B={BUDGET:.0f}, eps={BASE.epsilon:.0f}) ===")
    cert_mean = float(per_policy[cert_label]["finals"].mean())
    for lab in labels:
        a = per_policy[lab]["finals"]
        ratio = (a.mean() / cert_mean) if cert_mean > 1e-9 else float("nan")
        extra = "" if lab == cert_label else f"  ({ratio:.2f}x CERT)"
        print(f"  {lab:22s} mean={a.mean():5.2f}  median={np.median(a):5.2f}"
              f"{extra}")
    print(f"  grid seed {GRID_SEED}: departs at round {depart + 1}, "
          f"{len(frame_rounds)} animation frames "
          f"({len(sensing_rounds)} sensing + {len(driving_rounds)} driving)")
    return used_mp4, poster_rounds


if __name__ == "__main__":
    per_policy, horizon, n_seeds = capture()
    used_mp4, poster_rounds = render(per_policy, horizon, n_seeds)
    if not used_mp4:
        print("\nNOTE: ffmpeg MP4 writer was unavailable; produced GIF only.")
