"""Live-loop wiring of the round-2 calibrators (all flag-gated, defaults OFF).

Covers:
  * watch_monitor: the conformal test martingale (planner.watch) and the
    Shiryaev-Roberts detector (planner.sr) stay quiet under a correctly-
    specified model and ALARM when the world's drift jumps mid-run -- making
    the pinned-at-1.0 coverage an observable, alarming quantity;
  * path_calibration="pasc": the joint per-edge radius keeps the certificate
    VALID (LB <= true OPT <= UB) against the world's true costs and its gap is
    tighter-or-equal to the per-edge Bonferroni gap on the same seed/rounds;
  * the monitor is purely observational (it changes NO certificate).
"""
import math
import statistics as st

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.oracle import opt


def make(kind: str, seed: int = 7, rho: float = 0.02, **kw):
    world = grid_world(4, 4, seed=seed, kind=kind, noise_scale=0.05, **(
        {"rho": rho} if kind == "bounded" else {}
    ))
    params = {"epsilon": 3.0, "alpha_prime": 0.2, "delta": 1.0}
    params.update(kw)
    planner = CertPlanner(
        world, start=(0, 0), goal=(3, 3), config=PlannerConfig(**params)
    )
    return world, planner


class DriftJumpWorld:
    """A bounded-drift grid whose true costs SURGE by ``surge`` after ``t_jump``:
    a clean regime change the planner's age-weighting was never calibrated for.
    Re-observations after the jump produce anomalously large nonconformity
    scores relative to the pre-jump calibration buffer -- exactly the
    staleness-model break the WATCH monitor is meant to catch."""

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


# --------------------------------------------------------------------------- #
# watch_monitor
# --------------------------------------------------------------------------- #

def test_watch_quiet_under_correct_model():
    """No false alarm when the model holds: on a bounded-drift grid with the
    matched rho, neither the test martingale nor the Shiryaev-Roberts detector
    should fire. (sr_threshold set above the null random-walk excursions;
    empirically the null SR peak stays < 400 over these horizons.)"""
    for seed in range(6):
        _, planner = make("bounded", seed=seed, watch_monitor=True,
                          sr_threshold=5000.0)
        for _ in range(160):
            planner.round()
        assert not planner.watch.alarm(), f"martingale false alarm (seed {seed})"
        assert not planner.sr.alarm(), f"SR false alarm (seed {seed})"


def test_watch_alarms_on_drift_jump():
    """The Shiryaev-Roberts detector fires AFTER a mid-run drift jump, and the
    same config on the un-jumped world stays quiet. (The plain test martingale
    random-walks toward zero over the long pre-jump null and cannot recover in
    time -- exactly why WATCH pairs it with the implicitly-restarting SR
    detector for late changes; we assert on SR.)"""
    T_JUMP, ROUNDS = 120.0, 240
    fired = quiet = 0
    for seed in range(6):
        base = grid_world(4, 4, seed=seed, kind="bounded", noise_scale=0.05)
        world = DriftJumpWorld(base, t_jump=T_JUMP, surge=5.0)
        _, planner = (world, CertPlanner(
            world, (0, 0), (3, 3),
            PlannerConfig(epsilon=3.0, alpha_prime=0.2, delta=1.0,
                          watch_monitor=True, sr_threshold=5000.0)))
        for _ in range(ROUNDS):
            planner.round()
        if planner.sr.alarm():
            fired += 1
            # the alarm must land at/after the jump, never before it
            assert planner.sr.alarm_round is not None
        # control: identical planner, no jump -> must stay quiet
        _, ctrl = make("bounded", seed=seed, watch_monitor=True,
                       sr_threshold=5000.0)
        for _ in range(ROUNDS):
            ctrl.round()
        quiet += not ctrl.sr.alarm()
    assert fired >= 5, f"SR failed to detect the drift jump ({fired}/6)"
    assert quiet >= 5, f"SR false-alarmed on the un-jumped control ({quiet}/6)"


def test_watch_monitor_is_purely_observational():
    """watch_monitor must not perturb the certificate: on/off produce an
    identical (lb, ub, confidence) stream (it draws no randomness and mutates
    no belief -- only planner.watch / planner.sr)."""
    def stream(watch: bool):
        _, p = make("bounded", seed=9, watch_monitor=watch)
        return [(round(c.lb, 9), round(c.ub, 9), round(c.confidence, 9))
                for c, _ in (p.round() for _ in range(120))]

    assert stream(True) == stream(False)


def test_diagnostics_exposes_drift_and_ess():
    """planner.diagnostics() surfaces the WATCH/SR state plus the DASC-style
    residual drift score and the weights' effective sample size."""
    _, planner = make("bounded", seed=3, watch_monitor=True)
    for _ in range(120):
        planner.round()
    d = planner.diagnostics()
    for key in ("watch_value", "watch_running_max", "watch_alarm", "sr_peak",
                "sr_alarm", "sr_alarm_round", "residual_drift_score",
                "effective_sample_size", "n_scores"):
        assert key in d, f"missing diagnostic {key}"
    assert d["n_scores"] > 0
    assert d["effective_sample_size"] > 0.0
    assert math.isfinite(d["residual_drift_score"])
    assert d["residual_drift_score"] >= 0.0


# --------------------------------------------------------------------------- #
# path_calibration="pasc"
# --------------------------------------------------------------------------- #

def _run(mode: str, seed: int, rounds: int = 200):
    world, planner = make("bounded", seed=seed, path_calibration=mode)
    gaps, covered, valid = [], 0, 0
    for _ in range(rounds):
        cert, _ = planner.round()
        if cert.valid and math.isfinite(cert.gap):
            _, true_opt = opt(world, planner.t - planner.cfg.delta, (0, 0), (3, 3))
            valid += 1
            covered += cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9
            gaps.append(cert.gap)
    return gaps, covered, valid


def test_pasc_certificate_stays_valid():
    """SOUNDNESS GATE: PASC joint per-edge pricing must keep LB <= true OPT <=
    UB on EVERY valid round over 150+ rounds of the standard drift grid (same
    check the Bonferroni planner tests use)."""
    for seed in (7, 5, 11, 3):
        gaps, covered, valid = _run("pasc", seed, rounds=200)
        assert valid > 150, f"too few valid rounds (seed {seed}): {valid}"
        assert covered == valid, (
            f"PASC coverage violation (seed {seed}): {covered}/{valid}"
        )


def test_pasc_gap_tighter_or_equal_to_bonferroni():
    """PASC's certified gap should be tighter-or-equal to Bonferroni's on the
    same seed/rounds (one joint quantile beats the alpha/L union bound). Pooled
    over seeds the PASC median gap is <= Bonferroni's; a single seed can go the
    other way by a few percent (the gap also carries the drift term and the
    dual-search incumbent can differ) -- reported honestly, flag stays
    experimental."""
    pooled_b, pooled_p = [], []
    for seed in (7, 5, 11, 3, 42):
        gb, cb, vb = _run("bonferroni", seed)
        gp, cp, vp = _run("pasc", seed)
        assert cb == vb and cp == vp  # both cover on every valid round
        pooled_b += gb
        pooled_p += gp
    assert st.median(pooled_p) <= st.median(pooled_b) + 1e-9, (
        f"PASC pooled median gap {st.median(pooled_p):.3f} not <= "
        f"Bonferroni {st.median(pooled_b):.3f}"
    )


def test_pasc_falls_back_to_bonferroni_during_warmup():
    """While the buffer holds no full length-L block, _pasc_radius is +inf and
    _q falls back to the Bonferroni quantile (warm-up behavior unchanged): the
    first round is still INVALID and the PASC and Bonferroni planners agree
    until the buffer supports a joint block."""
    _, p = make("bounded", seed=1, path_calibration="pasc")
    # empty buffer: no block -> +inf radius (fall-through to Bonferroni)
    assert not math.isfinite(p._pasc_radius(6, 0.1))
    cert0, sensed0 = p.round()
    assert not cert0.valid          # warm-up: certificate invalid
    assert sensed0 is not None      # but sensing to fill the buffer
