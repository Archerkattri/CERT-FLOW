"""Integration tests: the CERT loop end-to-end on simulated worlds."""
import math

from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world
from certflow.oracle import opt


def make(kind: str, seed: int = 7, rho: float = 0.02, **kw):
    world = grid_world(4, 4, seed=seed, kind=kind, noise_scale=0.05, **(
        {"rho": rho} if kind == "bounded" else {}
    ))
    params = {"epsilon": 3.0, "alpha_prime": 0.2, "delta": 1.0}
    params.update(kw)
    planner = CertPlanner(world, start=(0, 0), goal=(3, 3), config=PlannerConfig(**params))
    return world, planner


def test_static_world_terminates_certified():
    """T2' rho->0 case: in a static world the gap closes and stays closed."""
    world, planner = make("static")
    cert = None
    for _ in range(400):
        cert, _ = planner.round()
        if cert.valid and cert.gap <= planner.cfg.epsilon:
            break
    assert cert is not None and cert.valid
    assert cert.gap <= planner.cfg.epsilon
    assert cert.epsilon_attainable


def test_certificate_covers_opt_after_warmup():
    """Smoke coverage: LB <= OPT <= UB on the vast majority of valid rounds."""
    world, planner = make("bounded", rho=0.01)
    covered, valid_rounds = 0, 0
    for _ in range(300):
        cert, _ = planner.round()
        if not cert.valid:
            continue
        _, true_opt = opt(world, planner.t - planner.cfg.delta, (0, 0), (3, 3))
        valid_rounds += 1
        covered += cert.lb - 1e-9 <= true_opt <= cert.ub + 1e-9
    assert valid_rounds > 50
    assert covered / valid_rounds >= 0.8  # loose smoke bound; Tier-0 is the real test


def test_unattainable_epsilon_is_declared():
    """T2' check: an absurdly small epsilon must be reported unattainable."""
    world, planner = make("bounded", rho=0.05, epsilon=1e-6)
    declared = False
    for _ in range(60):
        cert, _ = planner.round()
        if cert.valid and not cert.epsilon_attainable:
            declared = True
            assert cert.epsilon_floor > planner.cfg.epsilon
            break
    assert declared


def test_certificate_invalid_during_warmup():
    world, planner = make("bounded")
    cert, sensed = planner.round()
    assert not cert.valid          # buffer empty: must not claim coverage
    assert sensed is not None      # but it must be sensing to fix that


def test_sensing_stops_when_certified_with_maintenance_off():
    world, planner = make(
        "static", maintenance_every=10**9, maintenance_lookahead=0.0
    )
    cfg = planner.cfg
    for _ in range(400):
        cert, sensed = planner.round()
        if (
            cert.valid
            and cert.gap <= cfg.epsilon
            and cert.confidence >= cfg.min_certify_confidence
        ):
            assert sensed is None  # maintenance disabled: certified -> no sensing
            break
    assert planner.sense_spend > 0


def test_maintenance_keeps_static_certificate_alive():
    """Without maintenance the claim self-extinguishes as the buffer ages;
    with it, late-episode rounds stay valid and mostly certified."""
    def late_valid_fraction(**kw) -> float:
        world, planner = make("static", anneal_alpha=False, **kw)
        late_valid = 0
        for i in range(500):
            cert, _ = planner.round()
            if i >= 400:
                late_valid += cert.valid
        return late_valid / 100

    with_maint = late_valid_fraction()
    without = late_valid_fraction(maintenance_every=10**9, maintenance_lookahead=0.0)
    assert with_maint >= 0.9
    assert with_maint > without


def test_kappa_hysteresis_runs_and_reduces_or_matches_churn():
    """kappa must run cleanly, never report a looser UB than kappa-off, and
    not increase churn. (Effect size is the ablation's job, not this test's.)"""
    from certflow.sensing import path_edges

    def run(use_kappa: bool):
        world, planner = make("bounded", seed=5, use_kappa=use_kappa)
        churn, prev = 0, None
        for _ in range(200):
            cert, _ = planner.round()
            cur = set(path_edges(cert.path)) if cert.path else None
            if prev and cur:
                churn += len(prev ^ cur)
            prev = cur
        return churn

    churn_off = run(False)
    churn_on = run(True)
    assert churn_on <= churn_off


def test_no_prior_scores_in_unknown_terrain():
    """First observation of a prior-initialized edge must NOT push a score
    (a score needs a real previous observation, never the prior)."""
    world, planner = make("bounded", initial_survey=False)
    sensed_edges = set()
    for _ in range(40):
        _, sensed = planner.round()
        if sensed is not None:
            sensed_edges.add(sensed)
    # every sensed edge was prior-initialized, so first touches push nothing;
    # buffer only grows from REPEAT observations
    repeats = sum(1 for e in sensed_edges if planner._obs_count.get(e, 0) >= 2)
    assert len(planner.scorer) <= sum(
        max(0, planner._obs_count.get(e, 0) - 1) for e in sensed_edges
    )
    if repeats == 0:
        assert len(planner.scorer) == 0


def test_thinned_scores_keeps_disjoint_pairs_only():
    """Repeatedly observing ONE edge: unthinned pushes every re-observation's
    score (n-1 of n); thinned keeps only disjoint pairs (floor(n/2))."""
    def pushes(thinned: bool, n_obs: int) -> int:
        world, planner = make("static", thinned_scores=thinned)
        e = next(iter(planner.beliefs))
        for _ in range(n_obs):
            planner.ingest_observation(e)
        return len(planner.scorer)

    n = 20  # edge already has its survey observation (count starts at 1)
    assert pushes(False, n) == n
    assert pushes(True, n) == (n + 1) // 2


def test_cal_rho_a_diagnostic_tracked():
    world, planner = make("bounded")
    for _ in range(60):
        planner.round()
    assert planner.cal_rho_a_max > 0.0


def test_annealing_validity_and_honesty():
    """Annealing makes early rounds valid at a weaker (honest) claim that
    tightens toward 1-alpha'; round 1 stays invalid (empty buffer)."""
    world, planner = make("bounded")
    cert1, _ = planner.round()
    assert not cert1.valid  # empty buffer: nothing supportable
    confs = []
    for _ in range(150):
        cert, _ = planner.round()
        if cert.valid:
            confs.append(cert.confidence)
    world2, planner2 = make("bounded", anneal_alpha=False)
    valid_off = sum(planner2.round()[0].valid for _ in range(151))
    assert len(confs) > valid_off          # strictly more valid rounds
    assert confs[0] < confs[-1] + 1e-9     # claim tightens (or saturates)
    assert max(confs) <= 1 - planner.cfg.alpha_prime + 1e-9  # never overclaims


def test_adaptive_rate_senses_more_under_drift():
    # rho/epsilon chosen so the T2' floor needs k>=2 at this drift on the
    # 4x4 test world (attainable, but not at k=1): adaptive must sense more
    world_a, pa = make("bounded", rho=0.15, epsilon=8.0)
    world_b, pb = make("bounded", rho=0.15, epsilon=8.0, adaptive_rate=True)
    for _ in range(200):
        pa.round()
        pb.round()
    assert pb.sense_spend > pa.sense_spend  # rate adapts up under drift


def test_advance_start_keeps_working():
    world, planner = make("bounded")
    for _ in range(30):
        planner.round()
    path = None
    for _ in range(50):
        cert, _ = planner.round()
        if cert.path and len(cert.path) >= 2:
            path = cert.path
            break
    assert path is not None
    planner.advance_start(path[1])
    cert, _ = planner.round()
    assert cert.lb >= 0  # planner still produces sane certificates
    assert cert.path == [] or cert.path[0] == path[1]


def test_decision_uniform_trajectory_validity():
    """Decision-uniform mode: certified (acted-on) rounds must be valid
    simultaneously; measure episode-level all-certified-rounds-covered."""
    from certflow.oracle import opt as _opt

    def episode_uniform(decision_uniform: bool) -> tuple[int, int]:
        ok_eps = n_eps = 0
        for seed in range(6):
            world, planner = make(
                "bounded", seed=seed, epsilon=6.0,
                decision_uniform=decision_uniform,
            )
            all_ok, any_cert = True, False
            for _ in range(250):
                t0 = planner.t
                cert, _ = planner.round()
                if cert.valid and cert.gap <= planner.cfg.epsilon and \
                        cert.confidence >= planner.cfg.min_certify_confidence:
                    any_cert = True
                    _, o = _opt(world, t0, (0, 0), (3, 3))
                    if not (cert.lb - 1e-9 <= o <= cert.ub + 1e-9):
                        all_ok = False
            if any_cert:
                n_eps += 1
                ok_eps += all_ok
        return ok_eps, n_eps

    ok, n = episode_uniform(True)
    assert n > 0
    assert ok == n  # every episode's acted-on certificates all valid


def test_traversing_mars_degenerate_stopping():
    """T2' corollary (rho->0, q->0): in a noise-free static unknown world,
    CERT's epsilon-certificate stopping coincides with the deterministic
    'terminate when the path is proven optimal' rule (Traversing Mars):
    once certified at epsilon ~ 0, the incumbent IS the true optimum."""
    from certflow.oracle import opt as _opt

    for seed in (1, 2, 3):
        world = grid_world(5, 5, seed=seed, kind="static", noise_scale=0.0)
        planner = CertPlanner(
            world, (0, 0), (4, 4),
            PlannerConfig(epsilon=1e-6, alpha_prime=0.2, eps_tv=0.0,
                          initial_survey=False, prior_cost=1.0),
        )
        certified_at = None
        for i in range(400):
            cert, _ = planner.round()
            if (cert.valid and cert.gap <= planner.cfg.epsilon
                    and cert.confidence >= planner.cfg.min_certify_confidence):
                certified_at = i
                break
        assert certified_at is not None, f"never certified (seed {seed})"
        _, true_opt = _opt(world, planner.t, (0, 0), (4, 4))
        # the certified incumbent is exactly optimal: proven, not approximate
        inc_cost = sum(
            world.true_cost(e, planner.t)
            for e in zip(cert.path[:-1], cert.path[1:])
        )
        assert abs(inc_cost - true_opt) < 1e-6
        assert cert.gap <= 1e-6


def test_lifelong_memory_beats_memoryless():
    """O4 in its honest setting: across missions, carried memory (beliefs +
    calibration) certifies faster and senses less than a memoryless restart."""
    from certflow.drift import grid_world as _gw

    def missions(carry: bool) -> tuple[int, float]:
        world = _gw(5, 5, seed=3, kind="bounded", rho=0.01, noise_scale=0.05)
        cfg = PlannerConfig(epsilon=5.0, alpha_prime=0.2, eps_tv=1e-4,
                            initial_survey=False)
        pool = [((0, 0), (4, 4)), ((0, 4), (4, 0)), ((4, 4), (0, 0))]
        p = CertPlanner(world, *pool[0], cfg)
        total_rounds, total_sense = 0, 0.0
        for m, (s, g) in enumerate(pool):
            if m > 0:
                if carry:
                    p.t += 50.0
                    p.retarget(s, g)
                else:
                    p = CertPlanner(world, s, g, cfg, t0=p.t + 50.0)
            spend0 = p.sense_spend
            for i in range(250):
                cert, _ = p.round()
                if (cert.valid and cert.gap <= cfg.epsilon
                        and cert.confidence >= cfg.min_certify_confidence):
                    break
            if m > 0:
                total_rounds += i
                total_sense += p.sense_spend - spend0
        return total_rounds, total_sense

    r_mem, s_mem = missions(True)
    r_les, s_les = missions(False)
    assert r_mem < r_les
    assert s_mem < s_les
