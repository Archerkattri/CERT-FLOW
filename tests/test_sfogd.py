"""Scale-free OGD step-size for ACI (SAOCP, arXiv 2302.07869, Alg. 2)."""
import math

import numpy as np
import pytest

from certflow.conformal import ACITracker


def test_sfogd_validation_and_defaults():
    assert ACITracker(0.1).mode == "fixed"  # default unchanged
    with pytest.raises(ValueError):
        ACITracker(0.1, mode="bogus")
    with pytest.raises(ValueError):
        ACITracker(0.1, mode="sf-ogd", eta=0.0)
    # zero-gradient step (err rate == target exactly) leaves alpha put
    a = ACITracker(0.1, mode="sf-ogd")
    a.update(err=False)  # g = -0.1 (nonzero) -> moves, but stays finite/clipped
    assert 0.0 < a.working_alpha() < 1.0


def _biased_coverage_stream(mode, bias, seed, T=8000, **kw):
    """The realized miscoverage of the underlying predictor at working level a
    is clip(a + bias, 0, 1): the interval is mis-scaled by `bias`, so to hit
    target miscoverage the tracker must drive its working alpha to
    (target - bias) -- feasible only when that is in (0, 1), so we use bias < 0
    (the predictor over-covers; the tracker must RAISE alpha). Both trackers
    face the SAME noise stream."""
    rng = np.random.default_rng(seed)
    aci = ACITracker(alpha_target=0.1, mode=mode, **kw)
    errs = []
    for _ in range(T):
        a = aci.working_alpha()
        p_err = min(1.0, max(0.0, a + bias))
        err = bool(rng.random() < p_err)
        errs.append(err)
        aci.update(err)
    return float(np.mean(errs))


def test_sfogd_converges_and_beats_fixed_on_coverage_error():
    target = 0.1
    devs_fixed, devs_sf = [], []
    for seed in range(12):
        # common noise seed so the two trackers see the same randomness
        m_fixed = _biased_coverage_stream("fixed", bias=-0.2, seed=seed,
                                          gamma=0.005)
        m_sf = _biased_coverage_stream("sf-ogd", bias=-0.2, seed=seed, eta=0.1)
        devs_fixed.append(abs(m_fixed - target))
        devs_sf.append(abs(m_sf - target))
    mean_fixed = float(np.mean(devs_fixed))
    mean_sf = float(np.mean(devs_sf))
    # SF-OGD converges (small coverage error) ...
    assert mean_sf < 0.03, mean_sf
    # ... and its coverage error is <= fixed-eta ACI's on this stream
    assert mean_sf <= mean_fixed + 1e-9, (mean_sf, mean_fixed)


def test_sfogd_anytime_no_gamma_tuning():
    """Scale-free: the SAME eta controls coverage across very different bias
    magnitudes, where a single fixed gamma would be mistuned for one of them."""
    for bias in (-0.05, -0.2, -0.35):
        m = _biased_coverage_stream("sf-ogd", bias=bias, seed=7, eta=0.1)
        assert abs(m - 0.1) < 0.04, (bias, m)


def test_sfogd_planner_runs_and_is_sound():
    """A CertPlanner with aci_mode='sf-ogd' runs and stays sound."""
    from certflow import CertPlanner, PlannerConfig
    from certflow.drift import grid_world
    import heapq

    def opt(world, s, g, t):
        adj = {}
        for u in world.graph:
            for v in world.graph[u]:
                adj.setdefault(u, {})[v] = max(world.true_cost((u, v), t), 1e-9)
        dist = {s: 0.0}
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, c in adj.get(u, {}).items():
                nd = d + c
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist.get(g, math.inf)

    cfg = PlannerConfig(epsilon=5.0, alpha_prime=0.1, aci_mode="sf-ogd", aci_eta=0.1)
    hits = tot = 0
    for seed in range(4):
        w = grid_world(6, 6, seed=seed, kind="bounded", rho=0.02, noise_scale=0.05)
        p = CertPlanner(w, (0, 0), (5, 5), cfg)
        for r in range(120):
            c, _ = p.round()
            if r < 30 or not c.valid:
                continue
            o = opt(w, (0, 0), (5, 5), p.t - cfg.delta)
            if math.isinf(o):
                continue
            hits += c.lb <= o + 1e-6 and o <= c.ub + 1e-6
            tot += 1
    assert tot > 50
    assert hits / tot >= 0.9
