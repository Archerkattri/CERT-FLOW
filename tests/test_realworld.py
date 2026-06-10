"""Smoke tests for the METR-LA replay world (skipped if data absent)."""
import math
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data/metr-la/metr_la.h5"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="METR-LA not downloaded")


@pytest.fixture(scope="module")
def world():
    from certflow.realworld import TrafficWorld

    return TrafficWorld(seed=0, n_bins=144)  # half a day


def test_graph_and_costs_sane(world):
    n_edges = sum(len(nbrs) for nbrs in world.graph.values())
    assert n_edges > 500
    for e in list(world.edges())[:50]:
        c0, c1 = world.true_cost(e, 0.0), world.true_cost(e, 3600.0)
        assert 0 < c0 < 3600 and 0 < c1 < 3600  # seconds, sane travel times
    # piecewise-linear continuity
    e = next(world.edges())
    assert abs(world.true_cost(e, 299.0) - world.true_cost(e, 301.0)) < 5.0


def test_determinism_and_noise(world):
    from certflow.realworld import TrafficWorld

    w2 = TrafficWorld(seed=0, n_bins=144)
    e = next(world.edges())
    assert world.true_cost(e, 1234.0) == w2.true_cost(e, 1234.0)
    obs = [world.observe(e, 100.0) for _ in range(200)]
    truth = world.true_cost(e, 100.0)
    assert abs(sum(obs) / len(obs) - truth) < 2.0  # mean ~ truth (scale 5s)


def test_a1_violation_rate_measured(world):
    # rho is the p95 of |dc/dt|: violations exist by construction (~5%)
    assert 0.0 < world.a1_violation_rate < 0.15


def test_planner_runs_on_traffic(world):
    from certflow.cert import CertPlanner
    from certflow.realworld import far_endpoints, traffic_planner_config

    s, g = far_endpoints(world)
    p = CertPlanner(world, s, g, traffic_planner_config())
    valid = 0
    for _ in range(60):
        cert, _ = p.round()
        valid += cert.valid
    assert valid > 10  # annealing: valid well before full warm-up
    assert p.sense_spend > 0
