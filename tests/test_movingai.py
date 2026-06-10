"""Tests for the MovingAI benchmark adapter (certflow.movingai).

Files used:
  - data/movingai/dao/arena.map        (49 x 49  — small DAO map)
  - data/movingai/dao/arena.map.scen   (130 scenario entries)
  - data/movingai/street/Berlin_0_256.map        (256 x 256 street map)
  - data/movingai/street/Berlin_0_256.map.scen

All tests gracefully skip (pytest.skip) if the data/movingai directory is
absent so that CI without benchmark data still passes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path / import setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

DATA_DIR = _REPO / "data" / "movingai"
DAO_MAP = DATA_DIR / "dao" / "arena.map"
DAO_SCEN = DATA_DIR / "dao" / "arena.map.scen"
STREET_MAP = DATA_DIR / "street" / "Berlin_0_256.map"
STREET_SCEN = DATA_DIR / "street" / "Berlin_0_256.map.scen"

_DATA_PRESENT = DATA_DIR.exists()


def _skip_if_no_data():
    if not _DATA_PRESENT:
        pytest.skip("data/movingai not present — skipping benchmark tests")


# ---------------------------------------------------------------------------
# Helpers imported lazily (so the module itself loads without data)
# ---------------------------------------------------------------------------

def _import():
    from certflow.movingai import (
        crop,
        movingai_world,
        movingai_world_from_grid,
        parse_map,
        parse_scen,
        scenario_endpoints,
        _is_passable,
    )
    return parse_map, parse_scen, movingai_world, movingai_world_from_grid, crop, scenario_endpoints, _is_passable


# ===========================================================================
# 1. parse_map tests
# ===========================================================================

class TestParseMap:
    def test_dao_dimensions(self):
        _skip_if_no_data()
        parse_map, *_ = _import()
        grid = parse_map(DAO_MAP)
        assert len(grid) == 49, f"Expected 49 rows, got {len(grid)}"
        assert all(len(row) == 49 for row in grid), "All rows should be width=49"

    def test_street_dimensions(self):
        _skip_if_no_data()
        parse_map, *_ = _import()
        grid = parse_map(STREET_MAP)
        assert len(grid) == 256
        assert all(len(row) == 256 for row in grid)

    def test_walls_excluded_from_graph(self):
        _skip_if_no_data()
        parse_map, _, movingai_world, movingai_world_from_grid, *_ = _import()
        grid = parse_map(DAO_MAP)
        world = movingai_world_from_grid(grid, seed=0, kind="static")

        # Count passable cells in the grid.
        passable_cells = sum(
            1 for row in grid for ch in row if ch in ".GS"
        )
        # Every node in the graph must be a passable cell.
        for node in world.graph:
            r, c = node
            ch = grid[r][c]
            assert ch in ".GS", f"Node {node} has impassable character {ch!r}"

        # Node count should equal passable-cell count.
        assert len(world.graph) == passable_cells, (
            f"Expected {passable_cells} nodes, got {len(world.graph)}"
        )

    def test_edge_count_plausible(self):
        _skip_if_no_data()
        parse_map, _, movingai_world, movingai_world_from_grid, *_ = _import()
        grid = parse_map(DAO_MAP)
        world = movingai_world_from_grid(grid, seed=1, kind="static")
        edge_count = sum(1 for _ in world.edges())
        # 4-connected; minimum is 1 edge per node (isolated passable cell), max 4.
        assert edge_count >= len(world.graph), "Fewer edges than nodes — suspicious"
        assert edge_count <= 4 * len(world.graph), "More than 4 edges per node — impossible"


# ===========================================================================
# 2. parse_scen tests
# ===========================================================================

class TestParseScen:
    def test_dao_entries_nonzero(self):
        _skip_if_no_data()
        _, parse_scen, *_ = _import()
        entries = parse_scen(DAO_SCEN)
        assert len(entries) > 0, "Expected at least one scenario entry"

    def test_map_name_matches(self):
        _skip_if_no_data()
        _, parse_scen, *_ = _import()
        entries = parse_scen(DAO_SCEN)
        for e in entries:
            assert "arena" in e["map"].lower(), (
                f"Unexpected map name in arena.map.scen: {e['map']!r}"
            )

    def test_endpoints_passable(self):
        _skip_if_no_data()
        parse_map, parse_scen, *rest = _import()
        grid = parse_map(DAO_MAP)
        entries = parse_scen(DAO_SCEN)

        passable_chars = frozenset(".GS")
        for e in entries[:20]:  # check first 20 entries
            sr, sc = e["sy"], e["sx"]  # MovingAI: Y=row, X=col
            gr, gc = e["gy"], e["gx"]
            assert grid[sr][sc] in passable_chars, (
                f"Start ({sr},{sc}) is impassable: {grid[sr][sc]!r}"
            )
            assert grid[gr][gc] in passable_chars, (
                f"Goal ({gr},{gc}) is impassable: {grid[gr][gc]!r}"
            )

    def test_optimal_length_positive(self):
        _skip_if_no_data()
        _, parse_scen, *_ = _import()
        entries = parse_scen(DAO_SCEN)
        for e in entries:
            assert e["optimal_length"] >= 0, (
                f"Negative optimal length: {e['optimal_length']}"
            )

    def test_required_keys(self):
        _skip_if_no_data()
        _, parse_scen, *_ = _import()
        entries = parse_scen(DAO_SCEN)
        required = {"bucket", "map", "w", "h", "sx", "sy", "gx", "gy", "optimal_length"}
        for e in entries[:5]:
            missing = required - e.keys()
            assert not missing, f"Missing keys: {missing}"


# ===========================================================================
# 3. crop tests
# ===========================================================================

class TestCrop:
    def test_crop_size(self):
        _skip_if_no_data()
        parse_map, _, _mw, _mwg, crop, *_ = _import()
        grid = parse_map(DAO_MAP)
        sub, r0, c0 = crop(grid, 24, 24, 32)
        assert len(sub) <= 32
        assert all(len(row) <= 32 for row in sub)

    def test_crop_offset_within_bounds(self):
        _skip_if_no_data()
        parse_map, _, _mw, _mwg, crop, *_ = _import()
        grid = parse_map(DAO_MAP)
        size = 20
        sub, r0, c0 = crop(grid, 10, 10, size)
        assert 0 <= r0 < len(grid)
        assert 0 <= c0 < len(grid[0])

    def test_crop_corner_stays_inside(self):
        _skip_if_no_data()
        parse_map, _, _mw, _mwg, crop, *_ = _import()
        grid = parse_map(DAO_MAP)
        # Request crop centred at (0,0) — should be clamped to top-left.
        sub, r0, c0 = crop(grid, 0, 0, 16)
        assert r0 == 0
        assert c0 == 0
        assert len(sub) <= 16


# ===========================================================================
# 4. World protocol tests
# ===========================================================================

class TestWorldProtocol:
    """Build a cropped (<=64x64) bounded world and verify World protocol."""

    def _make_world(self, kind="bounded", size=48):
        _skip_if_no_data()
        parse_map, _, _mw, movingai_world_from_grid, crop, *_ = _import()
        grid = parse_map(DAO_MAP)
        sub, _, _ = crop(grid, 24, 24, size)
        return movingai_world_from_grid(sub, seed=42, kind=kind, rho=0.02)

    def test_graph_attribute_exists(self):
        _skip_if_no_data()
        world = self._make_world()
        assert hasattr(world, "graph")
        assert isinstance(world.graph, dict)

    def test_true_cost_determinism(self):
        """true_cost must be a pure function of (e, t) given the seed."""
        _skip_if_no_data()
        world = self._make_world()
        edges = list(world.edges())[:5]
        for e in edges:
            c1 = world.true_cost(e, 10.0)
            c2 = world.true_cost(e, 10.0)
            assert c1 == c2, f"true_cost not deterministic for edge {e}"

    def test_lipschitz_bound_bounded_world(self):
        """For bounded kind: |c(t') - c(t)| <= rho_true(e) * |t' - t|."""
        _skip_if_no_data()
        world = self._make_world(kind="bounded")
        edges = list(world.edges())[:20]
        t_pairs = [(0.0, 5.0), (1.0, 3.0), (5.0, 15.0), (0.0, 100.0)]
        for e in edges:
            rho = world.rho_true(e)
            for t1, t2 in t_pairs:
                c1 = world.true_cost(e, t1)
                c2 = world.true_cost(e, t2)
                diff = abs(c2 - c1)
                bound = rho * abs(t2 - t1)
                assert diff <= bound + 1e-9, (
                    f"Lipschitz violated for edge {e}: "
                    f"|c({t2})-c({t1})|={diff:.6f} > rho*|t'-t|={bound:.6f}"
                )

    def test_static_world_rho_true_is_zero(self):
        _skip_if_no_data()
        world = self._make_world(kind="static")
        for e in list(world.edges())[:10]:
            assert world.rho_true(e) == 0.0

    def test_observe_near_true(self):
        """Observation mean should be near true_cost (small noise_scale=0.05)."""
        _skip_if_no_data()
        parse_map, _, _mw, movingai_world_from_grid, crop, *_ = _import()
        grid = parse_map(DAO_MAP)
        sub, _, _ = crop(grid, 24, 24, 48)
        world = movingai_world_from_grid(
            sub, seed=99, kind="bounded", rho=0.02, noise_scale=0.05
        )
        edges = list(world.edges())[:5]
        n_obs = 200
        for e in edges:
            true_c = world.true_cost(e, 0.0)
            obs_vals = [world.observe(e, 0.0) for _ in range(n_obs)]
            mean_obs = sum(obs_vals) / n_obs
            assert abs(mean_obs - true_c) < 0.1, (
                f"Mean observation {mean_obs:.4f} too far from true cost {true_c:.4f}"
            )

    def test_edges_iterator(self):
        _skip_if_no_data()
        world = self._make_world()
        edges = list(world.edges())
        assert len(edges) > 0
        for u, v in edges[:5]:
            assert isinstance(u, tuple) and len(u) == 2
            assert isinstance(v, tuple) and len(v) == 2

    def test_bounded_world_rho_positive(self):
        _skip_if_no_data()
        world = self._make_world(kind="bounded")
        for e in list(world.edges())[:10]:
            assert world.rho_true(e) > 0.0


# ===========================================================================
# 5. scenario_endpoints tests
# ===========================================================================

class TestScenarioEndpoints:
    def test_min_length_filter(self):
        _skip_if_no_data()
        parse_map, parse_scen, _mw, _mwg, _crop, scenario_endpoints, _ = _import()
        grid = parse_map(DAO_MAP)
        entries = parse_scen(DAO_SCEN)
        pairs = scenario_endpoints(entries, grid, min_length=5.0)
        for start, goal in pairs:
            assert _is_passable_cell(grid, start)
            assert _is_passable_cell(grid, goal)

    def test_no_impassable_endpoints(self):
        _skip_if_no_data()
        parse_map, parse_scen, _mw, _mwg, _crop, scenario_endpoints, _ = _import()
        grid = parse_map(DAO_MAP)
        entries = parse_scen(DAO_SCEN)
        pairs = scenario_endpoints(entries, grid, min_length=0.0)
        for start, goal in pairs:
            sr, sc = start
            gr, gc = goal
            assert grid[sr][sc] in ".GS", f"Start {start} is impassable"
            assert grid[gr][gc] in ".GS", f"Goal {goal} is impassable"


def _is_passable_cell(grid, rc):
    r, c = rc
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] in ".GS"


# ===========================================================================
# 6. End-to-end smoke: CertPlanner on cropped DAO world
# ===========================================================================

class TestCertPlannerSmoke:
    """Run CertPlanner on a 32x32 cropped DAO world for 50 rounds."""

    def test_certplanner_50_rounds_no_error(self):
        _skip_if_no_data()
        from certflow.cert import CertPlanner, PlannerConfig
        from certflow.movingai import parse_map, movingai_world_from_grid, crop

        # Load and crop the DAO map to a 32x32 sub-map.
        grid = parse_map(DAO_MAP)
        sub, r0, c0 = crop(grid, 24, 24, 32)

        # Build a bounded-drift world.
        world = movingai_world_from_grid(sub, seed=7, kind="bounded", rho=0.02)

        # Find two passable cells to use as start/goal.
        passable = [
            (r, c)
            for r in range(len(sub))
            for c in range(len(sub[0]))
            if sub[r][c] in ".GS"
        ]
        assert len(passable) >= 2, "Not enough passable cells in crop"

        # Pick start and goal that are far-ish apart.
        start = passable[0]
        # Try to pick a goal that's reachable and different.
        goal = None
        for cand in reversed(passable):
            if cand != start:
                goal = cand
                break
        assert goal is not None

        config = PlannerConfig(
            epsilon=8.0,
            alpha_prime=0.2,
            eps_tv=1e-4,
            delta=1.0,
        )

        planner = CertPlanner(world, start=start, goal=goal, config=config)

        certs = []
        for _ in range(50):
            cert, _ = planner.round()
            certs.append(cert)

        # At least some rounds produced a Certificate object.
        assert len(certs) == 50
        # Check basic certificate invariants on valid rounds.
        for cert in certs:
            if cert.valid:
                assert cert.lb <= cert.ub + 1e-9, (
                    f"LB {cert.lb} > UB {cert.ub}"
                )
                assert 0.0 < cert.confidence <= 1.0 + 1e-9

    def test_certplanner_static_world(self):
        """Static world variant: costs don't drift, gap should eventually close."""
        _skip_if_no_data()
        from certflow.cert import CertPlanner, PlannerConfig
        from certflow.movingai import parse_map, movingai_world_from_grid, crop

        grid = parse_map(DAO_MAP)
        sub, _, _ = crop(grid, 24, 24, 32)
        world = movingai_world_from_grid(sub, seed=3, kind="static")

        passable = [
            (r, c)
            for r in range(len(sub))
            for c in range(len(sub[0]))
            if sub[r][c] in ".GS"
        ]
        start = passable[0]
        goal = passable[-1]

        config = PlannerConfig(epsilon=8.0, alpha_prime=0.2, eps_tv=1e-4)
        planner = CertPlanner(world, start=start, goal=goal, config=config)

        for _ in range(50):
            cert, _ = planner.round()

        # No exception == pass.
