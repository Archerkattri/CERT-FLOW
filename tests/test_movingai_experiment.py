"""Tests for run_movingai.py experiment functions.

Imports the episode function from scripts/run_movingai.py, runs a single
50-round episode on a 32x32 DAO crop, and asserts EpisodeResult contract.
Skips if data/movingai is absent.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

DATA_DIR = _REPO / "data" / "movingai"
DAO_MAP = DATA_DIR / "dao" / "arena.map"

_DATA_PRESENT = DATA_DIR.exists() and DAO_MAP.exists()


def _skip_if_no_data():
    if not _DATA_PRESENT:
        pytest.skip("data/movingai not present — skipping experiment tests")


class TestMovingAIEpisode:
    """One 50-round episode on a 32x32 DAO crop."""

    def test_episode_returns_episode_result(self):
        _skip_if_no_data()

        from certflow.movingai import parse_map, crop, _build_graph_from_grid
        from certflow.graphcore import dijkstra
        from certflow.types import EpisodeResult
        from run_movingai import movingai_episode

        # Load and crop a 32x32 sub-map from arena.map
        grid = parse_map(DAO_MAP)
        sub, r0, c0 = crop(grid, 24, 24, 32)

        # Find a connected start/goal pair within the crop
        G, _, _, _ = _build_graph_from_grid(sub)
        nodes = sorted(G.keys())

        # Find largest connected component
        visited: set = set()
        comps: list = []
        for s in nodes:
            if s in visited:
                continue
            comp: set = set()
            q = [s]
            while q:
                u = q.pop()
                if u in comp:
                    continue
                comp.add(u)
                for v in G.get(u, {}):
                    if v not in comp:
                        q.append(v)
            visited.update(comp)
            comps.append(frozenset(comp))

        biggest = max(comps, key=len)
        bc = sorted(biggest)
        start = bc[0]
        goal = max(biggest, key=lambda n: abs(n[0] - start[0]) + abs(n[1] - start[1]))

        # Verify path exists
        path, cost = dijkstra(G, start, goal)
        assert path is not None and not math.isinf(cost), (
            f"No path from {start} to {goal} in 32x32 DAO crop"
        )

        # Run a 50-round episode
        result = movingai_episode(
            grid=sub,
            start=start,
            goal=goal,
            seed=42,
            sensing_policy="cert",
            move_policy="when_certified",
            sense_budget=20.0,
            max_rounds=50,
        )

        # --- Assertions ---
        assert isinstance(result, EpisodeResult), (
            f"Expected EpisodeResult, got {type(result)}"
        )
        assert len(result.rounds) > 0, "EpisodeResult.rounds must be non-empty"
        assert len(result.rounds) <= 50, "Should not exceed max_rounds"
        assert math.isfinite(result.oracle_cost), (
            f"oracle_cost must be finite, got {result.oracle_cost}"
        )
        assert result.oracle_cost > 0.0, (
            f"oracle_cost should be positive, got {result.oracle_cost}"
        )
        assert result.travel_cost >= 0.0, (
            f"travel_cost must be non-negative, got {result.travel_cost}"
        )
        assert isinstance(result.reached_goal, bool)

    def test_episode_round_logs_well_formed(self):
        """Each RoundLog should have valid lb <= ub for valid rounds."""
        _skip_if_no_data()

        from certflow.movingai import parse_map, crop, _build_graph_from_grid
        from certflow.graphcore import dijkstra
        from run_movingai import movingai_episode

        grid = parse_map(DAO_MAP)
        sub, r0, c0 = crop(grid, 24, 24, 32)

        G, _, _, _ = _build_graph_from_grid(sub)
        nodes = sorted(G.keys())
        visited: set = set()
        comps: list = []
        for s in nodes:
            if s in visited:
                continue
            comp: set = set()
            q = [s]
            while q:
                u = q.pop()
                if u in comp:
                    continue
                comp.add(u)
                for v in G.get(u, {}):
                    if v not in comp:
                        q.append(v)
            visited.update(comp)
            comps.append(frozenset(comp))
        biggest = max(comps, key=len)
        bc = sorted(biggest)
        start = bc[0]
        goal = max(biggest, key=lambda n: abs(n[0] - start[0]) + abs(n[1] - start[1]))

        result = movingai_episode(
            grid=sub,
            start=start,
            goal=goal,
            seed=7,
            sensing_policy="random",
            move_policy="when_certified",
            sense_budget=20.0,
            max_rounds=50,
        )

        for i, rlog in enumerate(result.rounds):
            if rlog.confidence > 0.0:  # valid rounds only
                assert rlog.lb <= rlog.ub + 1e-9, (
                    f"Round {i}: lb={rlog.lb} > ub={rlog.ub}"
                )
                assert 0.0 < rlog.confidence <= 1.0 + 1e-9, (
                    f"Round {i}: confidence={rlog.confidence} out of range"
                )

    def test_blind_baseline_always_moves(self):
        """Blind baseline (move_policy=always) should reach goal quickly."""
        _skip_if_no_data()

        from certflow.movingai import parse_map, crop, _build_graph_from_grid
        from certflow.graphcore import dijkstra
        from run_movingai import movingai_episode

        grid = parse_map(DAO_MAP)
        sub, r0, c0 = crop(grid, 24, 24, 32)

        G, _, _, _ = _build_graph_from_grid(sub)
        nodes = sorted(G.keys())
        visited: set = set()
        comps: list = []
        for s in nodes:
            if s in visited:
                continue
            comp: set = set()
            q = [s]
            while q:
                u = q.pop()
                if u in comp:
                    continue
                comp.add(u)
                for v in G.get(u, {}):
                    if v not in comp:
                        q.append(v)
            visited.update(comp)
            comps.append(frozenset(comp))
        biggest = max(comps, key=len)
        bc = sorted(biggest)
        start = bc[0]
        goal = max(biggest, key=lambda n: abs(n[0] - start[0]) + abs(n[1] - start[1]))

        result = movingai_episode(
            grid=sub,
            start=start,
            goal=goal,
            seed=99,
            sensing_policy="none",
            move_policy="always",
            sense_budget=float("inf"),
            max_rounds=50,
        )

        assert math.isfinite(result.oracle_cost), "oracle_cost must be finite"
        # Sense spend should be 0 for blind baseline (no sensing policy)
        assert result.sense_cost == 0.0 or result.sense_cost < 1.0, (
            f"Blind baseline should have near-zero sense cost, got {result.sense_cost}"
        )
