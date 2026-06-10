"""Smoke tests for the Tier-1 replanning-latency benchmark (Theorem T3).

Runs a tiny benchmark (10x10, radius 1, 10 rounds) and asserts:
  1. incremental cost == scratch cost every round (correctness under sustained
     incremental use — no accumulated state drift).
  2. The result dict has all expected keys.
  3. The moving-start scenario also produces the expected keys.

These tests exercise the importable core of scripts/run_tier1_latency.py;
they do NOT import the __main__ block.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow import from both installed and in-tree layouts.
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo / "scripts"))

from run_tier1_latency import run_benchmark, run_moving_benchmark, compute_row  # noqa: E402

EXPECTED_BENCHMARK_KEYS = {
    "size",
    "radius",
    "rounds",
    "node_count",
    "inc_latencies_ms",
    "scratch_latencies_ms",
    "inc_pops",
    "cost_mismatches",
}

EXPECTED_ROW_KEYS = {
    "scenario",
    "size",
    "radius",
    "rounds",
    "node_count",
    "inc_p50_ms",
    "inc_p95_ms",
    "scr_p50_ms",
    "scr_p95_ms",
    "speedup_p50",
    "mean_pops",
    "cost_mismatches",
}


class TestBenchmarkKeys:
    def test_result_has_expected_keys(self):
        result = run_benchmark(rows=10, cols=10, radius=1, n_rounds=10, seed=7)
        assert EXPECTED_BENCHMARK_KEYS.issubset(result.keys()), (
            f"Missing keys: {EXPECTED_BENCHMARK_KEYS - result.keys()}"
        )

    def test_row_has_expected_keys(self):
        result = run_benchmark(rows=10, cols=10, radius=1, n_rounds=10, seed=7)
        row = compute_row(result, scenario="static")
        assert EXPECTED_ROW_KEYS.issubset(row.keys()), (
            f"Missing keys: {EXPECTED_ROW_KEYS - row.keys()}"
        )

    def test_moving_result_has_expected_keys(self):
        result = run_moving_benchmark(rows=10, cols=10, n_steps=10, seed=7)
        assert EXPECTED_BENCHMARK_KEYS.issubset(result.keys()), (
            f"Missing keys: {EXPECTED_BENCHMARK_KEYS - result.keys()}"
        )
        row = compute_row(result, scenario="moving")
        assert EXPECTED_ROW_KEYS.issubset(row.keys()), (
            f"Missing keys: {EXPECTED_ROW_KEYS - row.keys()}"
        )


class TestCorrectnessEveryRound:
    """Verify incremental cost == scratch cost across all rounds (Theorem T3 sanity)."""

    def test_no_cost_mismatches_radius1(self):
        result = run_benchmark(rows=10, cols=10, radius=1, n_rounds=10, seed=42)
        mismatches = result["cost_mismatches"]
        assert mismatches == [], (
            f"Cost mismatches on rounds {mismatches} — possible graphcore bug!\n"
            "Incremental and scratch Dijkstra should agree on every round."
        )

    def test_no_cost_mismatches_radius2(self):
        result = run_benchmark(rows=10, cols=10, radius=2, n_rounds=10, seed=99)
        mismatches = result["cost_mismatches"]
        assert mismatches == [], (
            f"Cost mismatches on rounds {mismatches} — possible graphcore bug!"
        )

    def test_no_cost_mismatches_moving(self):
        result = run_moving_benchmark(rows=10, cols=10, n_steps=10, seed=42)
        mismatches = result["cost_mismatches"]
        assert mismatches == [], (
            f"Moving-start cost mismatches on steps {mismatches} — possible graphcore bug!"
        )


class TestResultSanity:
    """Basic sanity on numeric outputs."""

    def test_rounds_count_matches(self):
        n_rounds = 10
        result = run_benchmark(rows=10, cols=10, radius=1, n_rounds=n_rounds, seed=1)
        assert result["rounds"] == n_rounds
        assert len(result["inc_latencies_ms"]) == n_rounds
        assert len(result["scratch_latencies_ms"]) == n_rounds
        assert len(result["inc_pops"]) == n_rounds

    def test_latencies_are_non_negative(self):
        result = run_benchmark(rows=10, cols=10, radius=1, n_rounds=10, seed=3)
        assert all(v >= 0 for v in result["inc_latencies_ms"])
        assert all(v >= 0 for v in result["scratch_latencies_ms"])

    def test_pops_are_non_negative(self):
        result = run_benchmark(rows=10, cols=10, radius=1, n_rounds=10, seed=5)
        assert all(p >= 0 for p in result["inc_pops"])

    def test_node_count(self):
        result = run_benchmark(rows=10, cols=10, radius=1, n_rounds=5, seed=2)
        assert result["node_count"] == 100

    def test_speedup_positive(self):
        result = run_benchmark(rows=10, cols=10, radius=1, n_rounds=10, seed=6)
        row = compute_row(result, scenario="static")
        # speedup can be < 1 for tiny grids but must be finite and positive.
        assert row["speedup_p50"] > 0
