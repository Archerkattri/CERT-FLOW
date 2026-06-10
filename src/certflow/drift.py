"""Ground-truth world simulators for CERT experiments.

Implements the ``World`` protocol from ``certflow.types``.  All worlds are
deterministic given a seed (``numpy.random.Generator``): ``true_cost`` is a
pure function of ``(e, t)`` once the world is constructed; ``observe`` draws
from the world's internal RNG so successive calls differ.

Usage::

    world = grid_world(10, 10, seed=42, kind="bounded", rho=0.05)
    cost  = world.true_cost(((0, 0), (0, 1)), 3.7)
    obs   = world.observe(((0, 0), (0, 1)), 3.7)
"""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from numpy.random import Generator

from certflow.types import Edge, Node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COST_FLOOR: float = 1e-6
_COST_CAP: float = 1e4


def _make_rng(seed: int | Generator) -> Generator:
    if isinstance(seed, Generator):
        return seed
    return np.random.default_rng(seed)


def _lognormal_costs(
    rng: Generator, n: int, sigma: float = 0.5
) -> np.ndarray:
    """Log-normal samples with median ~1.0 (mu=0 in log-space)."""
    return rng.lognormal(mean=0.0, sigma=sigma, size=n)


# ---------------------------------------------------------------------------
# World base utilities
# ---------------------------------------------------------------------------

class _GridBase:
    """Shared graph structure and edge enumeration for grid worlds."""

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: Generator,
        sigma: float = 0.5,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._noise_family = noise_family
        self._noise_scale = noise_scale
        self._rng = rng  # shared RNG for observe()

        # Build directed graph: 4-connected, both directions independent.
        edge_list: list[Edge] = []
        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    edge_list.append(((r, c), (r, c + 1)))
                    edge_list.append(((r, c + 1), (r, c)))
                if r + 1 < rows:
                    edge_list.append(((r, c), (r + 1, c)))
                    edge_list.append(((r + 1, c), (r, c)))

        self._edge_list: list[Edge] = edge_list
        self._edge_index: dict[Edge, int] = {e: i for i, e in enumerate(edge_list)}
        n_edges = len(edge_list)

        # Draw initial costs (log-normal, median ~1.0).
        init_costs = _lognormal_costs(rng, n_edges, sigma=sigma)

        # Build adjacency dict (initial true costs).
        self.graph: dict[Node, dict[Node, float]] = {}
        for i, (u, v) in enumerate(edge_list):
            self.graph.setdefault(u, {})[v] = float(init_costs[i])

    # ------------------------------------------------------------------
    def edges(self) -> Iterator[Edge]:
        return iter(self._edge_list)

    # ------------------------------------------------------------------
    def _edge_idx(self, e: Edge) -> int:
        return self._edge_index[e]

    # ------------------------------------------------------------------
    def _draw_noise(self) -> float:
        family = self._noise_family
        s = self._noise_scale
        if family == "gaussian":
            return float(self._rng.normal(0.0, s))
        elif family == "laplace":
            return float(self._rng.laplace(0.0, s))
        elif family == "student_t":
            return float(self._rng.standard_t(df=3) * s)
        elif family == "skewed":
            # centered lognormal: mean 0, right-skewed, sd ~ s. Violates the
            # planner's A3 (symmetry) by design — the Gaussian-break stressor.
            z = float(self._rng.normal(0.0, 1.0))
            raw = math.exp(z) - math.exp(0.5)
            return raw * s / math.sqrt((math.e - 1.0) * math.e)
        else:
            raise ValueError(f"Unknown noise family: {family!r}")

    # ------------------------------------------------------------------
    def observe(self, e: Edge, t: float) -> float:
        """Noisy observation: true_cost(e, t) + additive noise."""
        return self.true_cost(e, t) + self._draw_noise()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def rho_true(self, e: Edge) -> float:  # noqa: ARG002
        raise NotImplementedError


# ---------------------------------------------------------------------------
# StaticWorld
# ---------------------------------------------------------------------------

class StaticWorld(_GridBase):
    """Costs never change.  rho_true = 0."""

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: Generator,
        sigma: float = 0.5,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
    ) -> None:
        super().__init__(rows, cols, rng, sigma, noise_family, noise_scale)
        # Store costs as 1-D array indexed by edge index.
        n = len(self._edge_list)
        self._costs: np.ndarray = np.array(
            [self.graph[u][v] for u, v in self._edge_list], dtype=float
        )
        # Rebuild graph from _costs to be consistent.
        for i, (u, v) in enumerate(self._edge_list):
            self.graph[u][v] = float(self._costs[i])

    def true_cost(self, e: Edge, t: float) -> float:  # noqa: ARG002
        return float(self._costs[self._edge_idx(e)])

    def rho_true(self, e: Edge) -> float:  # noqa: ARG002
        return 0.0


# ---------------------------------------------------------------------------
# BoundedDriftWorld
# ---------------------------------------------------------------------------

class BoundedDriftWorld(_GridBase):
    """Piecewise-linear rate-limited random walk per edge.

    Each edge cost follows a trajectory whose slope is resampled in
    ``[-rho_e, +rho_e]`` at Poisson-arrival times (exponential inter-arrivals).
    The trajectory is reflected at ``[cost_floor, cost_cap]`` so that the
    Lipschitz bound ``|c(t') - c(t)| <= rho_e |t' - t|`` holds exactly.

    Per-edge ``rho_e = rho * U(0.5, 1.5)`` drawn once at construction.

    Trajectories are precomputed lazily and cached per edge to guarantee that
    repeated calls with the same ``(e, t)`` always return the same value.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: Generator,
        rho: float = 0.1,
        resampling_rate: float = 0.5,
        sigma: float = 0.5,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
        cost_floor: float = _COST_FLOOR,
        cost_cap: float = _COST_CAP,
        max_t: float = 1000.0,
    ) -> None:
        super().__init__(rows, cols, rng, sigma, noise_family, noise_scale)
        self._rho_global = rho
        self._resampling_rate = resampling_rate
        self._cost_floor = cost_floor
        self._cost_cap = cost_cap
        self._max_t = max_t

        n = len(self._edge_list)
        # Per-edge rho: rho * U(0.5, 1.5)
        self._rho_e: np.ndarray = rho * rng.uniform(0.5, 1.5, size=n)

        # Initial costs from parent graph.
        self._c0: np.ndarray = np.array(
            [self.graph[u][v] for u, v in self._edge_list], dtype=float
        )

        # Precompute trajectory breakpoints for each edge.
        # Each entry: (times, slopes) arrays describing the piecewise-linear path.
        self._trajectories: list[tuple[np.ndarray, np.ndarray] | None] = [
            None
        ] * n

        # Use a *separate* per-edge RNG seeded from the main rng (deterministic).
        # We draw all per-edge seeds now so the determinism contract is satisfied
        # regardless of the order in which edges are first queried.
        seeds = rng.integers(0, 2**31, size=n)
        self._edge_rngs: list[Generator] = [
            np.random.default_rng(int(s)) for s in seeds
        ]

    # ------------------------------------------------------------------
    def _build_trajectory(self, idx: int) -> None:
        """Precompute the full piecewise-linear trajectory for edge idx."""
        rho_e = float(self._rho_e[idx])
        c0 = float(self._c0[idx])
        rate = self._resampling_rate
        erng = self._edge_rngs[idx]
        floor = self._cost_floor
        cap = self._cost_cap

        # Generate Poisson breakpoints up to max_t.
        times = [0.0]
        t_cur = 0.0
        while t_cur < self._max_t:
            dt = erng.exponential(1.0 / rate)
            t_cur += dt
            times.append(t_cur)
        times_arr = np.array(times, dtype=float)

        # At each breakpoint, draw a new slope in [-rho_e, +rho_e].
        n_segs = len(times_arr)
        slopes = erng.uniform(-rho_e, rho_e, size=n_segs)

        # Build cost values at each breakpoint using reflection.
        values = np.empty(n_segs, dtype=float)
        values[0] = np.clip(c0, floor, cap)
        for k in range(1, n_segs):
            dt = times_arr[k] - times_arr[k - 1]
            proposed = values[k - 1] + slopes[k - 1] * dt
            # Reflect at boundaries.
            v = proposed
            while True:
                if v < floor:
                    v = 2 * floor - v
                    slopes[k - 1] = abs(slopes[k - 1])
                elif v > cap:
                    v = 2 * cap - v
                    slopes[k - 1] = -abs(slopes[k - 1])
                else:
                    break
            values[k] = v

        self._trajectories[idx] = (times_arr, values)

    # ------------------------------------------------------------------
    def _eval_trajectory(self, idx: int, t: float) -> float:
        """Evaluate the piecewise-linear trajectory at time t."""
        if self._trajectories[idx] is None:
            self._build_trajectory(idx)
        times, values = self._trajectories[idx]  # type: ignore[misc]

        if t <= 0.0:
            return float(values[0])

        # Find the segment: largest k with times[k] <= t.
        k = int(np.searchsorted(times, t, side="right")) - 1
        k = min(k, len(times) - 2)  # clamp to last full segment

        t0 = times[k]
        v0 = values[k]
        # Slope for the segment starting at t0.
        # After reflection the effective slope for value evolution from t0
        # is (values[k+1] - values[k]) / (times[k+1] - times[k]).
        dt_seg = times[k + 1] - t0
        if dt_seg < 1e-15:
            return float(v0)
        slope = (values[k + 1] - values[k]) / dt_seg
        raw = v0 + slope * (t - t0)
        return float(np.clip(raw, self._cost_floor, self._cost_cap))

    # ------------------------------------------------------------------
    def true_cost(self, e: Edge, t: float) -> float:
        idx = self._edge_idx(e)
        return self._eval_trajectory(idx, t)

    def rho_true(self, e: Edge) -> float:
        return float(self._rho_e[self._edge_idx(e)])


# ---------------------------------------------------------------------------
# JumpWorld
# ---------------------------------------------------------------------------

class JumpWorld(_GridBase):
    """Poisson jumps per edge; cost multiplies by a random factor.

    Off-model stress: violates A1 (unbounded instantaneous derivative).
    ``rho_true`` returns ``inf`` to document model misspecification.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: Generator,
        jump_rate: float = 0.1,
        jump_scale: float = 0.5,
        sigma: float = 0.5,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
        max_t: float = 1000.0,
    ) -> None:
        super().__init__(rows, cols, rng, sigma, noise_family, noise_scale)
        self._jump_rate = jump_rate
        self._jump_scale = jump_scale
        self._max_t = max_t

        n = len(self._edge_list)
        self._c0: np.ndarray = np.array(
            [self.graph[u][v] for u, v in self._edge_list], dtype=float
        )
        self._trajectories: list[tuple[np.ndarray, np.ndarray] | None] = [
            None
        ] * n
        seeds = rng.integers(0, 2**31, size=n)
        self._edge_rngs: list[Generator] = [
            np.random.default_rng(int(s)) for s in seeds
        ]

    # ------------------------------------------------------------------
    def _build_trajectory(self, idx: int) -> None:
        erng = self._edge_rngs[idx]
        c0 = float(self._c0[idx])
        rate = self._jump_rate
        scale = self._jump_scale

        # Poisson jump times.
        times = [0.0]
        t_cur = 0.0
        while t_cur < self._max_t:
            t_cur += erng.exponential(1.0 / rate)
            times.append(t_cur)
        times_arr = np.array(times, dtype=float)

        # Values: jump at each breakpoint by multiplying by lognormal factor.
        n_pts = len(times_arr)
        factors = erng.lognormal(mean=0.0, sigma=scale, size=n_pts)
        values = np.empty(n_pts, dtype=float)
        values[0] = c0
        for k in range(1, n_pts):
            values[k] = max(_COST_FLOOR, values[k - 1] * factors[k])

        self._trajectories[idx] = (times_arr, values)

    # ------------------------------------------------------------------
    def true_cost(self, e: Edge, t: float) -> float:
        idx = self._edge_idx(e)
        if self._trajectories[idx] is None:
            self._build_trajectory(idx)
        times, values = self._trajectories[idx]  # type: ignore[misc]
        # Piecewise-constant: use value at the last jump before t.
        k = int(np.searchsorted(times, t, side="right")) - 1
        k = max(0, min(k, len(values) - 1))
        return float(values[k])

    def rho_true(self, e: Edge) -> float:  # noqa: ARG002
        return math.inf


# ---------------------------------------------------------------------------
# PeriodicWorld
# ---------------------------------------------------------------------------

class PeriodicWorld(_GridBase):
    """Sinusoidal modulation per edge with random phase.

    ``c_e(t) = c0_e * (1 + amplitude * sin(2π t / period + phase_e))``.
    Off-model stress (FreMEn-style): not a bounded-rate random walk.
    ``rho_true`` returns the maximum instantaneous rate as a documented surrogate.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: Generator,
        period: float = 60.0,
        amplitude: float = 0.3,
        sigma: float = 0.5,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
    ) -> None:
        super().__init__(rows, cols, rng, sigma, noise_family, noise_scale)
        self._period = period
        self._amplitude = amplitude

        n = len(self._edge_list)
        self._c0: np.ndarray = np.array(
            [self.graph[u][v] for u, v in self._edge_list], dtype=float
        )
        # Random phase per edge in [0, 2π).
        self._phase: np.ndarray = rng.uniform(0.0, 2 * math.pi, size=n)

    # ------------------------------------------------------------------
    def true_cost(self, e: Edge, t: float) -> float:
        idx = self._edge_idx(e)
        c0 = float(self._c0[idx])
        phase = float(self._phase[idx])
        mod = 1.0 + self._amplitude * math.sin(
            2 * math.pi * t / self._period + phase
        )
        return max(_COST_FLOOR, c0 * mod)

    def rho_true(self, e: Edge) -> float:
        """Maximum instantaneous rate: d/dt [c0*(1+A*sin(...))] = c0*A*(2π/T)."""
        idx = self._edge_idx(e)
        c0 = float(self._c0[idx])
        return c0 * self._amplitude * 2 * math.pi / self._period


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def grid_world(
    rows: int,
    cols: int,
    seed: int | Generator,
    kind: str = "bounded",
    **kwargs,
) -> StaticWorld | BoundedDriftWorld | JumpWorld | PeriodicWorld:
    """Create a 4-connected directed grid world.

    Parameters
    ----------
    rows, cols:
        Grid dimensions.
    seed:
        Integer seed or ``numpy.random.Generator``.  All randomness in
        ``true_cost`` is deterministic given the seed; ``observe`` draws from
        the same generator (not pure).
    kind:
        One of ``"static"``, ``"bounded"``, ``"jump"``, ``"periodic"``.
    **kwargs:
        Forwarded to the world constructor (see class docstrings).

    Returns
    -------
    A world satisfying the ``World`` protocol.
    """
    rng = _make_rng(seed)
    if kind == "static":
        return StaticWorld(rows, cols, rng, **kwargs)
    elif kind == "bounded":
        return BoundedDriftWorld(rows, cols, rng, **kwargs)
    elif kind == "jump":
        return JumpWorld(rows, cols, rng, **kwargs)
    elif kind == "periodic":
        return PeriodicWorld(rows, cols, rng, **kwargs)
    else:
        raise ValueError(f"Unknown world kind: {kind!r}")
