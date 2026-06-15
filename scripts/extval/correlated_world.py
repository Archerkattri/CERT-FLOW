"""Spatially-correlated + heavy-tailed drift worlds for the CERT-FLOW RSS
extended-validation cell (ADDITIONAL results, NOT part of the published paper).

These worlds subclass ``certflow.drift._GridBase`` READ-ONLY (no package edits)
to reuse the exact 4-connected grid topology, the initial log-normal cost draw,
and the observation-noise families. Only the *drift dynamics* are new:

CorrelatedDriftWorld -- a one-factor "moving congestion front":

    c_e(t) = clip( c0_e + g_e(t) * L(t) + s_idio * I_e(t) ,  floor, cap )

  * L(t) is a SINGLE shared latent process with HEAVY-TAILED increments
    (student-t df=3 or Pareto). Its increments dL_t drive every edge at once,
    so neighboring edges move TOGETHER -- the spatial correlation under test.
  * g_e(t) is the edge's time-varying loading on the latent: a Gaussian bump
    in grid-x centered on a front position that SWEEPS across the grid with t
    (front_speed columns / unit time). Edges near the front have large g and
    are strongly co-driven; the cluster of correlated edges MOVES.
  * I_e(t) is a small per-edge idiosyncratic heavy-tailed walk (independent),
    so each edge has private noise on top of the shared factor.

IndepDriftWorld -- the MATCHED control. Same per-edge MARGINAL increment
distribution and magnitude as the correlated world, but every edge is driven
by its OWN independent heavy-tailed walk (no shared factor). Concretely the
correlated edge has increment variance Var(dc_e) = g_e^2 Var(dL) + s_idio^2
Var(dI); the control draws a single independent increment whose scale is set so
its per-edge marginal variance MATCHES that exactly. Thus the two worlds differ
ONLY in cross-edge dependence, at matched per-edge magnitude -- the fair
control the cell asks for.

Both worlds:
  * Precompute the full cost trajectory on a fixed time grid at construction,
    so true_cost(e, t) is a PURE deterministic function of (e, t) (linear
    interpolation between grid points) -- satisfies the World contract.
  * Expose rho_true(e) = empirical per-edge quantile of |dc/dt| over the grid,
    and a measured ``a1_violation_rate`` (fraction of steps exceeding the
    implied A1 bound) -- the SAME faithful construction realworld.TrafficWorld
    uses (realworld.py lines 108-119). Heavy tails mean A1 is stressed by
    design; the violation rate is measured and reported, never hidden.
"""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from numpy.random import Generator

from certflow.drift import _COST_CAP, _COST_FLOOR, _GridBase
from certflow.types import Edge


def _heavy_increments(
    rng: Generator, shape, family: str, scale: float
) -> np.ndarray:
    """Mean-zero, UNIT-std heavy-tailed increments times ``scale`` (so ``scale``
    is the population standard deviation of the increment for BOTH families and
    BOTH worlds -- this is what makes the matched-magnitude control exact).

    student_t: standard_t(df=3) has variance df/(df-2)=3, so divide by sqrt(3).
               Heavy tails, finite variance, symmetric.
    pareto:    symmetric Lomax (Pareto type II, a=3) shifted to zero mean,
               sign-randomised, then divided by its population std. a=3 ->
               finite variance with a heavy right tail.
    """
    if family == "student_t":
        df = 3.0
        return rng.standard_t(df=df, size=shape) / math.sqrt(df / (df - 2.0)) * scale
    elif family == "pareto":
        a = 3.0
        mag = rng.pareto(a, size=shape) - 1.0 / (a - 1.0)  # center (mean 1/(a-1))
        sign = rng.choice((-1.0, 1.0), size=shape)
        std = math.sqrt(a / ((a - 1.0) ** 2 * (a - 2.0)))  # Lomax population std
        return sign * mag / std * scale
    else:
        raise ValueError(f"unknown heavy-tail family {family!r}")


class _PrecomputedDriftBase(_GridBase):
    """Shared: build per-edge cost series on a time grid, expose rho_true and
    a measured A1-violation rate exactly like realworld.TrafficWorld."""

    def _finalize(self, costs: np.ndarray, dt_grid: float, rho_quantile: float) -> None:
        """costs: (n_steps, n_edges) precomputed true costs on the time grid.
        Stores them, derives empirical rho_true and the A1-violation rate."""
        self._costs = costs  # (T, E)
        self._dt_grid = dt_grid
        n_edges = costs.shape[1]
        # refresh the adjacency the planner reads at t0 with our t0 costs
        for i, (u, v) in enumerate(self._edge_list):
            self.graph[u][v] = float(costs[0, i])
        # empirical per-edge drift bound + measured A1 violations (realworld.py)
        rates = np.abs(np.diff(costs, axis=0)) / dt_grid  # (T-1, E)
        self._rho_arr = np.maximum(
            np.quantile(rates, rho_quantile, axis=0), 1e-9
        )
        viol = int((rates > self._rho_arr[None, :] + 1e-12).sum())
        total = rates.size
        self.a1_violation_rate = viol / total if total else 0.0
        # diagnostics: cross-edge correlation of the increments (neighbours)
        self._mean_abs_increment = float(np.abs(np.diff(costs, axis=0)).mean())

    def _eval(self, idx: int, t: float) -> float:
        b = min(max(t / self._dt_grid, 0.0), self._costs.shape[0] - 1.001)
        i, frac = int(b), b - int(b)
        v = (1 - frac) * self._costs[i, idx] + frac * self._costs[i + 1, idx]
        return float(v)

    def true_cost(self, e: Edge, t: float) -> float:
        return self._eval(self._edge_idx(e), t)

    def rho_true(self, e: Edge) -> float:
        return float(self._rho_arr[self._edge_idx(e)])

    def edges(self) -> Iterator[Edge]:
        return iter(self._edge_list)

    # --- correlation diagnostics over the precomputed increments ----------
    def neighbour_increment_corr(self) -> float:
        """Mean Pearson correlation of per-step increments between edges that
        share a node (adjacent edges) -- the spatial-correlation summary."""
        d = np.diff(self._costs, axis=0)  # (T-1, E)
        # build adjacency among edges sharing an endpoint
        node_to_edges: dict = {}
        for i, (u, v) in enumerate(self._edge_list):
            node_to_edges.setdefault(u, []).append(i)
            node_to_edges.setdefault(v, []).append(i)
        pairs = set()
        for ids in node_to_edges.values():
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    pairs.add((min(ids[a], ids[b]), max(ids[a], ids[b])))
        if not pairs:
            return float("nan")
        # standardise columns once
        dm = d - d.mean(axis=0, keepdims=True)
        sd = dm.std(axis=0)
        corrs = []
        for a, b in pairs:
            if sd[a] > 1e-12 and sd[b] > 1e-12:
                corrs.append(float((dm[:, a] * dm[:, b]).mean() / (sd[a] * sd[b])))
        return float(np.mean(corrs)) if corrs else float("nan")


class CorrelatedDriftWorld(_PrecomputedDriftBase):
    """One-factor moving-front drift with heavy-tailed shared increments."""

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: Generator,
        latent_scale: float = 0.06,
        idio_scale: float = 0.02,
        front_speed: float = 0.08,   # columns swept per unit time
        front_width: float = 1.2,    # Gaussian bump std in column units
        heavy_family: str = "student_t",
        dt_grid: float = 1.0,
        max_t: float = 1100.0,
        rho_quantile: float = 0.95,
        sigma: float = 0.5,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
        cost_floor: float = _COST_FLOOR,
        cost_cap: float = _COST_CAP,
    ) -> None:
        super().__init__(rows, cols, rng, sigma, noise_family, noise_scale)
        n = len(self._edge_list)
        n_steps = int(math.ceil(max_t / dt_grid)) + 2
        c0 = np.array([self.graph[u][v] for u, v in self._edge_list], dtype=float)

        # edge midpoint column (x) for the front loading
        edge_x = np.array(
            [(u[1] + v[1]) / 2.0 for u, v in self._edge_list], dtype=float
        )

        # shared latent path: heavy-tailed increments, mean zero
        dL = _heavy_increments(rng, n_steps, heavy_family, latent_scale)
        L = np.cumsum(dL)  # (T,)
        # per-edge idiosyncratic heavy-tailed walks (independent)
        dI = _heavy_increments(rng, (n_steps, n), heavy_family, idio_scale)
        I = np.cumsum(dI, axis=0)  # (T, E)

        ts = np.arange(n_steps) * dt_grid
        front_pos = (front_speed * ts) % (cols + 2 * front_width)  # sweep & wrap

        costs = np.empty((n_steps, n), dtype=float)
        for k in range(n_steps):
            # loading: Gaussian bump of front -- neighbours near front co-driven
            g = np.exp(-0.5 * ((edge_x - front_pos[k]) / front_width) ** 2)
            raw = c0 + g * L[k] + I[k]
            costs[k] = np.clip(raw, cost_floor, cost_cap)

        # stash the per-edge increment-scale profile so the matched control can
        # reproduce the SAME marginal magnitude (Var(dc_e) = g_e^2 Var(dL) +
        # Var(dI_e)); we hand the realised increment std per edge to the control
        self._edge_increment_std = np.diff(costs, axis=0).std(axis=0)  # (E,)
        self._heavy_family = heavy_family
        self._finalize(costs, dt_grid, rho_quantile)

    def observe(self, e: Edge, t: float) -> float:
        return self.true_cost(e, t) + self._draw_noise()


class IndepDriftWorld(_PrecomputedDriftBase):
    """Matched independent control: each edge an independent heavy-tailed walk
    whose per-edge increment std MATCHES a reference (correlated) world's, so
    only the cross-edge dependence differs (matched marginal magnitude)."""

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: Generator,
        edge_increment_std: np.ndarray,
        heavy_family: str = "student_t",
        dt_grid: float = 1.0,
        max_t: float = 1100.0,
        rho_quantile: float = 0.95,
        shared_rho: np.ndarray | None = None,
        sigma: float = 0.5,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
        cost_floor: float = _COST_FLOOR,
        cost_cap: float = _COST_CAP,
    ) -> None:
        super().__init__(rows, cols, rng, sigma, noise_family, noise_scale)
        n = len(self._edge_list)
        n_steps = int(math.ceil(max_t / dt_grid)) + 2
        c0 = np.array([self.graph[u][v] for u, v in self._edge_list], dtype=float)

        # per-edge independent heavy-tailed increments, scale = matched std
        # (unit-std increments times the reference world's per-edge std, so the
        # per-edge MARGINAL drift magnitude matches; only dependence differs).
        unit = _heavy_increments(rng, (n_steps, n), heavy_family, 1.0)
        dC = unit * edge_increment_std[None, :]
        C = np.cumsum(dC, axis=0)
        costs = np.clip(c0[None, :] + C, cost_floor, cost_cap)
        self._finalize(costs, dt_grid, rho_quantile)
        # IDENTICAL-INPUTS option: override the planner-visible A1 bound with the
        # reference (correlated) world's rho, so CERT consumes byte-identical
        # rho on both worlds. The realised A1-violation rate against this shared
        # bound is then measured separately and reported (may differ).
        if shared_rho is not None:
            self._rho_arr = np.asarray(shared_rho, dtype=float).copy()
            rates = np.abs(np.diff(self._costs, axis=0)) / self._dt_grid
            viol = int((rates > self._rho_arr[None, :] + 1e-12).sum())
            self.a1_violation_rate = viol / rates.size if rates.size else 0.0

    def observe(self, e: Edge, t: float) -> float:
        return self.true_cost(e, t) + self._draw_noise()
