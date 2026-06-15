"""Regime-switching grid world for the CERT-FLOW RSS extended-validation cell
CAL->TEST DISTRIBUTION SHIFT (the eps_tv / A2 TV-Lipschitz theorem).
ADDITIONAL results for the RSS version; NOT a change to the published paper.

This world subclasses ``certflow.drift._GridBase`` and reuses the published
``certflow.drift.BoundedDriftWorld`` READ-ONLY (no package edits), so the
4-connected grid topology, the initial log-normal cost draw, and the
observation-noise families are byte-identical to the published worlds. The ONLY
new behaviour is a single CHANGEPOINT at ``t_cp`` after which the *generating
regime* switches -- exactly the calibration -> test distribution shift that
non-exchangeable conformal prediction (Barber, Candes, Ramdas, Tibshirani 2023;
the TV-Lipschitz slack ``delta_stale = sum_i w~_i min(1, 2 eps_tv age_i)`` in
conformal.py) is built to absorb.

Two switch knobs, used together or singly (the cell asks for noise_family
AND/OR rho):
  * Before t_cp:  drift rate rho_pre, noise (family_pre, scale_pre).
  * At/after t_cp: drift rate rho_post (e.g. 3x), noise (family_post,
    scale_post) (e.g. gaussian -> student_t / skewed, larger scale).

Construction (faithful + deterministic):
  * Two independent ``BoundedDriftWorld`` segments are built (drift rho_pre and
    rho_post). Their per-edge piecewise-linear trajectories are sampled onto a
    fixed time grid and SPLICED continuously at t_cp: the post segment is
    shifted so its value at t_cp equals the pre segment's, and the pre segment
    is shifted so t=0 equals THIS world's shared initial cost c0. The result is
    one precomputed ``(n_steps, n_edges)`` array; ``true_cost(e, t)`` is then a
    cheap linear interpolation -- a PURE deterministic function of (e, t), as
    the World contract requires. (Same precompute-then-interpolate pattern as
    the sibling ``correlated_world.py`` / ``realworld.TrafficWorld``.)
  * ``observe()`` adds a single mean-zero noise draw from the regime active at
    its query time (family_pre/scale_pre if t < t_cp else family_post/
    scale_post), drawn from the world RNG -- so the calibration buffer the
    planner builds genuinely mixes pre- and post-shift residuals across the CP.

``rho_true(e)`` (the A1 bound the planner consumes) is, by default
(``rho_true_mode="pre"``), the PRE-shift empirical bound -- so the planner is
genuinely surprised by the post-shift regime, the way a deployed system
calibrated under one regime would be. This makes the shift bite: the test
distribution is no longer the calibration distribution and an eps_tv=0
(exchangeable) claim has no slack for it. The realised post-shift A1-violation
rate against that frozen bound is MEASURED (same construction as
realworld.py / correlated_world.py) and printed -- never hidden.
``rho_true_mode="post"`` instead exposes the true post-shift bound (isolating a
pure noise-family shift); ``"max"`` exposes the larger of the two.
"""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from numpy.random import Generator

from certflow.drift import BoundedDriftWorld, _COST_CAP, _COST_FLOOR, _GridBase
from certflow.types import Edge


def _draw_one(rng: Generator, family: str, scale: float) -> float:
    """Single mean-zero observation-noise draw, matching drift._GridBase's
    families EXACTLY (see certflow.drift._GridBase._draw_noise, copied verbatim
    so the noise the planner sees is identical to the published worlds, just
    regime-switched at t_cp)."""
    if family == "gaussian":
        return float(rng.normal(0.0, scale))
    elif family == "laplace":
        return float(rng.laplace(0.0, scale))
    elif family == "student_t":
        return float(rng.standard_t(df=3) * scale)
    elif family == "skewed":
        z = float(rng.normal(0.0, 1.0))
        raw = math.exp(z) - math.exp(0.5)
        return raw * scale / math.sqrt((math.e - 1.0) * math.e)
    else:
        raise ValueError(f"Unknown noise family: {family!r}")


class ShiftWorld(_GridBase):
    """Grid world with a single regime changepoint at t_cp (drift + noise)."""

    def __init__(
        self,
        rows: int,
        cols: int,
        rng: Generator,
        t_cp: float,
        rho_pre: float = 0.005,
        rho_post: float = 0.05,
        family_pre: str = "gaussian",
        scale_pre: float = 0.05,
        family_post: str = "student_t",
        scale_post: float = 0.25,
        sigma: float = 0.5,
        resampling_rate: float = 0.5,
        dt_grid: float = 1.0,
        max_t: float = 1100.0,
        rho_true_mode: str = "pre",      # "pre" | "post" | "max"
        rho_quantile: float = 0.95,
    ) -> None:
        super().__init__(rows, cols, rng, sigma, family_pre, scale_pre)
        self._t_cp = float(t_cp)
        self._family_pre, self._scale_pre = family_pre, scale_pre
        self._family_post, self._scale_post = family_post, scale_post
        self._rho_pre, self._rho_post = rho_pre, rho_post
        self._dt_grid = dt_grid

        n = len(self._edge_list)
        n_steps = int(math.ceil(max_t / dt_grid)) + 2
        c0 = np.array([self.graph[u][v] for u, v in self._edge_list], dtype=float)

        # --- two BoundedDriftWorld segments (independent seeds from rng) ------
        seed_a = int(rng.integers(0, 2**31))
        seed_b = int(rng.integers(0, 2**31))
        segA = BoundedDriftWorld(
            rows, cols, np.random.default_rng(seed_a), rho=rho_pre,
            resampling_rate=resampling_rate, sigma=sigma, max_t=max_t,
        )
        segB = BoundedDriftWorld(
            rows, cols, np.random.default_rng(seed_b), rho=rho_post,
            resampling_rate=resampling_rate, sigma=sigma, max_t=max_t,
        )

        ts = np.arange(n_steps) * dt_grid
        # sample each segment's trajectory onto the grid (per edge)
        A = np.empty((n_steps, n), dtype=float)
        B = np.empty((n_steps, n), dtype=float)
        for j, e in enumerate(self._edge_list):
            A[:, j] = [segA.true_cost(e, float(tt)) for tt in ts]
            B[:, j] = [segB.true_cost(e, float(tt)) for tt in ts]

        # re-base segment A so A(t=0) == our shared c0 (carry the offset)
        A = A + (c0 - A[0])[None, :]
        cp_idx = int(round(self._t_cp / dt_grid))
        cp_idx = max(1, min(cp_idx, n_steps - 1))
        # value of the re-based A at the changepoint (continuity target)
        A_at_cp = A[cp_idx]
        # re-base segment B so B(t_cp) == A_at_cp
        B = B - B[cp_idx][None, :] + A_at_cp[None, :]

        # spliced truth: A before cp, B at/after cp; then clip like the package
        costs = np.where(
            (np.arange(n_steps) < cp_idx)[:, None], A, B
        )
        self._costs = np.clip(costs, _COST_FLOOR, _COST_CAP)  # (T, E)
        self._cp_idx = cp_idx

        # --- planner-visible A1 bound (the byte-identical CERT input) ---------
        if rho_true_mode == "pre":
            base = segA._rho_e
        elif rho_true_mode == "post":
            base = segB._rho_e
        elif rho_true_mode == "max":
            base = np.maximum(segA._rho_e, segB._rho_e)
        else:
            raise ValueError(rho_true_mode)
        self._rho_arr = np.maximum(np.asarray(base, dtype=float), 1e-9).copy()
        self._rho_quantile = rho_quantile

        # refresh the t0 adjacency the planner reads to our spliced t0 costs
        for i, (u, v) in enumerate(self._edge_list):
            self.graph[u][v] = float(self._costs[0, i])

        # measured diagnostics over the spliced truth (cheap; vectorized)
        self._measure_diagnostics()

    # ------------------------------------------------------------------
    def _measure_diagnostics(self) -> None:
        rates = np.abs(np.diff(self._costs, axis=0)) / self._dt_grid  # (T-1,E)
        pre = rates[: self._cp_idx - 1] if self._cp_idx > 1 else rates[:0]
        post = rates[self._cp_idx :]
        bound = self._rho_arr[None, :]
        self.drift_q95_pre = float(np.quantile(pre, 0.95)) if pre.size else float("nan")
        self.drift_q95_post = float(np.quantile(post, 0.95)) if post.size else float("nan")
        self.a1_violation_pre = (
            float((pre > bound + 1e-12).mean()) if pre.size else float("nan")
        )
        self.a1_violation_post = (
            float((post > bound + 1e-12).mean()) if post.size else float("nan")
        )

    # ------------------------------------------------------------------
    def _eval(self, idx: int, t: float) -> float:
        b = min(max(t / self._dt_grid, 0.0), self._costs.shape[0] - 1.001)
        i, frac = int(b), b - int(b)
        v = (1 - frac) * self._costs[i, idx] + frac * self._costs[i + 1, idx]
        return float(v)

    def true_cost(self, e: Edge, t: float) -> float:
        return self._eval(self._edge_idx(e), t)

    def observe(self, e: Edge, t: float) -> float:
        if t < self._t_cp:
            noise = _draw_one(self._rng, self._family_pre, self._scale_pre)
        else:
            noise = _draw_one(self._rng, self._family_post, self._scale_post)
        return self.true_cost(e, t) + noise

    def rho_true(self, e: Edge) -> float:
        return float(self._rho_arr[self._edge_idx(e)])

    def edges(self) -> Iterator[Edge]:
        return iter(self._edge_list)
