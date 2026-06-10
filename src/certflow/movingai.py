"""MovingAI benchmark adapter for CERT.

Parses MovingAI .map and .scen files and builds World-protocol-compliant
environments by delegating time-varying dynamics to drift.py machinery.

Terrain encoding decisions (documented):
  - '.' plain floor  -> passable, base cost 1.0
  - 'G' grass        -> passable, base cost 1.0
  - 'S' swamp        -> passable, higher base cost (base_cost_swamp, default 2.0)
  - '@' out-of-bounds -> impassable (wall)
  - 'O' out-of-bounds -> impassable (wall)
  - 'T' trees        -> impassable (treated as wall for simplicity; no diagonal
                        movement in our 4-connected graph so partial passability
                        is irrelevant)
  - 'W' water        -> impassable (treated as wall; true water costs are
                        map-type-specific and we favour conservative behaviour)

4-connectivity only (no diagonals). Both (u->v) and (v->u) are added as
independent directed edges with the same initial cost (the cost is symmetric
at t=0, but drifts independently for each direction).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.random import Generator

from certflow.drift import (
    _COST_CAP,
    _COST_FLOOR,
    _make_rng,
)
from certflow.types import Edge, Node

# ---------------------------------------------------------------------------
# Passability helpers
# ---------------------------------------------------------------------------

#: Characters that count as passable terrain.
_PASSABLE = frozenset(".GS")
#: Characters that are walls / impassable.
_IMPASSABLE = frozenset("@OTW")


def _is_passable(ch: str) -> bool:
    """Return True if the terrain character is passable."""
    return ch in _PASSABLE


def _base_cost(ch: str, base_cost_swamp: float = 2.0) -> float:
    """Return the initial undirected edge cost for a destination cell character."""
    if ch == "S":
        return base_cost_swamp
    return 1.0


# ---------------------------------------------------------------------------
# 1. parse_map
# ---------------------------------------------------------------------------

def parse_map(path: str | Path) -> list[str]:
    """Parse a MovingAI .map file; return the grid as a list of strings.

    The header (type / height / width / map) is validated but not returned.
    Each string in the returned list is one row of the grid (length == width).

    Parameters
    ----------
    path:
        Path to the ``.map`` file.

    Returns
    -------
    list[str]
        ``grid[row][col]`` gives the terrain character at (row, col).
        Length of the list equals the declared map height.

    Raises
    ------
    ValueError
        If the header is malformed or the grid has the wrong dimensions.
    """
    path = Path(path)
    with path.open("r") as fh:
        lines = fh.readlines()

    if not lines:
        raise ValueError(f"Empty map file: {path}")

    # Strip trailing whitespace / newlines.
    lines = [ln.rstrip("\r\n") for ln in lines]

    # Header lines 0-3: type / height / width / map
    if not lines[0].strip().lower().startswith("type"):
        raise ValueError(f"Expected 'type ...' on line 0, got: {lines[0]!r}")

    try:
        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse height/width from lines 1-2: {exc}") from exc

    if lines[3].strip().lower() != "map":
        raise ValueError(f"Expected 'map' on line 3, got: {lines[3]!r}")

    grid = lines[4 : 4 + height]

    if len(grid) != height:
        raise ValueError(
            f"Expected {height} grid rows, got {len(grid)} in {path}"
        )

    # Some files have shorter rows (trailing spaces stripped); pad with '@'.
    padded: list[str] = []
    for i, row in enumerate(grid):
        if len(row) < width:
            row = row + "@" * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        padded.append(row)

    return padded


# ---------------------------------------------------------------------------
# 2. parse_scen
# ---------------------------------------------------------------------------

def parse_scen(path: str | Path) -> list[dict]:
    """Parse a MovingAI .scen file; return a list of scenario dicts.

    Each dict has keys:
      ``bucket`` (int), ``map`` (str), ``w`` (int), ``h`` (int),
      ``sx`` (int), ``sy`` (int), ``gx`` (int), ``gy`` (int),
      ``optimal_length`` (float).

    The MovingAI convention is that X indexes the column and Y indexes the row,
    so ``sx`` is the start column, ``sy`` is the start row, etc.

    Parameters
    ----------
    path:
        Path to the ``.scen`` file.

    Returns
    -------
    list[dict]
        One dict per valid scenario entry.
    """
    path = Path(path)
    entries: list[dict] = []

    with path.open("r") as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            # First line: "version 1"
            if lineno == 0:
                if line.lower().startswith("version"):
                    continue
                # Some files omit the version header; treat as data.

            parts = line.split("\t")
            if len(parts) < 9:
                continue  # skip malformed lines

            try:
                entries.append(
                    {
                        "bucket": int(parts[0]),
                        "map": parts[1],
                        "w": int(parts[2]),
                        "h": int(parts[3]),
                        "sx": int(parts[4]),  # start col (X)
                        "sy": int(parts[5]),  # start row (Y)
                        "gx": int(parts[6]),  # goal col (X)
                        "gy": int(parts[7]),  # goal row (Y)
                        "optimal_length": float(parts[8]),
                    }
                )
            except (ValueError, IndexError):
                continue  # skip malformed lines

    return entries


# ---------------------------------------------------------------------------
# 3. crop
# ---------------------------------------------------------------------------

def crop(
    grid: list[str],
    row0: int,
    col0: int,
    size: int,
) -> tuple[list[str], int, int]:
    """Extract a square sub-map of side ``size`` from ``grid``.

    The crop is centred at ``(row0, col0)`` but clamped so it stays inside the
    full map boundaries.

    Parameters
    ----------
    grid:
        Full grid from ``parse_map``.
    row0, col0:
        Desired centre (row, col).
    size:
        Side length of the square crop (both height and width).

    Returns
    -------
    (sub_grid, offset_row, offset_col)
        ``offset_row`` and ``offset_col`` are the top-left corner of the crop
        in the original grid's coordinate system, so a cell ``(r, c)`` in the
        original maps to ``(r - offset_row, c - offset_col)`` in the crop.
    """
    full_h = len(grid)
    full_w = len(grid[0]) if grid else 0

    half = size // 2
    r_start = max(0, row0 - half)
    c_start = max(0, col0 - half)
    r_end = min(full_h, r_start + size)
    c_end = min(full_w, c_start + size)

    # Re-clamp start in case the end was clamped.
    r_start = max(0, r_end - size)
    c_start = max(0, c_end - size)

    sub = [row[c_start:c_end] for row in grid[r_start:r_end]]
    return sub, r_start, c_start


# ---------------------------------------------------------------------------
# 4. scenario_endpoints
# ---------------------------------------------------------------------------

def scenario_endpoints(
    scen_entries: list[dict],
    map_grid: list[str],
    min_length: float = 50.0,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return usable (start, goal) pairs from a list of scenario entries.

    Filters for entries with ``optimal_length >= min_length``, converts
    MovingAI (X=col, Y=row) coordinates to ``(row, col)`` tuples, and
    verifies that both endpoints lie on passable cells.

    Parameters
    ----------
    scen_entries:
        Parsed entries from ``parse_scen``.
    map_grid:
        Full grid from ``parse_map``.
    min_length:
        Minimum optimal path length threshold (inclusive).

    Returns
    -------
    list[tuple[tuple[int,int], tuple[int,int]]]
        Each element is ``(start_rc, goal_rc)`` where both are ``(row, col)``.
    """
    rows = len(map_grid)
    cols = len(map_grid[0]) if map_grid else 0
    result = []

    for entry in scen_entries:
        if entry["optimal_length"] < min_length:
            continue

        # MovingAI: sx=X=col, sy=Y=row
        start_rc = (entry["sy"], entry["sx"])
        goal_rc = (entry["gy"], entry["gx"])

        # Bounds check.
        sr, sc = start_rc
        gr, gc = goal_rc
        if not (0 <= sr < rows and 0 <= sc < cols):
            continue
        if not (0 <= gr < rows and 0 <= gc < cols):
            continue

        # Passability check.
        if not _is_passable(map_grid[sr][sc]):
            continue
        if not _is_passable(map_grid[gr][gc]):
            continue

        result.append((start_rc, goal_rc))

    return result


# ---------------------------------------------------------------------------
# Internal: graph builder from a grid
# ---------------------------------------------------------------------------

def _build_graph_from_grid(
    grid: list[str],
    base_cost_swamp: float = 2.0,
) -> tuple[
    dict[Node, dict[Node, float]],
    list[Edge],
    dict[Edge, int],
    np.ndarray,
]:
    """Build a 4-connected directed graph from a passable-cell grid.

    Returns
    -------
    graph:
        ``graph[u][v] = base_cost`` adjacency dict.
    edge_list:
        Ordered list of directed edges.
    edge_index:
        Reverse map from edge to its index in ``edge_list``.
    init_costs:
        1-D float array of initial costs aligned to ``edge_list``.
    """
    rows = len(grid)
    cols = len(grid[0]) if grid else 0

    edge_list: list[Edge] = []
    edge_costs: list[float] = []

    for r in range(rows):
        for c in range(cols):
            if not _is_passable(grid[r][c]):
                continue
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not _is_passable(grid[nr][nc]):
                    continue
                # Cost of traversal to neighbour (destination cell governs cost).
                cost = _base_cost(grid[nr][nc], base_cost_swamp)
                edge_list.append(((r, c), (nr, nc)))
                edge_costs.append(cost)

    edge_index: dict[Edge, int] = {e: i for i, e in enumerate(edge_list)}
    init_costs = np.array(edge_costs, dtype=float)

    graph: dict[Node, dict[Node, float]] = {}
    for i, (u, v) in enumerate(edge_list):
        graph.setdefault(u, {})[v] = float(init_costs[i])

    return graph, edge_list, edge_index, init_costs


# ---------------------------------------------------------------------------
# 5. _MovingAIBase — mirrors _GridBase but built from a grid, not row x col
# ---------------------------------------------------------------------------

class _MovingAIBase:
    """Shared graph structure and edge enumeration for MovingAI worlds.

    Mirrors ``drift._GridBase`` but builds its graph from a passable-cell
    grid instead of a rectangular ``rows x cols`` grid.  The noise/observe
    machinery is a verbatim copy so no code in drift.py needs to change.
    """

    def __init__(
        self,
        grid: list[str],
        rng: Generator,
        base_cost_swamp: float = 2.0,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
    ) -> None:
        self._noise_family = noise_family
        self._noise_scale = noise_scale
        self._rng = rng

        graph, edge_list, edge_index, init_costs = _build_graph_from_grid(
            grid, base_cost_swamp
        )
        self.graph: dict[Node, dict[Node, float]] = graph
        self._edge_list: list[Edge] = edge_list
        self._edge_index: dict[Edge, int] = edge_index
        self._init_costs: np.ndarray = init_costs

    # ------------------------------------------------------------------
    def edges(self) -> Iterator[Edge]:
        return iter(self._edge_list)

    def _edge_idx(self, e: Edge) -> int:
        return self._edge_index[e]

    # ------------------------------------------------------------------
    def _draw_noise(self) -> float:
        """Verbatim copy of drift._GridBase._draw_noise."""
        family = self._noise_family
        s = self._noise_scale
        if family == "gaussian":
            return float(self._rng.normal(0.0, s))
        elif family == "laplace":
            return float(self._rng.laplace(0.0, s))
        elif family == "student_t":
            return float(self._rng.standard_t(df=3) * s)
        elif family == "skewed":
            z = float(self._rng.normal(0.0, 1.0))
            raw = math.exp(z) - math.exp(0.5)
            return raw * s / math.sqrt((math.e - 1.0) * math.e)
        else:
            raise ValueError(f"Unknown noise family: {family!r}")

    def observe(self, e: Edge, t: float) -> float:
        """Noisy observation: true_cost(e, t) + additive noise."""
        return self.true_cost(e, t) + self._draw_noise()  # type: ignore[attr-defined]

    def rho_true(self, e: Edge) -> float:  # noqa: ARG002
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete world classes
# ---------------------------------------------------------------------------

class MovingAIStaticWorld(_MovingAIBase):
    """Static (no drift) world over a MovingAI map.  ``rho_true`` == 0."""

    def __init__(
        self,
        grid: list[str],
        rng: Generator,
        base_cost_swamp: float = 2.0,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
    ) -> None:
        super().__init__(grid, rng, base_cost_swamp, noise_family, noise_scale)
        self._costs = self._init_costs.copy()
        # Rebuild graph from _costs to be consistent.
        for i, (u, v) in enumerate(self._edge_list):
            self.graph[u][v] = float(self._costs[i])

    def true_cost(self, e: Edge, t: float) -> float:  # noqa: ARG002
        return float(self._costs[self._edge_idx(e)])

    def rho_true(self, e: Edge) -> float:  # noqa: ARG002
        return 0.0


class MovingAIBoundedDriftWorld(_MovingAIBase):
    """Piecewise-linear rate-limited random walk over a MovingAI map.

    Semantics are identical to ``drift.BoundedDriftWorld``; the only
    difference is that the graph comes from a MovingAI grid instead of a
    rectangular grid.  Base costs are deterministic (set by terrain type);
    drift is layered on top via per-edge Poisson-resampled trajectories.
    """

    def __init__(
        self,
        grid: list[str],
        rng: Generator,
        rho: float = 0.02,
        resampling_rate: float = 0.5,
        base_cost_swamp: float = 2.0,
        noise_family: str = "gaussian",
        noise_scale: float = 0.05,
        cost_floor: float = _COST_FLOOR,
        cost_cap: float = _COST_CAP,
        max_t: float = 1000.0,
    ) -> None:
        super().__init__(grid, rng, base_cost_swamp, noise_family, noise_scale)
        self._rho_global = rho
        self._resampling_rate = resampling_rate
        self._cost_floor = cost_floor
        self._cost_cap = cost_cap
        self._max_t = max_t

        n = len(self._edge_list)
        self._rho_e: np.ndarray = rho * rng.uniform(0.5, 1.5, size=n)
        self._c0: np.ndarray = self._init_costs.copy()

        self._trajectories: list[tuple[np.ndarray, np.ndarray] | None] = [
            None
        ] * n
        seeds = rng.integers(0, 2**31, size=n)
        self._edge_rngs: list[Generator] = [
            np.random.default_rng(int(s)) for s in seeds
        ]

    # ------------------------------------------------------------------
    def _build_trajectory(self, idx: int) -> None:
        """Verbatim logic from drift.BoundedDriftWorld._build_trajectory."""
        rho_e = float(self._rho_e[idx])
        c0 = float(self._c0[idx])
        rate = self._resampling_rate
        erng = self._edge_rngs[idx]
        floor = self._cost_floor
        cap = self._cost_cap

        times = [0.0]
        t_cur = 0.0
        while t_cur < self._max_t:
            dt = erng.exponential(1.0 / rate)
            t_cur += dt
            times.append(t_cur)
        times_arr = np.array(times, dtype=float)

        n_segs = len(times_arr)
        slopes = erng.uniform(-rho_e, rho_e, size=n_segs)

        values = np.empty(n_segs, dtype=float)
        values[0] = np.clip(c0, floor, cap)
        for k in range(1, n_segs):
            dt = times_arr[k] - times_arr[k - 1]
            proposed = values[k - 1] + slopes[k - 1] * dt
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

    def _eval_trajectory(self, idx: int, t: float) -> float:
        """Verbatim logic from drift.BoundedDriftWorld._eval_trajectory."""
        if self._trajectories[idx] is None:
            self._build_trajectory(idx)
        times, values = self._trajectories[idx]  # type: ignore[misc]

        if t <= 0.0:
            return float(values[0])

        k = int(np.searchsorted(times, t, side="right")) - 1
        k = min(k, len(times) - 2)

        t0 = times[k]
        v0 = values[k]
        dt_seg = times[k + 1] - t0
        if dt_seg < 1e-15:
            return float(v0)
        slope = (values[k + 1] - values[k]) / dt_seg
        raw = v0 + slope * (t - t0)
        return float(np.clip(raw, self._cost_floor, self._cost_cap))

    def true_cost(self, e: Edge, t: float) -> float:
        idx = self._edge_idx(e)
        return self._eval_trajectory(idx, t)

    def rho_true(self, e: Edge) -> float:
        return float(self._rho_e[self._edge_idx(e)])


# ---------------------------------------------------------------------------
# 6. movingai_world — public factory
# ---------------------------------------------------------------------------

def movingai_world(
    map_path: str | Path,
    seed: int | Generator,
    kind: str = "bounded",
    rho: float = 0.02,
    base_cost_swamp: float = 2.0,
    noise_family: str = "gaussian",
    noise_scale: float = 0.05,
    **kw,
) -> MovingAIStaticWorld | MovingAIBoundedDriftWorld:
    """Build a CERT ``World`` over a MovingAI map.

    Parameters
    ----------
    map_path:
        Path to the ``.map`` file.
    seed:
        Integer seed or ``numpy.random.Generator``.
    kind:
        ``"static"`` (no drift) or ``"bounded"`` (piecewise-linear
        rate-limited random walk, same as ``drift.BoundedDriftWorld``).
    rho:
        Global drift-rate bound (used for ``kind="bounded"``).
    base_cost_swamp:
        Base traversal cost for swamp cells ('S').  Plain floor/grass cells
        use 1.0.  Impassable cells are excluded from the graph.
    noise_family:
        Additive observation noise: ``"gaussian"``, ``"laplace"``,
        ``"student_t"``, or ``"skewed"``.
    noise_scale:
        Scale parameter for the noise distribution.
    **kw:
        Additional keyword arguments forwarded to the world constructor.

    Returns
    -------
    A world satisfying the ``certflow.types.World`` protocol.
    """
    grid = parse_map(map_path)
    rng = _make_rng(seed)

    if kind == "static":
        return MovingAIStaticWorld(
            grid, rng,
            base_cost_swamp=base_cost_swamp,
            noise_family=noise_family,
            noise_scale=noise_scale,
            **kw,
        )
    elif kind == "bounded":
        return MovingAIBoundedDriftWorld(
            grid, rng,
            rho=rho,
            base_cost_swamp=base_cost_swamp,
            noise_family=noise_family,
            noise_scale=noise_scale,
            **kw,
        )
    else:
        raise ValueError(
            f"Unknown world kind: {kind!r}. "
            "MovingAI adapter supports 'static' and 'bounded'."
        )


# ---------------------------------------------------------------------------
# Convenience: build a world from a cropped grid
# ---------------------------------------------------------------------------

def movingai_world_from_grid(
    grid: list[str],
    seed: int | Generator,
    kind: str = "bounded",
    rho: float = 0.02,
    base_cost_swamp: float = 2.0,
    noise_family: str = "gaussian",
    noise_scale: float = 0.05,
    **kw,
) -> MovingAIStaticWorld | MovingAIBoundedDriftWorld:
    """Build a CERT ``World`` directly from a (possibly cropped) grid.

    Useful when you want to crop the map first (via ``crop()``) and then
    build a world over the sub-map without going back to the file.

    Parameters are identical to ``movingai_world`` except ``map_path`` is
    replaced by the already-parsed ``grid``.
    """
    rng = _make_rng(seed)

    if kind == "static":
        return MovingAIStaticWorld(
            grid, rng,
            base_cost_swamp=base_cost_swamp,
            noise_family=noise_family,
            noise_scale=noise_scale,
            **kw,
        )
    elif kind == "bounded":
        return MovingAIBoundedDriftWorld(
            grid, rng,
            rho=rho,
            base_cost_swamp=base_cost_swamp,
            noise_family=noise_family,
            noise_scale=noise_scale,
            **kw,
        )
    else:
        raise ValueError(
            f"Unknown world kind: {kind!r}. "
            "MovingAI adapter supports 'static' and 'bounded'."
        )
