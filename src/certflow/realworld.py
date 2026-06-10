"""Real-data World adapters: METR-LA / PEMS-BAY traffic replay.

The recorded loop-detector speeds ARE the ground truth: edge travel times are
replayed from the recording (piecewise-linear between 5-minute bins), so the
oracle is exact on real drifting costs. Observation noise is synthetic and
configurable (the recording does not separate sensor noise from state; we
state this rather than hide it). A1 is NOT guaranteed by construction here —
incidents produce drift-rate spikes — so rho_true is an empirical per-edge
quantile of |dc/dt| over the replay window, and the violation rate of the
implied A1 bound is a measured, reported quantity.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

from certflow.types import Edge, Node

BIN_SECONDS = 300.0
MPH_TO_MPS = 0.44704
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=2)
def _load_traffic(dataset: str) -> tuple[list[str], np.ndarray, dict[tuple[str, str], float]]:
    """(sensor_ids, speeds[bins x sensors] mph with missing filled, distances m)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised via a stubbed import
        raise ImportError(
            "The real-world traffic adapter (METR-LA / PEMS-BAY) needs pandas and "
            "PyTables, which are not part of the core install. Install them with:\n"
            "    pip install 'certflow[realworld]'"
        ) from exc

    if dataset == "metr-la":
        h5, dist_csv = DATA_DIR / "metr-la/metr_la.h5", DATA_DIR / "metr-la/distances_la_2012.csv"
    elif dataset == "pems-bay":
        h5, dist_csv = DATA_DIR / "pems-bay/pems_bay.h5", DATA_DIR / "pems-bay/distances_bay.csv"
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    df = pd.read_hdf(h5)
    ids = [str(c) for c in df.columns]
    speeds = df.values.astype(float)
    # missing readings are zeros: forward-fill then back-fill per sensor,
    # then clamp to a physical range so costs stay finite
    arr = np.where(speeds <= 0.0, np.nan, speeds)
    dfa = pd.DataFrame(arr).ffill().bfill()
    speeds = np.clip(dfa.values, 3.0, 75.0)
    d = pd.read_csv(dist_csv)
    # column-wise iteration: row-wise access would upcast int ids to floats
    dist = {
        (str(int(u)), str(int(v))): float(m)
        for u, v, m in zip(d["from"], d["to"], d["cost"])
        if int(u) != int(v)
    }
    return ids, speeds, dist


class TrafficWorld:
    """World over a replayed traffic recording.

    Nodes are sensor ids (strings); a directed edge (u, v) exists when the
    road-network distance u->v is at most max_dist_m. Cost = travel time in
    seconds at the tail sensor's current speed, piecewise-linear in t.
    The replay window is [offset_bins, offset_bins + n_bins); t=0 maps to
    the window start. Different seeds map to different windows (stride) and
    independent noise streams.
    """

    def __init__(
        self,
        dataset: str = "metr-la",
        seed: int = 0,
        n_bins: int = 288,              # one day
        window_stride_bins: int = 288,
        max_dist_m: float = 3000.0,
        noise_scale: float = 5.0,       # seconds, additive
        noise_family: str = "gaussian",
        rho_quantile: float = 0.95,
        offset_base_bins: int = 0,
    ) -> None:
        ids, speeds, dist = _load_traffic(dataset)
        usable = speeds.shape[0] - offset_base_bins
        n_windows = max(1, (usable - n_bins) // window_stride_bins)
        offset = offset_base_bins + (seed % n_windows) * window_stride_bins
        self._speeds = speeds[offset : offset + n_bins + 1]  # +1 for interp
        self._idx = {s: i for i, s in enumerate(ids)}
        self._noise_scale = noise_scale
        self._noise_family = noise_family
        self._rng = np.random.default_rng(seed + 10_000)
        self.n_bins = n_bins

        self.graph: dict[Node, dict[Node, float]] = {s: {} for s in ids}
        self._dist: dict[Edge, float] = {}
        for (u, v), m in dist.items():
            if 0.0 < m <= max_dist_m and u in self._idx and v in self._idx:
                self._dist[(u, v)] = m
                self.graph[u][v] = self.true_cost((u, v), 0.0)
        # drop isolated nodes
        used = {u for u, _ in self._dist} | {v for _, v in self._dist}
        self.graph = {n: nbrs for n, nbrs in self.graph.items() if n in used}

        # empirical per-edge drift-rate bound at the configured quantile,
        # and the measured violation rate of the implied A1 bound
        self._rho: dict[Edge, float] = {}
        viol = total = 0
        for e in self._dist:
            c = self._cost_series(e)
            rates = np.abs(np.diff(c)) / BIN_SECONDS
            r = float(np.quantile(rates, rho_quantile)) if len(rates) else 0.0
            self._rho[e] = max(r, 1e-9)
            viol += int((rates > self._rho[e] + 1e-12).sum())
            total += len(rates)
        self.a1_violation_rate = viol / total if total else 0.0

    def _cost_series(self, e: Edge) -> np.ndarray:
        u, _ = e
        sp = self._speeds[:, self._idx[u]]
        return self._dist[e] / (sp * MPH_TO_MPS)

    def true_cost(self, e: Edge, t: float) -> float:
        u, _ = e
        b = min(max(t / BIN_SECONDS, 0.0), self._speeds.shape[0] - 1.001)
        i, frac = int(b), b - int(b)
        col = self._idx[u]
        sp = (1 - frac) * self._speeds[i, col] + frac * self._speeds[i + 1, col]
        return self._dist[e] / (sp * MPH_TO_MPS)

    def observe(self, e: Edge, t: float) -> float:
        s = self._noise_scale
        if self._noise_family == "gaussian":
            eta = self._rng.normal(0.0, s)
        elif self._noise_family == "laplace":
            eta = self._rng.laplace(0.0, s)
        elif self._noise_family == "student_t":
            eta = self._rng.standard_t(3) * s
        else:
            raise ValueError(f"unknown noise family {self._noise_family!r}")
        return self.true_cost(e, t) + float(eta)

    def edges(self) -> Iterator[Edge]:
        return iter(self._dist)

    def rho_true(self, e: Edge) -> float:
        return self._rho[e]


def traffic_planner_config(**overrides):
    """PlannerConfig with TIME-UNIT-AWARE defaults for traffic replay.

    Real time runs in seconds with Delta = 300 s/round; the per-unit-time
    knobs (rho_w decay, eps_tv drift rate) must be rescaled from their
    per-round synthetic defaults or the staleness machinery runs 300x too
    fast (claims self-extinguish instantly). Costs are seconds of travel
    time, so epsilon/noise are in seconds too.
    """
    from certflow.cert import PlannerConfig

    defaults = dict(
        epsilon=120.0,                  # certify within 2 minutes of optimal
        alpha_prime=0.2,
        delta=BIN_SECONDS,              # one observation per 5-minute bin
        rho_w=0.99 ** (1.0 / BIN_SECONDS),   # ~0.99 per round
        eps_tv=1e-4 / BIN_SECONDS,           # ~1e-4 per round
        sense_cost=1.0,
        cost_floor=1.0,                 # 1 second
    )
    defaults.update(overrides)
    return PlannerConfig(**defaults)


def far_endpoints(world: TrafficWorld, min_hops: int = 6) -> tuple[Node, Node]:
    """A (start, goal) pair at least min_hops apart with goal reachable,
    deterministic for a given world graph."""
    from collections import deque

    nodes = sorted(world.graph)
    for s in nodes:
        seen = {s: 0}
        dq = deque([s])
        far: Node | None = None
        while dq:
            u = dq.popleft()
            for v in world.graph[u]:
                if v not in seen:
                    seen[v] = seen[u] + 1
                    if seen[v] >= min_hops:
                        far = v
                    dq.append(v)
        if far is not None:
            return s, far
    raise ValueError("graph has no pair at the requested distance")


def fit_spatial_predictor(
    dataset: str = "metr-la",
    train_bins: int = 18000,
    max_dist_m: float = 3000.0,
    fresh_age: float = 2.0 * BIN_SECONDS,
    ridge: float = 10.0,
    age_matched: bool = False,
):
    """P2 from the spatial-predictor study: per-sensor ridge of current speed
    on [stale own speed, fresh-neighbor mean speed], fit on the first
    train_bins of the recording (evaluation windows must start AFTER
    train_bins — see TrafficWorld offset_base_bins — so there is no leakage).

    Returns predictor(e, t, beliefs) -> predicted COST or None, suitable for
    CertPlanner(..., predictor=...). Fresh neighbors are read from the
    planner's own beliefs (edges with tail=neighbor and age <= fresh_age);
    None when no fresh neighbor exists (LOCF/model fallback upstream).

    age_matched=False (default) reproduces the original fresh-at-t training and
    runtime EXACTLY: neighbor means are read as the planner sees them now and
    regressed on [stale own speed, fresh neighbor mean].

    age_matched=True trains on the deployment distribution instead: each pair
    samples a target age a in {12,24,48} AND a neighbor age b in {0,6,12} bins,
    so the neighbor mean is computed at speed[t-b] (semi-stale, as deployed) and
    b is fed as a third regression feature. At runtime the actual neighbor age
    (min over used neighbors, snapped to the {0,6,12} ladder) is supplied as the
    b feature and the fresh-age gate widens to 12 bins so realistically-stale
    neighbors are accepted rather than discarded.
    """
    import numpy as np

    ids, speeds, dist = _load_traffic(dataset)
    idx = {s: i for i, s in enumerate(ids)}
    nbrs: dict[str, list[str]] = {s: [] for s in ids}
    for (u, v), m in dist.items():
        if 0.0 < m <= max_dist_m and u in idx and v in idx:
            nbrs[u].append(v)

    tr = speeds[:train_bins]
    rng = np.random.default_rng(0)
    ages = (12, 24, 48)
    nbr_age_bins = (0, 6, 12)  # neighbor ages used for age-matched training

    dist_map = {
        (u, v): m for (u, v), m in dist.items()
        if 0.0 < m <= max_dist_m and u in idx and v in idx
    }

    def _speed_from_belief(edge, beliefs):
        b = beliefs.get(edge)
        m = dist_map.get(edge)
        if b is None or m is None or b.c_hat <= 0:
            return None, math.inf
        return m / (b.c_hat * MPH_TO_MPS), b.t_obs

    if not age_matched:
        # ----- original fresh-at-t training (default; unchanged) -----
        coefs: dict[str, tuple[float, float, float]] = {}
        for s_id in ids:
            js = [idx[j] for j in nbrs[s_id]]
            if not js:
                continue
            i = idx[s_id]
            ts = rng.integers(max(ages), train_bins - 1, size=400)
            X, y = [], []
            for t in ts:
                a = int(rng.choice(ages))
                X.append([tr[t - a, i], tr[t, js].mean()])
                y.append(tr[t, i])
            X = np.asarray(X)
            Xb = np.hstack([X, np.ones((len(X), 1))])
            w = np.linalg.solve(Xb.T @ Xb + ridge * np.eye(3), Xb.T @ np.asarray(y))
            coefs[s_id] = (float(w[0]), float(w[1]), float(w[2]))

        def _predict(e, t, beliefs, cf):
            u, _ = e
            m_e = dist_map.get(e)
            if m_e is None:
                return None
            stale_speed, _ = _speed_from_belief(e, beliefs)
            if stale_speed is None:
                return None
            fresh = []
            for j in nbrs[u]:
                best_e, best_age = None, math.inf
                for k in nbrs.get(j, []):
                    bj = beliefs.get((j, k))
                    if bj is not None and (t - bj.t_obs) < best_age:
                        best_e, best_age = (j, k), t - bj.t_obs
                if best_e is not None and best_age <= fresh_age:
                    sp, _ = _speed_from_belief(best_e, beliefs)
                    if sp is not None:
                        fresh.append(sp)
            if not fresh:
                return None
            a0, a1, a2 = cf
            pred_speed = a0 * stale_speed + a1 * (sum(fresh) / len(fresh)) + a2
            pred_speed = min(max(pred_speed, 3.0), 75.0)
            return m_e / (pred_speed * MPH_TO_MPS)

        def predictor(e, t, beliefs):
            u, _ = e
            cf = coefs.get(u)
            if cf is None:
                return None
            if beliefs.get(e) is None:
                return None
            return _predict(e, t, beliefs, cf)

        return predictor

    # ----- age-matched training (deployment-distribution features) -----
    # neighbor age b feeds in as an extra feature; gate widens to 12 bins so
    # semi-stale neighbors are kept rather than thrown away.
    max_b = max(nbr_age_bins)
    am_fresh_age = max_b * BIN_SECONDS
    coefs4: dict[str, tuple[float, float, float, float]] = {}
    for s_id in ids:
        js = [idx[j] for j in nbrs[s_id]]
        if not js:
            continue
        i = idx[s_id]
        ts = rng.integers(max(max(ages), max_b), train_bins - 1, size=400)
        X, y = [], []
        for t in ts:
            a = int(rng.choice(ages))
            b = int(rng.choice(nbr_age_bins))
            # neighbor mean at the realistic neighbor age t-b (semi-stale)
            X.append([tr[t - a, i], tr[t - b, js].mean(), float(b)])
            y.append(tr[t, i])
        X = np.asarray(X)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        w = np.linalg.solve(Xb.T @ Xb + ridge * np.eye(4), Xb.T @ np.asarray(y))
        coefs4[s_id] = (float(w[0]), float(w[1]), float(w[2]), float(w[3]))

    def _snap_b(age_bins: float) -> int:
        """Snap a neighbor age (in bins) to the nearest training ladder rung."""
        return min(nbr_age_bins, key=lambda c: abs(c - age_bins))

    def _predict_am(e, t, beliefs, cf):
        u, _ = e
        m_e = dist_map.get(e)
        if m_e is None:
            return None
        stale_speed, _ = _speed_from_belief(e, beliefs)
        if stale_speed is None:
            return None
        fresh = []
        ages_used = []
        for j in nbrs[u]:
            best_e, best_age = None, math.inf
            for k in nbrs.get(j, []):
                bj = beliefs.get((j, k))
                if bj is not None and (t - bj.t_obs) < best_age:
                    best_e, best_age = (j, k), t - bj.t_obs
            if best_e is not None and best_age <= am_fresh_age:
                sp, _ = _speed_from_belief(best_e, beliefs)
                if sp is not None:
                    fresh.append(sp)
                    ages_used.append(best_age)
        if not fresh:
            return None
        # actual neighbor age = freshest used neighbor, snapped to {0,6,12} bins
        b_bins = _snap_b(min(ages_used) / BIN_SECONDS)
        a0, a1, a2, a3 = cf
        pred_speed = (a0 * stale_speed
                      + a1 * (sum(fresh) / len(fresh))
                      + a2 * float(b_bins)
                      + a3)
        pred_speed = min(max(pred_speed, 3.0), 75.0)
        return m_e / (pred_speed * MPH_TO_MPS)

    def predictor_am(e, t, beliefs):
        u, _ = e
        cf = coefs4.get(u)
        if cf is None:
            return None
        if beliefs.get(e) is None:
            return None
        return _predict_am(e, t, beliefs, cf)

    return predictor_am
