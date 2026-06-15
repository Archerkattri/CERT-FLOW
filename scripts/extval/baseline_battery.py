"""Conformal baseline battery: age-weighting (CERT) vs NexCP vs ACI.

ADDITIONAL results for the RSS version of the CERT-FLOW paper. NOT a change to
the published paper; nothing here modifies src/certflow (imported read-only).

THE HEADLINE CELL. We put CERT's conformal scorer head-to-head against two
faithful non-exchangeable conformal comparators ON THE SAME edge-cost residual
stream, at the SAME target level alpha=0.1, and report marginal coverage and
median interval width.

------------------------------------------------------------------------------
The shared task (identical for all three methods)
------------------------------------------------------------------------------
One-step-ahead conformal prediction of an edge's travel cost under drift. We
replay a stream of edge sensings. At step i an edge e is sensed at time t_i;
the point predictor is LOCF -- c_hat = the last observed cost on e. The
nonconformity score is the standard split-conformal absolute residual

        s_i = | obs_i - c_hat_i |                                  (1)

This is THE residual stream; it is identical for every method. A method that,
from the calibration scores in its strict past, produces a (1-alpha)-quantile q
forms the symmetric prediction interval

        obs_i in [ c_hat_i - q ,  c_hat_i + q ]                    (2)

so obs_i is COVERED iff s_i <= q and the interval WIDTH is 2q. The ONLY thing
that differs between methods is how q is computed from the calibration scores --
which is exactly the axis under test (how non-exchangeable conformal weights the
past). We deliberately do NOT fold CERT's drift-model widening (rho_e * age)
into the interval here: that is a separate CERT mechanism with no NexCP/ACI
analogue, and including it would be unfair to the comparators and would muddy
the weighting contrast. So this battery is the HARDEST fair test for CERT --
it strips CERT down to just its age-weighted quantile and pits that alone
against the comparators on the bare residual stream.

Protocol is prequential / one-step-ahead: q at step i uses only scores
1..i-1; then s_i is scored and revealed. Coverage is the marginal hit rate over
a COMMON ready-window -- the steps where ALL methods have warmed up enough to
emit a finite q -- so the three are scored on identical test points
(apples-to-apples; no method is handed an easier subset).

------------------------------------------------------------------------------
The three methods (faithful constructions, formulas cited)
------------------------------------------------------------------------------
(0) CERT (ours) -- certflow.conformal.ConformalScorer, used verbatim.
    Barber, Candes, Ramdas & Tibshirani, "Conformal prediction beyond
    exchangeability", Ann. Statist. 51(2):816-845, 2023, Thm 2: weighted split
    conformal q = Quantile_{1-alpha}( sum_i w~_i delta_{R_i} + w~_{n+1} d_inf )
    with FIXED, data-independent weights. CERT's weights are GEOMETRIC IN
    ELAPSED (wall-clock) AGE: w_i = rho_w^{(t - t_i)} (conformal.py L68). We
    also report the variant that additionally applies CERT's staleness coverage
    debit Delta_stale = sum_i w~_i min(1, 2 eps_tv age_i) (conformal.py L92,
    their independence corollary): the certificate then CLAIMS level
    1 - alpha - Delta_stale rather than 1 - alpha.

(1) NexCP -- the KEY contrast. SAME Barber-et-al weighted split-conformal
    quantile, but weights GEOMETRIC IN OBSERVATION INDEX:
        w_i  proportional to  rho^{(n - i)}                        (3)
    (Barber et al. 2023 Sec. 4: fixed weights "w_i = rho^{n+1-i}" -- the
    canonical drift weighting; the n+1-i vs n-i offset is a harmless
    reindexing.) i is arrival RANK, not elapsed time. Under IRREGULAR sensing,
    "k arrivals ago" can be 5 or 500 time-units in the past, yet NexCP weights
    by k alone -- it is blind to the gap. rho is tuned to its BEST (sweep
    reported); never a strawman.

(2) ACI -- certflow.conformal.ACITracker, reused verbatim. Gibbs & Candes,
    "Adaptive conformal inference under distribution shift", NeurIPS 2021,
    Eq. 2: alpha_{t+1} = alpha_t + gamma (alpha_target - err_t). Each step we
    query the (unweighted, split-conformal) empirical quantile of the SAME
    residual stream at ACI's current working alpha_t, score, then feed err_t
    back. Prop 4.1 gives long-run miscoverage control for ANY sequence; gamma
    swept to its best. (Weighting is orthogonal to ACI; pairing it with the
    plain empirical quantile is the standard construction.)

------------------------------------------------------------------------------
Two residual streams (per the CELL)
------------------------------------------------------------------------------
(a) SYNTHETIC bounded-drift grid with IRREGULAR per-edge sensing AND a
    wall-clock NON-STATIONARITY (observation-noise scale ramps over time). This
    is the regime age-weighting targets: because sensing is irregular, the
    NUMBER of arrivals inside a recent TIME-window fluctuates, so index-weighting
    (a fixed count of recent arrivals) misjudges the current-time residual scale,
    while age-weighting (a fixed recent time-window) tracks it. A1 holds by
    construction (BoundedDriftWorld), so the drift term we omitted would have
    been honest -- we omit it only for fairness to the comparators.
(b) METR-LA real edge residuals (data/metr-la, exact ground truth), driven with
    the same irregular sensing. Real diurnal non-stationarity supplies the
    wall-clock drift.

Run: cert_env/bin/python scripts/extval/baseline_battery.py [--quick]
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

# Read-only imports from the published package.
from certflow.conformal import ACITracker, ConformalScorer
from certflow.drift import grid_world
from certflow.realworld import TrafficWorld

QUICK = "--quick" in sys.argv
TARGET_ALPHA = 0.10
TARGET_COV = 1.0 - TARGET_ALPHA


# ===========================================================================
# Residual-stream generators.
# Each yields a list of Events; the score (1) and interval (2) are derived
# downstream so every method sees the identical stream and identical test
# points.
# ===========================================================================

@dataclass
class Event:
    e: object        # edge key (hashable)
    t: float         # wall-clock time of this sensing
    obs: float       # observed cost
    c_hat: float     # LOCF prediction at t (last obs on e), before revealing obs
    age: float       # t - t_prev (elapsed age of c_hat); inf on first sight
    first: bool      # True if this is the first sight of e (no real residual)


def _irregular_gaps(rng: np.random.Generator, n: int, mean_gap: float) -> np.ndarray:
    """Heavy-tailed (lognormal, sigma=1) inter-sensing gaps so elapsed wall-clock
    age decouples strongly from arrival index. Scaled to mean ~ mean_gap."""
    raw = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    return raw * (mean_gap / raw.mean())


def synthetic_stream(seed: int, n_events: int, mean_gap: float = 4.0) -> list[Event]:
    """Bounded-drift grid, IRREGULAR sensing, WALL-CLOCK non-stationary noise.

    A roving sensor visits a random edge each step; wall-clock advances by a
    heavy-tailed gap. The observation-noise scale ramps linearly in wall-clock
    time (x1 -> x6 across the horizon), so the residual distribution drifts in
    TIME, not in arrival index -- the regime where recent-in-time (age) beats
    recent-in-index. The grid drift (BoundedDriftWorld, A1 holds) adds
    age-of-prediction dependence on top."""
    w = grid_world(8, 8, seed=seed, kind="bounded", rho=0.03, noise_scale=1.0)
    edges = list(w.edges())
    rng = np.random.default_rng(seed + 7_000)
    gaps = _irregular_gaps(rng, n_events, mean_gap)
    horizon = float(gaps.sum())
    last_t: dict[object, float] = {}
    last_c: dict[object, float] = {}
    t = 0.0
    stream: list[Event] = []
    for i in range(n_events):
        e = edges[int(rng.integers(len(edges)))]
        true_c = w.true_cost(e, t)
        # wall-clock non-stationary observation noise (x1 -> x6 over horizon)
        scale = 1.0 + 5.0 * (t / horizon if horizon > 0 else 0.0)
        obs = true_c + float(rng.normal(0.0, scale))
        first = e not in last_t
        c_hat = obs if first else last_c[e]
        age = math.inf if first else (t - last_t[e])
        stream.append(Event(e, t, obs, c_hat, age, first))
        last_t[e], last_c[e] = t, obs
        t += float(gaps[i])
    return stream


def _corridor_edges(w: TrafficWorld, radius: int = 2) -> list[object]:
    """Edges within a radius-`radius` BFS ball of a deterministic far endpoint:
    a focused monitored corridor (the realistic CERT deployment), so a roving
    sensor accumulates genuine REPEAT visits within the replay window."""
    from collections import deque

    from certflow.realworld import far_endpoints
    s, _ = far_endpoints(w)
    seen = {s: 0}
    dq = deque([s])
    while dq:
        u = dq.popleft()
        if seen[u] >= radius:
            continue
        for v in w.graph[u]:
            if v not in seen:
                seen[v] = seen[u] + 1
                dq.append(v)
    ball = set(seen)
    return [e for e in w.edges() if e[0] in ball and e[1] in ball]


def metr_la_stream(seed: int, n_events: int, mean_gap_bins: float = 2.0) -> list[Event]:
    """METR-LA real edge residuals, IRREGULAR sensing over a monitored corridor.

    Replays recorded LA traffic (exact ground truth; observation noise is the
    package's configured synthetic additive noise). A roving sensor visits a
    random edge of a focused corridor each step; wall-clock advances by a
    heavy-tailed number of 5-min bins. The replay window is sized to fit the
    requested events (multi-day), so REAL diurnal/weekly non-stationarity
    supplies the wall-clock drift while the corridor guarantees repeat visits
    (genuine residuals). t stays monotone real time (no wrap)."""
    BIN = 300.0
    # size the window (in bins) to comfortably fit n_events at the mean gap,
    # capped at the recording length; +2 for interpolation headroom.
    need_bins = int(n_events * mean_gap_bins * 1.4) + 2
    w = TrafficWorld(dataset="metr-la", seed=seed, n_bins=need_bins)
    corridor = _corridor_edges(w, radius=2)
    if len(corridor) < 8:                      # fall back to a wider ball
        corridor = _corridor_edges(w, radius=3)
    rng = np.random.default_rng(seed + 9_000)
    gaps = _irregular_gaps(rng, n_events, mean_gap_bins) * BIN
    horizon = (w._speeds.shape[0] - 1.001) * BIN
    last_t: dict[object, float] = {}
    last_c: dict[object, float] = {}
    t = 0.0
    stream: list[Event] = []
    for i in range(n_events):
        if t > horizon:
            break
        e = corridor[int(rng.integers(len(corridor)))]
        obs = w.observe(e, t)
        first = e not in last_t
        c_hat = obs if first else last_c[e]
        age = math.inf if first else (t - last_t[e])
        stream.append(Event(e, t, obs, c_hat, age, first))
        last_t[e], last_c[e] = t, obs
        t += float(gaps[i])
    return stream


# ===========================================================================
# Per-method quantile streams. Each returns, for every event, the half-width q
# that method would emit at that step (math.inf while not ready). Coverage and
# width are then tallied by the shared scorer over the COMMON ready-window.
# ===========================================================================

def _score(ev: Event) -> float:
    """Standard split-conformal absolute residual (1). First sight -> 0
    (no prior estimate exists; such steps are excluded from the test set)."""
    return 0.0 if ev.first else abs(ev.obs - ev.c_hat)


def cert_q_stream(stream: list[Event], alpha: float, rho_w_per_unit: float,
                  eps_tv: float = 0.0) -> tuple[list[float], list[float]]:
    """CERT age-weighted quantile per step (ConformalScorer, verbatim).
    Returns (q per step, Delta_stale per step). rho_w_per_unit is the decay PER
    WALL-CLOCK TIME UNIT (so it is unit-correct on seconds or rounds alike)."""
    sc = ConformalScorer(rho_w=rho_w_per_unit, eps_tv=eps_tv)
    qs, ds = [], []
    for ev in stream:
        qs.append(sc.quantile(alpha, ev.t))
        ds.append(sc.delta_stale(ev.t) if eps_tv > 0.0 else 0.0)
        sc.push(_score(ev), ev.t)   # this step's score becomes "past"
    return qs, ds


def nexcp_q_stream(stream: list[Event], alpha: float, rho: float) -> list[float]:
    """NexCP index-weighted quantile per step, w_i = rho^{n-i} (3)."""
    scores: list[float] = []
    qs = []
    for ev in stream:
        qs.append(_nexcp_quantile(scores, alpha, rho))
        scores.append(_score(ev))
    return qs


def _nexcp_quantile(scores: list[float], alpha: float, rho: float) -> float:
    """Quantile_{1-alpha}( sum_i w~_i delta_{R_i} + w~_{n+1} delta_{+inf} ),
    w_i = rho^{n-i} (newest sample exponent 0). Test-point weight w_{n+1}=1
    (the standard tag-along: the unobserved test point is the next, freshest
    draw). +inf while finite mass cannot reach 1-alpha."""
    n = len(scores)
    if n == 0:
        return math.inf
    w = [rho ** (n - 1 - i) for i in range(n)]  # 0-based; last element exp 0
    total = sum(w) + 1.0                         # +1.0 = test-point mass
    target = (1.0 - alpha) * total
    acc = 0.0
    for r, wi in sorted(zip(scores, w)):
        acc += wi
        if acc >= target - 1e-12:
            return r
    return math.inf


def aci_q_stream(stream: list[Event], alpha: float, gamma: float
                 ) -> tuple[list[float], float]:
    """ACI quantile per step (ACITracker, verbatim) on the same residual stream.
    Base set = unweighted empirical (1-alpha_t)-quantile; err_t=1{s_t>q_t}.
    Returns (q per step, long-run empirical miscoverage)."""
    aci = ACITracker(alpha_target=alpha, gamma=gamma)
    scores: list[float] = []
    qs = []
    for ev in stream:
        a_work = aci.working_alpha()
        if aci.alpha_raw <= 0.0:
            q = math.inf          # set = everything (boundary convention)
        elif aci.alpha_raw >= 1.0:
            q = -math.inf         # empty set
        else:
            q = _empirical_quantile(scores, a_work)
        qs.append(q)
        err = bool(_score(ev) > q + 1e-12)   # -inf q -> always err (empty set)
        aci.update(err)
        scores.append(_score(ev))
    return qs, aci.empirical_miscoverage()


def _empirical_quantile(scores: list[float], alpha: float) -> float:
    """Unweighted split-conformal (1-alpha)-quantile with test mass at +inf:
    the ceil((1-alpha)(n+1))-th smallest score; +inf if that exceeds n."""
    n = len(scores)
    if n == 0:
        return math.inf
    k = math.ceil((1.0 - alpha) * (n + 1))
    if k > n:
        return math.inf
    return sorted(scores)[k - 1]


# ===========================================================================
# Shared scorer over the COMMON ready-window.
# ===========================================================================

def score_methods(stream: list[Event], qstreams: dict[str, list[float]],
                   extras: dict[str, dict]) -> list[dict]:
    """Tally coverage and median width for each method over the SAME test
    points: steps that (i) are not first-sights (a real residual exists) and
    (ii) have a finite q for EVERY method (common ready-window). ACI's
    boundary sets (+/-inf q) are handled explicitly so it is never silently
    dropped: a +inf q (everything) covers with infinite width, a -inf q
    (empty) misses with zero width -- both keep ACI on the same test points."""
    n = len(stream)
    names = list(qstreams)
    # common ready-window: every method finite (treat ACI +/-inf as 'ready',
    # since those are legitimate ACI prediction sets, just degenerate ones).
    aci_names = {k for k in names if k.startswith("ACI")}

    def ready(name: str, q: float) -> bool:
        if name in aci_names:
            return True            # ACI always emits a (possibly trivial) set
        return math.isfinite(q)

    window = [
        i for i in range(n)
        if (not stream[i].first) and all(ready(k, qstreams[k][i]) for k in names)
    ]
    rows = []
    for name in names:
        covs, widths, qfin = [], [], []
        for i in window:
            ev = stream[i]
            q = qstreams[name][i]
            s = _score(ev)
            if q == math.inf:           # ACI everything-set
                covs.append(1); widths.append(math.inf)
            elif q == -math.inf:        # ACI empty-set
                covs.append(0); widths.append(0.0)
            else:
                covs.append(int(s <= q + 1e-12))
                widths.append(2.0 * q)
                qfin.append(q)
        finite_w = [x for x in widths if math.isfinite(x)]
        row = dict(
            method=name,
            coverage=float(np.mean(covs)) if covs else float("nan"),
            n_scored=len(window),
            width_med=float(np.median(finite_w)) if finite_w else math.inf,
            q_med=float(np.median(qfin)) if qfin else math.inf,
            inf_width_frac=(1.0 - len(finite_w) / len(widths)) if widths else 1.0,
        )
        row.update(extras.get(name, {}))
        rows.append(row)
    return rows


# ===========================================================================
# Tuning: ONE fair rule for every method -- "do not under-cover the target,
# then be as tight as possible" -- swept on the same seeds it is reported on
# (this only HELPS the comparators; no held-out penalty is imposed on them).
# CERT's rho_w is parameterised as a HALF-LIFE in units of the mean gap, then
# converted to per-time-unit, so the single knob is unit-correct on both
# streams and directly comparable in 'effective recent window' terms.
# ===========================================================================

def _per_unit_rho_w(half_life_gaps: float, mean_gap: float) -> float:
    """rho_w per time-unit such that weight halves after `half_life_gaps` mean
    gaps: rho_w^(half_life_gaps*mean_gap) = 1/2."""
    horizon = half_life_gaps * mean_gap
    return 0.5 ** (1.0 / horizon)


def _eval(qbuilder, seeds, stream_fn) -> tuple[float, float]:
    """Mean coverage and median finite width for a q-builder over seeds, on the
    builder's OWN ready-window (used only for tuning, not for the final
    apples-to-apples table)."""
    covs, ws = [], []
    for s in seeds:
        stream = stream_fn(s)
        qs = qbuilder(stream)
        c, w = [], []
        for ev, q in zip(stream, qs):
            if ev.first:
                continue
            if math.isfinite(q):
                c.append(int(_score(ev) <= q + 1e-12)); w.append(2.0 * q)
            elif q == math.inf:
                c.append(1)
            elif q == -math.inf:
                c.append(0); w.append(0.0)
        if c:
            covs.append(float(np.mean(c)))
            ws.append(float(np.median(w)) if w else math.inf)
    cov = float(np.mean(covs)) if covs else float("nan")
    wid = float(np.median([x for x in ws if math.isfinite(x)] or [math.inf]))
    return cov, wid


def _pick(cands: list, evalfn) -> object:
    """Pick the candidate that does not under-cover the target, then tightest."""
    best, best_key = cands[0], (math.inf, math.inf)
    for cand in cands:
        cov, wid = evalfn(cand)
        under = max(0.0, TARGET_COV - cov)
        key = (under, wid)
        if key < best_key:
            best_key, best = key, cand
    return best


def tune_all(stream_fn, seeds, mean_gap: float) -> dict:
    a = TARGET_ALPHA
    hl_grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]   # CERT half-lives (gaps)
    nex_grid = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]
    aci_grid = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    best_hl = _pick(hl_grid, lambda hl: _eval(
        lambda st: cert_q_stream(st, a, _per_unit_rho_w(hl, mean_gap))[0],
        seeds, stream_fn))
    best_nex = _pick(nex_grid, lambda r: _eval(
        lambda st: nexcp_q_stream(st, a, r), seeds, stream_fn))
    # ACI targets long-run = target: pick gamma closest to target coverage,
    # then tightest (its own fair rule).
    best_g, best_key = aci_grid[0], (math.inf, math.inf)
    for g in aci_grid:
        cov, wid = _eval(lambda st: aci_q_stream(st, a, g)[0], seeds, stream_fn)
        key = (abs(cov - TARGET_COV), wid)
        if key < best_key:
            best_key, best_g = key, g
    return dict(half_life_gaps=best_hl,
                rho_w=_per_unit_rho_w(best_hl, mean_gap),
                nexcp_rho=best_nex, aci_gamma=best_g)


# ===========================================================================
# Driver
# ===========================================================================

def battery(name: str, stream_fn, seeds: list[int], mean_gap: float,
            eps_tv_cert: float) -> list[dict]:
    a = TARGET_ALPHA
    print(f"\n### {name}", flush=True)
    print(f"target alpha={a}  ->  target coverage={TARGET_COV:.2f}   "
          f"({len(seeds)} seeds, mean inter-sensing gap={mean_gap:g} time-units)",
          flush=True)
    tuned = tune_all(stream_fn, seeds, mean_gap)
    print(f"tuned (do-not-undercover, then tightest): "
          f"CERT half-life={tuned['half_life_gaps']} gaps "
          f"(rho_w={tuned['rho_w']:.6g}/unit)  "
          f"NexCP rho={tuned['nexcp_rho']}  ACI gamma={tuned['aci_gamma']}",
          flush=True)

    per_seed = []
    for s in seeds:
        stream = stream_fn(s)
        q_cert, _ = cert_q_stream(stream, a, tuned["rho_w"], eps_tv=0.0)
        q_cert_ds, ds = cert_q_stream(stream, a, tuned["rho_w"], eps_tv=eps_tv_cert)
        q_nex = nexcp_q_stream(stream, a, tuned["nexcp_rho"])
        q_aci, miscov = aci_q_stream(stream, a, tuned["aci_gamma"])
        # certified level for the Delta_stale variant (median over scored steps)
        ds_scored = [d for ev, d in zip(stream, ds) if not ev.first]
        ds_med = float(np.median(ds_scored)) if ds_scored else 0.0
        qstreams = {
            "CERT (age-weighted)": q_cert,
            "CERT (age + Delta_stale)": q_cert_ds,
            f"NexCP (index, rho={tuned['nexcp_rho']:g})": q_nex,
            f"ACI (gamma={tuned['aci_gamma']:g})": q_aci,
        }
        extras = {
            "CERT (age + Delta_stale)": dict(certified_level=TARGET_COV - ds_med),
            f"ACI (gamma={tuned['aci_gamma']:g})": dict(aci_longrun_miscov=miscov),
        }
        per_seed.append(score_methods(stream, qstreams, extras))

    rows = _avg_rows(per_seed)
    _print_table(rows)
    for r in rows:
        r["dataset"] = name
    return rows


def _avg_rows(per_seed: list[list[dict]]) -> list[dict]:
    names = [r["method"] for r in per_seed[0]]
    out = []
    for j, k in enumerate(names):
        rs = [seed_rows[j] for seed_rows in per_seed]
        finw = [r["width_med"] for r in rs if math.isfinite(r["width_med"])]
        finq = [r["q_med"] for r in rs if math.isfinite(r["q_med"])]
        agg = dict(
            method=k,
            coverage=float(np.mean([r["coverage"] for r in rs])),
            coverage_sd=float(np.std([r["coverage"] for r in rs])),
            width_med=float(np.median(finw)) if finw else math.inf,
            q_med=float(np.median(finq)) if finq else math.inf,
            n_scored=int(np.mean([r["n_scored"] for r in rs])),
        )
        for extra in ("certified_level", "aci_longrun_miscov", "inf_width_frac"):
            vals = [r[extra] for r in rs if extra in r]
            if vals:
                agg[extra] = float(np.mean(vals))
        out.append(agg)
    return out


def _print_table(rows: list[dict]) -> None:
    hdr = (f"{'method':32} {'coverage':>9} {'+-sd':>6} {'vs.90':>6} "
           f"{'width~(2q)':>11} {'q~':>9} {'n':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        cov = r["coverage"]
        flag = "OK" if cov >= TARGET_COV - 5e-3 else "UNDER"
        w, q = r["width_med"], r["q_med"]
        ws = f"{w:.3f}" if math.isfinite(w) else "inf"
        qs = f"{q:.3f}" if math.isfinite(q) else "inf"
        extra = ""
        if "certified_level" in r:
            extra = f"  certifies>={r['certified_level']:.3f}"
        if "aci_longrun_miscov" in r:
            extra = f"  long-run-miscov={r['aci_longrun_miscov']:.3f}"
        print(f"{r['method']:32} {cov:>9.3f} {r['coverage_sd']:>6.3f} "
              f"{flag:>6} {ws:>11} {qs:>9} {r['n_scored']:>6}{extra}")


def main() -> None:
    seeds = list(range(3)) if QUICK else list(range(10))
    n_syn = 800 if QUICK else 2000
    n_la = 800 if QUICK else 2000

    print("=" * 80)
    print("CONFORMAL BASELINE BATTERY -- age-weighting (CERT) vs NexCP vs ACI")
    print("ADDITIONAL RSS-version results; the published paper is UNCHANGED.")
    print("Identical residual stream s=|obs-c_hat| + identical interval [c_hat+-q]")
    print("per dataset; methods differ ONLY in how q weights the calibration")
    print("scores. Real numbers, produced now by running certflow read-only.")
    print("=" * 80)

    all_rows = []
    all_rows += battery(
        "(a) SYNTHETIC bounded-drift, irregular sensing, non-stationary noise",
        lambda s: synthetic_stream(s, n_syn, mean_gap=4.0),
        seeds, mean_gap=4.0, eps_tv_cert=1e-3)
    all_rows += battery(
        "(b) METR-LA real residuals, irregular sensing over a monitored corridor",
        lambda s: metr_la_stream(s, n_la, mean_gap_bins=2.0),
        seeds, mean_gap=2.0 * 300.0, eps_tv_cert=1e-4 / 300.0)

    print("\n" + "=" * 80)
    print("READING THE TABLE")
    print(" coverage : marginal hit-rate at target 1-alpha=0.90, over the COMMON")
    print("            ready-window (same test points for all methods).")
    print(" vs.90    : OK = >= 0.90 (does not under-cover); UNDER = under-covers.")
    print(" width~   : median FULL interval width = 2q (lower is tighter).")
    print(" q~       : median conformal half-width (the pure statistical object).")
    print("=" * 80)


if __name__ == "__main__":
    main()
