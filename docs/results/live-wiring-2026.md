# Live-wiring the round-2 calibrators — real METR-LA benchmark (2026-07)

The round-2 calibrators (PASC joint per-edge radius; WATCH conformal test
martingale + Shiryaev-Roberts detector) were standalone and tested but not wired
into the live `round()` loop. This note records wiring them behind
`PlannerConfig` flags (all default **OFF**) and benchmarking them on **real
METR-LA** traffic replay against the recording's true costs.

Reproduce: `PYTHONPATH=src python scripts/run_live_wiring.py` (20 seeds × 288
rounds = one replay day each; JSON at `results/live_wiring/table.json`).
Synthetic-grid checks: `pytest tests/test_live_wiring.py`.

## What was wired (flag-gated, defaults unchanged)

- **`watch_monitor=True`** — inside `ingest_observation`, each new score's
  weighted conformal p-value (against the current buffer, with the scorer's
  age-weights) feeds a `ConformalTestMartingale` (`planner.watch`) **and** a
  `ShiryaevRobertsDetector` (`planner.sr`). `planner.diagnostics()` exposes the
  martingale value/alarm, the SR peak/alarm, the recent-vs-buffer residual drift
  score (W1) and the age-weights' effective sample size. Purely observational —
  it changes **no** certificate and **no** pricing (verified: on/off produce a
  byte-identical `(lb, ub, confidence)` stream).
- **`path_calibration="pasc"`** — in `_q(path_len)`, when supported, edges are
  priced with the PASC **joint** radius `Q` (age-weighted `(1-α)` quantile of the
  per-block **max signed score**, disjoint length-`path_len` blocks of the live
  buffer) instead of the per-edge Bonferroni `α/L` quantile. Falls back to
  Bonferroni while the buffer holds no full block (warm-up) or `α`-annealing
  pins the level at 1. **Signed** block-max (not `|·|`): the live buffer stores
  the drift-adjusted score `|obs−ĉ| − ρ·age`, so `abs()` (as the standalone
  `pasc_edge_radius` uses for *raw* residuals) would double-count the drift
  subtraction and blow the radius up. Signed-max is the exact joint sibling of
  the per-edge Bonferroni quantile, which consumes the same signed buffer.

Soundness is preserved: PASC's `Q ≥ max_e s_e` jointly (prob `≥ 1-α` under block
exchangeability) gives `Q + ρ·age_e ≥ |obs_e − ĉ_e|` for every edge at once — the
same per-edge magnitude bound Bonferroni certifies, calibrated jointly instead of
via the `α/L` union. The age-weighted retrofit inherits the same weighted-
coverage argument (Barber et al. 2023 Thm 2) as the rest of the module.

## Real METR-LA results (20 seeds × 288 rounds, `rho_quantile=0.95`)

Coverage = fraction of valid rounds with `LB ≤ true OPT ≤ UB` (oracle = exact
Dijkstra on the recording). Gap = certified width `UB − LB` (seconds).

| mode | valid % | violation rate | gap median (s) | gap mean (s) |
|------|--------:|---------------:|---------------:|-------------:|
| Bonferroni (default)      | 94.7 % | **0.0000** | 8797.1  | 9901.7  |
| PASC                      | 94.7 % | **0.0000** | 11006.6 | 11980.1 |
| Bonferroni + watch_monitor| 94.7 % | **0.0000** | 8797.1  | 9901.7  |

`watch_monitor` diagnostics (20 seeds): **martingale quiet 20/20, SR quiet
20/20** (zero false alarms), mean residual-drift score 1509.3 s, mean effective
sample size 178.1.

## Verdict

**WATCH monitor — HOLDS (positive).** On real METR-LA the staleness/weighting
null the certificate assumes is **not** violated: both detectors stayed quiet
across all 20 replay days, exactly consistent with the measured 0.0000
coverage-violation rate. The pinned-at-1.0 coverage is now a *live, alarming*
quantity rather than an untestable claim, at zero cost to the certificate. A
mid-run drift-jump stress test (synthetic, `tests/test_live_wiring.py`) confirms
the Shiryaev-Roberts detector fires on a genuine regime change (the plain
martingale intentionally stays flat after a long null — precisely why WATCH pairs
it with the implicitly-restarting SR detector).

**PASC — does NOT cut real METR-LA width (honest negative).** PASC *increases*
median certified width by **+25.1 %** (8797 → 11007 s) at equal (zero)
violations — the opposite of the synthetic 4×4 grid, where it was **4.5 %
tighter** (pooled median, `tests/test_live_wiring.py`). Mechanism (verified):
real optimistic paths are **long** (L ≈ 14–18) while the calibration buffer holds
≤ 288 scores, so the number of length-L blocks is tiny — **1 block early in the
run, 18 at full buffer** — and the joint block-max quantile is coarse and
conservative for most of the day (PASC wider than Bonferroni in 819/1090 paired
rounds). Per-edge Bonferroni, which estimates its `α/L` quantile from the **full
pooled** buffer of individual scores, stays far better-resolved. PASC only wins
when paths are **short relative to the buffer** (the grid, L = 6). 

Consequence: the default stays Bonferroni; PASC keeps its experimental flag. This
tracks the program's meta-lesson — the soundness/verification layer (WATCH,
coverage) survives real data; the width-tightening claim (PASC as "the real
Bonferroni replacement") breaks on it. The remaining real-data width win is more
likely CIA's `√L` path-sum bound (block length = L, but pricing the *sum* not the
per-edge max) or a longer/again-weighted buffer that supplies enough length-L
blocks — not the joint per-edge max.
