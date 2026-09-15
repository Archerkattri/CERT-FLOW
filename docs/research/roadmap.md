# CERT-FLOW v4 weakness-closure roadmap

This is the working release plan for the active improvement goal. A phase is
not complete because a benchmark is favorable; it is complete only when the
implementation, focused tests, full-run artifact, and gate are all present.

## Research decisions

1. **Selection is a validity boundary.** Adaptive model/path selection must be
   recorded and evaluated on a fresh audit stream. CAP is the reference design
   for selection-conditional conformal prediction:
   <https://jmlr.org/beta/papers/v26/24-0452.html>.
2. **Conditional claims are finite-family claims.** Group guarantees require a
   declared group family and sparse-group abstention. The new group calibrator
   uses alpha allocation and held-out lower bounds; it does not claim universal
   conditional coverage. Reference:
   <https://academic.oup.com/jrsssb/article/87/4/1100/8058684>.
3. **Drift alarms need formal sequential evidence.** The operational recovery
   gate is separate from the false-alarm claim. The new sequential monitor uses
   a conformal e-value detector; adaptive step-size recovery will be compared
   with ACI under time-varying distributions. Reference:
   <https://www.jmlr.org/beta/papers/v25/22-1218.html>.

## Closure gates

- [x] `G1` Group/conditional validity: independent calibration and audit
  streams; every declared group has support, coverage, one-sided LCB, and an
  explicit invalid/abstain state when support is insufficient.
- [x] `G2` Post-selection validity: selection ledger digest, fresh audit
  support, and a held-out selected-path audit on synthetic and traffic data.
- [x] `G3` Drift/recovery: formal alarm and recovery metrics (detection delay,
  false alarms, recovery delay, post-recovery coverage) against ACI, SR, and
  the existing heuristic manager under matched streams.
- [x] `G4` Decision quality: repair the external-baseline runner so every
  policy has nonzero, comparable goal/regret rows under identical worlds,
  budgets, seeds, and validity/abstention accounting.
- [x] `G5` Width/coverage: tune only on calibration seeds; report held-out
  coverage LCB, median/p90 width, and worst-group coverage. No test-set tuning.
- [x] `G6` Fleet and sensing: add correlated/shared-edge stress, budget-matched
  VOI/max-age/random controls, and joint fleet coverage/regret gates.
- [x] `G7` Systems/reproducibility: benchmark manifest, source revision,
  environment, dataset hashes, seed list, configuration hash, and a checker
  that rejects quick/incomplete artifacts.
- [x] `G8` Release decision: CERT-FLOW must beat the prior run and the strongest
  valid baseline on each promoted metric, or the report must name the exact
  unresolved tradeoff instead of claiming a win.

## Execution order

1. Land reusable validity/recovery primitives and tests (`G1`–`G3`).
2. Repair and extend competitive runners (`G4`–`G6`).
3. Add manifest/checker enforcement (`G7`).
4. Run calibration-only sweeps, lock configurations, then run held-out full
   benchmarks and evaluate `G8`.

All declared v4 gates are now closed by persisted full-run artifacts and the
strict checker. The active goal remains open only for the next research cycle:
improving abrupt-shift recovery beyond the current honest abstention behavior,
not for missing evidence in the current release lanes.

## Frontier follow-up (2026-09-13)

The first frontier intervention is complete. The formal SR-backed gate now
defaults to the historical scalar betting shape with five fresh post-alarm
samples, while the fixed multi-epsilon SR mixture remains an opt-in research
variant. On the matched 12-seed shift benchmark, post-shift edge coverage is
unchanged at 0.954, detection remains 23 rounds, fresh-evidence recovery drops
from 10 to 5 rounds, and valid re-entry is 5 rounds. The mixture was measured
but rejected for promotion because its full-run median detection increased to
26.5 rounds. The strict checker now enforces the accelerated five-sample
artifact and the no-false-alarm/coverage gates together.
