# CERT: Conformal Edge-cost Routing under Time-drift — Research & System Design Spec

**Companion theory note:** `docs/theory/theory-notes.md` (T2′ derivation,
verified theorem statements, binding design constraints — that note is normative; this
spec references it rather than restating proofs). Two foundational decisions:
(1) the certificate is grounded in staleness-corrected conformal prediction, not
Gaussian μ±βσ; (2) the headline claim is *sense-to-certify under drift*; the
flow-memory is an optional ablation module.

---

## 1. One-paragraph summary

CERT is an online path planner for graphs whose edge costs are continuous, uncertain,
and drift over time. It maintains, at every replanning round, a high-probability
certificate `LB ≤ OPT ≤ UB` on the true optimal route cost — built from age-weighted
non-exchangeable conformal intervals on edge costs — and allocates paid sensing actions
specifically to shrink the certified gap on route-critical edges, repairing its two
searches incrementally (D* Lite/LPA*) rather than replanning from scratch. It stops
sensing when `UB − LB ≤ ε`, and it can *prove when no ε is sustainable* given the drift
rate and sensing bandwidth (the certifiability threshold, Theorem T2′). Target venue:
ICRA/RSS first; the theory core (T1 + T2′) could alternatively carry a NeurIPS/L4DC
submission.

## 2. Problem statement and assumptions

**Given:** directed graph `G=(V,E)`, start `s`, goal `g`. True edge cost `c_e(t) > 0`
unknown; the robot can pay sensing cost `m_e` to observe `c_e` (noisily) at any time
(remote sensing allowed; locality constraints are a config option). Point estimates
`ĉ_e` from last observation at time `t_e`; age `a_e(t) = t − t_e`.

**Assumptions (stated up front, all ablated in experiments):**
- **A1 — bounded cost drift:** `|c_e(t') − c_e(t)| ≤ ρ_e (t'−t)` with known (or
  conservatively over-estimated) `ρ_e`.
- **A2 — TV-Lipschitz residual drift:** the distribution of observation residuals
  shifts in total variation at most `ε_TV` per unit age (the assumption that powers
  the conformal coverage correction; distinct from A1, see theory note §5.4.5).
- **A3 — sensing model:** one observation per period Δ (rate budget), additive
  observation noise with unknown distribution (no Gaussianity assumed — that is the
  point of conformal).

**Objective:** reach `g` minimizing realized travel + sensing cost, while at every
round reporting a valid certificate `(LB, UB, confidence)` and stopping sensing once
`UB − LB ≤ ε`.

## 3. Novelty positioning (the cell we occupy and its neighbors)

Verified by five independent literature sweeps (June 2026). The unoccupied
intersection: **{certificate-as-sensing-objective} × {pay-to-sense} × {drifting/stale
costs} × {online incremental maintenance}**. Must-cite-and-distinguish table:

| Prior | What it owns | What it lacks (our delta) |
|---|---|---|
| TASP (Weiss et al. 2023) | explicit path-cost LB/UB from bounded estimators | bounds tightened by *computation*, static costs, no sensing |
| ARA*/AD* (Likhachev et al. 2003/05) | measured suboptimality certificate | known costs; gap closed by search effort, not observation |
| CTP + Remote Sensing (Bnaya & Felner 2009) | pay-to-sense edges, VOI placement | expected-cost objective, binary blockages, no certificate, no drift |
| PAC combinatorial pure exploration (Chen et al. 2017; CSALE 2022) | (ε,δ)-certified best path by sampling arms | no sensing cost vs travel trade, no execution loop, stationary |
| BISECT / LazySP / GLS (2016–19) | near-optimal "which edge to evaluate" | binary validity, computational evaluation, no cost certificate |
| Traversing Mars (Ott et al. 2024) | scout sensing until path "proven optimal/infeasible" | deterministic stopping rule; no probabilistic bound, staleness, or incremental reuse — we strictly generalize it |
| IPP w/ guaranteed estimation uncertainty (Jakkala et al. 2026) | certified *field* variance via sensing | certifies the map uniformly, not the route decision |
| Luo & Zhou 2024 | conformal path-cost sums | exchangeable only; no drift, weights, sensing, or online loop |
| CQR-GAE (Tang et al. 2025) | conformal edge intervals → robust SP | exchangeable, one-shot, no coverage propagation, no sensing |
| Persistence filter (Rosen 2016) / FreMEn (Krajník 2017) / BRULE (2025) | staleness as map/feature uncertainty | none couples age → cost-interval inflation → route certificate |
| E-Graphs (Phillips 2012) / ACO / Physarum | memory-as-heuristic; decaying edge scalars | no observation-age decay on guaranteed incremental search (κ module's narrow delta) |

Claims we explicitly do NOT make: first certified planner; first active-sensing
replanner; first staleness model; first bio-inspired memory. The claim is the
intersection, plus T1's `Δ_stale` and T2′ as new theoretical objects.

## 4. Algorithm design

### 4.1 Per-edge state

`(ĉ_e, t_e, a_e, ρ_e, m_e, κ_e)` + global rolling calibration buffer
`B = {(residual_i, age_i, terrain_class_i)}`.

### 4.2 Intervals (the certificate substrate)

```
q       = weighted conformal quantile at level α_edge over B,
          weights w_i = ρ_w^(age_i)            (data-independent: ages only)
ℓ_e(t)  = ĉ_e − q − ρ_e a_e(t)     (clipped at cost floor > 0)
u_e(t)  = ĉ_e + q + ρ_e a_e(t)
```

Per-edge coverage: `1 − α_edge − Δ_stale(ages; ρ_w, ε_TV)` (T1, theory note §2, §5.1).
Path-level: Bonferroni `α_edge = α′/L` (default); sum-aware score (Luo & Zhou marriage)
is the stretch upgrade. Realized confidence `1 − α′ − ΣΔ_stale` is **reported with the
certificate every round** — confidence degrades visibly with staleness, never silently.

### 4.3 Main loop (per replanning round)

```
1. Age update: recompute ℓ_e, u_e for all edges whose age bucket changed.
2. LB search:  P_lb = incremental shortest path under ℓ   →  LB = ℓ-cost(P_lb)
3. UB search:  UB = u-cost(P_lb) if feasible under u, else u-cost of the
               conservative shortest path (incremental, second D* Lite instance)
4. Certifiability check (T2′): if ε < 2Lq + ρ̄ Δ L(L−1), declare ε unattainable;
   degrade to smallest sustainable ε̂ or trigger deadline fallback. Never sense forever.
5. If UB − LB ≤ ε: emit CERTIFIED(P, LB, UB, confidence); continue executing.
6. Else sense: over E_crit = edges(P_lb) ∪ edges(P_ub) ∪ edges of the k ℓ-shortest
   paths with ℓ-cost ≤ (1+δ)·LB   (k, δ config; default k=3, δ=0.1),
   pick argmax expected (UB−LB) shrink per unit sensing cost
   — backstopped by age-triggered forced round-robin over edges(P_lb)
     (the rule that makes T2′(a) hold for the deployed policy, not just the analysis).
7. Update: ĉ_e, t_e ← observation; push residual into B (sensing = calibration);
   ACI update α_t+1 = α_t + γ(α − err_t) on realized coverage events.
8. Incremental repair only where intervals changed; goto 1.
```

### 4.4 Execution policy

The robot executes only the conservative incumbent (the path validated under `u`).
LB exists for the certificate, never for execution. Replanning happens concurrently
with motion; sensing may be remote or en-route per config.

### 4.5 Optional module: conductivity memory κ (ablation, not headline)

`κ_e` reinforced when `e` lies on certified incumbents, decayed with age. Used ONLY to
warm-start search ordering / priority initialization — never in `ℓ/u`, never in
conformal weights (theory note §5.4.1). Kill-gate: stays in the paper only if it shows
≥20% median replanning-latency reduction vs plain D* Lite AND beats E-Graphs at equal
budget; thrashing/hysteresis check mandatory. Otherwise it ships as a negative-result
ablation paragraph.

## 5. Theory targets

- **T1 (staleness-corrected coverage):** per-edge and path-level certificate validity
  with explicit `Δ_stale` from Barber Thm 2 + A2; sharper age-profile-dependent form
  is the contribution (generic corollary gives `2ε_TV/(1−ρ_w)`).
- **T2′ (certifiability threshold):** ε sustainable iff
  `ε ≥ 2Lq + ρ̄ Δ L(L−1)` (achievability via round-robin; impossibility via cut
  pigeonhole at `2mq + ρ_min Δ m(m−1)`). Static case: finite termination; ρ→0, q→0
  recovers Traversing Mars's stopping rule.
- **T3 (incremental repair):** repair cost scales with the locally-changed region
  (inherited from LPA*/D* Lite, restated for interval changes).
- **Stretch:** sum-aware non-exchangeable path score (kills the Bonferroni n₀
  burden); regret-style bound for the greedy+round-robin sensing policy.

**Ordering: T1/T2′ assumptions are frozen BEFORE Phase-1 code** — the drift model
dictates the interval-update law, not the other way around.

## 6. System architecture (v1, simulation-first)

Python core (NumPy; Rust/C++ port only if profiling demands it). Components, each
independently testable:

| Component | Responsibility | Interface |
|---|---|---|
| `graphcore` | dual incremental D* Lite/LPA* instances over ℓ and u | `update_edge(e, ℓ, u)`, `shortest_path(metric)` |
| `conformal` | rolling buffer B, weighted quantile, Δ_stale, ACI α-tracker | `quantile(α_edge)`, `push(residual, age)`, `confidence()` |
| `drift` | drift models (bounded-rate, jump, periodic, adversarial-TV) for both planner assumption and world generation | `widen(e, a_e)`, `step_world(t)` |
| `sensing` | E_crit construction, gap-shrink VOI, round-robin backstop | `next_observation()` |
| `oracle` | ground-truth Dijkstra per round, coverage logging | `opt(t)`, `covered(LB, UB, t)` |
| `harness` | seeds, tiers, metrics, Clopper–Pearson aggregation, plots | `run(config) → logs` |
| `baselines` | D* Lite, AD*, TASP-style, CTP-RS-style, CPE-style, μ±βσ, E-Graphs, κ-on/off | same `harness` API |

## 7. Experimental plan

**Tier 0 — coverage validation (the headline table):** synthetic dynamic grids;
on-model drift ρ ∈ {0, low, med, high} + off-model stress (jumps, periodic,
adversarial-within-TV-budget); ≥1000 seeds; empirical coverage with Clopper–Pearson
vs claimed `1−α′−ΣΔ_stale`; misspecified-ρ sweep (ρ̂/ρ ∈ [0.5, 2]). μ±βσ as the
calibration baseline we beat on coverage.

**Tier 1 — known-map dynamic metrics:** local cost updates; latency + incumbent
quality vs D* Lite, AD*; κ ablation + E-Graphs comparison + thrashing check live here.

**Tier 2 — unknown/drifting terrain, single robot:** gap, time-to-ε-certificate,
sensing budget, path-time regret vs revealed oracle; baselines: CTP-RS-style VOI,
CPE-style sampling, generic info-gain IPP, no-certificate D* Lite.

**Tier 3 (cut-first if schedule slips):** scout-follower vs Traversing Mars — does
probabilistic gap-shrinking reduce scouting time vs their deterministic stopping rule?

**Primary metrics:** empirical coverage; certificate gap; time-to-ε; replanning
latency (wall-clock); sensing spend; regret; certifiability-declaration correctness.
Iteration counts secondary only (Physarum-evaluation lesson). Coverage is claimed
**only** in simulation; any field/realistic tier demonstrates utility, not coverage.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Δ_stale constants too loose at realistic ages | ACI safety net carries long-run claim; report both; loose-Δ_stale is degraded-not-dead |
| Bonferroni n₀ burden (α/L small ⇒ big buffer) | warm-up with coarse α′ annealed tight; terrain-class pooling (ablated); stretch sum-aware score |
| Greedy sensing starves edges | round-robin/age-triggered backstop is part of the deployed policy (not just analysis) |
| κ self-confirming / thrashing | κ outside certificate by construction; hysteresis test; kill-gate |
| ρ_e unknown in practice | conservative over-estimate + misspecification sweep; online ρ estimation listed as future work, not claimed |
| ε unattainable under fast drift | certifiability check declares it (T2′) and degrades gracefully — a feature, reported as such |
| Scope overrun (solo researcher) | Tier 3 pre-committed as first cut; hardware out of scope for v1 |

## 9. Non-goals (v1)

No real hardware; no VLM/learned priors (slot exists in `conformal` for any point
predictor, exploration deferred); no multi-robot beyond optional Tier 3; no kinodynamic
feasibility (graph-level routing; Hybrid-A*/RRT* composition is future work); no online
learning of ρ_e.

## 10. Milestones (theory-first reordering of the original timeline)

1. **M1 (≈ Jul 2026):** freeze A1/A2, write T1 + T2′ proofs to paper-draft level.
   *Gate: proofs check out before code.*
2. **M2 (Aug):** `graphcore` + `conformal` + `drift` + `oracle`; Tier-0 coverage runs.
   *Gate: empirical coverage ≥ claimed line on-model.*
3. **M3 (Sep–Oct):** `sensing` + baselines; Tier 1–2.
4. **M4 (Nov):** κ ablation + kill-gate decision; stretch theorem attempt.
5. **M5 (Dec):** Tier 3 or cut; paper writing.
6. **M6 (Jan–Feb 2027):** polish, rebuttal package, submission.
