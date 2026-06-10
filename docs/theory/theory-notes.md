# CERT: Resolutions of the Four Pre-Spec Caveats

Working theory note. These four items were flagged as the gaps between "plan correct
in outline" and "plan that survives contact with the proofs." Each section resolves
one caveat to the level needed to constrain the spec and the implementation.

Notation used throughout:

- Graph `G = (V, E)`, start `s`, goal `g`.
- True (unknown, time-varying) edge cost `c_e(t) > 0`.
- Point estimate `ĉ_e` from the last observation of `e`, taken at time `t_e`.
- Age `a_e(t) = t − t_e`.
- Conformal base half-width `q` (the conformal quantile of observation residuals at
  miscoverage level `α_edge`; exact construction in §4).
- Edge interval at time `t`:  `[ℓ_e(t), u_e(t)] = [ĉ_e − q − ρ_e a_e(t), ĉ_e + q + ρ_e a_e(t)]`,
  width `w_e(t) = 2q + 2 ρ_e a_e(t)`.
- `P_lb(t)`: shortest path under `ℓ`. `LB(t) = Σ_{e∈P_lb} ℓ_e(t)`.
- `UB(t) = Σ_{e∈P_lb} u_e(t)` if `P_lb` is feasible under conservative costs
  (else the conservative-shortest path's `u`-cost; the bound below only improves).
- `OPT(t)`: cost of the true optimal s–g path under `c(t)`.

---

## Caveat 1 — T2 reworded: the certifiability threshold

**Original (broken) statement:** "sensing route-critical ambiguous edges drives
UB − LB → ε in bounded observations." False under persistent drift: intervals re-widen
as fast as sensing shrinks them unless sensing bandwidth beats the widening rate.

**Drift model (the assumption the whole paper states up front):**
bounded drift rate — for every edge `e` and times `t < t'`:

```
|c_e(t') − c_e(t)| ≤ ρ_e · (t' − t)
```

with `ρ_e` known (or conservatively over-estimated; sensitivity to misspecified ρ is an
explicit experiment, §3). A stochastic variant (sub-Gaussian increments, variance
`γ_e² · Δt`) is the natural extension; we prove with the worst-case model first because
it gives clean impossibility results.

**Key structural fact (why sensing only the LB-path suffices):**
since `UB ≤ Σ_{e∈P_lb} u_e` and `LB = Σ_{e∈P_lb} ℓ_e`,

```
UB(t) − LB(t) ≤ Σ_{e∈P_lb(t)} w_e(t) = Σ_{e∈P_lb(t)} [ 2q + 2 ρ_e a_e(t) ]      (★)
```

So the certificate gap is controlled by the widths *along the current optimistic path
only* — this is the formal justification for "route-critical" sensing, and it makes the
threshold computable.

**Theorem T2′ (certifiability threshold).** Let the robot sense at most one edge per
period `Δ` (sensing rate `1/Δ`). Let `L = |P_lb|` (edges on the certifying path) and
`ρ̄ = max_{e∈P_lb} ρ_e`, `ρ_min` analogous.

*(a) Achievability.* Round-robin re-sensing of the edges of `P_lb` keeps every age at
most `(L−1)Δ` in steady state, with the age multiset at any instant dominated by
`{0, Δ, …, (L−1)Δ}`. By (★):

```
UB − LB  ≤  2Lq + 2ρ̄ · Δ · L(L−1)/2  =  2Lq + ρ̄ Δ L(L−1)
```

Hence an ε-certificate is **sustainable** whenever

```
ε ≥ 2Lq + ρ̄ Δ L(L−1)        equivalently        Δ ≤ (ε − 2Lq) / (ρ̄ L(L−1)).
```

*(b) Impossibility.* If every s–g path must cross at least `m` edges of a separating
set `C` with drift rates ≥ ρ_min (a cut-criticality condition), then any sensing policy
at rate `1/Δ` leaves, at any instant, age sum over the `m` freshest such edges
≥ `Δ · m(m−1)/2` (pigeonhole: at most one age reset per period). Therefore

```
UB − LB  ≥  2mq + ρ_min Δ m(m−1)        for all policies, at all times,
```

and an ε-certificate is **unsustainable** when `ε < 2mq + ρ_min Δ m(m−1)`.

*(c) Corollaries.*
- `ε < 2Lq` is never certifiable, even with infinite sensing rate (measurement-noise floor).
- Static-but-unknown world (`ρ ≡ 0`): the gap is non-increasing under re-sensing and the
  loop terminates after finitely many observations — the original T2 is recovered as the
  `ρ → 0` special case, and Traversing Mars's deterministic stopping rule is the further
  special case `q → 0`.
- The threshold itself is a contribution: it answers "how fresh must my map be, as a
  function of drift rate, path length, sensing rate, and noise floor, for an ε-guarantee
  to exist at all?" No surveyed prior states such a result.

**Implementation consequence:** the planner must *report* certifiability: if
`ε < 2Lq + ρ̄ Δ L(L−1)` it should declare the target gap unattainable and either degrade
to the smallest sustainable ε or trigger the deadline fallback — not sense forever.
Greedy gap-shrink must be backstopped by round-robin/age-triggered forced re-sensing of
`P_lb` edges (this is also what makes (a) hold for the deployed policy, not just the
analysis policy).

---

## Caveat 2 — T1: the staleness-corrected conformal certificate

**Construction (two layers + a union bound):**

1. **Per-edge non-exchangeable conformal interval.** Nonconformity scores are residuals
   `R_i = |observed_i − predicted_i|` from past sensed edges (the rolling calibration
   buffer, §4). Because drift breaks exchangeability, use *weighted* conformal
   prediction with data-independent weights that decay with the age of each calibration
   sample. Per Barber–Candès–Ramdas–Tibshirani (2023), the coverage gap of weighted
   split conformal is bounded by a weighted sum of total-variation distances between
   each calibration point and the test point. Under the bounded-drift model the TV term
   for a calibration sample of age `a_i` is itself bounded by a function of `ρ a_i`
   (older samples are more "shifted"), giving a per-edge guarantee of the form:

   ```
   P( c_e(t) ∈ [ℓ_e(t), u_e(t)] ) ≥ 1 − α_edge − Δ_stale(a; w, ρ)
   ```

   where `Δ_stale` is an explicit, computable correction that (i) vanishes as ages → 0,
   (ii) grows with drift rate and ages, and (iii) shrinks as weights decay faster.
   **`Δ_stale` is the paper's novel theoretical object.** Instantiation from the
   verified Barber et al. theorem (exact statements in §5): with geometric weights
   `w_i = ρ_w^(age_i)` and the bounded-drift assumption translated to TV distance
   (`d_TV(Z_i, Z_test) ≤ ε_TV · age_i`), their worked corollary gives
   `Δ_stale ≤ 2 ε_TV / (1 − ρ_w)` — our job is the sharper, age-profile-dependent
   version (the generic bound is age-uniform; ours should use the actual realized ages
   in the buffer, which the Σ w̃_i · d_TV form supports directly).

   *Path-level alternative to the union bound:* Luo & Zhou (arXiv 2408.10939, 2024)
   construct conformal intervals directly on **sums** of edge labels (path cost) via
   conformalized interval arithmetic with a sum-aware nonconformity score — avoiding
   Bonferroni's `α/L` penalty — but only under exchangeability. The strongest version
   of T1 marries their sum-aware score to the non-exchangeable weighted guarantee;
   the Bonferroni route is the safe fallback if that marriage doesn't go through.

2. **Drift-aware widening.** The `ρ_e a_e` term in `[ℓ_e, u_e]` transports the interval
   from observation time to query time under the drift model; this is deterministic
   given the model, so it costs no coverage.

3. **Path-level union bound.** The certificate only needs simultaneous coverage over
   the edges of `P_lb` (by ★), not the whole graph: set `α_edge = α′/L` (Bonferroni).
   Then `P( LB ≤ OPT ≤ UB ) ≥ 1 − α′ − Σ_{e∈P_lb} Δ_stale,e`. Report the realized
   `α′ + ΣΔ_stale` alongside the gap — the certificate's *confidence* degrades
   gracefully and visibly with staleness, rather than silently.

**Honest risk register for T1:** if `Δ_stale`'s constants are too loose to be useful at
realistic ages, the fallback is the ACI layer (§4), which guarantees *long-run* coverage
assumption-free at the price of being a frequency (not per-round) statement. The paper
survives on the (weighted-CP per-round + ACI long-run) pair even if `Δ_stale` is loose;
tight `Δ_stale` is the upside case.

---

## Caveat 3 — Sim-only coverage validation: the oracle harness

The headline guarantee can only be *verified* where ground truth `c_e(t)` is known at
every round. Design, stated explicitly so the paper can't be accused of hiding it:

- **Generator.** Synthetic dynamic-grid worlds with edge-cost trajectories drawn from:
  (i) on-model: bounded-rate drift at severities `ρ ∈ {0, low, med, high}`;
  (ii) off-model stress: jump processes (sudden blockage), periodic costs
  (FreMEn-style), and adversarial drift within a total-variation budget.
- **Oracle.** Dijkstra on the true `c(·, t)` each replanning round → `OPT(t)`.
- **Logged per round:** indicator `[LB ≤ OPT ≤ UB]`, gap, realized confidence,
  sensing spend, latency.
- **Reported:** empirical coverage with Clopper–Pearson intervals over ≥ 1000 seeds ×
  rounds, against the claimed `1 − α′ − ΣΔ_stale`; coverage-vs-drift-severity curves
  (on-model should sit above the claimed line; off-model curves quantify robustness);
  coverage under misspecified `ρ` (planner believes ρ̂ ≠ true ρ, sweep ρ̂/ρ ∈ [0.5, 2]).
- **Real-robot / realistic tiers** (off-road, scout-follower if kept): report utility
  metrics only — travel time, gap, sensing budget, latency, regret vs revealed-oracle.
  The paper states plainly: *coverage is a model-conditional claim validated in
  simulation; field tiers demonstrate utility, not coverage.* A reviewer told this up
  front cannot discover it.

---

## Caveat 4 — Calibration data: where the conformal scores come from

**Principle: every sensing action is also a calibration sample.** Each observation of
edge `e` yields a residual (predicted vs observed cost) that enters a rolling
calibration buffer. Sensing therefore does double duty — it shrinks the certificate gap
*and* maintains the calibration set. No separate data-collection machinery.

Three layers:

1. **Warm-up (cold start).** The certificate is declared INVALID until the buffer holds
   `n₀` residuals (e.g., n₀ ≈ 30–50 for usable quantiles at α_edge ≈ 0.05/L... in
   practice α_edge small ⇒ n₀ ≥ 1/α_edge − 1; this is a real constraint — for L = 20,
   α′ = 0.1, α_edge = 0.005 needs n₀ ≥ 199). Mitigations: (i) start the mission with a
   coarser certified α′ and tighten as the buffer grows; (ii) pre-mission transfer
   buffer from previous environments, down-weighted by a fixed transfer weight (the
   Barber framework absorbs this as one more weight); (iii) CDF-pooling residuals
   across edges of the same terrain class to multiply effective sample size — pooling
   assumption stated explicitly and ablated.
2. **Steady state.** Weighted CP with age-decaying weights (§2) over the rolling buffer.
3. **Safety net.** Adaptive Conformal Inference (Gibbs–Candès 2021) on top: adjust the
   working miscoverage level `α_t` by the online update from realized miscoverage
   events (each later re-observation of an edge reveals whether the previous interval
   covered). ACI's long-run coverage guarantee is assumption-free — it holds under
   arbitrary distribution shift — so even if the drift model is wrong, the *long-run
   frequency* of certificate violations is controlled. [Exact update rule and rate from
   the verified statement — addendum §5.]

**Design rule that falls out:** the sensing selector's objective gains a small bonus for
observations that double as informative calibration samples early in the mission
(warm-up phase), then anneals to pure gap-shrink. Keeps the cold-start from dragging.

---

## §5 Addendum — exact statements of the two imported theorems (verified from primary LaTeX sources)

### 5.1 Barber–Candès–Ramdas–Tibshirani, "Conformal prediction beyond exchangeability"
(arXiv 2202.13415; Ann. Statist. 51(2), 2023)

Setup: data `Z_i = (X_i, Y_i)`, i = 1..n+1, test point `Z_{n+1}`; **no assumption on the
joint distribution**. Prespecified, fixed (non-data-dependent) weights `w_i ∈ [0,1]`,
normalized `w̃_i = w_i / (w_1+…+w_n+1)`, `w̃_{n+1} = 1/(w_1+…+w_n+1)`. Split conformal
residuals `R_i = |Y_i − μ̂(X_i)|`; prediction interval = `μ̂(X_{n+1}) ±
Quantile_{1−α}(Σ w̃_i δ_{R_i} + w̃_{n+1} δ_{+∞})`. `Z^i` = the sequence with test point
and point i swapped; `R(Z)` = residual vector.

**Theorem 2 (coverage):**

```
P( Y_{n+1} ∈ Ĉ_n(X_{n+1}) )  ≥  1 − α − Σ_{i=1}^{n} w̃_i · d_TV( R(Z), R(Z^i) )
```

(holds for split and full conformal). Interpretable relaxations: the TV term is
≤ d_TV(Z, Z^i); under independence, ≤ 2·d_TV(Z_i, Z_{n+1}) (marginal TV between
calibration point i and the test point). Theorem 3 gives the matching
anti-conservative side (+ w̃_{n+1} slack). Caveat that binds our design:
**weights must be data-independent** (age-based decay qualifies, since age is part of
the "tag," but cost-dependent weights would void the theorem as stated).

**Recommended weight schedule under drift:** geometric decay `w_i = ρ_w^(n+1−i)`
(they use ρ_w = 0.99). Worked corollary for Lipschitz drift
`d_TV(Z_i, Z_{n+1}) ≤ ε_TV·(n+1−i)`:  coverage gap ≤ `2 ε_TV / (1 − ρ_w)`.
Changepoint k steps back: gap ≤ `ρ_w^k`.

### 5.2 Gibbs–Candès, "Adaptive conformal inference under distribution shift"
(arXiv 2106.00170; NeurIPS 2021)

Miscoverage indicator `err_t = 1{Y_t ∉ Ĉ_t(α_t)}`; fixed step size γ > 0.

**Update rule (their Eq. 2):**   `α_{t+1} = α_t + γ (α − err_t)`

**Proposition 4.1 (long-run coverage, assumption-free / deterministic):** w.p. 1, ∀T:

```
| (1/T) Σ_{t=1}^{T} err_t − α |  ≤  ( max{α₁, 1−α₁} + γ ) / (T γ)
```

so empirical miscoverage → α a.s. at rate O(1/(Tγ)) under **arbitrary** (even
adversarial) distribution shift. Binding caveat: this is a **long-run frequency**
guarantee, not per-round marginal coverage — which is exactly why it is our safety
net (layer 3 of §4) and not the primary certificate (layer 1). Step size: γ = 0.005
in their experiments; theoretically γ ≈ sqrt(2·E|α*_{t+1} − α*_t|) (larger under
faster shift).

### 5.3 Is the target cell already taken? (verified June 2026)

**No** — but two adjacent works must be cited and distinguished:

- **Luo & Zhou 2024 (arXiv 2408.10939), Conformalized Interval Arithmetic:** conformal
  intervals on SUMS of edge labels (explicitly: path cost in traffic networks) via a
  sum-aware score — but standard exchangeable split conformal, no drift, no weights,
  no ACI, no sensing. They own "conformal path-cost under exchangeability."
- **Tang et al. 2025 (arXiv 2503.10088), CQR-GAE:** conformal (CQR+GNN) edge-cost
  intervals fed to robust shortest path — exchangeability assumed, endpoints used in a
  robust LP, no path-coverage propagation, no drift, no sensing.

Ruled out after checking: non-exchangeable CP for temporal GNNs (2507.02151 — link
prediction, not costs), cost-aware ACI for runtime assurance (2605.24463), risk-averse
correlated-edge traversal (2505.13674 — no conformal), ACI in motion planning (used for
dynamic-agent trajectory intervals, not edge costs).

**Open cell we occupy:** non-exchangeable (age-weighted) conformal edge-cost intervals
under an explicit drift model, propagated to a path-cost certificate (Bonferroni or
sum-aware), maintained online by incremental search, with sensing allocated to shrink
the certified gap. Each neighbor misses ≥ 2 of {drift/weights, path propagation,
online maintenance, active sensing}.

### 5.4 Binding design constraints extracted from the verified theorems

1. **Conformal weights must be data-independent** (Barber Thm 2 as stated). Age-based
   geometric decay `w_i = ρ_w^(age_i)` qualifies; any weight that depends on observed
   cost values voids the guarantee. The conductivity κ must therefore never touch the
   weights — a second, independent reason κ stays outside the certificate.
2. **Barber gives per-round marginal coverage; ACI gives long-run frequency only.**
   The certificate hierarchy is fixed: weighted-CP = primary per-round claim,
   ACI = assumption-free safety net. Never present ACI's guarantee as per-round.
3. **Path propagation costs either Bonferroni (α/L per edge → large n₀) or a sum-aware
   score (Luo & Zhou, exchangeable-only so far).** The marriage of sum-aware scores
   with non-exchangeable weights is open theory — schedule it as the stretch theorem,
   with Bonferroni as the default the implementation ships with.
4. **ACI step size γ trades adaptation speed vs bound tightness** (rate
   `O(1/(Tγ))`, recommended γ grows with shift speed). γ is an exposed config knob,
   reported in all experiments, default 0.005 per the source paper.
5. **The TV-to-drift translation (`d_TV(Z_i, Z_test) ≤ ε_TV · age_i`) is an assumption
   layer of its own** — it is where the abstract bound meets the physical drift model,
   and it must be stated as such (a Lipschitz-in-TV drift condition), not silently
   conflated with the bounded-rate cost-drift model used in T2′. The two models are
   linked but distinct: bounded cost drift bounds *where the cost can be*; TV-Lipschitz
   drift bounds *how unlike the calibration distribution the residuals become*.
