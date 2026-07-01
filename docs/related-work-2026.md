# Related work (2026 update)

## Learning shortest paths when data is scarce (Matsypura et al., arXiv 2601.03629)

Matsypura et al., *"Learning Shortest Paths When Data is Scarce"* study certified
shortest-path intervals in a **scarce-data, offline/batch** regime: they fit edge
costs with a **Laplacian-regularized** estimator (neighboring edges are smoothed
toward each other), derive **sim2real bias bounds** on the learned costs, and
compose them into **union-bound anytime path intervals** over **stationary**
costs. The setting is fundamentally batch: a fixed dataset, no time axis, no
online loop.

**Positioning vs CERT-FLOW.** CERT-FLOW is the complementary regime: **online,
non-exchangeable drift**. It certifies the optimal route cost every replanning
round under costs that move over time, using age-weighted non-exchangeable
conformal quantiles plus an explicit `rho * age` drift term, and it *acts* — it
directs paid sensing at the edges that shrink the certified gap fastest. Neither
method uses the other's machinery: they have no Laplacian spatial prior and no
sim2real bias model; we have no online loop, no drift model, and no sensing.

Both, however, share the **same Bonferroni weakness**: composing per-edge
intervals into a path interval by a union bound pays an `L`-fold price on an
`L`-edge path. Our Task-4 CIA option (`path_calibration="cia"`, see below)
addresses exactly this on our side by scoring the path *sum* directly
(concentration `~sqrt(L)` instead of summing `L` per-edge margins), retrofitted
with our age weights so it survives drift.

**Hybrid (future work, complementary not competing).** A natural combination is a
**Laplacian-smoothed bias prior feeding CERT-FLOW's per-edge scores**: use their
spatial regularizer / sim2real bias bound as an informative prior on each edge's
cost, then let our online conformal + drift + sensing layer track and certify it
over time. This borrows their scarce-data strength (spatial smoothing when
observations are few) without giving up our drift validity and decision loop. It
is out of scope here and left as future work.

## Adopted 2025 conformal machinery (this revision)

- **LP-shift (Lévy–Prokhorov distribution shifts, arXiv 2502.14105).** Optional
  `shift_model="lp"` staleness model: the worst-case quantile is
  `Quant(1 - alpha + rho) + eps` and worst-case coverage `F_P(q - eps) - rho`,
  replacing the TV-Lipschitz `Delta_stale` coverage correction (`eps` = smooth
  drift per unit staleness, `rho` = mass of abruptly/adversarially changed edges).
  TV remains the default.
- **CIA (Conformalized Interval Arithmetic w/ symmetric calibration, arXiv
  2408.10939).** Optional `path_calibration="cia"`: group-sum path-level
  calibration (`ceil((1+K)(1-alpha))`-th smallest of the signed path-sum scores),
  with symmetric calibration for overlap (honest coverage `>= 1 - alpha - delta`,
  `delta` exposed) and the age-weighted-CDF drift retrofit. Replaces per-edge
  Bonferroni on the UB side; experimental, per-edge Bonferroni is the default.
- **SAOCP / SF-OGD (scale-free online gradient descent, arXiv 2302.07869, Alg.
  2).** Optional `aci_mode="sf-ogd"` step size for the ACI safety net:
  `s_{t+1} = s_t - eta * g_t / sqrt(sum_{i<=t} ||g_i||^2)`, anytime and
  scale-free (no `gamma` tuned to the err/score magnitude). Fixed-`gamma` ACI is
  the default.

## Round 2 (this revision): joint per-edge calibration + a testability layer

The round-1 CIA option tightens the path *sum*. Round 2 adds the joint *per-edge*
Bonferroni replacement and, more importantly, turns the pinned-at-1.0 coverage
into an **observable** quantity — the single most-cited weakness of the
certificate.

- **PASC (Pipeline-Aware Conformal Prediction, arXiv 2605.18812).**
  `PASCCalibrator` prices every edge of a path at one radius `Q` = the weighted
  `(1-alpha)` quantile of the per-path **maximum** absolute residual. By the set
  identity `∩_e {s_e ≤ Q} = {max_e s_e ≤ Q}` (PASC Prop. 4, guarantee-agnostic),
  a fresh path's edges are **jointly** within `±Q` with probability `≥ 1-alpha` in
  a single scalar quantile — no `alpha/L` correction. Since `Q` is the module's
  already-valid split-conformal quantile with the `∪{+inf}` test point, soundness
  does not rest on any PASC-specific level constant; the age-weighted variant
  inherits the same Barber et al. weighted-coverage argument as CIA. PASC's own
  finite-sample proof assumes group exchangeability (its weighted extension is
  stated as future work); we surface `delta` (max pairwise edge-overlap) as the
  honest caveat.
- **WATCH (weighted conformal test martingale, arXiv 2505.04608).**
  `ConformalTestMartingale` bets on the weighted conformal p-values
  (`conformal_p_value`, WATCH Eq. 9) with a decreasing betting density; the wealth
  is a nonnegative supermartingale, so Ville gives `P(sup_t M_t ≥ 1/delta) ≤
  delta` under the (weighted-)exchangeability null. Two uses: (1) a **validity
  monitor** that alarms when the staleness/weighting model breaks; (2) a
  **tightness stress test** — replay with a shrunken radius, and the largest
  shrink that keeps `M_t` below the alarm is the tightest *safe* certificate. This
  is the observable the pinned-at-1.0 coverage never provided.
- **Conformal e-values (Vovk–Wang; arXiv 2503.13050).** `conformal_e_value`
  (betting form) and `score_ratio_e_value` (canonical `S_test / mean(S)`, Balinsky
  Eq. 4) with `E[E] ≤ 1` under the null (Markov alarm at `1/alpha`).
  `merge_e_values` implements the two admissible merges: **average** (valid under
  arbitrary dependence — the safe merge across a path's edges) and **product**
  (the sequential test martingale). The score-ratio e-value is `~1` for a
  trivially-wide/uninformative certificate and `≫1` when a score is anomalously
  large — the requested "collapses toward the null when uninformative" diagnostic.
- **Shiryaev-Roberts change detector (WATCH Prop 3.3).**
  `ShiryaevRobertsDetector` (`R_t = (1 + R_{t-1}) e_t`) is the companion to the
  test martingale for detecting a violation *after a long null run*: a plain
  martingale can random-walk toward zero over a long null and then miss a late
  change, whereas SR restarts implicitly every step (`E[R_t] = t` under the null,
  ARL `>= threshold`) and catches it. `CertPlanner.pasc_edge_radius()` exposes
  PASC's joint per-edge radius on the live buffer.
- **Empirical demonstration.** `scripts/run_watch_testability.py` shows all three
  observables on controlled streams where the ground truth is known: the validity
  monitor stays flat with coverage tracking `1-alpha`; the SR detector catches a
  sharp regime shift ~7 rounds after it (peak `~3e8` vs threshold `1e4`) where the
  plain martingale, decayed over the long null, misses it; and — honestly — the
  Bonferroni-vs-PASC width gap appears only under *positive edge correlation*
  (Bonferroni is tight under independence): at correlation `0.9` on an `L=20`
  path, Bonferroni over-covers (`0.97`) while PASC holds `~0.91` at **16.5%** less
  width. Under independence PASC barely helps — reported, not hidden.
- **DASC (drift-aware spectral CP, arXiv 2606.15953) — diagnostics only, by
  design.** DASC's coverage theorem is *not* distribution-free (it depends on
  unknown Lipschitz/mismatch constants), and its drift-gated calibration weights
  become label-dependent, which would forfeit the hard `LB ≤ OPT ≤ UB` guarantee.
  We therefore adopt only its genuinely-valid observables: `residual_drift_score`
  (the 1-D Wasserstein drift magnitude `D_t`) and `effective_sample_size` (Kish
  `n_eff`), exposed as a drift dashboard alongside the WATCH martingale — **not**
  wired into the coverage-critical weights. DASC's adaptive-`alpha` update
  (Theorem 3) is already present as `ACITracker`.
