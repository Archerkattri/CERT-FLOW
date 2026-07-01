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
