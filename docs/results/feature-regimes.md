# Feature-regime validation: spatial predictor (dense sensing) and decision-uniform

*Two opt-in CERT-FLOW features stress-tested in their intended regimes — the spatial predictor under dense sensing (METR-LA), and decision-uniform certificate spending — reporting honest, mostly-negative outcomes.*

**Reproduce:** `scripts/run_feature_regimes.py`

> **Finding —** Neither feature delivers its headline benefit in these regimes: the dense-sensing predictor gain at the highest sensing rate is real but tiny (and it reverses at the moderate rate, and is a warm-up artifact at the lowest rate), while decision-uniform mode is indistinguishable from baseline here except for the mechanically-expected confidence inflation. Soundness (full coverage) holds throughout.

---

## Experiment A — Spatial predictor in its designed dense-sensing regime (METR-LA)

**Setup.** 6 seeds x 200 rounds, TrafficWorld(seed, n_bins=200, offset_base_bins=20000)
(validation windows past the predictor's training region, no leakage).
`traffic_planner_config(rho_mode="online")`. Predictor: P2 ridge regression fit on
the first 18,000 METR-LA bins (`fresh_age=6*300s`). Observations per round k in {1, 4, 8}:
one standard `planner.round()` observation plus (k−1) additional edges chosen by
max-age across ALL graph edges via `planner.ingest_observation(e)` — simulating a
dense reporting sensor network rather than a single rover.

**Hypothesis.** With many observations per round neighborhoods stay fresh and the P2
predictor pays. Predictor-on should beat predictor-off on gap at k=8, be
neutral-or-negative at k=1 (replicating the documented negative from the sparse regime).

Rows ordered best -> worst by the primary metric (gap median, lower is better).
Coverage is tied across every condition (soundness held throughout), so it carries no
bold.

| condition | valid rounds ↑ | coverage ↑ | gap median (s) ↓ | mean confidence ↑ | pred_used_rounds · |
|---|---:|---:|---:|---:|---:|
| k=1, pred=on  |   867 | 1.000 | **2961.5** | 0.407 |   4846.5 |
| k=4, pred=off |  1160 | 1.000 | 2969.6 | 0.651 |      0.0 |
| k=4, pred=on  |  1156 | 1.000 | 3075.3 | 0.579 |  16959.8 |
| k=8, pred=on  |  1169 | 1.000 | 3305.7 | 0.613 |  25843.7 |
| k=8, pred=off |  **1184** | 1.000 | 3385.2 | **0.663** |      0.0 |
| k=1, pred=off |  1038 | 1.000 | 3528.6 | 0.577 |      0.0 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

Soundness held throughout: coverage 1.000 in all six conditions.

**Findings — honest.**

The dense-regime claim is **partially supported at k=8 and k=1, refuted at k=4**:

- **k=8:** predictor-on shows a gap improvement of 2.3% (3305.7 s vs 3385.2 s) and
  nearly the same number of valid rounds (1169 vs 1184). The per-bin annealing charge
  still costs mean confidence (0.613 vs 0.663), but the gap direction is correct.
  The improvement is small — an order of magnitude below the +42% the offline study
  projected — because even with k=8 additional sensing the predictor's benefit is
  diluted: the conformal width shrinks only when the predictor reliably outperforms
  LOCF across the age distribution, but most edges are still relatively fresh
  (age < the 12-bin gate where P2 materially helps) at 8 observations per 5-minute bin.

- **k=1:** predictor-on shows a larger gap improvement (2961.5 vs 3528.6, −16%), but
  also 171 fewer valid rounds. The lower valid-round count reflects per-bin annealing
  suppressing certificate issuance during predictor warm-up; the gap metric is
  conditioned on valid rounds only, creating selection bias. The apparent improvement
  cannot be attributed to predictor quality at k=1 — at this sensing rate
  neighborhoods are stale (replicating the documented negative mechanism) and the
  predictor is inferring from sparse context. The result is therefore an artifact of
  differential warm-up, not a genuine P2 gain.

- **k=4:** predictor-on is *worse* on gap (3075.3 vs 2969.6, +3.5%) while having
  nearly identical valid counts (1156 vs 1160). This is an honest refutation at k=4.
  The moderate sensing rate produces neighborhoods that are partially fresh — enough
  to trigger the predictor on many edges but not fresh enough for the predictor to
  improve on LOCF, exactly the mixed-regime predicted by the spatial-predictor study's
  half-fresh sensitivity analysis.

**Conclusion for Exp A.** The dense-regime hypothesis receives only weak support.
At k=8 the gap trend is in the right direction but tiny; at k=4 it reverses; at k=1
the apparent improvement is a warm-up artifact. The offline +42% upper bound requires
ALL neighbors fresh at query time, which does not hold even at k=8 in a route-focused
network. The predictor's structural cost — per-bin annealing charge reducing claimed
confidence — is consistent and non-trivial. **Recommendation: predictor remains opt-in;
the designed regime (fixed sensor networks reporting continuously at high rates) remains
plausible but was not demonstrated here.**

---

## Experiment B — Decision-uniform mode (T6)

**Setup.** 6×6 BoundedDriftWorld, rho=0.01, noise_scale=0.05; epsilon=5.0, alpha'=0.2,
eps_tv=1e-4, delta=1.0, rho_w=0.99. 12 seeds x 300 rounds.
`decision_uniform=False` vs `decision_uniform=True` (max_decisions=5: claim level
alpha'/5 = 0.04 per certificate instead of 0.2). Acted-on rounds = certified rounds
(cert.valid AND gap <= epsilon AND confidence >= min_certify_confidence).
Episode-level metric: fraction of episodes where every acted-on certificate was
geometrically valid (LB <= true OPT <= UB).

Rows ordered best -> worst by the primary distinguishing metric (mean confidence, higher
is better). Every other ranked column is tied between conditions (valid%, coverage, cert%,
gap median, and ep_all_valid all identical), so only mean confidence carries a bold.

| condition | valid% ↑ | coverage ↑ | cert% ↑ | gap median ↓ | mean confidence ↑ | ep_all_valid ↑ |
|---|---:|---:|---:|---:|---:|---:|
| decision_uniform=on  | 96.7% | 1.000 | 22.1% | 5.77 | **0.696** | 1.000 |
| decision_uniform=off | 96.7% | 1.000 | 22.1% | 5.77 | 0.647 | 1.000 |

*↑ higher is better · ↓ lower is better · · informational · **bold** = best*

**Findings.**

Coverage is 1.000 in both conditions and ep_all_valid is 1.000 for both. The T6 quantity
(episode-level all-certifications-valid) cannot be distinguished between on/off at this
regime because the per-round certificate is already conservative enough that no
miscoverage occurs at all — the cert% and gap are identical between conditions.

The visible effect of decision-uniform is a **+0.049 increase in mean claimed confidence**
(0.696 vs 0.647) at unchanged cert rate. This is the correct mechanical signature:
decision-uniform tightens the effective alpha per decision (0.04 vs 0.20), which means
the planner claims a higher confidence level for each certificate it does issue. The
trade-off — paying width at alpha'/N_dec instead of alpha' — does not visibly manifest
here because coverage is already far above the 1-alpha' floor (slack absorbed by
conservative ACI + tight q at rho=0.01).

The T6 distinction becomes material in regimes where the per-round miscoverage rate
is non-negligible (heavier drift, adversarial incident injection, or the T1b provable
mode at alpha'=0.2). Under those conditions the decision-uniform spending prevents the
union-bound failure-accumulation over N_dec decisions. In this clean regime, its effect
is captured entirely in the confidence inflation — correct behavior by construction.

**Conclusion for Exp B.** Decision-uniform mode behaves as designed:
(i) it increases claimed confidence per certificate (+0.049) at unchanged cert%,
(ii) ep_all_valid = 1.000 throughout (consistent with the T6 guarantee that the
mode prevents false certification at trajectory level), and (iii) cert% and gap are
identical between conditions (the alpha-spending does not gate MORE or FEWER
certifications — only the confidence level differs). The mode is appropriately
conservative for mission-critical use where every departure decision must be
simultaneously valid.
