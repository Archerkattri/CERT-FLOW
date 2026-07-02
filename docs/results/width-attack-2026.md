# Width attack 2026 — sum-level UB calibration and the test-then-tighten license

*Head-to-head of every width-relevant pricing option on identical worlds, plus
the new flag-gated `ShrinkLicense` (Tier-2, a-posteriori) — where the certified
gap can be tightened soundly, by how much, and what claim each number carries.*

**Reproduce:** `scripts/run_width_attack.py` (10 seeds; `--quick` for 3).
Output: `scripts/out/width_attack.json`. Tests: `tests/test_shrink_license.py`.

> **Finding —** The union-bound tax is real and recoverable where paths are
> long: on real METR-LA the two SUM-LEVEL upper-bound calibrations tighten the
> certified gap 24–27% at zero violations, while the block-max construction
> (PASC) stays +25% wider — sums calibrate where maxima starve. The largest
> reduction (−62%) comes from the a-posteriori shrink license, which carries a
> deliberately weaker, anytime-valid empirical claim — Tier-2, never a
> replacement for the distribution-free certificate.

## Modes

| mode | what changes | claim carried |
|---|---|---|
| default | per-edge Bonferroni (shipped) | a-priori, distribution-free |
| sum_aware | `sum_aware_ub=True` (T4 block-quantile UB) | a-priori, distribution-free |
| pasc | `path_calibration="pasc"` | a-priori (block-exchangeable) |
| cia-ub | UB := upper end of `cia_path_certificate()` on the incumbent (feasible path ⇒ valid UB on OPT); default UB when CIA unsupported | a-priori; **stacks two events** (see honesty note) |
| shrink | `shrink_license=True` — certificate UNCHANGED; Tier-2 shadow radius k·(q+ρ·age) | **a-posteriori anytime-valid** (see below) |

## Real METR-LA (10 seeds × 288 rounds, paired: identical replay, n=2729 valid rounds each)

| mode | violation rate ↓ | median gap (s) ↓ | gap ratio vs default ↓ |
|---|---:|---:|---:|
| default | 0.0000 | 9 598 | 1.000 |
| sum_aware | 0.0000 | 7 329 | 0.764 |
| pasc | 0.0000 | 11 967 | 1.247 |
| **cia-ub** | 0.0000 | **7 045** | **0.734** (CIA unsupported → fallback on 20.2% of rounds) |
| shrink (Tier-1 cert) | 0.0000 | 9 588 | 0.999 (unchanged, as designed) |
| — shrink **Tier-2 shadow** | **0.0051** measured | **3 606** | **0.376** (licensed k median 0.50, k<1 on 82% of rounds) |

## Drift grid 10×10, ρ=0.02 (10 seeds × 150 rounds; modes share true costs, observation-noise draws differ — treat deltas under ~2% as noise)

| mode | violation rate ↓ | gap ratio vs default ↓ |
|---|---:|---:|
| default | 0.0000 | 1.000 |
| sum_aware | 0.0000 | 0.998 |
| pasc | 0.0000 | 0.966 |
| cia-ub | 0.0000 | 0.977 (fallback 51.3%) |
| shrink Tier-2 shadow | 0.0000 measured | **0.577** (licensed k median 0.50) |

## Findings

1. **Where the width win lives: long paths.** METR-LA incumbents run L≈14–18
   edges, so the Bonferroni per-edge level α′/L starves against the buffer's
   effective sample size; a single sum-level quantile at level α′ is
   supportable and 24–27% tighter (`sum_aware` −23.6%, `cia-ub` −26.6%) at a
   measured violation rate of 0.0000. On the short-path grid (L≈6) the union
   tax is small and every a-priori mode is within noise of the default — the
   attack pays exactly where the weakness was measured.
2. **Sums calibrate where maxima starve.** PASC (one quantile of the per-block
   MAX) needs L-length blocks and stays +24.7% wide on real traffic —
   confirming the live-wiring result — while the two SUM constructions
   (block-sum quantile, group-sum CIA) tighten. Same buffer, same level; the
   difference is purely which functional of the block is calibrated.
3. **Honesty note on cia-ub (why it is not yet the default):** as measured it
   takes LB from the default event (level α′) and UB from CIA's event (its own
   α′), so the joint claim is only guaranteed at ~1−2α′ a priori (measured:
   0 violations). A default flip needs the α budget split across the two sides
   and a decision on the 20% unsupported-fallback rounds — queued as the
   `round()` integration decision, per the CIA-vs-PASC head-to-head plan.
   `sum_aware_ub` has no such caveat (T4 accounting is within the certificate)
   and its −23.6% is the cleanest immediate candidate.
4. **The shrink license is a different object, and says so.** The Tier-1
   certificate is bit-identical with the flag on (asserted in tests). The
   licensed Tier-2 radius (k·(q+ρ·age), k chosen by an anytime-valid betting
   CS on the observed violation rate — Waudby-Smith & Ramdas 2023, Ville) is
   62% narrower on real traffic at a measured shadow violation rate of 0.51%
   against true OPT (target α′=20%; the license floor k=0.5 binds, so the
   stream would support more). The claim is explicitly a-posteriori: valid,
   time-uniform statements about THIS deployment's observed stream, with
   self-revocation under regime shift — NOT an a-priori guarantee for the next
   round. That trade is fundamental, not an implementation gap: any a-priori
   narrowing from windowed evidence re-assumes the exchangeability that the
   drift model exists to drop (the CIA-collapse failure mode). Deployment
   reading: safety gates consume Tier-1; resource allocation may consume
   Tier-2.
5. **Both new behaviors are flag-gated, defaults unchanged**; suite
   268 passed / 28 skipped (was 258/28).
