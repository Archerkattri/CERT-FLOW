# Limitations

Every limitation documented in the paper, theory note, and results docs, with
its disposition. Categories: CLOSED (implemented + measured or proved),
CHARACTERIZED (root cause + partial mitigation + honest bound), INHERENT
(cannot be closed in software), and AUDIT-FOUND-AND-FIXED (defects surfaced by
adversarial review and resolved).

## (a) Closed

1. **Online estimation of rho_e.** Implemented (`rho_mode="online"`): pooled
   quantile of observed |dc|/age rates; conformal absorbs the tail as in the
   misspecification analysis. Coverage unchanged at 1.000 while gaps tighten
   1.7x on synthetic (10.14 -> 5.94, cert 7% -> 24%) and 2.4x on METR-LA
   (5441s -> 2230s). The planner no longer needs any world-supplied drift bound.
2. **Provable-mode validity cost (was 9% valid).** Closed by alpha annealing:
   96% valid at coverage 1.000.
3. **Off-model degradation now bounded.** theory.tex Lemma (A1-violation
   robustness): coverage loses at most the violation mass, worst case; measured
   violations visible to calibration are absorbed (traffic: 5-49% violation
   rates, coverage 1.000).
4. **pi_cal now bounded.** pi_cal <= 4 f_max <rho a>_cal under a bounded-density
   addition to A3 (theory.tex lemma), with <rho a>_cal tracked per run.
5. **Selection-bias residual.** Resolved by construction: with the freshness
   gate, gate-closed rounds use the unconditional Bonferroni bound and gate-open
   rounds are conditionally valid (data splitting). No uncontrolled mass remains.
6. **Synthetic observation noise on traffic.** Stress-closed: noise-free replay
   (pure staleness) gives coverage 1.000 and nearly identical gaps (5573s vs
   5441s given-rho; 2772s vs 2230s online-rho) — conclusions are
   staleness-dominated, not noise-driven.
7. **ESS/rho_w footgun.** Planner warns at construction when 1/(1-rho_w) cannot
   support alpha_prime without annealing.
8. **Maintenance/backstop ablation rows uninformative at eps=5.** Rerun at
   eps=12 (above the T2' floor) makes them speak — see ablations.md.
9. **O4/H1 (the failed objective) — resolved in its intended setting.**
   Within-mission latency still honestly fails (pre-widening owns it). Lifelong
   (Tier-L): full memory re-certifies 4.3x faster at 4.2x less sensing than
   memoryless restarts; the ablation shows calibration carryover -> instant
   validity, beliefs -> route knowledge, kappa -> zero marginal speed (stability
   only). See lifelong.md.
10. **Sum-aware lower bound (open theory thread).** Closed by impossibility
    (theory.tex T5): on layered graphs any valid uniform LB pays
    Omega(L*sigma*sqrt(ln w)) slack (greedy-path posterior argument); per-edge
    Bonferroni matches up to log factors, so the certificate's asymmetry is a
    theorem, not a limitation.
11. **Churn factor (open theory thread).** Closed by T7 + measurement: the
    floor and adaptive rate use the online-tracked churn measure K-hat; focused
    sensing suppresses churn (K 59 -> 11 ~ L; rotation refuted: same cert%, +20%
    spend); certification in the test regime improved 5.6% -> 36.7% across the
    churn-directed changes at coverage 1.000.
12. **Static-grid speed boundary — crossed where provable.**
    Certificate-gated preprocessing (snapshot.py): when the certificate proves
    the map tight, an all-pairs oracle on certified estimates serves cost queries
    in 269-394 ns and path queries in 8.7 us with explicit per-query certificates
    and drift-triggered expiry. Road scale: ALT on landmark lower-bounds absorbs
    bounded cost changes in 0.015-0.067 ms vs CRP's ~1 s; a certificate-friendly
    CH reaches the CH-class query (231 us on NY) and absorbs +-20% changes for a
    0.34 ms array write. See published-speed-comparison.md.

## (b) Characterized (honest residual)

13. **P_lb-churn factor above the T2' floor.** Attacked four ways (focused
    sensing, stabilized sensing target, online rho, gap-stall rate feedback).
    Online rho delivered the win (gap 14.5 -> 9.5, cert 5.6% -> 23% at rho=0.05,
    eps=8); stabilization and feedback add little. Residual ~1.6x with a
    structural root cause: unsensed edges' lower bounds fall with age, so
    optimism attracts P_lb to the stalest region; a sound uniform LB cannot
    ignore stale-cheap regions — the residual is the price of soundness.
14. **Uniform sum-aware lower bound for general graphs.** The layered-graph
    impossibility (item 10) bounds what any construction can achieve; the cover
    construction for arbitrary graphs remains open.

## (c) Inherent (not closable in software)

15. **Field coverage verification.** Coverage requires ground truth every round;
    replayed recordings provide it (two cities), live robot deployments cannot.
    Field tiers demonstrate utility.
16. **Maze-type topologies.** With no route alternatives, route-critical sensing
    provably cannot differentiate (MovingAI negative control); this is the
    claim's boundary, not a defect.

## (d) Audit-found-and-fixed

17. **Unobserved-edge soundness hole.** The Traversing-Mars degenerate ablation
    (T2' corollary made executable) exposed that never-observed edges entered the
    certificate with prior-centered intervals — the prior is not an observation
    and no theorem prices it; noise masked the hole everywhere else. Fixed:
    unobserved edges are unknown (ell at floor, u unbounded), so certification
    requires a fully observed path. The degenerate test now verifies the
    certified incumbent is exactly optimal in the noise-free static corner.
18. **Path corollary constant (GAP-A).** The path corollary must carry
    L_max <= |V|-1, not L, to cover the unknown optimum — statement fixed;
    strict_lb_alpha mode implements the exact constant (valid 76.5%, coverage
    1.000, claims ~0.13).
19. **Block-level staleness charge (GAP-B).** block_delta_stale was dead code;
    sum-aware confidence now charges the block-level staleness term,
    conservatively via max.
20. **Forensics audit dispositions.** 12/12 claims traced to executable code
    (2 reproduced bit-identically); mutation testing confirmed tests
    fail-on-break; datasets authentic (METR-LA replay bit-exact vs raw h5);
    29/30 citations exact — 1 author-attribution defect fixed
    (rockenbauer2025traversing, was ott2024), 2 transcription cells corrected.
