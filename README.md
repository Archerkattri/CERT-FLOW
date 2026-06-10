<p align="center"><img src="assets/banner.svg" alt="CERT-FLOW" width="100%"/></p>

<p align="center">
  <a href="#reproducing-every-number"><img alt="tests" src="https://img.shields.io/badge/tests-223%20passing-0072B2"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-56B4E9">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-1a7f37">
  <img alt="coverage claim" src="https://img.shields.io/badge/certificate%20coverage-1.000%20measured-D55E00">
</p>

A robot replanning through a world whose costs drift faces a question classical
planners never answer: **how good is my current route, given that most of the
map is stale?** CERT-FLOW answers it every round, with a proof: a
high-probability certificate `LB ≤ OPT ≤ UB` on the optimal route cost, built
from age-weighted non-exchangeable conformal prediction over drift-adjusted
observation residuals — and it spends paid sensing exactly where the
certificate says the gap shrinks fastest.

<p align="center"><img src="assets/overview.png" alt="One CERT round" width="92%"/></p>

## Why it's different

| | classical replanning (D\* Lite, AD\*) | exchangeable conformal (CIA) | **CERT-FLOW** |
|---|---|---|---|
| stale map | silently trusts it | coverage collapses (0.95 → **0.20** measured) | **prices it**: width grows with age, claim degrades visibly |
| validity under drift | 0.02–0.59 measured | gap-dependent | **0.95–1.00, every condition ever run** |
| sensing | none / heuristic | none | **certificate-directed** (oracle-level regret) |
| static regime | fast | tight | **proof-gated preprocessing**: ns–µs queries that self-expire |

## Headline results (all reproducible below)

- **Coverage ≥ claimed confidence on every condition ever run** — 17 synthetic
  regimes, off-model worlds, and two real cities (METR-LA, PEMS-BAY) at up to
  49% drift-model violation rates.
- **Route quality**: exactly optimal on known maps (≡ Dijkstra, plus the
  certificate); travel-regret −0.12 ≈ a clairvoyant oracle in unknown drifting
  terrain; 2–3× lower regret than freshness/uncertainty/random sensing at
  equal budget.
- **Speed**: 3.7 ms p50 / 12 ms p95 per fully-certified round at 60×60 (one
  CPU core). Certificate-gated preprocessing answers static queries in
  **269–394 ns** (cost) / 8.7 µs (path) — at or below published static-SOTA —
  and at road scale absorbs cost changes in **0.015–0.34 ms vs ~1 s** for
  CRP-style recustomization, exact under ±20% perturbation.
- **Theory T1–T7**: coverage (observable + latent), a certifiability
  *threshold* (gap ε is sustainable iff sensing rate beats drift — both
  directions), a √L sum-aware upper certificate with a measured
  selection-bias hazard and its gate, an **impossibility theorem** (no uniform
  lower bound can beat Bonferroni by more than log factors — the certificate's
  asymmetry is optimal), decision-uniform validity, and a churn-measured floor.
- **Honest negatives, kept**: the corridor-memory speed hypothesis failed
  (documented), a predictor's regime claim was downgraded after its test, and
  the maze negative-control shows exactly where route-critical sensing cannot
  help.

## Quickstart

```bash
python -m venv cert_env && source cert_env/bin/activate
pip install -e ".[dev,fast]" pandas h5py tables   # "fast" = numba (needed to reproduce the speed numbers)
pytest   # full suite: 223 with datasets; data-dependent tests skip cleanly without data/
python - <<'PY'
from certflow.cert import CertPlanner, PlannerConfig
from certflow.drift import grid_world

world = grid_world(6, 6, seed=0, kind="bounded", rho=0.02, noise_scale=0.05)
planner = CertPlanner(world, (0, 0), (5, 5),
                      PlannerConfig(epsilon=5.0, alpha_prime=0.2))
for _ in range(150):
    cert, sensed = planner.round()
print(f"[{cert.lb:.2f}, {cert.ub:.2f}] @ confidence {cert.confidence:.2f}, "
      f"gap {cert.gap:.2f}")
PY
```

## Reproducing every number

Every quantitative claim traces to a script; the core sweep runs in ~100 s on
a multicore machine (`CERTFLOW_WORKERS=N` parallelizes seeds bit-identically).

| Result | Script | Documented in |
|---|---|---|
| Tier-0 coverage (17 conditions, provable + strict modes) | `scripts/run_tier0.py` | `docs/results/tier0-coverage.md` |
| CERT vs Gaussian (path level) | `scripts/run_tier0_baselines.py` | `docs/results/tier0-coverage.md` |
| Edge-level audit (Gaussian break) | `scripts/run_gaussian_break.py` | `docs/results/gaussian-break.md` |
| Incremental repair latency (T3) | `scripts/run_tier1_latency.py` | `docs/results/tier1-latency.md` |
| Ablations (κ churn, pre-widening) | `scripts/run_ablations.py` | `docs/results/ablations.md` |
| Travel regret, unknown terrain | `scripts/run_tier2.py` | `docs/results/tier2-regret.md` |
| Real traffic (METR-LA / PEMS-BAY) | `scripts/run_metr_la.py [--pems-bay]` | `docs/results/metr-la.md` |
| MovingAI maps + maze negative control | `scripts/run_movingai.py` | `docs/results/movingai.md` |
| External algorithms (AD\*, VOI, TASP-degenerate) | `scripts/run_extern_baselines.py` | `docs/results/extern-baselines.md` |
| CIA exchangeability collapse | `scripts/run_cia_comparison.py` | `docs/results/cia-comparison.md` |
| E-Graphs + networkx anchors | `scripts/run_repeated_queries.py` | `docs/results/extern-baselines.md` |
| Lifelong missions (memory vs memoryless) | `scripts/run_lifelong.py` | `docs/results/lifelong.md` |
| Feature regimes (predictor, decision-uniform) | `scripts/run_feature_regimes.py` | `docs/results/feature-regimes.md` |
| Scale + engine benchmarks | `scripts/run_scale.py` | `docs/results/scale.md` |
| Road networks (DIMACS NY/FLA, ALT) | `scripts/run_roadnet.py` | `docs/results/published-speed-comparison.md` |
| Certified Contraction Hierarchies | `scripts/run_ch.py` | `docs/results/published-speed-comparison.md` |

All scripts accept `--quick`. Real-data runs need `data/` (sources and loaders
in `data/README.md`; ~230 MB total, links inside).

## How it works

1. **Score** every paid observation with a drift-adjusted residual; weight by
   age (data-independent geometric weights — exchangeability is *not* assumed).
2. **Price** each edge as `ĉ ± (λq + ρ·age)`: the conformal quantile pays for
   noise, the drift term pays for staleness.
3. **Bound** the optimum from both sides with two incremental searches
   (optimistic ℓ, conservative u) over a flat-array engine (numba kernels).
4. **Claim** `LB ≤ OPT ≤ UB` at an honestly-annealed confidence — weak claims
   during warm-up instead of silence; the claim visibly decays as the map ages.
5. **Sense** the edge that shrinks the certified gap fastest (route-critical,
   churn-aware); certification is a *rate*, not a state (T2′).
6. **When the certificate proves the map tight**, that proof licenses
   preprocessing — an all-pairs oracle or certified Contraction Hierarchy
   answering in ns–µs — revoked the instant drift exceeds tolerance.

## Layout

```
src/certflow/
  types.py      contracts (World, EdgeBelief, Certificate)
  conformal.py  weighted non-exchangeable quantiles, Δ_stale, ACI, blocks
  cert.py       the planner: certify → sense → repair loop, gates, annealing
  sensing.py    gap-shrink selection + baseline policies
  fastgraph.py  flat-array CSR engine (numba D* Lite, Dijkstra kernels)
  snapshot.py   certificate-gated all-pairs oracle (ns queries)
  ch.py         certified Contraction Hierarchies (231 µs on 264k-node NY)
  roadnet.py    DIMACS road graphs + exact ALT on landmark lower-bounds
  drift.py / realworld.py / movingai.py   synthetic, traffic-replay, game maps
  episodes.py / harness.py / baselines.py runners, seeds, parametric strawman
docs/results/   one markdown per experiment — numbers, anomalies, verdicts
docs/specs/     design spec; docs/theory/ working notes
```

## Citation

Paper: *CERT: Certified Route Planning under Drifting Costs* (preprint
forthcoming — citation entry will be updated with the arXiv ID).

```bibtex
@misc{attri2026certflow,
  author = {Attri, Krishi},
  title  = {{CERT-FLOW}: Certified Route Planning under Drifting Costs},
  year   = {2026},
  url    = {https://github.com/Archerkattri/CERT-FLOW}
}
```

## License

MIT — see [LICENSE](LICENSE).
