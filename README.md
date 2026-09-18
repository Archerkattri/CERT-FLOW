# CERT-FLOW

## Certified route planning under drifting costs

CERT-FLOW is a route planner that knows when its map is stale. Every planning
round returns an explicit interval

```text
LB <= OPT <= UB
```

with a stated confidence level, then spends its sensing budget where new
observations reduce the certified decision gap. The system combines
age-weighted conformal prediction, drift accounting, dual incremental search,
active sensing, live validity monitoring, selection-aware audits, fleet
certificates, and proof-gated fast routing.

The design principle is simple: speed is useful only while the evidence that
supports the route is still valid.

<p align="center">
  <a href="https://github.com/Archerkattri/CERT-FLOW/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Archerkattri/CERT-FLOW/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-56B4E9">
  <img alt="MIT license" src="https://img.shields.io/badge/license-MIT-1a7f37">
  <img alt="Tests" src="https://img.shields.io/badge/tests-337%20passing-009E73">
  <img alt="Release gates" src="https://img.shields.io/badge/release%20gates-83%2F83-009E73">
</p>

<p align="center"><a href="https://archerkattri.github.io/CERT-FLOW/">Project page and visual demos →</a></p>

![Current CERT-FLOW system dashboard](assets/current_system_dashboard.png)

The dashboard is generated from the persisted full-run fleet, recovery, and
real-world audit artifacts. It shows budgeted sensing, abrupt-shift recovery,
fresh conditional audits, and release evidence in one view.

## Why this is different

Most planners produce a path. Some planners produce a nominal cost. CERT-FLOW
produces a path together with a live, inspectable statement about the unknown
optimum and a policy for improving that statement.

| Capability | CERT-FLOW behavior |
|---|---|
| Stale map | Age widens each edge interval and reduces the supported confidence. |
| Unknown observation noise | Split-conformal residuals provide the statistical radius. |
| Drifting costs | A1/A2 drift terms, weighted calibration, and explicit shift penalties account for staleness. |
| Adaptive route choice | The selection ledger records candidates, context, and the selected path. |
| Paid sensing | Sensing is chosen by certified gap reduction, with objective-matched hybrid sensing when closing the gap is impossible. |
| Regime breaks | A formal e-value monitor can revoke the certificate, clear stale calibration, and reopen only after fresh evidence. |
| Fleet operation | Additive certificates compose agents; joint fleet calibration handles shared-load dependence without assuming agent independence. |
| Fast routing | Snapshot or Contraction Hierarchy preprocessing is usable only while a certificate-gated freshness check passes. |
| Failure behavior | Warm-up, sparse groups, unsupported conditional claims, and stale regimes become invalid or abstaining states—not silent promises. |

## The planning loop

1. Observe selected edges and form drift-adjusted residuals.
2. Weight calibration evidence by age using data-independent geometric weights.
3. Price every edge as a lower and upper interval, including uncertainty and drift.
4. Run optimistic and conservative incremental searches over the same graph.
5. Emit the route, `LB`, `UB`, confidence, validity state, and diagnostic data.
6. Select the next observation by expected certified gap reduction and sensing cost.
7. Execute or advance the route, absorbing free observations as they arrive.
8. Monitor the residual stream for invalidation, selection effects, and recovery.

The default certificate is distribution-free within its declared assumptions.
Optional layers are explicit about whether they are certificates, audits,
diagnostics, or weaker resource-allocation licenses.

## What is in the current system

### Core certificate and search

- Age-weighted, non-exchangeable split conformal quantiles.
- Drift-adjusted residuals and realized staleness correction.
- ACI adaptation and scale-free ACI updates.
- Optimistic lower search and conservative upper search.
- Flat-array Dijkstra/D* Lite kernels with a pure-Python fallback.
- Path-level confidence accounting, warm-up invalidity, annealing, and explicit
  finite-cost sentinels for unresolved upper bounds.
- LP-shift quantiles for smooth displacement plus abrupt-mass uncertainty.

### Tighter certified objects

- CIA-style group-sum path certificates with split alpha budgets.
- Block-sum upper bounds for long paths.
- PASC block-max calibration as an experimental path-level alternative.
- The `ShrinkLicense` anytime-valid shadow tier for resource allocation. It is
  never substituted for the distribution-free safety certificate.
- Trajectory-level conformal tubes and linear reachable tubes.

### Decision and evidence layers

- `SelectionLedger` and independent post-selection audit certificates.
- Finite-family group calibration with sparse-group invalidation rather than
  unsupported borrowing of marginal coverage.
- `EvidenceScoreModel` and pooled normalized conformal calibration using
  disjoint training and calibration splits.
- `DecisionRiskController` for action selection against supplied loss streams.
- Selection digests and diagnostics for reproducible audit trails.

### Sensing, fleet, and recovery

- Certificate-directed sensing, freshness, max-age, random, VOI, and hybrid
  policies.
- UCB gain-per-cost active sensing with realized-gain accounting.
- Additive fleet certificates and joint fleet calibration with deterministic
  shared-load penalties.
- WATCH conformal test martingale, conformal p-values/e-values, and
  Shiryaev–Roberts change detection.
- An optional fixed mixture of SR betting shapes for research comparison; the
  promoted default remains the scalar detector because it recovered faster in
  the matched full audit.
- Formal recovery: certificate revocation, stale-buffer clearing, fresh-sample
  re-entry, and a fresh testing epoch after recovery.

### Proof-gated acceleration

- Snapshot all-pairs oracle for certified static intervals.
- Certified Contraction Hierarchies for large road graphs.
- Exactness checks under bounded cost perturbations.
- Automatic invalidation when costs move outside the gate.

## Minimal usage

```bash
python -m pip install "certflow[fast]"
```

```python
from certflow import CertPlanner, PlannerConfig
from certflow.drift import grid_world

world = grid_world(
    6, 6,
    seed=0,
    kind="bounded",
    rho=0.02,
    noise_scale=0.05,
)
planner = CertPlanner(
    world,
    (0, 0),
    (5, 5),
    PlannerConfig(epsilon=5.0, alpha_prime=0.2),
)

for _ in range(150):
    certificate, sensed_edge = planner.round()

print({
    "path": certificate.path,
    "lb": certificate.lb,
    "ub": certificate.ub,
    "confidence": certificate.confidence,
    "valid": certificate.valid,
    "sensed": sensed_edge,
})
```

For a strong general-purpose configuration:

```python
from certflow.cert import recommended_config

planner = CertPlanner(
    world,
    (0, 0),
    (5, 5),
    recommended_config(epsilon=5.0),
)
```

Inspect the live state without changing the certificate stream:

```python
diagnostics = planner.diagnostics()
recovery = planner.recovery_diagnostics()
```

Enable formal regime recovery when stale calibration must be revoked on an
alarm:

```python
config = PlannerConfig(
    epsilon=5.0,
    alpha_prime=0.2,
    regime_recovery=True,
    recovery_formal=True,
    recovery_samples=5,
)
```

## Current measured evidence

The repository contains reproducible scripts and persisted full-run artifacts.
The release checker refuses quick artifacts as release evidence.

| Area | Current result |
|---|---|
| Full regression | 337 tests passed, 0 skipped (NY road graph, MovingAI maps/scenarios, METR-LA, PEMS-BAY all downloaded and loader-verified); one existing NumPy warning in the NaN/Inf serialization test. |
| Release audit | 83/83 strict gates passed. |
| Abrupt calibration shift | Formal SR post-shift edge coverage `0.954`, compared with ACI `0.937`; median detection `23` rounds, recovery `5`, valid re-entry `5`. |
| Formal recovery safety | Zero pre-shift false alarms in the full matched audit. |
| Fresh conditional audits | METR-LA and PEMS-BAY post-selection/group lower confidence bounds all clear the `0.80` release floor. |
| Fleet and sensing | Full budget/policy matrix completed with valid coverage lower bounds; hybrid sensing beats the budget-matched controls in the release audit. |
| Correlated drift | Full correlated/independent matched stress completed with no observed path-coverage drop. |
| Routing acceleration | Exactness checks pass for Contraction Hierarchies and bounded-change cost absorption. |
| Static raw latency | Specialized static methods remain faster. CERT-FLOW pays milliseconds for the certificate and wins on validity and bounded-change behavior, not on every raw-latency regime. |

The evidence is deliberately separated into certificate validity, decision
quality, conditional audits, recovery, and latency. A single composite score
would hide the tradeoffs.

## Visual gallery

### One round, one corridor

![Certified route corridor animation](assets/animations/certified_corridor.gif)

The planner warms up without making an unsupported claim, then maintains a
route-level corridor while the world drifts and sensing updates the evidence.

![Current-system recovery and sensing dashboard](assets/current_system_dashboard.png)

### Recovery and live validity

![Recovery frontier](assets/recovery_frontier.png)

![Live wiring and formal alarm figure](assets/live_wiring_2026.png)

![WATCH and Shiryaev-Roberts shift-detection animation](assets/animations/watch_alarm.gif)

The animation is an observational shift-detection demonstration; it does not
show the separate formal recovery/re-entry policy or change the certificate.

### Certificate versus an uncertified promise

![Synthetic certificate comparison](assets/animations/cert-break-grid.gif)

![MovingAI certificate comparison](assets/animations/cert-break-movingai.gif)

The comparison animations show why a narrow interval is not automatically a
good interval: the uncertified baseline can be smaller while failing to cover
the realized optimum.

### Sensing that improves decisions

![Sensing regret figure](assets/sensing_regret.png)

![Synthetic sensing animation](assets/animations/sensing-grid.gif)

![MovingAI sensing animation](assets/animations/sensing-movingai.gif)

The sensing policy changes when it can no longer close the certified gap within
budget. At that point hybrid sensing protects route quality without pretending
that the certificate closed.

### Width, dependence, and path aggregation

![Width methods](assets/width_methods.png)

![PASC versus Bonferroni](assets/animations/pasc_vs_bonferroni.png)

![Certified versus fast-planner crossover](assets/crossover_regret.png)

The tighter path-level objects recover part of the Bonferroni cost on suitable
streams. PASC is retained as an experimental alternative because its behavior
depends strongly on path length and support.

### Poster stills

| Certified grid | MovingAI comparison | Sensing grid | MovingAI sensing |
|---|---|---|---|
| ![grid 1](assets/gallery/cert-break-grid-poster-1.png) | ![MovingAI 1](assets/gallery/cert-break-movingai-poster-1.png) | ![sensing 1](assets/gallery/sensing-grid-poster-1.png) | ![MovingAI sensing 1](assets/gallery/sensing-movingai-poster-1.png) |
| ![grid 2](assets/gallery/cert-break-grid-poster-2.png) | ![MovingAI 2](assets/gallery/cert-break-movingai-poster-2.png) | ![sensing 2](assets/gallery/sensing-grid-poster-2.png) | ![MovingAI sensing 2](assets/gallery/sensing-movingai-poster-2.png) |
| ![grid 3](assets/gallery/cert-break-grid-poster-3.png) | ![MovingAI 3](assets/gallery/cert-break-movingai-poster-3.png) | ![sensing 3](assets/gallery/sensing-grid-poster-3.png) | ![MovingAI sensing 3](assets/gallery/sensing-movingai-poster-3.png) |
| ![grid 4](assets/gallery/cert-break-grid-poster-4.png) | ![MovingAI 4](assets/gallery/cert-break-movingai-poster-4.png) | ![sensing 4](assets/gallery/sensing-grid-poster-4.png) | ![MovingAI sensing 4](assets/gallery/sensing-movingai-poster-4.png) |

All GIFs are generated from real planner runs. MP4 output is optional when
ffmpeg is installed; the current render environment produced the GIF fallbacks
and the still images shown above.

## Reproduce the evidence

Clone and install the development dependencies:

```bash
git clone https://github.com/Archerkattri/CERT-FLOW
cd CERT-FLOW
python -m venv cert_env
source cert_env/bin/activate          # Windows: cert_env\Scripts\Activate.ps1
python -m pip install -e ".[dev,fast,realworld]"
python -m pytest -q
```

Run the strict release audit:

```bash
python scripts/check_release_gate.py --strict
```

Run the current recovery smoke benchmark:

```bash
python scripts/run_shift_recovery.py --quick
```

Run the full current audits when you want to regenerate the persisted evidence:

```bash
python scripts/run_validity_audit.py
python scripts/run_shift_recovery.py
python scripts/run_fleet_sensing.py
python scripts/run_realworld_audit.py
```

The `--quick` flag is for development feedback and smoke testing. It is not
release evidence. Real-data experiments require the datasets described by the
loaders under `src/certflow/realworld.py`, `src/certflow/movingai.py`, and
`src/certflow/roadnet.py`; absent datasets are skipped explicitly by tests.

Generate the visual gallery:

```bash
python scripts/viz_gen/current_system_dashboard.py
python scripts/viz_gen/certified_corridor.py
python scripts/viz_gen/watch_alarm.py
python scripts/viz_gen/live_wiring_fig.py
python scripts/viz_gen/sensing_regret.py
python scripts/viz_gen/width_methods.py
python scripts/viz_gen/pasc_vs_bonferroni.py
```

For the longer animations:

```bash
PYTHONPATH=src python scripts/viz_gen/cert-break-grid.py
PYTHONPATH=src python scripts/viz_gen/sensing-grid.py
PYTHONPATH=src python scripts/viz_gen/cert-break-movingai.py
PYTHONPATH=src python scripts/viz_gen/sensing-movingai.py
```

## Repository map

```text
src/certflow/
  types.py       World, EdgeBelief, Certificate contracts
  cert.py        certificate-directed planner and round loop
  conformal.py   quantiles, drift models, ACI, CIA/PASC, e-values, SR
  upgrades.py    selection, group, evidence, risk, tube, fleet, recovery
  sensing.py     gap-directed and baseline sensing policies
  fastgraph.py   flat-array Dijkstra/D* Lite engines
  snapshot.py    certificate-gated snapshot oracle
  ch.py          certified Contraction Hierarchies
  drift.py       synthetic drifting worlds
  realworld.py   traffic replay worlds
  movingai.py    MovingAI map/scenario loaders
  roadnet.py     DIMACS road graphs and routing utilities
  episodes.py    episode drivers and coverage aggregation
  harness.py     reproducible experiments and persistence
  team.py        additive fleet certificates

scripts/
  run_*.py       benchmark and audit runners
  check_release_gate.py strict persisted-evidence release checker
  viz_gen/       figures, animations, and the current dashboard renderer

assets/
  *.png          release figures
  animations/    GIF demonstrations
  gallery/       poster stills from the animation runs

docs/research/
  competitive-program.md  research evidence and comparisons
  certflow-v4-roadmap.md  research roadmap and dispositions
```

The repository describes one current CERT-FLOW system. Audit artifacts retain
their original schemas for reproducibility, while the public documentation and
commands use neutral current names.

## Scope and limitations

CERT-FLOW does not claim universal conditional coverage over arbitrary
covariates. Group certificates require a declared finite family, enough
support, and independent group calibration. Selection certificates require an
independent or post-selection audit stream. Learned evidence scores improve
pricing but do not turn an arbitrary feature model into a conditional theorem.
The exact supported statement, random objects, filtration, runtime trace and
open proof obligations are recorded in the
[mathematical review note](docs/research/certflow-proof-boundary.md). The
[independent day/map manifest](results/protocols/independent-day-map-v1.json)
is frozen with its final endpoints marked `LOCKED_UNRUN`.

The additive fleet certificate is the robust fleet composition. A tighter
joint congestion construction is retained only as a falsified comparison on
the real traffic audit. The block-max PASC option is experimental. Static
specialized routing remains faster when no certificate is required. These are
design boundaries, not hidden exceptions.

## Citation

If you use CERT-FLOW in research, cite the software and the accompanying
preprint:

```bibtex
@software{attri2026certflow,
  author = {Attri, Krishi},
  title  = {{CERT-FLOW}: Certified Route Planning under Drifting Costs},
  year   = {2026},
  doi    = {10.5281/zenodo.20631475},
  url    = {https://github.com/Archerkattri/CERT-FLOW}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable record and the
[engrXiv preprint](https://doi.org/10.31224/7306) for the formal paper.

## License

MIT. See [LICENSE](LICENSE).

## Current release status

The current checkout includes the repaired cache/configuration, malformed-input,
provenance, runner-safety and certification paths, plus the witness/risk-ledger
research seam. The strict promotion gate and full regression suite pass. The
mathematical review narrows the supported theorem and hardens post-selection
audit provenance. Independent proof review and locked deployment-day/map
execution remain research gates; no universal or always-valid conditional-
coverage theorem is claimed.
