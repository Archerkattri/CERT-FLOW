# Changelog

All notable changes to CERT-FLOW are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Multi-agent certificate + a 2026 conformal upgrade layer. All new behavior is
opt-in behind config flags / new classes; the default single-agent certificate
and its guarantees are unchanged.

### Added
- **Additive multi-agent certificate** (`certflow.team`): `additive_certificate`
  + `TeamCertificate` (`sum LB <= sum OPT <= sum UB`, union-bound confidence) —
  the one TEAM-CERT survivor, ported over a shared conformal store.
- **LP-shift staleness** (`ConformalScorer(shift_model="lp", eps_lp, rho_lp)`,
  arXiv 2502.14105): worst-case quantile `Quant(1-alpha+rho)+eps`. TV default.
- **CIA path-sum calibration** (`CIACalibrator`, `CertPlanner.cia_path_certificate`,
  arXiv 2408.10939): group-sum path calibration with symmetric-calibration overlap
  handling and the age-weighted drift retrofit. Bonferroni default.
- **SF-OGD ACI** (`ACITracker(mode="sf-ogd")`, arXiv 2302.07869): scale-free,
  anytime step size. Fixed-gamma default.
- **PASC joint per-edge calibration** (`PASCCalibrator`, arXiv 2605.18812): one
  `max`-score quantile prices all edges jointly at `>= 1-alpha`, replacing the
  `alpha/L` per-edge Bonferroni correction.
- **Testability layer** — making the pinned-at-1.0 coverage observable:
  `conformal_p_value` (WATCH Eq. 9), `ConformalTestMartingale` (WATCH, arXiv
  2505.04608; Ville alarm + tightness stress test), `conformal_e_value` /
  `score_ratio_e_value` / `merge_e_values` (arXiv 2503.13050).
- **Drift diagnostics** (from DASC, arXiv 2606.15953, as observables only —
  DASC's coverage bound is not distribution-free): `residual_drift_score`
  (1-D Wasserstein `D_t`), `effective_sample_size` (Kish `n_eff`).
- `docs/results/multiagent.md`, `docs/related-work-2026.md` (positioning vs
  arXiv 2601.03629 + the adopted machinery).

## [1.0.2] - 2026-06-10

Packaging and serialization fixes for the freshly published library.

### Fixed
- `pytest` now discovers the `src/`-layout package on a fresh checkout: the
  `[tool.pytest.ini_options]` `pythonpath` was `["."]` (repo root, no package
  there), so `python -m pytest` failed with `ModuleNotFoundError: certflow`
  unless `PYTHONPATH=src` was set or the package was installed. Set to
  `["src"]`.
- `EpisodeResult.oracle_cost` is now serialized by `save_results` and restored
  by `load_results`. It was dropped on save, so reloaded Tier-2 results came
  back with `oracle_cost = nan` and any regret analysis silently reported NaN.
  Legacy result files without the field still load (oracle_cost stays nan).

### Added
- `realworld` optional extra (`pip install 'certflow[realworld]'`) declaring
  the `pandas` and `tables` dependencies the METR-LA / PEMS-BAY traffic
  adapter needs. The core install stays numpy/scipy only; `_load_traffic` now
  raises a clear `ImportError` pointing at the extra when pandas is absent.

## [1.0.1] - 2026-06-10

First PyPI release (`pip install certflow`).

### Added
- Top-level package API: `from certflow import CertPlanner, PlannerConfig,
  Certificate, ConformalScorer, ACITracker, EdgeBelief, World`, plus
  `certflow.__version__` (previously `certflow/__init__.py` was empty and
  everything had to be imported from submodules; the old submodule imports
  still work).
- `CITATION.cff` (validated, concept DOI 10.5281/zenodo.20631475) and this
  changelog.
- Full PyPI packaging metadata: readme, keywords, classifiers, project URLs.

### Changed
- README: pip-based 30-second quickstart, static DOI badge pointing at the
  concept DOI, link to the limitations ledger, Python badge corrected to
  3.10+ (matching `requires-python`).
- Package version aligned with the release tag (pyproject said 0.1.0 while
  the repository was at v1.0.0).

### Fixed
- Lint sweep over `src/`: removed unused imports and dead local assignments
  (no behavior change; the full test suite passes bit-identically).

## [1.0.0] - 2026-06-10

First public release, accompanying the preprint *CERT: Certified Route
Planning under Drifting Costs (Extended Version)*.

- Conformal route certificates (LB <= OPT <= UB) under drifting edge costs:
  age-weighted non-exchangeable quantiles, staleness correction, honest
  annealing.
- Certificate-directed sensing (route-critical, churn-aware) and dual
  incremental search on a flat-array engine (numba kernels with pure-Python
  fallback).
- Certificate-gated preprocessing: all-pairs snapshot oracle and certified
  Contraction Hierarchies (ns-to-microsecond queries that expire under
  drift).
- 200+ tests; 16 reproduction pipelines covering 17 synthetic regimes,
  METR-LA / PEMS-BAY traffic replay, MovingAI maps, and DIMACS road
  networks.
- Theory T1-T7 documented in `docs/` (coverage, certifiability threshold,
  sum-aware certificate, impossibility of a tighter lower bound,
  decision-uniform validity, churn floor).

[1.0.2]: https://github.com/Archerkattri/CERT-FLOW/releases/tag/v1.0.2
[1.0.1]: https://github.com/Archerkattri/CERT-FLOW/releases/tag/v1.0.1
[1.0.0]: https://github.com/Archerkattri/CERT-FLOW/releases/tag/v1.0.0
