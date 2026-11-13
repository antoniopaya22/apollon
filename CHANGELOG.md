# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## Unreleased

### Added

- Documentation site: animated landing page (GSAP), interactive Thompson sampler, figures
  of the preprocessing pipeline and of the attack settings.

## 2.0.0 - 2026-11-13

### Added

- Installable package `apollon` (`pip install apollon-ids`) with a `src/` layout and
  type information.
- `ApollonMAB` as a scikit-learn estimator: per-cluster Thompson sampling over a pool of
  classifiers, online `update` from feedback, `prior_strength` to control exploration,
  `selection_frequencies` and `posterior_means` for inspection, `save_model` and
  `load_model`.
- `apollon.pool`: registry of arms (RF, DT, NB, LR, MLP, KNN, SVC) and the pool of the
  paper.
- `apollon.data`: loaders and preprocessing for CIC-IDS2017, CSE-CIC-IDS2018 and
  CIC-DDoS2019, and a portable `.npz` dataset format.
- `apollon.attacks`: ZOO and HopSkipJump through ART, in direct and transfer settings,
  with functional-feature constraints.
- `apollon.metrics`: accuracy, detection rate, F1, false positive rate and ROC AUC.
- Command line interface: `apollon prepare | train | evaluate | attack | demo`.
- Documentation site, continuous integration, tests, contribution guide and issue
  templates.

### Changed

- The original scripts and notebooks of the paper moved to `research/` and import the
  package.

## 1.0.0 - 2023-06-10

### Added

- Code of the paper published in *Computers & Security* 136 (2024): IDS models,
  preprocessing of the CIC datasets, Multi-Armed Bandit selection and adversarial attack
  notebooks.
