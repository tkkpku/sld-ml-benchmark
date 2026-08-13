# DATA CARD · SLD-ML Benchmark v1

## 1. Dataset identity

- Name: SLD-ML Benchmark v1
- Version: 1.0 (2026-08-13)
- License: CC-BY-4.0
- Distribution: GitHub Releases v1.0.0
  (https://github.com/tkkpku/sld-ml-benchmark)
- Language/domain: mechanical engineering, milling chatter, stability lobe
  diagrams (SLD), machine-learning surrogates

## 2. Dataset summary

90 parameter sets of the standard two-DOF regenerative-chatter milling
model, each with coarse (16x10, 32x20) and fine (128x80) stability maps.
Labels are exact Floquet spectral radii produced by a first-order
semi-discretization solver that was independently audited (Gauss
quadrature, exact constant-coefficient characteristic equation, two
time-domain integrators). Validation/test splits carry a second fine label
at m=160.

## 3. Intended use

- Training and benchmarking machine-learning surrogates that map coarse
  stability fields to fine stability lobe diagrams;
- Calibrating learned vs analytic (ZOA) vs interpolation baselines under a
  common protocol;
- Reproducibility studies of ML-SLD methods.

Not intended for: real cutting-process control without experimental
validation; extrapolation beyond the parameter box below.

## 4. Provenance and generation

- Model: 2-DOF symmetric milling DDE (Insperger & Stepan 2004 parameters:
  N=2, fn=922 Hz baseline, zeta=0.011, m_t=0.03993 kg, Kt=6e8, Kn=2e8
  N/m^2, down milling).
- Solver: first-order (linear-in-delay) SDM, block-exponential interval
  flow, Floquet spectral radius.
- Audit: three historical solver bugs (interval-weight swap, matrix-order
  reversal, state-space sign error) were found by independent audit and
  fixed before any label was produced; spot-check vs the audited CPU solver
  over 45 points x 9 cases gives max |rho_label - rho_cpu| = 2.12e-6.
- Generation pipeline: GPU-batched assembly (PyTorch, CUDA 12.8) + exact
  CPU LAPACK eigenvalues; one-command scripts included in `code/`.
- AI assistance: dataset generation, code and documentation were produced
  with heavy AI assistance and then subjected to six independent
  adversarial reviews; all headline numbers were independently recomputed.

## 5. Parameter space

- Radial immersion a/D: 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50,
  0.60, 0.75, 0.90, 1.00
- Damping ratio zeta: 0.005, 0.008, 0.011, 0.015, 0.020, 0.025, 0.030
- Natural frequency fn: 700, 800, 850, 922, 1000, 1100 Hz
- Domain: 4000-16000 rpm x 0.05-1.5 mm
- 90 cases are a deterministic subset of the 12x7x6 grid (anchors +
  extrapolation coverage + in-domain diversity; not the full 504-product);
  splits 70/10/10, test cases are all extrapolation points.

## 6. Files and formats

- `data/rho/case_<idx>_<grid>_m<m>.npz`: rho (flat float64, row-major
  depth x speed), n_rpms, a_p_mm, case, m.
- `meta.json`: full case list, splits, grids, license.
- `results/`: precomputed baselines, U-Net reference metrics, label
  uncertainty, safety analysis, stratified (difficulty-layer) evaluation,
  ZOA region scan, per-method timings, label spot-check, failure-zone
  experiments, and the safety-budget curve.
- `results/models/`: reference U-Net weights (58,481 params; release and
  ZOA-conditioned variants).

## 7. Quality and validation

- Label discretization uncertainty (m80 vs m160, val+test): mean 0.44%,
  max 2.65% mask disagreement (narrow a/D=0.1 lobe).
- Convergence: m80 vs m160 mask disagreement 0.11%-2.65% over three test
  grids; rho 99th-percentile difference 0.05-0.08.
- Time-domain verification: 13/15 direct agreement with two independent
  integrators, 2 documented boundary cases.
- Stability labels are column-monotone in depth in 99.67% of columns.
- Reproducibility: every result JSON is regenerable from `code/`; the
  solver audit is one command (`run_all_audits.py` in the source repo,
  included here as `code/` scripts).

## 8. Known limitations

- Linear time-invariant two-DOF modal model only (no process damping,
  asymmetric modes, helix, thin-wall dynamics, parameter-dependent
  direction matrices).
- No experimental cutting data in v1.
- m=80 labels carry discretization uncertainty quantified above.
- The parameter box is moderate (90 cases); generalization beyond it is
  unvalidated.
- The analytic ZOA baseline is the strongest method inside the box
  (F1 0.965, false-stable 0.007); learned surrogates in this release do
  not beat it, which is a calibration result, not a defect.

## 9. Maintenance

This is a v1 release. Errata and v2 extensions (larger parameter space,
experimental data, process damping, uncertainty-aware evaluation) will be
released under the same repository as new GitHub Releases.
