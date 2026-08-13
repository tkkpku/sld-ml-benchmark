# SLD-ML Benchmark v1: Dataset Descriptor

Version 1.0 (2026-08-13) · License: CC-BY-4.0 · DOI: TBD

## 1. Motivation and value

Machine-learning surrogates for milling stability lobe diagrams (SLDs) are
widely proposed, but each paper trains and evaluates on its own private
dataset, with no common baselines and no public coarse-to-fine label pairs.
This release provides:

1. a public paired coarse/fine SLD dataset (to the authors' knowledge, as
   of the 2026 literature review; retrieval method documented in the source
   repository's `references/first-dataset-search-log.md`);
2. audited labels: the SDM solver's three historical bugs were found and
   fixed by independent audit; every label is produced by the fixed solver
   and spot-checked against a CPU reference (max |rho_label - rho_cpu| =
   2.12e-6 over 45 points x 9 cases);
3. label uncertainty: m80 vs m160 mask disagreement mean 0.44%, max 2.65%;
4. honest baselines: the analytic ZOA boundary still beats learned
   surrogates in this regime (F1 0.965 vs 0.905), a useful calibration for
   the community;
5. reproducibility: one-command generation pipeline (GPU assembly + exact
   CPU eigenvalues) and audited evaluation protocol.

## 2. Data description

- 90 cases drawn deterministically from a structured 12 x 7 x 6 parameter
  grid (anchors, systematic extrapolation coverage, in-domain diversity;
  not the full 504-combination Cartesian product; no random draws);
- domain: 4000-16000 rpm x 0.05-1.5 mm;
- grids: fine 128x80 (m=80), fine160 128x80 (m=160, val+test only), c32
  32x20 (m=40), c16 16x10 (m=20);
- stored fields: spectral radius rho (float64) per grid point; binary
  masks derived by rho < 1;
- splits: 70/10/10; test cases are all extrapolation points (low
  immersion, low damping, low natural frequency);
- format: NPZ per case; metadata JSON; license CC-BY-4.0.

## 3. Methods (label generation)

The 2-DOF milling DDE

    M q'' + C q' + K q = a_p Kt A(t) [q(t) - q(t - tau)]

is discretized with the first-order (linear-in-delay) SDM
(Insperger & Stepan 2004), using the block-exponential interval flow and
the Floquet spectral radius. The solver was audited:

- interval flow vs entrywise Gauss quadrature: err 1.4e-14 to 2.8e-14;
- matrix product order vs a fresh 1000-step RK4: position err
  1.3e-9 to 5.2e-9, full-state err 1.3e-6 to 4.8e-5;
- state-space signs: exact constant-A0 characteristic equation agreement
  (ZOA rel. err 0);
- time-domain verification: 13/15 direct agreement + 2 documented
  exceptions (an m=80 boundary-resolution flip and an inconclusive
  method-of-steps ratio near the exact boundary).

The GPU assembly pipeline is numerically identical to the audited CPU
solver (spot-check max |rho_label - rho_cpu| = 2.12e-6, 45 points x 9
cases spanning train/validation/test, `results/label_spotcheck.json`);
exact eigenvalues are computed with LAPACK.

## 4. Evaluation protocol and baseline results

Metrics: pixel F1, precision/recall, false-stable fraction, mean boundary
distance, wall-clock time; difficulty layers (trivial >= 0.99, medium
0.5-0.99, hard < 0.5 stable fraction); dual headline (all 10 cases vs the
7 non-trivial cases). Results on the 10 test cases (mean):

| method | all-10 F1 | non-trivial-7 F1 | all-10 false-stable |
| --- | --- | --- | --- |
| ZOA | **0.965** | **0.952** | **0.007** |
| bilinear32 | 0.931 | 0.902 | 0.045 |
| bilinear16 | 0.870 | 0.814 | 0.101 |
| nearest16 | 0.854 | 0.791 | 0.152 |
| coarse-ZOA + bilinear | 0.855 | 0.793 | 0.146 |
| U-Net (3 seeds) | 0.905 +/- 0.006 | 0.865 | 0.107 |
| ZOA-conditioned U-Net + monotone projection | 0.932 +/- 0.001 | 0.905 | 0.064 |

Per-layer results, safety at benchmark scale (t* = 0.65 for the release
U-Net), the ZOA failure-region scan over all 90 cases, and the
safety-budget operating-point curve for the conditioned surrogate are in
`results/`. The analytic baseline remains the strongest and only
high-safety operating point inside the box.

## 5. Usage notes

- downstream methods should report per-case F1 and false-stable fraction;
- checkpoints must be selected on validation only; test used once;
- report the operating point (threshold or conservative shift) and,
  ideally, a safety-matched comparison against the ZOA baseline at the
  same false-stable budget;
- report the dual headline (all 10 cases and the 7 non-trivial cases) and
  per-layer means;
- for boundary-focused evaluation, use the m=160 label where available;
- scoring any new method through the standard harness:

    python code/benchmark_eval_cli.py --pred <pred.npz> --case <idx>

## 6. Limitations

Linear symmetric 2-DOF model; no experiments; m=80 label discretization
uncertainty up to 2.65%; extrapolation beyond the studied parameter box is
not validated; the parameter channel interference effect and the safety
shortfalls of learned surrogates are documented in `results/`.

## 7. Availability

Zenodo DOI: TBD. GitHub: TBD. License: CC-BY-4.0.

## 8. References

1. Takamoto, M., et al. (2022). PDEBench. NeurIPS 2022 D&B. arXiv:2210.07182
2. Kim, E., et al. (2026). Scientific Data, 13(1). DOI: 10.1038/s41597-026-07255-7
3. Ströbel, R., et al. (2025). Data in Brief.
4. Li, G., et al. (2024). Data in Brief, 55, 110703. DOI: 10.1016/j.dib.2024.110703
5. IDEKO (2019). Zenodo. DOI: 10.5281/zenodo.3531471
6. Qin, Z., et al. (2025). IJAMT, 136(7-8), 2945-2985. DOI: 10.1007/s00170-024-14971-0
7. Insperger, T., Stepan, G. (2004). IJNME, 61(1), 117-141. DOI: 10.1002/nme.1061
8. Rezaei, S., et al. (2025). JIM, 36(2), 1201-1235. DOI: 10.1007/s10845-023-02291-1
