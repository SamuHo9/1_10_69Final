# SPHARM volume exploration

Analysis date: 2026-10-03. Groups follow the filename labels Healthy and TLE.

Unit assumption: original NIfTI masks omit spatial units (code 0). Values below assume the coordinate units are millimetres, following the ICP pipeline convention. Percent differences do not require this assumption.

## SPHARM surface volume (mm³, assuming mm coordinates)

| Side | Group | Inputs | Valid surfaces | Mean ± SD | Median | Q1–Q3 | Min–max |
|---|---|---:|---:|---:|---:|---:|---:|
| left | Healthy | 240 | 239 | 4699.28 ± 480.55 | 4679.60 | 4315.41–5014.87 | 3552.17–5924.61 |
| left | TLE | 141 | 134 | 3907.21 ± 672.84 | 3834.65 | 3424.15–4383.40 | 2122.33–5867.63 |
| right | Healthy | 276 | 275 | 4817.11 ± 529.65 | 4807.28 | 4471.28–5177.53 | 3163.69–6607.20 |
| right | TLE | 105 | 102 | 3845.99 ± 641.82 | 3782.78 | 3446.47–4241.95 | 2500.87–5762.20 |

## Group comparisons

| Side | TLE − Healthy (mm³) | Change vs Healthy | 95% CI of difference | Welch p | Holm p (2 sides) | Hedges g |
|---|---:|---:|---:|---:|---:|---:|
| left | -792.07 | -16.86% | -922.01 to -662.13 | 1.148e-25 | 1.148e-25 | -1.419 |
| right | -971.12 | -20.16% | -1111.62 to -830.63 | 2.314e-28 | 4.627e-28 | -1.724 |

## Cross-check against original binary masks

| Side | Group | Original mask mean ± SD (mm³) | Mean individual SPHARM change |
|---|---|---:|---:|
| left | Healthy | 4349.67 ± 461.46 | 8.08% |
| left | TLE | 3584.13 ± 641.63 | 9.14% |
| right | Healthy | 4472.14 ± 504.91 | 7.75% |
| right | TLE | 3532.33 ± 617.36 | 9.03% |

## Method and limitations

- One native `_SPHARM.vtk` per successful subject/side; grid, para, ellalign, medial-axis and template surfaces are not additional subjects.
- All included surfaces are finite, closed, and have no boundary/nonmanifold edges. VTK mass-properties volume agrees with an independent signed-tetrahedron sum.
- Physical volume = normalized surface volume / scale³. The ICP metadata scale matches every saved transform; template hashes and every SPHARM input hash were checked.
- Original-mask volume = count of label-1 voxels × voxel volume; affine voxel volume agrees with pixdim. Original headers have unspecified spatial units (code 0), so reported mm³ assumes millimetres as used by the ICP pipeline.
- Surface processing includes resampling, label cleanup, and harmonic approximation; SPHARM volume is a processed-surface estimate, distinct from original segmentation volume.
- All valid samples and IQR outliers are retained. Summary and tests use the available sample for each side separately.
- Welch tests and confidence intervals compare independent filename groups within each side. Holm correction is across the two sides; original-mask comparisons are a separate sensitivity analysis.
- No adjustment for age, sex, intracranial volume, scan site, or dataset. This is a descriptive group comparison, not a diagnostic threshold or causal inference.
- Exact IDs shared across valid sides: 369; matching group labels: 134; different labels: 235. Side-specific labels must not be treated as a person-level diagnosis without metadata.

## Failed processing inputs

left: 8 failed of 381 inputs.
- left_Healthy_sub-451_hippocampus_lh_aligned
- left_TLE_sub-109_hippocampus_lh_aligned
- left_TLE_sub-202_hippocampus_lh_aligned
- left_TLE_sub-37_hippocampus_lh_aligned
- left_TLE_sub-388_hippocampus_lh_aligned
- left_TLE_sub-441_hippocampus_lh_aligned
- left_TLE_sub-460_hippocampus_lh_aligned
- left_TLE_sub-92_hippocampus_lh_aligned

right: 4 failed of 381 inputs.
- right_Healthy_sub-37149_hippocampus_rh_aligned
- right_TLE_sub-203_hippocampus_rh_aligned
- right_TLE_sub-405_hippocampus_rh_aligned
- right_TLE_sub-419_hippocampus_rh_aligned
