# SPHARM: inferred person-level volume comparison

Date: 2026-10-03. Input IDs: 381; inferred controls: 135; inferred TLE: 246 (141 left-label TLE and 105 right-label TLE).

**Grouping assumption:** Control means Healthy in both filename labels. TLE means TLE in at least one side. The same ID is assumed to refer to the same person across sides. This is inferred from file naming; clinical metadata has not confirmed it.

Healthy on one side alone does not establish that the person is a healthy control: 235 of 369 IDs with valid bilateral surfaces have different labels between sides. See [the separate filename-label report](report.md).

**Units:** mm³ assumes source coordinates are millimetres, following the ICP convention. All source NIfTI headers omit spatial units (code 0). Percent differences remain valid in the shared coordinate unit.

## Control versus all inferred TLE participants

| Side | Person group | Valid surfaces | SPHARM mean ± SD (mm³) | Median | Q1–Q3 | Original mask mean ± SD (mm³) |
|---|---|---:|---:|---:|---:|---:|
| left | Control | 135 | 4654.09 ± 450.28 | 4646.02 | 4272.05–4968.60 | 4301.15 ± 432.38 |
| left | TLE | 238 | 4278.96 ± 739.91 | 4356.65 | 3767.93–4817.37 | 3946.18 ± 711.13 |
| right | Control | 134 | 4718.92 ± 469.07 | 4702.77 | 4327.70–5064.69 | 4371.15 ± 446.78 |
| right | TLE | 243 | 4463.63 ± 797.06 | 4600.87 | 3861.49–5035.79 | 4133.34 ± 767.71 |

| Side | TLE − Control (mm³) | Change vs Control | 95% CI | Welch p | Holm p (2 sides) | Hedges g |
|---|---:|---:|---:|---:|---:|---:|
| left | -375.13 | -8.06% | -496.38 to -253.88 | 2.938e-09 | 5.876e-09 | -0.576 |
| right | -255.29 | -5.41% | -383.57 to -127.00 | 0.0001083 | 0.0001083 | -0.365 |

## Inferred laterality subgroups

All-patient means combine the side labelled TLE with the opposite side. The subgroups below keep the inferred laterality visible.

| Surface side | Inferred cohort | Valid surfaces | SPHARM mean ± SD (mm³) |
|---|---|---:|---:|
| left | Control | 135 | 4654.09 ± 450.28 |
| left | Left_TLE | 134 | 3907.21 ± 672.84 |
| left | Right_TLE | 104 | 4757.94 ± 513.50 |
| right | Control | 134 | 4718.92 ± 469.07 |
| right | Left_TLE | 141 | 4910.43 ± 567.45 |
| right | Right_TLE | 102 | 3845.99 ± 641.82 |

## Verification and scope

- Included 373 left and 377 right surfaces; 8 left and 4 right processing failures are excluded. Each available hemisphere is analysed without requiring a complete bilateral pair.
- Every included native SPHARM surface was closed, finite and without boundary/nonmanifold edges. Volume was checked with two independent calculations.
- The ICP physical-to-normalized scale was checked against every transformation matrix, and SPHARM input provenance hashes matched the current aligned inputs.
- Values are processed SPHARM surface estimates. Original mask volumes are shown separately as a sensitivity check.
- Every valid surface is retained, including IQR outliers. Statistical comparisons use Welch tests and Holm correction over two sides.
- These are unadjusted comparisons: age, sex, intracranial volume, acquisition site and dataset effects are not controlled. No diagnostic threshold or causal inference is established.
- Detailed statistics, including original-mask comparisons and Mann–Whitney tests, are in person_group_results.json. Per-subject values and both grouping definitions are in subject_volumes.json.
