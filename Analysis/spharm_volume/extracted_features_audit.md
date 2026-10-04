# Extracted SPHARM features audit

Date: 2026-10-03. The feature CSVs were read and compared without modifying them.

Technical extraction errors: 0. Files checked: 8.

| Split | Side | Rows (each feature type) | Coef features | XYZ features | BinaryClass 0 | BinaryClass 1 | Person-label disagreements |
|---|---|---:|---:|---:|---:|---:|---:|
| train | left | 300 | 507 | 3006 | 192 | 108 | 84 |
| train | right | 300 | 507 | 3006 | 219 | 81 | 112 |
| test | left | 73 | 507 | 3006 | 47 | 26 | 20 |
| test | right | 77 | 507 | 3006 | 56 | 21 | 29 |

## Checks performed

- Exact subject membership and counts against the original split manifest; no duplicate subjects or train/test person overlap.
- All feature names and column orders match the Prepare input schema. All values are numeric and finite.
- Every coefficient and coordinate value was compared with the matching ellalign source file. Allowed error: 5.1e-9, matching the extractor’s eight-decimal output.
- CSV, source and processing-sidecar SHA-256 checksums match the extraction manifests.
- Coefficient and XYZ labels agree. Mesh edges exactly match every source surface topology in their batch.
- Train and test use the same feature variant and processing contract for each side.
- The actual Prepare `load_pair` function accepted all four train/test pairs (coefficient and XYZ, left and right), tested with PythonSlicer.

## Target label definition

245 unique subject-side records have BinaryClass=0 while their inferred person group is TLE. This is consistent with the filename-based hemisphere target. It is a label-definition problem if the intended task is person-level Healthy Control versus TLE classification.
Both coefficient and XYZ files contain these same records. There are 245 affected hemisphere records, not 490 independent observations. The inferred person group is based on paired filename labels, not independently confirmed clinical metadata.

## Reference provenance

The source SPHARM contracts report icp_mode=unknown and have no ICP/SPHARM reference hashes; every extracted manifest correctly has fixed_reference_validated=false.
The ICP reference metadata shows fit_reference on all 381 inputs for both sides, including the subjects now assigned to test. For a fully held-out evaluation, rebuild data-dependent preprocessing on training inputs and transform test using the frozen training reference.

## Errors

None found in the technical extraction checks.
