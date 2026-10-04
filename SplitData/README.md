# SPHARM train/test split

Seed: 42. Split unit: person ID, shared across both hemispheres. Stratified by the inferred Control, Left_TLE and Right_TLE cohorts.
Train: 304 persons (79.79%). Test: 77 persons (20.21%). Rounding is necessary for 381 persons.

| Split | Persons | Control | Left TLE | Right TLE | Left surfaces | Right surfaces |
|---|---:|---:|---:|---:|---:|---:|
| train | 304 | 108 | 112 | 84 | 300 | 300 |
| test | 77 | 27 | 29 | 21 | 73 | 77 |

## Folder structure

```text
SplitData/
  train/
    SPHARM_LEFT/spharm_results/
    SPHARM_RIGHT/spharm_results/
    subjects.json
    subject_ids.txt
  test/
    SPHARM_LEFT/spharm_results/
    SPHARM_RIGHT/spharm_results/
    subjects.json
    subject_ids.txt
  split_manifest.json
  file_manifest.json
  split_summary.json
```

All subject-specific files from successful SPHARM runs are copied, retaining their original names. Failed-run residual files are excluded and listed in split_summary.json. Missing hemispheres are recorded in subjects.json.

## Labels and validation

Control/TLE person groups and TLE laterality are inferred from paired filename labels. A Healthy hemisphere may belong to a person whose opposite hemisphere is labelled TLE. Both the original hemisphere labels and inferred person labels are recorded in the manifest.

Copied and verified 11250 files by byte comparison; SHA-256 checksums are recorded. All 750 successful subject-side surfaces are included exactly once. Train/test ID overlap is zero.

## Existing preprocessing

This split uses existing ICP/SPHARM outputs. The existing ICP references were fit using all 381 subjects, including IDs now assigned to test. This split prevents subject overlap but does not undo reference-fitting leakage. For a fully held-out evaluation, fit data-dependent preprocessing on train only, then transform test using the frozen training reference.

Creation script: Analysis/spharm_volume/split_spharm_data.py. The script refuses to overwrite an existing SplitData folder.

## Dataset details CSV

`dataset_details.csv` contains one record per person and hemisphere: 762 records, including 750 passed SPHARM runs and 12 failed runs. It uses UTF-8 with a BOM for Excel compatibility. IDs use the `sub-` prefix to preserve leading zeros.

- `split` is the person's assigned `train` or `test` set, including when that hemisphere failed processing.
- `side` is `left` or `right`; `dataset_partition` combines these as `train_left`, `train_right`, `test_left`, or `test_right`.
- `spharm_status` is `passed` or `failed`, taken from the original SPHARM status JSON.
- `included_in_split` is 1 when the processed surface is present in SplitData, and 0 when it is absent. Failed runs have an assigned split but are not included as usable samples.
- `hemisphere_label_original` preserves the original filename label. `person_group_inferred` and `cohort_inferred` retain the grouping inferred from both hemispheres.
- Coefficient/grid availability flags are 1 or 0. `split_mesh_path` is relative to SplitData and blank for failed runs. `status_source` is relative to the project root and identifies the source status record.
