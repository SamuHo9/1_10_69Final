# Prepared datasets

Input: current extracted coefficient and XYZ train/test CSVs in SplitData.
Labels: original BinaryClass (hemisphere target). Seed: 42. PLS components: 8. Augmentation: train only, same-class, children-per-pair parameter 8.
The final augmentation pair may contain fewer than 8 children to achieve exact class balance.

| Method | Side | Status | Train rows | Synthetic rows | Test rows | Features |
|---|---|---|---:|---:|---:|---:|
| coef_plsda | left | completed | 384 | 84 | 73 | 507 |
| coef_plsda | right | completed | 438 | 138 | 77 | 507 |
| coef_raw | left | completed | 300 | 0 | 73 | 507 |
| coef_raw | right | completed | 300 | 0 | 77 | 507 |
| coef_raw_balanced_jitter | left | completed | 384 | 84 | 73 | 507 |
| coef_raw_balanced_jitter | right | completed | 438 | 138 | 77 | 507 |
| plsda_direct | left | completed | 300 | 0 | 73 | 507 |
| plsda_direct | right | completed | 300 | 0 | 77 | 507 |
| plsda_latent_features | left | completed | 300 | 0 | 73 | 8 |
| plsda_latent_features | right | completed | 300 | 0 | 77 | 8 |
| pointnet_plsda | left | completed | 384 | 84 | 73 | 3006 |
| pointnet_plsda | right | completed | 438 | 138 | 77 | 3006 |
| pointnet_raw | left | completed | 300 | 0 | 73 | 3006 |
| pointnet_raw | right | completed | 300 | 0 | 77 | 3006 |
| pointnet_raw_balanced_jitter | left | completed | 384 | 84 | 73 | 3006 |
| pointnet_raw_balanced_jitter | right | completed | 438 | 138 | 77 | 3006 |

Each method contains left/ and right/ output folders. Raw and augmented methods export train_prepared.csv and test_prepared.csv. Latent-feature outputs use train_plsda_latent_features.csv and test_plsda_latent_features.csv. Direct PLS-DA exports models, metrics and predictions.

prepare_plsda_direct.py aliases run_plsda_direct.py, so this classifier is executed once per side. Raw baselines and optional raw balanced-jitter augmentation are stored separately.

Validation checked feature schemas, finite values, row counts, unchanged source CSV hashes, test preprocessing, same-class train parents and PairID provenance. run_all_summary.json records exact commands and validation details; _logs/ contains execution logs.

Known input limitations: labels are hemisphere labels; contralateral hemispheres of inferred TLE persons are labelled 0. Existing ICP references were fit on all 381 inputs including test; these outputs are not an independently held-out preprocessing evaluation.
