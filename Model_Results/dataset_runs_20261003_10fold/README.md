# Dataset model runs

Scope: all. Seed: 42. Neural epochs: 80. Coefficient model PLS components: 8.
Each prepared dataset and hemisphere is trained independently. Existing repository architecture classes are loaded without legacy hardcoded dataset paths.
Coefficient models use train-fitted StandardScaler and binary-target PLS. Latent data are rebuilt using one-hot PLS from the corresponding original coefficients. PointNet uses the supplied 1002-point XYZ clouds.
Exactly 10 stratified folds of original Train people for every model. Each original row has one OOF prediction; synthetic rows are not evaluated as OOF. No silent fold reduction.
Augmentation, scaling and supervised PLS are regenerated/fitted inside each training fold. Latent inputs are rebuilt from original coefficients inside each fold.
Fixed settings (PLS components and epochs), no outer OOF/Test-based model or epoch selection. After CV, refit on full Train and predict the held-out Test once.
Prepared synthetic CSV rows and pre-fitted latent coordinates are not reused in outer folds. The dataset preparation method is rebuilt from its original training people and data_manifest.json.
Input limitations: current labels are hemisphere labels; existing ICP reference included test. These results are an exploratory comparison on the supplied datasets.

| Dataset | Side | Model | Status | Test accuracy | Balanced accuracy | ROC-AUC | OOF accuracy | OOF ROC-AUC |
|---|---|---|---|---:|---:|---:|---:|---:|
| coef_raw | left | SVM | completed | 0.7808 | 0.7696 | 0.8412 | 0.7500 | 0.7608 |
| coef_raw | left | MLP | completed | 0.7397 | 0.7377 | 0.8511 | 0.7300 | 0.7420 |
| coef_raw | left | ResNet | completed | 0.7671 | 0.7504 | 0.8372 | 0.7433 | 0.7352 |
| coef_raw | left | ResNetAE | completed | 0.7397 | 0.7377 | 0.8372 | 0.7200 | 0.7423 |
| coef_raw | left | MobileNet | completed | 0.7808 | 0.7696 | 0.8470 | 0.7333 | 0.7640 |
| coef_raw | left | SqueezeNet | completed | 0.7260 | 0.7271 | 0.8535 | 0.7533 | 0.7534 |
| coef_raw | right | SVM | completed | 0.8182 | 0.7262 | 0.8401 | 0.8467 | 0.8783 |
| coef_raw | right | MLP | completed | 0.7922 | 0.7381 | 0.8384 | 0.8500 | 0.8621 |
| coef_raw | right | ResNet | completed | 0.8442 | 0.7589 | 0.8427 | 0.8433 | 0.8530 |
| coef_raw | right | ResNetAE | completed | 0.8442 | 0.7589 | 0.8588 | 0.8467 | 0.8610 |
| coef_raw | right | MobileNet | completed | 0.8182 | 0.7411 | 0.8427 | 0.8433 | 0.8473 |
| coef_raw | right | SqueezeNet | completed | 0.8701 | 0.8214 | 0.8576 | 0.8400 | 0.8609 |
| coef_raw_balanced_jitter | left | SVM | completed | 0.7808 | 0.7696 | 0.8478 | 0.7433 | 0.7592 |
| coef_raw_balanced_jitter | left | MLP | completed | 0.7534 | 0.7484 | 0.8511 | 0.7300 | 0.7355 |
| coef_raw_balanced_jitter | left | ResNet | completed | 0.7808 | 0.7782 | 0.8470 | 0.7333 | 0.7424 |
| coef_raw_balanced_jitter | left | ResNetAE | completed | 0.7671 | 0.7504 | 0.8650 | 0.7300 | 0.7474 |
| coef_raw_balanced_jitter | left | MobileNet | completed | 0.7123 | 0.7164 | 0.7954 | 0.7433 | 0.7545 |
| coef_raw_balanced_jitter | left | SqueezeNet | completed | 0.7808 | 0.7353 | 0.8355 | 0.7400 | 0.7547 |
| coef_raw_balanced_jitter | right | SVM | completed | 0.8182 | 0.7113 | 0.8376 | 0.8400 | 0.8602 |
| coef_raw_balanced_jitter | right | MLP | completed | 0.8312 | 0.7500 | 0.8384 | 0.8367 | 0.8553 |
| coef_raw_balanced_jitter | right | ResNet | completed | 0.8052 | 0.7024 | 0.8376 | 0.8367 | 0.8451 |
| coef_raw_balanced_jitter | right | ResNetAE | completed | 0.7922 | 0.6786 | 0.8495 | 0.8400 | 0.8539 |
| coef_raw_balanced_jitter | right | MobileNet | completed | 0.8312 | 0.7351 | 0.8333 | 0.8433 | 0.8490 |
| coef_raw_balanced_jitter | right | SqueezeNet | completed | 0.8312 | 0.7500 | 0.8401 | 0.8500 | 0.8434 |
| coef_plsda | left | SVM | completed | 0.7808 | 0.7696 | 0.8519 | 0.7433 | 0.7658 |
| coef_plsda | left | MLP | completed | 0.7260 | 0.7271 | 0.8560 | 0.7167 | 0.7381 |
| coef_plsda | left | ResNet | completed | 0.7808 | 0.7696 | 0.8715 | 0.7333 | 0.7481 |
| coef_plsda | left | ResNetAE | completed | 0.7671 | 0.7504 | 0.8543 | 0.7333 | 0.7446 |
| coef_plsda | left | MobileNet | completed | 0.7808 | 0.7610 | 0.8572 | 0.7267 | 0.7340 |
| coef_plsda | left | SqueezeNet | completed | 0.7671 | 0.7418 | 0.8331 | 0.7633 | 0.7733 |
| coef_plsda | right | SVM | completed | 0.8701 | 0.7917 | 0.8503 | 0.8467 | 0.8766 |
| coef_plsda | right | MLP | completed | 0.8442 | 0.7738 | 0.8333 | 0.8467 | 0.8641 |
| coef_plsda | right | ResNet | completed | 0.8182 | 0.7113 | 0.7972 | 0.8433 | 0.8652 |
| coef_plsda | right | ResNetAE | completed | 0.8052 | 0.7024 | 0.8223 | 0.8333 | 0.8506 |
| coef_plsda | right | MobileNet | completed | 0.7792 | 0.6845 | 0.8333 | 0.8400 | 0.8345 |
| coef_plsda | right | SqueezeNet | completed | 0.8312 | 0.7500 | 0.8486 | 0.8000 | 0.8376 |
| augment_plsda_balanced | left | SVM | completed | 0.7534 | 0.7398 | 0.8372 | 0.7167 | 0.7377 |
| augment_plsda_balanced | left | MLP | completed | 0.7260 | 0.7185 | 0.8314 | 0.7067 | 0.7200 |
| augment_plsda_balanced | left | ResNet | completed | 0.7397 | 0.7377 | 0.8101 | 0.7333 | 0.7207 |
| augment_plsda_balanced | left | ResNetAE | completed | 0.7671 | 0.7504 | 0.8282 | 0.7133 | 0.7310 |
| augment_plsda_balanced | left | MobileNet | completed | 0.7671 | 0.7504 | 0.7848 | 0.7133 | 0.7165 |
| augment_plsda_balanced | left | SqueezeNet | completed | 0.7260 | 0.7185 | 0.8114 | 0.7067 | 0.7175 |
| augment_plsda_balanced | right | SVM | completed | 0.8182 | 0.7262 | 0.7934 | 0.8067 | 0.8371 |
| augment_plsda_balanced | right | MLP | completed | 0.8182 | 0.7411 | 0.8053 | 0.7933 | 0.8215 |
| augment_plsda_balanced | right | ResNet | completed | 0.8442 | 0.7738 | 0.8061 | 0.8167 | 0.8197 |
| augment_plsda_balanced | right | ResNetAE | completed | 0.8182 | 0.7262 | 0.7942 | 0.7833 | 0.8160 |
| augment_plsda_balanced | right | MobileNet | completed | 0.8312 | 0.7500 | 0.8104 | 0.7900 | 0.8100 |
| augment_plsda_balanced | right | SqueezeNet | completed | 0.7922 | 0.7232 | 0.7768 | 0.8000 | 0.8162 |
| plsda_latent_features | left | SVM | completed | 0.7671 | 0.7590 | 0.8421 | 0.7433 | 0.7511 |
| plsda_latent_features | left | MLP | completed | 0.7534 | 0.7484 | 0.8437 | 0.7233 | 0.7268 |
| plsda_latent_features | left | ResNet | completed | 0.7534 | 0.7398 | 0.8322 | 0.7333 | 0.7429 |
| plsda_latent_features | left | ResNetAE | completed | 0.7671 | 0.7590 | 0.8412 | 0.7133 | 0.7447 |
| plsda_latent_features | left | MobileNet | completed | 0.7808 | 0.7610 | 0.8216 | 0.7133 | 0.7355 |
| plsda_latent_features | left | SqueezeNet | completed | 0.7123 | 0.7164 | 0.8425 | 0.7533 | 0.7652 |
| plsda_latent_features | right | SVM | completed | 0.8182 | 0.7262 | 0.8325 | 0.8400 | 0.8685 |
| plsda_latent_features | right | MLP | completed | 0.7792 | 0.7292 | 0.8316 | 0.8233 | 0.8503 |
| plsda_latent_features | right | ResNet | completed | 0.7792 | 0.6845 | 0.8257 | 0.8333 | 0.8625 |
| plsda_latent_features | right | ResNetAE | completed | 0.8571 | 0.7827 | 0.8469 | 0.8400 | 0.8595 |
| plsda_latent_features | right | MobileNet | completed | 0.8052 | 0.7619 | 0.8359 | 0.8267 | 0.8415 |
| plsda_latent_features | right | SqueezeNet | completed | 0.8701 | 0.8661 | 0.8622 | 0.8533 | 0.8637 |
| pointnet_raw | left | PointNet | completed | 0.6849 | 0.6866 | 0.7610 | 0.6833 | 0.7215 |
| pointnet_raw | right | PointNet | completed | 0.7013 | 0.6607 | 0.7134 | 0.6800 | 0.7467 |
| pointnet_raw_balanced_jitter | left | PointNet | completed | 0.7397 | 0.7034 | 0.7561 | 0.6800 | 0.7193 |
| pointnet_raw_balanced_jitter | right | PointNet | completed | 0.7403 | 0.6429 | 0.7534 | 0.7567 | 0.7675 |
| pointnet_plsda | left | PointNet | completed | 0.7260 | 0.6240 | 0.7619 | 0.7033 | 0.7392 |
| pointnet_plsda | right | PointNet | completed | 0.6494 | 0.6399 | 0.7049 | 0.7767 | 0.7416 |
| plsda_direct | left | PLSDA | completed | 0.7671 | 0.7590 | 0.8429 | 0.7467 | 0.7448 |
| plsda_direct | right | PLSDA | completed | 0.8312 | 0.7500 | 0.8427 | 0.8433 | 0.8634 |

Each dataset/side/model folder contains model/preprocessing files, metrics.json, predictions, training_history.csv, plots, run_manifest.json and validation_report.json.
Logs are in _logs/<dataset>_<side>_<model>.log. results_summary.csv contains all metrics; run_summary.json records exact commands, timings and failures.
Every model, including Direct PLS-DA, uses the same 10-fold person assignments within each hemisphere. oof_predictions.csv/.npz, oof_metrics.json and fold_metrics.csv are saved per job; fold_01..fold_10 contain checkpoints and provenance.
