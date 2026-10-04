# Dataset model runs

Scope: all. Seed: 42. Neural epochs: 80. Coefficient model PLS components: 8.
Each prepared dataset and hemisphere is trained independently. Existing repository architecture classes are loaded without legacy hardcoded dataset paths.
Coefficient models use train-fitted StandardScaler and binary-target PLS; supplied latent datasets are scaled without another PLS step. PointNet uses the supplied 1002-point XYZ clouds.
Fixed training settings, full supplied train fit, no CV or test-based model/epoch selection. MLP uses training-loss stopping (no random validation split). Test is used only for final prediction.
For cross-validation, split original persons and redo supervised preprocessing and augmentation inside each training fold. Do not split these pre-augmented inputs into CV folds.
Input limitations: current labels are hemisphere labels; existing ICP reference included test. These results are an exploratory comparison on the supplied datasets.

| Dataset | Side | Model | Status | Test accuracy | Balanced accuracy | ROC-AUC |
|---|---|---|---|---:|---:|---:|
| coef_raw | left | SVM | completed | 0.7808 | 0.7696 | 0.8412 |
| coef_raw | left | MLP | completed | 0.7397 | 0.7377 | 0.8511 |
| coef_raw | left | ResNet | completed | 0.7671 | 0.7504 | 0.8372 |
| coef_raw | left | ResNetAE | completed | 0.7397 | 0.7377 | 0.8372 |
| coef_raw | left | MobileNet | completed | 0.7808 | 0.7696 | 0.8470 |
| coef_raw | left | SqueezeNet | completed | 0.7260 | 0.7271 | 0.8535 |
| coef_raw | right | SVM | completed | 0.8182 | 0.7262 | 0.8401 |
| coef_raw | right | MLP | completed | 0.7922 | 0.7381 | 0.8384 |
| coef_raw | right | ResNet | completed | 0.8442 | 0.7589 | 0.8427 |
| coef_raw | right | ResNetAE | completed | 0.8442 | 0.7589 | 0.8588 |
| coef_raw | right | MobileNet | completed | 0.8182 | 0.7411 | 0.8427 |
| coef_raw | right | SqueezeNet | completed | 0.8701 | 0.8214 | 0.8576 |
| coef_raw_balanced_jitter | left | SVM | completed | 0.7808 | 0.7610 | 0.8412 |
| coef_raw_balanced_jitter | left | MLP | completed | 0.7123 | 0.7164 | 0.8314 |
| coef_raw_balanced_jitter | left | ResNet | completed | 0.7260 | 0.7185 | 0.8159 |
| coef_raw_balanced_jitter | left | ResNetAE | completed | 0.7671 | 0.7590 | 0.8331 |
| coef_raw_balanced_jitter | left | MobileNet | completed | 0.7534 | 0.7398 | 0.8151 |
| coef_raw_balanced_jitter | left | SqueezeNet | completed | 0.7671 | 0.7504 | 0.8265 |
| coef_raw_balanced_jitter | right | SVM | completed | 0.8442 | 0.7440 | 0.8274 |
| coef_raw_balanced_jitter | right | MLP | completed | 0.8182 | 0.6964 | 0.8376 |
| coef_raw_balanced_jitter | right | ResNet | completed | 0.8182 | 0.6964 | 0.8206 |
| coef_raw_balanced_jitter | right | ResNetAE | completed | 0.8182 | 0.6964 | 0.8274 |
| coef_raw_balanced_jitter | right | MobileNet | completed | 0.8052 | 0.7024 | 0.7976 |
| coef_raw_balanced_jitter | right | SqueezeNet | completed | 0.8312 | 0.7649 | 0.8295 |
| coef_plsda | left | SVM | completed | 0.7808 | 0.7610 | 0.8511 |
| coef_plsda | left | MLP | completed | 0.7397 | 0.7377 | 0.8535 |
| coef_plsda | left | ResNet | completed | 0.7945 | 0.7717 | 0.8502 |
| coef_plsda | left | ResNetAE | completed | 0.7671 | 0.7418 | 0.8584 |
| coef_plsda | left | MobileNet | completed | 0.7808 | 0.7610 | 0.8453 |
| coef_plsda | left | SqueezeNet | completed | 0.6849 | 0.6952 | 0.8584 |
| coef_plsda | right | SVM | completed | 0.8312 | 0.7500 | 0.8282 |
| coef_plsda | right | MLP | completed | 0.8312 | 0.7649 | 0.8342 |
| coef_plsda | right | ResNet | completed | 0.8312 | 0.7202 | 0.7993 |
| coef_plsda | right | ResNetAE | completed | 0.8182 | 0.7262 | 0.8461 |
| coef_plsda | right | MobileNet | completed | 0.8571 | 0.7827 | 0.8427 |
| coef_plsda | right | SqueezeNet | completed | 0.8182 | 0.7411 | 0.8308 |
| augment_plsda_balanced | left | SVM | completed | 0.7534 | 0.7398 | 0.8372 |
| augment_plsda_balanced | left | MLP | completed | 0.7260 | 0.7185 | 0.8314 |
| augment_plsda_balanced | left | ResNet | completed | 0.7397 | 0.7377 | 0.8101 |
| augment_plsda_balanced | left | ResNetAE | completed | 0.7671 | 0.7504 | 0.8282 |
| augment_plsda_balanced | left | MobileNet | completed | 0.7671 | 0.7504 | 0.7848 |
| augment_plsda_balanced | left | SqueezeNet | completed | 0.7260 | 0.7185 | 0.8114 |
| augment_plsda_balanced | right | SVM | completed | 0.8182 | 0.7262 | 0.7934 |
| augment_plsda_balanced | right | MLP | completed | 0.8182 | 0.7411 | 0.8053 |
| augment_plsda_balanced | right | ResNet | completed | 0.8442 | 0.7738 | 0.8061 |
| augment_plsda_balanced | right | ResNetAE | completed | 0.8182 | 0.7262 | 0.7942 |
| augment_plsda_balanced | right | MobileNet | completed | 0.8312 | 0.7500 | 0.8104 |
| augment_plsda_balanced | right | SqueezeNet | completed | 0.7922 | 0.7232 | 0.7768 |
| plsda_latent_features | left | SVM | completed | 0.7671 | 0.7590 | 0.8421 |
| plsda_latent_features | left | MLP | completed | 0.7534 | 0.7484 | 0.8437 |
| plsda_latent_features | left | ResNet | completed | 0.7534 | 0.7398 | 0.8322 |
| plsda_latent_features | left | ResNetAE | completed | 0.7671 | 0.7590 | 0.8412 |
| plsda_latent_features | left | MobileNet | completed | 0.7808 | 0.7610 | 0.8216 |
| plsda_latent_features | left | SqueezeNet | completed | 0.7123 | 0.7164 | 0.8425 |
| plsda_latent_features | right | SVM | completed | 0.8182 | 0.7262 | 0.8325 |
| plsda_latent_features | right | MLP | completed | 0.7792 | 0.7292 | 0.8316 |
| plsda_latent_features | right | ResNet | completed | 0.7792 | 0.6845 | 0.8257 |
| plsda_latent_features | right | ResNetAE | completed | 0.8571 | 0.7827 | 0.8469 |
| plsda_latent_features | right | MobileNet | completed | 0.8052 | 0.7619 | 0.8359 |
| plsda_latent_features | right | SqueezeNet | completed | 0.8701 | 0.8661 | 0.8622 |
| pointnet_raw | left | PointNet | completed | 0.6849 | 0.6866 | 0.7610 |
| pointnet_raw | right | PointNet | completed | 0.7013 | 0.6607 | 0.7134 |
| pointnet_raw_balanced_jitter | left | PointNet | completed | 0.7534 | 0.7226 | 0.7430 |
| pointnet_raw_balanced_jitter | right | PointNet | completed | 0.8052 | 0.7024 | 0.7577 |
| pointnet_plsda | left | PointNet | completed | 0.7671 | 0.6817 | 0.7070 |
| pointnet_plsda | right | PointNet | completed | 0.5844 | 0.5952 | 0.6820 |
| plsda_direct | left | PLSDA | completed | 0.7671 | 0.7590 | 0.8429 |
| plsda_direct | right | PLSDA | completed | 0.8312 | 0.7500 | 0.8427 |

Each dataset/side/model folder contains model/preprocessing files, metrics.json, predictions, training_history.csv, plots, run_manifest.json and validation_report.json.
Logs are in _logs/<dataset>_<side>_<model>.log. results_summary.csv contains all metrics; run_summary.json records exact commands, timings and failures.
Direct PLS-DA is retrained through the existing prepare_plsda_direct.py entry point. It also performs its own 5-fold grouped CV on the original coefficient data.
