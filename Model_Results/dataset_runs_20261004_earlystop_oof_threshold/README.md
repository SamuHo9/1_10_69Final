# Dataset model runs

Scope: all. Seed: 42. Neural maximum epochs: 100. Coefficient model PLS components: 8.
Each prepared dataset and hemisphere is trained independently. Existing repository architecture classes are loaded without legacy hardcoded dataset paths.
Coefficient models use train-fitted StandardScaler and binary-target PLS. Latent data are rebuilt using one-hot PLS from the corresponding original coefficients. PointNet uses the supplied 1002-point XYZ clouds.
Exactly 10 stratified folds of original Train people for every model. Each original row has one OOF prediction; synthetic rows are not evaluated as OOF. No silent fold reduction.
Augmentation, scaling and supervised PLS are regenerated/fitted inside each training fold. Latent inputs are rebuilt from original coefficients inside each fold.
Select fold checkpoints using separate inner stopping people, then select a probability threshold maximizing original Train OOF accuracy. Freeze the threshold before full-Train refit and Test prediction. Final refit epochs are ceil(median(inner best epochs)); tuned OOF scores are selection scores.
Prepared synthetic CSV rows and pre-fitted latent coordinates are not reused in outer folds. The dataset preparation method is rebuilt from its original training people and data_manifest.json.
Input limitations: current labels are hemisphere labels; existing ICP reference included test. These results are an exploratory comparison on the supplied datasets.

| Dataset | Side | Model | Status | Test accuracy | Balanced accuracy | ROC-AUC | OOF accuracy | OOF ROC-AUC |
|---|---|---|---|---:|---:|---:|---:|---:|
| coef_raw | left | SVM | completed | 0.8356 | 0.7864 | 0.8412 | 0.7500 | 0.7608 |
| coef_raw | left | MLP | completed | 0.8356 | 0.7950 | 0.8609 | 0.7633 | 0.7702 |
| coef_raw | left | ResNet | completed | 0.7945 | 0.7717 | 0.8592 | 0.7767 | 0.7882 |
| coef_raw | left | ResNetAE | completed | 0.7945 | 0.7545 | 0.8601 | 0.7533 | 0.7681 |
| coef_raw | left | MobileNet | completed | 0.7808 | 0.7610 | 0.8658 | 0.7100 | 0.7226 |
| coef_raw | right | SVM | completed | 0.8182 | 0.6964 | 0.8401 | 0.8467 | 0.8783 |
| coef_raw | left | SqueezeNet | completed | 0.7397 | 0.7377 | 0.8543 | 0.7167 | 0.7439 |
| coef_raw | right | MLP | completed | 0.8052 | 0.7768 | 0.8741 | 0.8467 | 0.8334 |
| coef_raw | right | ResNet | completed | 0.8052 | 0.6875 | 0.8163 | 0.8333 | 0.8294 |
| coef_raw | right | ResNetAE | completed | 0.8182 | 0.7113 | 0.8452 | 0.8333 | 0.8504 |
| coef_raw | right | MobileNet | completed | 0.8182 | 0.7560 | 0.8639 | 0.8367 | 0.8075 |
| coef_raw_balanced_jitter | left | SVM | completed | 0.8356 | 0.7950 | 0.8478 | 0.7433 | 0.7592 |
| coef_raw_balanced_jitter | left | MLP | completed | 0.7945 | 0.7889 | 0.8486 | 0.7667 | 0.7915 |
| coef_raw | right | SqueezeNet | completed | 0.8182 | 0.6815 | 0.8546 | 0.7567 | 0.7854 |
| coef_raw_balanced_jitter | left | ResNet | completed | 0.8356 | 0.8036 | 0.8363 | 0.7700 | 0.8002 |
| coef_raw_balanced_jitter | left | ResNetAE | completed | 0.7808 | 0.7696 | 0.8519 | 0.7633 | 0.7682 |
| coef_raw_balanced_jitter | left | MobileNet | completed | 0.8219 | 0.7930 | 0.8699 | 0.7200 | 0.7611 |
| coef_raw_balanced_jitter | right | SVM | completed | 0.8052 | 0.6875 | 0.8376 | 0.8400 | 0.8602 |
| coef_raw_balanced_jitter | left | SqueezeNet | completed | 0.6712 | 0.6330 | 0.7643 | 0.7433 | 0.7627 |
| coef_raw_balanced_jitter | right | MLP | completed | 0.8442 | 0.7589 | 0.8529 | 0.8300 | 0.8414 |
| coef_raw_balanced_jitter | right | ResNet | completed | 0.8312 | 0.7500 | 0.8512 | 0.8533 | 0.8357 |
| coef_raw_balanced_jitter | right | ResNetAE | completed | 0.7922 | 0.6637 | 0.8690 | 0.8367 | 0.8369 |
| coef_raw_balanced_jitter | right | MobileNet | completed | 0.8442 | 0.7738 | 0.8299 | 0.8200 | 0.8174 |
| coef_plsda | left | SVM | completed | 0.8356 | 0.7864 | 0.8519 | 0.7433 | 0.7658 |
| coef_plsda | left | MLP | completed | 0.7945 | 0.7803 | 0.8527 | 0.7600 | 0.7788 |
| coef_raw_balanced_jitter | right | SqueezeNet | completed | 0.7922 | 0.6637 | 0.8478 | 0.7700 | 0.7616 |
| coef_plsda | left | ResNet | completed | 0.7945 | 0.7631 | 0.8363 | 0.7900 | 0.8036 |
| coef_plsda | left | ResNetAE | completed | 0.7945 | 0.7717 | 0.8519 | 0.7467 | 0.7753 |
| coef_plsda | left | MobileNet | completed | 0.7397 | 0.7205 | 0.8380 | 0.6567 | 0.7395 |
| coef_plsda | right | SVM | completed | 0.8052 | 0.6726 | 0.8503 | 0.8467 | 0.8766 |
| coef_plsda | left | SqueezeNet | completed | 0.7397 | 0.7034 | 0.7921 | 0.7467 | 0.7682 |
| coef_plsda | right | MLP | completed | 0.8571 | 0.7679 | 0.8767 | 0.8167 | 0.8420 |
| coef_plsda | right | ResNet | completed | 0.8442 | 0.7589 | 0.8520 | 0.8367 | 0.8525 |
| coef_plsda | right | ResNetAE | completed | 0.8312 | 0.7202 | 0.8095 | 0.8367 | 0.8311 |
| coef_plsda | right | MobileNet | completed | 0.8182 | 0.7560 | 0.8146 | 0.8300 | 0.8247 |
| augment_plsda_balanced | left | SVM | completed | 0.7671 | 0.7160 | 0.8372 | 0.7167 | 0.7377 |
| coef_plsda | right | SqueezeNet | completed | 0.8312 | 0.7500 | 0.8512 | 0.7700 | 0.7352 |
| augment_plsda_balanced | left | MLP | completed | 0.7534 | 0.7398 | 0.8470 | 0.7433 | 0.7661 |
| augment_plsda_balanced | left | ResNet | completed | 0.7397 | 0.7291 | 0.8290 | 0.7300 | 0.7493 |
| augment_plsda_balanced | left | ResNetAE | completed | 0.7534 | 0.7312 | 0.8380 | 0.7133 | 0.7400 |
| augment_plsda_balanced | left | MobileNet | completed | 0.7534 | 0.7398 | 0.8142 | 0.7433 | 0.7737 |
| augment_plsda_balanced | left | SqueezeNet | completed | 0.6986 | 0.6886 | 0.7668 | 0.7067 | 0.7352 |
| augment_plsda_balanced | right | SVM | completed | 0.8182 | 0.7262 | 0.7934 | 0.8067 | 0.8371 |
| augment_plsda_balanced | right | MLP | completed | 0.7922 | 0.6935 | 0.8172 | 0.7833 | 0.7838 |
| augment_plsda_balanced | right | ResNet | completed | 0.8182 | 0.7262 | 0.7466 | 0.7933 | 0.8035 |
| augment_plsda_balanced | right | ResNetAE | completed | 0.8312 | 0.7351 | 0.7781 | 0.7767 | 0.7948 |
| augment_plsda_balanced | right | MobileNet | completed | 0.8182 | 0.7262 | 0.7721 | 0.8067 | 0.8000 |
| plsda_latent_features | left | SVM | completed | 0.7945 | 0.7631 | 0.8421 | 0.7433 | 0.7511 |
| plsda_latent_features | left | MLP | completed | 0.8082 | 0.7909 | 0.8478 | 0.7267 | 0.7259 |
| plsda_latent_features | left | ResNet | completed | 0.7671 | 0.7504 | 0.8658 | 0.7467 | 0.7709 |
| augment_plsda_balanced | right | SqueezeNet | completed | 0.7662 | 0.7351 | 0.7594 | 0.7533 | 0.7623 |
| plsda_latent_features | left | ResNetAE | completed | 0.7808 | 0.7525 | 0.8625 | 0.7467 | 0.7746 |
| plsda_latent_features | left | SqueezeNet | completed | 0.7123 | 0.7164 | 0.8445 | 0.6867 | 0.7363 |
| plsda_latent_features | left | MobileNet | completed | 0.7808 | 0.7696 | 0.8502 | 0.6667 | 0.7143 |
| plsda_latent_features | right | SVM | completed | 0.8052 | 0.6875 | 0.8325 | 0.8400 | 0.8685 |
| plsda_latent_features | right | MLP | completed | 0.8312 | 0.7351 | 0.8342 | 0.8100 | 0.7994 |
| plsda_latent_features | right | ResNet | completed | 0.8312 | 0.7649 | 0.8223 | 0.8233 | 0.8187 |
| plsda_latent_features | right | ResNetAE | completed | 0.8182 | 0.6964 | 0.8350 | 0.8300 | 0.8188 |
| plsda_latent_features | right | MobileNet | completed | 0.8312 | 0.7649 | 0.8019 | 0.8300 | 0.8179 |
| plsda_latent_features | right | SqueezeNet | completed | 0.8701 | 0.8214 | 0.8614 | 0.7933 | 0.7955 |
| pointnet_raw | left | PointNet | completed | 0.6438 | 0.5000 | 0.4959 | 0.6233 | 0.6569 |
| pointnet_raw | right | PointNet | completed | 0.7013 | 0.4821 | 0.6726 | 0.6567 | 0.6406 |
| pointnet_raw_balanced_jitter | left | PointNet | completed | 0.4795 | 0.5614 | 0.7848 | 0.6967 | 0.7110 |
| pointnet_raw_balanced_jitter | right | PointNet | completed | 0.7403 | 0.6429 | 0.7007 | 0.7867 | 0.7606 |
| pointnet_plsda | left | PointNet | completed | 0.7534 | 0.6710 | 0.6718 | 0.7233 | 0.7380 |
| plsda_direct | left | PLSDA | completed | 0.8356 | 0.7950 | 0.8429 | 0.7467 | 0.7448 |
| plsda_direct | right | PLSDA | completed | 0.8312 | 0.7649 | 0.8427 | 0.8433 | 0.8634 |
| pointnet_plsda | right | PointNet | completed | 0.7143 | 0.5208 | 0.7560 | 0.7700 | 0.7719 |

Each dataset/side/model folder contains model/preprocessing files, metrics.json, predictions, training_history.csv, plots, run_manifest.json and validation_report.json.
Logs are in _logs/<dataset>_<side>_<model>.log. results_summary.csv contains all metrics; run_summary.json records exact commands, timings and failures.
Every model, including Direct PLS-DA, uses the same 10-fold person assignments within each hemisphere. oof_predictions.csv/.npz, oof_metrics.json and fold_metrics.csv are saved per job; fold_01..fold_10 contain checkpoints and provenance.

Epoch models split each outer Train into inner fit/stop people before augmentation/scaling/PLS. Inner stopping loss alone selects the best checkpoint; outer OOF is predicted only after restoring it.
Neural maximum epochs 100; MLP maximum epochs 100. Final full-Train refit uses ceil(median(best epochs)) from inner stops; no Test-based epoch selection.
cv_oof reports default-rule outer OOF. Threshold-tuned OOF scores are selection scores, not independent performance estimates. Test is the primary evaluation of the frozen threshold.