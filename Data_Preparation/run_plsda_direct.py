"""Run the direct PLS-DA classifier on coefficient CSVs.

This is not an augmentation or latent-feature export script.  PLS-DA itself
is the classifier: it is fit separately inside grouped CV folds for OOF
predictions, then once on all original train rows for the untouched test set.
Run without arguments to open the GUI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from data_prep_common import (
    COEF_COLUMNS,
    as_path,
    class_counts,
    fit_plsda,
    launch_gui,
    load_pair,
    parse_float,
    parse_int,
    patient_group_id,
    save_pickle,
    write_json,
)


def _probability(bundle: dict, x: np.ndarray) -> np.ndarray:
    scores = np.asarray(
        bundle["pls"].predict(bundle["scaler"].transform(x)),
        dtype=float,
    )
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError(f"Expected two PLS-DA class scores, got {scores.shape}")
    scores = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(scores)
    return np.clip(exponent[:, 1] / exponent.sum(axis=1), 0.0, 1.0)


def _metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "sensitivity": float(
            recall_score(y_true, prediction, pos_label=1, zero_division=0)
        ),
        "specificity": float(
            recall_score(y_true, prediction, pos_label=0, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, prediction, average="macro", zero_division=0)
        ),
        "threshold": float(threshold),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
    except ValueError:
        result["roc_auc"] = None
    return result


def _grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    requested_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups, first_index = np.unique(groups, return_index=True)
    group_y = y[first_index]
    if any(
        np.unique(y[groups == group]).size != 1
        for group in unique_groups
    ):
        raise ValueError("A patient group contains conflicting class labels")
    class_group_counts = np.bincount(group_y.astype(int), minlength=2)
    folds = int(requested_folds)
    if folds != 10:
        raise ValueError('Exactly 10 grouped folds are required')
    if int(class_group_counts.min()) < folds:
        raise ValueError(
            "Exactly 10-fold CV requires at least 10 original patient groups per class"
        )

    for split_seed in (seed, 42, 123, 2026, 7, 99):
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=split_seed,
        )
        candidate = list(splitter.split(np.zeros(len(y)), y, groups))
        valid = True
        for train_index, validation_index in candidate:
            if len(np.unique(y[train_index])) < 2:
                valid = False
            if len(np.unique(y[validation_index])) < 2:
                valid = False
            if set(groups[train_index]) & set(groups[validation_index]):
                valid = False
        if valid:
            return candidate
    raise ValueError(
        "Could not construct grouped stratified folds containing both classes"
    )


def process(
    train_csv: str,
    test_csv: str,
    output_dir: str,
    label_column: str = "BinaryClass",
    n_components: int = 8,
    folds: int = 10,
    seed: int = 42,
    threshold: float = 0.5,
) -> str:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    train, test = load_pair(train_csv, test_csv, "coef", label_column)
    x = train[COEF_COLUMNS].to_numpy(dtype=float)
    y = train["BinaryClass"].to_numpy(dtype=int)
    test_x = test[COEF_COLUMNS].to_numpy(dtype=float)
    groups = np.asarray([patient_group_id(value) for value in train["Subject"]])
    test_groups = np.asarray(
        [patient_group_id(value) for value in test["Subject"]]
    )
    if set(groups) & set(test_groups):
        raise ValueError("Patient overlap between train and test")

    splits = _grouped_splits(y, groups, folds, seed)
    oof_probability = np.full(len(y), np.nan, dtype=float)
    fold_records: list[dict] = []
    for fold_number, (train_index, validation_index) in enumerate(
        splits,
        start=1,
    ):
        bundle = fit_plsda(x[train_index], y[train_index], n_components)
        oof_probability[validation_index] = _probability(
            bundle,
            x[validation_index],
        )
        fold_records.append(
            {
                "fold": fold_number,
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "train_patient_groups": int(len(set(groups[train_index]))),
                "validation_patient_groups": int(
                    len(set(groups[validation_index]))
                ),
                "effective_components": int(bundle["effective_components"]),
            }
        )
    if not np.isfinite(oof_probability).all():
        raise RuntimeError("Grouped CV did not produce an OOF probability for every row")

    final_bundle = fit_plsda(x, y, n_components)
    test_probability = _probability(final_bundle, test_x)
    root = as_path(output_dir, "output directory")
    root.mkdir(parents=True, exist_ok=True)
    save_pickle(root / "direct_plsda_model.pkl", final_bundle)

    oof_prediction = (oof_probability >= threshold).astype(int)
    test_prediction = (test_probability >= threshold).astype(int)
    pd.DataFrame(
        {
            "Subject": train["Subject"].astype(str),
            "PatientID": groups,
            "BinaryClass": y,
            "Probability": oof_probability,
            "Prediction": oof_prediction,
        }
    ).to_csv(root / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "Subject": test["Subject"].astype(str),
            "PatientID": test_groups,
            "BinaryClass": test["BinaryClass"].to_numpy(dtype=int),
            "Probability": test_probability,
            "Prediction": test_prediction,
        }
    ).to_csv(root / "test_predictions.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "cv_oof": _metrics(y, oof_probability, threshold),
        "test": _metrics(
            test["BinaryClass"].to_numpy(dtype=int),
            test_probability,
            threshold,
        ),
    }
    write_json(root / "metrics.json", metrics)
    write_json(
        root / "run_manifest.json",
        {
            "schema": "direct_plsda_run_manifest_v1",
            "protocol": "plsda_direct",
            "classifier": "PLSRegression(one-hot target)+two-class-softmax",
            "model_input": "Coef_1..Coef_507",
            "pls_da_used_for_augmentation": False,
            "pls_da_applied_to_model_input": True,
            "synthetic_rows": 0,
            "input_train": str(as_path(train_csv, "train CSV")),
            "input_test": str(as_path(test_csv, "test CSV")),
            "output_dir": str(root),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_class_counts": class_counts(train),
            "test_class_counts": class_counts(test),
            "requested_components": int(n_components),
            "effective_components": int(
                final_bundle["effective_components"]
            ),
            "requested_folds": int(folds),
            "actual_folds": int(len(splits)),
            "seed": int(seed),
            "threshold": float(threshold),
            "test_sha256": __import__(
                "data_prep_common"
            ).sha256_file(as_path(test_csv, "test CSV", True)),
            "folds": fold_records,
        },
    )
    return f"plsda_direct completed. Output: {root}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="BinaryClass")
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--folds", type=int, choices=[10], default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def gui() -> None:
    fields = [
        {"name": "train_csv", "label": "Train CSV", "browse": "file"},
        {"name": "test_csv", "label": "Test CSV", "browse": "file"},
        {"name": "output_dir", "label": "Output folder", "browse": "dir"},
        {"name": "label_column", "label": "Label column", "default": "BinaryClass"},
        {"name": "n_components", "label": "PLS components", "default": "8"},
        {"name": "folds", "label": "Grouped CV folds (10)", "default": "10"},
        {"name": "seed", "label": "Seed", "default": "42"},
        {"name": "threshold", "label": "Decision threshold", "default": "0.5"},
    ]

    def run(values: dict[str, str]) -> str:
        return process(
            values["train_csv"],
            values["test_csv"],
            values["output_dir"],
            values["label_column"] or "BinaryClass",
            parse_int(values["n_components"] or "8", "PLS components", 1),
            parse_int(values["folds"] or "10", "folds", 10),
            parse_int(values["seed"] or "42", "seed", 0),
            parse_float(values["threshold"] or "0.5", "threshold", 0.0),
        )

    launch_gui(
        "Run direct PLS-DA",
        fields,
        run,
        "This file trains PLS-DA directly. It creates no synthetic rows and "
        "does not export PLS latent features.",
    )


def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] == "--gui":
        gui()
        return 0
    args = build_parser().parse_args()
    print(
        process(
            args.train_csv,
            args.test_csv,
            args.output_dir,
            args.label_column,
            args.n_components,
            args.folds,
            args.seed,
            args.threshold,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
