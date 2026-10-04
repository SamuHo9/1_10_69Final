"""Prepare raw SPHARM coefficient data for training.

The default mode preserves the original 507 coefficient features.  Optional
balanced_jitter augmentation is same-class only and is applied to train only.
Run without command-line arguments to open the GUI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from data_prep_common import (
    SYNTH_PROVENANCE_COLUMNS,
    augment_balanced_jitter,
    load_pair,
    launch_gui,
    parse_float,
    parse_int,
    write_preparation_outputs,
)


def _empty_outputs(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    synthetic = pd.DataFrame(
        columns=list(train.columns) + SYNTH_PROVENANCE_COLUMNS
    )
    pairs = pd.DataFrame(
        columns=[
            "PairID",
            "Class",
            "ParentSubject1",
            "ParentSubject2",
            "Method",
        ]
    )
    return synthetic, pairs


def process(
    train_csv: str,
    test_csv: str,
    output_dir: str,
    label_column: str = "BinaryClass",
    augmentation: str = "none",
    seed: int = 42,
    noise_scale: float = 0.02,
    children_per_pair: int = 8,
    augmentation_size: str = "maximum",
) -> str:
    train, test = load_pair(train_csv, test_csv, "coef", label_column)
    if augmentation == "none":
        prepared = train.copy()
        synthetic, pairs = _empty_outputs(train)
        info = {
            "augmentation_protocol": "none",
            "augmentation_kind": "none",
            "same_class_pairing": False,
            "original_train_n": len(train),
            "synthetic_train_n": 0,
            "final_train_n": len(train),
        }
    elif augmentation == "balanced_jitter":
        prepared, synthetic, pairs, info = augment_balanced_jitter(
            train,
            "coef",
            seed=seed,
            noise_scale=noise_scale,
            children_per_pair=children_per_pair,
            augmentation_size=augmentation_size,
            stream=True,
            protocol="coef_raw_balanced_jitter",
        )
    else:
        raise ValueError("augmentation must be none or balanced_jitter")

    root = write_preparation_outputs(
        output_dir=output_dir,
        protocol="coef_raw",
        kind="coef",
        input_train=train_csv,
        input_test=test_csv,
        original_train=train,
        prepared_train=prepared,
        prepared_test=test,
        synthetic=synthetic,
        pairs=pairs,
        info=info,
        extra_manifest={
            "model_input": "Coef_1..Coef_507",
            "pls_da_used": False,
            "test_preprocessing": "none",
            "seed": int(seed),
        },
    )
    return f"coef_raw completed. Output: {root}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="BinaryClass")
    parser.add_argument(
        "--augmentation",
        choices=("none", "balanced_jitter"),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    parser.add_argument("--children-per-pair", type=int, default=8)
    parser.add_argument("--augmentation-size", choices=("maximum", "balance_only"), default="maximum")
    return parser


def gui() -> None:
    fields = [
        {"name": "train_csv", "label": "Train CSV", "browse": "file"},
        {"name": "test_csv", "label": "Test CSV", "browse": "file"},
        {"name": "output_dir", "label": "Output folder", "browse": "dir"},
        {"name": "label_column", "label": "Label column", "default": "BinaryClass"},
        {"name": "augmentation", "label": "Augmentation", "default": "none"},
        {"name": "seed", "label": "Seed", "default": "42"},
        {"name": "noise_scale", "label": "Noise scale", "default": "0.02"},
        {
            "name": "children_per_pair",
            "label": "Children per pair",
            "default": "8",
        },
        {"name": "augmentation_size", "label": "Augmentation size (maximum/balance_only)", "default": "maximum"},
    ]

    def run(values: dict[str, str]) -> str:
        return process(
            values["train_csv"],
            values["test_csv"],
            values["output_dir"],
            values["label_column"] or "BinaryClass",
            values["augmentation"] or "none",
            parse_int(values["seed"] or "42", "seed", 0),
            parse_float(values["noise_scale"] or "0.02", "noise scale", 0.0),
            parse_int(
                values["children_per_pair"] or "8",
                "children per pair",
                1,
            ),
            values["augmentation_size"] or "maximum",
        )

    launch_gui(
        "Prepare coef_raw",
        fields,
        run,
        "Raw coefficient input. Optional balanced_jitter uses same-class "
        "training rows only; test is never augmented.",
    )


def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] == "--gui":
        gui()
        return 0
    args = build_parser().parse_args()
    message = process(
        args.train_csv,
        args.test_csv,
        args.output_dir,
        args.label_column,
        args.augmentation,
        args.seed,
        args.noise_scale,
        args.children_per_pair,
        args.augmentation_size,
    )
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
