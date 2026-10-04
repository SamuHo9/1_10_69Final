"""Prepare coefficient data with same-class PLS-DA augmentation.

PLS-DA is fit only on the supplied train CSV.  Unique parent pairs are selected
inside each class in PLS score space, synthetic rows are reconstructed back to
the original 507 coefficient space, and the test CSV is left unaugmented.
Run without arguments to open the GUI.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from data_prep_common import (
    SYNTH_PROVENANCE_COLUMNS,
    augment_plsda_same_class,
    load_pair,
    launch_gui,
    parse_int,
    write_preparation_outputs,
)


def process(
    train_csv: str,
    test_csv: str,
    output_dir: str,
    label_column: str = "BinaryClass",
    n_components: int = 8,
    seed: int = 42,
    children_per_pair: int = 8,
    augmentation_size: str = "maximum",
) -> str:
    train, test = load_pair(train_csv, test_csv, "coef", label_column)
    prepared, synthetic, pairs, bundle, info = augment_plsda_same_class(
        train,
        "coef",
        seed=seed,
        requested_components=n_components,
        children_per_pair=children_per_pair,
        augmentation_size=augmentation_size,
        stream=True,
        protocol="coef_plsda_same_class",
    )
    root = write_preparation_outputs(
        output_dir=output_dir,
        protocol="coef_plsda",
        kind="coef",
        input_train=train_csv,
        input_test=test_csv,
        original_train=train,
        prepared_train=prepared,
        prepared_test=test,
        synthetic=synthetic,
        pairs=pairs,
        info=info,
        scaler=bundle["scaler"],
        pls_bundle=bundle,
        extra_manifest={
            "model_input": "Coef_1..Coef_507",
            "pls_da_used_for_augmentation": True,
            "pls_da_applied_to_model_input": False,
            "test_preprocessing": "none",
            "seed": int(seed),
        },
    )
    return f"coef_plsda completed. Output: {root}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="BinaryClass")
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--children-per-pair", type=int, default=8)
    parser.add_argument("--augmentation-size", choices=("maximum", "balance_only"), default="maximum")
    return parser


def gui() -> None:
    fields = [
        {"name": "train_csv", "label": "Train CSV", "browse": "file"},
        {"name": "test_csv", "label": "Test CSV", "browse": "file"},
        {"name": "output_dir", "label": "Output folder", "browse": "dir"},
        {"name": "label_column", "label": "Label column", "default": "BinaryClass"},
        {"name": "n_components", "label": "PLS components", "default": "8"},
        {"name": "seed", "label": "Seed", "default": "42"},
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
            parse_int(values["n_components"] or "8", "PLS components", 1),
            parse_int(values["seed"] or "42", "seed", 0),
            parse_int(
                values["children_per_pair"] or "8",
                "children per pair",
                1,
            ),
            values["augmentation_size"] or "maximum",
        )

    launch_gui(
        "Prepare coef_plsda",
        fields,
        run,
        "PLS-DA is used only to make same-class synthetic coefficient rows. "
        "The downstream model still receives 507 coefficients.",
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
            args.seed,
            args.children_per_pair,
            args.augmentation_size,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
