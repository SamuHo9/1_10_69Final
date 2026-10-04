"""Export PLS-DA latent features from coefficient CSVs.

This is the PLS-DA_Latent_Features preparation method.  It does not augment
the data.  A scaler and PLS-DA model are fit on the supplied train CSV only,
then both train and test are transformed to PLS_1 ... PLS_k columns.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from data_prep_common import (
    COEF_COLUMNS,
    fit_plsda,
    launch_gui,
    load_pair,
    parse_int,
    pls_scores,
    write_latent_outputs,
)


def process(
    train_csv: str,
    test_csv: str,
    output_dir: str,
    label_column: str = "BinaryClass",
    n_components: int = 8,
) -> str:
    train, test = load_pair(train_csv, test_csv, "coef", label_column)
    x_train = train[COEF_COLUMNS].to_numpy(dtype=float)
    x_test = test[COEF_COLUMNS].to_numpy(dtype=float)
    y_train = train["BinaryClass"].to_numpy(dtype=int)
    bundle = fit_plsda(x_train, y_train, n_components)
    train_latent = pls_scores(bundle, x_train)
    test_latent = pls_scores(bundle, x_test)
    latent_columns = [
        f"PLS_{index}"
        for index in range(1, int(bundle["effective_components"]) + 1)
    ]
    train_output = pd.concat(
        [
            train[["Subject", "Group", "Class", "BinaryClass", "DataType"]].reset_index(
                drop=True
            ),
            pd.DataFrame(train_latent, columns=latent_columns),
        ],
        axis=1,
    )
    test_output = pd.concat(
        [
            test[["Subject", "Group", "Class", "BinaryClass", "DataType"]].reset_index(
                drop=True
            ),
            pd.DataFrame(test_latent, columns=latent_columns),
        ],
        axis=1,
    )
    root = write_latent_outputs(
        output_dir=output_dir,
        input_train=train_csv,
        input_test=test_csv,
        train=train,
        test=test,
        train_scores=train_output,
        test_scores=test_output,
        bundle=bundle,
        requested_components=n_components,
    )
    return f"plsda_latent_features completed. Output: {root}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="BinaryClass")
    parser.add_argument("--n-components", type=int, default=8)
    return parser


def gui() -> None:
    fields = [
        {"name": "train_csv", "label": "Train CSV", "browse": "file"},
        {"name": "test_csv", "label": "Test CSV", "browse": "file"},
        {"name": "output_dir", "label": "Output folder", "browse": "dir"},
        {"name": "label_column", "label": "Label column", "default": "BinaryClass"},
        {"name": "n_components", "label": "PLS components", "default": "8"},
    ]

    def run(values: dict[str, str]) -> str:
        return process(
            values["train_csv"],
            values["test_csv"],
            values["output_dir"],
            values["label_column"] or "BinaryClass",
            parse_int(values["n_components"] or "8", "PLS components", 1),
        )

    launch_gui(
        "Prepare PLS-DA latent features",
        fields,
        run,
        "Output contains metadata plus PLS_1 ... PLS_k. No synthetic rows "
        "are created.",
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
