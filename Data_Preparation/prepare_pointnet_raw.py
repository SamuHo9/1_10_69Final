"""Prepare raw XYZ point-cloud data for PointNet.

The script validates the 1,002-point XYZ schema.  Per-cloud normalization is
the default because it matches the existing PointNet preprocessing.  Optional
balanced_jitter augmentation is same-class only and is applied to train only.
Run without command-line arguments to open the GUI.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from data_prep_common import (
    SYNTH_PROVENANCE_COLUMNS,
    augment_balanced_jitter,
    load_pair,
    launch_gui,
    normalize_xyz_frame,
    parse_float,
    parse_int,
    write_preparation_outputs,
)


def process(
    train_csv: str,
    test_csv: str,
    output_dir: str,
    label_column: str = "BinaryClass",
    augmentation: str = "none",
    normalize: str = "per_cloud",
    seed: int = 42,
    noise_scale: float = 0.02,
    children_per_pair: int = 8,
    augmentation_size: str = "maximum",
) -> str:
    train, test = load_pair(train_csv, test_csv, "xyz", label_column)
    train = normalize_xyz_frame(train, normalize)
    test = normalize_xyz_frame(test, normalize)

    if augmentation == "none":
        prepared = train.copy()
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
            "xyz",
            seed=seed,
            noise_scale=noise_scale,
            children_per_pair=children_per_pair,
            augmentation_size=augmentation_size,
            stream=True,
            protocol="pointnet_raw_balanced_jitter",
        )
    else:
        raise ValueError("augmentation must be none or balanced_jitter")

    root = write_preparation_outputs(
        output_dir=output_dir,
        protocol="pointnet_raw",
        kind="xyz",
        input_train=train_csv,
        input_test=test_csv,
        original_train=train,
        prepared_train=prepared,
        prepared_test=test,
        synthetic=synthetic,
        pairs=pairs,
        info=info,
        extra_manifest={
            "model_input": "x_0,y_0,z_0 through x_1001,y_1001,z_1001",
            "pls_da_used": False,
            "xyz_normalization": normalize,
            "test_preprocessing": normalize,
            "seed": int(seed),
        },
    )
    return f"pointnet_raw completed. Output: {root}"


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
    parser.add_argument(
        "--normalize",
        choices=("none", "per_cloud"),
        default="per_cloud",
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
        {"name": "normalize", "label": "Normalization", "default": "per_cloud"},
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
            values["normalize"] or "per_cloud",
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
        "Prepare pointnet_raw",
        fields,
        run,
        "XYZ columns must be x_0,y_0,z_0 through x_1001,y_1001,z_1001. "
        "Normalization is per point cloud and uses no cross-subject statistics.",
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
            args.augmentation,
            args.normalize,
            args.seed,
            args.noise_scale,
            args.children_per_pair,
            args.augmentation_size,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
