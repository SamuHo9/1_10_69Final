"""Prepare PointNet XYZ data with same-class PLS-DA augmentation.

The supplied clouds are normalized independently, PLS-DA is fit on flattened
normalized XYZ values from train only, unique parent pairs are selected within each
class, and reconstructed synthetic clouds are normalized again.  PointNet
still receives XYZ values, not PLS scores.
"""

from __future__ import annotations

import argparse
import sys

from data_prep_common import (
    augment_plsda_same_class,
    load_pair,
    launch_gui,
    normalize_xyz_frame,
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
    normalize: str = "per_cloud",
    augmentation_size: str = "maximum",
) -> str:
    train, test = load_pair(train_csv, test_csv, "xyz", label_column)
    train = normalize_xyz_frame(train, normalize)
    test = normalize_xyz_frame(test, normalize)
    prepared, synthetic, pairs, bundle, info = augment_plsda_same_class(
        train,
        "xyz",
        seed=seed,
        requested_components=n_components,
        children_per_pair=children_per_pair,
        augmentation_size=augmentation_size,
        stream=True,
        protocol="pointnet_plsda_same_class",
        normalize_xyz_after_reconstruction=(normalize == "per_cloud"),
    )
    root = write_preparation_outputs(
        output_dir=output_dir,
        protocol="pointnet_plsda",
        kind="xyz",
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
            "model_input": "x_0,y_0,z_0 through x_1001,y_1001,z_1001",
            "pls_da_used_for_augmentation": True,
            "pls_da_applied_to_model_input": False,
            "xyz_normalization": normalize,
            "test_preprocessing": normalize,
            "seed": int(seed),
        },
    )
    return f"pointnet_plsda completed. Output: {root}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="BinaryClass")
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--children-per-pair", type=int, default=8)
    parser.add_argument(
        "--normalize",
        choices=("none", "per_cloud"),
        default="per_cloud",
    )
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
        {"name": "normalize", "label": "Normalization", "default": "per_cloud"},
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
            values["normalize"] or "per_cloud",
            values["augmentation_size"] or "maximum",
        )

    launch_gui(
        "Prepare pointnet_plsda",
        fields,
        run,
        "PLS-DA creates same-class synthetic XYZ clouds. The PointNet input "
        "remains 3 x 1002 XYZ values.",
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
            args.normalize,
            args.augmentation_size,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
