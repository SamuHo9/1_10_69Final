"""Shared utilities for the five GUI/CLI data-preparation entry points.

The utilities intentionally implement a fixed train/test workflow:

* only the supplied training CSV is used to fit scalers and PLS-DA;
* synthetic rows, when requested, are made from same-class training rows;
* the test CSV is never used to select pairs or create synthetic rows;
* every run writes a manifest and provenance tables.

For grouped cross-validation, the same functions should be called inside each
training fold rather than pre-augmenting the complete training cohort once.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler


META_COLUMNS = ["Subject", "Group", "Class", "BinaryClass", "DataType"]
COEF_COLUMNS = [f"Coef_{index}" for index in range(1, 508)]
XYZ_COLUMNS = [
    f"{axis}_{point}"
    for point in range(1002)
    for axis in ("x", "y", "z")
]
SYNTH_PROVENANCE_COLUMNS = [
    "PairID",
    "ParentSubject1",
    "ParentSubject2",
    "Alpha",
    "AugmentationMethod",
]


def as_path(value: str | Path, name: str, must_exist: bool = False) -> Path:
    """Resolve a user-supplied path and optionally require it to exist."""

    if value is None or not str(value).strip():
        raise ValueError(f"{name} is required")
    path = Path(str(value).strip()).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def json_ready(value: Any) -> Any:
    """Convert common NumPy/Path values to JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_group_id(subject: Any) -> str:
    """Extract the stable subject/patient token used for overlap checks."""

    value = str(subject).strip().split("/", 1)[-1]
    match = re.search(
        r"(?i)(?:^|[_-])"
        r"(sub[-_]?[a-z0-9]+|subject[-_]?[a-z0-9]+|case[-_]?[a-z0-9]+)"
        r"(?:_|$)",
        value,
    )
    if match:
        return match.group(1).replace("_", "-").lower()
    return re.sub(r"(?i)[^a-z0-9]+", "-", value).strip("-").lower()


def _binary_labels(series: pd.Series, name: str) -> pd.Series:
    """Convert a numeric or common Healthy/TLE label column to 0/1."""

    if series.isna().any():
        raise ValueError(f"{name} contains missing labels")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()).issubset({0, 1}):
        return numeric.astype(np.int64)

    normalized = series.astype(str).str.strip().str.lower()
    healthy = {
        "0",
        "healthy",
        "normal",
        "control",
        "hc",
        "non-tle",
        "non_tle",
    }
    tle = {"1", "tle", "epilepsy", "patient", "case"}
    unknown = set(normalized.unique()) - healthy - tle
    if unknown:
        raise ValueError(
            f"{name} must contain 0/1 or Healthy/TLE labels; unknown values: "
            f"{sorted(unknown)}"
        )
    return normalized.map(lambda value: 0 if value in healthy else 1).astype(np.int64)


def feature_columns(kind: str) -> list[str]:
    if kind == "coef":
        return list(COEF_COLUMNS)
    if kind == "xyz":
        return list(XYZ_COLUMNS)
    raise ValueError(f"Unknown data kind: {kind}")


def _normalize_metadata(
    frame: pd.DataFrame,
    label_column: str,
    name: str,
) -> pd.DataFrame:
    result = frame.copy()
    if "Subject" not in result.columns:
        raise ValueError(f"{name} must contain a Subject column")
    if result["Subject"].isna().any():
        raise ValueError(f"{name} contains a missing Subject")
    if result["Subject"].astype(str).duplicated().any():
        raise ValueError(f"{name} contains duplicate Subject values")

    if label_column not in result.columns:
        if label_column != "BinaryClass":
            raise ValueError(f"{name} does not contain label column {label_column!r}")
        if "BinaryClass" not in result.columns and "Class" not in result.columns:
            raise ValueError(f"{name} must contain BinaryClass or Class")

    source_column = label_column if label_column in result.columns else "Class"
    binary = _binary_labels(result[source_column], f"{name}.{source_column}")
    if "BinaryClass" in result.columns:
        declared = _binary_labels(result["BinaryClass"], f"{name}.BinaryClass")
        if not np.array_equal(binary.to_numpy(), declared.to_numpy()):
            raise ValueError(f"{name} has disagreement between label columns")

    result["BinaryClass"] = binary
    result["Class"] = binary.astype(np.int64)
    if "Group" not in result.columns:
        result["Group"] = binary.map({0: "Healthy", 1: "TLE"})
    result["Group"] = result["Group"].fillna("").astype(str)
    if "DataType" not in result.columns:
        result["DataType"] = "Original"
    result["DataType"] = result["DataType"].fillna("Original").astype(str)

    datatype_bad = ~result["DataType"].str.lower().isin({"original", "real"})
    subject_bad = result["Subject"].astype(str).str.contains(
        r"aug|synthetic|synth|interp",
        case=False,
        regex=True,
    )
    if datatype_bad.any() or subject_bad.any():
        raise ValueError(
            f"{name} contains pre-augmented rows. Supply original train/test CSVs."
        )

    return result


def validate_frame(frame: pd.DataFrame, kind: str, name: str) -> pd.DataFrame:
    expected = feature_columns(kind)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing {len(missing)} {kind} features; "
            f"first missing columns: {missing[:5]}"
        )
    values = frame[expected].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite feature values")
    if frame["BinaryClass"].nunique() != 2:
        raise ValueError(f"{name} must contain both classes 0 and 1")
    return frame[META_COLUMNS + expected].reset_index(drop=True)


def load_pair(
    train_csv: str | Path,
    test_csv: str | Path,
    kind: str,
    label_column: str = "BinaryClass",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate one original train/test CSV pair."""

    train_path = as_path(train_csv, "train CSV", must_exist=True)
    test_path = as_path(test_csv, "test CSV", must_exist=True)
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    train = _normalize_metadata(train, label_column, "train")
    test = _normalize_metadata(test, label_column, "test")
    train = validate_frame(train, kind, "train")
    test = validate_frame(test, kind, "test")

    train_groups = {patient_group_id(value) for value in train["Subject"]}
    test_groups = {patient_group_id(value) for value in test["Subject"]}
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise ValueError(f"Patient overlap between train and test: {overlap[:10]}")
    return train, test


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    if isinstance(frame, PreparedBatches):
        return dict(frame.final_counts)
    counts = frame["BinaryClass"].value_counts().to_dict()
    return {str(label): int(counts.get(label, 0)) for label in (0, 1)}


def normalize_xyz_frame(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Apply the PointNet per-cloud normalization when requested."""

    if mode not in {"none", "per_cloud"}:
        raise ValueError("XYZ normalization must be 'none' or 'per_cloud'")
    result = frame.copy()
    if mode == "none":
        return result
    clouds = result[XYZ_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 1002, 3)
    mean = clouds.mean(axis=1, keepdims=True)
    std = clouds.std(axis=1, keepdims=True)
    normalized = (clouds - mean) / (std + 1e-8)
    result.loc[:, XYZ_COLUMNS] = normalized.reshape(-1, len(XYZ_COLUMNS))
    return result


def _one_hot(labels: np.ndarray) -> np.ndarray:
    output = np.zeros((len(labels), 2), dtype=np.float64)
    output[np.arange(len(labels)), labels.astype(int)] = 1.0
    return output


def fit_plsda(
    x: np.ndarray,
    y: np.ndarray,
    requested_components: int,
) -> dict[str, Any]:
    """Fit the scaler and PLS-DA model using only the supplied rows."""

    if requested_components < 1:
        raise ValueError("n_components must be at least 1")
    if len(np.unique(y)) != 2:
        raise ValueError("PLS-DA requires both classes in the training rows")
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    effective = min(int(requested_components), x.shape[1], len(y) - 1)
    if effective < 1:
        raise ValueError("Not enough rows/features for PLS-DA")
    model = PLSRegression(
        n_components=effective,
        scale=False,
        max_iter=1000,
    )
    model.fit(scaled, _one_hot(y))
    return {
        "scaler": scaler,
        "pls": model,
        "requested_components": int(requested_components),
        "effective_components": int(effective),
    }


def pls_scores(bundle: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return np.asarray(
        bundle["pls"].transform(bundle["scaler"].transform(x)),
        dtype=np.float64,
    )


def _same_class_pairs(
    indices: Sequence[int],
    space: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """All unordered distinct-parent pairs, nearest first in score space."""
    indices = np.asarray(indices, dtype=int)
    first, second = np.triu_indices(len(indices), k=1)
    pairs = np.column_stack((indices[first], indices[second]))
    rng.shuffle(pairs)  # Seeded tie-breaking; each unordered pair occurs once.
    distances = np.linalg.norm(space[pairs[:, 0]] - space[pairs[:, 1]], axis=1)
    return [tuple(map(int, pair)) for pair in pairs[np.argsort(distances, kind="stable")]]


def augmentation_plan(frame: pd.DataFrame, children_per_pair: int = 8,
                      augmentation_size: str = "maximum") -> dict[str, Any]:
    """Largest equal class size with original rows and at most k children/pair."""
    if children_per_pair < 1:
        raise ValueError("children_per_pair must be positive")
    if augmentation_size not in {"maximum", "balance_only"}:
        raise ValueError("augmentation_size must be maximum or balance_only")
    if not frame["Subject"].is_unique:
        raise ValueError("Distinct original Subject values are required")
    if set(frame["BinaryClass"].unique()) != {0, 1}:
        raise ValueError("Augmentation requires both classes 0 and 1")
    counts = class_counts(frame)
    available = {key: n * (n - 1) // 2 for key, n in counts.items()}
    capacities = {key: counts[key] + children_per_pair * available[key] for key in counts}
    target = min(capacities.values()) if augmentation_size == "maximum" else max(counts.values())
    if target < max(counts.values()) or target > min(capacities.values()):
        raise ValueError("Cannot balance while retaining all originals: too few unique "
                         "same-class pairs for this children_per_pair setting")
    return {
        "augmentation_size": augmentation_size,
        "children_per_pair": int(children_per_pair),
        "original_class_counts": counts,
        "available_unique_pairs": available,
        "maximum_class_capacities": capacities,
        "target_per_class": int(target),
        "required_synthetic_per_class": {key: target - n for key, n in counts.items()},
        "final_class_counts": {key: int(target) for key in counts},
        "unique_unordered_pairs": True,
        "self_pairing": False,
        "parent_reuse_across_different_pairs": True,
    }


class SyntheticBatches:
    """Repeatable synthetic batches; wide datasets need not reside in RAM."""
    def __init__(self, columns, size, factory):
        self.columns = list(columns)
        self.size = int(size)
        self.factory = factory

    def __len__(self):
        return self.size

    def iter_chunks(self):
        return self.factory()


class PreparedBatches:
    def __init__(self, original, synthetic, final_counts):
        self.original = original
        self.synthetic = synthetic
        self.final_counts = final_counts

    def __len__(self):
        return len(self.original) + len(self.synthetic)


def _pair_schedule(frame, plan, rng, protocol, scores=None):
    """Use each unordered pair once, truncating only the final child group."""
    y = frame["BinaryClass"].to_numpy(dtype=int)
    rows = []
    for label in (0, 1):
        remaining = plan["required_synthetic_per_class"][str(label)]
        if not remaining:
            continue
        indices = np.flatnonzero(y == label)
        if scores is None:
            a, b = np.triu_indices(len(indices), k=1)
            pairs = np.column_stack((indices[a], indices[b]))
            rng.shuffle(pairs)
        else:
            pairs = _same_class_pairs(indices, scores, rng)
        for pair_index, (first, second) in enumerate(pairs):
            if not remaining:
                break
            n = min(plan["children_per_pair"], remaining)
            row = {"PairID": f"{protocol}_class{label}_pair{pair_index:06d}",
                   "Class": label,
                   "ParentSubject1": str(frame.iloc[first]["Subject"]),
                   "ParentSubject2": str(frame.iloc[second]["Subject"]),
                   "Method": protocol, "Children": n,
                   "FirstIndex": int(first), "SecondIndex": int(second)}
            if scores is not None:
                row["PLSScoreDistance"] = float(np.linalg.norm(scores[first] - scores[second]))
            rows.append(row)
            remaining -= n
        assert remaining == 0
    return rows


def _batch_factory(frame, columns, schedule, value_factory, seed, protocol):
    def generate():
        rng = np.random.default_rng(seed)
        metadata, values = [], []
        total = 0
        for pair in schedule:
            first, second = pair["FirstIndex"], pair["SecondIndex"]
            alpha, block = value_factory(first, second, pair["Children"], rng)
            if not np.isfinite(block).all():
                raise ValueError("Augmentation produced non-finite features")
            values.append(block)
            for weight in alpha:
                metadata.append(_base_synthetic_record(
                    frame, first, second, pair["Class"],
                    f"aug_{protocol}_class{pair['Class']}_child{total:07d}",
                    pair["PairID"], float(weight), protocol, [], []))
                total += 1
            if len(metadata) >= 256:
                yield pd.concat([pd.DataFrame(metadata),
                                 pd.DataFrame(np.vstack(values), columns=columns)], axis=1)[
                                     list(frame.columns) + SYNTH_PROVENANCE_COLUMNS]
                metadata, values = [], []
        if metadata:
            yield pd.concat([pd.DataFrame(metadata),
                             pd.DataFrame(np.vstack(values), columns=columns)], axis=1)[
                                 list(frame.columns) + SYNTH_PROVENANCE_COLUMNS]
    return generate


def _augmentation_result(frame, columns, plan, schedule, factory, info, stream):
    size = sum(plan["required_synthetic_per_class"].values())
    synthetic = SyntheticBatches(list(frame.columns) + SYNTH_PROVENANCE_COLUMNS, size, factory)
    pairs = pd.DataFrame(schedule, columns=["PairID", "Class", "ParentSubject1",
                         "ParentSubject2", "Method", "Children", "FirstIndex", "SecondIndex"]
                         + (["PLSScoreDistance"] if schedule and "PLSScoreDistance" in schedule[0] else []))
    pairs = pairs.drop(columns=["FirstIndex", "SecondIndex"])
    info.update(plan)
    info.update(original_train_n=len(frame), synthetic_train_n=size,
                final_train_n=len(frame) + size,
                used_unique_pairs={str(label): int((pairs["Class"] == label).sum()) for label in (0, 1)})
    if stream:
        prepared = PreparedBatches(frame, synthetic, plan["final_class_counts"])
    else:
        chunks = list(synthetic.iter_chunks())
        synthetic = pd.concat(chunks, ignore_index=True) if chunks else _empty_synthetic_frame(frame)
        prepared = pd.concat([frame, synthetic[frame.columns]], ignore_index=True) if size else frame.copy()
    return prepared, synthetic, pairs, info


def _base_synthetic_record(
    frame: pd.DataFrame,
    first: int,
    second: int,
    label: int,
    subject: str,
    pair_id: str,
    alpha: float,
    method: str,
    values: np.ndarray,
    columns: Sequence[str],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "Subject": subject,
        "Group": str(frame.iloc[first]["Group"]),
        "Class": int(label),
        "BinaryClass": int(label),
        "DataType": "Synthetic",
        "PairID": pair_id,
        "ParentSubject1": str(frame.iloc[first]["Subject"]),
        "ParentSubject2": str(frame.iloc[second]["Subject"]),
        "Alpha": float(alpha),
        "AugmentationMethod": method,
    }
    record.update({column: float(value) for column, value in zip(columns, values)})
    return record


def _empty_synthetic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=list(frame.columns) + SYNTH_PROVENANCE_COLUMNS)


def augment_balanced_jitter(
    frame: pd.DataFrame,
    kind: str,
    seed: int = 42,
    noise_scale: float = 0.02,
    children_per_pair: int = 8,
    protocol: str = "balanced_jitter",
    augmentation_size: str = "maximum",
    stream: bool = False,
):
    """Maximize balanced same-class mixup using unique unordered pairs."""
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    plan = augmentation_plan(frame, children_per_pair, augmentation_size)
    columns = feature_columns(kind)
    x = frame[columns].to_numpy(dtype=np.float64)
    feature_std = np.std(x, axis=0)
    feature_std = np.where(feature_std > 1e-8, feature_std, 1.0)
    schedule = _pair_schedule(frame, plan, np.random.default_rng(seed), protocol)

    def values(first, second, n, rng):
        alpha = rng.uniform(0.25, 0.75, size=n)
        block = (alpha[:, None] * x[first] + (1.0 - alpha[:, None]) * x[second]
                 + rng.normal(0.0, noise_scale * feature_std, size=(n, x.shape[1])))
        return alpha, block

    factory = _batch_factory(frame, columns, schedule, values, seed, protocol)
    info = {"augmentation_protocol": protocol,
            "augmentation_kind": "same_class_mixup_plus_jitter",
            "same_class_pairing": True, "noise_scale": float(noise_scale)}
    return _augmentation_result(frame, columns, plan, schedule, factory, info, stream)


def augment_plsda_same_class(
    frame: pd.DataFrame,
    kind: str,
    seed: int = 42,
    requested_components: int = 8,
    children_per_pair: int = 8,
    protocol: str = "fold-local-plsda-augmentation",
    normalize_xyz_after_reconstruction: bool = False,
    augmentation_size: str = "maximum",
    stream: bool = False,
):
    """Use unique same-class pairs in PLS space for maximum equal class sizes."""
    plan = augmentation_plan(frame, children_per_pair, augmentation_size)
    columns = feature_columns(kind)
    x = frame[columns].to_numpy(dtype=np.float64)
    y = frame["BinaryClass"].to_numpy(dtype=np.int64)
    bundle = fit_plsda(x, y, requested_components)
    scores = pls_scores(bundle, x)
    schedule = _pair_schedule(frame, plan, np.random.default_rng(seed), protocol, scores)

    def values(first, second, n, rng):
        alpha = 0.1 + 0.8 * (np.arange(n) + 0.5) / children_per_pair
        new_scores = (1.0 - alpha[:, None]) * scores[first] + alpha[:, None] * scores[second]
        block = bundle["scaler"].inverse_transform(bundle["pls"].inverse_transform(new_scores))
        if kind == "xyz" and normalize_xyz_after_reconstruction:
            clouds = block.reshape(-1, 1002, 3)
            clouds = ((clouds - clouds.mean(axis=1, keepdims=True))
                      / (clouds.std(axis=1, keepdims=True) + 1e-8))
            block = clouds.reshape(-1, len(columns))
        return alpha, block

    factory = _batch_factory(frame, columns, schedule, values, seed, protocol)
    info = {"augmentation_protocol": protocol,
            "augmentation_kind": "same_class_plsda_score_interpolation",
            "same_class_pairing": True,
            "pls_fit_scope": "supplied_training_csv_only",
            "pls_components_requested": int(requested_components),
            "pls_components_effective": int(bundle["effective_components"]),
            "normalize_xyz_after_reconstruction": bool(normalize_xyz_after_reconstruction)}
    prepared, synthetic, pairs, info = _augmentation_result(
        frame, columns, plan, schedule, factory, info, stream)
    return prepared, synthetic, pairs, bundle, info


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _scaler_stats(scaler: StandardScaler, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Feature": list(columns),
            "Mean": np.asarray(scaler.mean_, dtype=float),
            "Scale": np.asarray(scaler.scale_, dtype=float),
        }
    )


def write_preparation_outputs(
    output_dir: str | Path,
    protocol: str,
    kind: str,
    input_train: str | Path,
    input_test: str | Path,
    original_train: pd.DataFrame,
    prepared_train: pd.DataFrame,
    prepared_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    pairs: pd.DataFrame,
    info: dict[str, Any],
    scaler: StandardScaler | None = None,
    pls_bundle: dict[str, Any] | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    """Write the common CSV, manifest, provenance, and model artifacts."""

    root = as_path(output_dir, "output directory")
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(prepared_test, root / "test_prepared.csv")
    if isinstance(prepared_train, PreparedBatches):
        # Write both CSVs in bounded batches without materializing the full dataset.
        _write_csv(prepared_train.original, root / "train_prepared.csv")
        _write_csv(pd.DataFrame(columns=synthetic.columns), root / "synthetic_rows.csv")
        written = 0
        with (root / "train_prepared.csv").open("a", encoding="utf-8", newline="") as train_stream, \
                (root / "synthetic_rows.csv").open("a", encoding="utf-8", newline="") as synth_stream:
            for chunk in synthetic.iter_chunks():
                chunk[prepared_train.original.columns].to_csv(train_stream, index=False, header=False)
                chunk.to_csv(synth_stream, index=False, header=False)
                written += len(chunk)
        if written != len(synthetic):
            raise ValueError("Synthetic output count does not match augmentation plan")
    else:
        _write_csv(prepared_train, root / "train_prepared.csv")
        _write_csv(synthetic, root / "synthetic_rows.csv")
    _write_csv(pairs, root / "pair_manifest.csv")

    columns = feature_columns(kind)
    if scaler is not None:
        _write_csv(_scaler_stats(scaler, columns), root / "scaler_stats.csv")
    if pls_bundle is not None:
        save_pickle(root / "plsda_model.pkl", pls_bundle)

    manifest: dict[str, Any] = {
        "schema": "data_preparation_manifest_v1",
        "protocol": protocol,
        "kind": kind,
        "input_train": str(as_path(input_train, "train CSV")),
        "input_test": str(as_path(input_test, "test CSV")),
        "output_dir": str(root),
        "features": columns,
        "original_train_rows": int(len(original_train)),
        "prepared_train_rows": int(len(prepared_train)),
        "test_rows": int(len(prepared_test)),
        "synthetic_rows": int(len(synthetic)),
        "original_train_class_counts": class_counts(original_train),
        "prepared_train_class_counts": class_counts(prepared_train),
        "test_class_counts": class_counts(prepared_test),
        "augmentation_applied_to_test": False,
        "augmentation_applied_to_validation": False,
        "same_class_pairing": bool(info.get("same_class_pairing", False)),
        "cross_class_pairs": False,
        "test_sha256": sha256_file(as_path(input_test, "test CSV", True)),
        "info": info,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    write_json(root / "data_manifest.json", manifest)
    return root


def write_latent_outputs(
    output_dir: str | Path,
    input_train: str | Path,
    input_test: str | Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    bundle: dict[str, Any],
    requested_components: int,
) -> Path:
    root = as_path(output_dir, "output directory")
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(train_scores, root / "train_plsda_latent_features.csv")
    _write_csv(test_scores, root / "test_plsda_latent_features.csv")
    save_pickle(root / "scaler.pkl", bundle["scaler"])
    save_pickle(root / "plsda_model.pkl", bundle["pls"])
    manifest = {
        "schema": "plsda_latent_features_manifest_v1",
        "protocol": "plsda_latent_features",
        "input_feature_kind": "coef",
        "output_feature_prefix": "PLS_",
        "input_train": str(as_path(input_train, "train CSV")),
        "input_test": str(as_path(input_test, "test CSV")),
        "output_dir": str(root),
        "requested_components": int(requested_components),
        "effective_components": int(bundle["effective_components"]),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_class_counts": class_counts(train),
        "test_class_counts": class_counts(test),
        "pls_fit_scope": "supplied_training_csv_only",
        "synthetic_rows": 0,
        "augmentation_applied_to_test": False,
        "test_sha256": sha256_file(as_path(input_test, "test CSV", True)),
    }
    write_json(root / "data_manifest.json", manifest)
    return root


def _launch_qt_gui(
    title: str,
    fields: Sequence[dict[str, Any]],
    runner: Callable[[dict[str, str]], str],
    notes: str = "",
) -> None:
    """Qt fallback for SlicerSALT Python, whose runtime has no tkinter."""

    try:
        import qt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "GUI requires tkinter or SlicerSALT Qt. Run this file with "
            "PythonSlicer.exe."
        ) from exc

    if not hasattr(qt, "QDialog"):
        raise RuntimeError(
            "The standalone PythonSlicer runtime does not expose Qt widgets. "
            "For GUI use SlicerSALT.exe or a regular Python with tkinter; "
            "PythonSlicer.exe remains suitable for command-line mode."
        )

    application = None
    application_class = getattr(qt, "QApplication", None)
    if application_class is not None:
        instance = getattr(application_class, "instance", None)
        if instance is not None:
            application = instance()
        if application is None:
            application = application_class([])
    else:
        try:
            import slicer

            application = getattr(slicer, "app", None)
        except ImportError:
            application = None

    dialog = qt.QDialog(None)
    dialog.setWindowTitle(title)
    dialog.resize(900, 680)
    layout = qt.QVBoxLayout(dialog)
    form = qt.QFormLayout()
    layout.addLayout(form)
    widgets: dict[str, Any] = {}

    def selected_file(parent: Any, key: str) -> str:
        value = qt.QFileDialog.getOpenFileName(
            parent,
            f"Select {key}",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if isinstance(value, (tuple, list)):
            return str(value[0]) if value else ""
        return str(value or "")

    def selected_directory(parent: Any, key: str) -> str:
        return str(
            qt.QFileDialog.getExistingDirectory(parent, f"Select {key}") or ""
        )

    for spec in fields:
        name = str(spec["name"])
        edit = qt.QLineEdit(str(spec.get("default", "")))
        widgets[name] = edit
        browse_kind = spec.get("browse")
        if browse_kind:
            button = qt.QPushButton("Browse")
            row = qt.QHBoxLayout()
            row.addWidget(edit)
            row.addWidget(button)

            def choose(
                _checked: bool = False,
                key: str = name,
                kind: str = str(browse_kind),
                target: Any = edit,
            ) -> None:
                value = (
                    selected_file(dialog, key)
                    if kind == "file"
                    else selected_directory(dialog, key)
                )
                if value:
                    target.setText(value)

            button.clicked.connect(choose)
            form.addRow(str(spec["label"]), row)
        else:
            form.addRow(str(spec["label"]), edit)

    if notes:
        note = qt.QLabel(notes)
        note.setWordWrap(True)
        layout.addWidget(note)

    log = qt.QPlainTextEdit()
    log.setReadOnly(True)
    layout.addWidget(log)
    buttons = qt.QHBoxLayout()
    buttons.addStretch(1)
    close_button = qt.QPushButton("Close")
    run_button = qt.QPushButton("Run")
    run_button.setDefault(True)
    buttons.addWidget(close_button)
    buttons.addWidget(run_button)
    layout.addLayout(buttons)

    def execute() -> None:
        values = {
            key: str(widget.text()).strip()
            for key, widget in widgets.items()
        }
        try:
            run_button.setEnabled(False)
            message = runner(values)
            log.appendPlainText(message)
            qt.QMessageBox.information(dialog, "Completed", message)
        except Exception as exc:  # pragma: no cover - exercised through GUI
            message = f"{type(exc).__name__}: {exc}"
            log.appendPlainText(message)
            qt.QMessageBox.critical(dialog, "Failed", message)
        finally:
            run_button.setEnabled(True)

    close_button.clicked.connect(dialog.reject)
    run_button.clicked.connect(execute)
    exec_method = getattr(dialog, "exec_", None) or getattr(dialog, "exec")
    exec_method()
    del application


def launch_gui(
    title: str,
    fields: Sequence[dict[str, Any]],
    runner: Callable[[dict[str, str]], str],
    notes: str = "",
) -> None:
    """Launch a small Tkinter form shared by all entry points."""

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from tkinter.scrolledtext import ScrolledText
    except Exception:
        _launch_qt_gui(title, fields, runner, notes)
        return

    root = tk.Tk()
    root.title(title)
    root.geometry("900x680")
    root.minsize(760, 560)
    variables: dict[str, tk.StringVar] = {}

    outer = ttk.Frame(root, padding=12)
    outer.pack(fill="both", expand=True)
    outer.columnconfigure(1, weight=1)

    for row, spec in enumerate(fields):
        name = str(spec["name"])
        variables[name] = tk.StringVar(value=str(spec.get("default", "")))
        ttk.Label(outer, text=str(spec["label"])).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        entry = ttk.Entry(outer, textvariable=variables[name])
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        browse = spec.get("browse")
        if browse:
            def choose(
                key: str = name,
                browse_kind: str = str(browse),
            ) -> None:
                if browse_kind == "file":
                    selected = filedialog.askopenfilename(
                        title=f"Select {key}",
                        filetypes=[
                            ("CSV files", "*.csv"),
                            ("All files", "*.*"),
                        ],
                    )
                else:
                    selected = filedialog.askdirectory(title=f"Select {key}")
                if selected:
                    variables[key].set(selected)

            ttk.Button(
                outer,
                text="Browse",
                command=choose,
            ).grid(row=row, column=2, sticky="e", padx=(8, 0), pady=4)

    note_row = len(fields)
    if notes:
        ttk.Label(
            outer,
            text=notes,
            foreground="#444444",
            justify="left",
            wraplength=820,
        ).grid(row=note_row, column=0, columnspan=3, sticky="w", pady=(12, 8))
        note_row += 1

    log = ScrolledText(outer, height=12, wrap="word")
    log.grid(row=note_row, column=0, columnspan=3, sticky="nsew", pady=(8, 8))
    outer.rowconfigure(note_row, weight=1)

    def execute() -> None:
        values = {key: variable.get().strip() for key, variable in variables.items()}
        try:
            run_button.configure(state="disabled")
            root.update_idletasks()
            message = runner(values)
            log.insert("end", message + "\n")
            log.see("end")
            messagebox.showinfo("Completed", message, parent=root)
        except Exception as exc:  # pragma: no cover - exercised through GUI
            message = f"{type(exc).__name__}: {exc}"
            log.insert("end", message + "\n")
            log.see("end")
            messagebox.showerror("Failed", message, parent=root)
        finally:
            run_button.configure(state="normal")

    run_button = ttk.Button(outer, text="Run", command=execute)
    run_button.grid(row=note_row + 1, column=0, columnspan=3, pady=(4, 0))
    root.mainloop()


def parse_int(value: str, name: str, minimum: int = 0) -> int:
    try:
        number = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def parse_float(value: str, name: str, minimum: float = 0.0) -> float:
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number
