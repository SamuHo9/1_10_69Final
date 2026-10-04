#!/usr/bin/env python3
"""Create a fixed SPHARM template from training SPHARM outputs.

The input directory must contain matching pairs of::

    <subject>_SPHARM.vtk
    <subject>_SPHARM.coef

The script checks that every mesh has the same point order and topology,
averages the corresponding mesh coordinates and SPHARM coefficients, and
writes a new ``template_spharm_<side>.vtk/.coef`` pair plus a manifest.

This is deliberately a separate step from ``run_spharm_parallel.py``.  The
parallel runner consumes a template; it does not create one.  For a leakage-
controlled experiment, pass outputs made from the training subjects only.
Run this file with SlicerSALT's PythonSlicer because it uses VTK::

    PythonSlicer.exe create_spharm_template.py \
        --side left \
        --input_dir path/to/train/spharm_results \
        --output_dir path/to/generated_template

The safety check rejects an input path that does not contain ``train`` unless
``--allow-nontrain`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

def _bootstrap_slicer_python(import_error: ImportError) -> None:
    marker = "_SPHARM_TEMPLATE_RELAUNCHED"
    if os.environ.get(marker) == "1":
        raise SystemExit(
            "SlicerSALT Python could not import NumPy/VTK: {}".format(import_error)
        ) from import_error

    candidates = []
    configured = os.environ.get("SLICER_SALT_PYTHON")
    if configured:
        candidates.append(Path(configured))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates.append(program_files / "SlicerSALT 6.0.0" / "bin" / "PythonSlicer.exe")
    on_path = shutil.which("PythonSlicer.exe")
    if on_path:
        candidates.append(Path(on_path))
    python_slicer = next((candidate for candidate in candidates if candidate.is_file()), None)
    if python_slicer is None:
        raise SystemExit(
            "NumPy/VTK are required. Could not find SlicerSALT PythonSlicer.exe; "
            "set SLICER_SALT_PYTHON to its full path."
        ) from import_error

    script_path = Path(__file__).resolve()
    child_env = os.environ.copy()
    child_env[marker] = "1"
    print("[INFO] Relaunching with SlicerSALT Python:", python_slicer, flush=True)
    result = subprocess.run(
        [str(python_slicer), str(script_path), *sys.argv[1:]],
        cwd=str(script_path.parent),
        env=child_env,
        check=False,
    )
    raise SystemExit(result.returncode)


try:
    import numpy as np
    import vtk
except ImportError as exc:
    _bootstrap_slicer_python(exc)


REQUIRED_VTK_SUFFIX = "_SPHARM.vtk"
REQUIRED_COEF_SUFFIX = "_SPHARM.coef"
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COEF_TRIPLE_RE = re.compile(
    rf"\{{\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*\}}"
)
COEF_HEADER_RE = re.compile(r"\{\s*(\d+)\s*,")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_coef(path: Path) -> np.ndarray:
    """Read the ASCII SPHARM coefficient format used by SlicerSALT."""
    text = path.read_text(encoding="utf-8", errors="replace")
    header = COEF_HEADER_RE.match(text)
    if not header:
        raise ValueError(f"Invalid SPHARM coefficient header: {path}")
    expected = int(header.group(1))
    triples = [
        (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        for match in COEF_TRIPLE_RE.finditer(text[header.end() :])
    ]
    if len(triples) != expected:
        raise ValueError(
            f"Coefficient count mismatch in {path}: header={expected}, parsed={len(triples)}"
        )
    values = np.asarray(triples, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite coefficient found in {path}")
    return values


def write_coef(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected coefficient array with shape (N, 3), got {values.shape}")
    lines = [f"{{ {values.shape[0]},"]
    for index, (x, y, z) in enumerate(values):
        suffix = "}}" if index == values.shape[0] - 1 else "},"
        lines.append(f"{{{x:.6f}, {y:.6f}, {z:.6f}}}{suffix}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_polydata(path: Path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetPoints() is None:
        raise ValueError(f"Could not read VTK polydata: {path}")
    if poly.GetNumberOfPoints() < 10 or poly.GetNumberOfCells() == 0:
        raise ValueError(
            f"Invalid mesh {path}: points={poly.GetNumberOfPoints()}, cells={poly.GetNumberOfCells()}"
        )
    points = np.asarray(
        [poly.GetPoint(index) for index in range(poly.GetNumberOfPoints())], dtype=np.float64
    )
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite mesh coordinate found in {path}")
    return poly, points


def topology_hash(poly) -> str:
    """Hash cell types and point IDs to verify identical mesh topology/order."""
    digest = hashlib.sha256()
    ids = vtk.vtkIdList()
    for cell_index in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(cell_index)
        digest.update(str(poly.GetCellType(cell_index)).encode("ascii"))
        digest.update(b":")
        digest.update(str(cell.GetNumberOfPoints()).encode("ascii"))
        digest.update(b":")
        for point_index in range(cell.GetNumberOfPoints()):
            digest.update(str(cell.GetPointId(point_index)).encode("ascii"))
            digest.update(b",")
        digest.update(b";")
    return digest.hexdigest()


def subject_stem(path: Path) -> str:
    return path.name[: -len(REQUIRED_VTK_SUFFIX)]


def read_include_list(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(Path(line).name)
        if line.endswith(REQUIRED_VTK_SUFFIX):
            names.add(line[: -len(REQUIRED_VTK_SUFFIX)])
    if not names:
        raise ValueError(f"Include list is empty: {path}")
    return names


def discover_pairs(input_dir: Path, include_list: Path | None) -> list[tuple[Path, Path]]:
    allowed = read_include_list(include_list) if include_list else None
    pairs: list[tuple[Path, Path]] = []
    for vtk_path in sorted(input_dir.rglob(f"*{REQUIRED_VTK_SUFFIX}")):
        stem = subject_stem(vtk_path)
        if allowed is not None and vtk_path.name not in allowed and stem not in allowed:
            continue
        coef_path = vtk_path.with_name(f"{stem}{REQUIRED_COEF_SUFFIX}")
        if not coef_path.is_file():
            raise FileNotFoundError(f"Missing matching coefficient file for {vtk_path.name}: {coef_path}")
        pairs.append((vtk_path, coef_path))
    if not pairs:
        raise FileNotFoundError(
            f"No {REQUIRED_VTK_SUFFIX} files with matching {REQUIRED_COEF_SUFFIX} found in {input_dir}"
        )
    return pairs


def build_template(
    *,
    side: str,
    input_dir: Path,
    output_dir: Path,
    include_list: Path | None,
    allow_nontrain: bool,
    name: str,
    degree: int,
    subdivision: int,
    overwrite: bool,
) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not allow_nontrain and "train" not in {part.lower() for part in input_dir.parts}:
        raise ValueError(
            "Input path does not contain a 'train' directory. "
            "Use training-only SPHARM outputs or pass --allow-nontrain explicitly."
        )

    pairs = discover_pairs(input_dir, include_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtk_out = output_dir / f"{name}.vtk"
    coef_out = output_dir / f"{name}.coef"
    manifest_out = output_dir / f"{name}.manifest.json"
    if not overwrite and any(path.exists() for path in (vtk_out, coef_out, manifest_out)):
        raise FileExistsError(
            f"Output already exists in {output_dir}; use a new folder/name or pass --overwrite."
        )

    reference_poly = None
    point_arrays: list[np.ndarray] = []
    coef_arrays: list[np.ndarray] = []
    subjects = []
    expected_points = None
    expected_cells = None
    expected_topology = None
    expected_coef_count = None

    for vtk_path, coef_path in pairs:
        poly, points = read_polydata(vtk_path)
        mesh_topology = topology_hash(poly)
        coefficients = read_coef(coef_path)
        if reference_poly is None:
            reference_poly = poly
            expected_points = poly.GetNumberOfPoints()
            expected_cells = poly.GetNumberOfCells()
            expected_topology = mesh_topology
            expected_coef_count = coefficients.shape[0]
        else:
            if poly.GetNumberOfPoints() != expected_points:
                raise ValueError(
                    f"Point-count mismatch for {vtk_path.name}: "
                    f"expected {expected_points}, got {poly.GetNumberOfPoints()}"
                )
            if poly.GetNumberOfCells() != expected_cells or mesh_topology != expected_topology:
                raise ValueError(f"Topology mismatch for {vtk_path.name}; no template was written")
        if coefficients.shape[0] != expected_coef_count:
            raise ValueError(
                f"Coefficient-count mismatch for {coef_path.name}: "
                f"expected {expected_coef_count}, got {coefficients.shape[0]}"
            )
        point_arrays.append(points)
        coef_arrays.append(coefficients)
        subjects.append(
            {
                "subject": subject_stem(vtk_path),
                "vtk": str(vtk_path),
                "coef": str(coef_path),
                "vtk_sha256": sha256(vtk_path),
                "coef_sha256": sha256(coef_path),
            }
        )

    mean_points = np.mean(np.stack(point_arrays, axis=0), axis=0)
    mean_coefficients = np.mean(np.stack(coef_arrays, axis=0), axis=0)

    output_poly = vtk.vtkPolyData()
    output_poly.DeepCopy(reference_poly)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(mean_points.shape[0])
    for index, (x, y, z) in enumerate(mean_points):
        vtk_points.SetPoint(index, float(x), float(y), float(z))
    output_poly.SetPoints(vtk_points)
    # Subject-specific arrays must not leak into a template.
    output_poly.GetPointData().Initialize()
    output_poly.GetCellData().Initialize()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(vtk_out))
    writer.SetInputData(output_poly)
    writer.SetFileTypeToASCII()
    if writer.Write() != 1:
        raise RuntimeError(f"VTK writer failed: {vtk_out}")
    write_coef(coef_out, mean_coefficients)

    # Read back the generated artifacts before writing the manifest.
    check_poly, check_points = read_polydata(vtk_out)
    check_coef = read_coef(coef_out)
    if check_poly.GetNumberOfPoints() != expected_points or topology_hash(check_poly) != expected_topology:
        raise RuntimeError("Generated template failed topology validation")
    if check_coef.shape != mean_coefficients.shape or not np.allclose(check_coef, mean_coefficients, atol=5e-6):
        raise RuntimeError("Generated coefficient template failed read-back validation")

    manifest = {
        "schema": "spharm_template_build_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "side": side,
        "method": "coordinate_mean_and_coefficient_mean",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "template_vtk": str(vtk_out),
        "template_coef": str(coef_out),
        "n_subjects": len(subjects),
        "train_only_guard": not allow_nontrain,
        "allow_nontrain": allow_nontrain,
        "spharm_degree": degree,
        "subdivision_level": subdivision,
        "mesh_points": int(expected_points),
        "mesh_cells": int(expected_cells),
        "coefficient_count": int(expected_coef_count),
        "topology_sha256": expected_topology,
        "template_vtk_sha256": sha256(vtk_out),
        "template_coef_sha256": sha256(coef_out),
        "subjects": subjects,
        "notes": [
            "Template was calculated from the listed input meshes and coefficient files.",
            "The input directory and subject list must be training-only for an independent evaluation.",
            "This is a coordinate/coefficient mean; it does not refit ICP or rerun SPHARM.",
        ],
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SPHARM template from selected mesh/coefficient pairs")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="folder containing *_SPHARM.vtk and matching *_SPHARM.coef")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="folder where template files and manifest will be written")
    parser.add_argument("--include_list", type=Path, default=None,
                        help="optional text file with one VTK filename or subject stem per line")
    parser.add_argument("--name", default=None,
                        help="output base name; default template_spharm_<side>")
    parser.add_argument("--degree", type=int, default=12,
                        help="SPHARM degree recorded in the manifest")
    parser.add_argument("--subdivision", type=int, default=10,
                        help="subdivision level recorded in the manifest")
    parser.add_argument(
        "--allow-nontrain",
        "--all-files",
        dest="allow_nontrain",
        action="store_true",
        help="use all matching files; required when the input path is not a train directory",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    name = args.name or f"template_spharm_{args.side}"
    manifest = build_template(
        side=args.side,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        include_list=args.include_list,
        allow_nontrain=args.allow_nontrain,
        name=name,
        degree=args.degree,
        subdivision=args.subdivision,
        overwrite=args.overwrite,
    )
    print(f"Created: {manifest['template_vtk']}")
    print(f"Created: {manifest['template_coef']}")
    print(f"Manifest: {Path(manifest['output_dir']) / (name + '.manifest.json')}")
    print(f"Subjects: {manifest['n_subjects']} | points: {manifest['mesh_points']} | cells: {manifest['mesh_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
