#!/usr/bin/env python3
"""Run one fixed-reference ICP step through the normal Python/Slicer bootstrap.

From the project root, the default inputs are:
  ../merged_ds005602_ds004469_spharm_ready/left_hippocampus
  ../merged_ds005602_ds004469_spharm_ready/right_hippocampus
  (with a fallback to ../merged_ds005602_ds004469)

Examples:
  python ICP/run_icp_with_reference.py --side left --check-only
  python ICP/run_icp_with_reference.py --side left
  python ICP/run_icp_with_reference.py --side right
  python ICP/run_icp_with_reference.py --gui
  python ICP/run_icp_with_reference.py              # opens the same GUI
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ICP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ICP_DIR.parent
ICP_SCRIPT = ICP_DIR / "ICP.py"
_INPUT_ROOT_CANDIDATES = (
    PROJECT_ROOT.parent / "merged_ds005602_ds004469_spharm_ready",
    PROJECT_ROOT.parent / "merged_ds005602_ds004469",
)
DEFAULT_INPUT_ROOT = next(
    (candidate for candidate in _INPUT_ROOT_CANDIDATES if candidate.is_dir()),
    _INPUT_ROOT_CANDIDATES[0],
)
_REFERENCE_ROOT_CANDIDATES = (
    ICP_DIR / "Referen_ICP" / "legacy_groupwise_all_381_v1",
    ICP_DIR / "references" / "legacy_groupwise_all_381_v1",
)
DEFAULT_REFERENCE_ROOT = next(
    (candidate for candidate in _REFERENCE_ROOT_CANDIDATES if candidate.is_dir()),
    _REFERENCE_ROOT_CANDIDATES[0],
)
DEFAULT_SLICER_EXE = Path(r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe")
SUPPORTED_REFERENCE_VERSIONS = {
    "fixed-legacy-groupwise-reference-v1",
    "fixed-training-reference-v1",
}


def default_reference_root(side):
    """Find the newest complete reference bundle for the requested side."""
    timestamped_root = ICP_DIR / "Referen_ICP"
    timestamped = []
    if timestamped_root.is_dir():
        for candidate in timestamped_root.glob("ICP_reference_*"):
            template = candidate / side / "mean_shape.ply"
            contract = Path(str(template) + ".json")
            if template.is_file() and contract.is_file():
                timestamped.append(candidate)
    if timestamped:
        return sorted(timestamped, key=lambda path: (path.name, path.stat().st_mtime), reverse=True)[0]
    for candidate in _REFERENCE_ROOT_CANDIDATES:
        if (candidate / side / "mean_shape.ply").is_file():
            return candidate
    return DEFAULT_REFERENCE_ROOT


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supported_volume(path):
    name = path.name.lower()
    return name.endswith((".nii.gz", ".nii", ".hdr", ".nrrd"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Align one hemisphere to its audited fixed ICP reference using SlicerSALT."
    )
    parser.add_argument("--side", choices=("left", "right"))
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Batch input folder. Defaults to the selected side under the sibling merged dataset.",
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Run one mask file instead of a batch.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Parent of left_hippocampus/right_hippocampus; defaults to the sibling merged dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Fresh output folder. If omitted, a timestamped folder is created under reruns/.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional shared run folder name, so separate left/right commands can share one rerun root.",
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Reference bundle root (defaults to the newest complete ICP/Referen_ICP bundle for the selected side).",
    )
    reference_group.add_argument(
        "--reference-template",
        type=Path,
        default=None,
        help="Select one reference mesh directly. The adjacent .json contract is required.",
    )
    parser.add_argument(
        "--slicer-exe",
        type=Path,
        default=DEFAULT_SLICER_EXE,
        help="SlicerSALT executable path.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths, side, and reference without starting ICP.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the Windows GUI for selecting side, input, reference, and output.",
    )
    return parser.parse_args(argv)


def resolve_inputs(args):
    if args.input_file is not None:
        input_file = args.input_file.expanduser().resolve()
        if not input_file.is_file():
            raise FileNotFoundError("Input file not found: {}".format(input_file))
        if not supported_volume(input_file):
            raise ValueError("Unsupported input volume type: {}".format(input_file.name))
        input_files = [input_file]
        input_path = input_file
        input_mode = "single"
    else:
        if args.input_dir is not None:
            input_dir = args.input_dir.expanduser().resolve()
        else:
            input_root = (args.input_root or DEFAULT_INPUT_ROOT).expanduser().resolve()
            input_dir = input_root / (args.side + "_hippocampus")
        if not input_dir.is_dir():
            raise FileNotFoundError("Input directory not found: {}".format(input_dir))
        input_files = sorted(
            (path for path in input_dir.rglob("*") if path.is_file() and supported_volume(path)),
            key=lambda path: path.name.casefold(),
        )
        labelled = [path for path in input_files if "label" in path.name.casefold()]
        if labelled:
            input_files = labelled
        if not input_files:
            raise ValueError("No supported NIfTI/NRRD input volumes found in {}".format(input_dir))
        input_path = input_dir
        input_mode = "batch"

    opposite_markers = "right|rh" if args.side == "left" else "left|lh"
    opposite_pattern = re.compile(
        r"(?i)(?:^|[_-])(?:{})(?:[_-]|\.)".format(opposite_markers)
    )
    wrong_side = [path.name for path in input_files if opposite_pattern.search(path.name)]
    if wrong_side:
        raise ValueError(
            "Input side does not match --side {}. Opposite-side file: {}".format(
                args.side, wrong_side[0]
            )
        )
    return input_mode, input_path, input_files


def load_reference(args):
    if args.reference_template is not None:
        reference_template = args.reference_template.expanduser().resolve()
    else:
        reference_root = (
            args.reference_root.expanduser().resolve()
            if args.reference_root is not None
            else default_reference_root(args.side)
        )
        reference_template = reference_root / args.side / "mean_shape.ply"
    reference_metadata = Path(str(reference_template) + ".json")
    if not reference_template.is_file() or not reference_metadata.is_file():
        raise FileNotFoundError(
            "Reference bundle is incomplete for {}: {} and {}".format(
                args.side, reference_template, reference_metadata
            )
        )

    contract = json.loads(reference_metadata.read_text(encoding="utf-8"))
    version = contract.get("version")
    if version not in SUPPORTED_REFERENCE_VERSIONS:
        raise ValueError("Unexpected reference version: {}".format(version))
    actual_hash = sha256_file(reference_template)
    if contract.get("template_sha256") != actual_hash:
        raise ValueError("Reference SHA-256 does not match its metadata: {}".format(reference_template))
    if not contract.get("output_voxels") or not contract.get("output_spacing"):
        raise ValueError("Reference metadata is missing the output grid definition")
    expected_independent = contract.get(
        "independent_test_reference",
        version == "fixed-training-reference-v1",
    )
    return reference_template, contract, actual_hash, bool(expected_independent)


def launch_gui():
    """Launch a small Tk GUI without changing the command-line workflow.

    The GUI deliberately calls this same file again in CLI mode.  Therefore
    all existing preflight checks, reference hash checks, Slicer invocation,
    and output verification remain the single source of truth.
    """
    try:
        import queue
        import threading
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
    except Exception as exc:  # pragma: no cover - depends on the local Python build
        print("GUI requires Tkinter. Run with standard Windows Python.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title("Hippocampal ICP - Fixed Reference Runner")
    root.geometry("980x700")
    root.minsize(820, 560)

    side_var = tk.StringVar(value="left")
    mode_var = tk.StringVar(value="batch")
    input_var = tk.StringVar(value=str(DEFAULT_INPUT_ROOT / "left_hippocampus"))
    reference_var = tk.StringVar(
        value=str(default_reference_root("left") / "left" / "mean_shape.ply")
    )
    output_var = tk.StringVar(value="")
    run_id_var = tk.StringVar(value=datetime.now().strftime("%Y%m%d_%H%M%S"))
    slicer_var = tk.StringVar(value=str(DEFAULT_SLICER_EXE))
    check_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value="พร้อมตรวจสอบ")
    last_defaults = {
        "input": input_var.get(),
        "reference": reference_var.get(),
    }
    messages = queue.Queue()

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(8, weight=1)

    ttk.Label(frame, text="ICP fixed-reference runner", font=("Segoe UI", 15, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
    )
    ttk.Label(
        frame,
        text="เลือก reference .ply ได้โดยตรง ระบบจะตรวจสอบไฟล์ .json และ SHA-256 ก่อนเริ่ม SlicerSALT",
    ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

    ttk.Label(frame, text="Side").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    side_box = ttk.Combobox(frame, textvariable=side_var, values=("left", "right"), state="readonly", width=12)
    side_box.grid(row=2, column=1, sticky="w", pady=4)

    ttk.Label(frame, text="Input mode").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
    mode_frame = ttk.Frame(frame)
    mode_frame.grid(row=3, column=1, columnspan=2, sticky="w", pady=4)
    ttk.Radiobutton(mode_frame, text="Batch folder", variable=mode_var, value="batch").pack(side="left", padx=(0, 12))
    ttk.Radiobutton(mode_frame, text="Single file", variable=mode_var, value="single").pack(side="left")

    def path_row(row, label, variable, browse_command):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="เลือก...", command=browse_command).grid(row=row, column=2, sticky="e", padx=(8, 0), pady=4)

    def choose_input():
        if mode_var.get() == "batch":
            selected = filedialog.askdirectory(title="เลือกโฟลเดอร์ input mask")
        else:
            selected = filedialog.askopenfilename(
                title="เลือกไฟล์ input mask",
                filetypes=(("Mask volumes", "*.nii *.nii.gz *.nrrd *.hdr"), ("All files", "*.*")),
            )
        if selected:
            input_var.set(selected)

    def choose_reference():
        selected = filedialog.askopenfilename(
            title="เลือก ICP reference mesh",
            filetypes=(("ICP reference mesh", "*.ply *.vtk"), ("All files", "*.*")),
        )
        if selected:
            reference_var.set(selected)

    def choose_output():
        selected = filedialog.askdirectory(title="เลือกโฟลเดอร์ output ใหม่")
        if selected:
            output_var.set(selected)

    def choose_slicer():
        selected = filedialog.askopenfilename(
            title="เลือก SlicerSALT.exe",
            filetypes=(("SlicerSALT executable", "*.exe"), ("All files", "*.*")),
        )
        if selected:
            slicer_var.set(selected)

    path_row(4, "Input", input_var, choose_input)
    path_row(5, "Reference mesh", reference_var, choose_reference)
    path_row(6, "Output folder (optional)", output_var, choose_output)
    path_row(7, "SlicerSALT.exe", slicer_var, choose_slicer)

    ttk.Label(frame, text="Run ID").grid(row=8, column=0, sticky="nw", padx=(0, 8), pady=4)
    ttk.Entry(frame, textvariable=run_id_var).grid(row=8, column=1, sticky="new", pady=4)
    ttk.Checkbutton(frame, text="ตรวจสอบอย่างเดียว (ไม่เริ่ม ICP)", variable=check_var).grid(
        row=8, column=2, sticky="nw", padx=(8, 0), pady=4
    )

    log_box = scrolledtext.ScrolledText(frame, height=16, wrap="word", state="disabled", font=("Consolas", 9))
    log_box.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(12, 8))
    frame.rowconfigure(9, weight=1)

    button_frame = ttk.Frame(frame)
    button_frame.grid(row=10, column=0, columnspan=3, sticky="ew")
    button_frame.columnconfigure(0, weight=1)
    ttk.Label(button_frame, textvariable=status_var).grid(row=0, column=0, sticky="w")
    check_button = ttk.Button(button_frame, text="ตรวจสอบ / เริ่มรัน")
    check_button.grid(row=0, column=1, padx=(8, 0))
    close_button = ttk.Button(button_frame, text="ปิด", command=root.destroy)
    close_button.grid(row=0, column=2, padx=(8, 0))

    def append_log(text):
        log_box.configure(state="normal")
        log_box.insert("end", text + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def update_defaults(_event=None):
        side = side_var.get()
        new_input = str(DEFAULT_INPUT_ROOT / (side + "_hippocampus"))
        new_reference = str(default_reference_root(side) / side / "mean_shape.ply")
        if input_var.get() == last_defaults["input"]:
            input_var.set(new_input)
        if reference_var.get() == last_defaults["reference"]:
            reference_var.set(new_reference)
        last_defaults["input"] = new_input
        last_defaults["reference"] = new_reference

    side_box.bind("<<ComboboxSelected>>", update_defaults)

    def build_gui_args():
        side = side_var.get().strip().lower()
        input_path = Path(input_var.get().strip()).expanduser()
        reference_path = Path(reference_var.get().strip()).expanduser()
        slicer_path = Path(slicer_var.get().strip()).expanduser()
        if not input_var.get().strip():
            raise ValueError("กรุณาเลือก input")
        if not reference_var.get().strip():
            raise ValueError("กรุณาเลือก reference mesh")
        if not input_path.exists():
            raise ValueError("ไม่พบ input: {}".format(input_path))
        if not reference_path.is_file():
            raise ValueError("ไม่พบ reference mesh: {}".format(reference_path))
        if not slicer_path.is_file():
            raise ValueError("ไม่พบ SlicerSALT.exe: {}".format(slicer_path))
        argv = ["--side", side, "--reference-template", str(reference_path), "--slicer-exe", str(slicer_path)]
        argv.extend(("--input-dir" if mode_var.get() == "batch" else "--input-file", str(input_path)))
        if output_var.get().strip():
            argv.extend(("--output-dir", str(Path(output_var.get().strip()).expanduser())))
        elif run_id_var.get().strip():
            argv.extend(("--run-id", run_id_var.get().strip()))
        if check_var.get():
            argv.append("--check-only")
        return argv

    def finish_process(process, args):
        return_code = process.wait()
        messages.put(("done", return_code, args))

    def poll_messages():
        try:
            while True:
                item = messages.get_nowait()
                if item[0] == "line":
                    append_log(item[1])
                else:
                    _, return_code, _args = item
                    check_button.configure(state="normal")
                    close_button.configure(state="normal")
                    status_var.set("เสร็จสิ้น (exit code {})".format(return_code))
                    if return_code == 0:
                        messagebox.showinfo("ICP", "ตรวจสอบ/รัน ICP เสร็จเรียบร้อย")
                    else:
                        messagebox.showerror("ICP ล้มเหลว", "ดูรายละเอียดในช่อง log ด้านบน")
        except queue.Empty:
            pass
        root.after(100, poll_messages)

    def start_process():
        try:
            cli_args = build_gui_args()
        except Exception as exc:
            messagebox.showerror("ข้อมูลไม่ครบหรือไม่ถูกต้อง", str(exc))
            return
        append_log(">>> python {}".format(" ".join(cli_args)))
        check_button.configure(state="disabled")
        close_button.configure(state="disabled")
        status_var.set("กำลังตรวจสอบ/รัน SlicerSALT...")
        try:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve())] + cli_args,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            check_button.configure(state="normal")
            close_button.configure(state="normal")
            status_var.set("เริ่ม process ไม่สำเร็จ")
            messagebox.showerror("เริ่ม ICP ไม่สำเร็จ", str(exc))
            return

        def stream_output():
            assert process.stdout is not None
            for line in process.stdout:
                messages.put(("line", line.rstrip()))
            finish_process(process, cli_args)

        threading.Thread(target=stream_output, daemon=True).start()

    check_button.configure(command=start_process)
    root.after(100, poll_messages)
    root.mainloop()
    return 0


def make_output_dir(args, input_mode, input_path):
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("--run-id may contain only letters, numbers, dot, underscore, and hyphen")
    run_root = PROJECT_ROOT / "reruns" / ("icp_legacy_reference_" + run_id)
    if input_mode == "single":
        case_name = input_path.name
        if case_name.lower().endswith(".nii.gz"):
            case_name = case_name[:-7]
        else:
            case_name = input_path.stem
        subject_match = re.search(r"sub-[A-Za-z0-9-]+", case_name, flags=re.IGNORECASE)
        if subject_match:
            case_name = subject_match.group(0)
        else:
            case_name = re.sub(r"^(left|right)[_-]", "", case_name, flags=re.IGNORECASE)
        return run_root / ("single_{}_{}".format(args.side, case_name))
    return run_root / args.side


def build_icp_arguments(args, input_mode, input_path, output_dir, reference_template, contract):
    arguments = [
        "--output_dir", str(output_dir),
        "--reference_template", str(reference_template),
        "--output_voxels", str(int(contract["output_voxels"])),
        "--output_spacing", str(float(contract["output_spacing"])),
        "--max_iterations", "20",
        "--tolerance", "0.00005",
        "--pairwise_iterations", "100",
        "--pairwise_tolerance", "0.0001",
        "--pairwise_landmarks", "200",
        "--interpolation", "nn",
        "--invert_transform", "auto",
    ]
    if input_mode == "single":
        arguments.extend(("--input_file", str(input_path)))
    else:
        arguments.extend(("--input_dir", str(input_path)))
    return arguments


def verify_outputs(args, input_mode, input_path, expected_count, output_dir, reference_template,
                   contract, reference_hash, expected_independent_test_reference,
                   process_returncode):
    status_path = output_dir / "icp_status.json"
    log_path = output_dir / "icp_debug_log.txt"
    if not status_path.is_file():
        raise RuntimeError(
            "ICP.py/Slicer finished without icp_status.json (exit code {}). Check {}".format(
                process_returncode, log_path
            )
        )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("success") is not True or status.get("mode") != "fixed_reference":
        raise RuntimeError("ICP status is not a successful fixed-reference run: {}".format(status))
    if status.get("reference_sha256") != reference_hash:
        raise RuntimeError("ICP status reports a different reference hash")
    if status.get("geometry_version") != contract.get("version"):
        raise RuntimeError(
            "ICP status reports geometry version {}, expected {}".format(
                status.get("geometry_version"), contract.get("version")
            )
        )
    if bool(status.get("independent_test_reference")) != bool(expected_independent_test_reference):
        raise RuntimeError(
            "ICP status provenance flag mismatch: expected {}, got {}".format(
                expected_independent_test_reference, status.get("independent_test_reference")
            )
        )
    if int(status.get("subjects", -1)) != expected_count:
        raise RuntimeError(
            "ICP status count mismatch: expected {}, got {}".format(
                expected_count, status.get("subjects")
            )
        )
    aligned_dir = output_dir / "aligned_nifti"
    aligned_count = sum(
        1 for path in aligned_dir.iterdir()
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
    ) if aligned_dir.is_dir() else 0
    if aligned_count != expected_count:
        raise RuntimeError(
            "Aligned output count mismatch: expected {}, found {} in {}".format(
                expected_count, aligned_count, aligned_dir
            )
        )
    if process_returncode != 0:
        raise RuntimeError(
            "ICP outputs passed count checks, but Python/Slicer returned exit code {}".format(
                process_returncode
            )
        )

    summary = {
        "schema": "fixed_reference_icp_run_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "side": args.side,
        "input_mode": input_mode,
        "input_path": str(input_path),
        "subjects": aligned_count,
        "reference": str(reference_template),
        "reference_sha256": reference_hash,
        "reference_version": contract["version"],
        "independent_test_reference": bool(expected_independent_test_reference),
        "output": str(output_dir),
        "parameters": status.get("parameters", {}),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return aligned_count


def main(argv=None):
    args = parse_args(argv)
    # A no-argument launch is treated as the interactive workflow.  This
    # keeps manual reference selection available without requiring users to
    # remember the --gui switch, while preserving normal CLI behavior once a
    # side or input argument is supplied.
    no_cli_selection = all(
        value is None
        for value in (
            args.side,
            args.input_dir,
            args.input_file,
            args.input_root,
            args.output_dir,
            args.run_id,
            args.reference_root,
            args.reference_template,
        )
    ) and not args.check_only
    if args.gui or no_cli_selection:
        return launch_gui()
    try:
        if args.side is None:
            raise ValueError("--side is required unless --gui is used")
        if args.input_root is not None and args.input_dir is not None:
            raise ValueError("Use either --input-root or --input-dir, not both")
        input_mode, input_path, input_files = resolve_inputs(args)
        reference_template, contract, reference_hash, expected_independent_test_reference = load_reference(args)
        slicer_exe = args.slicer_exe.expanduser().resolve()
        if not slicer_exe.is_file():
            raise FileNotFoundError("SlicerSALT executable not found: {}".format(slicer_exe))
        output_dir = make_output_dir(args, input_mode, input_path)
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise FileExistsError(
                "Output directory must be new or empty: {}".format(output_dir)
            )

        print("ICP step: fixed reference ({}, {})".format(args.side, input_mode), flush=True)
        print("Input: {}".format(input_path), flush=True)
        print("Input volumes: {}".format(len(input_files)), flush=True)
        print("Reference: {}".format(reference_template), flush=True)
        print("Reference SHA-256: {}".format(reference_hash), flush=True)
        print("Output: {}".format(output_dir), flush=True)
        if expected_independent_test_reference:
            print(
                "Reference provenance: contract marks this template as independent of the held-out test subjects.",
                flush=True,
            )
        else:
            print(
                "Reference provenance: this template is a legacy groupwise reference and is not independent of held-out subjects.",
                flush=True,
            )
        if args.check_only:
            print("Check passed. ICP was not started.", flush=True)
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        icp_arguments = build_icp_arguments(
            args, input_mode, input_path, output_dir, reference_template, contract
        )
        command = [
            str(slicer_exe),
            "--no-main-window",
            "--no-splash",
            "--python-script",
            str(ICP_SCRIPT),
        ] + icp_arguments
        environment = os.environ.copy()
        print("Starting SlicerSALT with ICP.py; waiting for Slicer to finish...", flush=True)
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, check=False)
        count = verify_outputs(
            args, input_mode, input_path, len(input_files), output_dir,
            reference_template, contract, reference_hash,
            expected_independent_test_reference, completed.returncode
        )
        print("Verified {} aligned volume(s).".format(count), flush=True)
        print("Run summary: {}".format(output_dir / "run_summary.json"), flush=True)
        return 0
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
