"""
run_pipeline.py
===============
Automated Hippocampus Preprocessing Pipeline

How it works:
  1. Select MRI files (.nii.gz) via GUI or command line
  2. Run FastSurfer Segmentation automatically
  3. Extract Hippocampus (Label 17/53) + Preprocessing (moderate mode)
  4. Organize output into outputpreprocess/left_hippocampus & right_hippocampus

Usage:
  python run_pipeline.py                    # Open GUI file picker
  python run_pipeline.py --input <file>     # Specify file directly
  python run_pipeline.py --input_dir <dir>  # Specify folder (batch)
"""

import os
import sys
import glob
import time
import shutil
import argparse
import subprocess
from datetime import datetime

# Import config
try:
    from config import (
        FASTSURFER_DIR, OUTPUT_DIR, FASTSURFER_OUTPUT_DIR,
        EXTRACTION_MODE, CLOSE_ITERATIONS,
        VOX_SIZE, BATCH_SIZE, DEVICE, THREADS
    )
except ImportError:
    print("[ERROR] config.py not found. Make sure it is in the same folder as run_pipeline.py")
    sys.exit(1)

# =============================================================================
# Constants
# =============================================================================
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_SCRIPT = os.path.join(PIPELINE_DIR, "extract_hippocampus.py")
FASTSURFER_PREDICT = os.path.join(FASTSURFER_DIR, "FastSurferCNN", "run_prediction.py")

SUPPORTED_EXTENSIONS = (".nii.gz", ".nii", ".mgz")


def print_banner():
    print()
    print("=" * 60)
    print("  [BRAIN] Hippocampus Preprocessing Pipeline")
    print("  FastSurfer Segmentation -> Hippocampus Extraction")
    print("=" * 60)
    print(f"  FastSurfer : {FASTSURFER_DIR}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"  Mode       : {EXTRACTION_MODE}")
    print("=" * 60)
    print()


def validate_setup():
    errors = []

    if not os.path.isdir(FASTSURFER_DIR):
        errors.append(f"FastSurfer directory not found: {FASTSURFER_DIR}")

    if not os.path.isfile(FASTSURFER_PREDICT):
        errors.append(f"FastSurfer run_prediction.py not found: {FASTSURFER_PREDICT}")

    if not os.path.isfile(EXTRACT_SCRIPT):
        errors.append(f"extract_hippocampus.py not found: {EXTRACT_SCRIPT}")

    # Check checkpoints
    ckpt_dir = os.path.join(FASTSURFER_DIR, "checkpoints")
    if os.path.isdir(ckpt_dir):
        pkls = glob.glob(os.path.join(ckpt_dir, "*.pkl"))
        if len(pkls) == 0:
            errors.append(f"No model checkpoints found in: {ckpt_dir}")
    else:
        errors.append(f"Checkpoints directory not found: {ckpt_dir}")

    # Check Python packages
    for pkg in ["nibabel", "numpy", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"Python package '{pkg}' not installed. Run: pip install {pkg}")

    if errors:
        print("[ERROR] Setup validation failed:")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    print("[OK] Setup validation passed")


def select_files_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        print("[ERROR] tkinter not available. Use --input or --input_dir instead")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askquestion(
        "Hippocampus Pipeline",
        "Select 'Yes' to choose MRI file(s)\n"
        "Select 'No' to choose a folder (batch mode)",
        icon="question"
    )

    files = []

    if choice == "yes":
        selected = filedialog.askopenfilenames(
            title="Select MRI file(s) (T1w)",
            filetypes=[
                ("NIfTI files", "*.nii.gz *.nii"),
                ("MGZ files", "*.mgz"),
                ("All files", "*.*")
            ]
        )
        files = list(selected)
    else:
        folder = filedialog.askdirectory(title="Select folder containing MRI files")
        if folder:
            files = find_mri_files(folder)

    root.destroy()

    if not files:
        print("[INFO] No files selected. Exiting.")
        sys.exit(0)

    return files


def find_mri_files(directory):
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))

    # Filter: keep only T1w-like files (not masks or segmentations)
    filtered = []
    skip_keywords = ["mask", "seg", "aseg", "aparc", "label", "hippo"]
    for f in files:
        basename = os.path.basename(f).lower()
        if not any(kw in basename for kw in skip_keywords):
            filtered.append(f)

    if not filtered and files:
        filtered = files

    return sorted(set(filtered))


def get_subject_id(filepath):
    basename = os.path.basename(filepath)
    for ext in [".nii.gz", ".nii", ".mgz"]:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    for suffix in ["_T1w", "_t1w", "_T1", "_t1"]:
        basename = basename.replace(suffix, "")
    return basename


def run_fastsurfer(input_file, subject_id):
    output_dir = os.path.join(FASTSURFER_OUTPUT_DIR, subject_id)
    seg_file = os.path.join(output_dir, "mri", "aparc.DKTatlas+aseg.deep.mgz")

    # Skip if output already exists
    if os.path.isfile(seg_file):
        print(f"  [SKIP] FastSurfer output already exists")
        return seg_file

    os.makedirs(os.path.join(output_dir, "mri"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "stats"), exist_ok=True)

    cmd = [
        sys.executable,
        FASTSURFER_PREDICT,
        "--t1", input_file,
        "--sid", subject_id,
        "--sd", FASTSURFER_OUTPUT_DIR,
        "--asegdkt_segfile", seg_file,
        "--conformed_name", os.path.join(output_dir, "mri", "orig.mgz"),
        "--brainmask_name", os.path.join(output_dir, "mri", "mask.mgz"),
        "--aseg_name", os.path.join(output_dir, "mri", "aseg.auto_noCCseg.mgz"),
        "--seg_log", os.path.join(output_dir, "scripts", "deep-seg.log"),
        "--vox_size", VOX_SIZE,
        "--batch_size", str(BATCH_SIZE),
        "--viewagg_device", DEVICE,
        "--device", DEVICE,
        "--threads", str(THREADS),
    ]

    print(f"  [RUN] FastSurfer segmentation...")
    
    # FastSurfer requires its root directory in PYTHONPATH to import FastSurferCNN
    env = os.environ.copy()
    env["PYTHONPATH"] = FASTSURFER_DIR + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
            env=env
        )
        if result.returncode != 0:
            print(f"  [ERROR] FastSurfer failed:")
            print(result.stderr[-500:] if result.stderr else "No error output")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] FastSurfer timeout (>30 min)")
        return None

    if os.path.isfile(seg_file):
        print(f"  [OK] Segmentation complete")
        return seg_file
    else:
        print(f"  [ERROR] Segmentation file not created")
        return None


def run_extraction(seg_file, subject_id, temp_dir):
    cmd = [
        sys.executable,
        EXTRACT_SCRIPT,
        "--input", seg_file,
        "--output_dir", temp_dir,
        "--mode", EXTRACTION_MODE,
        "--close_iter", str(CLOSE_ITERATIONS),
        "--prefix", subject_id,
    ]

    print(f"  [RUN] Extracting hippocampus (mode={EXTRACTION_MODE})...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [ERROR] Extraction failed:")
            print(result.stderr[-300:] if result.stderr else "No error output")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Extraction timeout")
        return False

    # Check output files
    lh = os.path.join(temp_dir, f"lh_{subject_id}_hippocampus.nii.gz")
    rh = os.path.join(temp_dir, f"rh_{subject_id}_hippocampus.nii.gz")

    if os.path.isfile(lh) and os.path.isfile(rh):
        print(f"  [OK] Extraction complete (LH + RH)")
        return True
    else:
        print(f"  [ERROR] Output files missing in {temp_dir}")
        return False


def organize_output(temp_dir, subject_id, orig_mgz=None):
    lh_dir = os.path.join(OUTPUT_DIR, "left_hippocampus")
    rh_dir = os.path.join(OUTPUT_DIR, "right_hippocampus")
    mri_dir = os.path.join(OUTPUT_DIR, "mri")
    os.makedirs(lh_dir, exist_ok=True)
    os.makedirs(rh_dir, exist_ok=True)
    os.makedirs(mri_dir, exist_ok=True)

    lh_src = os.path.join(temp_dir, f"lh_{subject_id}_hippocampus.nii.gz")
    rh_src = os.path.join(temp_dir, f"rh_{subject_id}_hippocampus.nii.gz")

    lh_dst = os.path.join(lh_dir, f"lh_{subject_id}_hippocampus.nii.gz")
    rh_dst = os.path.join(rh_dir, f"rh_{subject_id}_hippocampus.nii.gz")
    mri_dst = os.path.join(mri_dir, f"{subject_id}_t1.nii.gz")

    moved = 0
    if os.path.isfile(lh_src):
        shutil.copy2(lh_src, lh_dst)
        moved += 1
    if os.path.isfile(rh_src):
        shutil.copy2(rh_src, rh_dst)
        moved += 1

    # Save matching conformed MRI as NIfTI so it is paired with the masks
    if orig_mgz and os.path.isfile(orig_mgz):
        try:
            import nibabel as nib
            img = nib.load(orig_mgz)
            nii_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
            nib.save(nii_img, mri_dst)
            print(f"  [SAVED] Paired MRI saved to {mri_dst}")
        except Exception as e:
            print(f"  [WARNING] Could not save paired MRI: {e}")

    return moved == 2


def main():
    parser = argparse.ArgumentParser(
        description="Hippocampus Preprocessing Pipeline - FastSurfer -> Extraction -> Ready for SPHARM-PDM",
    )
    parser.add_argument("--input", "-i", nargs="+",
                        help="MRI file(s) (.nii.gz), one or more")
    parser.add_argument("--input_dir", "-d",
                        help="Folder containing MRI files (batch mode)")
    parser.add_argument("--output_dir", "-o",
                        help="Custom output folder path")
    parser.add_argument("--skip_fastsurfer", action="store_true",
                        help="Skip FastSurfer, use existing segmentation (input = mgz files)")
    args = parser.parse_args()

    # Configure output dirs before printing banner
    global OUTPUT_DIR, FASTSURFER_OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = os.path.join(os.path.abspath(args.output_dir), "fastsurfer")
        FASTSURFER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "fastsurfer_temp")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FASTSURFER_OUTPUT_DIR, exist_ok=True)

    print_banner()
    validate_setup()

    # === Step 1: Gather input files ===
    input_files = []

    if args.input:
        input_files = args.input
    elif args.input_dir:
        if args.skip_fastsurfer:
            input_files = glob.glob(
                os.path.join(args.input_dir, "**", "aparc.DKTatlas+aseg.deep.mgz"),
                recursive=True
            )
        else:
            input_files = find_mri_files(args.input_dir)
    else:
        # Open GUI
        input_files = select_files_gui()

    if not input_files:
        print("[ERROR] No input files found")
        sys.exit(1)

    print(f"\n[INFO] Found {len(input_files)} file(s)")
    for f in input_files[:5]:
        print(f"  - {f}")
    if len(input_files) > 5:
        print(f"  ... and {len(input_files)-5} more")

    # === Steps 2-4: Process each subject ===
    temp_extract_dir = os.path.join(PIPELINE_DIR, "_temp_extract")

    log_path = os.path.join(OUTPUT_DIR, "pipeline_log.txt")
    os.makedirs(temp_extract_dir, exist_ok=True)

    total = len(input_files)
    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    log_lines = []
    log_lines.append(f"Pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Total files: {total}")
    log_lines.append(f"Mode: {EXTRACTION_MODE}")
    log_lines.append("")

    for i, input_file in enumerate(input_files):
        print(f"\n{'-'*60}")
        print(f"[{i+1}/{total}] {os.path.basename(input_file)}")
        print(f"{'-'*60}")

        if args.skip_fastsurfer:
            # Input is already a segmentation file (.mgz)
            seg_file = input_file
            parent = os.path.basename(os.path.dirname(os.path.dirname(input_file)))
            subject_id = parent.replace("_T1w", "")
        else:
            subject_id = get_subject_id(input_file)

        # Check if already processed (supports both new and old filename conventions)
        lh_check_new = os.path.join(OUTPUT_DIR, "left_hippocampus", f"lh_{subject_id}_hippocampus.nii.gz")
        rh_check_new = os.path.join(OUTPUT_DIR, "right_hippocampus", f"rh_{subject_id}_hippocampus.nii.gz")
        lh_check_old = os.path.join(OUTPUT_DIR, "left_hippocampus", f"{subject_id}_hippocampus_lh.nii.gz")
        rh_check_old = os.path.join(OUTPUT_DIR, "right_hippocampus", f"{subject_id}_hippocampus_rh.nii.gz")
        
        if (os.path.isfile(lh_check_new) and os.path.isfile(rh_check_new)) or (os.path.isfile(lh_check_old) and os.path.isfile(rh_check_old)):
            print(f"  [SKIP] Already processed: {subject_id}")
            skipped += 1
            log_lines.append(f"[{i+1}/{total}] {subject_id}: SKIPPED")
            continue

        # Step 2: FastSurfer Segmentation
        if args.skip_fastsurfer:
            seg_file = input_file
        else:
            seg_file = run_fastsurfer(input_file, subject_id)
            if seg_file is None:
                failed += 1
                log_lines.append(f"[{i+1}/{total}] {subject_id}: FAILED (FastSurfer)")
                continue

        # Step 3: Hippocampus Extraction + Preprocessing
        extract_ok = run_extraction(seg_file, subject_id, temp_extract_dir)
        if not extract_ok:
            failed += 1
            log_lines.append(f"[{i+1}/{total}] {subject_id}: FAILED (Extraction)")
            continue

        # Step 4: Organize output
        orig_mgz = os.path.join(FASTSURFER_OUTPUT_DIR, subject_id, "mri", "orig.mgz")
        if not os.path.isfile(orig_mgz):
            orig_mgz = os.path.join(os.path.dirname(seg_file), "orig.mgz")
        org_ok = organize_output(temp_extract_dir, subject_id, orig_mgz=orig_mgz)
        if org_ok:
            success += 1
            log_lines.append(f"[{i+1}/{total}] {subject_id}: SUCCESS")
        else:
            failed += 1
            log_lines.append(f"[{i+1}/{total}] {subject_id}: FAILED (Organize)")

    # === Summary ===
    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"  Total:    {total}")
    print(f"  Success:  {success}")
    print(f"  Failed:   {failed}")
    print(f"  Skipped:  {skipped}")
    print(f"  Time:     {elapsed_min:.1f} minutes")
    print(f"{'='*60}")
    print(f"\n  Output: {OUTPUT_DIR}")

    lh_count = len(glob.glob(os.path.join(OUTPUT_DIR, "left_hippocampus", "*.nii.gz")))
    rh_count = len(glob.glob(os.path.join(OUTPUT_DIR, "right_hippocampus", "*.nii.gz")))
    print(f"  Left hippocampus:  {lh_count} files")
    print(f"  Right hippocampus: {rh_count} files")
    print()

    # Save log
    log_lines.append("")
    log_lines.append(f"Pipeline finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Success: {success}, Failed: {failed}, Skipped: {skipped}")
    log_lines.append(f"Time: {elapsed_min:.1f} minutes")

    log_path = os.path.join(OUTPUT_DIR, "pipeline_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"  Log: {log_path}")

    # Cleanup temp
    if os.path.isdir(temp_extract_dir):
        shutil.rmtree(temp_extract_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
