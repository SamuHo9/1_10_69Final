import os
import glob

def find_valid_mri_files(directory):
    """
    Scans for valid MRI images only (.nii.gz, .nii, .mgz).
    Excludes non-MRI files, masks, segmentations, and 3D meshes.
    """
    if not directory or not os.path.isdir(directory):
        return []

    mri_extensions = (".nii.gz", ".nii", ".mgz")
    skip_keywords = ["mask", "seg", "aseg", "aparc", "label", "hippo", ".vtk"]

    files = []
    for ext in mri_extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        files.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))

    seen = set()
    valid_mris = []
    for f in files:
        norm = os.path.normpath(f)
        if norm in seen:
            continue
        seen.add(norm)

        fname = os.path.basename(norm).lower()
        if any(keyword in fname for keyword in skip_keywords):
            continue

        valid_mris.append(norm)

    return valid_mris

def check_existing_stages(output_dir):
    """
    Checks which pipeline stages already have generated results in output_dir.
    Returns: (has_fastsurfer: bool, has_icp: bool, has_spharm: bool, has_result: bool)
    """
    if not output_dir or not os.path.isdir(output_dir):
        return False, False, False, False

    # 1. Check FastSurfer results
    fs_dir = os.path.join(output_dir, "fastsurfer")
    fs_files = []
    if os.path.isdir(fs_dir):
        for ext in ("*.nii*", "*.vtk", "*.mgz"):
            fs_files.extend(glob.glob(os.path.join(fs_dir, ext)))
            fs_files.extend(glob.glob(os.path.join(fs_dir, "**", ext), recursive=True))
    has_fastsurfer = len(fs_files) > 0

    # 2. Check ICP results
    icp_dir = os.path.join(output_dir, "output_ICP")
    icp_files = []
    if os.path.isdir(icp_dir):
        for ext in ("*.vtk", "*.ply", "*.nii*"):
            icp_files.extend(glob.glob(os.path.join(icp_dir, ext)))
            icp_files.extend(glob.glob(os.path.join(icp_dir, "**", ext), recursive=True))
        icp_files = [f for f in icp_files if "mean_shape" not in os.path.basename(f).lower() and not os.path.basename(f).lower().startswith("template_")]
    has_icp = len(icp_files) > 0

    # 3. Check SPHARM results
    spharm_dir = os.path.join(output_dir, "output_SPHARM")
    sph_files = []
    if os.path.isdir(spharm_dir):
        for ext in ("*SPHARM*.vtk", "*.vtk"):
            sph_files.extend(glob.glob(os.path.join(spharm_dir, ext)))
            sph_files.extend(glob.glob(os.path.join(spharm_dir, "**", ext), recursive=True))
        sph_files = [
            f for f in sph_files 
            if "mean_shape" not in os.path.basename(f).lower() 
            and not os.path.basename(f).lower().startswith("template_") 
            and not any(aux in os.path.basename(f).lower() for aux in ("_para.", "_surf.", "medialaxis", "_grid."))
        ]
    has_spharm = len(sph_files) > 0

    # 4. Check Result panel results (ResNet prediction & 3D Grad-CAM)
    res_dir = os.path.join(output_dir, "output_Result")
    summary_json = os.path.join(res_dir, "evaluation_summary.json")
    has_result = os.path.isfile(summary_json)

    return has_fastsurfer, has_icp, has_spharm, has_result

def find_spharm_mesh(out_dir, subject, side):
    """Finds matching SPHARM .vtk mesh in output_SPHARM directory for given subject and side."""
    if not out_dir or not subject or not os.path.isdir(out_dir):
        return None
    side_key = "left" if str(side).lower() in ("left", "lh") else "right"
    search_dirs = [
        os.path.join(out_dir, "output_SPHARM", side_key, "spharm_results"),
        os.path.join(out_dir, "output_SPHARM", side_key),
        os.path.join(out_dir, "output_SPHARM", "spharm_results"),
        os.path.join(out_dir, "output_SPHARM"),
        os.path.join(out_dir, f"output_{side_key}_hippocampus", f"spharm_results_{side_key}"),
    ]
    candidates = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "**", f"*{subject}*.vtk"), recursive=True):
            fname = os.path.basename(f).lower()
            if any(aux in fname for aux in ("_para.", "_surf.", "medialaxis", "_grid.", "template_", "mean_shape")):
                continue
            f_norm = f.replace("\\", "/").lower()
            if side_key == "left" and not (fname.startswith("lh_") or "left" in fname or "/left/" in f_norm):
                continue
            if side_key == "right" and not (fname.startswith("rh_") or "right" in fname or "/right/" in f_norm):
                continue
            candidates.append(os.path.normpath(f))

    if not candidates:
        return None

    for suf in ("_SPHARM_realigned.vtk", "_realigned.vtk", "_SPHARM_procalign.vtk", "_procalign.vtk", "_SPHARM_ellalign.vtk", "_ellalign.vtk", "_SPHARM.vtk"):
        for c in candidates:
            if c.endswith(suf):
                return c
    return candidates[0]
