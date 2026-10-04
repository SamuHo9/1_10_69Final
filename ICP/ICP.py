import os
import sys


def _launch_standard_gui():
    """Select ICP inputs before SlicerSALT starts.

    Normal Windows Python has Tk available even when Slicer is still loading
    its scripted modules.  Opening this selector before the Slicer bootstrap
    makes a no-argument ``python ICP.py`` invocation reliable.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from datetime import datetime

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_archive = os.path.join(script_dir, "Referen_ICP")
    mode_values = {
        "สร้าง reference ใหม่จาก training": "fit_reference",
        "ใช้ reference เดิม": "fixed_reference",
        "ทดลอง groupwise": "exploratory_groupwise",
    }
    result = [None]

    root = tk.Tk()
    root.title("Hippocampal ICP")
    root.geometry("820x560")
    root.minsize(720, 500)

    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    ttk.Label(frame, text="ICP alignment and reference builder",
              font=("Segoe UI", 16, "bold")).grid(
                  row=0, column=0, columnspan=3, sticky="w", pady=(0, 6)
              )
    ttk.Label(
        frame,
        text="เลือกโหมด, ฝั่ง, input, output และ reference แล้วกด เริ่ม ICP\n"
             "การสร้าง reference ควรใช้ข้อมูล training เท่านั้น",
    ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))

    mode_var = tk.StringVar(value="สร้าง reference ใหม่จาก training")
    side_var = tk.StringVar(value="left")
    input_mode_var = tk.StringVar(value="โฟลเดอร์ batch")
    input_var = tk.StringVar()
    output_var = tk.StringVar()
    reference_var = tk.StringVar()
    run_id_var = tk.StringVar(value=datetime.now().strftime("ICP_reference_%Y%m%d_%H%M%S"))
    archive_var = tk.StringVar(value=default_archive)

    ttk.Label(frame, text="โหมด:").grid(row=2, column=0, sticky="w", pady=4)
    mode_box = ttk.Combobox(frame, textvariable=mode_var,
                            values=list(mode_values.keys()), state="readonly")
    mode_box.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

    ttk.Label(frame, text="ฝั่ง:").grid(row=3, column=0, sticky="w", pady=4)
    side_box = ttk.Combobox(frame, textvariable=side_var,
                            values=("left", "right"), state="readonly", width=14)
    side_box.grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(frame, text="ชนิด input:").grid(row=4, column=0, sticky="w", pady=4)
    input_mode_box = ttk.Combobox(frame, textvariable=input_mode_var,
                                  values=("โฟลเดอร์ batch", "ไฟล์ mask เดี่ยว"),
                                  state="readonly", width=20)
    input_mode_box.grid(row=4, column=1, sticky="w", pady=4)

    def add_path_row(row, label, variable, button_text, callback):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(frame, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        button = ttk.Button(frame, text=button_text, command=callback)
        button.grid(row=row, column=2, sticky="e", padx=(8, 0), pady=4)
        return entry, button

    def choose_input():
        if input_mode_var.get() == "โฟลเดอร์ batch":
            selected = filedialog.askdirectory(title="เลือกโฟลเดอร์ input mask")
        else:
            selected = filedialog.askopenfilename(
                title="เลือกไฟล์ input mask",
                filetypes=(("Mask volumes", "*.nii *.nii.gz *.nrrd *.hdr"),
                           ("All files", "*.*")),
            )
        if selected:
            input_var.set(selected)

    def choose_output():
        selected = filedialog.askdirectory(title="เลือกโฟลเดอร์ output")
        if selected:
            output_var.set(selected)

    def choose_reference():
        selected = filedialog.askopenfilename(
            title="เลือก ICP reference",
            filetypes=(("ICP reference", "*.ply *.vtk"), ("All files", "*.*")),
        )
        if selected:
            reference_var.set(selected)

    def choose_archive():
        selected = filedialog.askdirectory(title="เลือกโฟลเดอร์เก็บ reference")
        if selected:
            archive_var.set(selected)

    add_path_row(5, "Input:", input_var, "เลือก...", choose_input)
    add_path_row(6, "Output (ว่าง = อัตโนมัติ):", output_var, "เลือก...", choose_output)
    reference_entry, reference_button = add_path_row(
        7, "Reference .ply:", reference_var, "เลือก...", choose_reference
    )
    ttk.Label(frame, text="Reference run ID:").grid(row=8, column=0, sticky="w", pady=4)
    run_id_entry = ttk.Entry(frame, textvariable=run_id_var)
    run_id_entry.grid(row=8, column=1, columnspan=2, sticky="ew", pady=4)
    archive_entry, archive_button = add_path_row(
        9, "Archive root:", archive_var, "เลือก...", choose_archive
    )

    note = ttk.Label(
        frame,
        text="ผล reference จะอยู่ใน ICP\\Referen_ICP\\<run_id>\\<left หรือ right>",
    )
    note.grid(row=10, column=0, columnspan=3, sticky="w", pady=(8, 12))

    def update_controls(*_args):
        mode = mode_values[mode_var.get()]
        fixed = mode == "fixed_reference"
        fit = mode == "fit_reference"
        reference_state = "normal" if fixed else "disabled"
        archive_state = "normal" if fit else "disabled"
        reference_entry.configure(state=reference_state)
        reference_button.configure(state=reference_state)
        run_id_entry.configure(state=archive_state)
        archive_entry.configure(state=archive_state)
        archive_button.configure(state=archive_state)

    def start_run():
        mode = mode_values[mode_var.get()]
        input_path = input_var.get().strip()
        reference_path = reference_var.get().strip()
        if not input_path:
            messagebox.showerror("ข้อมูลไม่ครบ", "กรุณาเลือก input")
            return
        if input_mode_var.get() == "โฟลเดอร์ batch" and not os.path.isdir(input_path):
            messagebox.showerror("ไม่พบ input", input_path)
            return
        if input_mode_var.get() == "ไฟล์ mask เดี่ยว" and not os.path.isfile(input_path):
            messagebox.showerror("ไม่พบ input", input_path)
            return
        if mode == "fixed_reference" and not os.path.isfile(reference_path):
            messagebox.showerror("ไม่พบ reference", "กรุณาเลือกไฟล์ mean_shape.ply")
            return

        argv = []
        argv.extend(("--input_dir" if input_mode_var.get() == "โฟลเดอร์ batch"
                     else "--input_file", input_path))
        if output_var.get().strip():
            argv.extend(("--output_dir", output_var.get().strip()))
        if mode == "fit_reference":
            argv.extend((
                "--fit_reference",
                "--reference_side", side_var.get(),
                "--reference_run_id", run_id_var.get().strip(),
                "--reference_archive_root", archive_var.get().strip() or default_archive,
            ))
        elif mode == "fixed_reference":
            argv.extend(("--reference_template", reference_path))
        else:
            argv.append("--exploratory_groupwise")
        result[0] = argv
        root.destroy()

    input_mode_box.bind("<<ComboboxSelected>>", update_controls)
    mode_box.bind("<<ComboboxSelected>>", update_controls)
    button_row = ttk.Frame(frame)
    button_row.grid(row=11, column=0, columnspan=3, sticky="e", pady=(10, 0))
    ttk.Button(button_row, text="ยกเลิก", command=root.destroy).pack(side="right", padx=(8, 0))
    ttk.Button(button_row, text="เริ่ม ICP", command=start_run).pack(side="right")
    update_controls()
    root.mainloop()
    return result[0]


def _bootstrap_slicer():
    try:
        import slicer
        return
    except ImportError:
        pass

    slicer_exe = os.environ.get(
        "SLICER_EXE",
        r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe",
    )
    if not os.path.exists(slicer_exe):
        print(f"[ERROR] SlicerSALT not found at: {slicer_exe}")
        print("        Set env var SLICER_EXE or edit the default path in ICP.py")
        sys.exit(1)

    import subprocess
    script_path = os.path.abspath(__file__)
    if len(sys.argv) == 1 or "--gui" in sys.argv[1:]:
        try:
            selected_args = _launch_standard_gui()
        except Exception as gui_error:
            print("[WARNING] Could not open the standard Python GUI: {}".format(gui_error))
            selected_args = "__GUI_FALLBACK__"
        if selected_args != "__GUI_FALLBACK__":
            if selected_args is None:
                print("[INFO] ICP GUI cancelled.")
                sys.exit(0)
            cmd = [slicer_exe, "--no-main-window", "--no-splash",
                   "--python-script", script_path] + selected_args
            print("[INFO] GUI selections accepted; starting SlicerSALT with ICP.py")
            sys.exit(subprocess.call(cmd))
    cmd = [slicer_exe, "--no-main-window", "--no-splash",
           "--python-script", script_path] + sys.argv[1:]
    print(f"[INFO] No 'slicer' module in this Python. Re-launching via SlicerSALT:")
    print(f"       {slicer_exe}")
    sys.exit(subprocess.call(cmd))

_bootstrap_slicer()

import vtk
import numpy as np
import glob
import argparse
import json
import hashlib
import re
import shutil
import slicer
from datetime import datetime
from vtk.util.numpy_support import vtk_to_numpy

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
    for arg in sys.argv:
        if arg.endswith(".py") and os.path.isfile(arg):
            SCRIPT_DIR = os.path.dirname(os.path.abspath(arg))
            break

DEBUG_LOG = os.path.join(SCRIPT_DIR, "icp_debug_log.txt")
sys.path.insert(0, SCRIPT_DIR)
from reference_contract import write_reference_contract, load_reference_contract

DEFAULT_REFERENCE_ARCHIVE_ROOT = os.path.join(SCRIPT_DIR, "Referen_ICP")

def sprint(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] [ICP-LOG] {msg}"
    print(formatted_msg)
    sys.stdout.flush()
    with open(DEBUG_LOG, "a") as f:
        f.write(formatted_msg + "\n")


def _sha256_file(path):
    """Return the SHA-256 digest used to bind a reference to its contract."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_reference_run_id(value):
    run_id = value or datetime.now().strftime("ICP_reference_%Y%m%d_%H%M%S")
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--reference_run_id may contain only letters, numbers, underscore, and hyphen")
    return run_id


def _infer_reference_side(args):
    """Infer left/right from paths, while allowing an explicit side override."""
    if getattr(args, "reference_side", None):
        return args.reference_side
    path_text = " ".join(str(value or "") for value in
                          (getattr(args, "input_dir", None),
                           getattr(args, "input_file", None),
                           getattr(args, "output_dir", None))).lower().replace("\\", "/")
    tokens = set(token for token in re.split(r"[^a-z0-9]+", path_text) if token)
    has_left = bool(tokens.intersection({"left", "lh"}))
    has_right = bool(tokens.intersection({"right", "rh"}))
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    raise ValueError("Cannot infer reference side from the paths; pass --reference_side left or right")


def _archive_fitted_reference(output_dir, args, reference_path):
    """Archive a successful fit-reference result directly from ICP.py.

    This keeps the one-command workflow reproducible: the output reference,
    contract, status, and convergence history are copied into a timestamped
    side-specific bundle without changing the ICP output directory.
    """
    output_dir = os.path.abspath(output_dir)
    reference_path = os.path.abspath(reference_path)
    contract_path = reference_path + ".json"
    status_path = os.path.join(output_dir, "icp_status.json")
    if not os.path.isfile(reference_path) or not os.path.isfile(contract_path):
        raise FileNotFoundError(
            "fit_reference finished without mean_shape.ply and mean_shape.ply.json"
        )
    if not os.path.isfile(status_path):
        raise FileNotFoundError("fit_reference finished without icp_status.json")

    with open(status_path, "r", encoding="utf-8-sig") as stream:
        status = json.load(stream)
    if status.get("success") is not True or status.get("mode") != "fit_reference":
        raise ValueError("ICP status is not a successful fit_reference run")

    with open(contract_path, "r", encoding="utf-8-sig") as stream:
        contract = json.load(stream)
    template_hash = _sha256_file(reference_path)
    if contract.get("template_sha256") != template_hash:
        raise ValueError("Reference contract hash does not match mean_shape.ply")
    if not contract.get("output_voxels") or not contract.get("output_spacing"):
        raise ValueError("Reference contract is missing output grid metadata")

    side = _infer_reference_side(args)
    run_id = _safe_reference_run_id(getattr(args, "reference_run_id", None))
    archive_root = os.path.abspath(
        getattr(args, "reference_archive_root", None) or DEFAULT_REFERENCE_ARCHIVE_ROOT
    )
    destination = os.path.join(archive_root, run_id, side)
    if os.path.isdir(destination) and os.listdir(destination):
        raise FileExistsError("Reference archive destination is not empty: " + destination)
    os.makedirs(destination, exist_ok=True)

    copied_names = ["mean_shape.ply", "mean_shape.ply.json", "icp_status.json"]
    history_path = os.path.join(output_dir, "icp_convergence_history.json")
    if os.path.isfile(history_path):
        copied_names.append("icp_convergence_history.json")
    for name in copied_names:
        shutil.copy2(os.path.join(output_dir, name), os.path.join(destination, name))

    manifest_path = os.path.join(archive_root, run_id, "reference_manifest.json")
    existing = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8-sig") as stream:
            existing = json.load(stream)
    references = [item for item in existing.get("references", []) if item.get("side") != side]
    references.append({
        "side": side,
        "source_output_dir": output_dir,
        "source_reference": reference_path,
        "archived_reference": os.path.join(destination, "mean_shape.ply"),
        "template_sha256": template_hash,
        "contract_sha256": _sha256_file(contract_path),
        "contract": contract,
        "status": status,
    })
    manifest = {
        "schema": "timestamped_icp_reference_bundle_v1",
        "created_at": existing.get("created_at", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
        "run_id": run_id,
        "archive_root": os.path.join(archive_root, run_id),
        "references": references,
    }
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return destination, manifest_path

def get_points_numpy(poly):
    if poly is None or poly.GetPoints() is None:
        return np.zeros((0, 3), dtype=np.float64)
    return vtk_to_numpy(poly.GetPoints().GetData())

def prompt_folder(title):
    import qt
    folder = qt.QFileDialog.getExistingDirectory(None, title)
    if not folder:
        return None
    return folder

def apply_poly_transform(poly, matrix_np):
    transform = vtk.vtkTransform()
    matrix_vtk = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            matrix_vtk.SetElement(r, c, float(matrix_np[r, c]))
    transform.SetMatrix(matrix_vtk)
    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetInputData(poly)
    transformer.SetTransform(transform)
    transformer.Update()
    return transformer.GetOutput()

def icp_distance(source_poly, target_poly, n_sample=500):
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(target_poly)
    locator.BuildLocator()
    src_pts = get_points_numpy(source_poly)
    n_sample = min(len(src_pts), n_sample)
    idx = np.linspace(0, len(src_pts) - 1, n_sample, dtype=int)
    total_dist = 0.0
    for i in idx:
        pt = src_pts[i].tolist()
        closest_id = locator.FindClosestPoint(pt)
        cp = target_poly.GetPoint(closest_id)
        total_dist += np.sqrt(sum((src_pts[i][k] - cp[k]) ** 2 for k in range(3)))
    return total_dist / n_sample

def load_and_mesh_node(filepath, max_retries=2):
    normalized = os.path.abspath(filepath).replace("\\", "/")
    last_err = None
    for attempt in range(max_retries + 1):
        node = None
        try:
            node = slicer.util.loadLabelVolume(normalized)
            if not node:
                last_err = "loadLabelVolume returned None"
                continue
            img = node.GetImageData()
            if img is None or img.GetNumberOfPoints() == 0:
                last_err = "empty image data"
                slicer.mrmlScene.RemoveNode(node)
                continue
            dmc = vtk.vtkDiscreteMarchingCubes()
            dmc.SetInputData(img)
            dmc.GenerateValues(1, 1, 100)
            dmc.Update()
            poly = dmc.GetOutput()
            ijkToRas = vtk.vtkMatrix4x4()
            node.GetIJKToRASMatrix(ijkToRas)
            t = vtk.vtkTransform()
            t.SetMatrix(ijkToRas)
            transformer = vtk.vtkTransformPolyDataFilter()
            transformer.SetTransform(t)
            transformer.SetInputData(poly)
            transformer.Update()
            result_poly = transformer.GetOutput()
            slicer.mrmlScene.RemoveNode(node)
            return result_poly
        except Exception as e:
            last_err = str(e).split("\n")[0][:200]
            if node is not None:
                try: slicer.mrmlScene.RemoveNode(node)
                except Exception: pass
            if attempt < max_retries:
                try:
                    import gc; gc.collect()
                    import time; time.sleep(0.5)
                except Exception:
                    pass
    sprint(f"  !!! load_and_mesh_node failed after {max_retries+1} attempts: {last_err}")
    return None

def principal_axis_align(poly):
    pts = get_points_numpy(poly)
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    z_ax = Vt[0]
    x_ax = Vt[1]
    y_ax = np.cross(z_ax, x_ax)
    y_ax /= np.linalg.norm(y_ax)
    x_ax = np.cross(y_ax, z_ax)
    x_ax /= np.linalg.norm(x_ax)
    R = np.eye(3)
    R[:, 0] = x_ax
    R[:, 1] = y_ax
    R[:, 2] = z_ax
    # กันกรณี degenerate: R ต้องเป็น proper rotation (det=+1) ไม่ใช่ reflection
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
        sprint("    NOTE: PCA frame เป็น left-handed, กลับแกน X เพื่อให้ det=+1")
    T_rot = np.eye(4)
    T_rot[:3, :3] = R.T
    T_cent = np.eye(4)
    T_cent[:3, 3] = -centroid
    T_combined = T_rot @ T_cent
    return apply_poly_transform(poly, T_combined), T_combined

_FLIP_CANDIDATES = [
    np.eye(4),
    np.diag([1.0, -1.0, -1.0, 1.0]),
    np.diag([-1.0, 1.0, -1.0, 1.0]),
    np.diag([-1.0, -1.0, 1.0, 1.0]),
]
_FLIP_NAMES = ["identity", "rotX180", "rotY180", "rotZ180"]

def pole_flip_correction(poly, reference_poly=None):
    if reference_poly is None:
        pts = get_points_numpy(poly)
        z_mid = (pts[:, 2].min() + pts[:, 2].max()) / 2.0
        pts_pos = pts[pts[:, 2] > z_mid]
        pts_neg = pts[pts[:, 2] <= z_mid]
        fat_pos = (np.std(pts_pos[:, 0]) + np.std(pts_pos[:, 1])) if len(pts_pos) > 0 else 0.0
        fat_neg = (np.std(pts_neg[:, 0]) + np.std(pts_neg[:, 1])) if len(pts_neg) > 0 else 0.0
        if fat_neg > fat_pos:
            idx = 2
        else:
            idx = 0
        sprint(f"    Pole check (no ref): fat_neg={fat_neg:.4f}, fat_pos={fat_pos:.4f} -> {_FLIP_NAMES[idx]}")
        T = _FLIP_CANDIDATES[idx].copy()
        return apply_poly_transform(poly, T), T, (idx != 0)

    best_idx = 0
    best_dist = float("inf")
    dists = []
    for idx, T in enumerate(_FLIP_CANDIDATES):
        poly_test = apply_poly_transform(poly, T)
        d = icp_distance(poly_test, reference_poly)
        dists.append(d)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    sprint(f"    Pole check: dists=[{', '.join(f'{d:.5f}' for d in dists)}] -> {_FLIP_NAMES[best_idx]}")
    T = _FLIP_CANDIDATES[best_idx].copy()
    return apply_poly_transform(poly, T), T, (best_idx != 0)

def compute_mean_poly(meshes):
    counts = [m.GetNumberOfPoints() for m in meshes]
    sprint(f"    compute_mean_poly: NN-matching, point counts {min(counts)}-{max(counts)}, N={len(meshes)}")

    ref_pts = get_points_numpy(meshes[0])
    mean_points = np.zeros_like(ref_pts)

    for m in meshes:
        locator = vtk.vtkKdTreePointLocator()
        locator.SetDataSet(m)
        locator.BuildLocator()
        pts = get_points_numpy(m)
        for j, pt in enumerate(ref_pts):
            closest_id = locator.FindClosestPoint(pt.tolist())
            mean_points[j] += pts[closest_id]

    mean_points /= len(meshes)

    mean_poly = vtk.vtkPolyData()
    mean_poly.DeepCopy(meshes[0])
    vtk_pts = mean_poly.GetPoints()
    for j in range(len(mean_points)):
        vtk_pts.SetPoint(j, mean_points[j].tolist())

    return mean_poly

EVAL_PAIRWISE_STEPS = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]

def run_vtk_icp(source_poly, target_poly, return_history=False, max_iter=100, tolerance=0.0001, landmarks=200):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source_poly)
    icp.SetTarget(target_poly)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(max_iter)
    icp.SetMaximumMeanDistance(tolerance)
    icp.SetMaximumNumberOfLandmarks(landmarks)
    icp.CheckMeanDistanceOn()
    icp.Update()
    matrix = icp.GetMatrix()
    res = np.eye(4)
    for r in range(4):
        for c in range(4):
            res[r, c] = matrix.GetElement(r, c)

    if not return_history:
        return res

    history = []
    for k in EVAL_PAIRWISE_STEPS:
        icp_k = vtk.vtkIterativeClosestPointTransform()
        icp_k.SetSource(source_poly)
        icp_k.SetTarget(target_poly)
        icp_k.GetLandmarkTransform().SetModeToRigidBody()
        icp_k.SetMaximumNumberOfIterations(k)
        icp_k.SetMaximumMeanDistance(tolerance)
        icp_k.SetMaximumNumberOfLandmarks(landmarks)
        icp_k.CheckMeanDistanceOn()
        icp_k.Update()
        history.append(float(icp_k.GetMeanDistance()))

    return res, history

def _strip_ext(path):
    """ตัดนามสกุลแบบปลอดภัย (split('.')[0] จะพังกับชื่อเช่น sub-01.3T.nii.gz)"""
    bn = os.path.basename(path)
    for ext in [".nii.gz", ".nii", ".mgz", ".nrrd", ".hdr", ".img"]:
        if bn.lower().endswith(ext):
            return bn[:-len(ext)]
    return os.path.splitext(bn)[0]


_INTERP_MAP = {
    "nn": "nn", "nearestneighbor": "nn", "nearestneighbour": "nn", "nearest": "nn",
    "linear": "linear",
    "ws": "ws", "windowedsinc": "ws",
    "bs": "bs", "bspline": "bs",
}


def _norm_interp(name):
    """
    แปลงชื่อ interpolation ให้เป็นค่าที่ ResampleScalarVectorDWIVolume รู้จัก
    (nn / linear / ws / bs) - สำหรับ label map ต้องเป็น nn เท่านั้น
    """
    key = str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    val = _INTERP_MAP.get(key)
    if val is None:
        sprint(f"    !!! ไม่รู้จัก interpolation '{name}' - ใช้ 'nn' แทน")
        return "nn"
    if val != "nn":
        sprint(f"    !!! เตือน: ใช้ interpolation '{val}' กับ label map "
               f"จะทำให้ค่า label เพี้ยนและชิ้นเนื้อถูกกร่อน - ควรใช้ 'nn'")
    return val


def _make_ref_node(n_voxels, spacing_mm, half):
    """สร้าง reference grid ที่ zero-init แล้ว (AllocateScalars ไม่เคลียร์ memory เอง)"""
    ref_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "ref")
    img = vtk.vtkImageData()
    img.SetDimensions(n_voxels, n_voxels, n_voxels)
    img.AllocateScalars(vtk.VTK_SHORT, 1)
    img.GetPointData().GetScalars().Fill(0)
    ref_node.SetAndObserveImageData(img)
    ref_node.SetSpacing(spacing_mm, spacing_mm, spacing_mm)
    ref_node.SetOrigin(-half, -half, -half)
    ref_node.SetIJKToRASDirections(1, 0, 0, 0, 1, 0, 0, 0, 1)
    return ref_node


def _resample_with_matrix(input_node, M, n_voxels, spacing_mm, half, interpolation):
    """
    Resample input_node ลงบน grid มาตรฐานด้วยเมทริกซ์ M
    คืน (out_node, status_string) - ผู้เรียกเป็นคนลบ out_node
    """
    ref_node = _make_ref_node(n_voxels, spacing_mm, half)
    out_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "out")

    t_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
    v_mat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            v_mat.SetElement(r, c, float(M[r, c]))
    t_node.SetMatrixTransformToParent(v_mat)

    params = {
        "inputVolume": input_node.GetID(),
        "referenceVolume": ref_node.GetID(),
        "outputVolume": out_node.GetID(),      # FIX: แยกจาก referenceVolume
        "transformationFile": t_node.GetID(),
        # FIX: โมดูลนี้ใช้คีย์ "interpolationType" ค่า nn/linear/ws/bs
        # ของเดิมส่ง "interpolationMode":"NearestNeighbor" (เป็นของ BRAINSResample)
        # -> พารามิเตอร์ถูกเมิน -> ใช้ default = linear -> binary label ถูกกร่อน 1 voxel
        "interpolationType": _norm_interp(interpolation)
    }
    cli_node = slicer.cli.run(
        slicer.modules.resamplescalarvectordwivolume,
        None, params, wait_for_completion=True
    )
    status = cli_node.GetStatusString()
    if status != "Completed":
        try:
            err = cli_node.GetErrorText()
        except Exception:
            err = "(no error text)"
        sprint(f"    !!! CLI resample status={status}: {str(err)[:300]}")

    slicer.mrmlScene.RemoveNode(t_node)
    slicer.mrmlScene.RemoveNode(ref_node)
    try:
        slicer.mrmlScene.RemoveNode(cli_node)
    except Exception:
        pass
    return out_node, status


def _occupied_bounds_ras(out_node, spacing_mm, half):
    """หา bounding box (RAS) ของ voxel ที่ไม่เป็นศูนย์ในผลลัพธ์"""
    arr = slicer.util.arrayFromVolume(out_node)   # shape (k, j, i)
    if arr is None:
        return None, 0
    ijk = np.argwhere(arr > 0)
    if len(ijk) == 0:
        return None, 0
    ras = ijk[:, ::-1] * spacing_mm + np.array([-half, -half, -half])
    b = np.array([ras[:, 0].min(), ras[:, 0].max(),
                  ras[:, 1].min(), ras[:, 1].max(),
                  ras[:, 2].min(), ras[:, 2].max()])
    return b, int(len(ijk))


def _bounds_mismatch(nifti_b, mesh_b):
    """ผลรวมความต่างของ bounds ทั้ง 6 ค่า ใช้เลือกทิศ transform ที่ถูก"""
    if nifti_b is None:
        return float("inf")
    return float(np.abs(nifti_b - np.array(mesh_b)).sum())


def export_aligned_nifti(file_list, T_matrices, output_dir, aligned_meshes=None,
                         spacing_mm=0.5, n_voxels=128, interpolation="NearestNeighbor",
                         invert_transform="auto"):
    out_vol_dir = os.path.join(output_dir, "aligned_nifti")
    os.makedirs(out_vol_dir, exist_ok=True)
    N = len(file_list)
    half = (n_voxels / 2.0) * spacing_mm
    sprint(f"  Output grid: {n_voxels}^3, spacing={spacing_mm:.6f}, box=[{-half:+.4f},{half:+.4f}]")

    use_inverse = (invert_transform == "yes")

    # ---- Auto-probe ทิศทางของ transform กับ subject แรก ----
    # ITK ResampleImageFilter map จุดจาก output space -> input space (pull-back)
    # ส่วน MRML transform ที่เราสร้างเป็น forward ทิศอาจต้องกลับด้าน
    if invert_transform == "auto" and aligned_meshes is not None and N > 0:
        sprint("  Probing transform direction (forward vs inverse) on subject 1...")
        probe_node = slicer.util.loadLabelVolume(file_list[0])
        mesh_b = aligned_meshes[0].GetBounds()
        scores = {}
        for tag, M in [("forward", T_matrices[0]),
                       ("inverse", np.linalg.inv(T_matrices[0]))]:
            o, st = _resample_with_matrix(probe_node, M, n_voxels, spacing_mm, half, interpolation)
            b, nvox = _occupied_bounds_ras(o, spacing_mm, half)
            score = _bounds_mismatch(b, mesh_b)
            scores[tag] = score
            if b is None:
                sprint(f"    [{tag}] ว่างเปล่า (0 voxels) -> ทิศนี้ผิดแน่นอน")
            else:
                sprint(f"    [{tag}] nonzero={nvox}, bounds X[{b[0]:+.3f},{b[1]:+.3f}] "
                       f"Y[{b[2]:+.3f},{b[3]:+.3f}] Z[{b[4]:+.3f},{b[5]:+.3f}], "
                       f"mismatch={score:.4f}")
            slicer.mrmlScene.RemoveNode(o)
        slicer.mrmlScene.RemoveNode(probe_node)
        sprint(f"    Mesh ref bounds     X[{mesh_b[0]:+.3f},{mesh_b[1]:+.3f}] "
               f"Y[{mesh_b[2]:+.3f},{mesh_b[3]:+.3f}] Z[{mesh_b[4]:+.3f},{mesh_b[5]:+.3f}]")
        use_inverse = scores["inverse"] < scores["forward"]
        sprint(f"  --> เลือกใช้ transform ทิศ: {'INVERSE' if use_inverse else 'FORWARD'}")
        if min(scores.values()) == float("inf"):
            sprint("  !!! ทั้งสองทิศให้ผลว่างเปล่า - transform หรือ grid ผิดอย่างหนัก")

    n_bad = 0
    for i, f in enumerate(file_list):
        basename = _strip_ext(f)
        sprint(f"Saving [{i+1}/{N}]: {basename}")
        node = slicer.util.loadLabelVolume(f)
        if node is None:
            sprint(f"    !!! โหลดไฟล์ไม่สำเร็จ ข้าม subject นี้")
            n_bad += 1
            continue

        M = np.linalg.inv(T_matrices[i]) if use_inverse else T_matrices[i]
        out_node, status = _resample_with_matrix(node, M, n_voxels, spacing_mm,
                                                 half, interpolation)

        if status != "Completed":
            sprint(f"    !!! ข้ามการบันทึก {basename} เพราะ resample ล้มเหลว")
            n_bad += 1
            slicer.mrmlScene.RemoveNode(node)
            slicer.mrmlScene.RemoveNode(out_node)
            continue

        # ---- ตรวจสอบผลลัพธ์เทียบกับ mesh ที่ align แล้ว (ground truth) ----
        b, nvox = _occupied_bounds_ras(out_node, spacing_mm, half)
        if b is None:
            sprint(f"    !!! {basename}: ผลลัพธ์ว่างเปล่า (0 voxels) - หลุดออกนอกกล่อง")
            n_bad += 1
        else:
            touching = (abs(abs(b[0]) - half) < spacing_mm or abs(abs(b[1]) - half) < spacing_mm or
                        abs(abs(b[2]) - half) < spacing_mm or abs(abs(b[3]) - half) < spacing_mm or
                        abs(abs(b[4]) - half) < spacing_mm or abs(abs(b[5]) - half) < spacing_mm)
            if touching:
                sprint(f"    !!! {basename}: ข้อมูลชนขอบกล่อง = ถูก CROP (เพิ่ม output_voxels "
                       f"หรือลด output_spacing)")
                n_bad += 1
            if aligned_meshes is not None:
                mb = np.array(aligned_meshes[i].GetBounds())

                # ผิว marching-cubes อยู่ห่างจากจุดศูนย์กลาง voxel ริมสุดออกไป "ครึ่ง voxel ต้นฉบับ"
                # ดังนั้น NIfTI ต้องเล็กกว่า mesh เสมอเป็นปกติ ไม่ใช่ความผิดพลาด
                in_sp = np.array(node.GetSpacing(), dtype=float)
                overall_scale = float(np.linalg.svd(T_matrices[i][:3, :3],
                                                    compute_uv=False).mean())
                half_in_vox = 0.5 * float(in_sp.max()) * overall_scale   # เป็นหน่วย normalized

                # ตรวจ 2 ทิศแยกกัน: ล้นออกนอก = บั๊กจริง / หดเข้าใน = discretization
                out_ex = max(float(mb[0] - b[0]), float(b[1] - mb[1]),
                             float(mb[2] - b[2]), float(b[3] - mb[3]),
                             float(mb[4] - b[4]), float(b[5] - mb[5]))
                in_sh = max(float(b[0] - mb[0]), float(mb[1] - b[1]),
                            float(b[2] - mb[2]), float(mb[3] - b[3]),
                            float(b[4] - mb[4]), float(mb[5] - b[5]))

                tol_out = 1.0 * spacing_mm                       # ล้นออกได้แค่ปัดเศษ
                tol_in = half_in_vox + 2.0 * spacing_mm          # หดเข้าได้ตามฟิสิกส์
                bad = False
                if out_ex > tol_out:
                    sprint(f"    !!! {basename}: NIfTI ล้นออกนอก mesh {out_ex/spacing_mm:.2f} voxel "
                           f"(> {tol_out/spacing_mm:.1f}) = สัญญาณ transform/scale ผิด")
                    sprint(f"        NIfTI X[{b[0]:+.3f},{b[1]:+.3f}] Y[{b[2]:+.3f},{b[3]:+.3f}] Z[{b[4]:+.3f},{b[5]:+.3f}]")
                    sprint(f"        Mesh  X[{mb[0]:+.3f},{mb[1]:+.3f}] Y[{mb[2]:+.3f},{mb[3]:+.3f}] Z[{mb[4]:+.3f},{mb[5]:+.3f}]")
                    bad = True
                elif in_sh > tol_in:
                    sprint(f"    ! {basename}: หดเข้า {in_sh/spacing_mm:.2f} voxel "
                           f"(คาด <= {tol_in/spacing_mm:.2f}) input spacing={in_sp} "
                           f"- ตรวจว่า subject นี้ภาพหยาบกว่าตัวอื่นหรือไม่")
                    bad = True
                if bad:
                    n_bad += 1
                else:
                    sprint(f"    OK: {nvox} voxels, หดเข้า {in_sh/spacing_mm:.2f} voxel "
                           f"(budget {tol_in/spacing_mm:.2f}), ล้นออก {out_ex/spacing_mm:.2f} voxel")

        out_path = os.path.join(out_vol_dir, f"{basename}_aligned.nii.gz")
        if not slicer.util.saveNode(out_node, out_path):
            n_bad += 1

        slicer.mrmlScene.RemoveNode(node)
        slicer.mrmlScene.RemoveNode(out_node)

    if n_bad:
        sprint(f"  !!! สรุป: มี {n_bad} subject ที่ผลลัพธ์น่าสงสัย - อ่าน log ด้านบน")
        raise RuntimeError(f'Aligned NIfTI quality check failed ({n_bad} checks); inspect the ICP log')
    else:
        sprint(f"  สรุป: ทั้ง {N} subject ผ่านการตรวจสอบ bounds เรียบร้อย")

OUTPUT_VOXELS = 128
# ข้อมูลหลัง normalize อยู่ในช่วง [-1, 1] -> spacing = 2/n_voxels ทำให้กล่องพอดีเป๊ะ
# (ของเดิม 0.02 ให้กล่อง +-1.28 ซึ่งเสียพื้นที่ ~22% ต่อด้าน)
OUTPUT_SPACING = 2.0 / OUTPUT_VOXELS
MAX_GW_ITERATIONS = 20
GW_TOLERANCE = 0.00005
PAIRWISE_ITERATIONS = 100
PAIRWISE_TOLERANCE = 0.0001
PAIRWISE_LANDMARKS = 200
INTERPOLATION_MODE = "nn"   # ต้องเป็น nn สำหรับ label map (ค่าที่รับ: nn/linear/ws/bs)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Group-wise rigid ICP alignment for NIfTI labels.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input_dir", default=None, help="Batch input directory containing NIfTI labels")
    input_group.add_argument("--input_file", default=None, help="Single NIfTI label file")
    parser.add_argument("--gui", action="store_true",
                        help="Open the ICP GUI to select mode, side, input, output, and reference")
    parser.add_argument("--output_dir", default=None, help="Output directory for aligned files")
    parser.add_argument("--reference_template", default=None, help="Fixed ICP template .vtk/.ply with matching .json metadata")
    parser.add_argument("--fit_reference", action="store_true", help="Build a new reference from TRAINING masks only; requires model retraining")
    parser.add_argument("--reference_side", choices=("left", "right"), default=None,
                        help="Side label for the archived fitted reference (otherwise inferred from input/output paths)")
    parser.add_argument("--reference_run_id", default=None,
                        help="Timestamped archive folder name; default ICP_reference_YYYYMMDD_HHMMSS")
    parser.add_argument("--reference_archive_root", default=DEFAULT_REFERENCE_ARCHIVE_ROOT,
                        help="Root folder for fitted-reference archives (default: ICP/Referen_ICP)")
    parser.add_argument("--exploratory_groupwise", action="store_true",
                        help="Explicitly allow a batch-dependent groupwise reference for exploratory analysis only")
    parser.add_argument("--output_spacing", type=float, default=None,
                        help="Output voxel spacing (ถ้าไม่ระบุ = 2.0/output_voxels ให้กล่องพอดี [-1,1])")
    parser.add_argument("--output_voxels", type=int, default=OUTPUT_VOXELS, help="Output volume dimension size")
    parser.add_argument("--max_iterations", type=int, default=MAX_GW_ITERATIONS, help="Max groupwise ICP iterations")
    parser.add_argument("--tolerance", type=float, default=GW_TOLERANCE, help="Groupwise convergence tolerance")
    parser.add_argument("--pairwise_iterations", type=int, default=PAIRWISE_ITERATIONS, help="Pairwise ICP max iterations")
    parser.add_argument("--pairwise_tolerance", type=float, default=PAIRWISE_TOLERANCE, help="Pairwise ICP tolerance")
    parser.add_argument("--pairwise_landmarks", type=int, default=PAIRWISE_LANDMARKS, help="Pairwise ICP landmarks count")
    parser.add_argument("--interpolation", type=str, default=INTERPOLATION_MODE, help="Resampling interpolation: nn (บังคับสำหรับ label), linear, ws, bs")
    parser.add_argument("--invert_transform", type=str, default="auto", choices=["auto", "yes", "no"],
                        help="ทิศของ transform ที่ส่งให้ ITK resample (auto = ทดสอบกับ subject แรกแล้วเลือกเอง)")
    args, _ = parser.parse_known_args(argv)
    if args.gui:
        return args
    alignment_modes = int(bool(args.reference_template)) + int(bool(args.fit_reference)) + int(bool(args.exploratory_groupwise))
    if alignment_modes != 1:
        parser.error('Choose exactly one of --reference_template, --fit_reference, or --exploratory_groupwise')
    if args.input_file:
        if not os.path.isfile(args.input_file):
            parser.error(f'--input_file does not exist: {args.input_file}')
        if not args.input_file.lower().endswith(('.nii.gz', '.nii', '.hdr', '.nrrd')):
            parser.error('--input_file must be a supported NIfTI/Nrrd volume')
        if not args.output_dir:
            parser.error('--output_dir is required with --input_file')
    if args.output_voxels < 8 or args.max_iterations < 1 or args.pairwise_iterations < 1:
        parser.error('Voxel size/iteration counts must be positive and valid')

    if args.output_spacing is None:
        args.output_spacing = 2.0 / float(args.output_voxels)
    return args


def align_to_fixed_reference(meshes, file_list, args):
    contract = load_reference_contract(args.reference_template)
    reader = vtk.vtkPLYReader() if args.reference_template.lower().endswith('.ply') else vtk.vtkPolyDataReader()
    reader.SetFileName(args.reference_template)
    reader.Update()
    reference = reader.GetOutput()
    if reference.GetNumberOfPoints() < 10:
        raise ValueError('Invalid reference mesh')
    scale = contract['physical_to_normalized_scale']
    args.output_spacing = contract['output_spacing']
    args.output_voxels = contract['output_voxels']
    aligned, transforms = [], []
    for mesh in meshes:
        initial = np.eye(4)
        initial[:3, :3] *= scale
        initial[:3, 3] = -get_points_numpy(mesh).mean(axis=0) * scale
        working = apply_poly_transform(mesh, initial)
        working, pca = principal_axis_align(working)
        working, flip, _ = pole_flip_correction(working, reference_poly=reference)
        rigid = run_vtk_icp(working, reference, max_iter=args.pairwise_iterations,
                            tolerance=args.pairwise_tolerance, landmarks=args.pairwise_landmarks)
        matrix = rigid @ flip @ pca @ initial
        transformed = apply_poly_transform(mesh, matrix)
        half = args.output_voxels * args.output_spacing / 2
        if max(abs(v) for v in transformed.GetBounds()) >= half:
            raise ValueError('Subject exceeds the fixed reference grid; do not rescale the patient batch')
        aligned.append(transformed)
        transforms.append(matrix)
    history = {'mode': 'fixed_reference', 'reference_template': args.reference_template,
               'reference_contract': contract, 'subject_names': [os.path.basename(f) for f in file_list],
               'rounds': [], 'mean_distances': []}
    return aligned, transforms, history

def find_input_files(input_dir):
    extensions = ["*.nii.gz", "*.nii", "*.hdr", "*.nrrd"]
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    file_list = sorted(list(set(file_list)))

    label_files = [f for f in file_list if "label" in os.path.basename(f).lower()]
    if label_files:
        file_list = label_files
        sprint(f"Prioritizing {len(file_list)} label files.")
    return file_list

def clean_previous_outputs(output_dir):
    for old_file in ["icp_convergence_history.json", "icp_convergence.png"]:
        old_path = os.path.join(output_dir, old_file)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
                sprint(f"  Cleaned up old output file: {old_file}")
            except Exception:
                pass

def step_load_and_mesh_all(file_list):
    sprint("Step A: Loading and meshing label volumes...")
    meshes = []
    valid_files = []
    for i, f in enumerate(file_list):
        sprint(f"  Loading [{i+1}/{len(file_list)}]: {os.path.basename(f)}")
        p = load_and_mesh_node(f)
        if p and p.GetNumberOfPoints() > 0:
            meshes.append(p)
            valid_files.append(f)
        else:
            sprint(f"  WARNING: Skipping {os.path.basename(f)} (no points)")
    return meshes, valid_files

def step1_per_mesh_normalize(meshes):
    sprint("Step 1: Per-mesh normalization (center + scale to |coord| <= 1)...")
    N = len(meshes)
    aligned_meshes = []
    T_initial = []
    S_initial = []
    for i in range(N):
        pts = get_points_numpy(meshes[i])
        centroid = pts.mean(axis=0)
        T_cent = np.eye(4)
        T_cent[:3, 3] = -centroid
        poly_centered = apply_poly_transform(meshes[i], T_cent)

        b = poly_centered.GetBounds()
        max_abs = max(abs(b[0]), abs(b[1]), abs(b[2]),
                      abs(b[3]), abs(b[4]), abs(b[5]))
        s = 1.0 / max_abs if max_abs > 0 else 1.0

        T_scale = np.eye(4)
        T_scale[0, 0] = T_scale[1, 1] = T_scale[2, 2] = s

        T_combined = T_scale @ T_cent
        aligned_meshes.append(apply_poly_transform(meshes[i], T_combined))
        T_initial.append(T_combined)
        S_initial.append(float(s))
        if i < 3 or i == N - 1:
            sprint(f"  Mesh {i+1}: centroid={centroid}, scale={s:.6f}")
    return aligned_meshes, T_initial, S_initial

def step2_pca_alignment(aligned_meshes):
    sprint("Step 2: Principal Axis Alignment (PCA)...")
    N = len(aligned_meshes)
    T_pca_list = []
    for i in range(N):
        aligned_meshes[i], T_p = principal_axis_align(aligned_meshes[i])
        T_pca_list.append(T_p)
    return T_pca_list

def step3_orientation_disambiguation(aligned_meshes, T_initial, T_pca_list):
    sprint("Step 3: Orientation disambiguation (proper rotations vs reference)...")
    N = len(aligned_meshes)
    T_flip = []
    aligned_meshes[0], tf0, flipped0 = pole_flip_correction(aligned_meshes[0], reference_poly=None)
    T_flip.append(tf0)
    sprint(f"  Mesh 1 (reference): {'REORIENTED' if flipped0 else 'OK'}")

    for i in range(1, N):
        aligned_meshes[i], tfi, flippedi = pole_flip_correction(
            aligned_meshes[i], reference_poly=aligned_meshes[0]
        )
        T_flip.append(tfi)
        sprint(f"  Mesh {i+1}: {'REORIENTED' if flippedi else 'OK'}")

    for i in range(N):
        T_initial[i] = T_flip[i] @ T_pca_list[i] @ T_initial[i]
    return T_initial

def step4_groupwise_icp(aligned_meshes, file_list, args):
    N = len(aligned_meshes)
    max_gw_iter = args.max_iterations
    gw_tolerance = args.tolerance
    sprint(f"Step 4: Groupwise ICP (rigid, max {max_gw_iter} rounds, tolerance={gw_tolerance})...")
    
    T_icp = [np.eye(4) for _ in range(N)]
    prev_mean_dist = float("inf")

    import time
    gw_start_time = time.time()

    gw_history = {
        "rounds": [],
        "elapsed_times_sec": [],
        "mean_distances": [],
        "dist_changes": [],
        "subject_names": [os.path.basename(f) for f in file_list],
        "subject_distances": {os.path.basename(f): [] for f in file_list}
    }

    for gw_iter in range(max_gw_iter):
        sprint(f"  [Groupwise Round {gw_iter+1}/{max_gw_iter}]")

        sprint(f"  [Round {gw_iter+1}] Pre-mean orientation re-check...")
        for i in range(1, N):
            aligned_meshes[i], tfi, flippedi = pole_flip_correction(
                aligned_meshes[i], reference_poly=aligned_meshes[0]
            )
            if flippedi:
                sprint(f"    NOTE: Mesh {i+1} reoriented before mean update")
                T_icp[i] = tfi @ T_icp[i]

        ref_mean = compute_mean_poly(aligned_meshes)

        pairwise_histories = []
        subj_pw_dict = {}
        for i in range(N):
            dT, p_hist = run_vtk_icp(aligned_meshes[i], ref_mean, return_history=True,
                                     max_iter=args.pairwise_iterations, tolerance=args.pairwise_tolerance,
                                     landmarks=args.pairwise_landmarks)
            aligned_meshes[i] = apply_poly_transform(aligned_meshes[i], dT)
            T_icp[i] = dT @ T_icp[i]
            pairwise_histories.append(p_hist)
            bname = os.path.basename(file_list[i])
            subj_pw_dict[bname] = [float(v) for v in p_hist]

        if gw_iter == 0:
            avg_pairwise = np.mean(pairwise_histories, axis=0).tolist()
            gw_history["pairwise_iterations"] = EVAL_PAIRWISE_STEPS
            gw_history["mean_pairwise_distances"] = [float(v) for v in avg_pairwise]
            gw_history["subject_pairwise_distances"] = subj_pw_dict

        subj_dists = [icp_distance(aligned_meshes[i], ref_mean) for i in range(N)]
        current_mean_dist = sum(subj_dists) / N
        sprint(f"  [Round {gw_iter+1}] Mean ICP dist to template: {current_mean_dist:.6f}")

        dist_change = abs(prev_mean_dist - current_mean_dist) if prev_mean_dist != float("inf") else 0.0
        t_elapsed = round(time.time() - gw_start_time, 2)
        gw_history["rounds"].append(gw_iter + 1)
        gw_history["elapsed_times_sec"].append(t_elapsed)
        gw_history["mean_distances"].append(float(current_mean_dist))
        gw_history["dist_changes"].append(float(dist_change))
        for i, f in enumerate(file_list):
            bname = os.path.basename(f)
            gw_history["subject_distances"][bname].append(float(subj_dists[i]))

        if gw_iter > 0 and dist_change <= gw_tolerance:
            sprint(f"  --> Groupwise ICP CONVERGED at round {gw_iter+1} (change: {dist_change:.6f} <= {gw_tolerance})")
            break
        prev_mean_dist = current_mean_dist

    return T_icp, gw_history

def step5_global_bounding_box_normalize(aligned_meshes, T_initial, T_icp, S_initial):
    sprint("Step 5: Global bounding-box normalization (preserving relative physical sizes)...")
    N = len(aligned_meshes)

    physical_aligned_meshes = []
    T_scale_back_list = []
    for i in range(N):
        # --- FIX: ห้ามอ่าน scale จาก T_initial[i][0,0] ---
        # step3 เขียนทับ T_initial[i] = T_flip @ T_pca @ (s*I) แล้ว
        # ทำให้ block 3x3 กลายเป็น s*R ดังนั้น [0,0] = s*R[0,0] ไม่ใช่ s
        # (R[0,0] ติดลบได้ -> scale ติดลบ -> เมชถูก mirror)
        s = S_initial[i]

        # ตรวจความถูกต้องของเมทริกซ์สะสม: ต้องเป็น similarity (scale เท่ากันทุกแกน, det > 0)
        M = T_initial[i][:3, :3]
        sv = np.linalg.svd(M, compute_uv=False)
        det = float(np.linalg.det(M))
        if det <= 0:
            sprint(f"  !!! WARNING: Mesh {i+1} T_initial has det={det:+.6f} (reflection!) "
                   f"- ตรวจสอบ PCA / flip candidates")
        if sv.mean() > 0 and (sv.max() - sv.min()) / sv.mean() > 1e-6:
            sprint(f"  !!! WARNING: Mesh {i+1} scale ไม่ uniform: singular values = {sv}")
        if abs(sv.mean() - s) / max(s, 1e-12) > 1e-6:
            sprint(f"  !!! WARNING: Mesh {i+1} scale mismatch: stored={s:.6g} vs matrix={sv.mean():.6g}")

        T_scale_back = np.eye(4)
        T_scale_back[0, 0] = T_scale_back[1, 1] = T_scale_back[2, 2] = 1.0 / s
        T_scale_back_list.append(T_scale_back)
        m_phys = apply_poly_transform(aligned_meshes[i], T_scale_back)
        physical_aligned_meshes.append(m_phys)

    g_min = np.array([float('inf')] * 3)
    g_max = np.array([float('-inf')] * 3)
    for m in physical_aligned_meshes:
        b = m.GetBounds()
        g_min[0] = min(g_min[0], b[0]); g_max[0] = max(g_max[0], b[1])
        g_min[1] = min(g_min[1], b[2]); g_max[1] = max(g_max[1], b[3])
        g_min[2] = min(g_min[2], b[4]); g_max[2] = max(g_max[2], b[5])
        
    sprint(f"  Physical Union bounds: "
           f"X[{g_min[0]:+.4f},{g_max[0]:+.4f}]  "
           f"Y[{g_min[1]:+.4f},{g_max[1]:+.4f}]  "
           f"Z[{g_min[2]:+.4f},{g_max[2]:+.4f}]")

    g_center = (g_min + g_max) / 2.0
    g_half = (g_max - g_min) / 2.0
    max_half = float(g_half.max())
    # Reserve a 10% border for voxelization instead of touching the output grid.
    global_scale = 0.9 / max_half if max_half > 0 else 1.0

    T_g_cent = np.eye(4)
    T_g_cent[:3, 3] = -g_center
    T_g_scale = np.eye(4)
    T_g_scale[0, 0] = T_g_scale[1, 1] = T_g_scale[2, 2] = global_scale
    T_global = T_g_scale @ T_g_cent
    sprint(f"  Global center = ({g_center[0]:+.4f}, {g_center[1]:+.4f}, {g_center[2]:+.4f})")
    sprint(f"  Global scale  = {global_scale:.6f}  "
           f"(max physical half-extent {max_half:.4f} -> 1.0)")

    for i in range(N):
        aligned_meshes[i] = apply_poly_transform(physical_aligned_meshes[i], T_global)

    v_min = np.array([float('inf')] * 3)
    v_max = np.array([float('-inf')] * 3)
    for m in aligned_meshes:
        b = m.GetBounds()
        v_min[0] = min(v_min[0], b[0]); v_max[0] = max(v_max[0], b[1])
        v_min[1] = min(v_min[1], b[2]); v_max[1] = max(v_max[1], b[3])
        v_min[2] = min(v_min[2], b[4]); v_max[2] = max(v_max[2], b[5])
    sprint(f"  Union bounds (after) : "
           f"X[{v_min[0]:+.4f},{v_max[0]:+.4f}]  "
           f"Y[{v_min[1]:+.4f},{v_max[1]:+.4f}]  "
           f"Z[{v_min[2]:+.4f},{v_max[2]:+.4f}]")

    T_matrices = [T_global @ T_scale_back_list[i] @ T_icp[i] @ T_initial[i] for i in range(N)]
    return T_matrices

def step6_save_outputs(output_dir, file_list, aligned_meshes, T_matrices, gw_history, args):
    sprint("Step 6: Saving results...")
    N = len(file_list)
    np.save(os.path.join(output_dir, "T_matrices.npy"), np.array(T_matrices))
    sprint(f"  Saved T_matrices.npy ({N} matrices)")

    import json
    history_json_path = os.path.join(output_dir, "icp_convergence_history.json")
    with open(history_json_path, "w") as f:
        json.dump(gw_history, f, indent=2)
    sprint("  Saved icp_convergence_history.json")

    derived_mean_shape = None
    if args.reference_template:
        sprint("  Fixed reference is active; skipping a batch-derived mean_shape.ply to keep it distinct from the template.")
    else:
        mean_poly = compute_mean_poly(aligned_meshes)
        writer = vtk.vtkPLYWriter()
        derived_mean_shape = os.path.join(output_dir, "mean_shape.ply")
        writer.SetFileName(derived_mean_shape)
        writer.SetInputData(mean_poly)
        writer.Write()
        sprint("  Saved mean_shape.ply")

    # Save individual aligned meshes for each subject
    aligned_mesh_dir = os.path.join(output_dir, "aligned_meshes")
    os.makedirs(aligned_mesh_dir, exist_ok=True)
    for i, orig_f in enumerate(file_list):
        bn = os.path.basename(orig_f)
        for ext in [".nii.gz", ".nii", ".mgz"]:
            if bn.endswith(ext):
                bn = bn[:-len(ext)]
                break
        mesh_out = os.path.join(aligned_mesh_dir, f"{bn}_aligned.vtk")
        writer_m = vtk.vtkPolyDataWriter()
        writer_m.SetFileName(mesh_out)
        writer_m.SetInputData(aligned_meshes[i])
        writer_m.Write()
    sprint(f"  Saved {N} aligned meshes to: {aligned_mesh_dir}")

    export_aligned_nifti(file_list, T_matrices, output_dir,
                         aligned_meshes=aligned_meshes,
                         spacing_mm=args.output_spacing,
                         n_voxels=args.output_voxels,
                         interpolation=args.interpolation,
                         invert_transform=args.invert_transform)
    if args.fit_reference:
        physical_scale = float(np.linalg.svd(T_matrices[0][:3, :3], compute_uv=False).mean())
        write_reference_contract(os.path.join(output_dir, 'mean_shape.ply'), physical_scale,
                                 file_list, args.output_spacing, args.output_voxels)
    mode = ('fixed_reference' if args.reference_template else
            'fit_reference' if args.fit_reference else 'exploratory_groupwise')
    reference_sha256 = None
    reference_path = args.reference_template
    reference_contract = None
    if args.reference_template:
        reference_contract = load_reference_contract(args.reference_template)
        reference_sha256 = reference_contract['template_sha256']
    elif args.fit_reference:
        reference_path = os.path.join(output_dir, 'mean_shape.ply')
        contract_path = reference_path + '.json'
        if os.path.isfile(contract_path):
            with open(contract_path, 'r', encoding='utf-8') as contract_stream:
                reference_contract = json.load(contract_stream)
                reference_sha256 = reference_contract.get('template_sha256')
    geometry_version = (
        reference_contract.get('version') if reference_contract else
        'exploratory-groupwise-v1'
    )
    status_path = os.path.join(output_dir, 'icp_status.json')
    with open(status_path, 'w', encoding='utf-8') as stream:
        json.dump({'success': True, 'subjects': len(file_list),
                   'geometry_version': geometry_version,
                   'reference_template': reference_path,
                   'reference_sha256': reference_sha256,
                   'mode': mode,
                   'reference_origin': (reference_contract or {}).get('reference_origin'),
                   'derived_mean_shape': derived_mean_shape,
                   'independent_test_reference': (
                       (reference_contract or {}).get('independent_test_reference',
                           (reference_contract or {}).get('version') == 'fixed-training-reference-v1')
                       if reference_contract else None
                   ),
                   'parameters': {
                       'output_voxels': int(args.output_voxels),
                       'output_spacing': float(args.output_spacing),
                       'max_groupwise_iterations': int(args.max_iterations),
                       'groupwise_tolerance': float(args.tolerance),
                       'pairwise_iterations': int(args.pairwise_iterations),
                       'pairwise_tolerance': float(args.pairwise_tolerance),
                       'pairwise_landmarks': int(args.pairwise_landmarks),
                       'interpolation': str(args.interpolation),
                       'invert_transform': str(args.invert_transform),
                   }}, stream)

    if args.fit_reference:
        try:
            archive_dir, manifest_path = _archive_fitted_reference(output_dir, args, reference_path)
            # Keep the source status self-describing without making the archive
            # depend on a second command or a separate post-processing script.
            with open(status_path, 'r', encoding='utf-8') as stream:
                status_payload = json.load(stream)
            status_payload['reference_archive_dir'] = archive_dir
            status_payload['reference_manifest'] = manifest_path
            with open(status_path, 'w', encoding='utf-8') as stream:
                json.dump(status_payload, stream, indent=2)
            # Refresh the archived copy after adding its final provenance
            # fields, so source and archived status files agree.
            shutil.copy2(status_path, os.path.join(archive_dir, 'icp_status.json'))
            sprint(f"  Archived fitted reference: {archive_dir}")
            sprint(f"  Reference manifest: {manifest_path}")
        except Exception as archive_error:
            # The alignment output remains available for debugging, but a
            # failed archive must not be mistaken for a usable reference.
            try:
                with open(status_path, 'r', encoding='utf-8') as stream:
                    failed_status = json.load(stream)
                failed_status['success'] = False
                failed_status['state'] = 'reference_archive_failed'
                failed_status['error'] = str(archive_error)
                with open(status_path, 'w', encoding='utf-8') as stream:
                    json.dump(failed_status, stream, indent=2)
            finally:
                raise

    sprint(f"  All {N} aligned NIfTI saved to: {os.path.join(output_dir, 'aligned_nifti')}")
    sprint("--- ICP.py FINISHED ---")


def launch_gui():
    """Open a self-contained Qt selector so ICP.py can be run by itself."""
    try:
        import qt
    except ImportError as error:
        print("[ERROR] ICP GUI requires SlicerSALT's Qt environment: {}".format(error))
        return 1

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setWindowTitle("Hippocampal ICP")
    dialog.resize(760, 520)

    layout = qt.QVBoxLayout(dialog)
    title = qt.QLabel("ICP alignment and reference builder")
    title.setStyleSheet("font-size: 18px; font-weight: bold;")
    layout.addWidget(title)
    description = qt.QLabel(
        "เลือกโหมดการทำงาน ฝั่งข้อมูล และไฟล์ที่ต้องใช้ จากนั้นกด เริ่ม ICP\n"
        "โหมดสร้าง reference ควรใช้ข้อมูล training เท่านั้น"
    )
    description.setWordWrap(True)
    layout.addWidget(description)

    form = qt.QFormLayout()
    layout.addLayout(form)

    mode_box = qt.QComboBox()
    mode_box.addItem("สร้าง reference ใหม่จาก training (--fit_reference)")
    mode_box.addItem("ใช้ reference เดิม (--reference_template)")
    mode_box.addItem("ทดลอง groupwise (--exploratory_groupwise)")
    form.addRow("โหมด:", mode_box)

    side_box = qt.QComboBox()
    side_box.addItems(["left", "right"])
    form.addRow("ฝั่ง:", side_box)

    input_mode_box = qt.QComboBox()
    input_mode_box.addItems(["โฟลเดอร์ batch", "ไฟล์ mask เดี่ยว"])
    form.addRow("ชนิด input:", input_mode_box)

    input_edit = qt.QLineEdit()
    input_button = qt.QPushButton("เลือก...")
    input_row = qt.QHBoxLayout()
    input_row.addWidget(input_edit)
    input_row.addWidget(input_button)
    form.addRow("Input:", input_row)

    output_edit = qt.QLineEdit()
    output_button = qt.QPushButton("เลือก...")
    output_row = qt.QHBoxLayout()
    output_row.addWidget(output_edit)
    output_row.addWidget(output_button)
    form.addRow("Output (เว้นว่างใช้ค่าอัตโนมัติ):", output_row)

    reference_edit = qt.QLineEdit()
    reference_button = qt.QPushButton("เลือก...")
    reference_row = qt.QHBoxLayout()
    reference_row.addWidget(reference_edit)
    reference_row.addWidget(reference_button)
    form.addRow("Reference .ply:", reference_row)

    run_id_edit = qt.QLineEdit(datetime.now().strftime("ICP_reference_%Y%m%d_%H%M%S"))
    form.addRow("Reference run ID:", run_id_edit)

    archive_edit = qt.QLineEdit(DEFAULT_REFERENCE_ARCHIVE_ROOT)
    archive_button = qt.QPushButton("เลือก...")
    archive_row = qt.QHBoxLayout()
    archive_row.addWidget(archive_edit)
    archive_row.addWidget(archive_button)
    form.addRow("Reference archive root:", archive_row)

    note = qt.QLabel(
        "ไฟล์ที่สร้าง reference จะถูกเก็บใน ICP/Referen_ICP/<run_id>/<left หรือ right>/"
    )
    note.setWordWrap(True)
    layout.addWidget(note)

    buttons = qt.QHBoxLayout()
    buttons.addStretch(1)
    cancel_button = qt.QPushButton("ยกเลิก")
    run_button = qt.QPushButton("เริ่ม ICP")
    run_button.setDefault(True)
    buttons.addWidget(cancel_button)
    buttons.addWidget(run_button)
    layout.addLayout(buttons)

    selected_argv = [None]

    def selected_text(widget):
        return str(widget.text()).strip()

    def choose_input():
        if input_mode_box.currentIndex() == 0:
            selected = qt.QFileDialog.getExistingDirectory(dialog, "เลือกโฟลเดอร์ input mask")
        else:
            selected = qt.QFileDialog.getOpenFileName(
                dialog, "เลือกไฟล์ input mask", "",
                "Mask volumes (*.nii *.nii.gz *.nrrd *.hdr);;All files (*)"
            )
        if isinstance(selected, (tuple, list)):
            selected = selected[0]
        if selected:
            input_edit.setText(str(selected))

    def choose_output():
        selected = qt.QFileDialog.getExistingDirectory(dialog, "เลือกโฟลเดอร์ output")
        if selected:
            output_edit.setText(str(selected))

    def choose_reference():
        selected = qt.QFileDialog.getOpenFileName(
            dialog, "เลือก ICP reference", "",
            "ICP reference (*.ply *.vtk);;All files (*)"
        )
        if isinstance(selected, (tuple, list)):
            selected = selected[0]
        if selected:
            reference_edit.setText(str(selected))

    def choose_archive():
        selected = qt.QFileDialog.getExistingDirectory(dialog, "เลือกโฟลเดอร์เก็บ reference")
        if selected:
            archive_edit.setText(str(selected))

    def update_controls(*_args):
        fixed = mode_box.currentIndex() == 1
        fit = mode_box.currentIndex() == 0
        reference_edit.setEnabled(fixed)
        reference_button.setEnabled(fixed)
        run_id_edit.setEnabled(fit)
        archive_edit.setEnabled(fit)
        archive_button.setEnabled(fit)

    def show_error(message):
        qt.QMessageBox.critical(dialog, "ข้อมูลไม่ครบ", str(message))

    def start_run():
        input_path = selected_text(input_edit)
        if not input_path:
            show_error("กรุณาเลือก input")
            return
        mode_index = mode_box.currentIndex()
        if mode_index == 1 and not selected_text(reference_edit):
            show_error("โหมดใช้ reference เดิมต้องเลือกไฟล์ reference .ply")
            return
        if not os.path.exists(input_path):
            show_error("ไม่พบ input: {}".format(input_path))
            return
        if mode_index == 1 and not os.path.isfile(selected_text(reference_edit)):
            show_error("ไม่พบ reference: {}".format(selected_text(reference_edit)))
            return

        argv = []
        if input_mode_box.currentIndex() == 0:
            argv.extend(["--input_dir", input_path])
        else:
            argv.extend(["--input_file", input_path])
        if selected_text(output_edit):
            argv.extend(["--output_dir", selected_text(output_edit)])
        if mode_index == 0:
            argv.extend([
                "--fit_reference",
                "--reference_side", str(side_box.currentText()),
                "--reference_run_id", selected_text(run_id_edit),
                "--reference_archive_root", selected_text(archive_edit) or DEFAULT_REFERENCE_ARCHIVE_ROOT,
            ])
        elif mode_index == 1:
            argv.extend(["--reference_template", selected_text(reference_edit)])
        else:
            argv.append("--exploratory_groupwise")
        selected_argv[0] = argv
        dialog.accept()

    input_button.clicked.connect(choose_input)
    output_button.clicked.connect(choose_output)
    reference_button.clicked.connect(choose_reference)
    archive_button.clicked.connect(choose_archive)
    mode_box.currentIndexChanged.connect(update_controls)
    cancel_button.clicked.connect(dialog.reject)
    run_button.clicked.connect(start_run)
    update_controls()

    exec_method = getattr(dialog, "exec_", None) or getattr(dialog, "exec")
    result = exec_method()
    accepted_value = getattr(qt.QDialog, "Accepted", 1)
    if result != accepted_value or selected_argv[0] is None:
        print("[INFO] ICP GUI cancelled.")
        return 0
    return main(selected_argv[0])


def main(argv=None):
    args = parse_args(argv)
    if args.gui:
        return launch_gui()
    input_dir = args.input_dir
    input_file = args.input_file
    if not input_dir and not input_file:
        sprint("No input given. Opening folder picker...")
        input_dir = prompt_folder("Select input folder containing NIfTI labels")
        if not input_dir:
            sprint("ERROR: No folder selected. Exiting.")
            return

    output_dir = args.output_dir
    if not output_dir:
        basename = os.path.basename(os.path.normpath(input_dir))
        output_dir = os.path.join(SCRIPT_DIR, f"output_{basename}")
        sprint(f"No --output_dir given. Using default: {output_dir}")

    if input_dir:
        input_dir = input_dir.replace("\\", "/")
    if input_file:
        input_file = input_file.replace("\\", "/")
    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)
    global DEBUG_LOG
    DEBUG_LOG = os.path.join(output_dir, "icp_debug_log.txt")
    with open(DEBUG_LOG, "w", encoding="utf-8") as f:
        f.write(f"--- LOG START: {datetime.now()} ---\n")
    sprint("--- ICP.py STARTING (rigid ICP) ---")
    if args.exploratory_groupwise:
        sprint('[WARNING] Explicit exploratory groupwise alignment: the reference depends on this entire input batch.')
    sprint(f"Input  dir: {input_dir or '(single-file mode)'}")
    if input_file:
        sprint(f"Input file: {input_file}")
    sprint(f"Output dir: {output_dir}")

    clean_previous_outputs(output_dir)
    status_path = os.path.join(output_dir, 'icp_status.json')
    with open(status_path, 'w', encoding='utf-8') as stream:
        json.dump({'success': False, 'state': 'running'}, stream)

    file_list = [input_file] if input_file else find_input_files(input_dir)
    sprint(f"Total files to process: {len(file_list)}")
    if not file_list:
        raise ValueError('No files found in input_dir')

    meshes, valid_files = step_load_and_mesh_all(file_list)
    N = len(meshes)
    if N == 0:
        raise ValueError('No valid label meshes found')
    if N != len(file_list):
        raise RuntimeError(f'Only {N}/{len(file_list)} input masks loaded; refusing partial alignment')
    file_list = valid_files
    sprint(f"  Loaded {N} meshes successfully.")
    if args.reference_template:
        aligned_meshes, T_matrices, history = align_to_fixed_reference(meshes, file_list, args)
        step6_save_outputs(output_dir, file_list, aligned_meshes, T_matrices, history, args)
        return

    aligned_meshes, T_initial, S_initial = step1_per_mesh_normalize(meshes)
    T_pca_list = step2_pca_alignment(aligned_meshes)
    T_initial = step3_orientation_disambiguation(aligned_meshes, T_initial, T_pca_list)
    T_icp, gw_history = step4_groupwise_icp(aligned_meshes, file_list, args)
    T_matrices = step5_global_bounding_box_normalize(aligned_meshes, T_initial, T_icp, S_initial)
    step6_save_outputs(output_dir, file_list, aligned_meshes, T_matrices, gw_history, args)

if __name__ == "__main__":
    import sys, os
    exit_code = 0
    try:
        # Running `python ICP.py` with no options opens the selector.  A
        # command-line invocation with options remains fully non-interactive.
        if len(sys.argv) == 1:
            exit_code = launch_gui()
        else:
            exit_code = main()
    except Exception as e:
        import traceback
        sprint(f"FATAL ERROR: {str(e)}")
        with open(DEBUG_LOG, "a") as f:
            f.write(traceback.format_exc())
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            # Request a normal Slicer shutdown first.  Some SlicerSALT builds
            # block inside slicer.util.exit(), so do not wait on that call.
            if getattr(slicer, "app", None) is not None:
                slicer.app.quit()
        except Exception:
            pass
        try:
            # ICP.py is a command-line batch worker.  Once the status and
            # output files are flushed, terminate the worker so the parent
            # Python/PowerShell command cannot remain blocked in Slicer.
            os._exit(exit_code)
        except Exception:
            pass
