import os
import re
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal

from .vtk_viewer import VtkViewer

class RightPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_template_toggled = pyqtSignal(bool)
    signal_overlay_toggled = pyqtSignal(bool)
    signal_step_sd = pyqtSignal(float)
    signal_reset_sd = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        right_layout = QVBoxLayout(self)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        self.viewer = VtkViewer(self)
        right_layout.addWidget(self.viewer)
        
        self.viewer.signal_log_message.connect(self.signal_log_message)
        self.viewer.signal_template_toggled.connect(self.signal_template_toggled)
        self.viewer.signal_overlay_toggled.connect(self.signal_overlay_toggled)
        self.viewer.signal_step_sd.connect(self.signal_step_sd.emit)
        self.viewer.signal_reset_sd.connect(self.signal_reset_sd.emit)

    def set_view_mode(self, mode: str, module_name: str = ""):
        self.viewer.set_view_mode(mode, module_name)

    def set_overlay_visible(self, visible: bool):
        self.viewer.set_overlay_visible(visible)

    def set_side_filter(self, side_filter: str):
        self.viewer.set_side_filter(side_filter)

    def set_diagnostic_info(self, info_html: str):
        self.viewer.set_diagnostic_info(info_html)

    def display_all_meshes(self, filepaths, side_filter="all"):
        self.signal_log_message.emit(f"Superimposing {len(filepaths)} meshes in 3D View (Filter: {side_filter.upper()})")
        self.viewer.display_all_meshes(filepaths, side_filter=side_filter)

    def display_subject(self, filepath):
        self.viewer.set_mesh_view_visible(False)
        self.set_view_mode("quad", "Data Importer")
        self.viewer.set_3d_plane_buttons_visible(False)
        if hasattr(self.viewer, 'slice_mgr'):
            self.viewer.slice_mgr.hide_3d_planes(uncheck_buttons=True)
        if not filepath or not os.path.isfile(filepath):
            if hasattr(self.viewer, 'slice_mgr'):
                self.viewer.slice_mgr.clear_views()
            return

        self.signal_log_message.emit(f"Displaying subject: {os.path.basename(filepath)}")
        self.viewer.display_subject(filepath)

    def display_mesh(self, filepath, side_filter="all"):
        if isinstance(filepath, list):
            if not filepath:
                self.viewer.display_mesh("", side_filter=side_filter)
                self.signal_log_message.emit("[INFO] Deselected all meshes from 3D view.")
                return
            elif len(filepath) == 1:
                filepath = filepath[0]
            else:
                self.signal_log_message.emit(f"Displaying {len(filepath)} selected meshes in 3D View")
                self.viewer.display_all_meshes(filepath, side_filter=side_filter)
                return

        if not filepath:
            self.viewer.display_mesh("", side_filter=side_filter)
            self.signal_log_message.emit("[INFO] Deselected mesh from 3D view.")
            return

        # Check current module or mesh type: if Main Panel, ICP, SPHARM, or Result Panel, enforce full_3d
        current_mod = getattr(self.viewer, 'current_module_name', '')
        is_3d_only_mesh = ("spharm" in filepath.lower()) or ("_aligned" in filepath.lower()) or ("gradcam" in filepath.lower())
        if "main" in str(current_mod).lower() or current_mod in ("Main Panel", "ICP Registration", "SPHARM Processing", "Result Panel") or is_3d_only_mesh:
            target_mod = current_mod if current_mod in ("Main Panel", "ICP Registration", "SPHARM Processing", "Result Panel") else "Main Panel"
            if getattr(self.viewer, 'view_mode', '') != "full_3d":
                self.set_view_mode("full_3d", target_mod)

        self.signal_log_message.emit(f"Displaying 3D Mesh: {os.path.basename(filepath)}")
        self.viewer.display_mesh(filepath, side_filter=side_filter)
        
        # If in full_3d mode (Main Panel / ICP / SPHARM / Result), skip 2D MRI slice search and overlay
        if getattr(self.viewer, 'view_mode', 'quad') == "full_3d":
            self.signal_log_message.emit(f"[INFO] 3D mesh rendered for {os.path.basename(filepath)} in Full 3D View.")
            return

        if "fastsurfer" in str(current_mod).lower():
            self.viewer.set_3d_plane_buttons_visible(True)
        else:
            self.viewer.set_3d_plane_buttons_visible(False)
            if hasattr(self.viewer, 'slice_mgr'):
                self.viewer.slice_mgr.hide_3d_planes(uncheck_buttons=True)
        
        # Infer output directory and subject id from the mesh filepath
        # Filepath looks like: .../output_dir/left_hippocampus/lh_sub-XXXX_hippocampus.nii.gz
        parent_dir = os.path.dirname(os.path.abspath(filepath))
        output_dir = os.path.dirname(parent_dir)
        filename = os.path.basename(filepath)
        
        subject_id = None
        clean_name = filename
        for sfx in ["_aligned_SPHARM.vtk", "_SPHARM.vtk", "_aligned.vtk", "_hippocampus.nii.gz", "_hippocampus_lh.nii.gz", "_hippocampus_rh.nii.gz", ".nii.gz", ".vtk"]:
            if clean_name.endswith(sfx):
                clean_name = clean_name[:-len(sfx)]
                break

        if clean_name.startswith("lh_") or clean_name.startswith("rh_"):
            clean_name = clean_name[3:]
        if clean_name.endswith("_lh") or clean_name.endswith("_rh"):
            clean_name = clean_name[:-3]
            
        if clean_name:
            subject_id = clean_name
            
        if subject_id:
            # Search for matching conformed MRI (.nii.gz or .mgz)
            mri_file = None
            search_bases = [output_dir, parent_dir, os.path.dirname(output_dir), os.path.join(output_dir, "fastsurfer")]
            
            for base_d in search_bases:
                if not os.path.isdir(base_d):
                    continue
                # 1. Check for conformed NIfTI MRI in mri folder
                nii_candidate = os.path.join(base_d, "mri", f"{subject_id}_t1.nii.gz")
                if os.path.isfile(nii_candidate):
                    mri_file = nii_candidate
                    break
                
                # 2. Check for orig.mgz in fastsurfer_temp
                mgz_candidates = [
                    os.path.join(base_d, "fastsurfer_temp", subject_id, "mri", "orig.mgz"),
                    os.path.join(base_d, subject_id, "mri", "orig.mgz"),
                ]
                for mgz_c in mgz_candidates:
                    if os.path.isfile(mgz_c):
                        # Convert mgz to NIfTI on the fly for VTK
                        try:
                            import nibabel as nib
                            import numpy as np
                            mri_dir = os.path.join(base_d, "mri")
                            os.makedirs(mri_dir, exist_ok=True)
                            nii_dst = os.path.join(mri_dir, f"{subject_id}_t1.nii.gz")
                            if not os.path.exists(nii_dst):
                                img = nib.load(mgz_c)
                                nii_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
                                nib.save(nii_img, nii_dst)
                            mri_file = nii_dst
                            break
                        except Exception as e:
                            print(f"[ERROR] Could not convert mgz: {e}")
                if mri_file:
                    break
                    
            # Search for LH and RH mask files
            lh_mask = None
            rh_mask = None
            for base_d in search_bases:
                if not os.path.isdir(base_d):
                    continue
                lh_candidates = [
                    os.path.join(base_d, "left_hippocampus", f"lh_{subject_id}_hippocampus.nii.gz"),
                    os.path.join(base_d, "left_hippocampus", f"{subject_id}_hippocampus_lh.nii.gz"),
                    os.path.join(base_d, f"lh_{subject_id}_hippocampus.nii.gz"),
                    os.path.join(base_d, f"{subject_id}_hippocampus_lh.nii.gz"),
                ]
                rh_candidates = [
                    os.path.join(base_d, "right_hippocampus", f"rh_{subject_id}_hippocampus.nii.gz"),
                    os.path.join(base_d, "right_hippocampus", f"{subject_id}_hippocampus_rh.nii.gz"),
                    os.path.join(base_d, f"rh_{subject_id}_hippocampus.nii.gz"),
                    os.path.join(base_d, f"{subject_id}_hippocampus_rh.nii.gz"),
                ]
                if not lh_mask:
                    for c in lh_candidates:
                        if os.path.exists(c):
                            lh_mask = c
                            break
                if not rh_mask:
                    for c in rh_candidates:
                        if os.path.exists(c):
                            rh_mask = c
                            break
            
            # If the currently selected file is directly one of the masks, prioritize it
            if filename.startswith("lh_") or "left_hippocampus" in filepath:
                lh_mask = filepath
            elif filename.startswith("rh_") or "right_hippocampus" in filepath:
                rh_mask = filepath
                
            if mri_file:
                self.viewer.display_segmentation_overlays(mri_file, lh_mask, rh_mask, side_filter=side_filter)
            else:
                self.signal_log_message.emit(f"[INFO] 3D mesh rendered for {subject_id}.")

    def display_gradcam_mesh(self, mesh_path, scalar_mode="GradCAM_Importance", lut_type="gradcam", title="Grad-CAM Attention", side="left", opacity=1.0):
        self.viewer.display_gradcam_mesh(mesh_path, scalar_mode=scalar_mode, lut_type=lut_type, title=title, side=side, opacity=opacity)

    def set_patient_overlay(self, mesh_path, visible=True, opacity=0.35, side="left"):
        self.viewer.set_patient_overlay(mesh_path, visible=visible, opacity=opacity, side=side)

    def clear_gradcam_view(self, render_now=True):
        self.viewer.clear_gradcam_view(render_now=render_now)

    def reset_3d_camera(self, side=None):
        self.viewer.reset_3d_camera(side=side)

