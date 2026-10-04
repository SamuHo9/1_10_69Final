import os
import glob
import re
from PyQt6.QtCore import Qt

class LatentShapeExplorer:
    """
    Manages interactive latent shape space navigation along PLS-DA components
    (-3SD to +3SD), dynamic step mesh resolution, and 3D viewport synchronization.
    """
    def __init__(self, panel):
        self.p = panel

    def get_repo_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def find_step_mesh(self, side: str, sd_val: float, cohort: str = "All_Augment_tain", component: str = "PLS1"):
        repo_root = self.get_repo_root()
        base_dir = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", cohort, side, component)
        steps_dir = os.path.join(base_dir, "steps")
        if os.path.isdir(steps_dir):
            files = glob.glob(os.path.join(steps_dir, "*.vtk"))
            best_f = None
            best_diff = float('inf')
            for f in files:
                m = re.search(r"val_([+-]?\d+\.?\d*)", os.path.basename(f))
                if m:
                    diff = abs(float(m.group(1)) - sd_val)
                    if diff < best_diff:
                        best_diff = diff
                        best_f = f
            if best_f and best_diff < 0.08:
                return best_f

        # Fallback to standard milestone VTK if steps folder is absent
        if sd_val <= -2.5:
            milestone = "minus3SD"
        elif sd_val <= -1.5:
            milestone = "minus2SD"
        elif sd_val <= -0.5:
            milestone = "minus1SD"
        elif sd_val < 0.5:
            milestone = "Mean"
        elif sd_val < 1.5:
            milestone = "plus1SD"
        elif sd_val < 2.5:
            milestone = "plus2SD"
        else:
            milestone = "plus3SD"

        milestone_path = os.path.join(base_dir, f"{component}_{milestone}.vtk")
        if os.path.isfile(milestone_path):
            return milestone_path
        return None

    def step_sd(self, delta: float):
        cur_val = self.p.current_sd_value
        new_val = max(-3.0, min(3.0, round(cur_val + delta, 1)))
        if abs(new_val - cur_val) > 0.01:
            self.set_sd_value(new_val)

    def set_sd_value(self, sd_val: float):
        sd_val = max(-3.0, min(3.0, round(sd_val, 1)))
        step_idx = int(round((sd_val + 3.0) * 10))
        self.p.sd_slider.blockSignals(True)
        self.p.sd_slider.setValue(step_idx)
        self.p.sd_slider.blockSignals(False)
        self.p.current_sd_step_idx = step_idx
        self.p.current_sd_value = sd_val
        self.update_sd_badge(sd_val, step_idx)
        self.update_3d_view()

    def on_sd_slider_changed(self, val: int):
        sd_val = round(-3.0 + val * 0.1, 1)
        self.p.current_sd_step_idx = val
        self.p.current_sd_value = sd_val
        self.update_sd_badge(sd_val, val)
        self.update_3d_view()

    def update_sd_badge(self, sd_val: float, step_idx: int):
        if abs(sd_val) < 0.05:
            desc = "Mean"
            color = "#2980b9"
            bg = "#ebf5fb"
            border = "#aed6f1"
        elif sd_val <= -2.0:
            desc = "Severe Atrophy" if sd_val <= -2.5 else "Atrophy"
            color = "#c0392b"
            bg = "#fdf2f2"
            border = "#f5b7b1"
        elif sd_val < 0.0:
            desc = "Mild Atrophy"
            color = "#d35400"
            bg = "#fef5e7"
            border = "#fad7a0"
        else:
            desc = "Expansion"
            color = "#27ae60"
            bg = "#f2fbf6"
            border = "#abebc6"
        self.p.sd_val_badge.setText(f"{sd_val:+.1f} SD ({desc})")
        self.p.sd_val_badge.setStyleSheet(f"""
            QLabel {{
                background: {bg};
                color: {color};
                font-weight: bold;
                font-size: 11px;
                padding: 2px 6px;
                border-radius: 3px;
                border: 1px solid {border};
            }}
        """)

    def update_3d_view(self):
        side = "left" if self.p.rb_cam_left.isChecked() else "right"

        if self.p.rb_signed.isChecked():
            scalar_mode = "SignedDistance"
            lut_type = "signed_distance"
            mode_desc = "Signed Atrophy"
        elif self.p.rb_dist.isChecked():
            scalar_mode = "DistanceMapping"
            lut_type = "distance_mapping"
            mode_desc = "Deformation Mag (mm)"
        else:
            scalar_mode = "GradCAM_Importance"
            lut_type = "gradcam"
            mode_desc = "ResNet Grad-CAM"

        title = f"{side.upper()} {mode_desc} (SD = {self.p.current_sd_value:+.1f})"
        mesh_path = self.find_step_mesh(side, self.p.current_sd_value)

        if not mesh_path or not os.path.isfile(mesh_path):
            mesh_path = self.p.predictor.get_gradcam_mesh_path(
                side=side,
                component="PLS1",
                milestone="Mean",
                cohort="All_Augment_tain"
            )

        if not mesh_path or not os.path.isfile(mesh_path):
            self.p.signal_log_message.emit(f"[WARNING] 3D Grad-CAM mesh not found for {side}. Checking template...")
            mesh_path = self.p.predictor.get_template_mesh_path(side)

        if mesh_path and os.path.isfile(mesh_path):
            self.p.signal_gradcam_mesh_requested.emit(
                mesh_path, scalar_mode, lut_type, title, side, 1.0
            )

        self.p.on_patient_overlay_toggled()

    def handle_key_press(self, event):
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_BracketLeft, Qt.Key.Key_Comma):
            self.step_sd(-0.1)
            event.accept()
            return True
        elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_BracketRight, Qt.Key.Key_Period):
            self.step_sd(+0.1)
            event.accept()
            return True
        elif event.key() in (Qt.Key.Key_0, Qt.Key.Key_R, Qt.Key.Key_Home, Qt.Key.Key_Space):
            self.set_sd_value(0.0)
            event.accept()
            return True
        return False
