import os
import sys
import glob
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox,
    QGroupBox, QHBoxLayout, QLineEdit, QFileDialog,
    QHeaderView, QTabBar, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal

from .table_widget import ToggleTableWidget
from .spharm_worker import SPHARMWorker, SpharmWorker, get_project_root, find_slicer_salt_exe
from .adv_params import AdvParamsWidget
from .table_manager import SPHARMTableManager

class SpharmPanel(QWidget):
    """
    Coordinator Panel for SPHARM-PDM spherical parameterization and shape model
    generation for ICP-aligned hippocampus meshes.
    """
    signal_log_message = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str) # filepath can be str or list of str
    signal_template_toggled = pyqtSignal(bool)
    signal_overlay_all_toggled = pyqtSignal(bool, list, str) # enabled, file_list, side_filter
    signal_side_changed = pyqtSignal(str) # "all", "lh", "rh"
    signal_spharm_completed = pyqtSignal()
    signal_spharm_finished = pyqtSignal(bool)

    def __init__(self, get_folder_func, get_output_folder_func=None, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.get_output_folder = get_output_folder_func

        self.adv_widget = AdvParamsWidget(self)
        self.tbl_mgr = SPHARMTableManager(self)
        self.setup_ui()

    # Delegate property access for backward compatibility
    @property
    def all_files(self):
        return self.tbl_mgr.all_files

    @all_files.setter
    def all_files(self, val):
        self.tbl_mgr.all_files = val

    @property
    def current_side_filter(self):
        return self.tbl_mgr.current_side_filter

    @current_side_filter.setter
    def current_side_filter(self, val):
        self.tbl_mgr.current_side_filter = val

    def setup_ui(self):
        spharm_layout = QVBoxLayout(self)
        spharm_layout.setContentsMargins(10, 10, 10, 10)
        spharm_layout.setSpacing(10)

        help_label = QLabel("SPHARM-PDM spherical parameterization and shape model generation for ICP-aligned hippocampus meshes.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #555; font-size: 11px;")
        spharm_layout.addWidget(help_label)

        # Internal directory inputs for pipeline & left_panel integration
        self.mesh_input_dir = QLineEdit()
        self.mesh_input_dir.textChanged.connect(self.on_input_dir_changed)
        self.spharm_dir_input = QLineEdit()
        self.spharm_dir_input.textChanged.connect(self.populate_results_table)

        # 1. Side Selection
        side_group = QGroupBox("Side Execution Option")
        side_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 10px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        side_layout = QHBoxLayout(side_group)
        side_layout.setContentsMargins(10, 15, 10, 10)

        self.side_both_rb = QRadioButton("Both Sides (LH && RH)")
        self.side_both_rb.setChecked(True)
        self.side_lh_rb = QRadioButton("Left (LH) Only")
        self.side_rh_rb = QRadioButton("Right (RH) Only")

        self.side_btn_group = QButtonGroup(self)
        self.side_btn_group.addButton(self.side_both_rb, 0)
        self.side_btn_group.addButton(self.side_lh_rb, 1)
        self.side_btn_group.addButton(self.side_rh_rb, 2)

        side_layout.addWidget(self.side_both_rb)
        side_layout.addWidget(self.side_lh_rb)
        side_layout.addWidget(self.side_rh_rb)
        spharm_layout.addWidget(side_group)

        # 3. Main Action Button & Status Hint
        self.run_spharm_btn = QPushButton("Run SPHARM-PDM Pipeline")
        self.run_spharm_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
                padding: 9px 15px;
                border: 1px solid #ced6e0;
                border-radius: 5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #dee2e6);
                border: 1px solid #b2bec3;
                color: #1a252f;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #dee2e6, stop:1 #ced4da);
                border: 1px solid #95a5a6;
            }
            QPushButton:disabled {
                background: #f1f2f6;
                color: #a4b0be;
                border: 1px solid #dfe4ea;
            }
        """)
        self.run_spharm_btn.clicked.connect(self.run_spharm_process)
        spharm_layout.addWidget(self.run_spharm_btn)

        self.spharm_status_hint = QLabel("")
        self.spharm_status_hint.setWordWrap(True)
        self.spharm_status_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spharm_layout.addWidget(self.spharm_status_hint)

        # 4. Modular Collapsible Advanced Parameters
        spharm_layout.addWidget(self.adv_widget)

        # 5. Results Table with Category Tabs
        res_group = QGroupBox("SPHARM Surface Meshes && Models")
        res_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        res_layout = QVBoxLayout(res_group)
        res_layout.setContentsMargins(10, 16, 10, 10)
        res_layout.setSpacing(6)

        template_bar = QHBoxLayout()
        template_bar.setContentsMargins(0, 0, 0, 2)
        template_bar.setSpacing(12)
        self.template_cb = QCheckBox("Show Reference Template")
        self.template_cb.setToolTip("Overlay standard reference template (mean shape) in 3D view")
        self.template_cb.setStyleSheet("""
            QCheckBox {
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                spacing: 5px;
            }
            QCheckBox::indicator:checked {
                background: #f39c12;
                border: 1px solid #d68910;
            }
        """)
        self.template_cb.toggled.connect(self.signal_template_toggled.emit)
        template_bar.addWidget(self.template_cb)

        self.overlay_cb = QCheckBox("Overlay All Meshes")
        self.overlay_cb.setToolTip("Superimpose and view all SPHARM meshes together in 3D view")
        self.overlay_cb.setStyleSheet("""
            QCheckBox {
                color: #16a085;
                font-weight: bold;
                font-size: 11px;
                spacing: 5px;
            }
            QCheckBox::indicator:checked {
                background: #1abc9c;
                border: 1px solid #16a085;
            }
        """)
        self.overlay_cb.toggled.connect(self.on_overlay_cb_toggled)
        template_bar.addWidget(self.overlay_cb)
        template_bar.addStretch()
        res_layout.addLayout(template_bar)

        self.tab_bar = QTabBar()
        self.tab_bar.addTab("All")
        self.tab_bar.addTab("Left (LH)")
        self.tab_bar.addTab("Right (RH)")
        self.tab_bar.setExpanding(True)
        self.tab_bar.setStyleSheet("""
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f9fa, stop:1 #e9ecef);
                color: #2c3e50;
                padding: 6px 14px;
                margin-right: 3px;
                font-weight: bold;
                font-size: 11px;
                border: 1px solid #ced6e0;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f2f6);
                color: #2c3e50;
                border: 1px solid #b2bec3;
                border-bottom: none;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f2f6);
            }
        """)
        self.tab_bar.currentChanged.connect(self.tbl_mgr.on_tab_changed)
        res_layout.addWidget(self.tab_bar)

        self.results_table = ToggleTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Subject SPHARM Model", "Side", "File Path"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dcdde1;
                gridline-color: #ecf0f1;
                font-size: 11px;
                background-color: #ffffff;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #dcdde1;
                font-size: 11px;
            }
        """)
        self.results_table.itemSelectionChanged.connect(self.tbl_mgr.on_mesh_selected)
        res_layout.addWidget(self.results_table)

        spharm_layout.addWidget(res_group)
        self.update_run_button_state()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_run_button_state()

    # Forwarding methods for table manager integration
    def populate_results_table(self):
        self.tbl_mgr.populate_results_table()

    def update_table_display(self):
        self.tbl_mgr.update_table_display()

    def on_mesh_selected(self):
        self.tbl_mgr.on_mesh_selected()

    def on_tab_changed(self, index):
        self.tbl_mgr.on_tab_changed(index)

    def set_template_visible(self, visible: bool):
        self.template_cb.blockSignals(True)
        self.template_cb.setChecked(visible)
        self.template_cb.blockSignals(False)

    def set_overlay_visible(self, visible: bool):
        self.overlay_cb.blockSignals(True)
        self.overlay_cb.setChecked(visible)
        self.overlay_cb.blockSignals(False)
        if visible:
            self.tbl_mgr.emit_overlay_meshes()

    def on_overlay_cb_toggled(self, checked: bool):
        if checked:
            self.tbl_mgr.emit_overlay_meshes()
        else:
            self.signal_overlay_all_toggled.emit(False, [], self.current_side_filter)

    # Directories and input handling
    def browse_input_directory(self):
        initial = self.mesh_input_dir.text().strip() or ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select Mesh Folder (FastSurfer or ICP Output)", initial)
        if folder:
            self.mesh_input_dir.setText(folder)
            self.spharm_dir_input.setText(self.get_default_output_dir(folder))
            self.update_run_button_state()
            self.populate_results_table()

    def reset_to_pipeline_input(self):
        self.mesh_input_dir.clear()
        self.spharm_dir_input.setText(self.get_default_output_dir())
        self.update_run_button_state()
        self.populate_results_table()
        self.signal_log_message.emit("[INFO] Reset SPHARM input to default pipeline output.")

    def on_input_dir_changed(self, text):
        if text.strip() and os.path.isdir(text.strip()):
            self.spharm_dir_input.setText(self.get_default_output_dir(text.strip()))
        elif not text.strip():
            self.spharm_dir_input.setText(self.get_default_output_dir())
        self.update_run_button_state()
        self.populate_results_table()

    def browse_output_directory(self):
        initial = self.spharm_dir_input.text().strip() or ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select SPHARM Output Directory (output_SPHARM)", initial)
        if folder:
            self.spharm_dir_input.setText(folder)
            self.populate_results_table()
            self.update_run_button_state()

    def _has_spharm_inputs(self, folder):
        if not folder or not os.path.isdir(folder):
            return False
        for p in ["*.nii*", "*.vtk", "*.stl", "*.mgz"]:
            if glob.glob(os.path.join(folder, p)) or glob.glob(os.path.join(folder, "**", p), recursive=True):
                return True
        return False

    def resolve_input_folders(self):
        custom_input = self.mesh_input_dir.text().strip()
        candidates = []
        if custom_input and os.path.isdir(custom_input):
            candidates.append((custom_input, "Custom Folder"))

        out_base = self.get_output_folder().strip() if self.get_output_folder else ""
        if out_base and os.path.isdir(out_base):
            candidates.append((os.path.join(out_base, "output_ICP"), "Pipeline output_ICP"))
            candidates.append((os.path.join(out_base, "icp"), "Pipeline ICP"))
            candidates.append((out_base, "Pipeline Output"))

        for base, src_label in candidates:
            if not os.path.isdir(base):
                continue

            lh_cand = os.path.join(base, "left", "aligned_nifti")
            rh_cand = os.path.join(base, "right", "aligned_nifti")
            if self._has_spharm_inputs(lh_cand) or self._has_spharm_inputs(rh_cand):
                return lh_cand, rh_cand, base, src_label

            lh_cand_m = os.path.join(base, "left", "aligned_meshes")
            rh_cand_m = os.path.join(base, "right", "aligned_meshes")
            if self._has_spharm_inputs(lh_cand_m) or self._has_spharm_inputs(rh_cand_m):
                return lh_cand_m, rh_cand_m, base, src_label

            lh_cand2 = os.path.join(base, "left")
            rh_cand2 = os.path.join(base, "right")
            if self._has_spharm_inputs(lh_cand2) or self._has_spharm_inputs(rh_cand2):
                return lh_cand2, rh_cand2, base, src_label

            base_name = os.path.basename(base.rstrip(r'\/')).lower()
            parent = os.path.dirname(base.rstrip(r'\/'))
            if base_name in ("left", "lh", "left_hippocampus"):
                for r_name in ("right", "rh", "right_hippocampus"):
                    r_cand = os.path.join(parent, r_name)
                    if os.path.isdir(r_cand) and glob.glob(os.path.join(r_cand, "*.nii*")):
                        return base, r_cand, parent, src_label
                return base, None, parent, src_label
            elif base_name in ("right", "rh", "right_hippocampus"):
                for l_name in ("left", "lh", "left_hippocampus"):
                    l_cand = os.path.join(parent, l_name)
                    if os.path.isdir(l_cand) and glob.glob(os.path.join(l_cand, "*.nii*")):
                        return l_cand, base, parent, src_label
                return None, base, parent, src_label

            direct_nii = glob.glob(os.path.join(base, "*.nii*"))
            if direct_nii:
                lh_f = [f for f in direct_nii if os.path.basename(f).startswith("lh_") or "_lh" in os.path.basename(f).lower() or "left" in os.path.basename(f).lower()]
                rh_f = [f for f in direct_nii if os.path.basename(f).startswith("rh_") or "_rh" in os.path.basename(f).lower() or "right" in os.path.basename(f).lower()]
                if lh_f or rh_f:
                    sub_lh = os.path.join(base, "left")
                    sub_rh = os.path.join(base, "right")
                    os.makedirs(sub_lh, exist_ok=True)
                    os.makedirs(sub_rh, exist_ok=True)
                    for f in lh_f:
                        dst = os.path.join(sub_lh, os.path.basename(f))
                        if not os.path.exists(dst):
                            try: os.link(f, dst)
                            except Exception:
                                import shutil; shutil.copy2(f, dst)
                    for f in rh_f:
                        dst = os.path.join(sub_rh, os.path.basename(f))
                        if not os.path.exists(dst):
                            try: os.link(f, dst)
                            except Exception:
                                import shutil; shutil.copy2(f, dst)
                    return sub_lh, sub_rh, base, src_label

        return None, None, None, "None"

    def get_default_output_dir(self, resolved_base=None):
        custom_input = self.mesh_input_dir.text().strip()
        if custom_input and os.path.isdir(custom_input):
            p_dir = os.path.dirname(custom_input.rstrip(r'\/'))
            if p_dir and os.path.isdir(p_dir):
                return os.path.join(p_dir, "output_SPHARM")
            return os.path.join(custom_input, "output_SPHARM")

        out_base = self.get_output_folder().strip() if self.get_output_folder else ""
        if out_base and os.path.isdir(out_base):
            return os.path.join(out_base, "output_SPHARM")
        elif resolved_base and os.path.isdir(resolved_base):
            p_dir = os.path.dirname(resolved_base.rstrip(r'\/'))
            if p_dir and os.path.isdir(p_dir):
                return os.path.join(p_dir, "output_SPHARM")
            return os.path.join(resolved_base, "output_SPHARM")
        return ""

    def get_source_paths(self):
        return self.resolve_input_folders()

    def update_run_button_state(self):
        lh_dir, rh_dir, resolved_base, src_label = self.resolve_input_folders()

        has_lh = bool(lh_dir and self._has_spharm_inputs(lh_dir))
        has_rh = bool(rh_dir and self._has_spharm_inputs(rh_dir))

        default_out = self.get_default_output_dir(resolved_base)
        custom_dir = self.spharm_dir_input.text().strip()
        if not custom_dir and default_out:
            self.spharm_dir_input.setText(default_out)

        self.populate_results_table()

        if not has_lh and not has_rh:
            self.run_spharm_btn.setEnabled(False)
            self.run_spharm_btn.setToolTip("Cannot run SPHARM: No ICP-aligned meshes found.")
            self.spharm_status_hint.setText("⚠️ ICP output not found. Please run ICP registration first.")
            self.spharm_status_hint.setStyleSheet("color: #e67e22; font-size: 11px;")
        else:
            self.run_spharm_btn.setEnabled(True)
            self.run_spharm_btn.setToolTip("Click to run Batch SPHARM-PDM Pipeline")
            if len(self.all_files) > 0:
                self.spharm_status_hint.setText(f"✓ SPHARM results loaded ({len(self.all_files)} meshes). Ready to view or re-run.")
                self.spharm_status_hint.setStyleSheet("color: #27ae60; font-size: 11px;")
            else:
                self.spharm_status_hint.setText("✓ ICP output detected. Ready to run SPHARM-PDM pipeline.")
                self.spharm_status_hint.setStyleSheet("color: #2980b9; font-size: 11px;")

    def run_spharm_process(self):
        lh_dir, rh_dir, resolved_base, _ = self.resolve_input_folders()
        target_base = self.spharm_dir_input.text().strip() or self.get_default_output_dir(resolved_base)

        if not target_base:
            self.signal_log_message.emit("[ERROR] Output directory is missing. Please configure Output Directory.")
            return

        os.makedirs(target_base, exist_ok=True)
        tasks = []
        mode = self.side_btn_group.checkedId()

        # 0: Both, 1: Left only, 2: Right only
        if mode in (0, 1):
            if lh_dir and os.path.isdir(lh_dir):
                out_left = os.path.join(target_base, "left")
                os.makedirs(out_left, exist_ok=True)
                tasks.append(("left", lh_dir, out_left))
            else:
                self.signal_log_message.emit("[WARNING] Left hippocampus ICP folder not found.")

        if mode in (0, 2):
            if rh_dir and os.path.isdir(rh_dir):
                out_right = os.path.join(target_base, "right")
                os.makedirs(out_right, exist_ok=True)
                tasks.append(("right", rh_dir, out_right))
            else:
                self.signal_log_message.emit("[WARNING] Right hippocampus ICP folder not found.")

        if not tasks:
            self.signal_log_message.emit("[ERROR] No valid hippocampus folders found to run SPHARM-PDM.")
            return

        adv_params = self.adv_widget.get_adv_params()

        self.run_spharm_btn.setEnabled(False)
        self.results_table.setRowCount(0)
        self.signal_log_message.emit(f">>> Initiating Batch SPHARM-PDM Pipeline (Output: {target_base})...")

        self.worker = SPHARMWorker(tasks, adv_params)
        self.worker.signal_log.connect(self.signal_log_message.emit)
        self.worker.signal_finished.connect(self.on_spharm_finished)
        self.worker.start()

    def on_spharm_finished(self, success):
        self.update_run_button_state()
        if success:
            self.signal_log_message.emit(">>> Batch SPHARM-PDM Pipeline completed successfully.")
            self.signal_spharm_completed.emit()
        else:
            self.signal_log_message.emit("[ERROR] SPHARM Pipeline completed with warnings or errors.")
        self.signal_spharm_finished.emit(success)
        self.populate_results_table()

SPHARMPanel = SpharmPanel
