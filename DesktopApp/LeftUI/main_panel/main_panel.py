import os
import sys
import glob
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, 
    QLineEdit, QGroupBox, QMenu, QTabBar, QCheckBox
)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QColor, QFont

from .pipeline_utils import (
    find_valid_mri_files,
    check_existing_stages,
    find_spharm_mesh
)
from .pipeline_runner import PipelineRunner

class MainPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str)
    signal_diagnostic_info = pyqtSignal(str)
    signal_jump_to_module = pyqtSignal(str)

    find_valid_mri_files = staticmethod(find_valid_mri_files)
    check_existing_stages = staticmethod(check_existing_stages)
    find_spharm_mesh = staticmethod(find_spharm_mesh)

    def __init__(self, import_panel=None, fastsurfer_panel=None, icp_panel=None, spharm_panel=None, result_panel=None, parent=None):
        super().__init__(parent)
        self.import_panel = import_panel
        self.fastsurfer_panel = fastsurfer_panel
        self.icp_panel = icp_panel
        self.spharm_panel = spharm_panel
        self.result_panel = result_panel
        self.is_running = False
        self.all_pipeline_results = []
        self.runner = PipelineRunner(self)

        self.setup_ui()
        self.connect_pipeline_signals()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_stage_preview()
        self.populate_main_table()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # 1. Directory & File Selection Properties
        config_group = QGroupBox("Pipeline Input & Output Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                margin-top: 10px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
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
        cg_layout = QVBoxLayout(config_group)
        cg_layout.setContentsMargins(10, 16, 10, 10)
        cg_layout.setSpacing(8)

        # Row 1: Choose Data Directory (MRI only)
        dir_select_btn = QPushButton("Choose MRI Data Directory")
        dir_select_btn.setToolTip("Select directory containing raw MRI scans (*.nii.gz, *.nii, *.mgz)")
        dir_select_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 10px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
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
        """)
        dir_select_btn.clicked.connect(self.select_directory)
        cg_layout.addWidget(dir_select_btn)

        # Row 2: Choose Output Directory
        out_dir_btn = QPushButton("Choose Output Directory")
        out_dir_btn.setToolTip("Select base output directory. Subfolders (fastsurfer, output_ICP, output_SPHARM) will be organized automatically.")
        out_dir_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 7px 10px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
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
        """)
        out_dir_btn.clicked.connect(self.select_out_directory)
        cg_layout.addWidget(out_dir_btn)

        # Row 3: Display Input & Output Paths
        path_row = QHBoxLayout()
        path_row.setSpacing(8)

        input_col = QVBoxLayout()
        input_col.setSpacing(2)
        input_lbl = QLabel("MRI Input:")
        input_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        self.folder_input.setPlaceholderText("Select folder with MRI scans (.nii.gz, .nii, .mgz)...")
        self.folder_input.setStyleSheet("background: white; border: 1px solid #ced6e0; border-radius: 4px; padding: 5px 8px; font-size: 11px;")
        input_col.addWidget(input_lbl)
        input_col.addWidget(self.folder_input)
        path_row.addLayout(input_col)

        output_col = QVBoxLayout()
        output_col.setSpacing(2)
        output_lbl = QLabel("Output Folder:")
        output_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.out_folder_input = QLineEdit()
        self.out_folder_input.setReadOnly(True)
        self.out_folder_input.setPlaceholderText("Select base output directory...")
        self.out_folder_input.setStyleSheet("background: white; border: 1px solid #ced6e0; border-radius: 4px; padding: 5px 8px; font-size: 11px;")
        output_col.addWidget(output_lbl)
        output_col.addWidget(self.out_folder_input)
        path_row.addLayout(output_col)

        cg_layout.addLayout(path_row)
        main_layout.addWidget(config_group)

        # 2. Main Panel Pipeline Results & Subjects Table
        table_group = QGroupBox("Pipeline Subjects & Output Status")
        table_group.setStyleSheet("""
            QGroupBox {
                margin-top: 10px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
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
        tg_layout = QVBoxLayout(table_group)
        tg_layout.setContentsMargins(10, 16, 10, 10)
        tg_layout.setSpacing(6)

        # Tab Bar for Hemisphere Filtering
        self.tab_bar = QTabBar()
        self.tab_bar.addTab("All Meshes (0)")
        self.tab_bar.addTab("Left (0)")
        self.tab_bar.addTab("Right (0)")
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
        self.tab_bar.currentChanged.connect(self.on_tab_changed)
        tg_layout.addWidget(self.tab_bar)

        self.table = QTableWidget(0, 4)
        self.table.setMinimumHeight(240)
        self.table.setHorizontalHeaderLabels([
            "Subject", "Side", "Diagnosis", "Probability"
        ])
        h_header = self.table.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        h_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #dcdde1;
                font-size: 11px;
                gridline-color: #ecf0f1;
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
        self.table.itemSelectionChanged.connect(self.on_table_row_selected)
        tg_layout.addWidget(self.table)
        main_layout.addWidget(table_group, stretch=1)

        # 3. Pipeline Stages & Execution Controls
        workflow_group = QGroupBox("Pipeline Workflow Execution")
        workflow_group.setStyleSheet("""
            QGroupBox {
                margin-top: 10px;
                border: 1px solid #dcdde1;
                border-radius: 6px;
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
        wg_layout = QVBoxLayout(workflow_group)
        wg_layout.setContentsMargins(10, 16, 10, 10)
        wg_layout.setSpacing(8)

        # Progress Stages Display with Quick Jump buttons
        stages_layout = QVBoxLayout()
        stages_layout.setSpacing(5)

        jump_btn_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f2f6);
                color: #2c3e50;
                font-size: 10px;
                font-weight: bold;
                padding: 3px 9px;
                border: 1px solid #ced6e0;
                border-radius: 4px;
                min-width: 52px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: 1px solid #2471a3;
            }
            QPushButton:pressed {
                background: #2471a3;
                color: white;
            }
        """

        # Stage 1 Row: FastSurfer
        s1_row = QHBoxLayout()
        s1_row.setSpacing(6)
        self.stage1_lbl = QLabel("  1. FastSurfer Hippocampal Segmentation:  Pending")
        self.stage1_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        self.stage1_jump_btn = QPushButton("Open →")
        self.stage1_jump_btn.setToolTip("Switch to FastSurfer Segmentation Panel")
        self.stage1_jump_btn.setStyleSheet(jump_btn_style)
        self.stage1_jump_btn.clicked.connect(lambda: self.jump_to_panel("FastSurfer Segmentation"))
        s1_row.addWidget(self.stage1_lbl, stretch=1)
        s1_row.addWidget(self.stage1_jump_btn)
        stages_layout.addLayout(s1_row)

        # Stage 2 Row: ICP
        s2_row = QHBoxLayout()
        s2_row.setSpacing(6)
        self.stage2_lbl = QLabel("  2. Groupwise ICP Mesh Registration:         Pending")
        self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        self.stage2_jump_btn = QPushButton("Open →")
        self.stage2_jump_btn.setToolTip("Switch to ICP Registration Panel")
        self.stage2_jump_btn.setStyleSheet(jump_btn_style)
        self.stage2_jump_btn.clicked.connect(lambda: self.jump_to_panel("ICP Registration"))
        s2_row.addWidget(self.stage2_lbl, stretch=1)
        s2_row.addWidget(self.stage2_jump_btn)
        stages_layout.addLayout(s2_row)

        # Stage 3 Row: SPHARM
        s3_row = QHBoxLayout()
        s3_row.setSpacing(6)
        self.stage3_lbl = QLabel("  3. SPHARM-PDM Shape Analysis:            Pending")
        self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        self.stage3_jump_btn = QPushButton("Open →")
        self.stage3_jump_btn.setToolTip("Switch to SPHARM Processing Panel")
        self.stage3_jump_btn.setStyleSheet(jump_btn_style)
        self.stage3_jump_btn.clicked.connect(lambda: self.jump_to_panel("SPHARM Processing"))
        s3_row.addWidget(self.stage3_lbl, stretch=1)
        s3_row.addWidget(self.stage3_jump_btn)
        stages_layout.addLayout(s3_row)

        # Stage 4 Row: Prediction
        s4_row = QHBoxLayout()
        s4_row.setSpacing(6)
        self.stage4_lbl = QLabel("  4. Prediction & 3D Grad-CAM: Pending")
        self.stage4_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
        self.stage4_jump_btn = QPushButton("Open →")
        self.stage4_jump_btn.setToolTip("Switch to Result Panel")
        self.stage4_jump_btn.setStyleSheet(jump_btn_style)
        self.stage4_jump_btn.clicked.connect(lambda: self.jump_to_panel("Result Panel"))
        s4_row.addWidget(self.stage4_lbl, stretch=1)
        s4_row.addWidget(self.stage4_jump_btn)
        stages_layout.addLayout(s4_row)

        wg_layout.addLayout(stages_layout)

        # Force Re-run Checkbox
        self.force_rerun_cb = QCheckBox("Force Re-run all stages (Overwrite existing results)")
        self.force_rerun_cb.setToolTip("When checked, the pipeline will re-execute all stages from scratch even if previous results exist in output directory.")
        self.force_rerun_cb.setStyleSheet("""
            QCheckBox {
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                spacing: 6px;
                margin-top: 4px;
                margin-bottom: 2px;
            }
            QCheckBox::indicator:checked {
                background: #e74c3c;
                border: 1px solid #c0392b;
            }
        """)
        self.force_rerun_cb.toggled.connect(lambda: self.update_stage_preview())
        wg_layout.addWidget(self.force_rerun_cb)

        # Status notification label (hidden from UI)
        self.status_lbl = QLabel("")

        # Run Button (matches FastSurfer, ICP, SPHARM, Result panels)
        self.run_btn = QPushButton("Run Full Pipeline")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
                padding: 10px 15px;
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
        self.run_btn.clicked.connect(self.run_full_pipeline)
        wg_layout.addWidget(self.run_btn)

        main_layout.addWidget(workflow_group)

        # Initial check for stage status preview & populate table
        self.update_stage_preview()
        self.populate_main_table()

    def connect_pipeline_signals(self):
        if self.fastsurfer_panel:
            self.fastsurfer_panel.signal_fastsurfer_finished.connect(self.on_fastsurfer_finished_step)
        if self.icp_panel:
            self.icp_panel.signal_icp_finished.connect(self.on_icp_finished_step)
        if self.spharm_panel:
            self.spharm_panel.signal_spharm_finished.connect(self.on_spharm_finished_step)
        if self.result_panel and hasattr(self.result_panel, 'signal_batch_prediction_finished'):
            self.result_panel.signal_batch_prediction_finished.connect(self.on_result_finished_step)

    def set_directories(self, in_dir, out_dir):
        if in_dir:
            self.folder_input.setText(in_dir)
            self.last_input_dir = in_dir
        if out_dir:
            self.out_folder_input.setText(out_dir)
            self.last_output_dir = out_dir
            self.sync_panel_output_folders(out_dir)
        self.update_stage_preview()
        self.populate_main_table(in_dir, out_dir)

    def sync_panel_output_folders(self, out_dir):
        """Pre-configures individual subfolders matching each panel's standard structure and refreshes tables."""
        if not out_dir:
            return
        fs_dir = os.path.join(out_dir, "fastsurfer")
        icp_dir = os.path.join(out_dir, "output_ICP")
        spharm_dir = os.path.join(out_dir, "output_SPHARM")
        res_dir = os.path.join(out_dir, "output_Result")

        if self.fastsurfer_panel and hasattr(self.fastsurfer_panel, 'fs_dir_input'):
            self.fastsurfer_panel.fs_dir_input.setText(fs_dir)
            if os.path.isdir(fs_dir):
                self.fastsurfer_panel.populate_results_table()
        if self.icp_panel and hasattr(self.icp_panel, 'icp_dir_input'):
            self.icp_panel.icp_dir_input.setText(icp_dir)
            if os.path.isdir(icp_dir):
                self.icp_panel.populate_results_table()
        if self.spharm_panel and hasattr(self.spharm_panel, 'spharm_dir_input'):
            self.spharm_panel.spharm_dir_input.setText(spharm_dir)
            if os.path.isdir(spharm_dir):
                self.spharm_panel.populate_results_table()
        if self.result_panel:
            if hasattr(self.result_panel, 'spharm_dir_input'):
                self.result_panel.spharm_dir_input.setText(spharm_dir)
            if hasattr(self.result_panel, 'result_dir_input'):
                self.result_panel.result_dir_input.setText(res_dir)
            if os.path.isdir(res_dir) and hasattr(self.result_panel, 'load_existing_results'):
                self.result_panel.load_existing_results()

    def jump_to_panel(self, module_name: str):
        self.signal_jump_to_module.emit(module_name)
        win = self.window()
        if win and hasattr(win, 'module_combo'):
            win.module_combo.setCurrentText(module_name)
            return

        parent = self.parent()
        while parent:
            if hasattr(parent, 'switch_module'):
                name_to_idx = {
                    "Main Panel": 0,
                    "Data Importer": 1,
                    "FastSurfer Segmentation": 2,
                    "ICP Registration": 3,
                    "SPHARM Processing": 4,
                    "Result Panel": 5
                }
                if module_name in name_to_idx:
                    parent.switch_module(name_to_idx[module_name])
                    return
            parent = parent.parent()

    def update_stage_preview(self):
        """Checks output directory and previews which stages exist and which will be run."""
        out_dir = self.out_folder_input.text().strip()
        if not out_dir or not os.path.isdir(out_dir):
            self.stage1_lbl.setText("  1. FastSurfer Hippocampal Segmentation:  Pending")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Pending")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Pending")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.stage4_lbl.setText("  4. Prediction & 3D Grad-CAM: Pending")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #57606f;")
            self.status_lbl.setText("Ready: Select Input MRI Directory & Output Directory to begin.")
            self.status_lbl.setStyleSheet("color: #2c3e50; background-color: #eaf2f8; border: 1px solid #d4e6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
            return

        has_fs, has_icp, has_spharm, has_result = self.check_existing_stages(out_dir)
        force_rerun = hasattr(self, 'force_rerun_cb') and self.force_rerun_cb.isChecked()

        # Stage 1 label
        if has_fs:
            action = "(Will overwrite)" if force_rerun else "(Will skip)"
            color = "#d35400" if force_rerun else "#27ae60"
            self.stage1_lbl.setText(f"  1. FastSurfer Hippocampal Segmentation:  Results exist {action}")
            self.stage1_lbl.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: bold;")
        else:
            self.stage1_lbl.setText("  1. FastSurfer Hippocampal Segmentation:  Missing (Will run first)")
            self.stage1_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")

        # Stage 2 label
        if has_icp:
            action = "(Will overwrite)" if force_rerun else "(Will skip)"
            color = "#d35400" if force_rerun else "#27ae60"
            self.stage2_lbl.setText(f"  2. Groupwise ICP Mesh Registration:         Results exist {action}")
            self.stage2_lbl.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: bold;")
        elif has_fs or force_rerun:
            self.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Will run after Stage 1")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        else:
            self.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Pending")
            self.stage2_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

        # Stage 3 label
        if has_spharm:
            action = "(Will overwrite)" if force_rerun else "(Will skip)"
            color = "#d35400" if force_rerun else "#27ae60"
            self.stage3_lbl.setText(f"  3. SPHARM-PDM Shape Analysis:            Results exist {action}")
            self.stage3_lbl.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: bold;")
        elif (has_fs and has_icp) or force_rerun:
            self.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Will run after Stage 2")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        else:
            self.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Pending")
            self.stage3_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

        # Stage 4 label
        if has_result:
            action = "(Will overwrite)" if force_rerun else "(Will skip)"
            color = "#d35400" if force_rerun else "#27ae60"
            self.stage4_lbl.setText(f"  4. Prediction & 3D Grad-CAM: Results exist {action}")
            self.stage4_lbl.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: bold;")
        elif (has_fs and has_icp and has_spharm) or force_rerun:
            self.stage4_lbl.setText("  4. Prediction & 3D Grad-CAM: Will run after Stage 3")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        else:
            self.stage4_lbl.setText("  4. Prediction & 3D Grad-CAM: Pending")
            self.stage4_lbl.setStyleSheet("font-size: 11px; color: #57606f;")

        # Status text summary
        if force_rerun:
            self.status_lbl.setText("Force Re-run enabled: Running will re-compute all 4 pipeline stages from scratch.")
            self.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px; font-weight: bold;")
        elif has_fs and has_icp and has_spharm and has_result:
            self.status_lbl.setText("Complete results exist in output folder. Running will refresh all tables without re-computing.")
            self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        elif has_fs and has_icp and has_spharm and not has_result:
            self.status_lbl.setText("FastSurfer, ICP & SPHARM results exist (Skipping 1-3). Running will execute Prediction & 3D Grad-CAM.")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        elif has_fs and has_icp and not has_spharm:
            self.status_lbl.setText("FastSurfer & ICP results exist (Skipping both). Running will execute SPHARM-PDM -> Prediction.")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        elif has_fs and not has_icp:
            self.status_lbl.setText("FastSurfer results exist (Skipping FastSurfer). Running will execute ICP -> SPHARM -> Prediction.")
            self.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        else:
            self.status_lbl.setText("Ready to run full pipeline.")
            self.status_lbl.setStyleSheet("color: #2c3e50; background-color: #eaf2f8; border: 1px solid #d4e6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")


    def on_tab_changed(self, index):
        self.render_table_rows()

    def populate_main_table(self, in_dir=None, out_dir=None):
        """
        Loads paired Grad-CAM evaluation results and SPHARM 3D meshes into the Main Panel table.
        Keeps table empty on fresh startup until output directory is selected or pipeline finishes.
        """
        if not out_dir:
            out_dir = self.out_folder_input.text().strip()
        if not out_dir and getattr(self, 'import_panel', None):
            out_dir = self.import_panel.get_output_folder().strip()

        if not out_dir or not os.path.isdir(out_dir):
            self.all_pipeline_results = []
            self.table.setRowCount(0)
            self.tab_bar.blockSignals(True)
            self.tab_bar.setTabText(0, "All Meshes (0)")
            self.tab_bar.setTabText(1, "Left (0)")
            self.tab_bar.setTabText(2, "Right (0)")
            self.tab_bar.blockSignals(False)
            return

        json_file = os.path.join(out_dir, "output_Result", "evaluation_summary.json")

        records = []
        if os.path.isfile(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for item in data:
                    s_raw = item.get('subject', '')
                    s_clean = s_raw.strip()
                    if not s_clean:
                        continue
                    summary = item.get('summary', {})
                    lh_mesh = item.get('left_mesh')
                    rh_mesh = item.get('right_mesh')
                    left_prob = summary.get('left_prob')
                    right_prob = summary.get('right_prob')
                    overall_prob = summary.get('probability', 0.0)

                    # Left Hemisphere record: ONLY include if BOTH real SPHARM mesh and evaluated result exist
                    lh_spharm = self.find_spharm_mesh(out_dir, s_clean, "left")
                    if lh_spharm and left_prob is not None and lh_mesh and os.path.isfile(lh_mesh):
                        diag_str = "Epilepsy (TLE)" if left_prob > 0.5 else "Normal (HC)"
                        prob_str = f"{left_prob * 100:.1f}%"
                        records.append({
                            "subject": s_clean,
                            "side": "left",
                            "side_str": "LH",
                            "diag": diag_str,
                            "prob": left_prob,
                            "prob_str": prob_str,
                            "gradcam_mesh": lh_mesh,
                            "spharm_mesh": lh_spharm
                        })

                    # Right Hemisphere record: ONLY include if BOTH real SPHARM mesh and evaluated result exist
                    rh_spharm = self.find_spharm_mesh(out_dir, s_clean, "right")
                    if rh_spharm and right_prob is not None and rh_mesh and os.path.isfile(rh_mesh):
                        diag_str = "Epilepsy (TLE)" if right_prob > 0.5 else "Normal (HC)"
                        prob_str = f"{right_prob * 100:.1f}%"
                        records.append({
                            "subject": s_clean,
                            "side": "right",
                            "side_str": "RH",
                            "diag": diag_str,
                            "prob": right_prob,
                            "prob_str": prob_str,
                            "gradcam_mesh": rh_mesh,
                            "spharm_mesh": rh_spharm
                        })
            except Exception as e:
                self.signal_log_message.emit(f"[WARNING] Main Panel could not parse {json_file}: {e}")

        self.all_pipeline_results = records
        self.render_table_rows()

    def render_table_rows(self):
        tab_idx = self.tab_bar.currentIndex()

        total_lh = sum(1 for r in self.all_pipeline_results if r["side"] == "left")
        total_rh = sum(1 for r in self.all_pipeline_results if r["side"] == "right")

        self.tab_bar.blockSignals(True)
        self.tab_bar.setTabText(0, f"All Meshes ({len(self.all_pipeline_results)})")
        self.tab_bar.setTabText(1, f"Left ({total_lh})")
        self.tab_bar.setTabText(2, f"Right ({total_rh})")
        self.tab_bar.blockSignals(False)

        if tab_idx == 1:
            filtered = [r for r in self.all_pipeline_results if r["side"] == "left"]
        elif tab_idx == 2:
            filtered = [r for r in self.all_pipeline_results if r["side"] == "right"]
        else:
            filtered = list(self.all_pipeline_results)

        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.setRowCount(len(filtered))

        for row, rec in enumerate(filtered):
            subj = rec["subject"]
            side_str = rec["side_str"]
            diag_str = rec["diag"]
            prob_str = rec["prob_str"]
            gradcam_p = rec["gradcam_mesh"]
            spharm_p = rec["spharm_mesh"]

            item_subj = QTableWidgetItem(subj)
            item_subj.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            item_subj.setData(Qt.ItemDataRole.UserRole, rec)
            item_subj.setToolTip(f"Subject: {subj}\nSide: {side_str}")

            item_side = QTableWidgetItem(side_str)
            item_side.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if side_str == "LH":
                item_side.setForeground(QColor("#2980b9"))
            else:
                item_side.setForeground(QColor("#d35400"))
            item_side.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

            item_diag = QTableWidgetItem(diag_str)
            item_diag.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if "TLE" in diag_str or "Epilepsy" in diag_str:
                item_diag.setForeground(QColor("#c0392b"))
            elif "Normal" in diag_str or "HC" in diag_str:
                item_diag.setForeground(QColor("#27ae60"))
            else:
                item_diag.setForeground(QColor("#7f8c8d"))
            item_diag.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

            item_prob = QTableWidgetItem(prob_str)
            item_prob.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table.setItem(row, 0, item_subj)
            self.table.setItem(row, 1, item_side)
            self.table.setItem(row, 2, item_diag)
            self.table.setItem(row, 3, item_prob)

        self.table.blockSignals(False)

        if self.table.rowCount() > 0:
            if self.isVisible():
                self.table.selectRow(0)
                self.on_table_row_selected()
            else:
                self.table.blockSignals(True)
                self.table.selectRow(0)
                self.table.blockSignals(False)
        else:
            if self.isVisible():
                self.signal_mesh_selected.emit("", "all")
                self.signal_diagnostic_info.emit("")

    def on_table_row_selected(self):
        if not self.isVisible():
            return
        selected = self.table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        item = self.table.item(row, 0)
        if not item:
            return
        rec = item.data(Qt.ItemDataRole.UserRole)
        if not rec or not isinstance(rec, dict):
            return

        spharm_path = rec.get("spharm_mesh", "")
        side = rec.get("side", "left")
        subj = rec.get("subject", "")
        diag = rec.get("diag", "")
        prob_str = rec.get("prob_str", "")
        gradcam = rec.get("gradcam_mesh", "")

        is_tle = "TLE" in diag or "Epilepsy" in diag
        diag_color = "#e74c3c" if is_tle else "#2ecc71"
        diag_badge = f'<span style="color: {diag_color}; font-weight: bold;">&#9679; {diag} ({prob_str})</span>'
        g_name = os.path.basename(gradcam) if gradcam and gradcam != "—" else "None"
        info_html = f'{diag_badge}  |  <span style="color: #bdc3c7;">Grad-CAM: {g_name}</span>'

        win = self.window()
        if win and hasattr(win, 'right_panel'):
            win.right_panel.set_view_mode("full_3d", "Main Panel")

        self.signal_diagnostic_info.emit(info_html)

        if spharm_path and os.path.isfile(spharm_path):
            self.signal_mesh_selected.emit(spharm_path, side)
            self.signal_log_message.emit(f"Main Panel displaying SPHARM mesh: {os.path.basename(spharm_path)} for {subj} [{side.upper()}] (Diagnosis: {diag}, Prob: {prob_str})")
        else:
            self.signal_mesh_selected.emit("", side)
            self.signal_log_message.emit(f"Main Panel: No SPHARM mesh found on disk for {subj} [{side.upper()}].")


    def select_directory(self):
        parent_win = self.window() if self.window() else self
        initial_dir = "D:/" if os.path.exists("D:/") else "C:/"
        folder = QFileDialog.getExistingDirectory(parent_win, "Select Directory with MRI Scans", initial_dir)
        if folder:
            mri_files = self.find_valid_mri_files(folder)
            if not mri_files:
                self.signal_log_message.emit(f"[WARNING] No MRI scan files (*.nii.gz, *.nii, *.mgz) found in: {folder}")
                self.status_lbl.setText("Warning: No valid MRI scan files (*.nii.gz, *.nii, *.mgz) found in selected directory.")
                self.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
            else:
                self.status_lbl.setText(f"Detected {len(mri_files)} MRI image(s). Ready to process.")
                self.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px;")

            self.folder_input.setText(folder)
            self.signal_log_message.emit(f"Main Panel selected MRI directory: {folder} (Found {len(mri_files)} MRI files)")

            if self.import_panel:
                self.import_panel.folder_input.setText(folder)
                self.import_panel.load_subjects_from_directory(folder)
                self.import_panel.signal_directories_changed.emit(folder, self.out_folder_input.text().strip())

            self.populate_main_table(folder, self.out_folder_input.text().strip())

    def select_out_directory(self):
        parent_win = self.window() if self.window() else self
        initial_dir = getattr(self, 'last_output_dir', None) if getattr(self, 'last_output_dir', None) and os.path.isdir(self.last_output_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(parent_win, "Select Output Directory", initial_dir)
        if folder:
            self.out_folder_input.setText(folder)
            self.last_output_dir = folder
            win = self.window()
            if win and hasattr(win, 'left_panel') and hasattr(win.left_panel, 'set_global_output_directory'):
                win.left_panel.set_global_output_directory(folder)
            else:
                self.sync_panel_output_folders(folder)
                self.update_stage_preview()
                self.populate_main_table(self.folder_input.text().strip(), folder)
                if self.import_panel:
                    self.import_panel.out_folder_input.setText(folder)
                    self.import_panel.load_subjects_from_output(folder)
            self.signal_log_message.emit(f"Main Panel selected output directory: {folder}")


    # Delegate pipeline execution to PipelineRunner
    def run_full_pipeline(self):
        self.runner.run_full_pipeline()

    def on_fastsurfer_finished_step(self, success):
        self.runner.on_fastsurfer_finished_step(success)

    def on_icp_finished_step(self, success):
        self.runner.on_icp_finished_step(success)

    def on_spharm_finished_step(self, success):
        self.runner.on_spharm_finished_step(success)

    def run_result_step(self, out_dir):
        self.runner.run_result_step(out_dir)

    def on_result_finished_step(self, success):
        self.runner.on_result_finished_step(success)

    def reset_run_state(self):
        self.runner.reset_run_state()
