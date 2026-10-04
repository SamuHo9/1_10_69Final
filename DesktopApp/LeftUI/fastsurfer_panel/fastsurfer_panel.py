import os
import sys
import glob
import subprocess
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView, QTabBar, QLineEdit, QFileDialog)
from PyQt6.QtCore import pyqtSignal, Qt, QThread

class ToggleTableWidget(QTableWidget):
    """QTableWidget supporting ExtendedSelection (Ctrl/Shift multi-select)
    and single-click toggle/deselect when clicking an already selected sole row."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                row = item.row()
                modifiers = event.modifiers()
                has_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

                selected_rows = list(set(it.row() for it in self.selectedItems()))

                if not has_ctrl and not has_shift:
                    if selected_rows == [row]:
                        self.clearSelection()
                        return
                    elif row in selected_rows and len(selected_rows) > 1:
                        self.clearSelection()
                        self.selectRow(row)
                        return

        super().mousePressEvent(event)

def get_project_root():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

class FastSurferWorker(QThread):
    signal_log = pyqtSignal(str)
    signal_finished = pyqtSignal(bool)

    def __init__(self, input_dir, output_dir=None):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self):
        self.signal_log.emit(f"Running run_pipeline.py with input_dir={self.input_dir}")
        root_dir = get_project_root()
        pipeline_script = os.path.join(root_dir, "run_pipeline.py")
        if not os.path.isfile(pipeline_script):
            cand = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "run_pipeline.py"))
            if os.path.isfile(cand):
                pipeline_script = cand
            else:
                cand2 = os.path.join(root_dir, "FastSurfer", "run_pipeline.py")
                if os.path.isfile(cand2):
                    pipeline_script = cand2

        if not os.path.isfile(pipeline_script):
            self.signal_log.emit(f"[ERROR] run_pipeline.py not found at: {pipeline_script}")
            self.signal_finished.emit(False)
            return
        
        try:
            cmd = [sys.executable, pipeline_script, "--input_dir", self.input_dir]
            if self.output_dir:
                cmd.extend(["--output_dir", self.output_dir])
            
            process = subprocess.Popen(
                cmd,
                cwd=root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            for line in process.stdout:
                self.signal_log.emit(line.strip())
            process.wait()
            self.signal_finished.emit(process.returncode == 0)
        except Exception as e:
            self.signal_log.emit(f"[ERROR] Failed to run pipeline: {str(e)}")
            self.signal_finished.emit(False)


class FastsurferPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str) # filepath can be str or list of str
    signal_fastsurfer_completed = pyqtSignal()
    signal_fastsurfer_finished = pyqtSignal(bool)
    signal_overlay_all_toggled = pyqtSignal(bool, list, str) # enabled, file_list, side_filter
    signal_side_changed = pyqtSignal(str)
    
    def __init__(self, get_folder_func, get_output_folder_func=None, parent=None):
        super().__init__(parent)
        self.get_folder = get_folder_func
        self.get_output_folder = get_output_folder_func
        self.all_files = []
        self.current_side_filter = "all"
        self.last_selected_row = None
        self.setup_ui()
        
    def setup_ui(self):
        fs_layout = QVBoxLayout(self)
        fs_layout.setContentsMargins(10, 10, 10, 10)
        fs_layout.setSpacing(10)
        
        help_label = QLabel("Run FastSurfer segmentation or load existing Hippocampus mesh results.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #555; font-size: 11px;")
        fs_layout.addWidget(help_label)
        
        # Internal reference to maintain compatibility with left_panel & main_panel
        self.fs_dir_input = QLineEdit()
        
        # 1. FastSurfer Execution Parameters
        fs_group = QGroupBox("FastSurfer Execution Parameters")
        fs_group.setStyleSheet("""
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
        fs_form = QFormLayout(fs_group)
        fs_form.setContentsMargins(10, 16, 10, 10)
        fs_form.setSpacing(6)
        
        self.fs_gpu_cb = QCheckBox("Use GPU Acceleration (if available)")
        self.fs_gpu_cb.setChecked(True)
        self.fs_seg_only_cb = QCheckBox("Run Segmentation Only")
        self.fs_seg_only_cb.setChecked(True)
        
        fs_form.addRow("Compute Device:", self.fs_gpu_cb)
        fs_form.addRow("Pipeline Mode:", self.fs_seg_only_cb)
        fs_layout.addWidget(fs_group)
        
        self.run_fs_btn = QPushButton("Run FastSurfer Pipeline")
        self.run_fs_btn.setStyleSheet("""
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
        self.run_fs_btn.clicked.connect(self.run_fastsurfer_process)
        fs_layout.addWidget(self.run_fs_btn)

        self.fs_status_hint = QLabel("")
        
        # 3. Segmentation Results Table with Category Tabs (All / Left / Right)
        res_group = QGroupBox("Segmentation Results & Meshes")
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

        # Overlay All Meshes checkbox
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 2)
        top_bar.setSpacing(12)

        self.overlay_cb = QCheckBox("Overlay All Meshes")
        self.overlay_cb.setToolTip("Superimpose and view all segmented meshes together in 3D view")
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
        top_bar.addWidget(self.overlay_cb)
        top_bar.addStretch()
        res_layout.addLayout(top_bar)
        
        # Category Tabs (All / Left / Right)
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
        self.tab_bar.currentChanged.connect(self.on_tab_changed)
        res_layout.addWidget(self.tab_bar)
        
        # Results Table with 3 Columns
        self.results_table = ToggleTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Segmented Mesh Name", "Side", "File Path"])
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
        self.results_table.itemSelectionChanged.connect(self.on_mesh_selected)
        res_layout.addWidget(self.results_table)
        
        fs_layout.addWidget(res_group)
        self.update_run_button_state()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_run_button_state()

    def get_effective_input_dir(self):
        """
        Determines the effective MRI input directory.
        Priority 1: 'data import' or 'data_import' repository inside the output directory.
        Priority 2: get_folder() from Data Importer.
        """
        out_dir = self.get_output_folder().strip() if self.get_output_folder else ""
        if out_dir and os.path.isdir(out_dir):
            for cand_name in ["data import", "data_import"]:
                cand_path = os.path.join(out_dir, cand_name)
                if os.path.isdir(cand_path):
                    # Check if any MRI files exist
                    mris = [f for f in os.listdir(cand_path) if f.lower().endswith((".nii.gz", ".nii", ".mgz", ".nrrd"))]
                    if mris:
                        return cand_path

        raw_in = self.get_folder().strip() if self.get_folder else ""
        if raw_in and os.path.isdir(raw_in):
            return raw_in
        return ""

    def update_run_button_state(self):
        effective_in = self.get_effective_input_dir()
        out_dir = self.get_output_folder().strip() if self.get_output_folder else ""
        
        has_in = bool(effective_in and os.path.isdir(effective_in))
        has_out = bool(out_dir and os.path.isdir(out_dir))
        
        if not has_out or not has_in:
            self.run_fs_btn.setEnabled(False)
            self.run_fs_btn.setToolTip("Please select Input and Output directories in Data Importer")
        else:
            self.run_fs_btn.setEnabled(True)
            source_desc = "from 'data import'" if ("data import" in effective_in.lower() or "data_import" in effective_in.lower()) else "from input folder"
            self.run_fs_btn.setToolTip(f"Click to run FastSurfer Pipeline ({source_desc})")
                
            # If fs_dir_input is empty, pre-fill it with output_dir/fastsurfer
            if not self.fs_dir_input.text().strip():
                expected_fs = os.path.join(os.path.abspath(out_dir), "fastsurfer")
                self.fs_dir_input.setText(expected_fs)

    def browse_results_directory(self):
        initial_dir = "D:/" if os.path.exists("D:/") else "C:/"
        folder = QFileDialog.getExistingDirectory(self, "Select FastSurfer Results Directory", initial_dir)
        if folder:
            self.fs_dir_input.setText(folder)
            self.signal_log_message.emit(f"Selected FastSurfer results directory: {folder}")
            self.populate_results_table()

    def run_fastsurfer_process(self):
        out_dir = self.get_output_folder().strip() if self.get_output_folder else ""
        if not out_dir or not os.path.isdir(out_dir):
            self.signal_log_message.emit("[ERROR] Please select an Output Directory in Data Importer before running.")
            return

        input_dir = self.get_effective_input_dir()
        if not input_dir or not os.path.isdir(input_dir):
            self.signal_log_message.emit("[ERROR] Please import MRI scans to 'data import' or select a valid input directory first.")
            return

        is_from_data_import = ("data import" in input_dir.lower() or "data_import" in input_dir.lower())
        if is_from_data_import:
            self.signal_log_message.emit(f">>> FastSurfer Pipeline using MRI images from 'data import' repository: {input_dir}")
        else:
            self.signal_log_message.emit(f">>> FastSurfer Pipeline using input directory: {input_dir}")
            
        self.run_fs_btn.setEnabled(False)
        self.results_table.setRowCount(0)
        self.signal_log_message.emit(">>> Starting FastSurfer Pipeline...")
        
        self.worker = FastSurferWorker(input_dir, out_dir)
        self.worker.signal_log.connect(self.signal_log_message.emit)
        self.worker.signal_finished.connect(self.on_fastsurfer_finished)
        self.worker.start()

    run_process = run_fastsurfer_process

    def on_fastsurfer_finished(self, success):
        self.update_run_button_state()
        if success:
            self.signal_log_message.emit(">>> FastSurfer Pipeline completed successfully.")
            self.signal_fastsurfer_completed.emit()
        else:
            self.signal_log_message.emit("[ERROR] FastSurfer Pipeline failed or finished with errors.")
        self.signal_fastsurfer_finished.emit(success)
        
        # Sync input path with output folder
        if self.get_output_folder and self.get_output_folder():
            default_dir = os.path.join(os.path.abspath(self.get_output_folder()), "fastsurfer")
            self.fs_dir_input.setText(default_dir)
        
        self.populate_results_table()

    def on_tab_changed(self, index):
        if index == 1:
            self.current_side_filter = "lh"
        elif index == 2:
            self.current_side_filter = "rh"
        else:
            self.current_side_filter = "all"
        self.signal_side_changed.emit(self.current_side_filter)
        self.update_table_display()
        if hasattr(self, 'overlay_cb') and self.overlay_cb.isChecked():
            self.emit_overlay_meshes()

    def set_overlay_visible(self, visible: bool):
        if hasattr(self, 'overlay_cb'):
            self.overlay_cb.blockSignals(True)
            self.overlay_cb.setChecked(visible)
            self.overlay_cb.blockSignals(False)
            if visible:
                self.emit_overlay_meshes()

    def on_overlay_cb_toggled(self, checked: bool):
        if checked:
            self.emit_overlay_meshes()
        else:
            self.signal_overlay_all_toggled.emit(False, [], self.current_side_filter)

    def get_current_display_files(self):
        if self.current_side_filter == "lh":
            return [f for f in self.all_files if f["side_key"] == "lh"]
        elif self.current_side_filter == "rh":
            return [f for f in self.all_files if f["side_key"] == "rh"]
        return self.all_files

    def emit_overlay_meshes(self):
        display_files = self.get_current_display_files()
        filepaths = [f["filepath"] for f in display_files]
        self.signal_overlay_all_toggled.emit(True, filepaths, self.current_side_filter)

    def populate_results_table(self):
        target_dir = ""
        if self.get_output_folder and self.get_output_folder():
            out_f = self.get_output_folder().strip()
            if out_f and os.path.isdir(out_f):
                cand = os.path.join(os.path.abspath(out_f), "fastsurfer")
                target_dir = cand if os.path.isdir(cand) else out_f
                self.fs_dir_input.setText(target_dir)

        if not target_dir:
            target_dir = self.fs_dir_input.text().strip()
            
        self.all_files = []
        if os.path.isdir(target_dir):
            search_dirs = [target_dir]
            # Check for subdirectories
            for sub in ["left_hippocampus", "right_hippocampus", "fastsurfer_temp"]:
                p = os.path.join(target_dir, sub)
                if os.path.isdir(p):
                    search_dirs.append(p)
            # Also check if target_dir has a 'fastsurfer' child
            fs_child = os.path.join(target_dir, "fastsurfer")
            if os.path.isdir(fs_child):
                search_dirs.append(fs_child)
                search_dirs.append(os.path.join(fs_child, "left_hippocampus"))
                search_dirs.append(os.path.join(fs_child, "right_hippocampus"))
                
            found = []
            for d in search_dirs:
                if os.path.isdir(d):
                    for ext in ("*.nii.gz", "*.nii", "*.vtk"):
                        found.extend(glob.glob(os.path.join(d, ext)))
                        found.extend(glob.glob(os.path.join(d, "*_hippocampus", ext)))
                    
            seen = set()
            for filepath in found:
                norm_p = os.path.normpath(filepath)
                if norm_p not in seen and os.path.isfile(norm_p):
                    basename = os.path.basename(norm_p)
                    # Filter only hippocampus masks
                    norm_low = norm_p.lower()
                    if "hippocampus" in basename.lower() or basename.startswith("lh_") or basename.startswith("rh_") or "left_hippocampus" in norm_low or "right_hippocampus" in norm_low or "_lh." in norm_low or "_rh." in norm_low:
                        seen.add(norm_p)
                        
                        # Determine side
                        side_key = "unknown"
                        side_label = "Hippocampus"
                        if basename.startswith("lh_") or basename.endswith("_lh.nii.gz") or "left_hippocampus" in norm_p:
                            side_key = "lh"
                            side_label = "Left (LH)"
                        elif basename.startswith("rh_") or basename.endswith("_rh.nii.gz") or "right_hippocampus" in norm_p:
                            side_key = "rh"
                            side_label = "Right (RH)"
                            
                        self.all_files.append({
                            "filename": basename,
                            "side": side_label,
                            "side_key": side_key,
                            "filepath": norm_p
                        })
                        
        # Update tab texts with counts
        all_count = len(self.all_files)
        lh_count = sum(1 for f in self.all_files if f["side_key"] == "lh")
        rh_count = sum(1 for f in self.all_files if f["side_key"] == "rh")
        
        self.tab_bar.setTabText(0, f"All ({all_count})")
        self.tab_bar.setTabText(1, f"Left ({lh_count})")
        self.tab_bar.setTabText(2, f"Right ({rh_count})")
        
        self.update_table_display()
        if hasattr(self, 'overlay_cb') and self.overlay_cb.isChecked():
            self.emit_overlay_meshes()

    def update_table_display(self):
        display_files = self.get_current_display_files()
            
        self.results_table.blockSignals(True)
        self.results_table.setRowCount(len(display_files))
        for i, item_info in enumerate(display_files):
            # Col 0: File name
            name_item = QTableWidgetItem(item_info["filename"])
            name_item.setData(Qt.ItemDataRole.UserRole, item_info["filepath"])
            name_item.setData(Qt.ItemDataRole.UserRole + 1, item_info.get("side_key", ""))
            self.results_table.setItem(i, 0, name_item)
            
            # Col 1: Side
            side_item = QTableWidgetItem(item_info["side"])
            side_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if item_info["side_key"] == "lh":
                side_item.setForeground(Qt.GlobalColor.blue)
            elif item_info["side_key"] == "rh":
                side_item.setForeground(Qt.GlobalColor.darkYellow)
            self.results_table.setItem(i, 1, side_item)
            
            # Col 2: Path
            path_item = QTableWidgetItem(item_info["filepath"])
            path_item.setToolTip(item_info["filepath"])
            self.results_table.setItem(i, 2, path_item)
            
        self.results_table.blockSignals(False)
        self.results_table.clearSelection()

    def on_mesh_selected(self):
        selected_rows = sorted(list(set(index.row() for index in self.results_table.selectedIndexes())))
        if not selected_rows:
            # 0 items selected -> clear 3D mesh
            self.signal_mesh_selected.emit("", self.current_side_filter)
            return

        if len(selected_rows) == 1:
            row = selected_rows[0]
            name_item = self.results_table.item(row, 0)
            if name_item:
                filepath = name_item.data(Qt.ItemDataRole.UserRole)
                side_key = name_item.data(Qt.ItemDataRole.UserRole + 1) or self.current_side_filter
                if filepath:
                    self.signal_mesh_selected.emit(filepath, side_key)
            return

        # Multi-select (> 1 meshes selected via Ctrl / Shift)
        filepaths = []
        for row in selected_rows:
            name_item = self.results_table.item(row, 0)
            if name_item:
                fp = name_item.data(Qt.ItemDataRole.UserRole)
                if fp:
                    filepaths.append(fp)

        if filepaths:
            self.signal_mesh_selected.emit(filepaths, self.current_side_filter)

