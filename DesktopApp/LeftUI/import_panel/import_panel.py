import os
import glob
import json
import re
import shutil
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QLineEdit, QCheckBox, QGroupBox, QMenu, QMessageBox)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint

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

class ImportPanel(QWidget):
    signal_log_message = pyqtSignal(str)
    signal_subject_selected = pyqtSignal(str)
    signal_mesh_selected = pyqtSignal(object, str)
    signal_directories_changed = pyqtSignal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_input_dir = None
        self.last_output_dir = None
        self.loaded_directory = None
        self.imported_data_dir = None
        self.setup_ui()
        
    def setup_ui(self):
        import_layout = QVBoxLayout(self)
        
        # 1. Import Data Properties
        import_group = QGroupBox("Import Data Properties")
        import_group.setStyleSheet("""
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
        ig_layout = QVBoxLayout(import_group)
        ig_layout.setContentsMargins(10, 16, 10, 10)
        ig_layout.setSpacing(8)

        # Row 1: Choose Data Directory
        dir_select_btn = QPushButton("Choose Data Directory")
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
        ig_layout.addWidget(dir_select_btn)

        # Row 2: Choose Output Directory
        out_dir_btn = QPushButton("Choose Output Directory")
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
        ig_layout.addWidget(out_dir_btn)

        # Row 2: Input Path and Output Path side by side
        path_row = QHBoxLayout()
        path_row.setSpacing(8)

        input_col = QVBoxLayout()
        input_col.setSpacing(2)
        input_lbl = QLabel("Input:")
        input_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        self.folder_input.setPlaceholderText("No input directory selected...")
        self.folder_input.setStyleSheet("background: white; border: 1px solid #ced6e0; border-radius: 4px; padding: 5px 8px; font-size: 11px;")
        input_col.addWidget(input_lbl)
        input_col.addWidget(self.folder_input)
        path_row.addLayout(input_col)

        output_col = QVBoxLayout()
        output_col.setSpacing(2)
        output_lbl = QLabel("Output:")
        output_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #444;")
        self.out_folder_input = QLineEdit()
        self.out_folder_input.setReadOnly(True)
        self.out_folder_input.setPlaceholderText("No output directory selected...")
        self.out_folder_input.setStyleSheet("background: white; border: 1px solid #ced6e0; border-radius: 4px; padding: 5px 8px; font-size: 11px;")
        output_col.addWidget(output_lbl)
        output_col.addWidget(self.out_folder_input)
        path_row.addLayout(output_col)

        ig_layout.addLayout(path_row)

        # Row 3: Standalone Import Data Button
        import_action_btn = QPushButton("Import Data")
        import_action_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 12px;
                padding: 8px 10px;
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
        import_action_btn.clicked.connect(self.on_import_clicked)
        ig_layout.addWidget(import_action_btn)
        
        import_layout.addWidget(import_group)

        # 2. Imported Subjects
        subj_group = QGroupBox("Imported Subjects")
        subj_group.setStyleSheet("""
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
        subj_layout = QVBoxLayout(subj_group)
        subj_layout.setContentsMargins(10, 16, 10, 10)
        subj_layout.setSpacing(10)
        self.subjects_table = ToggleTableWidget(0, 1)
        self.subjects_table.setHorizontalHeaderLabels(["Subject name"])
        self.subjects_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.subjects_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.subjects_table.setStyleSheet("""
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
        self.subjects_table.itemSelectionChanged.connect(self.on_subject_selection_changed)
        subj_layout.addWidget(self.subjects_table)
        
        display_layout = QHBoxLayout()
        
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 6px 10px;
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
        self.remove_selected_btn.clicked.connect(self.remove_selected_subject)
        
        self.display_on_click_cb = QCheckBox("Display on click")
        self.display_on_click_cb.setChecked(True)
        
        display_layout.addWidget(self.remove_selected_btn)
        display_layout.addWidget(self.display_on_click_cb)
        subj_layout.addLayout(display_layout)
        
        import_layout.addWidget(subj_group)

    def add_to_history(self, path):
        pass

    def get_imported_data_folder(self):
        """Returns the path to 'data import' or 'data_import' folder in output directory."""
        out_dir = self.out_folder_input.text().strip()
        if not out_dir:
            return getattr(self, 'imported_data_dir', None)
        for cand in ["data import", "data_import"]:
            cand_p = os.path.join(out_dir, cand)
            if os.path.isdir(cand_p):
                return cand_p
        return os.path.join(out_dir, "data import")

    def get_folder(self):
        """
        Returns the effective input directory.
        If 'data import' exists in output folder and contains MRI files, returns 'data import' folder!
        """
        out_dir = self.get_output_folder()
        if out_dir and os.path.isdir(out_dir):
            for cand in ["data import", "data_import"]:
                cand_p = os.path.join(out_dir, cand)
                if os.path.isdir(cand_p):
                    mris = self.find_mri_files(cand_p)
                    if mris:
                        return cand_p
        return self.folder_input.text().strip()

    def get_output_folder(self):
        return self.out_folder_input.text().strip()

    def find_mri_files(self, directory):
        """Finds all valid MRI scan files in directory."""
        if not directory or not os.path.isdir(directory):
            return []
        patterns = ["*.nrrd", "*.nii.gz", "*.nii", "*.mgz"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(directory, pattern)))
            files.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))

        seen = set()
        unique = []
        skip_keywords = ["mask", "seg", "aseg", "aparc", "label", "hippo", ".vtk", ".stl", ".ply"]
        for f in files:
            norm = os.path.normpath(f)
            if norm not in seen:
                seen.add(norm)
                fname = os.path.basename(norm).lower()
                if not any(kw in fname for kw in skip_keywords):
                    unique.append(norm)
        return sorted(unique)

    def select_directory(self):
        initial_dir = self.last_input_dir if getattr(self, 'last_input_dir', None) and os.path.isdir(self.last_input_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select Directory with NIFTI files", initial_dir)
        if folder:
            self.set_source_directory(folder)

    def set_source_directory(self, folder):
        if not folder or not os.path.isdir(folder):
            return
        self.folder_input.setText(folder)
        self.last_input_dir = folder
        mris = self.find_mri_files(folder)
        self.signal_log_message.emit(f"Selected input directory: {folder} ({len(mris)} MRI file(s) found)")
        self.signal_directories_changed.emit(self.get_folder(), self.get_output_folder())

    def select_out_directory(self):
        initial_dir = self.last_output_dir if getattr(self, 'last_output_dir', None) and os.path.isdir(self.last_output_dir) else ("D:/" if os.path.exists("D:/") else "C:/")
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", initial_dir)
        if folder:
            self.out_folder_input.setText(folder)
            self.last_output_dir = folder
            self.load_subjects_from_output(folder)
            self.signal_log_message.emit(f"Selected output directory: {folder}")
            self.signal_directories_changed.emit(self.get_folder(), self.get_output_folder())

    select_output_directory = select_out_directory

    def on_import_clicked(self):
        """
        When Import Data is clicked:
        1. Checks input and output directories.
        2. Creates 'data import' folder in the chosen output directory.
        3. Copies MRI files systematically into 'data import'.
        4. Populates the table with subjects from 'data import'.
        5. Displays the first MRI subject in 3 slice views.
        """
        source_dir = self.folder_input.text().strip()
        out_dir = self.out_folder_input.text().strip()

        if not source_dir or not os.path.isdir(source_dir):
            self.signal_log_message.emit("[ERROR] Please select a valid input directory first.")
            QMessageBox.warning(self, "Input Required", "Please select a valid Input Directory containing MRI scans first.")
            return

        if not out_dir or not os.path.isdir(out_dir):
            self.signal_log_message.emit("[ERROR] Please select an output directory first.")
            QMessageBox.warning(self, "Output Directory Required", "Please select an Output Directory first to store the 'data import' folder.")
            return

        mri_files = self.find_mri_files(source_dir)
        if not mri_files:
            self.signal_log_message.emit(f"[ERROR] No valid MRI files found in {source_dir}")
            QMessageBox.information(self, "No Files", f"No valid MRI images (.nii.gz, .nii, .mgz) found in:\n{source_dir}")
            return

        # Create 'data import' folder inside chosen output directory
        target_import = os.path.join(out_dir, "data import")
        os.makedirs(target_import, exist_ok=True)
        self.imported_data_dir = target_import

        self.signal_log_message.emit(f">>> Importing {len(mri_files)} MRI file(s) into: {target_import}")

        copied_count = 0
        for src in mri_files:
            fname = os.path.basename(src)
            dst = os.path.join(target_import, fname)
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                try:
                    shutil.copy2(src, dst)
                    copied_count += 1
                except Exception as e:
                    self.signal_log_message.emit(f"  [ERROR] Failed to copy {fname}: {e}")

        self.signal_log_message.emit(f">>> Data Imported Successfully. Total {len(mri_files)} MRI scans stored in 'data import'.")

        # Load table directly from 'data import'
        self.load_subjects_from_import_folder(target_import)

        # Notify pipeline to use 'data import'
        self.signal_directories_changed.emit(target_import, out_dir)

    def load_subjects_from_import_folder(self, import_dir):
        """Loads and displays subjects from 'data import' folder in the table."""
        if not import_dir or not os.path.isdir(import_dir):
            return

        files = self.find_mri_files(import_dir)
        self.subjects_table.setRowCount(0)
        self.subjects_table.setRowCount(len(files))

        for row, fpath in enumerate(files):
            fname = os.path.basename(fpath)
            item = QTableWidgetItem(fname)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setData(Qt.ItemDataRole.UserRole, fpath)
            self.subjects_table.setItem(row, 0, item)

        self.signal_log_message.emit(f"Import Panel loaded {len(files)} subject MRI file(s) from 'data import' folder.")

        if self.subjects_table.rowCount() > 0:
            self.subjects_table.blockSignals(True)
            if not self.subjects_table.selectedItems():
                self.subjects_table.selectRow(0)
            self.subjects_table.blockSignals(False)
            if self.isVisible():
                self.display_selected_subject()

    def clear_table_and_views(self):
        """Clears the table and all slice viewers when no data import folder exists."""
        self.subjects_table.setRowCount(0)
        win = self.window()
        if not win or not hasattr(win, 'right_panel'):
            p = self.parent()
            while p is not None:
                if hasattr(p, 'right_panel'):
                    win = p
                    break
                p = p.parent()
        if win and hasattr(win, 'right_panel'):
            win.right_panel.set_view_mode("quad", "Data Importer")
            if hasattr(win.right_panel, 'viewer'):
                win.right_panel.viewer.set_mesh_view_visible(False)
                if hasattr(win.right_panel.viewer, 'slice_mgr'):
                    win.right_panel.viewer.slice_mgr.clear_views()
            if hasattr(win.right_panel, 'clear_gradcam_view'):
                win.right_panel.clear_gradcam_view(render_now=False)
        self.signal_subject_selected.emit("")

    def load_subjects_from_output(self, out_dir):
        """
        Loads subjects ONLY from 'data import' folder created from Import Data button.
        STRICT: If no 'data import' folder exists in out_dir, Data Importer shows NO files and NO images.
        """
        if not out_dir or not os.path.isdir(out_dir):
            self.clear_table_and_views()
            return

        # Check ONLY 'data import' or 'data_import' folder created from Import Data
        for cand in ["data import", "data_import"]:
            cand_p = os.path.join(out_dir, cand)
            if os.path.isdir(cand_p):
                mris = self.find_mri_files(cand_p)
                if mris:
                    self.imported_data_dir = cand_p
                    self.load_subjects_from_import_folder(cand_p)
                    return

        # STRICT: If no 'data import' folder exists, clear table and slice views completely!
        self.clear_table_and_views()
        self.signal_log_message.emit("Data Importer: No 'data import' folder found in output directory. (Table and slice views cleared)")

    def load_subjects_from_directory(self, directory, clear_existing=False):
        """
        Sets the source input directory.
        STRICT: Never populates the table or displays images from the raw input directory.
        Only when 'Import Data' is clicked (or if 'data import' already exists in output folder)
        will Data Importer show subjects.
        """
        if not directory or not os.path.isdir(directory):
            return

        self.set_source_directory(directory)

        # STRICT: Only populate from output's 'data import' folder if it exists.
        out_dir = self.get_output_folder().strip()
        if out_dir and os.path.isdir(out_dir):
            for cand in ["data import", "data_import"]:
                cand_p = os.path.join(out_dir, cand)
                if os.path.isdir(cand_p):
                    mris = self.find_mri_files(cand_p)
                    if mris:
                        self.load_subjects_from_import_folder(cand_p)
                        return

        # If no 'data import' folder exists, table and slice views MUST stay empty!
        self.clear_table_and_views()

    def on_subject_selection_changed(self):
        if not self.isVisible():
            return
        if self.display_on_click_cb.isChecked():
            self.display_selected_subject()

    def display_selected_subject(self):
        """Displays selected MRI scan in the 3 slice views (Axial, Coronal, Sagittal)."""
        selected_items = self.subjects_table.selectedItems()
        if not selected_items:
            win = self.window()
            if win and hasattr(win, 'right_panel'):
                win.right_panel.set_view_mode("quad", "Data Importer")
                if hasattr(win.right_panel, 'viewer'):
                    win.right_panel.viewer.set_mesh_view_visible(False)
                    if hasattr(win.right_panel.viewer, 'slice_mgr'):
                        win.right_panel.viewer.slice_mgr.clear_views()
            return
        
        selected_rows = list(set([item.row() for item in selected_items]))
        if len(selected_rows) > 1:
            return
            
        row = selected_rows[0]
        item = self.subjects_table.item(row, 0)
        if not item:
            self.clear_table_and_views()
            return
            
        subject_name = item.text()
        filepath = item.data(Qt.ItemDataRole.UserRole)
        if not filepath or not os.path.isfile(filepath):
            self.clear_table_and_views()
            return

        # Ensure right panel stays in 3 slice views (quad) mode without 3D mesh
        win = self.window()
        if win and hasattr(win, 'right_panel'):
            win.right_panel.set_view_mode("quad", "Data Importer")
            if hasattr(win.right_panel, 'viewer'):
                win.right_panel.viewer.set_mesh_view_visible(False)
            if hasattr(win.right_panel, 'clear_gradcam_view'):
                win.right_panel.clear_gradcam_view(render_now=False)

        self.signal_log_message.emit(f"Displaying subject: {subject_name}")
        if filepath.lower().endswith((".vtk", ".stl", ".ply")):
            self.signal_mesh_selected.emit(filepath, "all")
        else:
            self.signal_subject_selected.emit(filepath)

    def remove_selected_subject(self):
        selected_items = self.subjects_table.selectedItems()
        if not selected_items:
            return
            
        selected_rows = sorted(list(set([item.row() for item in selected_items])), reverse=True)
        count = len(selected_rows)
        if count == 0:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove {count} selected subject(s) from the list?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )

        if reply != QMessageBox.StandardButton.Ok:
            return
        
        for row in selected_rows:
            item = self.subjects_table.item(row, 0)
            subject_name = item.text() if item else f"Row {row}"
            self.subjects_table.removeRow(row)
            self.signal_log_message.emit(f"Removed subject from list: {subject_name}")
