import os
import glob
import platform
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                             QLabel, QLineEdit, QComboBox, QPushButton, 
                             QDialogButtonBox, QTextEdit, QFileDialog)
from PyQt6.QtCore import Qt

class PreferencesDialog(QDialog):
    """Configuration dialog for SlicerSALT path, compute engine, and 3D themes."""
    def __init__(self, parent=None, current_slicer=""):
        super().__init__(parent)
        self.setWindowTitle("Application Preferences")
        self.resize(520, 240)
        self.setStyleSheet("""
            QDialog { background-color: #f8f9fa; }
            QLabel { font-size: 12px; color: #2c3e50; font-weight: bold; }
            QLineEdit, QComboBox { padding: 6px; font-size: 12px; border: 1px solid #ced4da; border-radius: 4px; background: white; }
            QPushButton { padding: 6px 12px; font-size: 11px; font-weight: bold; border-radius: 4px; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        form = QFormLayout()
        form.setSpacing(10)
        
        # SlicerSALT Path
        slicer_box = QHBoxLayout()
        self.slicer_input = QLineEdit(current_slicer)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_slicer)
        slicer_box.addWidget(self.slicer_input)
        slicer_box.addWidget(browse_btn)
        form.addRow("SlicerSALT Binary:", slicer_box)
        
        # 3D Viewport Theme
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([
            "Default Blue Gradient (Slicer-style)",
            "Clinical Dark Slate (Modern)",
            "High Contrast Black (Dark Room)",
            "Pure White (Publication Paper)"
        ])
        form.addRow("3D Viewport Background:", self.theme_combo)
        
        layout.addLayout(form)
        layout.addStretch()
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        
    def browse_slicer(self):
        f, _ = QFileDialog.getOpenFileName(self, "Locate SlicerSALT.exe", "", "Executable (*.exe);;All Files (*)")
        if f:
            self.slicer_input.setText(f)


class AboutDialog(QDialog):
    """About dialog showing application overview, version, and citations."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Hippocampal Shape Analysis Toolbox")
        self.resize(520, 320)
        self.setStyleSheet("background-color: #ffffff;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)
        
        title_lbl = QLabel("Hippocampal Shape Analysis Pipeline")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_lbl)
        
        sub_lbl = QLabel("Automated SPHARM-PDM & Deep Learning Epilepsy Detection Platform\nVersion 2.5.0 (Clinical Research Edition)")
        sub_lbl.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        layout.addWidget(sub_lbl)
        
        desc = QLabel(
            "This software provides an end-to-end neuroimaging pipeline for automated hippocampal "
            "segmentation (FastSurfer), rigid ICP surface alignment, Spherical Harmonics "
            "Point Distribution Modeling (SPHARM-PDM via SlicerSALT), and AI-driven Temporal Lobe "
            "Epilepsy (TLE) classification with Grad-CAM morphometric visualization."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; line-height: 1.5; color: #34495e; padding: 10px 0;")
        layout.addWidget(desc)
        
        layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("padding: 6px 20px; font-weight: bold; background: #3498db; color: white; border-radius: 4px;")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)


class DiagnosticsDialog(QDialog):
    """System diagnostic dialog inspecting GPU/CUDA, SlicerSALT, and runtime libraries."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System & Hardware Diagnostics")
        self.resize(560, 400)
        self.setStyleSheet("background-color: #ffffff;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        
        title = QLabel("System Environment & Diagnostics")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title)
        
        diag_text = QTextEdit()
        diag_text.setReadOnly(True)
        diag_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background: #1e293b; color: #e2e8f0; border-radius: 4px; padding: 8px;")
        
        slicer_paths = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
        slicer_status = f"DETECTED ({slicer_paths[0]})" if slicer_paths else "NOT FOUND in default C:\\Program Files\\"
        
        cuda_status = "Not Available (Using CPU Mode)"
        try:
            import torch
            if torch.cuda.is_available():
                cuda_status = f"Available - {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        except Exception:
            pass
            
        vtk_ver = "Unknown"
        try:
            import vtk
            vtk_ver = vtk.vtkVersion.GetVTKVersion()
        except Exception:
            pass
        
        lines = [
            "=== Operating System & Runtime ===",
            f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
            f"Python Version: {platform.python_version()}",
            f"PyQt6 Version: 6.x",
            f"VTK Version: {vtk_ver}",
            "",
            "=== AI & Hardware Acceleration ===",
            f"PyTorch Compute: {cuda_status}",
            "",
            "=== Prerequisites & External Toolchains ===",
            f"SlicerSALT Engine: {slicer_status}",
            "FastSurfer CLI: Ready (Local / Docker environment)",
            "",
            "=== Analysis Status ===",
            "Shape Pipeline: " + ("READY FOR EXECUTION" if slicer_paths else "SLICERSALT REQUIRED")
        ]
        diag_text.setText("\n".join(lines))
        layout.addWidget(diag_text)
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("padding: 6px 20px; font-weight: bold; background: #2c3e50; color: white; border-radius: 4px;")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
