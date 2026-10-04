import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFrame, QFormLayout,
    QComboBox, QDoubleSpinBox, QSpinBox, QLabel
)
from PyQt6.QtCore import QLocale

class AdvParamsWidget(QWidget):
    """
    Advanced Parameter configuration widget for Groupwise ICP Registration.
    Manages preset mode (Production vs Custom) and parameter adjustments.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.toggle_btn = QPushButton("Advanced Parameters [+]")
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #57606f;
                font-weight: bold;
                font-size: 11px;
                text-align: left;
                padding: 4px 6px;
                border: none;
            }
            QPushButton:hover {
                color: #2f3542;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_advanced_params)
        layout.addWidget(self.toggle_btn)

        self.container = QFrame()
        self.container.setStyleSheet("""
            QFrame {
                border: 1px solid #e1e2e6;
                border-radius: 5px;
                background-color: #fafbfc;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 3px;
                background-color: white;
                color: #34495e;
                font-size: 11px;
            }
            QLabel {
                font-size: 11px;
                color: #2c3e50;
            }
        """)
        adv_form = QFormLayout(self.container)
        adv_form.setContentsMargins(10, 10, 10, 10)
        adv_form.setSpacing(6)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Production Mode (Default)", "Custom Mode"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setLocale(QLocale.c())
        self.spacing_spin.setRange(0.001, 10.0)
        self.spacing_spin.setDecimals(3)
        self.spacing_spin.setValue(0.02)
        self.spacing_spin.setEnabled(False)

        self.voxels_spin = QSpinBox()
        self.voxels_spin.setLocale(QLocale.c())
        self.voxels_spin.setRange(16, 1024)
        self.voxels_spin.setValue(128)
        self.voxels_spin.setEnabled(False)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 1000)
        self.max_iter_spin.setValue(20)
        self.max_iter_spin.setEnabled(False)

        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setLocale(QLocale.c())
        self.tol_spin.setRange(0.000001, 1.0)
        self.tol_spin.setDecimals(6)
        self.tol_spin.setValue(0.00005)
        self.tol_spin.setEnabled(False)

        self.pw_iter_spin = QSpinBox()
        self.pw_iter_spin.setRange(1, 1000)
        self.pw_iter_spin.setValue(100)
        self.pw_iter_spin.setEnabled(False)

        self.pw_tol_spin = QDoubleSpinBox()
        self.pw_tol_spin.setLocale(QLocale.c())
        self.pw_tol_spin.setRange(0.000001, 1.0)
        self.pw_tol_spin.setDecimals(6)
        self.pw_tol_spin.setValue(0.0001)
        self.pw_tol_spin.setEnabled(False)

        self.pw_landmarks_spin = QSpinBox()
        self.pw_landmarks_spin.setRange(10, 5000)
        self.pw_landmarks_spin.setValue(200)
        self.pw_landmarks_spin.setEnabled(False)

        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["NearestNeighbor", "Linear", "BSpline"])
        self.interp_combo.setEnabled(False)

        adv_form.addRow("Preset Mode:", self.mode_combo)
        adv_form.addRow("Output Spacing:", self.spacing_spin)
        adv_form.addRow("Output Voxels:", self.voxels_spin)
        adv_form.addRow("Groupwise Max Iterations:", self.max_iter_spin)
        adv_form.addRow("Groupwise Tolerance:", self.tol_spin)
        adv_form.addRow("Pairwise Max Iterations:", self.pw_iter_spin)
        adv_form.addRow("Pairwise Tolerance:", self.pw_tol_spin)
        adv_form.addRow("Pairwise Landmarks:", self.pw_landmarks_spin)
        adv_form.addRow("Interpolation Mode:", self.interp_combo)

        self.container.setVisible(False)
        layout.addWidget(self.container)

    def toggle_advanced_params(self):
        should_show = self.container.isHidden()
        self.container.setVisible(should_show)
        self.toggle_btn.setText("Advanced Parameters [-]" if should_show else "Advanced Parameters [+]")

    def on_mode_changed(self, index):
        is_custom = (index == 1)
        for widget in [self.spacing_spin, self.voxels_spin, self.max_iter_spin,
                       self.tol_spin, self.pw_iter_spin, self.pw_tol_spin,
                       self.pw_landmarks_spin, self.interp_combo]:
            widget.setEnabled(is_custom)

        if not is_custom:
            self.spacing_spin.setValue(0.02)
            self.voxels_spin.setValue(128)
            self.max_iter_spin.setValue(20)
            self.tol_spin.setValue(0.00005)
            self.pw_iter_spin.setValue(100)
            self.pw_tol_spin.setValue(0.0001)
            self.pw_landmarks_spin.setValue(200)
            self.interp_combo.setCurrentText("NearestNeighbor")

    def get_adv_params(self) -> dict:
        return {
            "spacing": self.spacing_spin.value(),
            "voxels": self.voxels_spin.value(),
            "max_iter": self.max_iter_spin.value(),
            "tolerance": self.tol_spin.value(),
            "pw_iter": self.pw_iter_spin.value(),
            "pw_tol": self.pw_tol_spin.value(),
            "pw_landmarks": self.pw_landmarks_spin.value(),
            "interp": self.interp_combo.currentText()
        }
