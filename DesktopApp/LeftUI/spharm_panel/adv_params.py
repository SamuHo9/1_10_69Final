from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFrame, QFormLayout,
    QComboBox, QSpinBox, QCheckBox, QLabel
)

class AdvParamsWidget(QWidget):
    """
    Advanced Parameter configuration widget for SPHARM-PDM Pipeline.
    Manages presets (Production, Fast Test, Custom) and parameter adjustments.
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
            QSpinBox, QComboBox {
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
        self.mode_combo.addItems(["Production Mode (Default)", "Fast Test Mode", "Custom Mode"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(50, 5000)
        self.iter_spin.setValue(1000)
        self.iter_spin.setSingleStep(100)
        self.iter_spin.setEnabled(False)

        self.subdiv_spin = QSpinBox()
        self.subdiv_spin.setRange(1, 30)
        self.subdiv_spin.setValue(10)
        self.subdiv_spin.setEnabled(False)

        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 30)
        self.degree_spin.setValue(12)
        self.degree_spin.setEnabled(False)

        self.regen_cb = QCheckBox("Regenerate SPHARM Only (Reuse existing _surf.vtk and _para.vtk)")
        self.regen_cb.setChecked(False)

        adv_form.addRow("Preset Mode:", self.mode_combo)
        adv_form.addRow("GenParaMesh Iterations:", self.iter_spin)
        adv_form.addRow("Subdivision Level:", self.subdiv_spin)
        adv_form.addRow("SPHARM Degree:", self.degree_spin)
        adv_form.addRow("", self.regen_cb)

        self.container.setVisible(False)
        layout.addWidget(self.container)

    def toggle_advanced_params(self):
        should_show = self.container.isHidden()
        self.container.setVisible(should_show)
        self.toggle_btn.setText("Advanced Parameters [-]" if should_show else "Advanced Parameters [+]")

    def on_mode_changed(self, index):
        if index == 0:  # Production
            self.iter_spin.setValue(1000)
            self.subdiv_spin.setValue(10)
            self.degree_spin.setValue(12)
            self.iter_spin.setEnabled(False)
            self.subdiv_spin.setEnabled(False)
            self.degree_spin.setEnabled(False)
        elif index == 1:  # Fast Test
            self.iter_spin.setValue(200)
            self.subdiv_spin.setValue(5)
            self.degree_spin.setValue(6)
            self.iter_spin.setEnabled(False)
            self.subdiv_spin.setEnabled(False)
            self.degree_spin.setEnabled(False)
        elif index == 2:  # Custom
            self.iter_spin.setEnabled(True)
            self.subdiv_spin.setEnabled(True)
            self.degree_spin.setEnabled(True)

    def get_adv_params(self) -> dict:
        return {
            "iterations": self.iter_spin.value(),
            "subdiv": self.subdiv_spin.value(),
            "degree": self.degree_spin.value(),
            "regen_only": self.regen_cb.isChecked()
        }
