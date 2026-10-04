import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QRadioButton, QButtonGroup, QSlider, QFrame
)
from PyQt6.QtCore import Qt

class GradCamControlsWidget(QWidget):
    """
    Manages Section 4: 3D Grad-CAM & Atrophy Visualization controls.
    Includes side selector, heatmap colormap modes, latent SD trajectory stepper,
    continuous slider (-3.0 to +3.0 SD), and milestone jump buttons.
    """
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.p = panel
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        cam_group = QGroupBox("4. 3D Grad-CAM & Atrophy Visualization")
        cam_group.setStyleSheet("""
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
        c_layout = QVBoxLayout(cam_group)
        c_layout.setContentsMargins(10, 16, 10, 10)
        c_layout.setSpacing(6)

        # Active view hemisphere
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("Side:"))
        self.rb_cam_left = QRadioButton("Left (LH)")
        self.rb_cam_right = QRadioButton("Right (RH)")
        self.rb_cam_left.setChecked(True)
        self.cam_side_group = QButtonGroup(self)
        self.cam_side_group.addButton(self.rb_cam_left, 0)
        self.cam_side_group.addButton(self.rb_cam_right, 1)
        self.cam_side_group.buttonClicked.connect(self.p.update_3d_view)
        side_row.addWidget(self.rb_cam_left)
        side_row.addWidget(self.rb_cam_right)
        side_row.addStretch()
        c_layout.addLayout(side_row)

        # Visualization mode options
        c_layout.addWidget(QLabel("Colormap Heatmap Mode:"))
        self.rb_dist = QRadioButton("Deformation Mag (mm)")
        self.rb_signed = QRadioButton("Signed Atrophy (Inward/Expansion)")
        self.rb_gradcam = QRadioButton("Grad-CAM Attention")
        self.rb_dist.setChecked(True)

        self.cam_mode_group = QButtonGroup(self)
        self.cam_mode_group.addButton(self.rb_dist, 0)
        self.cam_mode_group.addButton(self.rb_signed, 1)
        self.cam_mode_group.addButton(self.rb_gradcam, 2)
        self.cam_mode_group.buttonClicked.connect(self.p.update_3d_view)

        c_layout.addWidget(self.rb_dist)
        c_layout.addWidget(self.rb_signed)
        c_layout.addWidget(self.rb_gradcam)

        # Latent SD Trajectory & Atrophy Stepper Controls
        sd_box = QFrame()
        sd_box.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: 1px solid #ced6e0;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        sd_box_layout = QVBoxLayout(sd_box)
        sd_box_layout.setContentsMargins(4, 4, 4, 4)
        sd_box_layout.setSpacing(4)

        # Header Row with prominent SD indicator badge and quick Reset button
        sd_hdr_row = QHBoxLayout()
        sd_title = QLabel("SD Trajectory:")
        sd_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #2c3e50;")

        self.sd_reset_btn = QPushButton("Reset")
        self.sd_reset_btn.setToolTip("Reset latent SD trajectory to 0.0 SD (Mean) [Hotkeys: 0, R, Space, Home]")
        self.sd_reset_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 2px 8px;
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
        self.sd_reset_btn.clicked.connect(lambda: self.p.set_sd_value(0.0))

        self.sd_val_badge = QLabel("0.0 SD (Mean)")
        self.sd_val_badge.setStyleSheet("""
            QLabel {
                background: #ebf5fb;
                color: #2980b9;
                font-weight: bold;
                font-size: 11px;
                padding: 2px 6px;
                border-radius: 3px;
                border: 1px solid #aed6f1;
            }
        """)
        sd_hdr_row.addWidget(sd_title)
        sd_hdr_row.addStretch()
        sd_hdr_row.addWidget(self.sd_reset_btn)
        sd_hdr_row.addWidget(self.sd_val_badge)
        sd_box_layout.addLayout(sd_hdr_row)

        # Stepper buttons + Continuous Slider Row
        slider_row = QHBoxLayout()
        sd_step_btn_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 11px;
                padding: 4px 8px;
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
        """
        self.sd_prev_btn = QPushButton("-0.1 SD")
        self.sd_prev_btn.setStyleSheet(sd_step_btn_style)
        self.sd_prev_btn.setToolTip("Step latent deformation backward by -0.1 SD (or press Left Arrow key)")
        self.sd_prev_btn.clicked.connect(lambda: self.p.step_sd(-0.1))

        self.sd_slider = QSlider(Qt.Orientation.Horizontal)
        self.sd_slider.setRange(0, 60)
        self.sd_slider.setValue(30)  # 30 = 0.0 SD (Mean)
        self.sd_slider.setTickInterval(10)
        self.sd_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sd_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sd_slider.valueChanged.connect(self.p.on_sd_slider_changed)

        self.sd_next_btn = QPushButton("+0.1 SD")
        self.sd_next_btn.setStyleSheet(sd_step_btn_style)
        self.sd_next_btn.setToolTip("Step latent deformation forward by +0.1 SD (or press Right Arrow key)")
        self.sd_next_btn.clicked.connect(lambda: self.p.step_sd(+0.1))

        slider_row.addWidget(self.sd_prev_btn)
        slider_row.addWidget(self.sd_slider)
        slider_row.addWidget(self.sd_next_btn)
        sd_box_layout.addLayout(slider_row)

        # Milestone quick buttons row
        milestone_row = QHBoxLayout()
        milestone_row.setSpacing(2)
        m_btn_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e9ecef);
                color: #2c3e50;
                font-weight: bold;
                font-size: 10px;
                padding: 3px 2px;
                border: 1px solid #ced6e0;
                border-radius: 3px;
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
        """
        btn_m3 = QPushButton("-3 SD")
        btn_m3.setToolTip("Severe Atrophy (-3.0 SD)")
        btn_m2 = QPushButton("-2 SD")
        btn_m2.setToolTip("Atrophy (-2.0 SD)")
        btn_m1 = QPushButton("-1 SD")
        btn_m1.setToolTip("Mild Atrophy (-1.0 SD)")
        btn_mean = QPushButton("Mean")
        btn_mean.setToolTip("Normative Mean (0.0 SD)")
        btn_p1 = QPushButton("+1 SD")
        btn_p1.setToolTip("Expansion (+1.0 SD)")
        btn_p2 = QPushButton("+2 SD")
        btn_p2.setToolTip("Expansion (+2.0 SD)")
        btn_p3 = QPushButton("+3 SD")
        btn_p3.setToolTip("Expansion (+3.0 SD)")

        for b in (btn_m3, btn_m2, btn_m1, btn_mean, btn_p1, btn_p2, btn_p3):
            b.setFixedHeight(24)
            b.setStyleSheet(m_btn_style)

        btn_m3.clicked.connect(lambda: self.p.set_sd_value(-3.0))
        btn_m2.clicked.connect(lambda: self.p.set_sd_value(-2.0))
        btn_m1.clicked.connect(lambda: self.p.set_sd_value(-1.0))
        btn_mean.clicked.connect(lambda: self.p.set_sd_value(0.0))
        btn_p1.clicked.connect(lambda: self.p.set_sd_value(1.0))
        btn_p2.clicked.connect(lambda: self.p.set_sd_value(2.0))
        btn_p3.clicked.connect(lambda: self.p.set_sd_value(3.0))

        milestone_row.addWidget(btn_m3)
        milestone_row.addWidget(btn_m2)
        milestone_row.addWidget(btn_m1)
        milestone_row.addWidget(btn_mean)
        milestone_row.addWidget(btn_p1)
        milestone_row.addWidget(btn_p2)
        milestone_row.addWidget(btn_p3)
        sd_box_layout.addLayout(milestone_row)

        hint_lbl = QLabel("<i>Hotkeys: <b>Left / Right</b> Step SD  |  <b>0 / R / Space</b> Reset</i>")
        hint_lbl.setStyleSheet("font-size: 9px; color: #576574; font-style: italic;")
        sd_box_layout.addWidget(hint_lbl)

        c_layout.addWidget(sd_box)
        layout.addWidget(cam_group)
