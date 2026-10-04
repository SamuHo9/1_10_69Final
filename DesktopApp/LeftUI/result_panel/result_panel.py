import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QHeaderView, QGroupBox,
    QRadioButton, QButtonGroup, QProgressBar, QFrame,
    QScrollArea, QTabBar
)
from PyQt6.QtCore import Qt, pyqtSignal

from models.predictor import HippocampalPredictor
from .table_widget import ToggleTableWidget
from .latent_explorer import LatentShapeExplorer
from .gradcam_controls import GradCamControlsWidget
from .eval_manager import EvaluationManager

class ResultPanel(QWidget):
    """
    Result & Diagnostic Panel Coordinator.
    Coordinates trained ResNet1D model inference, structured output_Result persistence,
    and interactive 3D Grad-CAM attention and deformation heatmap visualization.
    """
    signal_log_message = pyqtSignal(str)
    signal_gradcam_mesh_requested = pyqtSignal(str, str, str, str, str, float)
    signal_patient_overlay_requested = pyqtSignal(str, bool, float, str)
    signal_clear_gradcam_requested = pyqtSignal()
    signal_batch_prediction_finished = pyqtSignal(bool)

    def __init__(self, parent=None, get_input_folder=None, get_output_folder=None):
        super().__init__(parent)
        self.get_input_folder = get_input_folder
        self.get_output_folder = get_output_folder

        self.predictor = HippocampalPredictor()
        self.current_sd_step_idx = 30  # 30 corresponds to 0.0 SD (Mean)
        self.current_sd_value = 0.0
        self.current_tab_filter = "all"
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.eval_mgr = EvaluationManager(self)
        self.cam_controls = GradCamControlsWidget(self)
        self.latent_explorer = LatentShapeExplorer(self)

        self.setup_ui()

    # Delegate property access for backward compatibility
    @property
    def current_patient_data(self):
        return self.eval_mgr.current_patient_data

    @current_patient_data.setter
    def current_patient_data(self, val):
        self.eval_mgr.current_patient_data = val

    @property
    def last_prediction_results(self):
        return self.eval_mgr.last_prediction_results

    @last_prediction_results.setter
    def last_prediction_results(self, val):
        self.eval_mgr.last_prediction_results = val

    @property
    def all_evaluation_results(self):
        return self.eval_mgr.all_evaluation_results

    @all_evaluation_results.setter
    def all_evaluation_results(self, val):
        self.eval_mgr.all_evaluation_results = val

    @property
    def rb_cam_left(self):
        return self.cam_controls.rb_cam_left

    @property
    def rb_cam_right(self):
        return self.cam_controls.rb_cam_right

    @property
    def rb_dist(self):
        return self.cam_controls.rb_dist

    @property
    def rb_signed(self):
        return self.cam_controls.rb_signed

    @property
    def rb_gradcam(self):
        return self.cam_controls.rb_gradcam

    @property
    def sd_slider(self):
        return self.cam_controls.sd_slider

    @property
    def sd_val_badge(self):
        return self.cam_controls.sd_val_badge

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.horizontalScrollBar().setEnabled(False)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)

        # ---------------------------------------------------------------------
        # 1. Diagnostic Model
        # ---------------------------------------------------------------------
        header_group = QGroupBox("1. Diagnostic Model")
        header_group.setStyleSheet("""
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
        h_layout = QVBoxLayout(header_group)
        h_layout.setContentsMargins(10, 16, 10, 10)
        h_layout.setSpacing(6)

        # Internal directory input for pipeline & left_panel integration
        self.spharm_dir_input = QLineEdit()
        self.spharm_dir_input.textChanged.connect(self.on_spharm_dir_changed)

        model_badge = QLabel("Architecture: <b>(1D-CNN) + PLS-DA</b> | Weights: <b>Trained & Frozen</b>")
        model_badge.setWordWrap(True)
        model_badge.setStyleSheet("font-size: 10px; color: #576574; background: #ffffff; padding: 4px 6px; border-radius: 3px; border: 1px solid #e9ecef;")
        h_layout.addWidget(model_badge)
        container_layout.addWidget(header_group)

        # ---------------------------------------------------------------------
        # 2. Execution & Batch Evaluation
        # ---------------------------------------------------------------------
        exec_group = QGroupBox("2. Batch Evaluation")
        exec_group.setStyleSheet(header_group.styleSheet())
        e_layout = QVBoxLayout(exec_group)
        e_layout.setContentsMargins(10, 16, 10, 10)
        e_layout.setSpacing(6)

        # Internal directory input for pipeline & left_panel integration
        self.result_dir_input = QLineEdit()

        eval_opts_layout = QHBoxLayout()
        eval_lbl = QLabel("Target:")
        eval_lbl.setStyleSheet("font-size: 11px; color: #2c3e50; font-weight: bold;")
        eval_opts_layout.addWidget(eval_lbl)
        self.side_eval_group = QButtonGroup(self)
        self.rb_both = QRadioButton("Both")
        self.rb_left = QRadioButton("Left (LH)")
        self.rb_right = QRadioButton("Right (RH)")
        self.rb_both.setChecked(True)
        self.side_eval_group.addButton(self.rb_both, 0)
        self.side_eval_group.addButton(self.rb_left, 1)
        self.side_eval_group.addButton(self.rb_right, 2)
        eval_opts_layout.addWidget(self.rb_both)
        eval_opts_layout.addWidget(self.rb_left)
        eval_opts_layout.addWidget(self.rb_right)
        eval_opts_layout.addStretch()
        e_layout.addLayout(eval_opts_layout)

        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.setStyleSheet("""
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
            QPushButton:disabled {
                background: #f1f2f6;
                color: #a4b0be;
                border: 1px solid #dfe4ea;
            }
        """)
        self.predict_btn.clicked.connect(self.run_batch_prediction)
        e_layout.addWidget(self.predict_btn)

        self.batch_prog_bar = QProgressBar()
        self.batch_prog_bar.setRange(0, 100)
        self.batch_prog_bar.setValue(0)
        self.batch_prog_bar.setFixedHeight(14)
        self.batch_prog_bar.setTextVisible(True)
        self.batch_prog_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ced6e0;
                border-radius: 3px;
                text-align: center;
                background-color: #ecf0f1;
                font-size: 9px;
                font-weight: bold;
                color: #2c3e50;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 2px;
            }
        """)
        self.batch_prog_bar.setVisible(False)
        e_layout.addWidget(self.batch_prog_bar)

        self.batch_status_hint = QLabel("")
        self.batch_status_hint.setWordWrap(True)
        self.batch_status_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        e_layout.addWidget(self.batch_status_hint)
        container_layout.addWidget(exec_group)

        # ---------------------------------------------------------------------
        # 3. Evaluation Results & Meshes Table
        # ---------------------------------------------------------------------
        results_group = QGroupBox("3. Diagnostic Results & Evaluated Meshes")
        results_group.setStyleSheet(header_group.styleSheet())
        r_layout = QVBoxLayout(results_group)
        r_layout.setContentsMargins(10, 16, 10, 10)
        r_layout.setSpacing(6)

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
        r_layout.addWidget(self.tab_bar)

        self.results_table = ToggleTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels([
            "Subject", "Side", "Diagnosis", "Probability", "Mesh (.vtk)"
        ])
        h_header = self.results_table.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.results_table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
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
        self.results_table.setFixedHeight(185)
        self.results_table.itemSelectionChanged.connect(self.on_result_selected)
        r_layout.addWidget(self.results_table)
        container_layout.addWidget(results_group)

        # ---------------------------------------------------------------------
        # 4. Modular 3D Grad-CAM & Deformation Heatmap Controls
        # ---------------------------------------------------------------------
        container_layout.addWidget(self.cam_controls)
        container_layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        self.update_run_button_state()

    # =========================================================================
    # Path Resolution & Navigation
    # =========================================================================
    def get_default_output_dir(self, resolved_base=None):
        if self.get_output_folder:
            out_base = self.get_output_folder().strip()
            if out_base:
                return os.path.join(out_base, "output_Result")
        if resolved_base:
            return os.path.join(resolved_base, "output_Result")
        return ""

    def browse_spharm_dir(self):
        start_dir = self.get_output_folder() if self.get_output_folder else os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Select SPHARM Results Directory", start_dir)
        if folder:
            self.spharm_dir_input.setText(folder)
            self.on_spharm_dir_changed()

    def browse_result_dir(self):
        start_dir = self.result_dir_input.text().strip() or (self.get_output_folder() if self.get_output_folder else os.getcwd())
        folder = QFileDialog.getExistingDirectory(self, "Select Result Output Directory (output_Result)", start_dir)
        if folder:
            self.result_dir_input.setText(folder)
            self.load_existing_results()

    def update_run_button_state(self):
        # 1. Resolve directories from global output folder if not set
        out_base = self.get_output_folder().strip() if self.get_output_folder else ""
        if out_base and os.path.isdir(out_base):
            spharm_cand = os.path.join(out_base, "output_SPHARM")
            if os.path.isdir(spharm_cand):
                self.spharm_dir_input.setText(spharm_cand)
            elif not self.spharm_dir_input.text().strip():
                self.spharm_dir_input.setText(out_base)

            res_cand = os.path.join(out_base, "output_Result")
            if not self.result_dir_input.text().strip():
                self.result_dir_input.setText(res_cand)

        # 2. Check if Result output already exists -> auto-load and display immediately
        target_res = self.result_dir_input.text().strip()
        if not target_res and out_base:
            target_res = os.path.join(out_base, "output_Result")
            self.result_dir_input.setText(target_res)

        if target_res and os.path.isdir(target_res):
            summary_json = os.path.join(target_res, "evaluation_summary.json")
            if os.path.isfile(summary_json) and not self.all_evaluation_results:
                self.load_existing_results()

        # 3. Check if SPHARM output exists
        subjs = self.eval_mgr.discover_spharm_subjects()
        count = len(subjs)

        if count == 0:
            self.predict_btn.setEnabled(False)
            self.predict_btn.setToolTip("SPHARM output not found. Please run SPHARM-PDM first.")
            if not self.all_evaluation_results:
                self.batch_status_hint.setText("⚠️ SPHARM output not found. Please run SPHARM-PDM first.")
                self.batch_status_hint.setStyleSheet("color: #e67e22; font-size: 11px;")
        else:
            self.predict_btn.setEnabled(True)
            self.predict_btn.setToolTip(f"Click to run diagnostic prediction for {count} subject(s)")
            if self.all_evaluation_results:
                self.batch_status_hint.setText(f"✓ Evaluation results loaded ({len(self.all_evaluation_results)} subjects). Ready to view or re-run.")
                self.batch_status_hint.setStyleSheet("color: #27ae60; font-size: 11px;")
            else:
                self.batch_status_hint.setText(f"✓ SPHARM output ready ({count} subjects found). Ready to run prediction.")
                self.batch_status_hint.setStyleSheet("color: #2980b9; font-size: 11px;")

    def on_spharm_dir_changed(self):
        self.update_run_button_state()

    def on_panel_activated(self):
        self.update_run_button_state()

    # Forwarding methods for EvaluationManager integration
    def run_batch_prediction(self):
        self.eval_mgr.run_batch_prediction()

    def load_existing_results(self):
        self.eval_mgr.load_existing_results()

    def on_tab_changed(self, index):
        self.eval_mgr.on_tab_changed(index)

    def populate_results_table(self):
        self.eval_mgr.populate_results_table()

    def on_result_selected(self):
        self.eval_mgr.on_result_selected()

    def display_prediction_results(self, results, selected_side=None):
        self.eval_mgr.display_prediction_results(results, selected_side)

    def clear_view(self):
        self.eval_mgr.clear_view()

    def copy_summary(self):
        self.eval_mgr.copy_summary()

    # Forwarding methods for LatentShapeExplorer integration
    def keyPressEvent(self, event):
        if not self.latent_explorer.handle_key_press(event):
            super().keyPressEvent(event)

    def step_sd(self, delta: float):
        self.latent_explorer.step_sd(delta)

    def set_sd_value(self, sd_val: float):
        self.latent_explorer.set_sd_value(sd_val)

    def on_sd_slider_changed(self, val: int):
        self.latent_explorer.on_sd_slider_changed(val)

    def update_3d_view(self):
        self.latent_explorer.update_3d_view()

    def find_step_mesh(self, side: str, sd_val: float, cohort="All_Augment_tain", component="PLS1"):
        return self.latent_explorer.find_step_mesh(side, sd_val, cohort, component)

    def on_patient_overlay_toggled(self):
        side = "left" if self.rb_cam_left.isChecked() else "right"
        self.signal_patient_overlay_requested.emit("", False, 0.0, side)
