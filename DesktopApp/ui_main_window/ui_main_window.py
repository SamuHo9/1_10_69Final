import os
import sys
import time
import glob
import webbrowser
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, 
                             QMessageBox, QApplication, QScrollArea, QFrame, 
                             QLineEdit, QTextEdit, QPlainTextEdit, QFileDialog)
from PyQt6.QtCore import Qt, QEvent

from LeftUI.left_panel import LeftPanel
from RightUI.right_panel import RightPanel
from .dialogs import PreferencesDialog, AboutDialog, DiagnosticsDialog
from .console_widget import ExecutionConsole
from .menu_toolbar import setup_menus_and_toolbar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hippocampal Shape Analysis Pipeline (Slicer-style)")
        self.resize(1200, 800)

        # 1. Menus and Toolbar (delegated to menu_toolbar.py)
        setup_menus_and_toolbar(self)

        # 2. Central Layout with Splitters
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.v_splitter.setHandleWidth(8)
        self.v_splitter.setChildrenCollapsible(True)
        self.v_splitter.setStyleSheet("""
            QSplitter::handle:vertical {
                background: #34495e;
                height: 8px;
                border-top: 1px solid #4a627a;
                border-bottom: 1px solid #243342;
            }
            QSplitter::handle:vertical:hover { background: #3498db; }
        """)
        main_layout.addWidget(self.v_splitter)

        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.h_splitter.setHandleWidth(6)
        self.h_splitter.setStyleSheet("""
            QSplitter::handle:horizontal {
                background: #dcdde1;
                width: 6px;
            }
            QSplitter::handle:horizontal:hover { background: #3498db; }
        """)

        # 3. Panels
        self.left_panel = LeftPanel()
        self.right_panel = RightPanel(self)

        # Wrap LeftPanel in scroll area
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.left_scroll.horizontalScrollBar().setEnabled(False)
        self.left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.left_scroll.setWidget(self.left_panel)
        self.left_scroll.setMinimumWidth(380)

        # Wrap RightPanel in scroll area
        self.right_scroll = QScrollArea()
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.right_panel.setMinimumHeight(450)
        self.right_panel.setMinimumWidth(500)
        self.right_scroll.setWidget(self.right_panel)

        self.h_splitter.addWidget(self.left_scroll)
        self.h_splitter.addWidget(self.right_scroll)
        self.h_splitter.setSizes([450, 830])
        self.saved_h_splitter_sizes = [450, 830]
        self.h_splitter.splitterMoved.connect(self.on_h_splitter_moved)

        self.v_splitter.addWidget(self.h_splitter)

        # 4. Execution Console Widget
        self.console = ExecutionConsole(self)
        self.console.signal_close_requested.connect(lambda: self.set_terminal_visible(False))
        self.console.signal_expand_requested.connect(self.toggle_console_expand)
        self.v_splitter.addWidget(self.console)
        self.v_splitter.setStretchFactor(0, 1)
        self.v_splitter.setStretchFactor(1, 0)
        self.v_splitter.setSizes([600, 135])
        self.v_splitter.splitterMoved.connect(self.on_splitter_moved)

        self.is_terminal_fullscreen = False
        self.saved_splitter_sizes = [600, 135]

        # 5. Connect Signals Between Panels
        self.left_panel.signal_subject_selected.connect(self.right_panel.display_subject)
        self.left_panel.signal_mesh_selected.connect(self.right_panel.display_mesh)
        self.left_panel.signal_log_message.connect(self.log)
        self.right_panel.signal_log_message.connect(self.log)
        self.left_panel.signal_template_toggled.connect(self.right_panel.viewer.set_template_visible)
        self.left_panel.signal_overlay_all_toggled.connect(self.on_overlay_all_toggled)
        self.right_panel.signal_overlay_toggled.connect(self.left_panel.set_overlay_visible)
        self.left_panel.signal_side_changed.connect(self.right_panel.set_side_filter)
        if hasattr(self.left_panel, 'signal_gradcam_mesh_requested'):
            self.left_panel.signal_gradcam_mesh_requested.connect(self.right_panel.display_gradcam_mesh)
            self.left_panel.signal_patient_overlay_requested.connect(self.right_panel.set_patient_overlay)
            self.left_panel.signal_clear_gradcam_requested.connect(self.right_panel.clear_gradcam_view)
        if hasattr(self.left_panel, 'signal_diagnostic_info'):
            self.left_panel.signal_diagnostic_info.connect(self.right_panel.viewer.set_diagnostic_info)
        if hasattr(self.right_panel, 'signal_step_sd'):
            self.right_panel.signal_step_sd.connect(self.on_step_sd_requested)
        if hasattr(self.right_panel, 'signal_reset_sd'):
            self.right_panel.signal_reset_sd.connect(self.on_reset_sd_requested)

        self.module_combo.currentTextChanged.connect(self.on_module_changed)
        self.on_module_changed(self.module_combo.currentText())

        # Event filter to ensure arrow keys scrub SD cleanly
        QApplication.instance().installEventFilter(self)

        self.log("SlicerSALT-style UI initialized successfully.")
        self.check_slicer_salt()

    # ====================================================
    # Action Handler Slots
    # ====================================================
    def open_dataset_folder(self):
        initial_dir = "D:/" if os.path.exists("D:/") else "C:/"
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Directory", initial_dir)
        if folder:
            if hasattr(self.left_panel, 'import_panel'):
                self.left_panel.import_panel.folder_input.setText(folder)
                self.left_panel.import_panel.load_subjects_from_directory(folder)
            if hasattr(self.left_panel, 'main_panel'):
                self.left_panel.main_panel.folder_input.setText(folder)
                self.left_panel.main_panel.update_stage_preview()
            self.left_panel.fastsurfer_panel.update_run_button_state()
            self.left_panel.icp_panel.update_run_button_state()
            self.left_panel.spharm_panel.update_run_button_state()
            self.log(f"Dataset directory set: {folder}")

    def choose_output_folder(self):
        initial_dir = getattr(self, '_last_output_dir', None)
        if not initial_dir or not os.path.isdir(initial_dir):
            if hasattr(self.left_panel, 'main_panel') and self.left_panel.main_panel.out_folder_input.text().strip():
                initial_dir = self.left_panel.main_panel.out_folder_input.text().strip()
            elif hasattr(self.left_panel, 'import_panel') and self.left_panel.import_panel.out_folder_input.text().strip():
                initial_dir = self.left_panel.import_panel.out_folder_input.text().strip()
            else:
                initial_dir = "D:/" if os.path.exists("D:/") else "C:/"

        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", initial_dir)
        if folder:
            self._last_output_dir = folder
            self.left_panel.set_global_output_directory(folder)
            self.log(f"Global output directory set: {folder}")

    def open_single_mesh(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open 3D Mesh File", "", "3D Mesh Files (*.vtk *.stl *.ply);;All Files (*)"
        )
        if filepath:
            self.right_panel.set_view_mode("full_3d")
            self.right_panel.display_mesh(filepath)
            self.log(f"Opened single 3D mesh: {os.path.basename(filepath)}")

    def capture_3d_screenshot(self):
        default_name = f"hippocampus_render_{int(time.time())}.png"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save 3D Viewport Screenshot", default_name, "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )
        if not filepath:
            return

        vtkWidget = getattr(self.right_panel.viewer, 'mesh_vtkWidget', None)
        if not vtkWidget:
            self.log("[ERROR] 3D mesh viewport is not available.")
            return

        saved = False
        try:
            import vtk
            rw = vtkWidget.GetRenderWindow()
            rw.Render()

            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(rw)
            w2if.SetInputBufferTypeToRGB()
            w2if.ReadFrontBufferOn()
            w2if.Update()

            ext = os.path.splitext(filepath)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                writer = vtk.vtkJPEGWriter()
                writer.SetQuality(95)
            else:
                writer = vtk.vtkPNGWriter()

            writer.SetFileName(filepath)
            writer.SetInputConnection(w2if.GetOutputPort())
            writer.Write()

            if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
                saved = True
        except Exception as e:
            self.log(f"[WARNING] VTK direct capture fallback: {e}")

        if not saved:
            pixmap = vtkWidget.grab()
            saved = pixmap.save(filepath)

        if saved:
            self.log(f"3D Screenshot saved successfully: {filepath}")
            self.console.set_status(f"Snapshot saved: {os.path.basename(filepath)}", "#2ecc71")
        else:
            self.log(f"[ERROR] Failed to save screenshot to {filepath}")

    def clear_3d_view(self):
        if hasattr(self.right_panel, 'viewer'):
            self.right_panel.viewer.display_mesh("")
        if hasattr(self.right_panel, 'clear_gradcam_view'):
            self.right_panel.clear_gradcam_view()
        self.log("Cleared 3D mesh view and overlays.")

    def reset_3d_camera(self):
        if hasattr(self.right_panel, 'viewer'):
            self.right_panel.viewer.reset_camera()

    def set_single_3d_view(self):
        if hasattr(self.right_panel, 'set_view_mode'):
            self.right_panel.set_view_mode("full_3d", self.module_combo.currentText())
            self.log("Switched to Single Full 3D Viewport.")

    def set_quad_view(self):
        if hasattr(self.right_panel, 'set_view_mode'):
            self.right_panel.set_view_mode("quad", self.module_combo.currentText())
            self.log("Switched to Quad 4-Viewports.")

    def toggle_view_mode(self):
        cur = getattr(self.right_panel.viewer, 'view_mode', 'quad')
        if cur == 'full_3d':
            self.set_quad_view()
        else:
            self.set_single_3d_view()

    def toggle_left_sidebar(self):
        is_vis = self.left_scroll.isVisible()
        self.left_scroll.setVisible(not is_vis)
        if hasattr(self, 'toggle_left_action'):
            self.toggle_left_action.setChecked(not is_vis)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
            if hasattr(self, 'fullscreen_action'):
                self.fullscreen_action.setChecked(False)
        else:
            self.showFullScreen()
            if hasattr(self, 'fullscreen_action'):
                self.fullscreen_action.setChecked(True)

    def run_full_pipeline(self):
        self.module_combo.setCurrentText("Main Panel")
        mp = getattr(self.left_panel, 'main_panel', None)
        if mp and hasattr(mp, 'run_all_button'):
            mp.run_all_button.click()
            self.log("Triggered automated Full Pipeline execution.")

    def show_preferences(self):
        slicer_paths = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
        current_slicer = slicer_paths[0] if slicer_paths else ""
        dlg = PreferencesDialog(self, current_slicer=current_slicer)
        if dlg.exec():
            theme_idx = dlg.theme_combo.currentIndex()
            viewer = getattr(self.right_panel, 'viewer', None)
            if viewer and hasattr(viewer, 'mesh_renderer'):
                ren = viewer.mesh_renderer
                if theme_idx == 0:  # Slicer Blue Gradient
                    ren.GradientBackgroundOn()
                    ren.SetBackground(0.741, 0.749, 0.902)
                    ren.SetBackground2(0.459, 0.475, 0.745)
                elif theme_idx == 1:  # Clinical Dark Slate
                    ren.GradientBackgroundOn()
                    ren.SetBackground(0.12, 0.15, 0.20)
                    ren.SetBackground2(0.06, 0.08, 0.12)
                elif theme_idx == 2:  # Pure Black
                    ren.GradientBackgroundOff()
                    ren.SetBackground(0.0, 0.0, 0.0)
                elif theme_idx == 3:  # Pure White
                    ren.GradientBackgroundOff()
                    ren.SetBackground(1.0, 1.0, 1.0)
                if hasattr(viewer, 'mesh_vtkWidget'):
                    viewer.mesh_vtkWidget.GetRenderWindow().Render()
                self.log(f"3D Viewport Theme changed to: {dlg.theme_combo.currentText()}")

    def show_diagnostics(self):
        dlg = DiagnosticsDialog(self)
        dlg.exec()

    def show_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def check_slicer_salt(self):
        slicer_paths = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
        if not slicer_paths:
            QMessageBox.critical(
                self, "SlicerSALT Required", 
                "SlicerSALT could not be found in the default installation directory (C:\\Program Files\\SlicerSALT*).\n\n"
                "This program requires SlicerSALT to function. The application will now close and open the download page."
            )
            webbrowser.open("https://salt.slicer.org/")
            sys.exit(1)
        else:
            self.log(f"SUCCESS: SlicerSALT detected at {slicer_paths[0]}")

    # ====================================================
    # Console & Logging Helpers
    # ====================================================
    def log(self, message):
        self.console.log(message)

    def copy_console_logs(self):
        self.console.copy_logs()

    def clear_console_logs(self):
        self.console.clear_logs()

    def toggle_console_expand(self):
        sizes = self.v_splitter.sizes()
        total_height = sum(sizes) if sum(sizes) > 0 else 800
        
        if sizes[0] > 50:
            self.saved_splitter_sizes = sizes
            self.v_splitter.setSizes([0, total_height])
            self.is_terminal_fullscreen = True
            self.console.expand_btn.setText("Restore")
            self.console.expand_btn.setToolTip("Restore terminal to original size")
            self.console.set_status("Terminal Fullscreen (Covering workspace)")
        else:
            if hasattr(self, 'saved_splitter_sizes') and self.saved_splitter_sizes[0] > 50:
                self.v_splitter.setSizes(self.saved_splitter_sizes)
            else:
                self.v_splitter.setSizes([total_height - 135, 135])
            self.is_terminal_fullscreen = False
            self.console.expand_btn.setText("Expand")
            self.console.expand_btn.setToolTip("Expand terminal to full screen")
            self.console.set_status("Ready")

    def on_splitter_moved(self, pos, index):
        sizes = self.v_splitter.sizes()
        if sizes[0] > 50 and getattr(self, 'is_terminal_fullscreen', False):
            self.is_terminal_fullscreen = False
            self.console.expand_btn.setText("Expand")
            self.console.expand_btn.setToolTip("Expand terminal to full screen")

    def on_h_splitter_moved(self, pos, index):
        if self.right_scroll.isVisible():
            sizes = self.h_splitter.sizes()
            if len(sizes) == 2 and sizes[0] > 50 and sizes[1] > 50:
                self.saved_h_splitter_sizes = sizes

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, 'v_splitter') and not getattr(self, '_initial_layout_done', False):
            self._initial_layout_done = True
            total_h = self.v_splitter.height()
            if total_h > 150:
                terminal_h = 135
                top_h = max(total_h - terminal_h, 540)
                self.v_splitter.setSizes([top_h, total_h - top_h])
                self.saved_splitter_sizes = [top_h, total_h - top_h]

    def set_terminal_visible(self, visible):
        self.console.setVisible(visible)
        if hasattr(self, 'toggle_terminal_action'):
            self.toggle_terminal_action.setChecked(visible)
        if visible:
            sizes = self.v_splitter.sizes()
            total_height = sum(sizes) if sum(sizes) > 0 else 800
            if len(sizes) >= 2 and sizes[1] < 50:
                self.v_splitter.setSizes([total_height - 135, 135])
            self.console.set_status("Ready")

    # ====================================================
    # Panel Signal & Module Routing
    # ====================================================
    def on_overlay_all_toggled(self, enabled: bool, filepaths: list, side_filter: str):
        self.right_panel.set_overlay_visible(enabled)
        if enabled:
            self.right_panel.display_all_meshes(filepaths, side_filter)
        else:
            self.right_panel.viewer.clear_multi_mesh_actors()
            self.right_panel.viewer.update_legend()
            self.right_panel.viewer.mesh_vtkWidget.GetRenderWindow().Render()

    def on_module_changed(self, module_name):
        index = self.module_combo.findText(module_name)
        self.left_panel.switch_module(index)
        self.right_panel.viewer.clear_all_patient_meshes()
        active_panel = self.left_panel.get_current_module_panel()

        modules_with_right_ui = {
            "Main Panel": True,
            "Data Importer": True,
            "FastSurfer Segmentation": True,
            "ICP Registration": True,
            "SPHARM Processing": True,
            "Result Panel": True,
        }

        has_right_ui = modules_with_right_ui.get(module_name, False)
        if not has_right_ui:
            if self.right_scroll.isVisible():
                sizes = self.h_splitter.sizes()
                if len(sizes) == 2 and sizes[0] > 50 and sizes[1] > 50:
                    self.saved_h_splitter_sizes = sizes
            self.right_scroll.setVisible(False)
        else:
            if not self.right_scroll.isVisible():
                self.right_scroll.setVisible(True)
                if hasattr(self, 'saved_h_splitter_sizes') and self.saved_h_splitter_sizes:
                    self.h_splitter.setSizes(self.saved_h_splitter_sizes)
                else:
                    self.h_splitter.setSizes([430, 850])

            if module_name in ("ICP Registration", "SPHARM Processing", "Result Panel", "Main Panel"):
                self.right_panel.set_view_mode("full_3d", module_name)
                self.right_panel.viewer.set_3d_plane_buttons_visible(False)
                if hasattr(self.right_panel.viewer, 'slice_mgr'):
                    self.right_panel.viewer.slice_mgr.hide_3d_planes(uncheck_buttons=True)
            elif module_name == "FastSurfer Segmentation":
                self.right_panel.set_view_mode("quad", module_name)
                self.right_panel.viewer.set_mesh_view_visible(True)
                self.right_panel.viewer.set_3d_plane_buttons_visible(True)
            elif module_name == "Data Importer":
                self.right_panel.viewer.set_mesh_view_visible(False)
                self.right_panel.set_view_mode("quad", module_name)
                self.right_panel.viewer.set_3d_plane_buttons_visible(False)
                if hasattr(self.right_panel.viewer, 'slice_mgr'):
                    self.right_panel.viewer.slice_mgr.hide_3d_planes(uncheck_buttons=True)
            else:
                self.right_panel.viewer.set_mesh_view_visible(False)
                self.right_panel.set_view_mode("quad", module_name)
                self.right_panel.viewer.set_3d_plane_buttons_visible(False)
                if hasattr(self.right_panel.viewer, 'slice_mgr'):
                    self.right_panel.viewer.slice_mgr.hide_3d_planes(uncheck_buttons=True)

        if module_name != "Result Panel":
            if hasattr(self.right_panel, 'clear_gradcam_view'):
                self.right_panel.clear_gradcam_view(render_now=False)

        if module_name == "Data Importer" and hasattr(active_panel, 'display_selected_subject'):
            active_panel.display_selected_subject()
        elif hasattr(active_panel, 'results_table') and active_panel.results_table.rowCount() > 0:
            if not active_panel.results_table.selectedItems():
                active_panel.results_table.selectRow(0)
        if hasattr(active_panel, 'overlay_cb'):
            is_overlay = active_panel.overlay_cb.isChecked()
            self.right_panel.set_overlay_visible(is_overlay)
            if is_overlay:
                active_panel.emit_overlay_meshes()
        if hasattr(active_panel, 'current_side_filter'):
            self.right_panel.set_side_filter(active_panel.current_side_filter)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if self.module_combo.currentText() == "Result Panel":
                if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_BracketLeft, Qt.Key.Key_Comma):
                    if not isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                        self.on_step_sd_requested(-0.1)
                        return True
                elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_BracketRight, Qt.Key.Key_Period):
                    if not isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                        self.on_step_sd_requested(0.1)
                        return True
                elif event.key() in (Qt.Key.Key_0, Qt.Key.Key_R, Qt.Key.Key_Home, Qt.Key.Key_Space):
                    if not isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit)):
                        self.on_reset_sd_requested()
                        return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        curr = self.module_combo.currentText()
        if curr == "Result Panel":
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_BracketLeft, Qt.Key.Key_Comma):
                self.on_step_sd_requested(-0.1)
                event.accept()
                return
            elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_BracketRight, Qt.Key.Key_Period):
                self.on_step_sd_requested(0.1)
                event.accept()
                return
            elif event.key() in (Qt.Key.Key_0, Qt.Key.Key_R, Qt.Key.Key_Home, Qt.Key.Key_Space):
                self.on_reset_sd_requested()
                event.accept()
                return
        super().keyPressEvent(event)

    def on_step_sd_requested(self, delta: float):
        res_panel = getattr(self.left_panel, 'result_panel', None)
        if res_panel and hasattr(res_panel, 'step_sd'):
            res_panel.step_sd(delta)

    def on_reset_sd_requested(self):
        res_panel = getattr(self.left_panel, 'result_panel', None)
        if res_panel and hasattr(res_panel, 'set_sd_value'):
            res_panel.set_sd_value(0.0)
