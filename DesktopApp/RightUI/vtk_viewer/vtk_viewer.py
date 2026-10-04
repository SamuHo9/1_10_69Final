import os
import vtk
from PyQt6.QtWidgets import (QWidget, QGridLayout, QHBoxLayout, QPushButton, 
                             QLabel, QCheckBox)
from PyQt6.QtCore import pyqtSignal, Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .slice_viewers import (
    CustomQVTKWidget,
    create_mask_actor,
    SliceViewersManager
)
from .template_manager import TemplateManager
from .gradcam_visualizer import GradCamVisualizer
from .lut_builder import (
    build_lut_gradcam,
    build_lut_signed_distance,
    build_lut_distance_mapping
)

class VtkViewer(QWidget):
    """
    Core VTK viewport orchestration widget.
    Coordinates 2D orthogonal MRI slice planes (Axial, Coronal, Sagittal)
    and interactive 3D mesh rendering, templates, and Grad-CAM attention.
    """
    signal_log_message = pyqtSignal(str)
    signal_template_toggled = pyqtSignal(bool)
    signal_overlay_toggled = pyqtSignal(bool)
    signal_step_sd = pyqtSignal(float)
    signal_reset_sd = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.maximized_frame = None
        self.mesh_view_enabled = False
        self.view_mode = "quad"  # 'quad' (4-view) or 'full_3d' (single large 3D viewport)
        self.current_module_name = ""
        self.current_mesh_path = None
        self.current_side_filter = "all"
        self.current_side = "left"
        self.mesh_actor = None
        self.current_diagnostic_info = None

        self.setup_ui()

    def setup_ui(self):
        # 2x2 Grid Layout
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2)

        # Initialize modular sub-managers
        self.slice_mgr = SliceViewersManager(self)
        self.tmpl_mgr = TemplateManager(self)
        self.gradcam_vis = GradCamVisualizer(self)

        # Build 2D slice viewers (Axial, Coronal, Sagittal)
        self.slice_mgr.setup_ui(self.grid_layout, self.toggle_maximize)

        # Build 3D Mesh Viewer Frame
        self.mesh_frame = self.slice_mgr.create_view_frame("#3498db")
        self.grid_layout.addWidget(self.mesh_frame, 0, 1)

        layout_mesh = self.mesh_frame.layout()
        mesh_top_bar = QHBoxLayout()
        mesh_top_bar.setContentsMargins(6, 4, 6, 4)
        mesh_top_bar.setSpacing(8)
        
        self.mesh_title_lbl = QLabel("3D Mesh View")
        self.mesh_title_lbl.setStyleSheet("color: white; font-weight: bold; border: none; background: transparent; font-size: 12px;")
        
        self.mesh_legend_lbl = QLabel("")
        self.mesh_legend_lbl.setStyleSheet("color: #ecf0f1; font-size: 11px; background: rgba(20,25,35,180); padding: 2px 8px; border-radius: 4px; border: 1px solid #3d4d65;")
        self.mesh_legend_lbl.setVisible(False)
        
        self.template_cb = QCheckBox("Show Reference Template")
        self.template_cb.setToolTip("Overlay standard reference template (mean shape) in 3D")
        self.template_cb.setStyleSheet("""
            QCheckBox {
                color: #f1c40f;
                font-weight: bold;
                font-size: 11px;
                spacing: 5px;
                background: rgba(241, 196, 15, 0.12);
                border: 1px solid #f39c12;
                border-radius: 4px;
                padding: 3px 8px;
            }
            QCheckBox:hover {
                background: rgba(241, 196, 15, 0.22);
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #f39c12;
                border-radius: 3px;
                background: #1e2230;
            }
            QCheckBox::indicator:checked {
                background: #f39c12;
            }
        """)
        self.template_cb.toggled.connect(self.on_template_cb_toggled)
        self.template_cb.setVisible(False)

        self.overlay_cb = QCheckBox("Overlay All Meshes")
        self.overlay_cb.setToolTip("Superimpose and view all aligned meshes together in 3D")
        self.overlay_cb.setStyleSheet("""
            QCheckBox {
                color: #1abc9c;
                font-weight: bold;
                font-size: 11px;
                spacing: 5px;
                background: rgba(26, 188, 156, 0.12);
                border: 1px solid #16a085;
                border-radius: 4px;
                padding: 3px 8px;
            }
            QCheckBox:hover {
                background: rgba(26, 188, 156, 0.22);
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #16a085;
                border-radius: 3px;
                background: #1e2230;
            }
            QCheckBox::indicator:checked {
                background: #1abc9c;
            }
        """)
        self.overlay_cb.toggled.connect(self.on_overlay_cb_toggled)
        self.overlay_cb.setVisible(False)
        
        self.mesh_max_btn = QPushButton("◻")
        self.mesh_max_btn.setFixedSize(24, 24)
        self.mesh_max_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        self.mesh_max_btn.clicked.connect(lambda: self.toggle_maximize(self.mesh_frame))
        
        mesh_top_bar.addWidget(self.mesh_title_lbl)
        mesh_top_bar.addWidget(self.mesh_legend_lbl)
        mesh_top_bar.addStretch()
        mesh_top_bar.addWidget(self.template_cb)
        mesh_top_bar.addWidget(self.overlay_cb)
        mesh_top_bar.addWidget(self.mesh_max_btn)
        
        layout_mesh.addLayout(mesh_top_bar)
        
        self.mesh_vtkWidget = QVTKRenderWindowInteractor(self.mesh_frame)
        self.mesh_vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.mesh_vtkWidget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout_mesh.addWidget(self.mesh_vtkWidget)
        
        self.mesh_renderer = vtk.vtkRenderer()
        self.mesh_renderer.GradientBackgroundOn()
        self.mesh_renderer.SetBackground(0.741, 0.749, 0.902)
        self.mesh_renderer.SetBackground2(0.459, 0.475, 0.745)
        self.mesh_vtkWidget.GetRenderWindow().AddRenderer(self.mesh_renderer)
        self.setup_3d_lighting()

        try:
            iren = self.mesh_vtkWidget.GetRenderWindow().GetInteractor()
            iren.AddObserver("KeyPressEvent", self._on_mesh_key_press)
            iren.AddObserver("LeftButtonPressEvent", self._on_mesh_mouse_press)
        except Exception:
            pass
        
        # Attach 3D slice planes and outline borders to the 3D renderer
        self.slice_mgr.attach_3d_planes_to_renderer(self.mesh_renderer)

        self.mesh_frame.hide()
        self.mesh_vtkWidget.Initialize()
        self.mesh_vtkWidget.GetRenderWindow().Render()

    # =========================================================================
    # Properties for Seamless Backwards Compatibility
    # =========================================================================
    @property
    def axial_frame(self): return self.slice_mgr.axial_frame
    @property
    def coronal_frame(self): return self.slice_mgr.coronal_frame
    @property
    def sagittal_frame(self): return self.slice_mgr.sagittal_frame
    @property
    def axial_viewer(self): return self.slice_mgr.axial_viewer
    @property
    def coronal_viewer(self): return self.slice_mgr.coronal_viewer
    @property
    def sagittal_viewer(self): return self.slice_mgr.sagittal_viewer
    @property
    def axial_slider(self): return self.slice_mgr.axial_slider
    @property
    def coronal_slider(self): return self.slice_mgr.coronal_slider
    @property
    def sagittal_slider(self): return self.slice_mgr.sagittal_slider
    @property
    def axial_slice_lbl(self): return self.slice_mgr.axial_slice_lbl
    @property
    def coronal_slice_lbl(self): return self.slice_mgr.coronal_slice_lbl
    @property
    def sagittal_slice_lbl(self): return self.slice_mgr.sagittal_slice_lbl
    @property
    def axial_3d_btn(self): return self.slice_mgr.axial_3d_btn
    @property
    def coronal_3d_btn(self): return self.slice_mgr.coronal_3d_btn
    @property
    def sagittal_3d_btn(self): return self.slice_mgr.sagittal_3d_btn
    @property
    def axial_3d_actor(self): return self.slice_mgr.axial_3d_actor
    @property
    def coronal_3d_actor(self): return self.slice_mgr.coronal_3d_actor
    @property
    def sagittal_3d_actor(self): return self.slice_mgr.sagittal_3d_actor
    @property
    def axial_outline_actor(self): return self.slice_mgr.axial_outline_actor
    @property
    def coronal_outline_actor(self): return self.slice_mgr.coronal_outline_actor
    @property
    def sagittal_outline_actor(self): return self.slice_mgr.sagittal_outline_actor
    @property
    def axial_voi(self): return self.slice_mgr.axial_voi
    @property
    def coronal_voi(self): return self.slice_mgr.coronal_voi
    @property
    def sagittal_voi(self): return self.slice_mgr.sagittal_voi
    @property
    def mask_actors(self): return self.slice_mgr.mask_actors
    @mask_actors.setter
    def mask_actors(self, val): self.slice_mgr.mask_actors = val

    @property
    def template_actor(self): return self.tmpl_mgr.template_actor
    @template_actor.setter
    def template_actor(self, val): self.tmpl_mgr.template_actor = val
    @property
    def template_actors(self): return self.tmpl_mgr.template_actors
    @template_actors.setter
    def template_actors(self, val): self.tmpl_mgr.template_actors = val
    @property
    def multi_mesh_actors(self): return self.tmpl_mgr.multi_mesh_actors
    @multi_mesh_actors.setter
    def multi_mesh_actors(self, val): self.tmpl_mgr.multi_mesh_actors = val
    @property
    def multi_mesh_paths(self): return self.tmpl_mgr.multi_mesh_paths
    @multi_mesh_paths.setter
    def multi_mesh_paths(self, val): self.tmpl_mgr.multi_mesh_paths = val
    @property
    def is_overlay_active(self): return self.tmpl_mgr.is_overlay_active
    @is_overlay_active.setter
    def is_overlay_active(self, val): self.tmpl_mgr.is_overlay_active = val

    @property
    def scalar_actor(self): return self.gradcam_vis.scalar_actor
    @scalar_actor.setter
    def scalar_actor(self, val): self.gradcam_vis.scalar_actor = val
    @property
    def scalar_mapper(self): return self.gradcam_vis.scalar_mapper
    @scalar_mapper.setter
    def scalar_mapper(self, val): self.gradcam_vis.scalar_mapper = val
    @property
    def scalar_bar_actor(self): return self.gradcam_vis.scalar_bar_actor
    @scalar_bar_actor.setter
    def scalar_bar_actor(self, val): self.gradcam_vis.scalar_bar_actor = val
    @property
    def patient_overlay_actor(self): return self.gradcam_vis.patient_overlay_actor
    @patient_overlay_actor.setter
    def patient_overlay_actor(self, val): self.gradcam_vis.patient_overlay_actor = val

    # =========================================================================
    # 2D Slice Viewers Delegation
    # =========================================================================
    def display_subject(self, filepath):
        self.slice_mgr.display_subject(filepath)

    def display_segmentation_overlays(self, base_img_path, lh_mask_path, rh_mask_path, side_filter="all"):
        self.slice_mgr.display_segmentation_overlays(base_img_path, lh_mask_path, rh_mask_path, side_filter=side_filter)

    def set_3d_plane_buttons_visible(self, visible: bool):
        self.slice_mgr.set_3d_plane_buttons_visible(visible)

    def reset_slice(self, orientation):
        self.slice_mgr.reset_slice(orientation)

    def toggle_3d_plane(self, orientation, visible):
        self.slice_mgr.toggle_3d_plane(orientation, visible)

    def toggle_flip(self, viewer):
        self.slice_mgr.toggle_flip(viewer)

    def change_slice(self, viewer, val, slider, slice_lbl):
        self.slice_mgr.change_slice(viewer, val, slider, slice_lbl)

    def create_view_frame(self, color):
        return self.slice_mgr.create_view_frame(color)

    # =========================================================================
    # Template & Multi-Mesh Delegation
    # =========================================================================
    def get_spharm_lh_transform(self):
        return self.tmpl_mgr.get_spharm_lh_transform()

    def get_template_paths(self, side=None):
        return self.tmpl_mgr.get_template_paths(side=side)

    def get_template_path(self, side=None):
        return self.tmpl_mgr.get_template_path(side=side)

    def update_template_overlay(self, side=None):
        self.tmpl_mgr.update_template_overlay(side=side)

    def clear_template_actors(self):
        self.tmpl_mgr.clear_template_actors()

    def set_template_visible(self, visible: bool):
        self.tmpl_mgr.set_template_visible(visible)

    def on_template_cb_toggled(self, checked: bool):
        self.signal_template_toggled.emit(checked)
        self.update_template_overlay()

    def on_overlay_cb_toggled(self, checked: bool):
        self.signal_overlay_toggled.emit(checked)

    def set_overlay_visible(self, visible: bool):
        self.overlay_cb.blockSignals(True)
        self.overlay_cb.setChecked(visible)
        self.overlay_cb.blockSignals(False)

    def clear_multi_mesh_actors(self):
        self.tmpl_mgr.clear_multi_mesh_actors()

    def display_all_meshes(self, filepaths, side_filter="all"):
        self.tmpl_mgr.display_all_meshes(filepaths, side_filter=side_filter)

    def update_legend(self, tmpl_name=None):
        self.tmpl_mgr.update_legend(tmpl_name=tmpl_name)

    def set_side_filter(self, side_filter: str):
        self.current_side_filter = side_filter
        if self.template_cb.isChecked():
            if side_filter in ("rh", "right"):
                chosen_side = "right"
            elif side_filter in ("lh", "left"):
                chosen_side = "left"
            else:
                chosen_side = "all"
            self.update_template_overlay(side=chosen_side)

    # =========================================================================
    # Grad-CAM Delegation & LUT Builders
    # =========================================================================
    def build_lut_gradcam(self):
        return build_lut_gradcam()

    def build_lut_signed_distance(self, max_val=0.16):
        return build_lut_signed_distance(max_val=max_val)

    def build_lut_distance_mapping(self, max_val=0.16):
        return build_lut_distance_mapping(max_val=max_val)

    def clear_gradcam_view(self, render_now=True):
        self.gradcam_vis.clear_gradcam_view(render_now=render_now)

    def display_gradcam_mesh(self, mesh_path, scalar_mode="DistanceMapping", lut_type="distance_mapping", title="Distance Mapping", side="left", opacity=1.0):
        self.gradcam_vis.display_gradcam_mesh(mesh_path, scalar_mode=scalar_mode, lut_type=lut_type, title=title, side=side, opacity=opacity)

    def set_patient_overlay(self, mesh_path, visible=True, opacity=0.35, side="left"):
        self.gradcam_vis.set_patient_overlay(mesh_path, visible=visible, opacity=opacity, side=side)

    # =========================================================================
    # 3D Viewport Controls & Single Mesh Rendering
    # =========================================================================
    def _on_mesh_key_press(self, obj, event):
        try:
            key = (self.mesh_vtkWidget.GetKeySym() or "").lower()
            is_result = ("result" in str(self.current_module_name).lower()) or (getattr(self, 'scalar_actor', None) is not None)
            if is_result:
                if key in ("right", "bracketright", "period"):
                    self.signal_step_sd.emit(0.1)
                elif key in ("left", "bracketleft", "comma"):
                    self.signal_step_sd.emit(-0.1)
                elif key in ("0", "r", "home", "space"):
                    self.signal_reset_sd.emit()
        except Exception:
            pass

    def _on_mesh_mouse_press(self, obj, event):
        try:
            self.mesh_vtkWidget.setFocus()
        except Exception:
            pass

    def setup_3d_lighting(self):
        """Balanced lighting matching view_gradcam_plsda_top3.py."""
        self.mesh_renderer.RemoveAllLights()

        key_light = vtk.vtkLight()
        key_light.SetLightTypeToCameraLight()
        key_light.SetPosition(0.3, 0.5, 1.0)
        key_light.SetIntensity(0.9)
        self.mesh_renderer.AddLight(key_light)

        fill_light = vtk.vtkLight()
        fill_light.SetLightTypeToCameraLight()
        fill_light.SetPosition(-0.4, -0.3, 0.7)
        fill_light.SetIntensity(0.4)
        self.mesh_renderer.AddLight(fill_light)

    def reset_3d_camera(self, side=None):
        """Sets canonical viewing direction without modifying underlying mesh data."""
        self.mesh_renderer.ResetCamera()
        camera = self.mesh_renderer.GetActiveCamera()
        fp = camera.GetFocalPoint()
        dist = camera.GetDistance()
        camera.SetFocalPoint(fp[0], fp[1], fp[2])

        is_result = ("result" in str(self.current_module_name).lower()) or (getattr(self, 'scalar_actor', None) is not None)
        is_spharm = ("spharm" in str(self.current_module_name).lower()) or (
            self.current_mesh_path and "spharm" in self.current_mesh_path.lower()
        ) or (
            self.multi_mesh_paths and any("spharm" in p.lower() for p in self.multi_mesh_paths[:3])
        )

        resolved_side = str(side or getattr(self, 'current_side', None) or getattr(self, 'current_side_filter', None) or "left").lower()

        if is_result:
            camera.SetParallelProjection(True)
            if "right" in resolved_side or "rh" in resolved_side:
                camera.SetPosition(fp[0], fp[1], fp[2] - dist)
                camera.SetViewUp(1.0, 0.0, 0.0)
            else:
                camera.SetPosition(fp[0], fp[1], fp[2] + dist)
                camera.SetViewUp(-1.0, 0.0, 0.0)
        elif is_spharm:
            camera.SetParallelProjection(False)
            camera.SetPosition(fp[0], fp[1] - dist, fp[2])
            camera.SetViewUp(-0.592, 0.0, 0.806)
        else:
            camera.SetParallelProjection(False)
            camera.SetPosition(fp[0], fp[1] - dist, fp[2])
            camera.SetViewUp(0.0, 0.0, 1.0)

        self.mesh_renderer.ResetCameraClippingRange()
        self.mesh_vtkWidget.GetRenderWindow().Render()

    def reset_camera(self):
        self.slice_mgr.reset_camera()
        self.reset_3d_camera()
        self.signal_log_message.emit("Camera reset to original position.")

    def set_diagnostic_info(self, text: str):
        self.current_diagnostic_info = text
        self.update_legend()
        if hasattr(self, 'mesh_vtkWidget'):
            self.mesh_vtkWidget.GetRenderWindow().Render()

    def clear_all_patient_meshes(self):
        """Clears all patient meshes (single and multi-overlaid) from 3D viewport."""
        if self.mesh_actor is not None:
            self.mesh_renderer.RemoveActor(self.mesh_actor)
            self.mesh_actor = None
        self.clear_multi_mesh_actors()
        self.clear_gradcam_view(render_now=False)
        self.current_mesh_path = None
        self.current_diagnostic_info = None
        self.is_overlay_active = False
        self.overlay_cb.blockSignals(True)
        self.overlay_cb.setChecked(False)
        self.overlay_cb.blockSignals(False)
        self.update_legend()
        self.mesh_vtkWidget.GetRenderWindow().Render()

    def set_mesh_view_visible(self, visible):
        self.mesh_view_enabled = visible
        if self.view_mode != "full_3d" and self.maximized_frame is None:
            self.mesh_frame.setVisible(visible)

    def toggle_maximize(self, frame):
        if self.view_mode == "full_3d":
            return

        frames = [self.axial_frame, self.coronal_frame, self.sagittal_frame, self.mesh_frame]
        
        if self.maximized_frame is None:
            self.maximized_frame = frame
            for f in frames:
                if f != frame:
                    f.hide()
        else:
            self.maximized_frame = None
            for f in frames:
                if f == self.mesh_frame:
                    f.setVisible(self.mesh_view_enabled)
                else:
                    f.show()

    def set_view_mode(self, mode: str, module_name: str = ""):
        """
        Switches between:
        - 'quad': 4-view layout (Axial, Coronal, Sagittal + 3D) for Data Importer & FastSurfer
        - 'full_3d': Single large 3D viewport for ICP Registration & SPHARM Processing
        """
        prev_module = self.current_module_name
        self.view_mode = mode
        self.current_module_name = module_name

        if prev_module and prev_module != module_name:
            if self.mesh_actor is not None:
                self.mesh_renderer.RemoveActor(self.mesh_actor)
                self.mesh_actor = None
            self.clear_multi_mesh_actors()
            self.clear_gradcam_view(render_now=False)
            self.current_mesh_path = None
            self.is_overlay_active = False
            self.overlay_cb.blockSignals(True)
            self.overlay_cb.setChecked(False)
            self.overlay_cb.blockSignals(False)
            self.clear_template_actors()
        
        if mode == "full_3d":
            self.maximized_frame = None
            self.slice_mgr.hide_all()
            
            self.grid_layout.removeWidget(self.mesh_frame)
            self.grid_layout.addWidget(self.mesh_frame, 0, 0, 2, 2)
            self.mesh_frame.show()
            
            self.set_3d_plane_buttons_visible(False)
            if hasattr(self, 'slice_mgr'):
                self.slice_mgr.hide_3d_planes(uncheck_buttons=True)
            self.mesh_max_btn.setVisible(False)
            
            is_result = ("result" in str(module_name).lower()) or ("main" in str(module_name).lower())
            self.template_cb.setVisible(not is_result)
            self.overlay_cb.setVisible(not is_result)
            if is_result:
                self.template_cb.blockSignals(True)
                self.template_cb.setChecked(False)
                self.template_cb.blockSignals(False)
                self.clear_template_actors()
                self.overlay_cb.blockSignals(True)
                self.overlay_cb.setChecked(False)
                self.overlay_cb.blockSignals(False)
                self.clear_multi_mesh_actors()
            
            if "main" in str(module_name).lower():
                disp = "3D View — SPHARM Shape & Diagnosis"
            elif module_name:
                disp = f"3D View — {module_name}"
            else:
                disp = "3D View"
            self.mesh_title_lbl.setText(disp)
            
            if not is_result and self.template_cb.isChecked():
                self.update_template_overlay()
                
            self.reset_3d_camera()
            self.mesh_vtkWidget.GetRenderWindow().Render()
            
        else: # quad mode
            self.maximized_frame = None
            is_fastsurfer = "fastsurfer" in str(self.current_module_name).lower()
            if is_fastsurfer:
                self.mesh_view_enabled = True
            elif module_name == "Data Importer":
                self.mesh_view_enabled = False
            
            self.grid_layout.removeWidget(self.mesh_frame)
            self.grid_layout.addWidget(self.mesh_frame, 0, 1, 1, 1)
            
            self.slice_mgr.show_all()
            self.mesh_frame.setVisible(self.mesh_view_enabled)
            
            self.mesh_title_lbl.setText("3D Mesh View")
            self.mesh_max_btn.setVisible(True)
            self.template_cb.setVisible(False)
            self.overlay_cb.setVisible(False)
            
            self.clear_template_actors()
            if self.multi_mesh_actors:
                self.clear_multi_mesh_actors()
                
            if is_fastsurfer:
                self.set_3d_plane_buttons_visible(True)
            else:
                self.set_3d_plane_buttons_visible(False)
                if hasattr(self, 'slice_mgr'):
                    self.slice_mgr.hide_3d_planes(uncheck_buttons=True)
                
            self.update_legend()
            self.mesh_vtkWidget.GetRenderWindow().Render()

    def display_mesh(self, filepath, side_filter="all"):
        if not filepath:
            if self.mesh_actor is not None:
                self.mesh_renderer.RemoveActor(self.mesh_actor)
                self.mesh_actor = None
            if self.multi_mesh_actors:
                self.clear_multi_mesh_actors()
            self.current_mesh_path = None
            if self.template_cb.isChecked():
                self.update_template_overlay()
            self.update_legend()
            self.mesh_vtkWidget.GetRenderWindow().Render()
            return

        if not (filepath.endswith(".nii.gz") or filepath.endswith(".mgz") or filepath.endswith(".vtk") or filepath.endswith(".ply")):
            self.signal_log_message.emit("[ERROR] Unsupported mesh format. Expected .vtk, .ply, .nii.gz, or .mgz")
            return
            
        self.current_mesh_path = filepath
        self.current_side_filter = side_filter

        is_fastsurfer = "fastsurfer" in str(getattr(self, 'current_module_name', '')).lower()
        if self.view_mode != "full_3d":
            self.set_mesh_view_visible(True)
            if is_fastsurfer:
                self.set_3d_plane_buttons_visible(True)
            else:
                self.set_3d_plane_buttons_visible(False)
                if hasattr(self, 'slice_mgr'):
                    self.slice_mgr.hide_3d_planes(uncheck_buttons=True)
        else:
            self.set_3d_plane_buttons_visible(False)
            if hasattr(self, 'slice_mgr'):
                self.slice_mgr.hide_3d_planes(uncheck_buttons=True)
        
        if self.mesh_actor is not None:
            self.mesh_renderer.RemoveActor(self.mesh_actor)
            self.mesh_actor = None
        self.clear_gradcam_view(render_now=False)

        if self.multi_mesh_actors:
            self.clear_multi_mesh_actors()
            self.overlay_cb.blockSignals(True)
            self.overlay_cb.setChecked(False)
            self.overlay_cb.blockSignals(False)
        
        if filepath.endswith(".vtk") or filepath.endswith(".ply"):
            if filepath.endswith(".ply"):
                reader = vtk.vtkPLYReader()
            else:
                reader = vtk.vtkPolyDataReader()
            reader.SetFileName(filepath)
            reader.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            mapper.ScalarVisibilityOff()
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            fp_low = filepath.replace("\\", "/").lower()
            bn = os.path.basename(filepath).lower()
            is_rh = ("rh" in bn or "right" in bn or "/right/" in fp_low or side_filter == "rh")
            is_lh = ("lh" in bn or "left" in bn or "/left/" in fp_low or side_filter == "lh")
            
            is_spharm = ("spharm" in str(self.current_module_name).lower()) or ("spharm" in fp_low)
            if is_spharm and is_lh:
                actor.SetUserTransform(self.get_spharm_lh_transform())

            if is_rh:
                actor.GetProperty().SetColor(0.95, 0.65, 0.55) # Soft Coral
            elif is_lh:
                actor.GetProperty().SetColor(0.55, 0.75, 0.95) # Soft Blue
            else:
                actor.GetProperty().SetColor(0.72, 0.82, 0.93) # Cyan
                
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetSpecular(0.25)
            actor.GetProperty().SetSpecularPower(15)
            actor.GetProperty().SetInterpolationToPhong()
            
            self.mesh_actor = actor
            self.mesh_renderer.AddActor(self.mesh_actor)
            
            chosen_side = "right" if is_rh else ("left" if is_lh else None)
            if self.template_cb.isChecked():
                self.update_template_overlay(side=chosen_side)
                
            self.reset_3d_camera(side=chosen_side)
            self.update_legend()
            self.mesh_vtkWidget.GetRenderWindow().Render()
            return

        # Load NIFTI mask (FastSurfer)
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(filepath)
        reader.Update()
        
        mc2 = vtk.vtkMarchingCubes()
        mc2.SetInputConnection(reader.GetOutputPort())
        mc2.SetValue(0, 0.5)
        
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(mc2.GetOutputPort())
        smoother.SetNumberOfIterations(15)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetPassBand(0.1)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetSpecularPower(15)
        actor.GetProperty().SetInterpolationToPhong()
        
        self.mesh_actor = actor
        self.mesh_renderer.AddActor(self.mesh_actor)
        self.reset_3d_camera()
        self.update_legend()
        self.mesh_vtkWidget.GetRenderWindow().Render()

VTKViewer = VtkViewer

