import os
import vtk
from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QSlider)
from PyQt6.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

def create_mask_actor(mask_path, r, g, b, alpha=0.5):
    """Creates a colored translucent vtkImageActor from a 2D/3D NIFTI binary mask."""
    if not os.path.exists(mask_path):
        return None
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(mask_path)
    reader.Update()
    
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(2)
    lut.SetTableRange(0, 1)
    lut.SetTableValue(0, 0, 0, 0, 0.0) # Transparent background
    lut.SetTableValue(1, r/255.0, g/255.0, b/255.0, alpha)
    
    map_colors = vtk.vtkImageMapToColors()
    map_colors.SetInputData(reader.GetOutput())
    map_colors.SetLookupTable(lut)
    map_colors.SetOutputFormatToRGBA()
    map_colors.Update()
    
    actor = vtk.vtkImageActor()
    actor.GetMapper().SetInputData(map_colors.GetOutput())
    return actor

class CustomQVTKWidget(QVTKRenderWindowInteractor):
    """Subclass that intercepts and drops Left Clicks to prevent VTK's default Window/Level adjustment."""
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            return
        super().mousePressEvent(ev)
        
    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            return
        super().mouseMoveEvent(ev)

class SliceViewersManager:
    """
    Manages the 2D orthogonal MRI slice viewers (Axial, Coronal, Sagittal),
    including slider controls, flip/reset actions, 2D segmentation overlays,
    and corresponding 3D orthogonal slice planes.
    """
    def __init__(self, parent_viewer):
        self.parent = parent_viewer
        self.current_dims = None
        self.mask_actors = []
        
        # 3D Orthogonal Slice Plane Actors & Outline Filters
        self._setup_3d_plane_actors()

    def _setup_3d_plane_actors(self):
        # Axial 3D
        self.axial_3d_actor = vtk.vtkImageActor()
        self.axial_3d_actor.SetVisibility(False)
        self.axial_voi = vtk.vtkExtractVOI()
        self.axial_outline = vtk.vtkOutlineFilter()
        self.axial_outline.SetInputConnection(self.axial_voi.GetOutputPort())
        axial_out_map = vtk.vtkPolyDataMapper()
        axial_out_map.SetInputConnection(self.axial_outline.GetOutputPort())
        self.axial_outline_actor = vtk.vtkActor()
        self.axial_outline_actor.SetMapper(axial_out_map)
        self.axial_outline_actor.GetProperty().SetColor(0.95, 0.25, 0.25)
        self.axial_outline_actor.GetProperty().SetLineWidth(2.5)
        self.axial_outline_actor.SetVisibility(False)

        # Coronal 3D
        self.coronal_3d_actor = vtk.vtkImageActor()
        self.coronal_3d_actor.SetVisibility(False)
        self.coronal_voi = vtk.vtkExtractVOI()
        self.coronal_outline = vtk.vtkOutlineFilter()
        self.coronal_outline.SetInputConnection(self.coronal_voi.GetOutputPort())
        coronal_out_map = vtk.vtkPolyDataMapper()
        coronal_out_map.SetInputConnection(self.coronal_outline.GetOutputPort())
        self.coronal_outline_actor = vtk.vtkActor()
        self.coronal_outline_actor.SetMapper(coronal_out_map)
        self.coronal_outline_actor.GetProperty().SetColor(0.25, 0.85, 0.35)
        self.coronal_outline_actor.GetProperty().SetLineWidth(2.5)
        self.coronal_outline_actor.SetVisibility(False)

        # Sagittal 3D
        self.sagittal_3d_actor = vtk.vtkImageActor()
        self.sagittal_3d_actor.SetVisibility(False)
        self.sagittal_voi = vtk.vtkExtractVOI()
        self.sagittal_outline = vtk.vtkOutlineFilter()
        self.sagittal_outline.SetInputConnection(self.sagittal_voi.GetOutputPort())
        sagittal_out_map = vtk.vtkPolyDataMapper()
        sagittal_out_map.SetInputConnection(self.sagittal_outline.GetOutputPort())
        self.sagittal_outline_actor = vtk.vtkActor()
        self.sagittal_outline_actor.SetMapper(sagittal_out_map)
        self.sagittal_outline_actor.GetProperty().SetColor(0.95, 0.8, 0.1)
        self.sagittal_outline_actor.GetProperty().SetLineWidth(2.5)
        self.sagittal_outline_actor.SetVisibility(False)

    def attach_3d_planes_to_renderer(self, renderer):
        renderer.AddActor(self.axial_3d_actor)
        renderer.AddActor(self.axial_outline_actor)
        renderer.AddActor(self.coronal_3d_actor)
        renderer.AddActor(self.coronal_outline_actor)
        renderer.AddActor(self.sagittal_3d_actor)
        renderer.AddActor(self.sagittal_outline_actor)

    def setup_ui(self, grid_layout, toggle_maximize_fn):
        """Creates the 3 view frames (Axial, Coronal, Sagittal) and adds them to grid_layout."""
        self.axial_frame = self.create_view_frame("#e74c3c")
        self.coronal_frame = self.create_view_frame("#2ecc71")
        self.sagittal_frame = self.create_view_frame("#f1c40f")

        grid_layout.addWidget(self.axial_frame, 0, 0)
        grid_layout.addWidget(self.coronal_frame, 1, 0)
        grid_layout.addWidget(self.sagittal_frame, 1, 1)

        # Axial setup
        self.axial_slider, self.axial_vtkWidget, self.axial_slice_lbl, self.axial_flip_btn, self.axial_3d_btn, self.axial_reset_btn = \
            self.setup_vtk_with_slider(self.axial_frame, "Axial View", toggle_maximize_fn)
        self.axial_viewer = vtk.vtkImageViewer2()
        self.axial_viewer.SetRenderWindow(self.axial_vtkWidget.GetRenderWindow())
        self.axial_viewer.SetupInteractor(self.axial_vtkWidget.GetRenderWindow().GetInteractor())
        self.axial_viewer.SetSliceOrientationToXY()
        self.axial_slider.valueChanged.connect(lambda val: self.change_slice(self.axial_viewer, val, self.axial_slider, self.axial_slice_lbl))
        self.axial_flip_btn.clicked.connect(lambda: self.toggle_flip(self.axial_viewer))
        self.axial_3d_btn.toggled.connect(lambda chk: self.toggle_3d_plane("axial", chk))
        self.axial_reset_btn.clicked.connect(lambda: self.reset_slice("axial"))

        # Coronal setup
        self.coronal_slider, self.coronal_vtkWidget, self.coronal_slice_lbl, self.coronal_flip_btn, self.coronal_3d_btn, self.coronal_reset_btn = \
            self.setup_vtk_with_slider(self.coronal_frame, "Coronal View", toggle_maximize_fn)
        self.coronal_viewer = vtk.vtkImageViewer2()
        self.coronal_viewer.SetRenderWindow(self.coronal_vtkWidget.GetRenderWindow())
        self.coronal_viewer.SetupInteractor(self.coronal_vtkWidget.GetRenderWindow().GetInteractor())
        self.coronal_viewer.SetSliceOrientationToXZ()
        self.coronal_slider.valueChanged.connect(lambda val: self.change_slice(self.coronal_viewer, val, self.coronal_slider, self.coronal_slice_lbl))
        self.coronal_flip_btn.clicked.connect(lambda: self.toggle_flip(self.coronal_viewer))
        self.coronal_3d_btn.toggled.connect(lambda chk: self.toggle_3d_plane("coronal", chk))
        self.coronal_reset_btn.clicked.connect(lambda: self.reset_slice("coronal"))

        # Sagittal setup
        self.sagittal_slider, self.sagittal_vtkWidget, self.sagittal_slice_lbl, self.sagittal_flip_btn, self.sagittal_3d_btn, self.sagittal_reset_btn = \
            self.setup_vtk_with_slider(self.sagittal_frame, "Sagittal View", toggle_maximize_fn)
        self.sagittal_viewer = vtk.vtkImageViewer2()
        self.sagittal_viewer.SetRenderWindow(self.sagittal_vtkWidget.GetRenderWindow())
        self.sagittal_viewer.SetupInteractor(self.sagittal_vtkWidget.GetRenderWindow().GetInteractor())
        self.sagittal_viewer.SetSliceOrientationToYZ()
        self.sagittal_slider.valueChanged.connect(lambda val: self.change_slice(self.sagittal_viewer, val, self.sagittal_slider, self.sagittal_slice_lbl))
        self.sagittal_flip_btn.clicked.connect(lambda: self.toggle_flip(self.sagittal_viewer))
        self.sagittal_3d_btn.toggled.connect(lambda chk: self.toggle_3d_plane("sagittal", chk))
        self.sagittal_reset_btn.clicked.connect(lambda: self.reset_slice("sagittal"))

        # Configure solid black background for 2D slice viewers
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            ren = viewer.GetRenderer()
            ren.GradientBackgroundOff()
            ren.SetBackground(0.0, 0.0, 0.0)
            viewer.GetImageActor().SetVisibility(False)

        # Initialize widgets
        self.axial_vtkWidget.Initialize()
        self.axial_vtkWidget.GetRenderWindow().Render()
        self.coronal_vtkWidget.Initialize()
        self.coronal_vtkWidget.GetRenderWindow().Render()
        self.sagittal_vtkWidget.Initialize()
        self.sagittal_vtkWidget.GetRenderWindow().Render()

    def create_view_frame(self, color):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ border: 2px solid {color}; background-color: #1e2230; }}")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        return frame

    def setup_vtk_with_slider(self, frame, title_str, toggle_maximize_fn):
        layout = frame.layout()
        
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(5, 5, 5, 5)
        top_bar.setSpacing(4)
        
        title_lbl = QLabel(title_str)
        title_lbl.setStyleSheet("color: white; font-weight: bold; border: none; background: transparent;")
        
        plane_3d_btn = QPushButton("🔲 3D")
        plane_3d_btn.setCheckable(True)
        plane_3d_btn.setChecked(False)
        plane_3d_btn.setVisible(False)
        plane_3d_btn.setToolTip("Show / Hide slice plane in 3D Mesh View")
        plane_3d_btn.setFixedSize(45, 24)
        plane_3d_btn.setStyleSheet("""
            QPushButton {
                background: #2c3e50;
                color: #bdc3c7;
                border: 1px solid #7f8c8d;
                border-radius: 3px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background: #34495e;
                color: white;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: 1px solid #1f618d;
            }
        """)
        
        flip_btn = QPushButton("⇅")
        flip_btn.setFixedSize(24, 24)
        flip_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        
        max_btn = QPushButton("◻")
        max_btn.setFixedSize(24, 24)
        max_btn.setStyleSheet("QPushButton { background: transparent; color: white; border: 1px solid #777; border-radius: 3px; font-weight: bold;} QPushButton:hover { background: #555; }")
        max_btn.clicked.connect(lambda: toggle_maximize_fn(frame))
        
        top_bar.addWidget(title_lbl)
        top_bar.addStretch()
        top_bar.addWidget(plane_3d_btn)
        top_bar.addWidget(flip_btn)
        top_bar.addWidget(max_btn)
        
        layout.addLayout(top_bar)
        
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(5, 0, 5, 5)
        slider_layout.setSpacing(6)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setEnabled(False)
        slider.setStyleSheet("""
            QSlider { background: transparent; border: none; }
            QSlider::groove:horizontal { border: 1px solid #999999; height: 8px; background: #333333; margin: 2px 0; }
            QSlider::handle:horizontal { background: #7c7c7c; border: 1px solid #999999; width: 18px; margin: -4px 0; border-radius: 3px; }
            QSlider::handle:horizontal:hover { background: #a0a0a0; }
        """)
        
        slice_lbl = QLabel("Slice: - / -")
        slice_lbl.setStyleSheet("color: white; font-size: 11px; border: none; background: transparent;")
        slice_lbl.setFixedWidth(75)
        slice_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        reset_slice_btn = QPushButton("↺ Reset")
        reset_slice_btn.setToolTip("Reset to initial slice")
        reset_slice_btn.setEnabled(False)
        reset_slice_btn.setFixedSize(58, 22)
        reset_slice_btn.setStyleSheet("""
            QPushButton {
                background: #34495e;
                color: #ecf0f1;
                border: 1px solid #7f8c8d;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #415b76;
                color: white;
            }
            QPushButton:pressed {
                background: #2c3e50;
            }
            QPushButton:disabled {
                background: #242735;
                color: #666;
                border: 1px solid #3d414d;
            }
        """)
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(slice_lbl)
        slider_layout.addWidget(reset_slice_btn)
        layout.addLayout(slider_layout)
        
        vtkWidget = CustomQVTKWidget(frame)
        layout.addWidget(vtkWidget)
        
        return slider, vtkWidget, slice_lbl, flip_btn, plane_3d_btn, reset_slice_btn

    def set_3d_plane_buttons_visible(self, visible: bool):
        current_mod = getattr(self.parent, 'current_module_name', '')
        if visible and "fastsurfer" not in str(current_mod).lower():
            visible = False

        self.axial_3d_btn.setVisible(visible)
        self.coronal_3d_btn.setVisible(visible)
        self.sagittal_3d_btn.setVisible(visible)
        if not visible:
            self.hide_3d_planes(uncheck_buttons=True)

    def hide_3d_planes(self, uncheck_buttons: bool = True):
        """Hides all 3D slice planes and outlines from the 3D viewport."""
        for act in [self.axial_3d_actor, self.coronal_3d_actor, self.sagittal_3d_actor,
                    self.axial_outline_actor, self.coronal_outline_actor, self.sagittal_outline_actor]:
            if act:
                act.SetVisibility(False)
        if uncheck_buttons:
            for btn in [self.axial_3d_btn, self.coronal_3d_btn, self.sagittal_3d_btn]:
                if btn:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
        if hasattr(self.parent, 'mesh_vtkWidget') and self.parent.mesh_vtkWidget:
            self.parent.mesh_vtkWidget.GetRenderWindow().Render()

    def reset_slice(self, orientation):
        if orientation == "axial" and hasattr(self, 'axial_initial_slice'):
            self.axial_slider.setValue(self.axial_initial_slice)
            self.parent.signal_log_message.emit(f"Axial view reset to initial slice {self.axial_initial_slice}.")
        elif orientation == "coronal" and hasattr(self, 'coronal_initial_slice'):
            self.coronal_slider.setValue(self.coronal_initial_slice)
            self.parent.signal_log_message.emit(f"Coronal view reset to initial slice {self.coronal_initial_slice}.")
        elif orientation == "sagittal" and hasattr(self, 'sagittal_initial_slice'):
            self.sagittal_slider.setValue(self.sagittal_initial_slice)
            self.parent.signal_log_message.emit(f"Sagittal view reset to initial slice {self.sagittal_initial_slice}.")

    def toggle_3d_plane(self, orientation, visible):
        current_mod = getattr(self.parent, 'current_module_name', '')
        if "fastsurfer" not in str(current_mod).lower():
            self.hide_3d_planes(uncheck_buttons=True)
            return

        if visible:
            self.parent.set_mesh_view_visible(True)
        
        if orientation == "axial":
            self.axial_3d_actor.SetVisibility(visible)
            self.axial_outline_actor.SetVisibility(visible)
            if visible and self.current_dims is not None:
                val = self.axial_slider.value()
                dims = self.current_dims
                self.axial_3d_actor.SetDisplayExtent(0, dims[0]-1, 0, dims[1]-1, val, val)
                self.axial_voi.SetVOI(0, dims[0]-1, 0, dims[1]-1, val, val)
        elif orientation == "coronal":
            self.coronal_3d_actor.SetVisibility(visible)
            self.coronal_outline_actor.SetVisibility(visible)
            if visible and self.current_dims is not None:
                val = self.coronal_slider.value()
                dims = self.current_dims
                self.coronal_3d_actor.SetDisplayExtent(0, dims[0]-1, val, val, 0, dims[2]-1)
                self.coronal_voi.SetVOI(0, dims[0]-1, val, val, 0, dims[2]-1)
        elif orientation == "sagittal":
            self.sagittal_3d_actor.SetVisibility(visible)
            self.sagittal_outline_actor.SetVisibility(visible)
            if visible and self.current_dims is not None:
                val = self.sagittal_slider.value()
                dims = self.current_dims
                self.sagittal_3d_actor.SetDisplayExtent(val, val, 0, dims[1]-1, 0, dims[2]-1)
                self.sagittal_voi.SetVOI(val, val, 0, dims[1]-1, 0, dims[2]-1)
                
        self.parent.mesh_renderer.ResetCameraClippingRange()
        self.parent.mesh_vtkWidget.GetRenderWindow().Render()
        self.parent.signal_log_message.emit(f"3D Slice Plane ({orientation.capitalize()}) {'shown' if visible else 'hidden'} in 3D View.")

    def toggle_flip(self, viewer):
        camera = viewer.GetRenderer().GetActiveCamera()
        vu = camera.GetViewUp()
        camera.SetViewUp(-vu[0], -vu[1], -vu[2])
        viewer.Render()

    def change_slice(self, viewer, val, slider, slice_lbl):
        viewer.SetSlice(val)
        min_val = viewer.GetSliceMin()
        max_val = viewer.GetSliceMax()
        slice_lbl.setText(f"Slice: {val} / {max_val}")
        
        # Sync mask actors' display extent
        extent = viewer.GetImageActor().GetDisplayExtent()
        if hasattr(self, 'mask_actors'):
            for orientation_actors in self.mask_actors:
                if orientation_actors['viewer'] == viewer:
                    for actor in orientation_actors['actors']:
                        if actor:
                            actor.SetDisplayExtent(extent)

        # Sync 3D slice plane
        if self.current_dims is not None:
            dims = self.current_dims
            if viewer == self.axial_viewer:
                self.axial_3d_actor.SetDisplayExtent(0, dims[0]-1, 0, dims[1]-1, val, val)
                self.axial_voi.SetVOI(0, dims[0]-1, 0, dims[1]-1, val, val)
                if self.axial_3d_actor.GetVisibility():
                    self.parent.mesh_vtkWidget.GetRenderWindow().Render()
            elif viewer == self.coronal_viewer:
                self.coronal_3d_actor.SetDisplayExtent(0, dims[0]-1, val, val, 0, dims[2]-1)
                self.coronal_voi.SetVOI(0, dims[0]-1, val, val, 0, dims[2]-1)
                if self.coronal_3d_actor.GetVisibility():
                    self.parent.mesh_vtkWidget.GetRenderWindow().Render()
            elif viewer == self.sagittal_viewer:
                self.sagittal_3d_actor.SetDisplayExtent(val, val, 0, dims[1]-1, 0, dims[2]-1)
                self.sagittal_voi.SetVOI(val, val, 0, dims[1]-1, 0, dims[2]-1)
                if self.sagittal_3d_actor.GetVisibility():
                    self.parent.mesh_vtkWidget.GetRenderWindow().Render()

    def clear_views(self):
        # Remove old mask actors if they exist
        if hasattr(self, 'mask_actors'):
            for orientation in self.mask_actors:
                for actor in orientation.get('actors', []):
                    if actor:
                        orientation['viewer'].GetRenderer().RemoveActor(actor)
            self.mask_actors = []
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            if viewer and viewer.GetImageActor():
                viewer.GetImageActor().SetVisibility(False)
            if viewer and viewer.GetRenderer():
                viewer.Render()
        if hasattr(self, 'axial_slice_lbl'):
            self.axial_slice_lbl.setText("Slice: - / -")
        if hasattr(self, 'coronal_slice_lbl'):
            self.coronal_slice_lbl.setText("Slice: - / -")
        if hasattr(self, 'sagittal_slice_lbl'):
            self.sagittal_slice_lbl.setText("Slice: - / -")
        if hasattr(self, 'axial_slider'):
            self.axial_slider.setEnabled(False)
        if hasattr(self, 'coronal_slider'):
            self.coronal_slider.setEnabled(False)
        if hasattr(self, 'sagittal_slider'):
            self.sagittal_slider.setEnabled(False)
        if hasattr(self, 'axial_reset_btn'):
            self.axial_reset_btn.setEnabled(False)
        if hasattr(self, 'coronal_reset_btn'):
            self.coronal_reset_btn.setEnabled(False)
        if hasattr(self, 'sagittal_reset_btn'):
            self.sagittal_reset_btn.setEnabled(False)
        for act in [self.axial_3d_actor, self.coronal_3d_actor, self.sagittal_3d_actor,
                    self.axial_outline_actor, self.coronal_outline_actor, self.sagittal_outline_actor]:
            if act:
                act.SetVisibility(False)
        for btn in [self.axial_3d_btn, self.coronal_3d_btn, self.sagittal_3d_btn]:
            if btn:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)

    def display_subject(self, filepath):
        if not filepath or not os.path.isfile(filepath):
            self.clear_views()
            return

        # Ensure any leftover mask overlay actors are removed
        if hasattr(self, 'mask_actors') and self.mask_actors:
            for orientation in self.mask_actors:
                for actor in orientation.get('actors', []):
                    if actor:
                        orientation['viewer'].GetRenderer().RemoveActor(actor)
            self.mask_actors = []

        filename = os.path.basename(filepath)
        ext = filepath.lower()
        
        try:
            if ext.endswith('.nii') or ext.endswith('.nii.gz'):
                reader = vtk.vtkNIFTIImageReader()
                reader.SetFileName(filepath)
            elif ext.endswith('.mgz'):
                import nibabel as nib
                import numpy as np
                temp_nii = filepath.replace('.mgz', '_conformed.nii.gz')
                if not os.path.exists(temp_nii):
                    img = nib.load(filepath)
                    nii_img = nib.Nifti1Image(np.asarray(img.dataobj), img.affine, img.header)
                    nib.save(nii_img, temp_nii)
                reader = vtk.vtkNIFTIImageReader()
                reader.SetFileName(temp_nii)
            elif ext.endswith('.nrrd'):
                reader = vtk.vtkNrrdReader()
                reader.SetFileName(filepath)
            elif ext.endswith('.mha'):
                reader = vtk.vtkMetaImageReader()
                reader.SetFileName(filepath)
            else:
                self.parent.signal_log_message.emit(f"Unsupported image format for slice viewer: {filepath}.")
                return
            reader.Update()
            
            image_data = reader.GetOutput()
            if not image_data or image_data.GetDimensions()[0] == 0:
                self.parent.signal_log_message.emit(f"Failed to load image data from {filename}")
                return
                
            self.axial_viewer.SetInputData(image_data)
            self.coronal_viewer.SetInputData(image_data)
            self.sagittal_viewer.SetInputData(image_data)
            
            dims = image_data.GetDimensions()
            self.current_dims = dims
            self.parent.current_dims = dims
            
            # Setup Sliders and Reset Buttons
            self.axial_slider.setEnabled(True)
            self.axial_slider.setRange(0, dims[2] - 1)
            self.axial_initial_slice = dims[2] // 2
            self.axial_slider.setValue(self.axial_initial_slice)
            self.axial_slice_lbl.setText(f"Slice: {self.axial_initial_slice} / {dims[2] - 1}")
            self.axial_reset_btn.setEnabled(True)
            
            self.coronal_slider.setEnabled(True)
            self.coronal_slider.setRange(0, dims[1] - 1)
            self.coronal_initial_slice = dims[1] // 2
            self.coronal_slider.setValue(self.coronal_initial_slice)
            self.coronal_slice_lbl.setText(f"Slice: {self.coronal_initial_slice} / {dims[1] - 1}")
            self.coronal_reset_btn.setEnabled(True)
            
            self.sagittal_slider.setEnabled(True)
            self.sagittal_slider.setRange(0, dims[0] - 1)
            self.sagittal_initial_slice = dims[0] // 2
            self.sagittal_slider.setValue(self.sagittal_initial_slice)
            self.sagittal_slice_lbl.setText(f"Slice: {self.sagittal_initial_slice} / {dims[0] - 1}")
            self.sagittal_reset_btn.setEnabled(True)
            
            # Auto window/level based on scalar range
            scalar_range = image_data.GetScalarRange()
            window = scalar_range[1] - scalar_range[0]
            level = (scalar_range[0] + scalar_range[1]) / 2.0

            # Connect 3D slice plane actors & VOIs
            self.axial_3d_actor.GetMapper().SetInputData(image_data)
            self.axial_3d_actor.SetDisplayExtent(0, dims[0] - 1, 0, dims[1] - 1, dims[2] // 2, dims[2] // 2)
            self.axial_voi.SetInputData(image_data)
            self.axial_voi.SetVOI(0, dims[0] - 1, 0, dims[1] - 1, dims[2] // 2, dims[2] // 2)

            self.coronal_3d_actor.GetMapper().SetInputData(image_data)
            self.coronal_3d_actor.SetDisplayExtent(0, dims[0] - 1, dims[1] // 2, dims[1] // 2, 0, dims[2] - 1)
            self.coronal_voi.SetInputData(image_data)
            self.coronal_voi.SetVOI(0, dims[0] - 1, dims[1] // 2, dims[1] // 2, 0, dims[2] - 1)

            self.sagittal_3d_actor.GetMapper().SetInputData(image_data)
            self.sagittal_3d_actor.SetDisplayExtent(dims[0] // 2, dims[0] // 2, 0, dims[1] - 1, 0, dims[2] - 1)
            self.sagittal_voi.SetInputData(image_data)
            self.sagittal_voi.SetVOI(dims[0] // 2, dims[0] // 2, 0, dims[1] - 1, 0, dims[2] - 1)

            for act_3d in [self.axial_3d_actor, self.coronal_3d_actor, self.sagittal_3d_actor]:
                act_3d.GetProperty().SetColorWindow(window)
                act_3d.GetProperty().SetColorLevel(level)
                act_3d.InterpolateOn()
            
            # Explicitly set the flipped ViewUp by default (as it usually loads upside down)
            self.axial_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, -1, 0)
            self.coronal_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, 0, -1)
            self.sagittal_viewer.GetRenderer().GetActiveCamera().SetViewUp(0, 0, -1)
            
            for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
                viewer.GetImageActor().SetVisibility(True)
                viewer.SetColorWindow(window)
                viewer.SetColorLevel(level)
                viewer.GetImageActor().InterpolateOn()
                
                renderer = viewer.GetRenderer()
                renderer.ResetCamera()
                viewer.Render()

            if any([self.axial_3d_actor.GetVisibility(), self.coronal_3d_actor.GetVisibility(), self.sagittal_3d_actor.GetVisibility()]):
                self.parent.mesh_vtkWidget.GetRenderWindow().Render()
                
            self.parent.signal_log_message.emit(f"Loaded MRI volume: {filename}")
            
        except Exception as e:
            self.parent.signal_log_message.emit(f"Error loading volume: {str(e)}")

    def display_segmentation_overlays(self, base_img_path, lh_mask_path, rh_mask_path, side_filter="all"):
        filter_text = "ALL (Left & Right)" if side_filter == "all" else ("LEFT (LH)" if side_filter == "lh" else "RIGHT (RH)")
        self.parent.signal_log_message.emit(f"Loading 2D MRI with {filter_text} Hippocampus overlays...")
        
        self.display_subject(base_img_path)
        
        # Remove old mask actors if they exist
        if hasattr(self, 'mask_actors'):
            for orientation in self.mask_actors:
                for actor in orientation['actors']:
                    if actor:
                        orientation['viewer'].GetRenderer().RemoveActor(actor)
        
        self.mask_actors = []
        viewers = [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]
        
        for viewer in viewers:
            actors = []
            
            # Left Hippocampus: #A7BCE9 (167, 188, 233)
            if side_filter in ("all", "lh") and lh_mask_path and os.path.exists(lh_mask_path):
                lh_actor = create_mask_actor(lh_mask_path, 167, 188, 233, alpha=0.55)
                if lh_actor:
                    viewer.GetRenderer().AddActor(lh_actor)
                    actors.append(lh_actor)
                    
            # Right Hippocampus: #FFA87B (255, 168, 123)
            if side_filter in ("all", "rh") and rh_mask_path and os.path.exists(rh_mask_path):
                rh_actor = create_mask_actor(rh_mask_path, 255, 168, 123, alpha=0.55)
                if rh_actor:
                    viewer.GetRenderer().AddActor(rh_actor)
                    actors.append(rh_actor)
                
            self.mask_actors.append({
                'viewer': viewer,
                'actors': actors
            })
            
            extent = viewer.GetImageActor().GetDisplayExtent()
            for actor in actors:
                actor.SetDisplayExtent(extent)
            
            viewer.Render()

    def reset_camera(self):
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            if viewer.GetInput():
                renderer = viewer.GetRenderer()
                renderer.ResetCamera()
                viewer.Render()

    def hide_all(self):
        self.axial_frame.hide()
        self.coronal_frame.hide()
        self.sagittal_frame.hide()

    def show_all(self):
        self.axial_frame.show()
        self.coronal_frame.show()
        self.sagittal_frame.show()
