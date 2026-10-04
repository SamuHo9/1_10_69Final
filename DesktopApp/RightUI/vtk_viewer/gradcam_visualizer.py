import os
import vtk
from .lut_builder import build_lut_gradcam, build_lut_signed_distance, build_lut_distance_mapping

class GradCamVisualizer:
    """
    Manages 3D Grad-CAM attention and deformation scalar heatmaps,
    colorbars, and patient ghost mesh overlays in the 3D viewport.
    """
    def __init__(self, parent_viewer):
        self.parent = parent_viewer
        self.scalar_actor = None
        self.scalar_mapper = None
        self.scalar_bar_actor = None
        self.patient_overlay_actor = None
        self._last_gradcam_side = None

    def clear_gradcam_view(self, render_now=True):
        if self.scalar_actor:
            self.parent.mesh_renderer.RemoveActor(self.scalar_actor)
            self.scalar_actor = None
        self.scalar_mapper = None
        if self.scalar_bar_actor:
            self.parent.mesh_renderer.RemoveActor(self.scalar_bar_actor)
            self.scalar_bar_actor = None
        if self.patient_overlay_actor:
            self.parent.mesh_renderer.RemoveActor(self.patient_overlay_actor)
            self.patient_overlay_actor = None
        self._last_gradcam_side = None
        if render_now:
            self.parent.mesh_vtkWidget.GetRenderWindow().Render()

    def display_gradcam_mesh(self, mesh_path, scalar_mode="DistanceMapping", lut_type="distance_mapping", title="Distance Mapping", side="left", opacity=1.0):
        if not mesh_path or not os.path.isfile(mesh_path):
            self.parent.signal_log_message.emit(f"[ERROR] Mesh file not found: {mesh_path}")
            return

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(mesh_path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            self.parent.signal_log_message.emit(f"[ERROR] Invalid polydata in: {mesh_path}")
            return

        pdata = poly.GetPointData()
        arr = pdata.GetArray(scalar_mode)
        if arr is not None:
            pdata.SetActiveScalars(scalar_mode)
            s_range = arr.GetRange()
        else:
            if pdata.GetNumberOfArrays() > 0:
                first_name = pdata.GetArrayName(0)
                pdata.SetActiveScalars(first_name)
                s_range = pdata.GetScalars().GetRange()
                scalar_mode = first_name
            else:
                s_range = (0.0, 1.0)

        # Build appropriate LUT based on mode
        if lut_type == "signed_distance":
            max_abs = max(0.16, abs(s_range[0]), abs(s_range[1]))
            lut = build_lut_signed_distance(max_val=max_abs)
            s_min, s_max = -max_abs, max_abs
            bar_title = "mm"
        elif lut_type == "distance_mapping" or "Distance" in scalar_mode:
            max_val = max(0.16, s_range[1])
            lut = build_lut_distance_mapping(max_val=max_val)
            s_min, s_max = 0.0, max_val
            bar_title = "mm"
        else:
            lut = build_lut_gradcam()
            s_min, s_max = 0.0, 1.0
            bar_title = "Score"

        # Update title text on top bar
        self.parent.mesh_title_lbl.setText(f"3D View — {title}")

        # Smooth in-place update if same side and actors exist
        is_same_side = (self._last_gradcam_side == side)
        can_update_in_place = (
            self.scalar_actor is not None and
            self.scalar_mapper is not None and
            self.scalar_bar_actor is not None and
            is_same_side
        )

        if can_update_in_place:
            self.scalar_mapper.SetInputData(poly)
            self.scalar_mapper.SetScalarRange(s_min, s_max)
            self.scalar_mapper.SetLookupTable(lut)
            self.scalar_mapper.SelectColorArray(scalar_mode)
            self.scalar_bar_actor.SetLookupTable(lut)
            self.scalar_bar_actor.SetTitle(bar_title)
            self.parent.mesh_vtkWidget.GetRenderWindow().Render()
            return

        # Full setup
        self.clear_gradcam_view(render_now=False)
        self.parent.clear_all_patient_meshes()
        self.parent.clear_template_actors()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetScalarRange(s_min, s_max)
        mapper.SetLookupTable(lut)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray(scalar_mode)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        is_spharm = "spharm" in str(self.parent.current_module_name).lower()
        if is_spharm and side.lower() == "left":
            actor.SetUserTransform(self.parent.get_spharm_lh_transform())

        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetAmbient(0.20)
        actor.GetProperty().SetDiffuse(0.80)
        actor.GetProperty().SetSpecular(0.25)
        actor.GetProperty().SetSpecularPower(30)
        actor.GetProperty().SetInterpolationToGouraud()

        self.parent.mesh_renderer.AddActor(actor)
        self.scalar_actor = actor
        self.scalar_mapper = mapper
        self._last_gradcam_side = side

        # Scalar Bar
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle(bar_title)
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.86, 0.12)
        scalar_bar.SetWidth(0.11)
        scalar_bar.SetHeight(0.55)
        tprop = scalar_bar.GetTitleTextProperty()
        tprop.SetColor(1.0, 1.0, 1.0)
        tprop.SetFontSize(11)
        tprop.BoldOn()
        lprop = scalar_bar.GetLabelTextProperty()
        lprop.SetColor(0.9, 0.9, 0.9)
        lprop.SetFontSize(10)
        
        self.parent.mesh_renderer.AddActor(scalar_bar)
        self.scalar_bar_actor = scalar_bar

        self.parent.reset_3d_camera(side=side)
        self.parent.mesh_vtkWidget.GetRenderWindow().Render()
        self.parent.signal_log_message.emit(f"SUCCESS: Rendered {title} on 3D mesh ({side.upper()})")

    def set_patient_overlay(self, mesh_path, visible=True, opacity=0.35, side="left"):
        if self.patient_overlay_actor:
            self.parent.mesh_renderer.RemoveActor(self.patient_overlay_actor)
            self.patient_overlay_actor = None

        if not visible or not mesh_path or not os.path.isfile(mesh_path):
            self.parent.mesh_vtkWidget.GetRenderWindow().Render()
            return

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(mesh_path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            return

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        is_spharm = "spharm" in str(self.parent.current_module_name).lower()
        if is_spharm and side.lower() == "left":
            actor.SetUserTransform(self.parent.get_spharm_lh_transform())

        # Cyan / Teal ghost outline / surface
        actor.GetProperty().SetColor(0.2, 0.85, 0.95)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        actor.GetProperty().SetInterpolationToPhong()

        self.parent.mesh_renderer.AddActor(actor)
        self.patient_overlay_actor = actor
        self.parent.mesh_vtkWidget.GetRenderWindow().Render()
        self.parent.signal_log_message.emit(f"Overlaid patient mesh ({os.path.basename(mesh_path)}) in 3D View.")
