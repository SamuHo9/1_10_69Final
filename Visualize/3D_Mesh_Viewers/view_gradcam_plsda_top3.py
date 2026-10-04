#!/usr/bin/env python3
"""
================================================================================
Interactive 3D Viewer: PLS-DA Components Comparison & Distance Mapping
================================================================================
Features:
- Side-by-side synchronized viewports comparing PLS-DA components (Component 1 & 2).
- Camera synchronization across viewports (rotate/pan one, all follow).
- Continuous scrubbing across -3.0 to +3.0 (step 0.1) or key SD milestones.
- Multiple visualization modes:
  * [M1] Distance Mapping (Absolute displacement magnitude in mm)
  * [M2] Signed Deformation (Inward atrophy [Blue] vs Outward expansion [Red])
  * [M3] Grad-CAM Heatmap (ResNet class activation attention on subfields)
- Color themes: Pure White background (#FFFFFF) by default, toggleable to Dark Navy via [B].
- Play/Pause animation loop across [-3SD, +3SD].
- Keyboard hotkeys:
  * Left/Right arrow or [ / ]: Step -0.1 / +0.1 SD
  * Space: Play / Pause continuous deformation animation
  * M: Cycle scalar mode (DistanceMapping -> SignedDistance -> GradCAM)
  * B: Toggle White / Dark background theme
  * F: Front view, T: Top view, S: Side view, R: Reset view
  * W: Toggle wireframe
  * Q / Esc: Exit viewer
================================================================================
"""

import os
import re
import sys
import glob
import vtk
from vtk.util import numpy_support
import numpy as np
import pandas as pd

# Suppress VTK warnings
vtk.vtkObject.GlobalWarningDisplayOff()

def popup_select_directory(title="Select Output Folder"):
    """Folder dialog fallback using tkinter."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title=title, initialdir=os.getcwd())
        root.destroy()
        return folder if folder else None
    except Exception:
        return None

def load_polydata(filepath):
    """Load VTK polydata file."""
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        return None
    return poly


# =============================================================================
# Color Lookup Tables
# =============================================================================

def build_lut_distance_mapping(max_val=0.16):
    """Jet / Turbo colormap for positive distance magnitude (0 to max_val mm)."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(0.0, max_val)
    lut.Build()
    # Blue (0) -> Cyan -> Green -> Yellow -> Red (max)
    for i in range(256):
        t = i / 255.0
        # Jet colormap formula
        r = np.clip(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0)
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
    return lut

def build_lut_signed_distance(max_val=0.16):
    """Diverging Blue-White-Red colormap for signed deformation (-max to +max mm)."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(-max_val, max_val)
    lut.Build()
    for i in range(256):
        t = (i / 255.0) * 2.0 - 1.0 # -1 to 1
        if t < 0:
            # Blue to White
            frac = 1.0 + t # 0 to 1
            r = frac
            g = frac
            b = 1.0
        else:
            # White to Red
            frac = 1.0 - t # 1 to 0
            r = 1.0
            g = frac
            b = frac
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
    return lut

def build_lut_gradcam():
    """Hot / Inferno colormap for Grad-CAM attention (0 to 1.0)."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(0.0, 1.0)
    lut.Build()
    for i in range(256):
        t = i / 255.0
        # Dark purple -> Red -> Orange -> Yellow
        r = np.clip(t * 1.5, 0.0, 1.0)
        g = np.clip((t - 0.3) * 1.5, 0.0, 1.0)
        b = np.clip((t - 0.7) * 3.0, 0.0, 1.0)
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
    return lut


# =============================================================================
# Main Top-3 Comparison Viewer Class
# =============================================================================

class Top3DistanceMappingViewer:
    """
    Side-by-side synchronized 3-panel viewer for Top 3 PLS-DA components.
    """
    MODES = ["DistanceMapping", "SignedDistance", "GradCAM_Importance"]
    MODE_TITLES = {
        "DistanceMapping": "Distance Mapping (Displacement Magnitude, mm)",
        "SignedDistance": "Signed Deformation (Blue: Inward Atrophy | Red: Expansion)",
        "GradCAM_Importance": "ResNet Grad-CAM Attention Heatmap (Subfield Focus)"
    }

    def __init__(self, comp_dirs, comp_names, white_bg=True):
        """
        comp_dirs: list of directory paths for components (Component 1 & 2).
        comp_names: list of strings (e.g. ['PLS1', 'PLS2']).
        white_bg: boolean flag for pure white background (#FFFFFF)
        """
        self.comp_dirs = comp_dirs
        self.comp_names = comp_names
        self.num_comps = len(comp_dirs)
        self.white_bg = white_bg
        
        # Discover steps in each component
        self.comp_steps = []
        for cdir in comp_dirs:
            steps_folder = os.path.join(cdir, "steps")
            files = sorted(glob.glob(os.path.join(steps_folder, "*.vtk"))) if os.path.isdir(steps_folder) else []
            if not files:
                # Fallback to milestone VTKs in cdir
                files = sorted(glob.glob(os.path.join(cdir, "*.vtk")))
            self.comp_steps.append(files)
            
        self.total_steps = len(self.comp_steps[0])
        if self.total_steps == 0:
            print("[ERROR] No VTK meshes found in component directories.")
            return
            
        # Parse SD values from filenames
        self.sd_values = []
        for f in self.comp_steps[0]:
            m = re.search(r"val_([+-]?\d+\.?\d*)", os.path.basename(f))
            if m:
                self.sd_values.append(float(m.group(1)))
            else:
                self.sd_values.append(0.0)
                
        # Scan meshes to determine exact physical scalar limits
        self.global_max_dist = 0.16
        self.global_max_signed = 0.16
        d_maxs = []
        for step_list in self.comp_steps:
            for f in step_list:
                if "plus3SD" in f or "minus3SD" in f or "val_+3.0" in f or "val_-3.0" in f:
                    poly = load_polydata(f)
                    if poly:
                        arr = poly.GetPointData().GetArray("DistanceMapping")
                        if arr:
                            d_maxs.append(arr.GetRange()[1])
        if d_maxs:
            self.global_max_dist = max(0.01, float(max(d_maxs)))
            self.global_max_signed = self.global_max_dist
        print(f"[INFO] Calibrated Colormap Range: 0.0 to {self.global_max_dist:.4f} mm")

        # Start at +2.0 SD (or closest) so red deformation is immediately visible on open!
        diffs_2 = [abs(v - 2.0) for v in self.sd_values]
        self.current_step_idx = int(np.argmin(diffs_2)) if diffs_2 else 0
        
        # State variables
        self.current_mode_idx = 0
        self.animating = False
        self.anim_direction = 1
        self.wireframe = False
        self.autoscale = False # False = calibrated fixed scale, True = dynamic per step
        
        # Pre-build color lookup tables
        self.luts = {
            "DistanceMapping": build_lut_distance_mapping(max_val=self.global_max_dist),
            "SignedDistance": build_lut_signed_distance(max_val=self.global_max_signed),
            "GradCAM_Importance": build_lut_gradcam()
        }
        
        self.setup_ui()
        self.apply_theme()
        self.update_all_meshes()
        self.sync_cameras()
        self.start()

    def setup_ui(self):
        # Create Main RenderWindow
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(1500, 780)
        self.render_window.SetWindowName(f"PLS-DA Components Distance Mapping Viewer ({' vs '.join(self.comp_names)})")
        self.render_window.SetMultiSamples(8)
        
        # Overlay Background Renderer (Help text, mode text, SD indicator)
        self.bg_ren = vtk.vtkRenderer()
        self.bg_ren.SetViewport(0, 0, 1, 1)
        self.bg_ren.InteractiveOff()
        self.bg_ren.SetLayer(0)
        self.render_window.SetNumberOfLayers(2)
        self.render_window.AddRenderer(self.bg_ren)
        
        # Top Header Banner
        self.header_actor = vtk.vtkTextActor()
        hp = self.header_actor.GetTextProperty()
        hp.SetFontSize(16)
        hp.BoldOn()
        hp.SetShadow(True)
        self.header_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.header_actor.GetPositionCoordinate().SetValue(0.015, 0.95)
        self.bg_ren.AddActor(self.header_actor)
        
        # Bottom Instructions & Controls Bar
        self.help_actor = vtk.vtkTextActor()
        help_prop = self.help_actor.GetTextProperty()
        help_prop.SetFontSize(12)
        self.help_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.help_actor.GetPositionCoordinate().SetValue(0.015, 0.015)
        self.help_actor.SetInput(
            "[Left/Right] Step SD \u00b10.1   [Space] Play/Pause   [M] Cycle Colormap   [B] White/Dark BG   [E] Export Image   [F] Front Arch  [T] Top  [S] Side   [Q] Quit"
        )
        self.bg_ren.AddActor(self.help_actor)
        
        # Interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Observers
        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)
        self.interactor.AddObserver("InteractionEvent", self.on_interaction)
        self.interactor.AddObserver("TimerEvent", self.on_timer)
        self.timer_id = None
        
        # 3 Sub-Viewports (one per component)
        self.renderers = []
        self.actors = []
        self.mappers = []
        self.panel_labels = []
        self.stats_labels = []
        self.scalar_bars = []
        
        y_bottom = 0.07
        y_top = 0.93
        for i in range(self.num_comps):
            x_left = i / self.num_comps
            x_right = (i + 1) / self.num_comps
            
            ren = vtk.vtkRenderer()
            ren.SetViewport(x_left, y_bottom, x_right, y_top)
            ren.SetBackground(0.08 + i*0.01, 0.09 + i*0.01, 0.12 + i*0.01)
            ren.SetLayer(1)
            ren.GetActiveCamera().SetParallelProjection(True)
            
            # Lights
            ren.RemoveAllLights()
            key_light = vtk.vtkLight()
            key_light.SetLightTypeToCameraLight()
            key_light.SetPosition(0.3, 0.5, 1.0)
            key_light.SetIntensity(0.9)
            ren.AddLight(key_light)
            
            fill_light = vtk.vtkLight()
            fill_light.SetLightTypeToCameraLight()
            fill_light.SetPosition(-0.4, -0.3, 0.7)
            fill_light.SetIntensity(0.4)
            ren.AddLight(fill_light)
            
            # Mapper & Actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetLookupTable(self.luts[self.MODES[self.current_mode_idx]])
            mapper.ScalarVisibilityOn()
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray(self.MODES[self.current_mode_idx])
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetInterpolationToGouraud()
            prop.SetAmbient(0.2)
            prop.SetDiffuse(0.8)
            prop.SetSpecular(0.25)
            prop.SetSpecularPower(30)
            ren.AddActor(actor)
            
            # Panel Top Title (Component Name & Rank)
            p_label = vtk.vtkTextActor()
            lp = p_label.GetTextProperty()
            lp.SetFontSize(16)
            lp.SetColor(1, 1, 1)
            lp.SetJustificationToCentered()
            lp.BoldOn()
            lp.SetShadow(True)
            p_label.SetInput(f"Rank {i+1}: {self.comp_names[i]}")
            p_label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            p_label.GetPositionCoordinate().SetValue(0.5, 0.94)
            ren.AddActor(p_label)
            
            # Panel Bottom Stats (Mean & Max mm)
            s_label = vtk.vtkTextActor()
            sp = s_label.GetTextProperty()
            sp.SetFontSize(12)
            sp.SetColor(0.8, 0.9, 1.0)
            sp.SetJustificationToCentered()
            s_label.SetInput("Displacement: -- mm")
            s_label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            s_label.GetPositionCoordinate().SetValue(0.5, 0.04)
            ren.AddActor(s_label)
            
            # Scalar bar legend
            sbar = vtk.vtkScalarBarActor()
            sbar.SetLookupTable(self.luts[self.MODES[self.current_mode_idx]])
            sbar.SetNumberOfLabels(4)
            sbar.GetTitleTextProperty().SetFontSize(9)
            sbar.GetTitleTextProperty().SetColor(0.9, 0.9, 0.9)
            sbar.GetLabelTextProperty().SetFontSize(8)
            sbar.GetLabelTextProperty().SetColor(0.9, 0.9, 0.9)
            sbar.SetWidth(0.12)
            sbar.SetHeight(0.32)
            sbar.SetPosition(0.86, 0.08)
            ren.AddActor(sbar)
            
            self.render_window.AddRenderer(ren)
            self.renderers.append(ren)
            self.mappers.append(mapper)
            self.actors.append(actor)
            self.panel_labels.append(p_label)
            self.stats_labels.append(s_label)
            self.scalar_bars.append(sbar)

    def apply_theme(self):
        """Applies pure white or dark navy theme across all renderers and text actors."""
        if self.white_bg:
            bg_color = (1.0, 1.0, 1.0)
            text_header = (0.1, 0.12, 0.18)
            text_help = (0.2, 0.25, 0.35)
            text_panel = (0.1, 0.12, 0.18)
            text_stats = (0.2, 0.25, 0.35)
            text_sbar = (0.1, 0.12, 0.18)
        else:
            bg_color = (0.06, 0.07, 0.10)
            text_header = (1.0, 0.95, 0.8)
            text_help = (0.65, 0.85, 1.0)
            text_panel = (1.0, 1.0, 1.0)
            text_stats = (0.8, 0.9, 1.0)
            text_sbar = (0.9, 0.9, 0.9)
            
        self.bg_ren.SetBackground(*bg_color)
        self.header_actor.GetTextProperty().SetColor(*text_header)
        self.help_actor.GetTextProperty().SetColor(*text_help)
        
        for i in range(self.num_comps):
            ren = self.renderers[i]
            if self.white_bg:
                ren.SetBackground(1.0, 1.0, 1.0)
            else:
                ren.SetBackground(0.08 + i*0.01, 0.09 + i*0.01, 0.12 + i*0.01)
            self.panel_labels[i].GetTextProperty().SetColor(*text_panel)
            self.stats_labels[i].GetTextProperty().SetColor(*text_stats)
            self.scalar_bars[i].GetTitleTextProperty().SetColor(*text_sbar)
            self.scalar_bars[i].GetLabelTextProperty().SetColor(*text_sbar)

    def sync_cameras(self, source_idx=0):
        """Synchronize camera view across all 3 viewports."""
        src_cam = self.renderers[source_idx].GetActiveCamera()
        pos = src_cam.GetPosition()
        focal = src_cam.GetFocalPoint()
        up = src_cam.GetViewUp()
        scale = src_cam.GetParallelScale()
        
        for idx, ren in enumerate(self.renderers):
            if idx != source_idx:
                cam = ren.GetActiveCamera()
                cam.SetPosition(pos)
                cam.SetFocalPoint(focal)
                cam.SetViewUp(up)
                cam.SetParallelScale(scale)
                ren.ResetCameraClippingRange()

    def update_all_meshes(self):
        """Update meshes in all 3 viewports for the current step and mode."""
        active_mode = self.MODES[self.current_mode_idx]
        current_sd = self.sd_values[self.current_step_idx]
        
        # Load the 3 meshes first to check their actual scalar ranges
        current_polys = []
        for i in range(self.num_comps):
            fpath = self.comp_steps[i][self.current_step_idx]
            poly = load_polydata(fpath)
            current_polys.append(poly)
            
        # Determine active colormap and range
        if active_mode == "DistanceMapping":
            if self.autoscale:
                step_maxs = [p.GetPointData().GetArray(active_mode).GetRange()[1] 
                             for p in current_polys if p and p.GetPointData().GetArray(active_mode)]
                cur_max = max(0.005, max(step_maxs)) if step_maxs else self.global_max_dist
            else:
                cur_max = self.global_max_dist
            active_lut = build_lut_distance_mapping(cur_max)
            scale_desc = f"[0.000 to {cur_max:.3f} mm]"
        elif active_mode == "SignedDistance":
            if self.autoscale:
                step_maxs = [max(abs(p.GetPointData().GetArray(active_mode).GetRange()[0]), 
                                 abs(p.GetPointData().GetArray(active_mode).GetRange()[1]))
                             for p in current_polys if p and p.GetPointData().GetArray(active_mode)]
                cur_max = max(0.005, max(step_maxs)) if step_maxs else self.global_max_signed
            else:
                cur_max = self.global_max_signed
            active_lut = build_lut_signed_distance(cur_max)
            scale_desc = f"[-{cur_max:.3f} to +{cur_max:.3f} mm]"
        else: # GradCAM
            active_lut = self.luts["GradCAM_Importance"]
            scale_desc = "[0.00 to 1.00]"
            
        mode_title = self.MODE_TITLES[active_mode]
        auto_label = "[Auto-Scale: ON]" if self.autoscale else "[Fixed Scale]"
        self.header_actor.SetInput(
            f"Mode: {mode_title}   |   Scale: {scale_desc} {auto_label}\nLatent Trajectory: SD = {current_sd:+.1f}   [Step {self.current_step_idx+1}/{self.total_steps}]"
        )
        
        for i in range(self.num_comps):
            poly = current_polys[i]
            if poly is None:
                continue
                
            self.mappers[i].SetInputData(poly)
            self.mappers[i].SetLookupTable(active_lut)
            self.mappers[i].SetScalarRange(active_lut.GetRange())
            self.mappers[i].SelectColorArray(active_mode)
            self.scalar_bars[i].SetLookupTable(active_lut)
            self.scalar_bars[i].SetTitle("mm" if "Distance" in active_mode else "Score")
            
            # Extract point array for stats
            arr = poly.GetPointData().GetArray(active_mode)
            if arr:
                np_arr = numpy_support.vtk_to_numpy(arr)
                if active_mode == "DistanceMapping":
                    self.stats_labels[i].SetInput(f"Mean: {np.mean(np_arr):.3f} mm | Max: {np.max(np_arr):.3f} mm")
                elif active_mode == "SignedDistance":
                    inward = np.mean(np.clip(-np_arr, 0, None))
                    outward = np.mean(np.clip(np_arr, 0, None))
                    self.stats_labels[i].SetInput(f"Inward: -{inward:.3f} mm | Outward: +{outward:.3f} mm")
                else:
                    self.stats_labels[i].SetInput(f"Mean Attention: {np.mean(np_arr):.3f} | Peak: {np.max(np_arr):.3f}")
                    
        self.render_window.Render()

    def on_interaction(self, obj, event):
        """Called whenever the user rotates/pans/zooms with the mouse."""
        # Find which renderer triggered the event, or default to 0
        self.sync_cameras(0)
        self.render_window.Render()

    def on_key_press(self, obj, event):
        key = (obj.GetKeySym() or "").lower()
        
        if key in ("right", "bracketright", "period"):
            if self.current_step_idx < self.total_steps - 1:
                self.current_step_idx += 1
                self.update_all_meshes()
        elif key in ("left", "bracketleft", "comma"):
            if self.current_step_idx > 0:
                self.current_step_idx -= 1
                self.update_all_meshes()
        elif key == "space":
            # Toggle animation
            self.animating = not self.animating
            if self.animating:
                if self.timer_id is None:
                    self.timer_id = self.interactor.CreateRepeatingTimer(40) # ~25 fps
            else:
                if self.timer_id is not None:
                    self.interactor.DestroyTimer(self.timer_id)
                    self.timer_id = None
        elif key == "m":
            # Cycle colormap mode
            self.current_mode_idx = (self.current_mode_idx + 1) % len(self.MODES)
            self.update_all_meshes()
        elif key == "b":
            # Toggle background between Pure White (#FFFFFF) and Dark Navy
            self.white_bg = not self.white_bg
            self.apply_theme()
            self.render_window.Render()
        elif key == "a":
            # Toggle auto-scale
            self.autoscale = not self.autoscale
            self.update_all_meshes()
        elif key == "w":
            self.wireframe = not self.wireframe
            for a in self.actors:
                if self.wireframe:
                    a.GetProperty().SetRepresentationToWireframe()
                else:
                    a.GetProperty().SetRepresentationToSurface()
            self.render_window.Render()
        elif key in ("e", "p"):
            self.export_screenshot()
        elif key == "f":
            self.set_camera_view("front")
        elif key == "t":
            self.set_camera_view("top")
        elif key == "s":
            self.set_camera_view("side")
        elif key == "r":
            self.set_camera_view("reset")
        elif key in ("q", "escape"):
            self.interactor.TerminateApp()

    def export_screenshot(self):
        """Export publication-quality PNG capture of the current window."""
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.render_window)
        w2if.SetInputBufferTypeToRGBA()
        w2if.ReadFrontBufferOff()
        w2if.Update()

        active_mode = self.MODES[self.current_mode_idx]
        current_sd = self.sd_values[self.current_step_idx]
        sd_str = f"sd_{current_sd:+.1f}".replace("+", "plus").replace("-", "minus")
        
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        out_dir = os.path.join(repo_root, "Model_Results_Excel", "10_GradCAM_and_Distance_Mapping", "interactive_exports")
        os.makedirs(out_dir, exist_ok=True)
        
        filename = f"capture_{active_mode}_{sd_str}_step{self.current_step_idx+1}.png"
        out_path = os.path.join(out_dir, filename)
        
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(out_path)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
        print(f"\n[SCREENSHOT SAVED] -> {out_path}")
        
        self.help_actor.SetInput(f"[SAVED]: {filename}  (Press [E] to export again)")
        self.render_window.Render()

    def on_timer(self, obj, event):
        if not self.animating:
            return
        self.current_step_idx += self.anim_direction
        if self.current_step_idx >= self.total_steps - 1:
            self.current_step_idx = self.total_steps - 1
            self.anim_direction = -1
        elif self.current_step_idx <= 0:
            self.current_step_idx = 0
            self.anim_direction = 1
        self.update_all_meshes()

    def set_camera_view(self, axis):
        is_left = any("left" in d.lower() for d in self.comp_dirs)
        for ren in self.renderers:
            cam = ren.GetActiveCamera()
            cam.SetParallelProjection(True)
            ren.ResetCamera()
            fp = cam.GetFocalPoint()
            dist = cam.GetDistance()
            if axis in ("front", "crescent", "arch"):
                # Calibrated crescent arch view (matches user's presentation)
                if is_left:
                    cam.SetPosition(fp[0], fp[1], fp[2] + dist)
                else:
                    cam.SetPosition(fp[0], fp[1], fp[2] - dist)
                cam.SetViewUp(0, -1, 0)
            elif axis == "top":
                cam.SetPosition(fp[0], fp[1], fp[2] + dist)
                cam.SetViewUp(0, 1, 0)
            elif axis == "side":
                cam.SetPosition(fp[0] + dist, fp[1], fp[2])
                cam.SetViewUp(0, 0, 1)
            elif axis == "reset":
                ren.ResetCamera()
            ren.ResetCameraClippingRange()
        self.sync_cameras(0)
        self.render_window.Render()

    def start(self):
        self.set_camera_view("front")
        self.interactor.Initialize()
        self.interactor.Start()


# =============================================================================
# CLI Main
# =============================================================================

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Priority directories to inspect if none provided
    candidate_dirs = [
        os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "Dataset_1", "right"),
        os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "Dataset_1", "left"),
        os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "All_Augment_tain", "right"),
        os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "All_Augment_tain", "left"),
        os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "Ds005602", "right"),
        os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "Ds005602", "left"),
    ]
    
    target_dir = None
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        target_dir = os.path.abspath(sys.argv[1])
    else:
        for cdir in candidate_dirs:
            if os.path.isdir(cdir):
                target_dir = cdir
                break
                
    if not target_dir:
        print("[INFO] Selecting output folder...")
        target_dir = popup_select_directory("Select Output_GradCAM_PLSDA Folder")
        if not target_dir:
            print("Exiting: No folder selected.")
            return

    # Check for summary CSV or discover component folders
    summary_csv = os.path.join(target_dir, "plsda_8_components_summary.csv")
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        top_df = df[df["Is_Top3"] == True].sort_values("Top3_Rank")
        all_comps = top_df["Component"].tolist()
    else:
        # Fallback: scan subfolders matching PLS*
        subfolders = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d)) and d.startswith("PLS")]
        all_comps = sorted(subfolders)

    # Per user request: "ขอแค่ 1 and 2" -> strictly select Component 1 and 2 (PLS1 and PLS2)
    # Check if user explicitly asked for 3 via CLI flag, otherwise default strictly to Top 2
    if "--all" in sys.argv or "--top3" in sys.argv:
        comp_names = all_comps[:3]
    else:
        # Strictly Component 1 and Component 2
        comp_names = [c for c in all_comps if c in ("PLS1", "PLS2")]
        if len(comp_names) < 2:
            comp_names = all_comps[:2]

    if not comp_names:
        print(f"[ERROR] No PLS component folders found in {target_dir}")
        return

    comp_dirs = [os.path.join(target_dir, c) for c in comp_names]
    print(f"Opening Top 2 Viewer (Components: {', '.join(comp_names)}) for:\n  " + "\n  ".join(comp_dirs))
    Top3DistanceMappingViewer(comp_dirs, comp_names, white_bg=True)


if __name__ == "__main__":
    main()
