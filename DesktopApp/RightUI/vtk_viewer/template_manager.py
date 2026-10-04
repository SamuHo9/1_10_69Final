import os
import vtk

class TemplateManager:
    """
    Manages SPHARM and ICP reference templates, coordinate transforms,
    multi-mesh cohort overlays, and 3D viewport legend text.
    """
    def __init__(self, parent_viewer):
        self.parent = parent_viewer
        self.template_actor = None
        self.template_actors = []
        self.multi_mesh_actors = []
        self.multi_mesh_paths = []
        self.is_overlay_active = False

    def get_spharm_lh_transform(self):
        """
        Precomputed Kabsch rigid transform from template_spharm_left to template_spharm_right,
        with 180° view rotation around the horizontal viewing axis so LH SPHARM meshes face
        the canonical upright C-crescent orientation matching RH without altering actual file coordinates.
        """
        tr = vtk.vtkTransform()
        tr.PostMultiply()
        tr.Translate(0.00260724, 0.05516517, 0.10794069)
        mat = vtk.vtkMatrix4x4()
        R = [
            [ 0.69549754,  0.70316967, -0.14776868],
            [-0.66605966,  0.55378502, -0.49968658],
            [-0.26953237,  0.44595355,  0.85350907]
        ]
        for i in range(3):
            for j in range(3):
                mat.SetElement(i, j, R[i][j])
        tr.Concatenate(mat)
        tr.Translate(-0.05795671, 0.04367124, -0.06888033)

        # 180° rotation around horizontal view axis to correct upside-down & backward orientation
        tr.Translate(-0.039, -0.093, -0.001)
        tr.RotateWXYZ(180, 0.806, 0.0, 0.592)
        tr.Translate(0.039, 0.093, 0.001)
        return tr

    def get_template_paths(self, side=None):
        """Returns (candidates_list, resolved_side, mod_type) where candidates_list is a list of (filepath, side_str)"""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        templates_dir = os.path.join(project_root, "Templates")
        
        is_spharm = ("spharm" in self.parent.current_module_name.lower()) or (
            self.parent.current_mesh_path and "spharm" in self.parent.current_mesh_path.lower()
        ) or (
            self.multi_mesh_paths and any("spharm" in p.lower() for p in self.multi_mesh_paths[:3])
        )
        mod_type = "SPHARM" if is_spharm else "ICP"
        sub_dir = "SPHARM" if is_spharm else "ICP"
        
        detected_side = None
        if side:
            s_low = str(side).lower()
            if s_low == "all":
                detected_side = "all"
            elif s_low in ("right", "rh") or "right" in s_low or "rh" in s_low:
                detected_side = "right"
            elif s_low in ("left", "lh") or "left" in s_low or "lh" in s_low:
                detected_side = "left"

        # 1. Prioritize active side filter
        if detected_side is None:
            if self.parent.current_side_filter in ("rh", "right"):
                detected_side = "right"
            elif self.parent.current_side_filter in ("lh", "left"):
                detected_side = "left"
            elif self.parent.current_side_filter == "all":
                if self.is_overlay_active or self.parent.mesh_actor is None:
                    detected_side = "all"

        # 2. If still None and a single mesh is loaded, infer from its filename
        if detected_side is None and self.parent.current_mesh_path and not self.is_overlay_active:
            mesh_to_check = self.parent.current_mesh_path
            fp_low = mesh_to_check.replace("\\", "/").lower()
            bn = os.path.basename(mesh_to_check).lower()
            if "rh_" in bn or "_rh." in bn or "_rh_" in bn or "right" in bn or "/right/" in fp_low or "/rh/" in fp_low or "right_hippocampus" in fp_low:
                detected_side = "right"
            elif "lh_" in bn or "_lh." in bn or "_lh_" in bn or "left" in bn or "/left/" in fp_low or "/lh/" in fp_low or "left_hippocampus" in fp_low:
                detected_side = "left"

        # 3. Default fallback
        if detected_side is None:
            detected_side = "all" if self.parent.current_side_filter == "all" else "left"

        def get_cand(s):
            if is_spharm:
                c = os.path.join(templates_dir, sub_dir, f"template_spharm_{s}.vtk")
                if not os.path.isfile(c):
                    c = os.path.join(templates_dir, sub_dir, f"template_spharm_{s}.ply")
            else:
                c = os.path.join(templates_dir, sub_dir, f"template_mean_{s}.vtk")
                if not os.path.isfile(c):
                    c = os.path.join(templates_dir, sub_dir, f"template_mean_{s}.ply")
            return c if os.path.isfile(c) else None

        results = []
        if detected_side == "all":
            left_cand = get_cand("left")
            right_cand = get_cand("right")
            if left_cand:
                results.append((left_cand, "left"))
            if right_cand:
                results.append((right_cand, "right"))
        else:
            cand = get_cand(detected_side)
            if cand:
                results.append((cand, detected_side))

        return results, detected_side, mod_type

    def get_template_path(self, side=None):
        results, detected_side, mod_type = self.get_template_paths(side=side)
        if results:
            return results[0][0], detected_side, mod_type
        return None, detected_side, mod_type

    def clear_template_actors(self):
        for act in self.template_actors:
            self.parent.mesh_renderer.RemoveActor(act)
        self.template_actors = []
        self.template_actor = None

    def update_template_overlay(self, side=None):
        if not self.parent.template_cb.isChecked():
            if self.template_actors or self.template_actor is not None:
                self.clear_template_actors()
                self.update_legend()
                self.parent.mesh_vtkWidget.GetRenderWindow().Render()
                self.parent.signal_log_message.emit("Reference template hidden.")
            return

        tmpl_specs, resolved_side, mod_type = self.get_template_paths(side=side)
        if not tmpl_specs:
            self.parent.signal_log_message.emit(f"[WARNING] Reference template not found for {resolved_side} ({mod_type}).")
            return

        self.clear_template_actors()

        is_spharm = ("spharm" in str(self.parent.current_module_name).lower()) or any(
            "spharm" in p[0].lower() for p in tmpl_specs
        )

        for tmpl_path, tmpl_side in tmpl_specs:
            if tmpl_path.endswith(".ply"):
                reader = vtk.vtkPLYReader()
            else:
                reader = vtk.vtkPolyDataReader()
            reader.SetFileName(tmpl_path)
            reader.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            if is_spharm and tmpl_side == "left":
                actor.SetUserTransform(self.get_spharm_lh_transform())
            
            # Warm Amber / Gold translucent ghost surface
            prop = actor.GetProperty()
            prop.SetColor(0.95, 0.76, 0.20)
            prop.SetOpacity(0.45)
            prop.SetSpecular(0.3)
            prop.SetSpecularPower(20)
            prop.SetInterpolationToPhong()

            self.parent.mesh_renderer.AddActor(actor)
            self.template_actors.append(actor)

        if self.template_actors:
            self.template_actor = self.template_actors[0]

        self.parent.reset_3d_camera(side=resolved_side)
        if resolved_side == "all":
            tmpl_name = f"{mod_type} Left & Right Mean"
        elif resolved_side == "left":
            tmpl_name = f"{mod_type} Left Mean"
        else:
            tmpl_name = f"{mod_type} Right Mean"

        self.update_legend(tmpl_name=tmpl_name)
        self.parent.mesh_vtkWidget.GetRenderWindow().Render()
        names_str = ", ".join([f"{os.path.basename(p)} ({s.upper()})" for p, s in tmpl_specs])
        self.parent.signal_log_message.emit(f"[INFO] Reference template overlaid: {names_str}")

    def set_template_visible(self, visible: bool):
        self.parent.template_cb.blockSignals(True)
        self.parent.template_cb.setChecked(visible)
        self.parent.template_cb.blockSignals(False)
        self.update_template_overlay()

    def clear_multi_mesh_actors(self):
        for act in self.multi_mesh_actors:
            self.parent.mesh_renderer.RemoveActor(act)
        self.multi_mesh_actors = []
        self.multi_mesh_paths = []
        self.is_overlay_active = False

    def display_all_meshes(self, filepaths, side_filter="all"):
        self.parent.current_side_filter = side_filter
        self.parent.current_mesh_path = None
        if not filepaths:
            self.clear_multi_mesh_actors()
            self.update_legend()
            self.parent.mesh_vtkWidget.GetRenderWindow().Render()
            return

        if self.parent.mesh_actor is not None:
            self.parent.mesh_renderer.RemoveActor(self.parent.mesh_actor)
            self.parent.mesh_actor = None
        self.clear_multi_mesh_actors()
        self.parent.clear_gradcam_view(render_now=False)
        self.is_overlay_active = True
        self.multi_mesh_paths = filepaths

        # Soft translucent color palettes
        left_colors = [
            (0.40, 0.70, 0.95), (0.35, 0.60, 0.90), (0.45, 0.80, 0.98),
            (0.50, 0.65, 0.95), (0.30, 0.75, 0.85), (0.55, 0.70, 1.00)
        ]
        right_colors = [
            (0.95, 0.65, 0.50), (0.90, 0.55, 0.45), (0.98, 0.70, 0.55),
            (0.92, 0.60, 0.40), (0.85, 0.50, 0.45), (0.95, 0.75, 0.60)
        ]

        loaded_count = 0
        for i, fp in enumerate(filepaths):
            if not (fp.endswith(".vtk") or fp.endswith(".ply") or fp.endswith(".nii.gz") or fp.endswith(".mgz") or fp.endswith(".nii")):
                continue
            if fp.endswith(".ply"):
                r = vtk.vtkPLYReader()
                r.SetFileName(fp)
                r.Update()
                output_port = r.GetOutputPort()
            elif fp.endswith(".vtk"):
                r = vtk.vtkPolyDataReader()
                r.SetFileName(fp)
                r.Update()
                output_port = r.GetOutputPort()
            else:
                r = vtk.vtkNIFTIImageReader()
                r.SetFileName(fp)
                r.Update()
                mc = vtk.vtkMarchingCubes()
                mc.SetInputConnection(r.GetOutputPort())
                mc.SetValue(0, 0.5)
                sm = vtk.vtkWindowedSincPolyDataFilter()
                sm.SetInputConnection(mc.GetOutputPort())
                sm.SetNumberOfIterations(15)
                sm.BoundarySmoothingOff()
                sm.FeatureEdgeSmoothingOff()
                sm.SetPassBand(0.1)
                sm.NonManifoldSmoothingOn()
                sm.NormalizeCoordinatesOn()
                sm.Update()
                output_port = sm.GetOutputPort()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(output_port)
            mapper.ScalarVisibilityOff()

            act = vtk.vtkActor()
            act.SetMapper(mapper)

            fp_low = fp.replace("\\", "/").lower()
            bn = os.path.basename(fp).lower()
            is_rh = ("rh" in bn or "right" in bn or "/right/" in fp_low or side_filter == "rh")
            if is_rh:
                c = right_colors[i % len(right_colors)]
            else:
                c = left_colors[i % len(left_colors)]

            is_spharm = ("spharm" in str(self.parent.current_module_name).lower()) or any(
                "spharm" in p.lower() for p in filepaths[:3]
            )
            if is_spharm and not is_rh:
                act.SetUserTransform(self.get_spharm_lh_transform())

            act.GetProperty().SetColor(*c)
            act.GetProperty().SetOpacity(0.38)
            act.GetProperty().SetSpecular(0.25)
            act.GetProperty().SetSpecularPower(15)
            act.GetProperty().SetInterpolationToPhong()

            self.parent.mesh_renderer.AddActor(act)
            self.multi_mesh_actors.append(act)
            loaded_count += 1

        if side_filter in ("rh", "right"):
            chosen_side = "right"
        elif side_filter in ("lh", "left"):
            chosen_side = "left"
        else:
            chosen_side = "all"

        if self.parent.template_cb.isChecked():
            self.update_template_overlay(side=chosen_side)

        self.parent.reset_3d_camera(side=chosen_side)
        self.update_legend()
        self.parent.mesh_vtkWidget.GetRenderWindow().Render()
        self.parent.signal_log_message.emit(f"[INFO] Superimposed {loaded_count} meshes in 3D view (Filter: {side_filter.upper()}).")

    def update_legend(self, tmpl_name=None):
        parts = []
        if self.is_overlay_active and self.multi_mesh_actors:
            side_str = "Right" if self.parent.current_side_filter == "rh" else ("Left" if self.parent.current_side_filter == "lh" else "All")
            parts.append(f'<span style="color: #1abc9c; font-weight: bold;">&#9679; Overlaid Meshes: {len(self.multi_mesh_actors)} ({side_str})</span>')
        elif self.parent.current_mesh_path and self.parent.mesh_actor:
            m_name = os.path.basename(self.parent.current_mesh_path)
            if len(m_name) > 36:
                m_name = m_name[:16] + "..." + m_name[-16:]
            parts.append(f'<span style="color: #79c0ff; font-weight: bold;">&#9679; Patient: {m_name}</span>')
            
        if self.parent.template_cb.isChecked() and (self.template_actors or self.template_actor is not None):
            if not tmpl_name:
                _, side, mod_type = self.get_template_paths()
                if side == "all":
                    tmpl_name = f"{mod_type} Left & Right Mean"
                elif side == "left":
                    tmpl_name = f"{mod_type} Left Mean"
                else:
                    tmpl_name = f"{mod_type} Right Mean"
            parts.append(f'<span style="color: #f1c40f; font-weight: bold;">&#9679; Template: {tmpl_name} (Ghost)</span>')
            
        if getattr(self.parent, 'current_diagnostic_info', None):
            parts.append(self.parent.current_diagnostic_info)
            
        if parts:
            self.parent.mesh_legend_lbl.setText("  |  ".join(parts))
            self.parent.mesh_legend_lbl.setVisible(True)
        else:
            self.parent.mesh_legend_lbl.setText("")
            self.parent.mesh_legend_lbl.setVisible(False)
