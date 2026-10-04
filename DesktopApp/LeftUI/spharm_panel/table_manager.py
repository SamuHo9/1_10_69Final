import os
import glob
from PyQt6.QtWidgets import QTableWidgetItem
from PyQt6.QtCore import Qt

class SPHARMTableManager:
    """
    Manages SPHARM results table population, side tab filtering (All / LH / RH),
    multi-selection synchronization, and best-mesh resolution for SPHARM Panel.
    """
    def __init__(self, panel):
        self.p = panel
        self.all_files = []
        self.current_side_filter = "all"

    def populate_results_table(self):
        target_base = self.p.spharm_dir_input.text().strip()
        if not target_base or not os.path.isdir(target_base):
            if hasattr(self.p, 'get_default_output_dir'):
                def_out = self.p.get_default_output_dir()
                if def_out and os.path.isdir(def_out):
                    target_base = def_out
                    self.p.spharm_dir_input.setText(target_base)

        if (not target_base or not os.path.isdir(target_base)) and self.p.get_output_folder:
            out_base = self.p.get_output_folder().strip()
            if out_base and os.path.isdir(out_base):
                cand = os.path.join(out_base, "output_SPHARM")
                if os.path.isdir(cand):
                    target_base = cand
                    self.p.spharm_dir_input.setText(cand)

        self.all_files = []
        if target_base and os.path.isdir(target_base):
            search_dirs = [
                (os.path.join(target_base, "left", "spharm_results"), "lh", "Left (LH)"),
                (os.path.join(target_base, "right", "spharm_results"), "rh", "Right (RH)"),
                (os.path.join(target_base, "spharm_results"), "all", "SPHARM"),
                (os.path.join(target_base, "output_left_hippocampus", "spharm_results_left"), "lh", "Left (LH)"),
                (os.path.join(target_base, "output_right_hippocampus", "spharm_results_right"), "rh", "Right (RH)"),
                (os.path.join(target_base, "split_data", "ALL_Left"), "lh", "Left (LH)"),
                (os.path.join(target_base, "split_data", "ALL_Right"), "rh", "Right (RH)"),
                (os.path.join(target_base, "left"), "lh", "Left (LH)"),
                (os.path.join(target_base, "right"), "rh", "Right (RH)"),
                (target_base, "all", "SPHARM"),
            ]

            seen = set()
            for directory, side_key, side_label in search_dirs:
                if os.path.isdir(directory):
                    all_vtk = glob.glob(os.path.join(directory, "*.vtk"))
                    subject_map = {}
                    for vtk_file in all_vtk:
                        bn = os.path.basename(vtk_file)
                        bn_lower = bn.lower()
                        if "mean_shape" in bn_lower or bn_lower.startswith("template_"):
                            continue
                        if any(aux in bn_lower for aux in ("_para.", "_surf.", "medialaxis", "_grid.")):
                            continue

                        s_key = bn
                        for suf in ("_SPHARM_realigned.vtk", "_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk", ".vtk"):
                            if s_key.endswith(suf):
                                s_key = s_key[:-len(suf)]
                                break

                        if s_key not in subject_map:
                            subject_map[s_key] = []
                        subject_map[s_key].append(os.path.normpath(vtk_file))

                    for s_key, f_list in sorted(subject_map.items()):
                        # Priority: _realigned.vtk > _procalign.vtk > _ellalign.vtk > _SPHARM.vtk
                        best_mesh = None
                        for suf in ("_SPHARM_realigned.vtk", "_SPHARM_procalign.vtk", "_SPHARM_ellalign.vtk", "_SPHARM.vtk"):
                            cand = [f for f in f_list if f.endswith(suf)]
                            if cand:
                                best_mesh = cand[0]
                                break
                        if not best_mesh and f_list:
                            best_mesh = f_list[0]

                        norm_p = best_mesh
                        if norm_p not in seen:
                            seen.add(norm_p)
                            basename = os.path.basename(norm_p)

                            cur_key = side_key
                            cur_label = side_label
                            norm_lower = norm_p.lower()
                            if cur_key == "all":
                                if basename.startswith("lh_") or "left" in norm_lower or "_lh" in norm_lower or "\\left\\" in norm_lower or "/left/" in norm_lower:
                                    cur_key, cur_label = "lh", "Left (LH)"
                                elif basename.startswith("rh_") or "right" in norm_lower or "_rh" in norm_lower or "\\right\\" in norm_lower or "/right/" in norm_lower:
                                    cur_key, cur_label = "rh", "Right (RH)"

                            self.all_files.append({
                                "filename": basename,
                                "side": cur_label,
                                "side_key": cur_key,
                                "filepath": norm_p
                            })

        # Update tab counts
        all_count = len(self.all_files)
        lh_count = sum(1 for f in self.all_files if f["side_key"] == "lh")
        rh_count = sum(1 for f in self.all_files if f["side_key"] == "rh")
        self.p.tab_bar.setTabText(0, f"All ({all_count})")
        self.p.tab_bar.setTabText(1, f"Left ({lh_count})")
        self.p.tab_bar.setTabText(2, f"Right ({rh_count})")

        self.update_table_display()
        if self.p.overlay_cb.isChecked():
            self.emit_overlay_meshes()

    def get_current_display_files(self):
        if self.current_side_filter == "lh":
            return [f for f in self.all_files if f["side_key"] == "lh"]
        elif self.current_side_filter == "rh":
            return [f for f in self.all_files if f["side_key"] == "rh"]
        return self.all_files

    def update_table_display(self):
        display_files = self.get_current_display_files()

        self.p.results_table.blockSignals(True)
        self.p.results_table.setRowCount(len(display_files))
        for i, item in enumerate(display_files):
            name_item = QTableWidgetItem(item["filename"])
            name_item.setData(Qt.ItemDataRole.UserRole, item["filepath"])
            name_item.setData(Qt.ItemDataRole.UserRole + 1, item["side_key"])
            self.p.results_table.setItem(i, 0, name_item)

            side_item = QTableWidgetItem(item["side"])
            side_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if item["side_key"] == "lh":
                side_item.setForeground(Qt.GlobalColor.blue)
            elif item["side_key"] == "rh":
                side_item.setForeground(Qt.GlobalColor.darkYellow)
            self.p.results_table.setItem(i, 1, side_item)

            path_item = QTableWidgetItem(item["filepath"])
            path_item.setToolTip(item["filepath"])
            self.p.results_table.setItem(i, 2, path_item)

        self.p.results_table.blockSignals(False)
        self.p.results_table.clearSelection()

    def on_tab_changed(self, index):
        if index == 1:
            self.current_side_filter = "lh"
        elif index == 2:
            self.current_side_filter = "rh"
        else:
            self.current_side_filter = "all"
        self.p.signal_side_changed.emit(self.current_side_filter)
        self.update_table_display()
        if self.p.overlay_cb.isChecked():
            self.emit_overlay_meshes()

    def emit_overlay_meshes(self):
        display_files = self.get_current_display_files()
        filepaths = [f["filepath"] for f in display_files]
        self.p.signal_overlay_all_toggled.emit(True, filepaths, self.current_side_filter)

    def on_mesh_selected(self):
        selected_rows = sorted(list(set(index.row() for index in self.p.results_table.selectedIndexes())))
        if not selected_rows:
            self.p.signal_mesh_selected.emit("", self.current_side_filter)
            return

        if len(selected_rows) == 1:
            row = selected_rows[0]
            name_item = self.p.results_table.item(row, 0)
            if name_item:
                filepath = name_item.data(Qt.ItemDataRole.UserRole)
                side_key = name_item.data(Qt.ItemDataRole.UserRole + 1) or self.current_side_filter
                if filepath:
                    self.p.signal_mesh_selected.emit(filepath, side_key)
            return

        filepaths = []
        for row in selected_rows:
            name_item = self.p.results_table.item(row, 0)
            if name_item:
                fp = name_item.data(Qt.ItemDataRole.UserRole)
                if fp:
                    filepaths.append(fp)

        if filepaths:
            self.p.signal_mesh_selected.emit(filepaths, self.current_side_filter)
