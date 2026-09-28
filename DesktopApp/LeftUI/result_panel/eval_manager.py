import os
import glob
import json
import shutil
from PyQt6.QtWidgets import (
    QTableWidgetItem, QHeaderView, QApplication, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

class EvaluationManager:
    """
    Manages batch evaluation execution via HippocampalPredictor,
    SPHARM subject discovery, JSON report persistence, and
    diagnostic results table synchronization for ResultPanel.
    """
    def __init__(self, panel):
        self.p = panel
        self.all_evaluation_results = []
        self.current_patient_data = None
        self.last_prediction_results = None

    def _extract_subject_id(self, filename: str) -> str:
        clean = os.path.splitext(filename)[0]
        for p in ("lh_", "rh_", "left_", "right_"):
            if clean.lower().startswith(p):
                clean = clean[len(p):]
                break
        for pat in (
            "_SPHARM_realigned", "_SPHARM_procalign", "_SPHARM_ellalign",
            "_SPHARM_grid", "_SPHARMMedialAxis", "_MedialAxisScalars",
            "_MedialAxis", "_SPHARM", "_realigned", "_procalign",
            "_ellalign", "_surf", "_para", "_aligned"
        ):
            clean = clean.replace(pat, "")
        if clean.endswith("_lh") or clean.endswith("_rh"):
            clean = clean[:-3]
        clean = clean.replace("_hippocampus", "")
        return clean.strip("_")

    def discover_spharm_subjects(self):
        spharm_dir = self.p.spharm_dir_input.text().strip()
        search_dirs = []
        if spharm_dir and os.path.isdir(spharm_dir):
            search_dirs.append(spharm_dir)
            search_dirs.append(os.path.join(spharm_dir, "left"))
            search_dirs.append(os.path.join(spharm_dir, "right"))
            search_dirs.append(os.path.join(spharm_dir, "left", "spharm_results"))
            search_dirs.append(os.path.join(spharm_dir, "right", "spharm_results"))
            search_dirs.append(os.path.join(spharm_dir, "spharm_results"))

        if self.p.get_output_folder:
            out_d = self.p.get_output_folder().strip()
            if out_d and os.path.isdir(out_d) and out_d not in search_dirs:
                search_dirs.extend([
                    out_d,
                    os.path.join(out_d, "output_SPHARM"),
                    os.path.join(out_d, "output_left_hippocampus", "spharm_results_left"),
                    os.path.join(out_d, "output_right_hippocampus", "spharm_results_right")
                ])

        coef_map = {}
        vtk_map = {}

        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for coef_file in glob.glob(os.path.join(d, "**", "*_SPHARM.coef"), recursive=True):
                fname = os.path.basename(coef_file)
                fname_lower = fname.lower()
                if any(aux in fname_lower for aux in ("_para.", "_surf.", "medialaxis", "_grid.", "template_")):
                    continue
                side = "left" if (fname.startswith("lh_") or "left" in fname_lower) else ("right" if (fname.startswith("rh_") or "right" in fname_lower) else "unknown")
                if side not in ("left", "right"):
                    continue
                subj = self._extract_subject_id(fname)
                if not subj:
                    continue
                coef_map.setdefault(subj, {}).setdefault(side, []).append(os.path.normpath(coef_file))

            for vtk_file in glob.glob(os.path.join(d, "**", "*.vtk"), recursive=True):
                fname = os.path.basename(vtk_file)
                fname_lower = fname.lower()
                if any(aux in fname_lower for aux in ("_para.", "_surf.", "medialaxis", "_grid.", "template_", "mean_shape")):
                    continue
                side = "left" if (fname.startswith("lh_") or "left" in fname_lower) else ("right" if (fname.startswith("rh_") or "right" in fname_lower) else "unknown")
                if side not in ("left", "right"):
                    continue
                subj = self._extract_subject_id(fname)
                if not subj:
                    continue
                vtk_map.setdefault(subj, {}).setdefault(side, []).append(os.path.normpath(vtk_file))

        all_subj_keys = sorted(set(list(coef_map.keys()) + list(vtk_map.keys())))
        subjects = {}

        def pick_best_coef(c_list):
            if not c_list:
                return None
            for c in c_list:
                if c.endswith("_SPHARM.coef"):
                    return c
            return None

        def pick_best_vtk(v_list):
            if not v_list:
                return None
            for suf in ("_SPHARM_realigned.vtk", "_realigned.vtk", "_SPHARM_procalign.vtk", "_procalign.vtk", "_SPHARM_ellalign.vtk", "_ellalign.vtk", "_SPHARM.vtk"):
                cand = [v for v in v_list if v.endswith(suf)]
                if cand:
                    return cand[0]
            return v_list[0]

        for subj in all_subj_keys:
            lh_coef = pick_best_coef(coef_map.get(subj, {}).get("left", []))
            rh_coef = pick_best_coef(coef_map.get(subj, {}).get("right", []))
            lh_vtk = pick_best_vtk(vtk_map.get(subj, {}).get("left", []))
            rh_vtk = pick_best_vtk(vtk_map.get(subj, {}).get("right", []))

            if not lh_coef and not rh_coef:
                continue

            subjects[subj] = {
                "left_coef": lh_coef,
                "right_coef": rh_coef,
                "left_vtk": lh_vtk,
                "right_vtk": rh_vtk
            }

        return subjects

    def run_batch_prediction(self):
        subjects = self.discover_spharm_subjects()
        if not subjects:
            self.p.signal_log_message.emit("[WARNING] No valid SPHARM .coef or .vtk files found in the specified directory.")
            self.p.batch_status_hint.setText("❌ No valid SPHARM files found in specified directory.")
            self.p.batch_status_hint.setStyleSheet("color: #c0392b; font-size: 11px;")
            self.p.signal_batch_prediction_finished.emit(False)
            return

        out_dir = self.p.result_dir_input.text().strip() or self.p.get_default_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        meshes_dir = os.path.join(out_dir, "meshes")
        os.makedirs(meshes_dir, exist_ok=True)

        eval_target = self.p.side_eval_group.checkedId()  # 0: both, 1: left, 2: right
        total_subjs = len(subjects)
        self.p.signal_log_message.emit(f">>> Running ResNet batch prediction for {total_subjs} subjects...")

        self.p.predict_btn.setEnabled(False)
        self.p.batch_prog_bar.setVisible(True)
        self.p.batch_prog_bar.setValue(0)
        self.p.batch_status_hint.setText(f"Evaluating {total_subjs} subjects across hippocampal meshes...")
        self.p.batch_status_hint.setStyleSheet("color: #2980b9; font-size: 11px; font-weight: bold;")

        evaluated_results = []

        for idx, (subj_name, data) in enumerate(sorted(subjects.items())):
            lh_coef = data.get("left_coef")
            rh_coef = data.get("right_coef")
            lh_vtk = data.get("left_vtk")
            rh_vtk = data.get("right_vtk")

            left_res = None
            right_res = None

            if eval_target in (0, 1) and lh_coef and os.path.isfile(lh_coef):
                try:
                    left_res = self.p.predictor.predict(lh_coef, side='left')
                except Exception as e:
                    self.p.signal_log_message.emit(f"[WARNING] Left predict failed for {subj_name}: {e}")

            if eval_target in (0, 2) and rh_coef and os.path.isfile(rh_coef):
                try:
                    right_res = self.p.predictor.predict(rh_coef, side='right')
                except Exception as e:
                    self.p.signal_log_message.emit(f"[WARNING] Right predict failed for {subj_name}: {e}")

            if left_res and right_res:
                avg_prob = (left_res['probability'] + right_res['probability']) / 2.0
                is_epilepsy = avg_prob > 0.5
                higher_side = 'Left' if left_res['probability'] >= right_res['probability'] else 'Right'
                summary = {
                    'probability': avg_prob,
                    'is_epilepsy': is_epilepsy,
                    'label': "Temporal Lobe Epilepsy (TLE)" if is_epilepsy else "Healthy Control (HC)",
                    'primary_side': higher_side,
                    'left_prob': left_res['probability'],
                    'right_prob': right_res['probability']
                }
            elif left_res:
                prob = left_res['probability']
                summary = {
                    'probability': prob,
                    'is_epilepsy': prob > 0.5,
                    'label': "Temporal Lobe Epilepsy (TLE)" if prob > 0.5 else "Healthy Control (HC)",
                    'primary_side': "Left",
                    'left_prob': prob,
                    'right_prob': None
                }
            elif right_res:
                prob = right_res['probability']
                summary = {
                    'probability': prob,
                    'is_epilepsy': prob > 0.5,
                    'label': "Temporal Lobe Epilepsy (TLE)" if prob > 0.5 else "Healthy Control (HC)",
                    'primary_side': "Right",
                    'left_prob': None,
                    'right_prob': prob
                }
            else:
                continue

            prob = summary['probability']
            risk_level = "High Risk (Epilepsy-aligned)" if prob > 0.75 else ("Moderate Risk" if summary['is_epilepsy'] else "Low Risk (Healthy-aligned)")
            summary['risk_level'] = risk_level

            out_lh_vtk = None
            out_rh_vtk = None

            if left_res and lh_coef and os.path.isfile(lh_coef):
                lh_milestone = "minus3SD" if left_res.get('probability', 0) > 0.75 else ("minus2SD" if left_res.get('probability', 0) > 0.5 else "Mean")
                src_lh_gradcam = self.p.predictor.get_gradcam_mesh_path(side="left", component="PLS1", milestone=lh_milestone, cohort="All_Augment_tain")
                if src_lh_gradcam and os.path.isfile(src_lh_gradcam):
                    out_lh_vtk = os.path.join(meshes_dir, f"{subj_name}_lh_gradcam.vtk")
                    try:
                        if not os.path.exists(out_lh_vtk) or os.path.getmtime(src_lh_gradcam) > os.path.getmtime(out_lh_vtk):
                            shutil.copy2(src_lh_gradcam, out_lh_vtk)
                    except Exception:
                        out_lh_vtk = src_lh_gradcam

            if right_res and rh_coef and os.path.isfile(rh_coef):
                rh_milestone = "minus3SD" if right_res.get('probability', 0) > 0.75 else ("minus2SD" if right_res.get('probability', 0) > 0.5 else "Mean")
                src_rh_gradcam = self.p.predictor.get_gradcam_mesh_path(side="right", component="PLS1", milestone=rh_milestone, cohort="All_Augment_tain")
                if src_rh_gradcam and os.path.isfile(src_rh_gradcam):
                    out_rh_vtk = os.path.join(meshes_dir, f"{subj_name}_rh_gradcam.vtk")
                    try:
                        if not os.path.exists(out_rh_vtk) or os.path.getmtime(src_rh_gradcam) > os.path.getmtime(out_rh_vtk):
                            shutil.copy2(src_rh_gradcam, out_rh_vtk)
                    except Exception:
                        out_rh_vtk = src_rh_gradcam

            item_record = {
                'subject_name': subj_name,
                'left_coef': lh_coef if left_res else None,
                'right_coef': rh_coef if right_res else None,
                'left_vtk': out_lh_vtk if left_res else None,
                'right_vtk': out_rh_vtk if right_res else None,
                'left_gradcam_mesh': out_lh_vtk if left_res else None,
                'right_gradcam_mesh': out_rh_vtk if right_res else None,
                'patient_left_mesh': lh_vtk if left_res else None,
                'patient_right_mesh': rh_vtk if right_res else None,
                'left_result': left_res,
                'right_result': right_res,
                'summary': summary,
                'output_left_mesh': out_lh_vtk if left_res else None,
                'output_right_mesh': out_rh_vtk if right_res else None
            }
            evaluated_results.append(item_record)

            prog_pct = int(((idx + 1) / total_subjs) * 100)
            self.p.batch_prog_bar.setValue(prog_pct)
            QApplication.processEvents()

        # Save evaluation_summary.json in output_Result
        json_file = os.path.join(out_dir, "evaluation_summary.json")
        try:
            json_data = []
            for r in evaluated_results:
                json_data.append({
                    'subject': r['subject_name'],
                    'summary': r['summary'],
                    'left_mesh': r['output_left_mesh'],
                    'right_mesh': r['output_right_mesh']
                })
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            self.p.signal_log_message.emit(f"Saved JSON report: {json_file}")
        except Exception as e:
            self.p.signal_log_message.emit(f"[WARNING] Failed to save JSON report: {e}")

        self.all_evaluation_results = evaluated_results
        self.populate_results_table()
        self.p.predict_btn.setEnabled(True)
        self.p.batch_status_hint.setText(f"Evaluated all {total_subjs} meshes successfully. Saved to: {os.path.basename(out_dir)}")
        self.p.batch_status_hint.setStyleSheet("color: #27ae60; font-size: 11px; font-weight: bold;")
        self.p.signal_log_message.emit(f"SUCCESS: Batch evaluation completed for {total_subjs} subjects. Output folder: {out_dir}")
        self.p.signal_batch_prediction_finished.emit(True)

    def load_existing_results(self):
        out_dir = self.p.result_dir_input.text().strip() or self.p.get_default_output_dir()
        if not os.path.isdir(out_dir):
            self.clear_view()
            self.p.batch_status_hint.setText(f"Folder not found: {os.path.basename(out_dir) if out_dir else 'None'}")
            self.p.batch_status_hint.setStyleSheet("color: #e67e22; font-size: 11px;")
            self.p.signal_log_message.emit(f"[INFO] Output folder '{out_dir}' does not exist. Cleared view.")
            return

        json_file = os.path.join(out_dir, "evaluation_summary.json")

        if os.path.isfile(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                recovered_map = {}
                for item in data:
                    s_raw = item.get('subject', '')
                    s_clean = self._extract_subject_id(s_raw)
                    if not s_clean:
                        continue
                    summary = item.get('summary', {})
                    lh_mesh = item.get('left_mesh')
                    rh_mesh = item.get('right_mesh')

                    if s_clean not in recovered_map or (summary.get('probability', 0.0) > 0 and not recovered_map[s_clean]['summary'].get('probability')):
                        recovered_map[s_clean] = {
                            'subject_name': s_clean,
                            'left_coef': None,
                            'right_coef': None,
                            'left_vtk': lh_mesh,
                            'right_vtk': rh_mesh,
                            'left_result': {'probability': summary.get('left_prob')} if summary.get('left_prob') is not None else None,
                            'right_result': {'probability': summary.get('right_prob')} if summary.get('right_prob') is not None else None,
                            'summary': summary,
                            'output_left_mesh': lh_mesh,
                            'output_right_mesh': rh_mesh
                        }
                    else:
                        existing = recovered_map[s_clean]
                        if not existing['left_vtk'] and lh_mesh:
                            existing['left_vtk'] = lh_mesh
                        if not existing['right_vtk'] and rh_mesh:
                            existing['right_vtk'] = rh_mesh

                recovered = list(recovered_map.values())
                if recovered:
                    self.all_evaluation_results = recovered
                    self.populate_results_table()
                    self.p.batch_status_hint.setText(f"Loaded {len(recovered)} evaluation results from {os.path.basename(out_dir)}")
                    self.p.batch_status_hint.setStyleSheet("color: #27ae60; font-size: 11px;")
                    return
            except Exception as e:
                self.p.signal_log_message.emit(f"[WARNING] Could not parse existing JSON report: {e}")

        self.clear_view()
        self.p.batch_status_hint.setText("No evaluation results found in output directory.")
        self.p.batch_status_hint.setStyleSheet("color: #e67e22; font-size: 11px;")
        self.p.signal_log_message.emit(f"[INFO] No evaluation results found in '{out_dir}'. Cleared view.")

    def on_tab_changed(self, index):
        self.populate_results_table()

    def populate_results_table(self):
        tab_idx = self.p.tab_bar.currentIndex()

        total_lh = 0
        total_rh = 0
        for r in self.all_evaluation_results:
            if r.get('left_result') and (r.get('left_vtk') or r.get('left_gradcam_mesh')):
                total_lh += 1
            if r.get('right_result') and (r.get('right_vtk') or r.get('right_gradcam_mesh')):
                total_rh += 1

        self.p.tab_bar.blockSignals(True)
        self.p.tab_bar.setTabText(0, f"All ({total_lh + total_rh})")
        self.p.tab_bar.setTabText(1, f"Left ({total_lh})")
        self.p.tab_bar.setTabText(2, f"Right ({total_rh})")
        self.p.tab_bar.blockSignals(False)

        filtered = []
        for r in self.all_evaluation_results:
            has_left = bool(r.get('left_result') and (r.get('left_vtk') or r.get('left_gradcam_mesh')))
            has_right = bool(r.get('right_result') and (r.get('right_vtk') or r.get('right_gradcam_mesh')))
            if tab_idx == 1:
                if has_left:
                    filtered.append((r, "left"))
            elif tab_idx == 2:
                if has_right:
                    filtered.append((r, "right"))
            else:
                if has_left:
                    filtered.append((r, "left"))
                if has_right:
                    filtered.append((r, "right"))

        self.p.results_table.blockSignals(True)
        self.p.results_table.setRowCount(0)
        self.p.results_table.setRowCount(len(filtered))

        for row, (record, view_side) in enumerate(filtered):
            subj_name = record['subject_name']
            summary = record.get('summary', {})
            prob = summary.get('probability', 0.0)

            if view_side == "left":
                lh_res = record.get('left_result')
                side_prob = lh_res['probability'] if (lh_res and lh_res.get('probability') is not None) else prob
                diag_str = "TLE" if side_prob > 0.5 else "HC"
                prob_str = f"{side_prob * 100:.1f}%"
                side_str = "LH"
                side_full = "Left (LH)"
                mesh_path = record.get('left_vtk')
            else:
                rh_res = record.get('right_result')
                side_prob = rh_res['probability'] if (rh_res and rh_res.get('probability') is not None) else prob
                diag_str = "TLE" if side_prob > 0.5 else "HC"
                prob_str = f"{side_prob * 100:.1f}%"
                side_str = "RH"
                side_full = "Right (RH)"
                mesh_path = record.get('right_vtk')

            mesh_str = os.path.basename(mesh_path) if mesh_path else "—"

            item_name = QTableWidgetItem(subj_name)
            item_name.setData(Qt.ItemDataRole.UserRole, (record, view_side))
            item_name.setToolTip(f"Subject: {subj_name}\nSide: {side_full}")

            item_side = QTableWidgetItem(side_str)
            item_side.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_side.setToolTip(side_full)
            if side_str == "LH" or view_side == "left":
                item_side.setForeground(QColor("#2980b9"))
            else:
                item_side.setForeground(QColor("#d35400"))
            item_side.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

            item_diag = QTableWidgetItem(diag_str)
            item_diag.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_prob = QTableWidgetItem(prob_str)
            item_prob.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_mesh = QTableWidgetItem(mesh_str)
            item_mesh.setToolTip(mesh_path if mesh_path else "")

            if "TLE" in diag_str:
                item_diag.setForeground(QColor("#c0392b"))
                item_diag.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            else:
                item_diag.setForeground(QColor("#27ae60"))
                item_diag.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

            self.p.results_table.setItem(row, 0, item_name)
            self.p.results_table.setItem(row, 1, item_side)
            self.p.results_table.setItem(row, 2, item_diag)
            self.p.results_table.setItem(row, 3, item_prob)
            self.p.results_table.setItem(row, 4, item_mesh)

        self.p.results_table.blockSignals(False)
        self.p.results_table.resizeColumnsToContents()
        self.p.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        if self.p.results_table.rowCount() > 0:
            if self.p.isVisible():
                self.p.results_table.selectRow(0)
            else:
                self.p.results_table.blockSignals(True)
                self.p.results_table.selectRow(0)
                self.p.results_table.blockSignals(False)

    def on_result_selected(self):
        if not self.p.isVisible():
            return
        selected = self.p.results_table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        item = self.p.results_table.item(row, 0)
        if not item:
            return

        user_data = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(user_data, tuple):
            record, view_side = user_data
        elif isinstance(user_data, dict):
            record, view_side = user_data, "left"
        else:
            return

        self.current_patient_data = record
        self.last_prediction_results = {
            'left': record.get('left_result'),
            'right': record.get('right_result'),
            'summary': record.get('summary')
        }

        self.display_prediction_results(self.last_prediction_results, selected_side=view_side)
        self.p.set_sd_value(0.0)
        self.p.signal_log_message.emit(f"Inspecting evaluation results for: {record['subject_name']} [{view_side.upper()}]")

    def display_prediction_results(self, results, selected_side=None):
        summary = results.get('summary')
        if not summary:
            return

        lh_prob = summary.get('left_prob')
        rh_prob = summary.get('right_prob')

        if selected_side in ("left", "right"):
            if selected_side == "right":
                self.p.cam_controls.rb_cam_right.setChecked(True)
            else:
                self.p.cam_controls.rb_cam_left.setChecked(True)
        elif lh_prob is not None and rh_prob is not None:
            if rh_prob > lh_prob:
                self.p.cam_controls.rb_cam_right.setChecked(True)
            else:
                self.p.cam_controls.rb_cam_left.setChecked(True)
        elif rh_prob is not None:
            self.p.cam_controls.rb_cam_right.setChecked(True)
        elif lh_prob is not None:
            self.p.cam_controls.rb_cam_left.setChecked(True)

    def clear_view(self):
        self.p.signal_clear_gradcam_requested.emit()
        self.all_evaluation_results = []
        self.current_patient_data = None
        self.last_prediction_results = None

        self.p.results_table.blockSignals(True)
        self.p.results_table.setRowCount(0)
        self.p.results_table.blockSignals(False)

        self.p.tab_bar.blockSignals(True)
        self.p.tab_bar.setTabText(0, "All (0)")
        self.p.tab_bar.setTabText(1, "Left (0)")
        self.p.tab_bar.setTabText(2, "Right (0)")
        self.p.tab_bar.blockSignals(False)

        self.p.signal_log_message.emit("Diagnostic panel reset.")

    def copy_summary(self):
        if not self.last_prediction_results or not self.current_patient_data:
            return

        summary = self.last_prediction_results.get('summary', {})
        subj = self.current_patient_data.get('subject_name', 'Patient')
        diag = summary.get('label', 'N/A')
        prob = summary.get('probability', 0.0) * 100.0
        lh = summary.get('left_prob')
        rh = summary.get('right_prob')
        focus = summary.get('primary_side', 'N/A')

        text = (
            f"=== HIPPOCAMPAL SHAPE ANALYSIS - DIAGNOSTIC REPORT ===\n"
            f"Subject ID: {subj}\n"
            f"Model: ResNet1D Residual Neural Network + PLS-DA\n"
            f"Diagnosis: {diag}\n"
            f"Overall Epilepsy Probability: {prob:.2f}%\n"
            + (f"Left Hippocampus Risk: {lh*100:.2f}%\n" if lh is not None else "")
            + (f"Right Hippocampus Risk: {rh*100:.2f}%\n" if rh is not None else "")
            + f"Suspected Seizure Focus: {focus} Hemisphere\n"
            f"Methodology: SPHARM-PDM Point Distribution Parameterization\n"
            f"======================================================\n"
        )
        cb = QApplication.clipboard()
        if cb:
            cb.setText(text)
            self.p.signal_log_message.emit(f"Diagnostic summary copied to clipboard for {subj}.")
            QMessageBox.information(self.p, "Summary Copied", f"Diagnostic summary for '{subj}' has been copied to your clipboard.")
