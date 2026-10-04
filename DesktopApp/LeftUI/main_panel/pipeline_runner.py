import os
import glob
import shutil

class PipelineRunner:
    """
    Coordinates sequential execution of the 4 pipeline stages:
    1. FastSurfer Hippocampal Segmentation (using 'data import' repository)
    2. Groupwise ICP Mesh Registration
    3. SPHARM-PDM Surface Parameterization
    4. ResNet Epilepsy Prediction & 3D Grad-CAM
    """
    def __init__(self, panel):
        self.p = panel

    def run_full_pipeline(self):
        in_dir = self.p.folder_input.text().strip()
        out_dir = self.p.out_folder_input.text().strip()

        if not out_dir:
            self.p.signal_log_message.emit("[ERROR] Main Panel: Please select an Output Directory first.")
            self.p.status_lbl.setText("Error: Output directory not specified.")
            self.p.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
            return

        # Create structured output folders exactly as each panel expects
        fs_dir = os.path.join(out_dir, "fastsurfer")
        icp_dir = os.path.join(out_dir, "output_ICP")
        spharm_dir = os.path.join(out_dir, "output_SPHARM")
        res_dir = os.path.join(out_dir, "output_Result")

        os.makedirs(fs_dir, exist_ok=True)
        os.makedirs(icp_dir, exist_ok=True)
        os.makedirs(spharm_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        self.p.sync_panel_output_folders(out_dir)

        # -------------------------------------------------------------
        # Continuous Execution: Ensure MRI scans are in 'data import'
        # -------------------------------------------------------------
        data_import_dir = None
        for cand in ["data import", "data_import"]:
            cand_p = os.path.join(out_dir, cand)
            if os.path.isdir(cand_p):
                mris = [f for f in os.listdir(cand_p) if f.lower().endswith((".nii.gz", ".nii", ".mgz", ".nrrd"))]
                if mris:
                    data_import_dir = cand_p
                    break

        if not data_import_dir and in_dir and os.path.isdir(in_dir):
            target_import = os.path.join(out_dir, "data import")
            os.makedirs(target_import, exist_ok=True)
            self.p.signal_log_message.emit(f">>> [MAIN PIPELINE] Preparing 'data import' repository in: {target_import}")
            
            raw_mris = []
            for ext in (".nii.gz", ".nii", ".mgz", ".nrrd"):
                raw_mris.extend(glob.glob(os.path.join(in_dir, f"*{ext}")))
                raw_mris.extend(glob.glob(os.path.join(in_dir, "**", f"*{ext}"), recursive=True))
            
            skip_kw = ["mask", "seg", "aseg", "aparc", "label", "hippo", ".vtk", ".stl", ".ply"]
            copied_count = 0
            for src in raw_mris:
                fname = os.path.basename(src)
                if any(kw in fname.lower() for kw in skip_kw):
                    continue
                dst = os.path.join(target_import, fname)
                if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                    try:
                        shutil.copy2(src, dst)
                        copied_count += 1
                    except Exception as e:
                        self.p.signal_log_message.emit(f"  [WARNING] Could not copy {fname}: {e}")

            data_import_dir = target_import
            self.p.signal_log_message.emit(f">>> [MAIN PIPELINE] Organized {copied_count} MRI scan(s) into 'data import' repository.")

        if data_import_dir:
            self.p.signal_log_message.emit(f">>> [MAIN PIPELINE] FastSurfer stage will use MRI scans from 'data import': {data_import_dir}")
            effective_mri_dir = data_import_dir
        else:
            effective_mri_dir = in_dir

        # Sync directories with ImportPanel
        if self.p.import_panel:
            if effective_mri_dir and os.path.isdir(effective_mri_dir):
                self.p.import_panel.folder_input.setText(effective_mri_dir)
                self.p.import_panel.load_subjects_from_output(out_dir)
            self.p.import_panel.out_folder_input.setText(out_dir)
            self.p.import_panel.signal_directories_changed.emit(effective_mri_dir, out_dir)

        # Check existing results in output dir (all 4 stages)
        has_fs, has_icp, has_spharm, has_result = self.p.check_existing_stages(out_dir)
        force_rerun = hasattr(self.p, 'force_rerun_cb') and self.p.force_rerun_cb.isChecked()

        if force_rerun:
            has_fs = has_icp = has_spharm = has_result = False
            self.p.signal_log_message.emit(">>> [MAIN PIPELINE] Force Re-run enabled: Overwriting existing results, starting all stages from scratch...")

        if has_fs and has_icp and has_spharm and has_result:
            self.p.stage1_lbl.setText("  1. FastSurfer Hippocampal Segmentation:  Found existing results (Skipped)")
            self.p.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Found existing results (Skipped)")
            self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Found existing results (Skipped)")
            self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.p.stage4_lbl.setText("  4. ResNet Epilepsy Prediction & 3D Grad-CAM: Found existing results (Skipped)")
            self.p.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

            self.p.status_lbl.setText("All results already exist in output folder! Skipped execution and refreshed all tables.")
            self.p.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px; font-weight: bold;")
            self.p.signal_log_message.emit(">>> [MAIN PIPELINE] Complete results for all panels (FastSurfer, ICP, SPHARM, Result) already exist in output folder. Skipping execution and populating all tables.")

            if self.p.fastsurfer_panel:
                self.p.fastsurfer_panel.populate_results_table()
            if self.p.icp_panel:
                self.p.icp_panel.populate_results_table()
            if self.p.spharm_panel:
                self.p.spharm_panel.populate_results_table()
            if self.p.result_panel and hasattr(self.p.result_panel, 'load_existing_results'):
                self.p.result_panel.load_existing_results()

            self.p.populate_main_table()
            return

        # Start execution
        self.p.is_running = True
        self.p.run_btn.setEnabled(False)
        self.p.run_btn.setText("⏳ Pipeline Running...")

        # If FastSurfer already exists, jump straight to ICP / SPHARM / Result
        if has_fs:
            self.p.stage1_lbl.setText("  1. FastSurfer Hippocampal Segmentation:  Found existing results (Skipped)")
            self.p.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.p.fastsurfer_panel:
                self.p.fastsurfer_panel.populate_results_table()

            if has_icp and has_spharm:
                self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Found existing results (Skipped)")
                self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Found existing results (Skipped)")
                self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                if self.p.icp_panel:
                    self.p.icp_panel.populate_results_table()
                if self.p.spharm_panel:
                    self.p.spharm_panel.populate_results_table()
                self.run_result_step(out_dir)
                return
            elif has_icp and not has_spharm:
                self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Found existing results (Skipped)")
                self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                if self.p.icp_panel:
                    self.p.icp_panel.populate_results_table()
                self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            In Progress...")
                self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
                self.p.status_lbl.setText("Step 3/4: Running SPHARM-PDM Processing (ICP skipped, results exist)...")
                self.p.signal_log_message.emit(">>> [MAIN PIPELINE] ICP results already exist. Starting SPHARM-PDM Processing...")
                if self.p.spharm_panel:
                    self.p.spharm_panel.update_run_button_state()
                    self.p.spharm_panel.run_spharm_process()
                else:
                    self.reset_run_state()
                return
            else:
                self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         In Progress...")
                self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
                self.p.status_lbl.setText("Step 2/4: Running Groupwise ICP Registration (FastSurfer skipped, results exist)...")
                self.p.signal_log_message.emit(">>> [MAIN PIPELINE] FastSurfer results already exist. Starting Groupwise ICP Registration...")
                if self.p.icp_panel:
                    self.p.icp_panel.update_run_button_state()
                    self.p.icp_panel.run_icp_process()
                else:
                    self.reset_run_state()
                return

        # Standard start from Step 1
        self.p.stage1_lbl.setText("  1. FastSurfer Hippocampal Segmentation:  In Progress...")
        self.p.stage1_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.p.status_lbl.setText("Step 1/4: Running FastSurfer Hippocampal Segmentation...")
        self.p.signal_log_message.emit(">>> [MAIN PIPELINE] Starting FastSurfer Hippocampal Segmentation...")

        if self.p.fastsurfer_panel:
            if hasattr(self.p.fastsurfer_panel, 'update_run_button_state'):
                self.p.fastsurfer_panel.update_run_button_state()
            if hasattr(self.p.fastsurfer_panel, 'run_fastsurfer_process'):
                self.p.fastsurfer_panel.run_fastsurfer_process()
            elif hasattr(self.p.fastsurfer_panel, 'run_process'):
                self.p.fastsurfer_panel.run_process()
        else:
            self.p.signal_log_message.emit("[ERROR] FastSurfer panel not configured.")
            self.reset_run_state()

    def on_fastsurfer_finished_step(self, success):
        if not self.p.is_running:
            return

        if not success:
            self.p.stage1_lbl.setText("  1. FastSurfer Hippocampal Segmentation:  Failed")
            self.p.stage1_lbl.setStyleSheet("font-size: 11px; color: #c0392b; font-weight: bold;")
            self.p.status_lbl.setText("Pipeline stopped: FastSurfer Segmentation encountered an error.")
            self.p.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
            self.reset_run_state()
            return

        self.p.stage1_lbl.setText("  1. FastSurfer Hippocampal Segmentation:  Completed")
        self.p.stage1_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

        if self.p.fastsurfer_panel:
            self.p.fastsurfer_panel.populate_results_table()
        self.p.populate_main_table()

        out_dir = self.p.out_folder_input.text().strip()
        _, has_icp, has_spharm, has_result = self.p.check_existing_stages(out_dir)
        force_rerun = hasattr(self.p, 'force_rerun_cb') and self.p.force_rerun_cb.isChecked()

        if not force_rerun:
            if has_icp and has_spharm and has_result:
                self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Found existing results (Skipped)")
                self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Found existing results (Skipped)")
                self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                self.p.stage4_lbl.setText("  4. ResNet Epilepsy Prediction & 3D Grad-CAM: Found existing results (Skipped)")
                self.p.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                if self.p.icp_panel:
                    self.p.icp_panel.populate_results_table()
                if self.p.spharm_panel:
                    self.p.spharm_panel.populate_results_table()
                if self.p.result_panel and hasattr(self.p.result_panel, 'load_existing_results'):
                    self.p.result_panel.load_existing_results()
                self.on_result_finished_step(True)
                return
            elif has_icp and has_spharm and not has_result:
                self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Found existing results (Skipped)")
                self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Found existing results (Skipped)")
                self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                if self.p.icp_panel:
                    self.p.icp_panel.populate_results_table()
                if self.p.spharm_panel:
                    self.p.spharm_panel.populate_results_table()
                self.run_result_step(out_dir)
                return
            elif has_icp and not has_spharm:
                self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Found existing results (Skipped)")
                self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                if self.p.icp_panel:
                    self.p.icp_panel.populate_results_table()
                self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            In Progress...")
                self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
                self.p.status_lbl.setText("Step 3/4: Running SPHARM-PDM Processing (ICP skipped, results exist)...")
                self.p.signal_log_message.emit(">>> [MAIN PIPELINE] ICP results already exist. Starting SPHARM-PDM Processing...")
                if self.p.spharm_panel:
                    self.p.spharm_panel.update_run_button_state()
                    self.p.spharm_panel.run_spharm_process()
                else:
                    self.reset_run_state()
                return

        # Step 2: ICP Registration
        self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         In Progress...")
        self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.p.status_lbl.setText("Step 2/4: Running Groupwise ICP Registration...")
        self.p.signal_log_message.emit(">>> [MAIN PIPELINE] FastSurfer step completed. Starting Groupwise ICP Registration...")

        if self.p.icp_panel:
            self.p.icp_panel.update_run_button_state()
            self.p.icp_panel.run_icp_process()
        else:
            self.p.signal_log_message.emit("[ERROR] ICP panel not configured.")
            self.reset_run_state()

    def on_icp_finished_step(self, success):
        if not self.p.is_running:
            return

        if not success:
            self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Failed")
            self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #c0392b; font-weight: bold;")
            self.p.status_lbl.setText("Pipeline stopped: Groupwise ICP Registration encountered an error.")
            self.p.status_lbl.setStyleSheet("color: #c0392b; background-color: #fdedec; border: 1px solid #f5b7b1; padding: 6px 8px; border-radius: 4px;")
            self.reset_run_state()
            return

        self.p.stage2_lbl.setText("  2. Groupwise ICP Mesh Registration:         Completed")
        self.p.stage2_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

        if self.p.icp_panel:
            self.p.icp_panel.populate_results_table()
        self.p.populate_main_table()

        out_dir = self.p.out_folder_input.text().strip()
        _, _, has_spharm, has_result = self.p.check_existing_stages(out_dir)
        force_rerun = hasattr(self.p, 'force_rerun_cb') and self.p.force_rerun_cb.isChecked()

        if not force_rerun:
            if has_spharm and has_result:
                self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Found existing results (Skipped)")
                self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                self.p.stage4_lbl.setText("  4. ResNet Epilepsy Prediction & 3D Grad-CAM: Found existing results (Skipped)")
                self.p.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                if self.p.spharm_panel:
                    self.p.spharm_panel.populate_results_table()
                if self.p.result_panel and hasattr(self.p.result_panel, 'load_existing_results'):
                    self.p.result_panel.load_existing_results()
                self.on_result_finished_step(True)
                return
            elif has_spharm and not has_result:
                self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Found existing results (Skipped)")
                self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
                if self.p.spharm_panel:
                    self.p.spharm_panel.populate_results_table()
                self.run_result_step(out_dir)
                return

        # Step 3: SPHARM Processing
        self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            In Progress...")
        self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.p.status_lbl.setText("Step 3/4: Running SPHARM-PDM Processing...")
        self.p.signal_log_message.emit(">>> [MAIN PIPELINE] ICP step completed. Starting SPHARM-PDM Processing...")

        if self.p.spharm_panel:
            self.p.spharm_panel.update_run_button_state()
            self.p.spharm_panel.run_spharm_process()
        else:
            self.p.signal_log_message.emit("[ERROR] SPHARM panel not configured.")
            self.reset_run_state()

    def on_spharm_finished_step(self, success):
        if not self.p.is_running:
            return

        if not success:
            self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Finished with warnings")
            self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #d35400; font-weight: bold;")
            self.p.status_lbl.setText("SPHARM-PDM finished with warnings or errors.")
            self.p.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
        else:
            self.p.stage3_lbl.setText("  3. SPHARM-PDM Shape Analysis:            Completed")
            self.p.stage3_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")

        out_dir = self.p.out_folder_input.text().strip()
        if self.p.import_panel and out_dir:
            self.p.import_panel.load_subjects_from_output(out_dir)
        if self.p.fastsurfer_panel:
            self.p.fastsurfer_panel.populate_results_table()
        if self.p.icp_panel:
            self.p.icp_panel.populate_results_table()
        if self.p.spharm_panel:
            self.p.spharm_panel.populate_results_table()
        self.p.populate_main_table()

        _, _, _, has_result = self.p.check_existing_stages(out_dir)
        force_rerun = hasattr(self.p, 'force_rerun_cb') and self.p.force_rerun_cb.isChecked()

        if not force_rerun and has_result:
            self.p.stage4_lbl.setText("  4. ResNet Epilepsy Prediction & 3D Grad-CAM: Found existing results (Skipped)")
            self.p.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            if self.p.result_panel and hasattr(self.p.result_panel, 'load_existing_results'):
                self.p.result_panel.load_existing_results()
            self.on_result_finished_step(True)
            return

        self.run_result_step(out_dir)

    def run_result_step(self, out_dir):
        self.p.stage4_lbl.setText("  4. ResNet Epilepsy Prediction & 3D Grad-CAM: In Progress...")
        self.p.stage4_lbl.setStyleSheet("font-size: 11px; color: #2980b9; font-weight: bold;")
        self.p.status_lbl.setText("Step 4/4: Running ResNet Epilepsy Prediction & 3D Grad-CAM...")
        self.p.status_lbl.setStyleSheet("color: #1a5276; background-color: #ebf5fb; border: 1px solid #aed6f1; padding: 6px 8px; border-radius: 4px; font-weight: 500;")
        self.p.signal_log_message.emit(">>> [MAIN PIPELINE] SPHARM step completed. Starting ResNet Epilepsy Prediction & 3D Grad-CAM...")

        if self.p.result_panel and hasattr(self.p.result_panel, 'run_batch_prediction'):
            sph_d = os.path.join(out_dir, "output_SPHARM")
            res_d = os.path.join(out_dir, "output_Result")
            self.p.result_panel.spharm_dir_input.setText(sph_d)
            self.p.result_panel.result_dir_input.setText(res_d)
            self.p.result_panel.run_batch_prediction()
        else:
            self.p.signal_log_message.emit("[WARNING] Result panel not configured or run_batch_prediction unavailable.")
            self.on_result_finished_step(True)

    def on_result_finished_step(self, success):
        if not self.p.is_running:
            return

        if not success:
            self.p.stage4_lbl.setText("  4. ResNet Epilepsy Prediction & 3D Grad-CAM: Finished with warnings")
            self.p.stage4_lbl.setStyleSheet("font-size: 11px; color: #d35400; font-weight: bold;")
            self.p.status_lbl.setText("ResNet Prediction & Grad-CAM finished with warnings or errors.")
            self.p.status_lbl.setStyleSheet("color: #d35400; background-color: #fef9e7; border: 1px solid #f9e79f; padding: 6px 8px; border-radius: 4px;")
        else:
            self.p.stage4_lbl.setText("  4. ResNet Epilepsy Prediction & 3D Grad-CAM: Completed")
            self.p.stage4_lbl.setStyleSheet("font-size: 11px; color: #27ae60; font-weight: bold;")
            self.p.status_lbl.setText("Complete Pipeline (All 4 Stages) Finished Successfully!")
            self.p.status_lbl.setStyleSheet("color: #1e8449; background-color: #eafaf1; border: 1px solid #a9dfbf; padding: 6px 8px; border-radius: 4px; font-weight: bold;")
            self.p.signal_log_message.emit(">>> [MAIN PIPELINE] All 4 pipeline stages (FastSurfer, ICP, SPHARM, Result) completed successfully!")

        out_dir = self.p.out_folder_input.text().strip()
        if self.p.import_panel and out_dir:
            self.p.import_panel.load_subjects_from_output(out_dir)
        if self.p.fastsurfer_panel:
            self.p.fastsurfer_panel.populate_results_table()
        if self.p.icp_panel:
            self.p.icp_panel.populate_results_table()
        if self.p.spharm_panel:
            self.p.spharm_panel.populate_results_table()
        if self.p.result_panel and hasattr(self.p.result_panel, 'load_existing_results'):
            self.p.result_panel.load_existing_results()

        self.p.populate_main_table()
        self.reset_run_state()

    def reset_run_state(self):
        self.p.is_running = False
        self.p.run_btn.setEnabled(True)
        self.p.run_btn.setText("Run Full Pipeline")
