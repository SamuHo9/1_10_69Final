import os
import sys
import glob
import subprocess
from PyQt6.QtCore import pyqtSignal, QThread

def get_project_root():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def find_slicer_salt_exe():
    env_path = os.environ.get("SLICER_EXE")
    if env_path and os.path.isfile(env_path):
        return env_path
    
    root = get_project_root()
    local_candidates = [
        os.path.join(root, "SlicerSALT", "SlicerSALT.exe"),
        os.path.join(root, "Prerequisites", "SlicerSALT", "SlicerSALT.exe"),
        os.path.join(root, "Prerequisites", "SlicerSALT 6.0.0", "SlicerSALT.exe"),
    ]
    for c in local_candidates:
        if os.path.isfile(c):
            return c

    candidates = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
    if candidates:
        return candidates[0]

    candidates_x86 = glob.glob(r"C:\Program Files (x86)\SlicerSALT*\SlicerSALT.exe")
    if candidates_x86:
        return candidates_x86[0]

    for user_dir in glob.glob(r"C:\Users\*"):
        user_cands = [
            os.path.join(user_dir, "AppData", "Local", "NA-MIC", "SlicerSALT 6.0.0", "SlicerSALT.exe"),
            os.path.join(user_dir, "AppData", "Local", "Programs", "SlicerSALT 6.0.0", "SlicerSALT.exe"),
        ]
        for uc in user_cands:
            if os.path.isfile(uc):
                return uc

    default_salt = r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe"
    if os.path.isfile(default_salt):
        return default_salt

    return None

class SPHARMWorker(QThread):
    signal_log = pyqtSignal(str)
    signal_finished = pyqtSignal(bool)

    def __init__(self, tasks, adv_params=None):
        super().__init__()
        self.tasks = tasks # list of (side_name, in_dir, out_dir)
        self.adv_params = adv_params or {}

    def run(self):
        slicer_exe = find_slicer_salt_exe()
        if not slicer_exe:
            self.signal_log.emit("[ERROR] SlicerSALT.exe not found! Please check installation.")
            self.signal_finished.emit(False)
            return

        project_root = get_project_root()
        batch_script = os.path.join(project_root, "SPHARM", "run_spharm_batch.py")
        realign_script = os.path.join(project_root, "SPHARM", "realign_spharm.py")

        if not os.path.isfile(batch_script):
            self.signal_log.emit(f"[ERROR] run_spharm_batch.py not found at: {batch_script}")
            self.signal_finished.emit(False)
            return

        overall_success = True
        for side_name, in_dir, out_dir in self.tasks:
            self.signal_log.emit(f"\n==================================================")
            self.signal_log.emit(f">>> Running Batch SPHARM for [{side_name.upper()} Hippocampus]")
            
            os.makedirs(out_dir, exist_ok=True)
            
            # Step 1: Batch SPHARM via SlicerSALT
            cmd = [
                slicer_exe,
                "--no-main-window",
                "--no-splash",
                "--python-script", batch_script,
                "--input_dir", in_dir,
                "--output_dir", out_dir,
                "--num_iterations", str(self.adv_params.get("num_iter", 1000)),
                "--subdiv_level", str(self.adv_params.get("subdiv", 10)),
                "--spharm_degree", str(self.adv_params.get("degree", self.adv_params.get("deg", 12)))
            ]
            
            # Use official template from Templates/SPHARM if present
            tmpl_file = os.path.join(project_root, "Templates", "SPHARM", f"template_spharm_{side_name.lower()}.vtk")
            coef_file = os.path.join(project_root, "Templates", "SPHARM", f"template_spharm_{side_name.lower()}.coef")
            if os.path.isfile(tmpl_file) and os.path.isfile(coef_file):
                cmd.extend(["--reference_template", tmpl_file])
                self.signal_log.emit(f"    Using Reference Template: {os.path.basename(tmpl_file)} and {os.path.basename(coef_file)}")
            else:
                self.signal_log.emit(f"    [INFO] Official template for {side_name} not found or incomplete (need both .vtk and .coef). Proceeding without --reference_template.")

            self.signal_log.emit(f"Executing: {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                for line in proc.stdout:
                    self.signal_log.emit(line.strip())
                proc.wait()
                
                if proc.returncode != 0:
                    self.signal_log.emit(f"[ERROR] SPHARM batch failed for {side_name} with return code {proc.returncode}")
                    overall_success = False
                    continue
                else:
                    self.signal_log.emit(f"[SUCCESS] SPHARM processing completed for {side_name}.")
                    
                # Step 2: Post-process Procrustes Re-alignment (Self-alignment against cohort mean)
                if os.path.isfile(realign_script):
                    self.signal_log.emit(f">>> Running Procrustes Re-alignment for [{side_name.upper()}]...")
                    realign_target = os.path.join(out_dir, "spharm_results") if os.path.isdir(os.path.join(out_dir, "spharm_results")) else out_dir
                    realign_cmd = [
                        sys.executable,
                        realign_script,
                        "--spharm_dir", realign_target,
                        "--tolerance", str(self.adv_params.get("tol", 0.0001)),
                        "--max_iterations", str(self.adv_params.get("max_iter", 50))
                    ]
                    # Pass official template as fixed target if available
                    if os.path.isfile(tmpl_file):
                        realign_cmd.extend(["--target_template", tmpl_file])
                        self.signal_log.emit(f"    Aligning cohort to official template: {os.path.basename(tmpl_file)}")
                        
                    re_proc = subprocess.Popen(
                        realign_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    for line in re_proc.stdout:
                        self.signal_log.emit(line.strip())
                    re_proc.wait()
                    self.signal_log.emit(f"[SUCCESS] Procrustes Re-alignment finished for {side_name}.")
                    
            except Exception as e:
                self.signal_log.emit(f"[ERROR] Exception running SPHARM {side_name}: {str(e)}")
                overall_success = False

        self.signal_finished.emit(overall_success)

SpharmWorker = SPHARMWorker
