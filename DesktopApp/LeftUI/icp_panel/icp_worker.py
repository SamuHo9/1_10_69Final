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
    for cand in local_candidates:
        if os.path.isfile(cand):
            return cand
            
    candidates = glob.glob(r"C:\Program Files\SlicerSALT*\SlicerSALT.exe")
    if candidates:
        return candidates[0]
    return "C:\\Program Files\\SlicerSALT 6.0.0\\SlicerSALT.exe"

class ICPWorker(QThread):
    signal_log = pyqtSignal(str)
    signal_finished = pyqtSignal(bool)

    def __init__(self, tasks, adv_params, parent=None):
        super().__init__(parent)
        self.tasks = tasks  # list of tuples: (side_name, input_dir, output_dir)
        self.adv_params = adv_params

    def run(self):
        project_root = get_project_root()
        icp_script = os.path.join(project_root, "ICP", "ICP.py")

        if not os.path.isfile(icp_script):
            self.signal_log.emit(f"[ERROR] ICP.py not found at: {icp_script}")
            self.signal_finished.emit(False)
            return

        overall_success = True
        for side_name, in_dir, out_dir in self.tasks:
            self.signal_log.emit(f"\n==================================================")
            self.signal_log.emit(f">>> Running Batch ICP for [{side_name.upper()} Hippocampus]")
            self.signal_log.emit(f"    Input:  {in_dir}")
            self.signal_log.emit(f"    Output: {out_dir}")
            self.signal_log.emit(f"==================================================")
            
            os.makedirs(out_dir, exist_ok=True)
            
            cmd = [
                sys.executable,
                icp_script,
                "--input_dir", in_dir,
                "--output_dir", out_dir,
                "--output_spacing", str(self.adv_params.get("spacing", 0.02)),
                "--output_voxels", str(self.adv_params.get("voxels", 128)),
                "--max_iterations", str(self.adv_params.get("max_iter", 20)),
                "--tolerance", str(self.adv_params.get("tol", 0.00005)),
                "--pairwise_max_iterations", str(self.adv_params.get("pw_iter", 100)),
                "--pairwise_tolerance", str(self.adv_params.get("pw_tol", 0.0001)),
                "--pairwise_landmarks", str(self.adv_params.get("pw_landmarks", 200)),
                "--interp_type", str(self.adv_params.get("interp", "NearestNeighbor"))
            ]

            # Check for standard reference template in Templates/ICP
            tmpl_cand = os.path.join(project_root, "Templates", "ICP", f"template_mean_{side_name.lower()}.vtk")
            if not os.path.isfile(tmpl_cand):
                tmpl_cand = os.path.join(project_root, "Templates", "ICP", f"template_mean_{side_name.lower()}.ply")
            if os.path.isfile(tmpl_cand):
                cmd.extend(["--reference_template", tmpl_cand])
                self.signal_log.emit(f"    Using Reference Template: {os.path.basename(tmpl_cand)}")

            try:
                kwargs = {}
                if os.name == 'nt':
                    kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                    
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    **kwargs
                )
                
                for line in process.stdout:
                    clean = line.strip()
                    if clean:
                        self.signal_log.emit(clean)
                process.wait()
                
                if process.returncode != 0:
                    self.signal_log.emit(f"[ERROR] ICP process failed for {side_name} with return code {process.returncode}")
                    overall_success = False
                else:
                    self.signal_log.emit(f"[OK] ICP completed successfully for {side_name}.")
            except Exception as e:
                self.signal_log.emit(f"[ERROR] Exception running ICP {side_name}: {str(e)}")
                overall_success = False

        self.signal_finished.emit(overall_success)

IcpWorker = ICPWorker
