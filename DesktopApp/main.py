import sys
import os
import time
import multiprocessing
import warnings

# Suppress harmless scikit-learn unpickle version and feature name warnings globally
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# CLI Runner Dispatch: Allows the frozen .exe to execute Python sub-scripts
# without requiring Python to be installed on the host machine.
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.py') or sys.argv[1] == "-m"):
        if sys.argv[1] == "-m" and len(sys.argv) > 2:
            module_name = sys.argv[2]
            sys.argv = [sys.argv[2]] + sys.argv[3:]
            import runpy
            runpy.run_module(module_name, run_name="__main__")
            sys.exit(0)
        elif sys.argv[1].endswith('.py'):
            script_path = os.path.abspath(sys.argv[1])
            sys.argv = sys.argv[1:]
            script_dir = os.path.dirname(script_path)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            import runpy
            runpy.run_path(script_path, run_name="__main__")
            sys.exit(0)

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QColor, QFont, QPainter, QPen
from PyQt6.QtCore import Qt, QLocale

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set default locale to C/English to force standard Arabic numerals (1000, 10, 12)
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
    
    # ----------------------------------------------------
    # Splash Screen (Slicer/SlicerSALT style startup)
    # ----------------------------------------------------
    pixmap = QPixmap(600, 350)
    pixmap.fill(QColor("#2c3e50")) # Dark blue Slicer-style background
    
    # Draw title directly on the splash image
    painter = QPainter(pixmap)
    painter.setPen(QPen(QColor("white")))
    painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Shape Analysis Toolbox\nInitializing...")
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setFont(QFont("Arial", 11, QFont.Weight.Bold))
    splash.show()
    app.processEvents()
    
    # 1. Real Loading: VTK Rendering Engine
    splash.showMessage("Loading VTK Rendering Engine...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
    
    # 2. Real Loading: System Requirements & SlicerSALT Verification
    splash.showMessage("Checking System Requirements & SlicerSALT...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    import glob
    import torch
    accel_text = "GPU / CUDA" if torch.cuda.is_available() else "CPU"
    
    # 3. Real Loading: Pre-load Deep Learning Models (ResNet1D & PLS-DA)
    splash.showMessage(f"Loading Deep Learning Predictor ({accel_text})...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    try:
        from models.predictor import HippocampalPredictor
        predictor = HippocampalPredictor()
        if predictor.is_model_available("left"):
            splash.showMessage(f"Loading Left Hippocampal Model ({accel_text})...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
            app.processEvents()
            predictor.load_model("left")
        if predictor.is_model_available("right"):
            splash.showMessage(f"Loading Right Hippocampal Model ({accel_text})...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
            app.processEvents()
            predictor.load_model("right")
    except Exception as e:
        print(f"[Warning] Model pre-loading: {e}")
    
    # 4. Real Loading: User Interface & 3D Viewport Modules
    splash.showMessage("Initializing User Interface Modules...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, Qt.GlobalColor.white)
    app.processEvents()
    from ui_main_window import MainWindow
    window = MainWindow()
    
    # Close splash and show main window maximized
    splash.finish(window)
    window.showMaximized()
    
    sys.exit(app.exec())
