@echo off
setlocal enabledelayedexpansion
title Build Standalone Executable (PyInstaller)

cd /d "%~dp0"

echo ============================================================
echo   Hippocampal Shape Analysis Toolbox - Exe Builder
echo ============================================================
echo.

set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"
set "BUILD_PYTHON=python"

if exist "%VENV_PYTHON%" (
    set "BUILD_PYTHON=%VENV_PYTHON%"
)

echo Checking PyInstaller...
"%BUILD_PYTHON%" -m pip show pyinstaller >nul 2>&1
if !errorlevel! neq 0 (
    echo [INFO] Installing PyInstaller into environment...
    "%BUILD_PYTHON%" -m pip install pyinstaller
)

echo.
echo [1/2] Packaging Desktop Application into standalone bundle...
echo (This may take 3-5 minutes to bundle PyTorch, VTK, and PyQt6...)
echo.

"%BUILD_PYTHON%" -m PyInstaller --noconfirm --onedir --windowed ^
    --name "HippocampalAnalysisApp" ^
    --add-data "DesktopApp/models;models" ^
    --add-data "Templates;Templates" ^
    --hidden-import "vtkmodules" ^
    --hidden-import "vtkmodules.all" ^
    --hidden-import "sklearn" ^
    --hidden-import "sklearn.cross_decomposition" ^
    --hidden-import "sklearn.preprocessing" ^
    --hidden-import "joblib" ^
    --hidden-import "torch" ^
    --hidden-import "scipy" ^
    --hidden-import "nibabel" ^
    --hidden-import "trimesh" ^
    "DesktopApp/main.py"

if !errorlevel! equ 0 (
    echo.
    echo ============================================================
    echo [SUCCESS] Standalone Application created successfully!
    echo Location: %~dp0dist\HippocampalAnalysisApp\
    echo.
    echo You can now zip or share the 'dist\HippocampalAnalysisApp' folder
    echo with anyone. They can run 'HippocampalAnalysisApp.exe' directly
    echo WITHOUT needing to install Python or any dependencies!
    echo ============================================================
) else (
    echo.
    echo [ERROR] Build failed. Please check the console output above.
)

pause
