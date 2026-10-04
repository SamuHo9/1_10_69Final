@echo off
setlocal enabledelayedexpansion
title Hippocampal Shape Analysis Toolbox - Launcher

cd /d "%~dp0"

echo ============================================================
echo   Hippocampal Shape Analysis Toolbox - Auto Launcher
echo ============================================================
echo.

set "VENV_DIR=%~dp0venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PYTHONW=%VENV_DIR%\Scripts\pythonw.exe"
set "REQ_FILE=%~dp0requirements.txt"
set "APP_ENTRY=%~dp0DesktopApp\main.py"

REM ------------------------------------------------------------
REM 1. Check if virtual environment already exists and is valid
REM ------------------------------------------------------------
if exist "%VENV_PYTHON%" (
    REM Quick test if PyQt6 is present
    "%VENV_PYTHON%" -c "import PyQt6, vtk, torch" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [OK] Environment verified. Starting application...
        if exist "%VENV_PYTHONW%" (
            start "" "%VENV_PYTHONW%" "%APP_ENTRY%"
        ) else (
            start "" "%VENV_PYTHON%" "%APP_ENTRY%"
        )
        exit /b 0
    )
)

REM ------------------------------------------------------------
REM 2. First-time setup: Detect System Python
REM ------------------------------------------------------------
echo [INFO] First-time setup or environment update required.
echo Detecting system Python...

set "SYS_PYTHON="
where python >nul 2>&1
if !errorlevel! equ 0 (
    set "SYS_PYTHON=python"
) else (
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        set "SYS_PYTHON=py"
    )
)

if "%SYS_PYTHON%"=="" (
    echo.
    echo ============================================================
    echo [ERROR] Python is not installed or not found in system PATH.
    echo.
    echo Please install Python 3.10 or 3.11 from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Make sure to check the box:
    echo   "Add python.exe to PATH" during installation.
    echo ============================================================
    echo.
    pause
    exit /b 1
)

echo Using system Python: %SYS_PYTHON%
%SYS_PYTHON% --version

REM ------------------------------------------------------------
REM 3. Create Virtual Environment
REM ------------------------------------------------------------
echo.
echo [1/3] Creating dedicated local environment (venv)...
if not exist "%VENV_DIR%" (
    %SYS_PYTHON% -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM ------------------------------------------------------------
REM 4. Install Dependencies from requirements.txt
REM ------------------------------------------------------------
echo.
echo [2/3] Installing required packages (PyQt6, VTK, PyTorch, Scikit-learn)...
echo (This is only done on first run and may take a couple minutes...)
echo.

"%VENV_PYTHON%" -m pip install --upgrade pip
if exist "%REQ_FILE%" (
    "%VENV_PYTHON%" -m pip install -r "%REQ_FILE%"
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Dependency installation encountered an issue.
        pause
        exit /b 1
    )
) else (
    echo [WARNING] requirements.txt not found.
)

REM ------------------------------------------------------------
REM 5. Launch the Application
REM ------------------------------------------------------------
echo.
echo [3/3] Setup complete! Launching Shape Analysis Toolbox...
if exist "%VENV_PYTHONW%" (
    start "" "%VENV_PYTHONW%" "%APP_ENTRY%"
) else (
    start "" "%VENV_PYTHON%" "%APP_ENTRY%"
)

REM Close launcher window cleanly
exit /b 0
