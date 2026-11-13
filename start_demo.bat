@echo off
TITLE GEO-SENTINEL AI: Landslide Prediction System
COLOR 0B
echo ======================================================================
echo    GEO-SENTINEL AI // AUTONOMOUS LANDSLIDE EARLY WARNING SYSTEM
echo ======================================================================
echo.
echo [1/2] Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not detected in PATH!
    pause
    exit /b 1
)

echo.
echo [2/2] Launching Backend Server and Futuristic Dashboard...
python run_project.py

pause
