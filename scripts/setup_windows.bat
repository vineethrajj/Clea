@echo off
REM One-time setup: creates a virtualenv, installs dependencies, runs the
REM hardware check. Requires Python 3.10+ and an NVENC-capable ffmpeg on PATH
REM (https://www.gyan.dev/ffmpeg/builds/ - the "full" build).
cd /d "%~dp0.."

where python >nul 2>nul || (echo Python not found on PATH & pause & exit /b 1)

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || (pause & exit /b 1)
)
call .venv\Scripts\activate.bat
echo Installing dependencies (first run takes a few minutes)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || (pause & exit /b 1)

echo.
python -m clea doctor
echo.
echo Setup complete. Start the app with scripts\run_windows.bat
pause
