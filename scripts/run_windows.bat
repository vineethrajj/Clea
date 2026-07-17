@echo off
REM Starts the Clea web UI and opens it in the default browser.
REM The console window must stay open while you use the app.
cd /d "%~dp0.."

if not exist .venv (
    echo Run scripts\setup_windows.bat first.
    pause & exit /b 1
)
call .venv\Scripts\activate.bat
start "" http://localhost:8000
python -m clea serve --port 8000
pause
