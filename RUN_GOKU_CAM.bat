@echo off
cd /d "%~dp0"
if not exist .venv (
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Python 3.11 was not found. Install Python 3.11 and try again.
    pause
    exit /b 1
  )
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-goku.txt
python goku_instant_cam.py
pause
