@echo off
cd /d "%~dp0"
title Goku Instant Cam

if not exist .venv (
  echo Creating Python 3.11 environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Python 3.11 64-bit was not found. Install it and try again.
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install --disable-pip-version-check --upgrade pip
pip install --disable-pip-version-check -r requirements-goku.txt
if errorlevel 1 (
  echo.
  echo Dependency installation failed. Check the error above.
  pause
  exit /b 1
)
python goku_instant_cam.py
pause
