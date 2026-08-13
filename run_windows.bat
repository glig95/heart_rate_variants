@echo off
REM Run the program from source on Windows, without building anything.
REM
REM Needs Python 3.9 or newer, installed with "Add python.exe to PATH" ticked.
REM The first run sets things up and takes a minute; later runs start straight
REM away.

if not exist .venv (
    echo First run: making an isolated environment...
    python -m venv .venv || goto :failed
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip >nul
    python -m pip install -r requirements.txt || goto :failed
) else (
    call .venv\Scripts\activate.bat
)

python run_gui.py
exit /b 0

:failed
echo.
echo Setup failed. The message above says why.
echo Check that Python is installed and on the PATH: try  python --version
pause
exit /b 1
