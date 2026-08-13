@echo off
REM Build the standalone Windows application.
REM
REM Needs Python 3.9 or newer installed, with the "Add python.exe to PATH" box
REM ticked during installation. Double-click this file, or run it from a command
REM prompt in this folder. It takes a few minutes the first time.

echo Making an isolated environment...
python -m venv .venv || goto :failed
call .venv\Scripts\activate.bat

echo Installing what the program needs...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt || goto :failed
python -m pip install pyinstaller || goto :failed

echo Building. This is the slow part...
pyinstaller --noconfirm hrv_variants.spec || goto :failed

echo.
echo Done. The program is in:  dist\HeartRateVariants\
echo Run HeartRateVariants.exe inside that folder, or make a shortcut to it.
echo The whole folder has to stay together; the data folder beside the program
echo can be edited or replaced.
pause
exit /b 0

:failed
echo.
echo Build failed. The message above says why.
pause
exit /b 1
