@echo off
TITLE Therapist AI
cd /d "%~dp0"

echo ========================================================
echo                 Starting Therapist AI...
echo ========================================================
echo.

:: Check if the virtual environment exists
IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. 
    echo Please make sure you have run the setup script.
    echo.
    pause
    exit /b 1
)

:: Activate the virtual environment
call .venv\Scripts\activate.bat

:: Run the application
set PYTHONPATH=%~dp0
python -m src.cli

:: Keep the window open if the app crashes or exits
echo.
pause
