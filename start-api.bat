@echo off
REM Start FastAPI server for AI Hair Shape Analysis
echo Starting AI Hair FastAPI Server...
echo.

REM Navigate to the script directory
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo No virtual environment found. Using system Python.
)

REM Install requirements if not already installed
echo Installing/Updating requirements...
pip install -q -r requirement.txt

REM Start the FastAPI server
echo.
echo Starting server on http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
