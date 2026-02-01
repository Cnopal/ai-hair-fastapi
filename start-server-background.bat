@echo off
REM AI Hair FastAPI Server Launcher
REM This script starts the FastAPI server in the background
REM It can be placed in the Windows Startup folder for automatic startup

echo.
echo ========================================
echo AI Hair FastAPI Server Starter
echo ========================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Navigate to the script directory
cd /d "%SCRIPT_DIR%"

REM Check if the venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Expected: "%SCRIPT_DIR%venv\Scripts\python.exe"
    echo.
    echo Please ensure you're running this from C:\laragon\www\ai-hair\
    pause
    exit /b 1
)

echo.
echo Starting FastAPI Server...
echo Server will run in the background
echo.
echo To stop the server:
echo   - Find "python.exe" in Task Manager
echo   - Right-click and select "End Task"
echo.
echo API will be available at:
echo   http://127.0.0.1:8888
echo.

REM Start the Python script in a hidden window
start "" "venv\Scripts\python.exe" "run_server.py"

REM Wait a moment for the server to start
timeout /t 3 /nobreak

REM Optionally open the browser to verify
REM start http://127.0.0.1:8888/

echo.
echo Server is starting. Check that it's running using:
echo   Invoke-WebRequest -Uri "http://127.0.0.1:8888/" 
echo.

exit /b 0
