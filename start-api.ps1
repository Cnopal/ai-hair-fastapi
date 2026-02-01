# AI Hair FastAPI Server Startup Script

Write-Host "Starting AI Hair FastAPI Server..." -ForegroundColor Cyan
Write-Host ""

# Set script directory as working directory
Set-Location -Path $PSScriptRoot

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment if it exists
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Host "No virtual environment found. Using system Python." -ForegroundColor Yellow
}

# Install/Update requirements
Write-Host "Installing/Updating requirements..." -ForegroundColor Cyan
pip install -q -r requirement.txt

# Start the FastAPI server
Write-Host ""
Write-Host "Starting server on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Read-Host "Press Enter to exit"
