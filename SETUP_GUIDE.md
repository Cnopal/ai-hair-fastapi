# AI Hair Shape Analysis - Setup & Troubleshooting Guide

## Issue Summary
When uploading images for face shape checking, the form keeps loading indefinitely.

## Root Cause
The FastAPI server on port 8888 that analyzes facial shapes wasn't running, so the Laravel application couldn't connect to it, causing the request to timeout.

## Solutions Implemented

### 1. **Fixed Model Path Issue**
- Updated `main.py` to use absolute path for the face_landmarker.task model
- **File**: `C:\laragon\www\ai-hair\main.py` (line 24-26)
- **Change**: From relative path to `os.path.join(os.path.dirname(__file__), "face_landmarker.task")`

### 2. **Updated FastAPI Port Configuration**
- Changed FastAPI server port from 8000 to 8888
- **File**: `C:\laragon\www\ai-hair\main.py` (line 328)
- **Updated .env**: `C:\laragon\www\fyp\.env` (line 70)
- **New URL**: `http://127.0.0.1:8888/face-shape`

### 3. **Improved Frontend Form Handling**
- Enhanced form submission to prevent stuck loading states
- **File**: `C:\laragon\www\fyp\resources\views\customer\hairstyles\ai-hair.blade.php`
- **Change**: Added proper form submission with submission delay

### 4. **Created Startup Scripts**
- **run_server.py**: Python wrapper to run FastAPI with error handling
- **start-api.bat**: Batch script to start the server
- **start-api.ps1**: PowerShell script to start the server

## How to Start the FastAPI Server

### Option 1: Run via Python (Recommended)
```bash
cd C:\laragon\www\ai-hair
C:\laragon\www\ai-hair\venv\Scripts\python.exe C:\laragon\www\ai-hair\run_server.py
```

### Option 2: Run via Batch File
```bash
C:\laragon\www\ai-hair\start-api.bat
```

### Option 3: Run via PowerShell
```powershell
C:\laragon\www\ai-hair\start-api.ps1
```

## Verify Server is Running
```bash
# Test the health endpoint
curl http://127.0.0.1:8888/health

# Or in PowerShell
Invoke-WebRequest -Uri "http://127.0.0.1:8888/" -UseBasicParsing | Select-Object -ExpandProperty Content
```

Expected response:
```json
{
  "message": "AI Hair Recommendation API",
  "version": "1.0.0",
  "endpoints": {
    "POST /face-shape": "Analyze face shape from uploaded image",
    "GET /": "API information"
  }
}
```

## Files Modified

1. **C:\laragon\www\ai-hair\main.py**
   - Updated model path handling (line 24-26)
   - Updated port to 8888 (line 328)

2. **C:\laragon\www\fyp\.env**
   - Updated FASTAPI_URL to port 8888

3. **C:\laragon\www\fyp\resources\views\customer\hairstyles\ai-hair.blade.php**
   - Improved form submission handling

4. **New files created**
   - C:\laragon\www\ai-hair\run_server.py
   - C:\laragon\www\ai-hair\start-api.bat
   - C:\laragon\www\ai-hair\start-api.ps1

## Testing the Upload Feature

1. Start the FastAPI server (see "How to Start the FastAPI Server" above)
2. Navigate to the customer dashboard
3. Go to "AI Hair Style Recommendation"
4. Upload a clear, front-facing photo
5. The system should analyze the image and provide hairstyle recommendations

## Troubleshooting

### Issue: "Address already in use" Error
```bash
# Find process using port 8888
netstat -ano | findstr :8888

# Kill the process (replace PID with the actual process ID)
taskkill /PID <PID> /F
```

### Issue: Module not found error
```bash
# Activate virtual environment and install dependencies
cd C:\laragon\www\ai-hair
venv\Scripts\activate
pip install -r requirement.txt
```

### Issue: "face_landmarker.task not found"
Ensure the file exists at: `C:\laragon\www\ai-hair\face_landmarker.task`

If not, you may need to download it from the MediaPipe website.

## Next Steps

To make this automatic, you could:
1. Create a Windows Task Scheduler job to run the FastAPI server at startup
2. Use a process manager like PM2 or Supervisor
3. Deploy to a cloud service or production server

## Environment Requirements
- Python 3.10.6 (in the virtual environment)
- FastAPI
- MediaPipe
- OpenCV (cv2)
- TensorFlow (for MediaPipe Face Landmarker)
(venv) PS C:\laragon\www\ai-hair> venv\Scripts\python.exe run_server.py