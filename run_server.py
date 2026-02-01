#!/usr/bin/env python
"""Wrapper to run FastAPI server with proper error handling"""

import sys
import os
import traceback

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Loading FastAPI application...")
    from main import app
    
    print("Loading Uvicorn...")
    import uvicorn
    
    print("=" * 60)
    print("Starting AI Hair FastAPI Server...")
    print("API will be available at http://127.0.0.1:8888")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=8888, log_level="info")
    
except Exception as e:
    print(f"\nERROR: {str(e)}", file=sys.stderr)
    print("\nTraceback:", file=sys.stderr)
    traceback.print_exc()
    print("\nKeeping window open for 10 seconds...")
    import time
    time.sleep(10)
    sys.exit(1)
