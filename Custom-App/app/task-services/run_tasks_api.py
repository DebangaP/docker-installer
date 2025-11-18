#!/usr/bin/env python3
"""
Wrapper script to run TaskAPI using uvicorn
This is needed because the directory name 'task-services' contains a hyphen,
which prevents direct Python module imports
"""
# need to fix this
# This is needed because the directory name 'task-services' contains a hyphen,
# which prevents direct Python module imports

import sys
import os

# Add the task-services directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app from TaskAPI
from TaskAPI import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)

