#!/bin/bash

# Change to the application directory
cd /app

# Ensure logs directory exists
mkdir -p /app/logs
chmod 755 /app/logs

# NOTE: Cron jobs are handled in the python-tasks container
# This container only runs the FastAPI frontend server

echo "Waiting for dependencies (like Postgres) to be ready..."
#sleep 2m

# Set PYTHONPATH to include /app so imports work correctly
export PYTHONPATH=/app

# Verify PYTHONPATH is set
echo "PYTHONPATH is set to: $PYTHONPATH"

# NOTE: Background Python scripts (KiteFetchData, KiteFetchFuture, KiteWS, InsertOHLC)
# are now handled in the python-tasks container via cron jobs

# Start Uvicorn server as the main foreground process.
# This will keep the container running.
echo "Starting Uvicorn server on 0.0.0.0:8000..."
uvicorn KiteAccessToken:app --host 0.0.0.0 --port 8000 --reload # reload for Development only
