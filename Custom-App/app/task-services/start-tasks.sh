#!/bin/bash

# Change to the application directory
cd /app

# Ensure logs directory exists (for cron jobs)
mkdir -p /app/logs
chmod 755 /app/logs

# Start the cron service in the background
cron
echo "Cron service started."

echo "Waiting for dependencies (like Postgres) to be ready..."
# Wait for postgres to be ready (optional, but recommended)
sleep 10

# Set PYTHONPATH to include /app so imports work correctly
export PYTHONPATH=/app

# Verify PYTHONPATH is set
echo "PYTHONPATH is set to: $PYTHONPATH"

# NOTE: All background tasks run via cron (defined in /etc/cron.d/crontab)
# - InsertOHLC.py runs once daily at 4:00 PM
# - PredictionScheduler.py runs once daily at 4:00 PM
# - All other scheduled tasks run according to their cron schedules

# Start Uvicorn server for on-demand task triggers (port 8001)
echo "Starting Uvicorn server for background tasks API on 0.0.0.0:8001..."
# Use the wrapper script to run TaskAPI (handles Python import issues with hyphens)
cd /app && PYTHONPATH=/app python /app/task-services/run_tasks_api.py &
UVICORN_PID=$!
echo "Uvicorn started with PID: $UVICORN_PID"

# Keep container running
echo "Cron jobs and Uvicorn API are active. Container will keep running..."
echo "Background tasks can be triggered via API on port 8001"
echo "Check /app/logs/ for task logs."

# Wait for Uvicorn process (keeps container alive)
wait $UVICORN_PID