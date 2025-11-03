#!/bin/bash

# Change to the application directory
cd /app

# Ensure logs directory exists (for cron jobs)
mkdir -p /app/logs
chmod 755 /app/logs

# Start the cron service in the background
cron
echo "Cron service started."

echo "Waiting for 2 minutes for dependencies (like Postgres) to be ready..."
#sleep 2m

# Start all scripts in the background BEFORE starting Uvicorn
echo "Starting background Python scripts..."
# Set PYTHONPATH to include /app so imports work correctly
export PYTHONPATH=/app:$PYTHONPATH
python -m kite.KiteFetchData &
python -m kite.KiteFetchFuture &
python -m kite.KiteWS &
#python -m kite.InsertOHLC &

# Start Uvicorn server as the main foreground process.
# This will keep the container running.
echo "Starting Uvicorn server on 0.0.0.0:8000..."
uvicorn KiteAccessToken:app --host 0.0.0.0 --port 8000 --reload # reload for Development only
