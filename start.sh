#!/bin/bash

# Start Uvicorn server in background
uvicorn KiteAccessToken:app --host 0.0.0.0 --port 8000 &

echo "Waiting for 3 minutes before continuing..."
sleep 4m

# Start all scripts in the background
python KiteFetchData.py &
python KiteFetchFuture.py &
python KiteWS.py &
#python InsertOHLC.py &

# Wait for all background processes to complete to keep container alive
wait