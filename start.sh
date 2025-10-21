#!/bin/bash

# Start Uvicorn server in background
uvicorn KiteAccessToken:app --host 0.0.0.0 --port 8000 &

echo "Waiting for 5 minutes before continuing..."
sleep 5m

# Start both scripts in the background
python KiteFetchData.py &
python KiteWS.py &

#echo "Waiting for Kite access token..."

# Poll Redis or check file for access token availability
#while ! redis-cli -h redis ping 2>/dev/null; do
#    echo "Waiting for Redis..."
#    sleep 1
#done

#while [ -z "$(redis-cli -h redis get kite_access_token)" ]; do
#    echo "Access token not found, sleeping 5 seconds..."
#    sleep 5
#done

#echo "Kite access token found. Starting Python scripts..."


# Wait for all background processes to complete to keep container alive
wait