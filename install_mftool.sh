#!/bin/bash
# Script to install mftool in running Docker container without rebuilding
# Usage: ./install_mftool.sh

echo "Installing mftool in Docker container..."

# Find the Python container
CONTAINER_NAME=$(docker ps --format "{{.Names}}" | grep -i "python\|custom\|app" | head -n 1)

if [ -z "$CONTAINER_NAME" ]; then
    echo "Error: Python container not found. Please start the container first."
    echo "Run: docker compose up -d"
    exit 1
fi

echo "Found container: $CONTAINER_NAME"
echo "Installing mftool..."

# Install mftool in the running container
docker exec -it $CONTAINER_NAME pip install --no-cache-dir mftool>=0.2.0

if [ $? -eq 0 ]; then
    echo "✓ Successfully installed mftool"
    echo "Verifying installation..."
    docker exec -it $CONTAINER_NAME python -c "import mftool; print('mftool version:', mftool.__version__ if hasattr(mftool, '__version__') else 'installed')"
    echo ""
    echo "Note: This installation will persist in the container until it is recreated."
    echo "If you recreate the container, you'll need to run this script again or rebuild the image."
else
    echo "✗ Failed to install mftool"
    exit 1
fi

