#!/bin/bash

# Convert line endings from CRLF to LF for start-tasks.sh (handles Windows line endings)
# This is needed because volume mounts can override files with CRLF endings

# Function to convert CRLF to LF using tr (more universal than sed)
convert_line_endings() {
    local file="$1"
    if [ -f "$file" ]; then
        # Use tr to remove carriage returns, then overwrite the file
        tr -d '\r' < "$file" > "$file.tmp" && mv "$file.tmp" "$file"
        chmod +x "$file"
        return 0
    fi
    return 1
}

# Convert the script that will be executed (copied during build)
convert_line_endings /start-tasks.sh

# Also convert the volume-mounted version if it exists (for reference)
convert_line_endings /app/task-services/start-tasks.sh

# Execute the start script
exec bash /start-tasks.sh

