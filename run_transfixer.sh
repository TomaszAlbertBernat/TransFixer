#!/bin/bash

# WSL-friendly TransFixer runner script
# This script properly handles Ctrl+C in WSL environments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/transfixer.py"
PID_FILE="/tmp/transfixer.pid"

# Function to cleanup
cleanup() {
    echo "Received interrupt signal, cleaning up..."
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Killing TransFixer process $PID and its children..."
        
        # First, try graceful termination with SIGTERM
        kill -TERM $PID 2>/dev/null || true
        pkill -P $PID 2>/dev/null || true
        
        # Wait briefly for graceful shutdown
        sleep 1
        
        # Check if still running and force kill
        if kill -0 $PID 2>/dev/null; then
            echo "Process still running, force killing..."
            kill -9 $PID 2>/dev/null || true
            pkill -9 -P $PID 2>/dev/null || true
        fi
        
        # Additional cleanup for any remaining processes
        sleep 1
        pkill -f "transfixer.py" 2>/dev/null || true
        
        rm -f "$PID_FILE"
    fi
    echo "Cleanup complete."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 $OLD_PID 2>/dev/null; then
        echo "TransFixer is already running with PID $OLD_PID"
        echo "Use 'kill $OLD_PID' to stop it, or delete $PID_FILE if it's stale"
        exit 1
    else
        echo "Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# Start the Python script in background and capture its PID
echo "Starting TransFixer..."
cd "$SCRIPT_DIR"

# Pass all arguments to the Python script
python3 "$PYTHON_SCRIPT" "$@" &
PYTHON_PID=$!

# Save the PID
echo $PYTHON_PID > "$PID_FILE"
echo "TransFixer started with PID $PYTHON_PID"
echo "Press Ctrl+C to stop gracefully"

# Wait for the background process
wait $PYTHON_PID

# Clean up PID file when done
rm -f "$PID_FILE"
echo "TransFixer finished." 