#!/bin/bash

# Force stop TransFixer script for WSL
# Use this if Ctrl+C doesn't work

PID_FILE="/tmp/transfixer.pid"

echo "Force stopping TransFixer..."

# Check PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Found PID file with process $PID"
    
    # Kill the process tree
    echo "Killing process tree..."
    pkill -P $PID 2>/dev/null || true
    kill $PID 2>/dev/null || true
    
    # Wait a bit
    sleep 2
    
    # Force kill if still running
    if kill -0 $PID 2>/dev/null; then
        echo "Force killing process $PID..."
        kill -9 $PID 2>/dev/null || true
    fi
    
    rm -f "$PID_FILE"
    echo "Removed PID file"
fi

# Clean up any remaining Python processes
echo "Cleaning up any remaining TransFixer processes..."
pkill -f "transfixer.py" 2>/dev/null || true

# Clean up any remaining multiprocessing processes
pkill -f "multiprocessing" 2>/dev/null || true

echo "TransFixer force-stopped." 