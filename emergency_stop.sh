#!/bin/bash

# Emergency stop script - kills ALL TransFixer processes immediately
echo "EMERGENCY STOP: Killing all TransFixer processes..."

# Kill by PID file first
PID_FILE="/tmp/transfixer.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Killing main process $PID..."
    kill -9 $PID 2>/dev/null || true
    rm -f "$PID_FILE"
fi

# Kill all Python processes containing "transfixer"
echo "Killing all transfixer.py processes..."
pkill -9 -f "transfixer.py" 2>/dev/null || true

# Kill all Python processes in this directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Killing Python processes in $SCRIPT_DIR..."
pkill -9 -f "$SCRIPT_DIR" 2>/dev/null || true

# Kill any remaining multiprocessing or torch processes
echo "Killing multiprocessing and torch processes..."
pkill -9 -f "multiprocessing" 2>/dev/null || true
pkill -9 -f "torch" 2>/dev/null || true

# Kill any remaining Whisper processes
pkill -9 -f "whisper" 2>/dev/null || true

echo "Emergency stop complete. All processes should be terminated."

# Show any remaining processes
echo "Checking for remaining processes..."
ps aux | grep -E "(transfixer|multiprocessing)" | grep -v grep || echo "No remaining processes found." 