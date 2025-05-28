#!/bin/bash

# TransFixer Runner Script with CUDA Support
# This script sets up the environment for faster-whisper with CUDA

# Activate virtual environment
source venv/bin/activate

# Get CUDA library paths
CUDA_LIBS=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')

# Set LD_LIBRARY_PATH for CUDA libraries
export LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH"

echo "🚀 Starting TransFixer with CUDA support..."
echo "📍 CUDA Libraries: $CUDA_LIBS"

# Run TransFixer with all arguments passed to this script
python3 transfixer.py "$@" 