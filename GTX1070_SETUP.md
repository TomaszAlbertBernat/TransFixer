# GTX 1070 Setup Guide for TransFixer

## 🎯 Quick Start for GTX 1070 Users

If you have an NVIDIA GTX 1070 and are experiencing cuDNN issues, this guide will help you get TransFixer working optimally.

### 🚀 TL;DR - Quick Fix

```bash
# Run the diagnostic tool first
python gtx1070_diagnostic.py

# Use optimal settings for GTX 1070
python transfixer.py --compute-type int8 --verbose-logging
```

---

## 📋 Why GTX 1070 Has Special Requirements

The GTX 1070 uses **Pascal architecture** (Compute Capability 6.1), which has specific limitations:

- ❌ **No Tensor Cores**: float16 operations are inefficient/problematic
- ✅ **Supports int8**: Excellent for quantized models  
- ⚠️ **cuDNN Compatibility**: Requires specific CUDA/cuDNN versions

---

## 🔧 Complete Setup Guide

### Step 1: Install Diagnostic Tool

```bash
# Download and run the GTX 1070 diagnostic
python gtx1070_diagnostic.py
```

This will check your entire setup and provide specific recommendations.

### Step 2: Install Compatible Software

#### Option A: CUDA 11.8 (Recommended)
```bash
# Install CUDA 11.8 from NVIDIA
# Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive

# Install compatible PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install faster-whisper  
pip install faster-whisper
```

#### Option B: CUDA 12.x (Alternative)
```bash
# If you already have CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper
```

### Step 3: Configure Environment

Add to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
# For GTX 1070 optimization
export CT2_CUDA_ALLOW_FP16=1  # Only if using float16 (not recommended)
export CUDA_VISIBLE_DEVICES=0  # If you have multiple GPUs
```

---

## ⚙️ Optimal TransFixer Settings

### Best Performance Settings

```bash
# Recommended: int8 quantization (fastest on GTX 1070)
python transfixer.py --compute-type int8 --backend faster-whisper

# Alternative: float32 (more memory usage but reliable)
python transfixer.py --compute-type float32 --backend faster-whisper

# Conservative: smaller batches
python transfixer.py --compute-type int8 --batch-size 4
```

### If CUDA Still Doesn't Work

```bash
# Force CPU mode (slower but reliable)
python transfixer.py --force-cpu

# Try transformers backend
python transfixer.py --backend transformers --compute-type float32
```

---

## 🔍 Troubleshooting Common Issues

### Issue 1: Process exits with code -1073740791

**Cause**: Missing cuDNN libraries in PATH

**Solution**:
```bash
# Add cuDNN to PATH (Windows)
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;%PATH%

# Linux/WSL - check cuDNN installation
sudo find /usr -name "libcudnn*"
```

### Issue 2: "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED"

**Cause**: float16 incompatibility with Pascal architecture

**Solution**:
```bash
# Use int8 instead of float16
python transfixer.py --compute-type int8
```

### Issue 3: "CUDA out of memory"

**Cause**: GTX 1070 has limited 8GB VRAM

**Solution**:
```bash
# Reduce batch size
python transfixer.py --batch-size 4 --compute-type int8

# Or use conservative performance mode
# Edit config.py: PERFORMANCE_MODE = "conservative"
```

### Issue 4: Very slow transcription

**Cause**: Using float16 on Pascal (no Tensor Cores)

**Solution**:
```bash
# Switch to int8 for 2-3x speed improvement
python transfixer.py --compute-type int8
```

---

## 📊 Expected Performance

With optimal settings on GTX 1070:

- **int8 quantization**: ~2-3x realtime speed
- **float32**: ~1.5-2x realtime speed  
- **float16**: ~0.5-1x realtime speed (not recommended)

### Memory Usage
- **int8**: ~3-4GB VRAM
- **float32**: ~6-7GB VRAM
- **float16**: ~4-5GB VRAM (but slow)

---

## 🛠️ Advanced Configuration

### Edit config.py for GTX 1070

```python
# Optimal settings for GTX 1070
PERFORMANCE_MODE = "balanced"  # or "conservative" if issues persist
CTRANSLATE2_COMPUTE_TYPE = "int8"  # Best for Pascal
USE_FASTER_WHISPER_BACKEND = True
FASTER_WHISPER_MODEL = "turbo"  # or "large-v3" for accuracy

# Memory optimization
MAX_BATCH_SIZE = 8  # Reduce if OOM errors
GPU_MEMORY_FRACTION = 0.8  # Use 80% of 8GB
```

### Environment Variables for Debugging

```bash
# Enable verbose logging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_LOGS="+dynamo"

# Disable problematic features if needed
export CUDNN_ENABLED=0  # Disable cuDNN entirely
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

---

## 🆘 Getting Help

### Run Diagnostics
```bash
# Comprehensive GTX 1070 diagnostics
python gtx1070_diagnostic.py

# Verbose TransFixer logging
python transfixer.py --verbose-logging --compute-type int8
```

### Common Working Configurations

#### Configuration 1: Maximum Performance
```bash
python transfixer.py --compute-type int8 --backend faster-whisper --batch-size 8
```

#### Configuration 2: Maximum Stability  
```bash
python transfixer.py --compute-type float32 --backend transformers --batch-size 4
```

#### Configuration 3: Fallback (CPU)
```bash
python transfixer.py --force-cpu --num-workers 2
```

---

## 📚 Additional Resources

- [NVIDIA CUDA 11.8 Download](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN 8.6.0 Download](https://developer.nvidia.com/rdp/cudnn-archive)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [faster-whisper Documentation](https://github.com/guillaumekln/faster-whisper)

---

## ✅ Verification Checklist

After setup, verify your installation:

- [ ] `nvidia-smi` shows GTX 1070 
- [ ] `nvcc --version` shows CUDA 11.8+
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] `python gtx1070_diagnostic.py` passes all tests
- [ ] `python transfixer.py --compute-type int8` works without errors

---

**🎉 Success!** Your GTX 1070 should now work optimally with TransFixer using int8 quantization. 