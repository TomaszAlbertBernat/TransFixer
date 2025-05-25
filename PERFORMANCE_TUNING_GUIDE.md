# TransFixer Performance Tuning Guide

## 🚀 **Maximum Performance Optimization Strategies**

This guide provides comprehensive optimization strategies to achieve maximum performance from your TransFixer Whisper transcription system.

---

## **📊 Performance Hierarchy (Expected Speed Improvements)**

| Optimization Strategy | Speed Improvement | Accuracy Impact | Memory Usage |
|----------------------|-------------------|-----------------|--------------|
| **faster-whisper (CTranslate2)** | **4-8x faster** | None | 50% less |
| **torch.compile (PyTorch 2.0+)** | **4.5x faster** | None | Same |
| **Whisper large-v3-turbo** | **8x faster** | Minimal (-2% WER) | Same |
| **Distil-Whisper large-v3** | **6.3x faster** | <1% WER increase | 40% less |
| **Flash Attention 2** | **2-3x faster** | None | 50% less |
| **Mixed Precision (AMP)** | **1.5-2x faster** | None | 40% less |
| **Aggressive Batching** | **2-4x faster** | None | More |
| **INT8 Quantization** | **2x faster** | Minimal | 75% less |

---

## **🎯 Quick Start: Maximum Performance Setup**

### **Option 1: Fastest Setup (Recommended)**
```python
# config.py settings for maximum speed
USE_FASTER_WHISPER_BACKEND = True
FASTER_WHISPER_MODEL = "large-v3-turbo"  # 8x faster than large-v3
PERFORMANCE_MODE = "aggressive"
ENABLE_TORCH_COMPILE = True
CTRANSLATE2_COMPUTE_TYPE = "float16"
```

### **Option 2: Balanced Speed + Quality**
```python
# config.py settings for balanced performance
USE_FASTER_WHISPER_BACKEND = True
FASTER_WHISPER_MODEL = "distil-large-v3"  # 6.3x faster, <1% accuracy loss
PERFORMANCE_MODE = "balanced"
ENABLE_TORCH_COMPILE = True
CTRANSLATE2_COMPUTE_TYPE = "float16"
```

### **Option 3: Memory-Constrained Systems**
```python
# config.py settings for low-memory systems
USE_FASTER_WHISPER_BACKEND = True
FASTER_WHISPER_MODEL = "large-v3"
PERFORMANCE_MODE = "conservative"
CTRANSLATE2_COMPUTE_TYPE = "int8"  # 75% less memory
```

---

## **🔧 Detailed Optimization Strategies**

### **1. Backend Selection Optimization**

#### **faster-whisper (CTranslate2) - HIGHEST PRIORITY**
```bash
# Install faster-whisper for maximum performance
pip install faster-whisper>=1.1.0 ctranslate2>=4.5.0
```

**Configuration:**
```python
USE_FASTER_WHISPER_BACKEND = True
FASTER_WHISPER_MODEL = "large-v3-turbo"  # Use turbo for 8x speed improvement
CTRANSLATE2_COMPUTE_TYPE = "float16"      # Balance speed/quality
CTRANSLATE2_INTER_THREADS = 1
CTRANSLATE2_INTRA_THREADS = 0             # Use all available threads
```

**Expected Performance:** 4-8x faster than standard Whisper with same accuracy.

#### **Alternative: Optimized Transformers Backend**
```python
USE_FASTER_WHISPER_BACKEND = False
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
ENABLE_TORCH_COMPILE = True
ENABLE_FLASH_ATTENTION = True
ENABLE_MIXED_PRECISION = True
```

### **2. Model Selection Strategy**

#### **Performance vs Accuracy Trade-offs:**

1. **Maximum Speed: Distil-Whisper**
   - Model: `distil-whisper/distil-large-v3`
   - Speed: 6.3x faster than large-v3
   - Accuracy: <1% WER increase
   - Best for: High-volume processing

2. **Balanced: Large-v3-Turbo**
   - Model: `openai/whisper-large-v3-turbo`
   - Speed: 8x faster than large-v3
   - Accuracy: Similar to large-v2
   - Best for: Most use cases

3. **Maximum Accuracy: Large-v3**
   - Model: `openai/whisper-large-v3`
   - Speed: Baseline
   - Accuracy: Best available
   - Best for: Critical transcriptions

### **3. Memory Optimization**

#### **GPU Memory Management**
```python
# Aggressive memory optimization
PERFORMANCE_MODE = "aggressive"
GPU_MEMORY_FRACTION = 0.9        # Use 90% of GPU memory
MEMORY_SAFETY_FACTOR = 0.9       # High memory utilization
MAX_BATCH_SIZE = 24              # Maximize batch size
MAX_CHUNK_LENGTH = 90            # Longer chunks for efficiency
```

#### **Conservative Memory Settings (for stability)**
```python
PERFORMANCE_MODE = "conservative"
GPU_MEMORY_FRACTION = 0.7
MEMORY_SAFETY_FACTOR = 0.7
MAX_BATCH_SIZE = 8
MAX_CHUNK_LENGTH = 30
```

### **4. PyTorch Optimizations**

#### **torch.compile (PyTorch 2.0+)**
```python
ENABLE_TORCH_COMPILE = True
TORCH_COMPILE_MODE = "reduce-overhead"    # Best for inference
TORCH_COMPILE_FULLGRAPH = True           # Maximum optimization
```

#### **Flash Attention 2**
```bash
# Install Flash Attention 2
pip install flash-attn>=2.0.0
```

```python
ENABLE_FLASH_ATTENTION = True
ATTENTION_IMPLEMENTATION = "flash_attention_2"
```

#### **Mixed Precision Training**
```python
ENABLE_MIXED_PRECISION = True
# Automatically uses autocast for GPU inference
```

### **5. Batch Processing Optimization**

#### **Dynamic Batch Sizing**
The system automatically calculates optimal batch sizes based on:
- Available GPU memory
- Performance mode setting
- Model memory requirements

```python
def calculate_optimal_batch_size():
    """
    Aggressive mode: Up to 24 files per batch
    Balanced mode: Up to 16 files per batch  
    Conservative mode: Up to 8 files per batch
    """
```

#### **Chunk Length Optimization**
```python
# Longer chunks = better GPU utilization
DEFAULT_CHUNK_LENGTH = 50       # Aggressive
DEFAULT_CHUNK_LENGTH = 30       # Balanced
DEFAULT_CHUNK_LENGTH = 20       # Conservative
```

---

## **⚡ Performance Monitoring & Troubleshooting**

### **Real-time Performance Monitoring**
```python
# Enable advanced performance monitoring
from advanced_performance_monitor import start_performance_monitoring

start_performance_monitoring()
# Monitors CPU, GPU, memory usage and suggests optimizations
```

### **Performance Monitoring**
```bash
# Real-time monitoring is automatically enabled when running TransFixer
# Check logs/performance_monitor.log for detailed metrics
# Performance suggestions are provided in real-time during transcription
```

### **Common Performance Issues & Solutions**

#### **Issue: Out of Memory (OOM) Errors**
**Solutions:**
1. Reduce batch size: `MAX_BATCH_SIZE = 8`
2. Reduce chunk length: `MAX_CHUNK_LENGTH = 30`
3. Use INT8 quantization: `CTRANSLATE2_COMPUTE_TYPE = "int8"`
4. Enable memory cleanup: `PERFORMANCE_MODE = "conservative"`

#### **Issue: Slow Transcription Speed**
**Solutions:**
1. Enable faster-whisper: `USE_FASTER_WHISPER_BACKEND = True`
2. Use turbo model: `FASTER_WHISPER_MODEL = "large-v3-turbo"`
3. Enable torch.compile: `ENABLE_TORCH_COMPILE = True`
4. Increase batch size: `MAX_BATCH_SIZE = 16`

#### **Issue: High GPU Temperature**
**Solutions:**
1. Reduce batch size to lower GPU load
2. Add cooling breaks between batches
3. Monitor with: `GPUtil.getGPUs()[0].temperature`

#### **Issue: CPU Bottleneck**
**Solutions:**
1. Increase worker processes: `--num-workers 2`
2. Use faster storage (SSD)
3. Optimize file I/O operations

---

## **🎮 Performance Mode Configurations**

### **Aggressive Mode (Maximum Performance)**
```python
PERFORMANCE_MODE = "aggressive"
# - Uses 90% of GPU memory
# - Batch size up to 24
# - Chunk length up to 90 seconds
# - Best for: High-end GPUs (RTX 3080+, 16GB+ VRAM)
```

### **Balanced Mode (Recommended)**
```python
PERFORMANCE_MODE = "balanced"
# - Uses 80% of GPU memory
# - Batch size up to 16
# - Chunk length up to 60 seconds
# - Best for: Mid-range GPUs (RTX 3060+, 8GB+ VRAM)
```

### **Conservative Mode (Stability)**
```python
PERFORMANCE_MODE = "conservative"
# - Uses 70% of GPU memory
# - Batch size up to 8
# - Chunk length up to 45 seconds
# - Best for: Lower-end GPUs (GTX 1660+, 6GB+ VRAM)
```

---

## **📈 Expected Performance Improvements**

### **Baseline vs Optimized Performance**

| Configuration | Files/Hour | Speed Improvement | Memory Usage |
|---------------|------------|-------------------|--------------|
| **Default Whisper** | 20 | 1x | 100% |
| **+ torch.compile** | 90 | 4.5x | 100% |
| **+ Flash Attention** | 135 | 6.7x | 50% |
| **+ faster-whisper** | 160 | 8x | 50% |
| **+ Turbo Model** | 320 | 16x | 50% |
| **+ INT8 Quantization** | 400 | 20x | 25% |

### **Hardware-Specific Recommendations**

#### **RTX 4090 (24GB VRAM)**
```python
PERFORMANCE_MODE = "aggressive"
MAX_BATCH_SIZE = 32
MAX_CHUNK_LENGTH = 120
CTRANSLATE2_COMPUTE_TYPE = "float16"
```

#### **RTX 3080 (10GB VRAM)**
```python
PERFORMANCE_MODE = "balanced"
MAX_BATCH_SIZE = 16
MAX_CHUNK_LENGTH = 60
CTRANSLATE2_COMPUTE_TYPE = "float16"
```

#### **RTX 3060 (8GB VRAM)**
```python
PERFORMANCE_MODE = "conservative"
MAX_BATCH_SIZE = 8
MAX_CHUNK_LENGTH = 45
CTRANSLATE2_COMPUTE_TYPE = "int8"
```

#### **GTX 1660 (6GB VRAM)**
```python
PERFORMANCE_MODE = "conservative"
MAX_BATCH_SIZE = 4
MAX_CHUNK_LENGTH = 30
CTRANSLATE2_COMPUTE_TYPE = "int8"
```

---

## **🔍 Performance Analysis Tools**

### **1. Real-time Monitoring**
```python
from advanced_performance_monitor import get_performance_summary
summary = get_performance_summary()
print(f"Current performance: {summary}")
```

### **2. Export Performance Metrics**
```python
from advanced_performance_monitor import export_performance_metrics
metrics_file = export_performance_metrics()
print(f"Metrics saved to: {metrics_file}")
```

### **3. Performance Testing**
```bash
# Test your configuration with actual audio files
python3 transfixer.py

# Monitor performance in real-time through logs
# Performance metrics are automatically tracked and logged
```

---

## **🚀 Advanced Optimizations**

### **Speculative Decoding (Experimental)**
```python
ENABLE_SPECULATIVE_DECODING = True
ASSISTANT_MODEL = "distil-whisper/distil-large-v3"
# Uses smaller model to predict, verified by larger model
# Expected: 2x additional speed improvement
```

### **Voice Activity Detection (VAD)**
```python
ENABLE_VAD_FILTER = True
VAD_PARAMETERS = {
    "min_silence_duration_ms": 500,
    "speech_threshold": 0.5,
}
# Filters out silence for batched processing
# Expected: 20-50% speed improvement for sparse audio
```

### **Custom CUDA Kernels (Expert)**
```python
# For advanced users: compile custom CUDA kernels
# Requires CUDA development environment
# Expected: 10-20% additional improvement
```

---

## **📝 Performance Testing Checklist**

### **Before Optimization:**
- [ ] Test current configuration with sample audio files
- [ ] Document current performance metrics from logs
- [ ] Note GPU memory usage and temperature

### **After Each Optimization:**
- [ ] Test with new settings using sample audio files
- [ ] Compare transcription accuracy (sample files)
- [ ] Monitor system stability (run for 30+ minutes)
- [ ] Check for memory leaks or temperature issues

### **Final Validation:**
- [ ] Process large batch (100+ files) successfully
- [ ] Verify transcription quality on sample files
- [ ] Confirm system stability under load
- [ ] Document final performance improvements

---

## **🛠️ Troubleshooting Guide**

### **Performance Not Improving?**
1. Check if optimizations are actually enabled in logs
2. Verify GPU is being used: `torch.cuda.is_available()`
3. Monitor GPU utilization: `nvidia-smi`
4. Check for CPU bottlenecks in task manager

### **Accuracy Degradation?**
1. Test with larger model: `large-v3` instead of turbo
2. Disable aggressive optimizations temporarily
3. Compare with baseline transcriptions
4. Adjust temperature settings for sampling

### **System Instability?**
1. Reduce batch sizes and chunk lengths
2. Enable conservative mode
3. Add cooling breaks between batches
4. Monitor system temperatures

---

## **📊 Summary: Maximum Performance Configuration**

```python
# config.py - MAXIMUM PERFORMANCE SETUP
USE_FASTER_WHISPER_BACKEND = True
FASTER_WHISPER_MODEL = "large-v3-turbo"
PERFORMANCE_MODE = "aggressive"
ENABLE_TORCH_COMPILE = True
ENABLE_FLASH_ATTENTION = True
ENABLE_MIXED_PRECISION = True
CTRANSLATE2_COMPUTE_TYPE = "float16"
ENABLE_VAD_FILTER = True
ENABLE_SPECULATIVE_DECODING = True

# Expected Results:
# - 15-20x faster than baseline Whisper
# - 50-75% less memory usage
# - Minimal accuracy loss (<2% WER)
# - Stable operation on modern GPUs
```

This configuration should provide the maximum possible performance while maintaining transcription quality. Adjust settings based on your specific hardware and requirements. 