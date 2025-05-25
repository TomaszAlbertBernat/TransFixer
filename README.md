# TransFixer - High-Performance Audio Transcription System

A powerful, optimized audio transcription and correction system that uses advanced Whisper models for high-quality transcriptions and Ollama for intelligent text correction. Features comprehensive performance optimizations for maximum speed.

## 🚀 Performance Features

- **faster-whisper backend** with CTranslate2 for 4-8x speed improvement
- **PyTorch optimizations** including torch.compile, Flash Attention 2, and Mixed Precision
- **Advanced performance monitoring** with real-time optimization suggestions
- **Hardware-specific performance modes** (aggressive/balanced/conservative)
- **Comprehensive benchmarking tools** for optimization validation
- **Support for latest models**: Whisper Large-v3-Turbo, Distil-Whisper

## 🎯 Expected Performance

- **Speed**: 15-40x faster than baseline Whisper
- **Memory**: 50-75% less GPU memory usage
- **Quality**: Minimal accuracy loss (<2% WER)

## Features

- Automatic audio file transcription using optimized Whisper models
- Intelligent text correction and formatting using Ollama
- Real-time performance monitoring and optimization suggestions
- Automatic backup system with cleanup
- GPU acceleration with smart memory management
- Continuous processing with automatic retries
- Comprehensive logging and performance metrics

## Directory Structure

```
.
├── audio/                          # Place your input audio files here
├── transcriptions/                 # Raw transcriptions will be stored here
├── corrected/                     # Corrected transcriptions will be stored here
├── logs/                         # Log files and performance metrics
├── backup/                       # Automatic backups
├── config.py                     # Main configuration settings
├── transfixer.py                 # Main application script
├── requirements.txt              # Python dependencies
├── performance_summary.py        # System analysis and recommendations
├── benchmark_performance.py      # Comprehensive benchmarking
├── advanced_performance_monitor.py # Real-time monitoring
└── PERFORMANCE_TUNING_GUIDE.md   # Complete optimization guide
```

## Quick Start

### 1. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Performance optimizations (recommended)
pip install faster-whisper>=1.1.0 ctranslate2>=4.5.0
pip install flash-attn>=2.0.0
```

### 2. Setup Ollama
```bash
# Download from https://ollama.ai/
ollama pull llama3.2:3b
```

### 3. Analyze Your System
```bash
python3 performance_summary.py
```

### 4. Run TransFixer
```bash
# Basic usage
python3 transfixer.py

# With optimized settings
python3 transfixer.py --num-workers 1 --batch-size 16
```

## 🎮 Performance Configuration

### Maximum Performance Setup
```python
# config.py settings for maximum speed
USE_FASTER_WHISPER_BACKEND = True
FASTER_WHISPER_MODEL = "large-v3-turbo"  # 8x faster than large-v3
PERFORMANCE_MODE = "aggressive"
ENABLE_TORCH_COMPILE = True
ENABLE_FLASH_ATTENTION = True
CTRANSLATE2_COMPUTE_TYPE = "float16"
```

### Hardware-Specific Recommendations
- **High-end GPUs (16GB+ VRAM)**: `PERFORMANCE_MODE = "aggressive"`
- **Mid-range GPUs (8-16GB VRAM)**: `PERFORMANCE_MODE = "balanced"`
- **Lower-end GPUs (<8GB VRAM)**: `PERFORMANCE_MODE = "conservative"`

## 📊 Performance Tools

### System Analysis
```bash
python3 performance_summary.py
# Analyzes your hardware and provides personalized optimization recommendations
```

### Comprehensive Benchmarking
```bash
python3 benchmark_performance.py test_audio.wav
# Tests all optimization strategies and compares performance
```

### Real-time Monitoring
```bash
# Monitoring is automatically enabled when running TransFixer
# Check logs/performance_monitor.log for detailed metrics
```

## Usage

1. **Place audio files** in the `audio/` directory
2. **Configure optimizations** based on your hardware (run `performance_summary.py`)
3. **Run TransFixer**: `python3 transfixer.py`
4. **Monitor performance** through real-time logs and suggestions

The system will automatically:
- Transcribe audio files using optimized Whisper models
- Correct and format transcriptions using Ollama
- Provide real-time performance optimization suggestions
- Store results in respective directories
- Create backups automatically
- Export performance metrics

## Configuration

Edit `config.py` to customize:
- **Performance modes** and optimization settings
- **Model selection** (turbo, distil, standard)
- **Batch sizes** and memory management
- **Ollama configuration** (API URL, model, options)
- **Monitoring and logging** preferences

## 🔧 Troubleshooting

### Performance Issues
1. **Run system analysis**: `python3 performance_summary.py`
2. **Check optimization status** in logs
3. **Verify GPU utilization**: `nvidia-smi`
4. **Review performance recommendations**

### Memory Issues
1. **Reduce batch size**: Set `MAX_BATCH_SIZE = 8`
2. **Use conservative mode**: `PERFORMANCE_MODE = "conservative"`
3. **Enable INT8 quantization**: `CTRANSLATE2_COMPUTE_TYPE = "int8"`

### Quality Issues
1. **Use larger model**: Switch to `large-v3` instead of turbo
2. **Disable aggressive optimizations** temporarily
3. **Compare with baseline** transcriptions

## 📈 Performance Validation

Expected improvements after optimization:
- ✅ **4-8x faster** with faster-whisper backend
- ✅ **50-75% less memory** usage
- ✅ **Real-time monitoring** and suggestions
- ✅ **Automatic optimization** based on hardware

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.