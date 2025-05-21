# TransFixer

A powerful audio transcription and correction system that uses Whisper large-v3-turbo for high-quality transcriptions and Ollama for intelligent text correction. All work is being done locally for privacy.

## Features

- Automatic audio file transcription using Whisper large-v3-turbo
- Intelligent text correction and formatting using Ollama
- Automatic backup system with configurable retention (default: 1 backup)
- GPU acceleration support with optional torch.compile optimization
- Continuous processing with automatic retries
- Comprehensive logging system
- Flexible processing modes:
  - Parallel processing with configurable worker count
  - Batched processing for memory efficiency
  - Optional transcription phase skipping

## System Requirements

- **CPU**: Modern multi-core processor
- **RAM**: Minimum 16GB (32GB recommended for parallel processing)
- **GPU**: NVIDIA GPU with at least 8GB VRAM (required for optimal performance)
- **Storage**: At least 10GB free space for models and temporary files
- **Operating System**: Windows 10/11, Linux, or macOS

## Directory Structure

```
.
├── audio/              # Place your input audio files here (supports .mp3 files)
├── transcriptions/     # Raw transcriptions will be stored here
├── corrected/         # Corrected transcriptions will be stored here
├── logs/             # Log files will be stored here
├── backup/           # Backup files will be stored here
├── config.py         # Configuration settings
├── transfixer.py     # Main application script
├── requirements.txt  # Python dependencies
└── setup.py         # Setup script
```

## Prerequisites

### For Local Setup
- Python 3.10 or later
- Ollama running locally on port 11434
- NVIDIA GPU with CUDA support (recommended)
- FFmpeg installed for audio processing

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install and start Ollama:
   - Download from https://ollama.ai/
   - Pull the required model:
     ```bash
     ollama pull qwen3:8b
     ```

4. Run the application:
   ```bash
   python transfixer.py [options]
   ```

## Command Line Options

The script supports several command line arguments:

- `--workers N`: Number of parallel Whisper instances (default: 2)
- `--batch-size N`: Number of audio files to process in a single batch (default: 4)
- `--skip-transcribing`: Skip the transcription phase and proceed directly to correction
- `--use-batching`: Use batched processing instead of multiple workers (more memory efficient)
- `--use-torch-compile`: Enable torch.compile for Whisper model (requires PyTorch 2.0+)

Example usage:
```bash
# Run with 4 parallel workers
python transfixer.py --workers 4

# Run in batch mode with torch.compile optimization
python transfixer.py --use-batching --use-torch-compile --batch-size 8

# Skip transcription and only run correction
python transfixer.py --skip-transcribing
```

## Usage

1. Place your audio files in the `audio/` directory
2. The system will automatically:
   - Transcribe audio files using Whisper large-v3-turbo
   - Correct and format transcriptions using Ollama
   - Store results in respective directories
   - Create backups automatically

## Configuration

Edit `config.py` to modify:
- Processing parameters (MIN_CHARS, MAX_RETRIES, CHECK_INTERVAL)
- Whisper model settings
- Ollama configuration (API URL, model, options)
- Backup settings (default: 1 backup)
- Directory paths
- Correction prompt template

## Key Dependencies

- PyTorch 2.7.0
- Transformers 4.51.3
- Whisper large-v3-turbo
- Ollama with qwen3:8b model
- CUDA 12.x (for GPU acceleration)

## Monitoring

- Check `logs/transcription_errors.log` for detailed operation logs
- Monitor the application output for real-time status
- Progress bars show transcription and correction progress

## Troubleshooting

### Common Issues

1. Transcription fails:
   - Check GPU memory usage
   - Verify audio file format
   - Check logs for specific error messages
   - Try using batch mode if running out of memory

2. Correction fails:
   - Verify Ollama is running on port 11434
   - Check if the correct model is pulled
   - Review logs for error details

3. Performance issues:
   - Reduce number of parallel workers if memory constrained
   - Enable batch processing for better memory efficiency
   - Consider using torch.compile for improved performance

## Maintenance

- Backups are automatically created before each processing cycle
- Old backups are automatically cleaned up (keeps last backup by default)
- Logs are rotated automatically
- Lock files are automatically cleaned up on startup and in case of crashes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.