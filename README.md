# TransFixer

A powerful audio transcription and correction system that uses Whisper large-v3-turbo for high-quality transcriptions and Ollama for intelligent text correction. All work is being done locally for privacy.

## Features

- Automatic audio file transcription using Whisper large-v3-turbo
- Intelligent text correction and formatting using Ollama
- Automatic backup system
- GPU acceleration support
- Continuous processing with automatic retries
- Comprehensive logging system

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
   python transfixer.py
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
- Backup settings
- Directory paths
- Correction prompt template

## Monitoring

- Check `logs/transcription_errors.log` for detailed operation logs
- Monitor the application output for real-time status

## Troubleshooting

### Common Issues

1. Transcription fails:
   - Check GPU memory usage
   - Verify audio file format
   - Check logs for specific error messages

2. Correction fails:
   - Verify Ollama is running on port 11434
   - Check if the correct model is pulled
   - Review logs for error details

## Maintenance

- Backups are automatically created before each processing cycle
- Old backups are automatically cleaned up (keeps last backup by default)
- Logs are rotated automatically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.