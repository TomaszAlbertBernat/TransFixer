# TransFixer

A powerful audio transcription and correction system that uses Whisper large-v3-turbo for high-quality transcriptions and LM Studio for intelligent text correction.

## Features

- Automatic audio file transcription using Whisper large-v3-turbo
- Intelligent text correction and formatting using LM Studio
- Automatic backup system
- GPU acceleration support
- Containerized deployment with Docker
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
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Docker Compose configuration
└── .dockerignore     # Docker ignore rules
```

## Prerequisites

### For Docker Setup
- Docker Engine 20.10.0 or later
- Docker Compose 2.0.0 or later
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA Driver 450.80.02 or later (for GPU support)

### For Local Setup
- Python 3.10 or later
- LM Studio running locally on port 1234
- NVIDIA GPU with CUDA support (recommended)

## Setup

### Option 1: Docker Setup (Recommended)

1. Install Docker and Docker Compose:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   
   # Windows
   # Download and install Docker Desktop from https://www.docker.com/products/docker-desktop
   ```

2. Install NVIDIA Container Toolkit (for GPU support):
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd TransFixer
   ```

4. Start the application:
   ```bash
   docker-compose up -d
   ```

5. Monitor the logs:
   ```bash
   docker-compose logs -f
   ```

### Option 2: Local Setup

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

3. Start LM Studio and ensure it's running on port 1234

4. Run the application:
   ```bash
   python transfixer.py
   ```

## Usage

1. Place your audio files in the `audio/` directory
2. The system will automatically:
   - Transcribe audio files using Whisper large-v3-turbo
   - Correct and format transcriptions using LM Studio
   - Store results in respective directories
   - Create backups automatically

## Configuration

Edit `config.py` to modify:
- Processing parameters
- API settings
- Directory paths
- Model settings
- Backup settings

## Monitoring

- Check `logs/transcription_errors.log` for detailed operation logs
- Monitor the Docker container:
  ```bash
  docker-compose logs -f
  ```

## Troubleshooting

### Docker Issues

1. Permission errors:
   ```bash
   sudo chown -R $USER:$USER .
   ```

2. GPU not detected:
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Test Docker GPU support
   docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. LM Studio connection issues:
   - Ensure LM Studio is running on port 1234
   - Check firewall settings
   - Test API endpoint: http://localhost:1234/v1/chat/completions

### Common Issues

1. Transcription fails:
   - Check GPU memory usage
   - Verify audio file format
   - Check logs for specific error messages

2. Correction fails:
   - Verify LM Studio is running
   - Check API endpoint accessibility
   - Review logs for error details

## Maintenance

- Backups are automatically created before each processing cycle
- Old backups are automatically cleaned up (keeps last 5 by default)
- Logs are rotated automatically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
