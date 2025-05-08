import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the recommended directory structure for the transcription system."""
    # Define directories
    directories = [
        "audio",           # For input audio files
        "transcriptions",  # For raw transcriptions
        "corrected",      # For corrected transcriptions
        "logs",          # For log files
        "backup",        # For backup files
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create a .gitkeep file in each directory to ensure they're tracked by git
    for directory in directories:
        gitkeep_path = os.path.join(directory, ".gitkeep")
        Path(gitkeep_path).touch()
        print(f"Created .gitkeep in: {directory}")
    
    # Create a README.md file
    readme_content = """# TransFixer

A transcription and correction system for audio files.

## Directory Structure

- `audio/`: Place your input audio files here (supports .mp3 files)
- `transcriptions/`: Raw transcriptions will be stored here
- `corrected/`: Corrected transcriptions will be stored here
- `logs/`: Log files will be stored here
- `backup/`: Backup files will be stored here

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure LM Studio is running locally on port 1234

3. Place your audio files in the `audio/` directory

4. Run the script:
```bash
python transfixer.py
```

## Configuration

Edit `config.py` to modify:
- Processing parameters
- API settings
- Directory paths
- Model settings

## Logs

Check `logs/transcription_errors.log` for detailed operation logs.
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("Created README.md")

    # Create a .gitignore file
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class

# Logs
logs/*
!logs/.gitkeep

# Audio files
audio/*
!audio/.gitkeep

# Transcriptions
transcriptions/*
!transcriptions/.gitkeep

# Corrected files
corrected/*
!corrected/.gitkeep

# Backup files
backup/*
!backup/.gitkeep

# Environment
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
"""
    
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print("Created .gitignore")

if __name__ == "__main__":
    create_directory_structure() 