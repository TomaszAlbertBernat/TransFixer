# config.py

import os

# --- Directory Configuration ---
AUDIO_DIR = "audio"
TRANSCRIPTIONS_DIR = "transcriptions"
CORRECTED_DIR = "corrected"
LOG_DIR = "logs"
BACKUP_DIR = "backup"
LOG_FILE = os.path.join(LOG_DIR, "transcription_errors.log")

# --- Processing Configuration ---
MIN_CHARS = 50  # Minimum characters for a valid transcription
MAX_RETRIES = 3 # Max retries for Whisper transcription attempts
CHECK_INTERVAL = 300  # Seconds between processing cycles

# --- Whisper Model Configuration ---
WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # Fast and accurate model

# --- Ollama Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "max_tokens": 1024
}

# --- Backup Configuration ---
MAX_BACKUPS = 1  # Maximum number of backups to keep

# --- Prompt Template for Correction ---
CORRECTION_PROMPT = """Please correct the following transcription text for any spelling mistakes, grammatical errors, punctuation issues, or transcription artifacts. 
Maintain the original meaning and flow of the text. Return only the corrected text without additional commentary or explanation."""

# --- Basic Performance Settings ---
ENABLE_MIXED_PRECISION = True  # Enable automatic mixed precision for faster inference
GPU_MEMORY_FRACTION = 0.8  # Use 80% of available GPU memory
MAX_BATCH_SIZE = 16  # Maximum batch size for transcription
DEFAULT_CHUNK_LENGTH = 30  # Default chunk length in seconds
MIN_CHUNK_LENGTH = 20  # Minimum chunk length
MAX_CHUNK_LENGTH = 60  # Maximum chunk length