# config.py

import os

# --- Directory Configuration ---
AUDIO_DIR = "audio"
TRANSCRIPTIONS_DIR = "transcriptions"

LOG_DIR = "logs"
BACKUP_DIR = "backup"
LOG_FILE = os.path.join(LOG_DIR, "transcription_errors.log")

# --- Processing Configuration ---
MIN_CHARS = 50  # Minimum characters for a valid transcription
MAX_RETRIES = 3 # Max retries for Whisper transcription attempts
CHECK_INTERVAL = 300  # Seconds between processing cycles

# --- Whisper Model Configuration ---
WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # Fast and accurate model



# --- Backup Configuration ---
MAX_BACKUPS = 1  # Maximum number of backups to keep



# --- Basic Performance Settings ---
ENABLE_MIXED_PRECISION = True  # Enable automatic mixed precision for faster inference
GPU_MEMORY_FRACTION = 0.95  # Increased from 0.9 to 0.95 for better GPU utilization
MAX_BATCH_SIZE = 32  # Increased from 24 to 32 for better GPU utilization
DEFAULT_CHUNK_LENGTH = 30  # Default chunk length in seconds
MIN_CHUNK_LENGTH = 20  # Minimum chunk length
MAX_CHUNK_LENGTH = 90  # Increased from 60 to 90 for better efficiency on longer audio

# --- Advanced Performance Settings ---
PERFORMANCE_MODE = "aggressive"  # Changed from "balanced" to "aggressive" for maximum GPU usage
MEMORY_SAFETY_FACTOR = 0.9  # Increased from 0.8 for more aggressive memory usage

# --- Flash Attention Configuration ---
ENABLE_FLASH_ATTENTION = True  # Changed from False to True - can improve GPU utilization
FLASH_ATTENTION_AVAILABLE = False  # Will be detected automatically
ATTENTION_IMPLEMENTATION = "flash_attention_2"  # Changed from "sdpa" to try flash attention first
ENABLE_SDPA = True  # Keep as fallback

# --- PyTorch Compile Configuration ---
ENABLE_TORCH_COMPILE = True  # Enable torch.compile for PyTorch 2.0+ - may improve GPU utilization
TORCH_COMPILE_MODE = "max-autotune"  # Changed from "reduce-overhead" to "max-autotune" for better GPU usage
TORCH_COMPILE_FULLGRAPH = True  # Enable fullgraph mode for torch.compile

# --- Performance Monitoring ---
PERFORMANCE_MONITORING_AVAILABLE = True  # Changed from False to True to monitor improvements