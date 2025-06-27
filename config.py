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
GPU_MEMORY_FRACTION = 0.9  # Use 80% of available GPU memory
MAX_BATCH_SIZE = 16  # Maximum batch size for transcription
DEFAULT_CHUNK_LENGTH = 30  # Default chunk length in seconds
MIN_CHUNK_LENGTH = 20  # Minimum chunk length
MAX_CHUNK_LENGTH = 60  # Maximum chunk length

# --- Advanced Performance Settings ---
PERFORMANCE_MODE = "balanced"  # Options: "conservative", "balanced", "aggressive"
MEMORY_SAFETY_FACTOR = 0.8  # Memory safety factor for batch sizing

# --- Flash Attention Configuration ---
ENABLE_FLASH_ATTENTION = False  # Enable Flash Attention 2 if available
FLASH_ATTENTION_AVAILABLE = False  # Will be detected automatically
ATTENTION_IMPLEMENTATION = "sdpa"  # Options: "sdpa", "flash_attention_2", "eager"
ENABLE_SDPA = True  # Enable Scaled Dot Product Attention

# --- PyTorch Compile Configuration ---
ENABLE_TORCH_COMPILE = False  # Enable torch.compile for PyTorch 2.0+
TORCH_COMPILE_MODE = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune"
TORCH_COMPILE_FULLGRAPH = True  # Enable fullgraph mode for torch.compile

# --- Performance Monitoring ---
PERFORMANCE_MONITORING_AVAILABLE = False  # Advanced performance monitoring (requires extra dependencies)