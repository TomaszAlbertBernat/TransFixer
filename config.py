# config.py

import os

# --- Directory Configuration ---
AUDIO_DIR = "audio"
TRANSCRIPTIONS_DIR = "transcriptions"
CORRECTED_DIR = "corrected"
LOG_DIR = "logs"
BACKUP_DIR = "backup" # This was implicitly defined in transfixer.py, good to have it here.
LOG_FILE = os.path.join(LOG_DIR, "transcription_errors.log")

# --- Processing Configuration ---
MIN_CHARS = 50  # Minimum characters for a valid transcription
MAX_RETRIES = 3 # Max retries for Whisper transcription attempts
CHECK_INTERVAL = 300  # Seconds between processing cycles

# --- Whisper Model Configuration ---
# Options for Hugging Face: "openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small",
# "openai/whisper-medium", "openai/whisper-large", "openai/whisper-large-v2", "openai/whisper-large-v3"
# Your "large-v3-turbo" might be a custom name or require specific handling if not a direct HF model ID.
# For standard Hugging Face Transformers, you'd typically use "openai/whisper-large-v3".
WHISPER_MODEL = "openai/whisper-large-v3-turbo" # Changed to a standard Hugging Face model ID. Adjust if you have a specific "large-v3-turbo".

# --- Ollama Configuration ---
# Ensure your Ollama instance is running and the model is pulled (e.g., `ollama pull phi3:mini`)
OLLAMA_API_URL = "http://localhost:11434/api/chat"  # Default Ollama API endpoint for chat
OLLAMA_MODEL = "llama3.2:3b"  # Change this to your preferred model
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

# =============================================================================
# GPU OPTIMIZATION SETTINGS
# =============================================================================

# Performance Mode Selection
PERFORMANCE_MODE = "aggressive"  # Options: "conservative", "balanced", "aggressive"

# Performance Mode Configurations
PERFORMANCE_CONFIGS = {
    "conservative": {
        "gpu_memory_fraction": 0.7,
        "memory_safety_factor": 0.7,  # Use 70% of available memory
        "max_batch_size": 12,
        "max_chunk_length": 45,
        "aggressive_batching": False,
    },
    "balanced": {
        "gpu_memory_fraction": 0.8,
        "memory_safety_factor": 0.8,  # Use 80% of available memory
        "max_batch_size": 16,
        "max_chunk_length": 60,
        "aggressive_batching": True,
    },
    "aggressive": {
        "gpu_memory_fraction": 0.9,
        "memory_safety_factor": 0.9,  # Use 90% of available memory
        "max_batch_size": 24,
        "max_chunk_length": 90,
        "aggressive_batching": True,
    }
}

# Get current performance config
_current_config = PERFORMANCE_CONFIGS[PERFORMANCE_MODE]

# Apply current settings
GPU_MEMORY_FRACTION = _current_config["gpu_memory_fraction"]
MEMORY_SAFETY_FACTOR = _current_config["memory_safety_factor"]
MAX_BATCH_SIZE = _current_config["max_batch_size"]
AGGRESSIVE_BATCHING = _current_config["aggressive_batching"]

# Legacy settings (kept for compatibility)
CONSERVATIVE_BATCH_SIZING = not AGGRESSIVE_BATCHING  # Inverse of aggressive batching
ENABLE_MIXED_PRECISION = True  # Enable automatic mixed precision for faster inference
SMART_GPU_SELECTION = True  # Automatically select GPU with most available memory

# Model Settings
WHISPER_MODEL_CACHE_TIMEOUT = 3600  # Cache model for 1 hour
MIN_MEMORY_THRESHOLD = 0.85 if PERFORMANCE_MODE == "aggressive" else 0.8  # Higher threshold for aggressive mode

# Performance Settings - Dynamic based on performance mode
DEFAULT_CHUNK_LENGTH = min(50 if PERFORMANCE_MODE == "aggressive" else 30, _current_config["max_chunk_length"])
MIN_CHUNK_LENGTH = 20
MAX_CHUNK_LENGTH = _current_config["max_chunk_length"]

# =============================================================================
# PERFORMANCE MODE SUMMARY
# =============================================================================
def get_performance_summary():
    """Get a summary of current performance settings."""
    return {
        "mode": PERFORMANCE_MODE,
        "gpu_memory_fraction": f"{GPU_MEMORY_FRACTION*100:.0f}%",
        "memory_safety_factor": f"{MEMORY_SAFETY_FACTOR*100:.0f}%",
        "max_batch_size": MAX_BATCH_SIZE,
        "chunk_length_range": f"{MIN_CHUNK_LENGTH}-{MAX_CHUNK_LENGTH}s",
        "default_chunk_length": f"{DEFAULT_CHUNK_LENGTH}s",
        "aggressive_batching": AGGRESSIVE_BATCHING,
        "mixed_precision": ENABLE_MIXED_PRECISION
    }