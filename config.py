import os

# Directory Configuration
AUDIO_DIR = "audio"
TRANSCRIPTIONS_DIR = "transcriptions"
CORRECTED_DIR = "corrected"
LOG_DIR = "logs"
BACKUP_DIR = "backup"
LOG_FILE = os.path.join(LOG_DIR, "transcription_errors.log")

# Processing Configuration
MIN_CHARS = 50
MAX_RETRIES = 3
WHISPER_MODEL = "base"  # Options: "tiny", "base", "small", "medium", "large"
CHECK_INTERVAL = 300  # Seconds between processing cycles

# Backup Configuration
MAX_BACKUPS = 5  # Maximum number of backups to keep

# API Configuration
LM_STUDIO_API = "http://localhost:1234/v1/chat/completions"

# Processing Parameters
LM_STUDIO_PARAMS = {
    "max_tokens": 4096,
    "temperature": 0.7
}

# Prompt Template
CORRECTION_PROMPT = (
    "Proofread the following transcription, remove any sponsor mentions, "
    "and structure it into clear paragraphs suitable for a RAG system. "
    "Ensure the output is concise and well-organized."
) 