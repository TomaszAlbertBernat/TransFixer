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
WHISPER_MODEL = "large-v3-turbo"  # Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"
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
    "Review and correct the following audio transcription for use in a Retrieval-Augmented Generation (RAG) system's knowledge base. "
    "Follow these instructions precisely:\n\n"
    "- Proofread thoroughly, fixing all grammar, spelling, and punctuation errors.\n"
    "- Remove all mentions of sponsors or advertisements.\n"
    "- Remove conversational filler such as 'uh', 'um', 'like', repetitions, false starts, and other non-essential speech artifacts.\n"
    "- Organize the corrected text into clear, logical paragraphs.\n"
    "- Ensure that all essential factual information and key points from the original transcription are preserved.\n"
    "- The final output should be clean, accurate, and easily readable plain text, ready for indexing in a RAG system."
)