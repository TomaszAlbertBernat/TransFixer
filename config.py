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
OLLAMA_MODEL = "phi4-mini"  # IMPORTANT: Set this to the Ollama model you have pulled and want to use (e.g., "llama3", "mistral", "phi3:latest")
OLLAMA_OPTIONS = { # Options to pass to the Ollama model for text generation
    "temperature": 0.7,
    "num_ctx": 4096, # Context window size. Adjust based on the model and your needs.
    # "top_k": 40,
    # "top_p": 0.9,
    # Add other Ollama options here as needed. Refer to Ollama documentation for available options.
}

# --- Backup Configuration ---
MAX_BACKUPS = 1  # Maximum number of backups to keep

# --- Prompt Template for Correction ---
CORRECTION_PROMPT = (
    "Review and correct the following audio transcription for use in a Retrieval-Augmented Generation (RAG) system's knowledge base. "
    "Follow these instructions precisely:\n\n"
    "- Proofread thoroughly, fixing all grammar, spelling, and punctuation errors.\n"
    "- Remove all mentions of sponsors or advertisements.\n"
    "- Remove conversational filler such as 'uh', 'um', 'like', repetitions, false starts, and other non-essential speech artifacts.\n"
    "- Organize the corrected text into clear, logical paragraphs.\n"
    "- Ensure that all essential factual information and key points from the original transcription are preserved.\n"
    "- The final output should be clean, accurate, and easily readable plain text, ready for indexing in a RAG system. "
    "Do not add any conversational fluff, preambles, or concluding remarks; output only the corrected text itself."
)