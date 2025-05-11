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
OLLAMA_MODEL = "phi4-mini"  # Example: "llama3", "phi3", "mistral". Ensure this model is pulled in Ollama.
OLLAMA_OPTIONS = {  # Options to pass to the Ollama model
    "temperature": 0.7,
    "num_ctx": 4096,  # Example context window size, adjust based on model
    # Add other Ollama options here as needed: e.g., top_k, top_p
}

# --- Backup Configuration ---
MAX_BACKUPS = 1  # Maximum number of backups to keep

# --- Prompt Template for Correction ---
CORRECTION_PROMPT = (
    "Please correct any grammar, spelling, and punctuation errors in the following text. "
    "Also, improve sentence structure and clarity where needed, while preserving the original meaning. "
    "The text is a transcription of spoken audio. Focus on readability and accuracy. "
    "Do not add any conversational fluff or introductory/concluding remarks, just output the corrected text."
)