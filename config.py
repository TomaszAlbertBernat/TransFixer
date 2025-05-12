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
OLLAMA_MODEL = "qwen3:8b"  # Example: "llama3", "phi3", "mistral". Ensure this model is pulled in Ollama.
OLLAMA_OPTIONS = {  # Options to pass to the Ollama model
    "temperature": 0.7,
    "num_ctx": 9000,  # Example context window size, adjust based on model
    # Add other Ollama options here as needed: e.g., top_k, top_p
}

# --- Backup Configuration ---
MAX_BACKUPS = 1  # Maximum number of backups to keep

# --- Prompt Template for Correction ---
CORRECTION_PROMPT = (
    "**Task:**\n"
    "> You are a sophisticated information extraction assistant. Your role is to analyze a provided transcription of a\n"
    "> conversation, identify key topics, extract structured information, and organize it into a standardized format for\n"
    "> use in Retrieval-Augmented Generation (RAG) systems.\n\n"
    "> **Instructions:**\n"
    "> 1. **Read the transcription carefully** and identify all explicitly discussed topics (e.g., projects, events,\n"
    "> decisions, conflicts, etc.).\n"
    "> 2. **Extract structured data** for each topic, including:\n"
    ">    - **Topic Name** (e.g., \"Project Launch\", \"Budget Review\")\n"
    ">    - **Key Points** (e.g., objectives, timelines, stakeholders, decisions made)\n"
    ">    - **Participants** (names/roles of individuals involved)\n"
    ">    - **Dates/Time** (if mentioned)\n"
    ">    - **Action Items** (tasks assigned or agreed upon)\n"
    ">    - **Contextual Notes** (additional details or nuances)\n"
    "> 3. **Organize the output** into a JSON or structured format, ensuring clarity and precision.\n"
    "> 4. **Avoid adding unverified information** or assumptions not explicitly stated in the transcription.\n"
    "> 5. **Highlight critical decisions or unresolved issues** if applicable.\n"
    "> 6. **Include metadata** such as the transcription source, date, and speaker roles (if available).\n\n"
    "> **Example Output Format:**\n"
    "> ```json\n"
    "> {\n"
    ">   \"transcription_metadata\": {\n"
    ">     \"source\": \"Meeting Transcript - Q3 Strategy Session\",\n"
    ">     \"date\": \"2023-10-15\",\n"
    ">     \"participants\": [\"Alice (Project Lead)\", \"Bob (CTO)\", \"Charlie (Marketing Head)\"]\n"
    ">   },\n"
    ">   \"topics\": [\n"
    ">     {\n"
    ">       \"topic_name\": \"Project Launch\",\n"
    ">       \"key_points\": [\n"
    ">         \"Launch date set for Q4 2023\",\n"
    ">         \"Budget allocation: $500k for development\",\n"
    ">         \"Stakeholders: Alice, Bob, Charlie\"\n"
    ">       ],\n"
    ">       \"action_items\": [\n"
    ">         \"Finalize vendor contracts by 2023-10-20\",\n"
    ">         \"Prepare marketing materials by 2023-10-25\"\n"
    ">       ],\n"
    ">       \"contextual_notes\": \"Discussed potential risks related to vendor delays.\"\n"
    ">     }\n"
    ">   ]\n"
    "> }\n"
    "> ```\n\n"
    "> **Notes for the Model:**\n"
    "> - Prioritize accuracy over completeness. If information is ambiguous, flag it as \"unclear\" or \"not specified.\"\n"
    "> - Use consistent terminology (e.g., \"action items\" vs. \"tasks\")\n"
    "> - If the transcription contains multiple languages, extract information in the original language unless\n"
    "> instructed otherwise\n"
    "> - For RAG integration, ensure the structured data is searchable and semantically rich"
)