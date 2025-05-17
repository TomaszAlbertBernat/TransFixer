import os
import subprocess
import shutil
import logging
from multiprocessing import Pool, set_start_method
import requests # For Ollama API calls
import time
from pathlib import Path
import signal
import sys
from datetime import datetime
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from config import OLLAMA_MODEL, CORRECTION_PROMPT, OLLAMA_API_URL, OLLAMA_OPTIONS
from tqdm import tqdm
import re

# --- START: Configuration ---
# Original Configuration variables
AUDIO_DIR = "audio"
TRANSCRIPTIONS_DIR = "transcriptions"
CORRECTED_DIR = "corrected"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "transcription_errors.log")
MIN_CHARS = 50
MAX_RETRIES = 3 # For Whisper transcription
WHISPER_MODEL = "openai/whisper-large-v3-turbo" # Changed from v3-turbo as it might not be a standard HF identifier without API key logic
CHECK_INTERVAL = 300  # Seconds between processing cycles
# --- END: Configuration ---

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    set_start_method('spawn')
except RuntimeError:
    # Method may have been set already
    pass

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)

# Create formatters
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create handlers
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Global variable to track lock files for cleanup
active_lock_files = set()

# Global variables for Whisper model and pipeline
whisper_model = None
whisper_processor = None
whisper_pipeline = None

def initialize_whisper():
    """Initialize Whisper model, processor and pipeline."""
    global whisper_model, whisper_processor, whisper_pipeline
    
    if whisper_model is not None:
        return  # Already initialized
        
    logger.info("Initializing Whisper model and processor...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32

    logger.info(f"Loading model {WHISPER_MODEL}...")
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True if device != "cpu" else False,
        use_safetensors=True
    )
    whisper_model.to(device)
    logger.info("Model loaded successfully")

    logger.info("Loading processor...")
    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
    logger.info("Processor loaded successfully")

    # Create the pipeline
    logger.info("Creating Whisper pipeline...")
    whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16 if device != "cpu" else 4,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    logger.info("Pipeline created successfully")

def cleanup_whisper():
    """Clean up Whisper model resources."""
    global whisper_model, whisper_processor, whisper_pipeline
    
    if whisper_model is not None:
        logger.info("Cleaning up Whisper model resources...")
        # Move model to CPU first to free GPU memory
        if hasattr(whisper_model, 'to'):
            whisper_model.to('cpu')
        del whisper_model
        del whisper_processor
        del whisper_pipeline
        whisper_model = None
        whisper_processor = None
        whisper_pipeline = None
        # Force CUDA cache clear if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Whisper resources cleaned up successfully")

def create_backup():
    """Create a backup of transcriptions and corrected files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("backup", timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    logger.info(f"Creating backup in {backup_dir}")
    
    # Backup transcriptions
    trans_backup = os.path.join(backup_dir, "transcriptions")
    if os.path.exists(TRANSCRIPTIONS_DIR):
        shutil.copytree(TRANSCRIPTIONS_DIR, trans_backup, dirs_exist_ok=True)
        logger.info("Transcriptions backed up successfully")
    
    # Backup corrected files
    corr_backup = os.path.join(backup_dir, "corrected")
    if os.path.exists(CORRECTED_DIR):
        shutil.copytree(CORRECTED_DIR, corr_backup, dirs_exist_ok=True)
        logger.info("Corrected files backed up successfully")
    
    return backup_dir

def cleanup_old_backups(max_backups=5):
    """Remove old backups keeping only the most recent ones."""
    backup_base_dir = "backup"
    if not os.path.exists(backup_base_dir):
        return
    backup_dirs = sorted([d for d in os.listdir(backup_base_dir) if os.path.isdir(os.path.join(backup_base_dir, d))])
    if len(backup_dirs) > max_backups:
        logger.info(f"Cleaning up old backups. Keeping {max_backups} most recent backups.")
        for old_dir in backup_dirs[:-max_backups]:
            shutil.rmtree(os.path.join(backup_base_dir, old_dir))
            logger.info(f"Removed old backup: {old_dir}")

def cleanup_lock_files():
    """Clean up any remaining lock files."""
    # Create a copy of the set to avoid issues if the set is modified during iteration elsewhere (though less likely here)
    locks_to_remove = list(active_lock_files)
    for lock_file in locks_to_remove:
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logger.info(f"Cleaned up lock file: {lock_file}")
            if lock_file in active_lock_files: # Check before removing
                 active_lock_files.remove(lock_file)
        except Exception as e:
            logger.error(f"Error cleaning up lock file {lock_file}: {e}")

def signal_handler(signum, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup_lock_files()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def is_valid_transcription(file_path):
    """Check if transcription is valid (exists and has more than MIN_CHARS)."""
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return len(content.strip()) >= MIN_CHARS # Use strip() to avoid counting only whitespace
    except Exception as e:
        logger.error(f"Error reading transcription {file_path}: {e}")
        return False

def transcribe_file(args):
    """Transcribe an audio file with Whisper using Hugging Face transformers."""
    audio_path, trans_path = args
    lock_file = audio_path + ".lock"
    
    if os.path.exists(lock_file):
        logger.info(f"Skipping transcription for {audio_path}: lock file exists.")
        return
    
    if os.path.exists(trans_path) and is_valid_transcription(trans_path):
        logger.info(f"Skipping transcription for {audio_path}: valid transcription already exists.")
        return
    
    try:
        Path(lock_file).touch()
        active_lock_files.add(lock_file)
    except Exception as e:
        logger.error(f"Error creating lock file for {audio_path}: {e}")
        return
    
    retries = 0
    success = False
    while retries < MAX_RETRIES:
        try:
            logger.info(f"Starting transcription of {audio_path} (attempt {retries + 1}/{MAX_RETRIES})")
            # Ensure Whisper is initialized
            initialize_whisper()
            
            # Create a progress bar for this file
            pbar = tqdm(total=100, desc=f"Transcribing {os.path.basename(audio_path)}", 
                       leave=False, position=1)
            
            # Transcribe using the pipeline
            result = whisper_pipeline(audio_path, generate_kwargs={"max_new_tokens": 256}) 
            transcription = result["text"]
            
            # Update progress bar to 100% when done
            pbar.update(100)
            pbar.close()
            
            ensure_dir(os.path.dirname(trans_path))
            with open(trans_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            if is_valid_transcription(trans_path):
                logger.info(f"Successfully transcribed {audio_path}")
                success = True
                break
            else:
                logger.warning(f"Transcription too short for {audio_path} (length {len(transcription.strip())}), retrying...")
                retries += 1
        except Exception as e:
            logger.error(f"Unexpected error during transcription of {audio_path}: {e}")
            retries += 1
        time.sleep(5)
    
    if not success:
        logger.error(f"Max retries reached for {audio_path}. Transcription failed or was too short.")
    
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
        if lock_file in active_lock_files:
            active_lock_files.remove(lock_file)
    except Exception as e:
        logger.error(f"Error removing lock file for {audio_path}: {e}")

def correct_file(args):
    """Correct a transcription using Ollama API."""
    trans_path, corrected_path = args
    lock_file = trans_path + ".correction.lock" # Use a different lock suffix for correction

    if os.path.exists(lock_file):
        logger.info(f"Skipping correction for {trans_path}: lock file exists.")
        return

    # Skip if corrected file exists (no need to check content length for corrected files, assume correction is good)
    if os.path.exists(corrected_path):
        logger.info(f"Skipping correction for {trans_path}: corrected file already exists.")
        return
    
    if not is_valid_transcription(trans_path): # Don't correct invalid or too short transcriptions
        logger.warning(f"Skipping correction for {trans_path}: source transcription is invalid or too short.")
        return

    try:
        Path(lock_file).touch()
        active_lock_files.add(lock_file)
    except Exception as e:
        logger.error(f"Error creating lock file for correction of {trans_path}: {e}")
        return
    
    try:
        logger.info(f"Starting correction of {trans_path} using Ollama model {OLLAMA_MODEL}")
        with open(trans_path, "r", encoding="utf-8") as f:
            text_to_correct = f.read()
        
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that corrects transcriptions."
                },
                {
                    "role": "user",
                    "content": f"File: {os.path.basename(trans_path)}\n\n{CORRECTION_PROMPT}\n\n---\n\n{text_to_correct}"
                }
            ],
            "options": OLLAMA_OPTIONS,
            "stream": False # We want the full response at once
        }
        
        logger.info(f"Sending request to Ollama API: {OLLAMA_API_URL}")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300) # 5 min timeout
        response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)
        
        response_data = response.json()
        corrected_text = response_data.get("message", {}).get("content", "")
        # Remove <thinking>...</thinking> blocks if present
        corrected_text = re.sub(r'<think>[\s\S]*?</think>', '', corrected_text, flags=re.IGNORECASE)

        if not corrected_text.strip():
            logger.warning(f"Ollama returned empty correction for {trans_path}. Original text: {text_to_correct[:100]}...")
            # Optionally, copy original if correction is empty, or log as error
            # For now, we just won't write an empty file.
            return

        ensure_dir(os.path.dirname(corrected_path))
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(corrected_text.strip())
        logger.info(f"Successfully corrected {trans_path} and saved to {corrected_path}")

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Ollama API connection error for {trans_path}: {e}. Ensure Ollama is running and reachable at {OLLAMA_API_URL}.")
    except requests.exceptions.Timeout:
        logger.error(f"Ollama API request timed out for {trans_path}.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed for {trans_path}: {e}")
        if e.response is not None:
            logger.error(f"Ollama API Response: {e.response.text}")
    except KeyError as e:
        logger.error(f"Unexpected response structure from Ollama for {trans_path}. Missing key: {e}. Response: {response_data}")
    except Exception as e:
        logger.error(f"Unexpected error during correction of {trans_path}: {e}")
    finally:
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
            if lock_file in active_lock_files:
                active_lock_files.remove(lock_file)
        except Exception as e:
            logger.error(f"Error removing correction lock file for {trans_path}: {e}")


def collect_transcription_tasks():
    """Collect audio files needing transcription."""
    tasks = []
    logger.info("Scanning for audio files to transcribe...")
    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            # Add more audio extensions if needed
            if file.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
                audio_path = os.path.join(root, file)
                base_filename, _ = os.path.splitext(file)
                rel_path_dir = os.path.relpath(root, AUDIO_DIR)
                
                # Handle cases where rel_path_dir might be '.' for files directly in AUDIO_DIR
                if rel_path_dir == ".":
                    trans_dir = TRANSCRIPTIONS_DIR
                else:
                    trans_dir = os.path.join(TRANSCRIPTIONS_DIR, rel_path_dir)
                 
                trans_path = os.path.join(trans_dir, base_filename + ".txt")
                
                ensure_dir(os.path.dirname(trans_path)) # Ensure specific subdirectory exists
                
                # Check for lock file first
                lock_file = audio_path + ".lock"
                if os.path.exists(lock_file):
                    logger.info(f"Skipping task for {audio_path}: lock file exists.")
                    continue

                if not (os.path.exists(trans_path) and is_valid_transcription(trans_path)):
                    tasks.append((audio_path, trans_path))
    return tasks

def collect_correction_tasks():
    """Collect transcriptions needing correction."""
    tasks = []
    logger.info("Scanning for transcriptions to correct...")
    for root, _, files in os.walk(TRANSCRIPTIONS_DIR):
        for file in files:
            if file.lower().endswith(".txt"):
                trans_path = os.path.join(root, file)
                rel_path = os.path.relpath(trans_path, TRANSCRIPTIONS_DIR)
                corrected_path = os.path.join(CORRECTED_DIR, rel_path)

                ensure_dir(os.path.dirname(corrected_path)) # Ensure specific subdirectory exists

                # Check for lock file first
                lock_file = trans_path + ".correction.lock"
                if os.path.exists(lock_file):
                    logger.info(f"Skipping task for {trans_path}: correction lock file exists.")
                    continue
                
                # Only add task if corrected file doesn't exist AND source transcription is valid
                if not os.path.exists(corrected_path) and is_valid_transcription(trans_path):
                    tasks.append((trans_path, corrected_path))
    return tasks

def main():
    """Main loop to process audio files in phases."""
    logger.info("Initializing directories...")
    ensure_dir(AUDIO_DIR)
    ensure_dir(TRANSCRIPTIONS_DIR)
    ensure_dir(CORRECTED_DIR)
    ensure_dir("backup")
    
    if not OLLAMA_MODEL or not CORRECTION_PROMPT:
        logger.error("OLLAMA_MODEL and CORRECTION_PROMPT must be set in the configuration.")
        sys.exit(1)

    logger.info(f"Using Ollama model: {OLLAMA_MODEL} via {OLLAMA_API_URL}")

    while True:
        try:
            logger.info("="*50)
            logger.info(f"Starting new processing cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*50)
            
            create_backup()
            cleanup_old_backups()
            
            # Phase 1: Transcribe audio files
            logger.info("-" * 10 + " Phase 1: Transcription " + "-" * 10)
            trans_tasks = collect_transcription_tasks()
            if trans_tasks:
                logger.info(f"Found {len(trans_tasks)} files to transcribe")
                
                num_parallel_transcriptions = 2 # Set to 2 for two Whisper instances

                # Each process in the pool will call transcribe_file, 
                # which in turn calls initialize_whisper().
                # Due to the 'spawn' start method, each process will load its own model instance.
                with tqdm(total=len(trans_tasks), desc=f"Overall Transcription Progress ({num_parallel_transcriptions} workers)", position=0, leave=True) as pbar:
                    # Using imap_unordered to update the progress bar as tasks complete
                    # and to allow tasks to be processed as they are available.
                    with Pool(processes=num_parallel_transcriptions) as pool:
                        for _ in pool.imap_unordered(transcribe_file, trans_tasks):
                            pbar.update(1)
                
                # This cleanup_whisper() call primarily affects the main process.
                # Models loaded by worker processes are cleaned up when those processes terminate.
                # It also calls torch.cuda.empty_cache(), which can be beneficial.
                cleanup_whisper()
            else:
                logger.info("No new audio files to transcribe")
                # Ensure Whisper resources in the main process are cleaned up if they were ever loaded.
                cleanup_whisper()
            
            # Phase 2: Correct transcriptions
            logger.info("-" * 10 + " Phase 2: Correction " + "-" * 10)
            correct_tasks = collect_correction_tasks()
            if correct_tasks:
                logger.info(f"Found {len(correct_tasks)} transcriptions to correct")
                with Pool(processes=1) as pool:
                    pool.map(correct_file, correct_tasks)
            else:
                logger.info("No new transcriptions to correct")
            
            logger.info(f"Cycle complete. Sleeping for {CHECK_INTERVAL} seconds.")
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.critical(f"Critical error in main loop: {e}", exc_info=True)
            logger.info("Performing emergency cleanup of lock files due to critical error.")
            cleanup_lock_files()
            cleanup_whisper()  # Also clean up Whisper resources on error
            logger.info(f"Sleeping for 60 seconds before attempting to restart loop...")
            time.sleep(60)

if __name__ == "__main__":
    try:
        logger.info("Starting TransFixer (Ollama Edition)...")
        main()
    except KeyboardInterrupt:
        logger.info("Script terminated by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Script terminated due to an unhandled error: {e}", exc_info=True)
    finally:
        logger.info("Performing final cleanup of lock files...")
        cleanup_lock_files()
        cleanup_whisper()  # Clean up Whisper resources on exit
        logger.info("Exiting TransFixer.")
        sys.exit(0)