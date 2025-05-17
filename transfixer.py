import os
import subprocess
import shutil
import logging
import ray
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

# Initialize Ray
ray.init(ignore_reinit_error=True)

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

@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
class WhisperWorker:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() and self.device != "cpu" else torch.float32
        self.model = None
        self.processor = None
        self.pipeline = None
        self.initialize()

    def initialize(self):
        """Initialize Whisper model, processor and pipeline."""
        logger.info("Initializing Whisper model and processor...")
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading model {WHISPER_MODEL}...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True if self.device != "cpu" else False,
            use_safetensors=True
        )
        self.model.to(self.device)
        logger.info("Model loaded successfully")

        logger.info("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
        logger.info("Processor loaded successfully")

        # Create the pipeline
        logger.info("Creating Whisper pipeline...")
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16 if self.device != "cpu" else 4,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        logger.info("Pipeline created successfully")

    def transcribe(self, audio_path):
        """Transcribe audio file using Whisper."""
        try:
            result = self.pipeline(audio_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return None

    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            self.model.to('cpu')
            del self.model
            del self.processor
            del self.pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

@ray.remote
def process_audio_file(audio_path, worker):
    """Process a single audio file using Ray."""
    try:
        # Create lock file
        lock_file = f"{audio_path}.lock"
        if os.path.exists(lock_file):
            return None
        
        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))
        active_lock_files.add(lock_file)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        transcription_path = os.path.join(TRANSCRIPTIONS_DIR, f"{base_name}.txt")
        
        # Skip if already transcribed
        if is_valid_transcription(transcription_path):
            return None
            
        # Transcribe using Ray worker
        transcription = ray.get(worker.transcribe.remote(audio_path))
        if not transcription:
            return None
            
        # Save transcription
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(transcription)
            
        return transcription_path
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return None
    finally:
        # Cleanup lock file
        if os.path.exists(lock_file):
            os.remove(lock_file)
        if lock_file in active_lock_files:
            active_lock_files.remove(lock_file)

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

def main():
    """Main function to process audio files."""
    try:
        # Create necessary directories
        for directory in [AUDIO_DIR, TRANSCRIPTIONS_DIR, CORRECTED_DIR, LOG_DIR]:
            ensure_dir(directory)
            
        # Create backup before processing
        backup_dir = create_backup()
        logger.info(f"Created backup in {backup_dir}")
        
        # Initialize Ray workers
        num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
        workers = [WhisperWorker.remote() for _ in range(num_workers)]
        
        while True:
            try:
                # Get list of audio files
                audio_files = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR)
                             if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))]
                
                if not audio_files:
                    logger.info("No new audio files to process")
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Process files in parallel using Ray
                futures = []
                for audio_file in audio_files:
                    worker = workers[len(futures) % num_workers]  # Round-robin worker assignment
                    futures.append(process_audio_file.remote(audio_file, worker))
                
                # Wait for all tasks to complete
                results = ray.get(futures)
                
                # Process results
                for result in results:
                    if result:
                        logger.info(f"Successfully processed: {result}")
                
                # Cleanup old backups
                cleanup_old_backups()
                
                time.sleep(CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(CHECK_INTERVAL)
                
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, cleaning up...")
    finally:
        # Cleanup Ray workers
        for worker in workers:
            ray.get(worker.cleanup.remote())
        ray.shutdown()
        cleanup_lock_files()

if __name__ == "__main__":
    main()