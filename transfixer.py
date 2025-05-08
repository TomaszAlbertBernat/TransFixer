import os
import subprocess
import shutil
import logging
from multiprocessing import Pool
import requests
import time
from pathlib import Path
import signal
import sys
from datetime import datetime
from config import *

# Configuration
AUDIO_DIR = "audio"
TRANSCRIPTIONS_DIR = "transcriptions"
CORRECTED_DIR = "corrected"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "transcription_errors.log")
MIN_CHARS = 50
MAX_RETRIES = 3
WHISPER_MODEL = "base"  # Use smaller model for less GPU usage (e.g., "tiny" or "base")
LM_STUDIO_API = "http://localhost:1234/v1/chat/completions"
CHECK_INTERVAL = 300  # Seconds between processing cycles

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global variable to track lock files for cleanup
active_lock_files = set()

def create_backup():
    """Create a backup of transcriptions and corrected files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("backup", timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup transcriptions
    trans_backup = os.path.join(backup_dir, "transcriptions")
    if os.path.exists(TRANSCRIPTIONS_DIR):
        shutil.copytree(TRANSCRIPTIONS_DIR, trans_backup, dirs_exist_ok=True)
    
    # Backup corrected files
    corr_backup = os.path.join(backup_dir, "corrected")
    if os.path.exists(CORRECTED_DIR):
        shutil.copytree(CORRECTED_DIR, corr_backup, dirs_exist_ok=True)
    
    logging.info(f"Created backup in {backup_dir}")
    return backup_dir

def cleanup_old_backups(max_backups=5):
    """Remove old backups keeping only the most recent ones."""
    backup_dirs = sorted([d for d in os.listdir("backup") if os.path.isdir(os.path.join("backup", d))])
    if len(backup_dirs) > max_backups:
        for old_dir in backup_dirs[:-max_backups]:
            shutil.rmtree(os.path.join("backup", old_dir))
            logging.info(f"Removed old backup: {old_dir}")

def cleanup_lock_files():
    """Clean up any remaining lock files."""
    for lock_file in active_lock_files:
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logging.info(f"Cleaned up lock file: {lock_file}")
        except Exception as e:
            logging.error(f"Error cleaning up lock file {lock_file}: {e}")

def signal_handler(signum, frame):
    """Handle termination signals."""
    logging.info(f"Received signal {signum}, cleaning up...")
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
            return len(content) > MIN_CHARS
    except Exception as e:
        logging.error(f"Error reading transcription {file_path}: {e}")
        return False

def transcribe_file(args):
    """Transcribe an audio file with Whisper-Turbo."""
    audio_path, trans_path = args
    lock_file = audio_path + ".lock"
    
    # Skip if valid transcription exists or file is being processed
    if os.path.exists(lock_file) or (os.path.exists(trans_path) and is_valid_transcription(trans_path)):
        logging.info(f"Skipping transcription for {audio_path}: already processed or locked")
        return
    
    # Create lock file
    try:
        open(lock_file, "w").close()
        active_lock_files.add(lock_file)
    except Exception as e:
        logging.error(f"Error creating lock file for {audio_path}: {e}")
        return
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Run Whisper-Turbo with specified model
            result = subprocess.run(
                ["whisper-turbo", audio_path, "--model", WHISPER_MODEL, "--output", trans_path],
                check=True,
                capture_output=True,
                text=True
            )
            if is_valid_transcription(trans_path):
                logging.info(f"Successfully transcribed {audio_path}")
                break
            else:
                logging.warning(f"Transcription too short for {audio_path}, retrying...")
                retries += 1
        except subprocess.CalledProcessError as e:
            logging.error(f"Transcription failed for {audio_path}: {e.stderr}")
            retries += 1
        except Exception as e:
            logging.error(f"Unexpected error during transcription of {audio_path}: {e}")
            retries += 1
    
    if retries >= MAX_RETRIES:
        logging.error(f"Max retries reached for {audio_path}")
    
    # Remove lock file
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
            active_lock_files.remove(lock_file)
    except Exception as e:
        logging.error(f"Error removing lock file for {audio_path}: {e}")

def correct_file(args):
    """Correct a transcription using LM Studio API."""
    trans_path, corrected_path = args
    lock_file = trans_path + ".lock"
    
    # Skip if corrected file exists or file is being processed
    if os.path.exists(lock_file) or os.path.exists(corrected_path):
        logging.info(f"Skipping correction for {trans_path}: already processed or locked")
        return
    
    # Create lock file
    try:
        open(lock_file, "w").close()
        active_lock_files.add(lock_file)
    except Exception as e:
        logging.error(f"Error creating lock file for {trans_path}: {e}")
        return
    
    try:
        with open(trans_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        payload = {
            "model": "local-model",
            "messages": [{"role": "user", "content": f"{CORRECTION_PROMPT}\n\n{text}"}],
            **LM_STUDIO_PARAMS
        }
        
        response = requests.post(LM_STUDIO_API, json=payload, timeout=300)
        response.raise_for_status()
        corrected_text = response.json()["choices"][0]["message"]["content"]
        
        # Save corrected transcription
        ensure_dir(os.path.dirname(corrected_path))
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(corrected_text)
        logging.info(f"Successfully corrected {trans_path}")
    except requests.RequestException as e:
        logging.error(f"Correction failed for {trans_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during correction of {trans_path}: {e}")
    finally:
        # Remove lock file
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                active_lock_files.remove(lock_file)
        except Exception as e:
            logging.error(f"Error removing lock file for {trans_path}: {e}")

def collect_transcription_tasks():
    """Collect audio files needing transcription."""
    tasks = []
    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith(".mp3"):
                audio_path = os.path.join(root, file)
                rel_path = os.path.relpath(audio_path, AUDIO_DIR)
                trans_path = os.path.join(TRANSCRIPTIONS_DIR, rel_path.replace(".mp3", ".txt"))
                ensure_dir(os.path.dirname(trans_path))
                if not (os.path.exists(trans_path) and is_valid_transcription(trans_path)):
                    tasks.append((audio_path, trans_path))
    return tasks

def collect_correction_tasks():
    """Collect transcriptions needing correction."""
    tasks = []
    for root, _, files in os.walk(TRANSCRIPTIONS_DIR):
        for file in files:
            if file.endswith(".txt"):
                trans_path = os.path.join(root, file)
                rel_path = os.path.relpath(trans_path, TRANSCRIPTIONS_DIR)
                corrected_path = os.path.join(CORRECTED_DIR, rel_path)
                if not os.path.exists(corrected_path) and is_valid_transcription(trans_path):
                    ensure_dir(os.path.dirname(corrected_path))
                    tasks.append((trans_path, corrected_path))
    return tasks

def main():
    """Main loop to process audio files in phases."""
    ensure_dir(AUDIO_DIR)
    ensure_dir(TRANSCRIPTIONS_DIR)
    ensure_dir(CORRECTED_DIR)
    ensure_dir("backup")
    
    while True:
        try:
            logging.info("Starting processing cycle")
            
            # Create backup before processing
            create_backup()
            cleanup_old_backups()
            
            # Phase 1: Transcribe audio files
            trans_tasks = collect_transcription_tasks()
            if trans_tasks:
                logging.info(f"Found {len(trans_tasks)} files to transcribe")
                with Pool(processes=1) as pool:  # Single process for GPU efficiency
                    pool.map(transcribe_file, trans_tasks)
            else:
                logging.info("No files to transcribe")
            
            # Phase 2: Correct transcriptions
            correct_tasks = collect_correction_tasks()
            if correct_tasks:
                logging.info(f"Found {len(correct_tasks)} transcriptions to correct")
                with Pool(processes=1) as pool:  # Single process for API stability
                    pool.map(correct_file, correct_tasks)
            else:
                logging.info("No transcriptions to correct")
            
            logging.info(f"Cycle complete, sleeping for {CHECK_INTERVAL} seconds")
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            cleanup_lock_files()
            time.sleep(60)  # Wait a minute before retrying after an error

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Script terminated by user")
        cleanup_lock_files()
        exit(0)
    except Exception as e:
        logging.error(f"Script terminated due to error: {e}")
        cleanup_lock_files()
        exit(1)