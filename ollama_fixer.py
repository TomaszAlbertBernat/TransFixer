import os
import logging
import time
import re
import requests
import argparse
import threading
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- START: Configuration ---
TRANSCRIPTIONS_DIR = "transcriptions"
CORRECTED_DIR = "corrected"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "ollama_fixer_errors.log")
MIN_CHARS = 50
MAX_RETRIES = 3
CHECK_INTERVAL = 300
BACKUP_DIR = "backup"
MAX_BACKUPS = 1

# Ollama Configuration
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi4-mini-reasoning"
OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "max_tokens": 1024
}

# Correction Prompt
CORRECTION_PROMPT = """Please correct the following transcription text for any spelling mistakes, grammatical errors, punctuation issues, or transcription artifacts. 
Maintain the original meaning and flow of the text. Return only the corrected text without additional commentary or explanation."""
# --- END: Configuration ---

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

# Global flag for verbose logging
verbose_logging = False

# Global shutdown event
shutdown_event = threading.Event()

def log_verbose(message, level=logging.INFO):
    """Log a message only if verbose logging is enabled."""
    if verbose_logging:
        logger.log(level, message)

def log_essential(message, level=logging.INFO):
    """Log essential messages that should always be shown."""
    if verbose_logging:
        logger.log(level, message)
    else:
        if level >= logging.WARNING:
            logger.log(level, message)
        else:
            print(message)

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

def correct_file(args):
    """Correct a transcription using Ollama API."""
    trans_path, corrected_path = args
    lock_file = trans_path + ".correction.lock"

    if os.path.exists(lock_file):
        log_verbose(f"Skipping correction for {trans_path}: lock file exists.")
        return

    if os.path.exists(corrected_path):
        log_verbose(f"Skipping correction for {trans_path}: corrected file already exists.")
        return
    
    if not is_valid_transcription(trans_path):
        logger.warning(f"Skipping correction for {trans_path}: source transcription is invalid or too short.")
        return

    try:
        Path(lock_file).touch()
    except Exception as e:
        logger.error(f"Error creating lock file for correction of {trans_path}: {e}")
        return
    
    try:
        log_essential(f"✏️  Correcting: {os.path.basename(trans_path)}")
        with open(trans_path, "r", encoding="utf-8") as f:
            text_to_correct = f.read()
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that corrects transcriptions."},
                {"role": "user", "content": f"File: {os.path.basename(trans_path)}\n\n{CORRECTION_PROMPT}\n\n---\n\n{text_to_correct}"}
            ],
            "options": OLLAMA_OPTIONS,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        response_data = response.json()
        corrected_text = response_data.get("message", {}).get("content", "")
        corrected_text = re.sub(r'<think>[\s\S]*?</think>', '', corrected_text, flags=re.IGNORECASE)
        if not corrected_text.strip():
            logger.warning(f"Ollama returned empty correction for {trans_path}.")
            return # Do not write empty file
        ensure_dir(os.path.dirname(corrected_path))
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(corrected_text.strip())
        log_verbose(f"✅ Successfully corrected {os.path.basename(trans_path)}")

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Ollama API connection error for {trans_path}: {e}.")
    except requests.exceptions.Timeout:
        logger.error(f"Ollama API request timed out for {trans_path}.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed for {trans_path}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Ollama API Response: {e.response.text}")
    except KeyError as e:
        logger.error(f"Unexpected response structure from Ollama for {trans_path}. Missing key: {e}.")
    except Exception as e:
        logger.error(f"Unexpected error during correction of {trans_path}: {e}")
    finally:
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception as e:
            logger.error(f"Error removing correction lock file for {trans_path}: {e}")

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

def cleanup_lock_files():
    """Clean up any remaining .correction.lock files by scanning directories."""
    log_verbose("Scanning for and removing orphaned correction lock files...")
    found_locks = 0

    # Scan for correction locks (.correction.lock) associated with files in TRANSCRIPTIONS_DIR
    if os.path.exists(TRANSCRIPTIONS_DIR):
        for root, _, files in os.walk(TRANSCRIPTIONS_DIR):
            for file_name in files:
                # Check for .correction.lock files that are siblings to non-lock files (e.g., .txt files)
                if not file_name.endswith(".lock") and not file_name.endswith(".correction.lock"):
                    lock_file_path = os.path.join(root, file_name + ".correction.lock")
                    if os.path.exists(lock_file_path):
                        try:
                            os.remove(lock_file_path)
                            log_verbose(f"Cleaned up orphaned correction lock file: {lock_file_path}")
                            found_locks += 1
                        except Exception as e:
                            logger.error(f"Error cleaning up lock file {lock_file_path}: {e}")
    
    if found_locks == 0:
        logger.info("No orphaned lock files found during scan.")
    else:
        logger.info(f"Cleaned up {found_locks} orphaned lock files.")

def create_backup():
    """Create a backup of corrected files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(BACKUP_DIR, timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    log_verbose(f"Creating backup in {backup_dir}")
    
    # Backup corrected files
    corr_backup = os.path.join(backup_dir, "corrected")
    if os.path.exists(CORRECTED_DIR):
        import shutil
        shutil.copytree(CORRECTED_DIR, corr_backup, dirs_exist_ok=True)
        log_verbose("Corrected files backed up successfully")
    
    return backup_dir

def cleanup_old_backups(max_backups=MAX_BACKUPS):
    """Remove old backups keeping only the most recent ones."""
    backup_base_dir = BACKUP_DIR
    if not os.path.exists(backup_base_dir):
        return
    backup_dirs = sorted([d for d in os.listdir(backup_base_dir) if os.path.isdir(os.path.join(backup_base_dir, d))])
    if len(backup_dirs) > max_backups:
        log_verbose(f"Cleaning up old backups. Keeping {max_backups} most recent backups.")
        import shutil
        for old_dir in backup_dirs[:-max_backups]:
            shutil.rmtree(os.path.join(backup_base_dir, old_dir))
            log_verbose(f"Removed old backup: {old_dir}")

def handle_shutdown(signum=None, frame=None):
    """Handle shutdown signals."""
    logger.info("Shutdown signal received, initiating graceful shutdown...")
    shutdown_event.set()

def main(verbose_logging_arg=False, model_arg=None, create_backups=True):
    global verbose_logging, OLLAMA_MODEL
    verbose_logging = verbose_logging_arg
    
    # Override configuration with command line arguments
    if model_arg:
        OLLAMA_MODEL = model_arg
        log_verbose(f"Model overridden via command line: {OLLAMA_MODEL}")

    # Register signal handlers for graceful shutdown
    import signal
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, handle_shutdown)

    # Create required directories
    ensure_dir(TRANSCRIPTIONS_DIR)
    ensure_dir(CORRECTED_DIR)
    ensure_dir(LOG_DIR)
    ensure_dir(BACKUP_DIR)

    # Main processing loop
    while not shutdown_event.is_set():
        try:
            # Collect tasks
            correction_tasks = collect_correction_tasks()

            if not correction_tasks:
                log_essential("No new transcriptions found to correct. Waiting for new files...")
                # Wait with ability to respond to shutdown
                timeout_reached = not shutdown_event.wait(timeout=CHECK_INTERVAL)
                if not timeout_reached:  # If wait was interrupted by shutdown
                    break
                continue

            # Process correction tasks
            log_essential(f"Found {len(correction_tasks)} files to correct")
            
            # Process corrections sequentially to avoid overwhelming Ollama
            for task in tqdm(correction_tasks, desc="Correcting transcriptions", unit="file"):
                if shutdown_event.is_set():
                    break
                correct_file(task)

            # Create backup after processing
            if create_backups:
                try:
                    backup_dir = create_backup()
                    log_verbose(f"Created backup in {backup_dir}")
                    
                    # Clean up old backups
                    cleanup_old_backups()
                except Exception as e:
                    logger.error(f"Error creating backup: {e}")

            # Wait before next check
            timeout_reached = not shutdown_event.wait(timeout=CHECK_INTERVAL)
            if not timeout_reached:  # If wait was interrupted by shutdown
                break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, initiating shutdown...")
            shutdown_event.set()
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            # Wait before retrying
            timeout_reached = not shutdown_event.wait(timeout=CHECK_INTERVAL)
            if not timeout_reached:  # If wait was interrupted by shutdown
                break

    # Final cleanup
    cleanup_lock_files()
    logger.info("Ollama Fixer shutdown complete")

if __name__ == "__main__":
    try:
        log_essential("🚀 Starting Ollama Fixer...")
        parser = argparse.ArgumentParser(description="Ollama Fixer: Correct transcription files using Ollama API.")
        parser.add_argument("--cleanup-locks", action="store_true", help="Remove all correction lock files and exit. Use this if processing was interrupted.")
        parser.add_argument("--verbose-logging", action="store_true", help="Enable verbose logging output.")
        parser.add_argument("--model", type=str, default=None, help="Ollama model to use (default: phi4-mini-reasoning)")
        parser.add_argument("--no-backups", action="store_true", help="Disable automatic backups")
        parser.add_argument("--interval", type=int, default=CHECK_INTERVAL, help=f"Check interval in seconds (default: {CHECK_INTERVAL})")
        parser.add_argument("--test-ollama", action="store_true", help="Test Ollama API connection and exit.")
        
        args = parser.parse_args()
        
        # Override CHECK_INTERVAL using the module's globals() instead of global declaration
        if args.interval != CHECK_INTERVAL:
            # This directly modifies the module-level variable without using global keyword
            globals()['CHECK_INTERVAL'] = args.interval
            log_essential(f"Check interval set to {CHECK_INTERVAL} seconds")
        
        # Handle cleanup locks option
        if args.cleanup_locks:
            log_essential("🧹 Cleaning up correction lock files...")
            cleanup_lock_files()
            log_essential("✅ Lock file cleanup completed. You can now run Ollama Fixer normally.")
            exit(0)
        
        # Test Ollama API connection
        if args.test_ollama:
            log_essential(f"Testing connection to Ollama API at {OLLAMA_API_URL}...")
            try:
                payload = {
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Respond with the word 'CONNECTED' if you can read this message."}
                    ],
                    "options": {"temperature": 0.0, "max_tokens": 10},
                    "stream": False
                }
                response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
                response.raise_for_status()
                response_data = response.json()
                content = response_data.get("message", {}).get("content", "")
                
                if "CONNECTED" in content.upper():
                    log_essential(f"✅ Successfully connected to Ollama model: {OLLAMA_MODEL}")
                else:
                    log_essential(f"⚠️ Connected to Ollama but received unexpected response: {content}")
                
                exit(0)
            except Exception as e:
                log_essential(f"❌ Failed to connect to Ollama: {e}")
                exit(1)
        
        # Run main application
        main(args.verbose_logging, args.model, not args.no_backups)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in __main__, requesting graceful shutdown...")
        shutdown_event.set()
    except SystemExit:
        pass
    except Exception as e:
        logger.error(f"Unhandled exception in __main__: {e}", exc_info=True)
    finally:
        logger.info("Ollama Fixer completed.") 