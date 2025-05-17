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
from config import OLLAMA_MODEL, CORRECTION_PROMPT, OLLAMA_API_URL, OLLAMA_OPTIONS, NUM_PARALLEL_WHISPER
from tqdm import tqdm
import re
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TransFixer - Audio Transcription and Correction System')
    parser.add_argument('--workers', type=int, default=NUM_PARALLEL_WHISPER,
                      help=f'Number of parallel Whisper instances (default: {NUM_PARALLEL_WHISPER})')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Number of audio files to process in a single batch (default: 4)')
    parser.add_argument('--skip-transcribing', action='store_true',
                      help='Skip the transcription phase and proceed directly to correction')
    parser.add_argument('--use-batching', action='store_true',
                      help='Use batched processing instead of multiple workers (more memory efficient)')
    return parser.parse_args()

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

def initialize_whisper(batch_size=None):
    """
    Initialize Whisper model, processor and pipeline.
    
    Args:
        batch_size: Optional batch size to use for the pipeline
    """
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
    
    # Apply torch.compile for faster inference if available (requires PyTorch 2.0+)
    # Note: torch.compile is not compatible with chunked processing
    use_compile = False  # Set to True if you prefer speed over handling very long audio files
    
    if use_compile and hasattr(torch, 'compile') and device != "cpu":
        try:
            logger.info("Applying torch.compile() to model for faster inference...")
            whisper_model = torch.compile(whisper_model)
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Could not compile model with torch.compile(): {e}")
    
    logger.info("Model loaded successfully")

    logger.info("Loading processor...")
    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
    logger.info("Processor loaded successfully")

    # Determine batch size for pipeline
    if batch_size is None:
        # Default batch sizes based on device
        pipeline_batch_size = 16 if device != "cpu" else 4
    else:
        pipeline_batch_size = batch_size
    
    logger.info(f"Creating Whisper pipeline with batch_size={pipeline_batch_size}...")
    
    # Create the pipeline with optimized parameters
    whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        chunk_length_s=30,
        batch_size=pipeline_batch_size,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        # Additional parameters to improve GPU utilization
        generate_kwargs={
            "max_new_tokens": 256,
            "do_sample": False,
            "use_cache": True
        }
    )
    
    # Set chunk length based on available memory
    if device != "cpu":
        try:
            # Get available GPU memory and set chunk size proportionally
            total_mem = torch.cuda.get_device_properties(device).total_memory
            allocated_mem = torch.cuda.memory_allocated(device)
            available_mem = total_mem - allocated_mem
            
            # Log memory information in GB for better readability
            logger.info(f"Total GPU memory: {total_mem / (1024**3):.2f}GB")
            logger.info(f"Available GPU memory: {available_mem / (1024**3):.2f}GB")
            
            # Use larger chunks if we have more memory available
            if available_mem > 6 * 1024 * 1024 * 1024:  # > 6GB free
                whisper_pipeline.model.config.forced_decoder_ids = None  # Allow model to decide
                logger.info("Using optimized settings for high-memory GPU")
            elif available_mem < 2 * 1024 * 1024 * 1024:  # < 2GB free
                # Use smaller chunks and more aggressive memory saving
                whisper_pipeline.chunk_length_s = 15
                logger.info("Using conservative settings for low-memory GPU")
        except Exception as e:
            logger.warning(f"Error optimizing for GPU memory: {e}")
    
    logger.info("Pipeline created successfully")
    
    # Return batch size actually used
    return pipeline_batch_size

def cleanup_whisper():
    """Clean up Whisper model resources."""
    global whisper_model, whisper_processor, whisper_pipeline
    
    if whisper_model is not None:
        logger.info("Cleaning up Whisper model resources...")
        try:
            # Move model to CPU first to free GPU memory
            if hasattr(whisper_model, 'to') and torch.cuda.is_available():
                whisper_model.to('cpu')
                
            # Delete model components
            del whisper_model
            del whisper_processor
            del whisper_pipeline
            whisper_model = None
            whisper_processor = None
            whisper_pipeline = None
            
            # Force CUDA cache clear if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Additional memory cleanup
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                allocated_mem = torch.cuda.memory_allocated(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory
                available_mem = total_mem - allocated_mem
                logger.info(f"GPU memory after cleanup: {allocated_mem / (1024**3):.2f}GB used, {available_mem / (1024**3):.2f}GB available")
                
            logger.info("Whisper resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during Whisper cleanup: {e}")

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

def transcribe_batch(batch_tasks, batch_size=4):
    """
    Transcribe a batch of audio files using a single Whisper model instance.
    
    Args:
        batch_tasks: List of (audio_path, trans_path) tuples
        batch_size: Size of batches to process
        
    Returns:
        List of results with success/failure information
    """
    results = []
    
    # Skip if batch is empty
    if not batch_tasks:
        return results
    
    # Create lock files for all tasks in the batch
    lock_files = []
    valid_tasks = []
    
    for audio_path, trans_path in batch_tasks:
        lock_file = audio_path + ".lock"
        
        # Skip if lock file exists or transcription already exists
        if os.path.exists(lock_file):
            logger.info(f"Skipping transcription for {audio_path}: lock file exists.")
            continue
            
        if os.path.exists(trans_path) and is_valid_transcription(trans_path):
            logger.info(f"Skipping transcription for {audio_path}: valid transcription already exists.")
            continue
        
        try:
            Path(lock_file).touch()
            active_lock_files.add(lock_file)
            lock_files.append(lock_file)
            valid_tasks.append((audio_path, trans_path))
        except Exception as e:
            logger.error(f"Error creating lock file for {audio_path}: {e}")
            continue
    
    if not valid_tasks:
        return results
    
    try:
        # Initialize Whisper model with proper batch size
        initialize_whisper(batch_size=batch_size)
        
        # Get all audio paths and create a mapping to trans_paths
        audio_paths = [task[0] for task in valid_tasks]
        trans_path_map = {task[0]: task[1] for task in valid_tasks}
        
        logger.info(f"Processing batch of {len(audio_paths)} files at once with batch_size={batch_size}")
        
        with tqdm(total=len(valid_tasks), desc="Batch transcription", leave=False, position=1) as batch_pbar:
            # Log memory usage before batch
            if torch.cuda.is_available():
                before_mem = torch.cuda.memory_allocated() / (1024**2)
                logger.info(f"GPU memory before batch: {before_mem:.2f}MB")
            
            try:
                # Process all files in a single call - this is the key change
                # The pipeline will handle batching internally based on batch_size
                batch_results = whisper_pipeline(
                    audio_paths,
                    batch_size=batch_size,
                    generate_kwargs={
                        "max_new_tokens": 256,
                        "do_sample": False,
                        "use_cache": True 
                    }
                )
                
                # Log memory usage after batch
                if torch.cuda.is_available():
                    after_mem = torch.cuda.memory_allocated() / (1024**2)
                    logger.info(f"GPU memory after batch: {after_mem:.2f}MB")
                    logger.info(f"Batch memory delta: {after_mem - before_mem:.2f}MB")
                
                # Handle results
                for i, (audio_path, result) in enumerate(zip(audio_paths, batch_results)):
                    trans_path = trans_path_map[audio_path]
                    transcription = result["text"]
                    
                    # Save transcription
                    ensure_dir(os.path.dirname(trans_path))
                    with open(trans_path, "w", encoding="utf-8") as f:
                        f.write(transcription)
                    
                    if is_valid_transcription(trans_path):
                        logger.info(f"Successfully transcribed {audio_path}")
                        results.append((audio_path, True, None))
                    else:
                        logger.warning(f"Transcription too short for {audio_path} (length {len(transcription.strip())})")
                        results.append((audio_path, False, "Transcription too short"))
                    
                    batch_pbar.update(1)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Mark all files in the failed batch as failed
                for audio_path, _ in valid_tasks:
                    results.append((audio_path, False, str(e)))
                    batch_pbar.update(1)
    
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
    finally:
        # Clean up all lock files
        for lock_file in lock_files:
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                if lock_file in active_lock_files:
                    active_lock_files.remove(lock_file)
            except Exception as e:
                logger.error(f"Error removing lock file {lock_file}: {e}")
    
    return results

def main():
    """Main loop to process audio files in phases."""
    # Parse command line arguments
    args = parse_arguments()
    num_workers = args.workers
    batch_size = args.batch_size
    use_batching = args.use_batching
    skip_transcribing = args.skip_transcribing
    
    # Calculate optimal batch size based on VRAM if using batching
    if use_batching and torch.cuda.is_available():
        try:
            # Simple heuristic: 
            # Whisper v3 turbo needs ~5GB for a single stream
            # Leave at least 2GB buffer for other operations
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            # Better method to estimate available memory
            # First run a small allocation to initialize CUDA context
            torch.cuda.empty_cache()
            dummy = torch.ones(1).cuda()
            del dummy
            torch.cuda.empty_cache()
            
            # Now get allocated memory after context initialization
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
            # Available memory is total minus what's already allocated
            available_mem = total_mem - allocated_mem
            
            logger.info(f"GPU memory: total={total_mem:.1f}GB, allocated={allocated_mem:.1f}GB, available={available_mem:.1f}GB")
            
            # Start with available memory, subtract buffer for safety
            usable_mem = available_mem - 1  # Reduce buffer from 2GB to 1GB
            # Reduce per-stream estimate from 5GB to 2.5GB based on observed usage
            mem_per_stream = 2.5
            max_streams = max(1, int(usable_mem / mem_per_stream))
            
            # Adjust batch size if needed, don't go below 1
            suggested_batch_size = max(1, min(batch_size, max_streams))
            
            if suggested_batch_size != batch_size:
                logger.info(f"Adjusting batch size from {batch_size} to {suggested_batch_size} based on available VRAM")
                batch_size = suggested_batch_size
            else:
                logger.info(f"Using requested batch size of {batch_size}")
            
            logger.info(f"Memory estimate: {mem_per_stream}GB per stream, expecting to use ~{mem_per_stream * batch_size:.1f}GB VRAM")
        except Exception as e:
            logger.warning(f"Could not calculate optimal batch size: {e}")
    
    if use_batching:
        logger.info(f"Starting with batch processing (batch size: {batch_size})")
    else:
        logger.info(f"Starting with {num_workers} parallel Whisper instances")
        
    if skip_transcribing:
        logger.info("Transcription phase will be skipped")
    
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
            
            # Phase 1: Transcribe audio files (skip if --skip-transcribing is used)
            if not skip_transcribing:
                logger.info("-" * 10 + " Phase 1: Transcription " + "-" * 10)
                trans_tasks = collect_transcription_tasks()
                if trans_tasks:
                    logger.info(f"Found {len(trans_tasks)} files to transcribe")
                    
                    if use_batching:
                        # Batch processing approach - single model instance processes multiple files
                        with tqdm(total=len(trans_tasks), desc=f"Overall Transcription Progress (batch size: {batch_size})", position=0, leave=True) as pbar:
                            # Process all files in appropriate batches
                            for i in range(0, len(trans_tasks), batch_size*2):  # Process in larger chunks to avoid frequent model reloading
                                large_batch = trans_tasks[i:i+batch_size*2]
                                results = transcribe_batch(large_batch, batch_size=batch_size)
                                pbar.update(len(large_batch))
                                
                                # Clear CUDA cache between large batches
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        
                        # Clean up resources after all batches
                        cleanup_whisper()
                    else:
                        # Original approach - multiple worker processes, each with its own model instance
                        with tqdm(total=len(trans_tasks), desc=f"Overall Transcription Progress ({num_workers} workers)", position=0, leave=True) as pbar:
                            with Pool(processes=num_workers) as pool:
                                for _ in pool.imap_unordered(transcribe_file, trans_tasks):
                                    pbar.update(1)
                        
                        # Clean up resources
                        cleanup_whisper()
                else:
                    logger.info("No new audio files to transcribe")
                    cleanup_whisper()
            else:
                logger.info("Skipping transcription phase as requested")
            
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