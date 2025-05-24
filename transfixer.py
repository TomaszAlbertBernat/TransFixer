import os
import subprocess
import shutil
import logging
from multiprocessing import Pool, set_start_method
import multiprocessing
import requests # For Ollama API calls
import time
from pathlib import Path
import signal
import sys
from datetime import datetime, timedelta
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from torch.cuda.amp import autocast  # Added for automatic mixed precision
from config import (
    OLLAMA_MODEL, CORRECTION_PROMPT, OLLAMA_API_URL, OLLAMA_OPTIONS,
    GPU_MEMORY_FRACTION, CONSERVATIVE_BATCH_SIZING, ENABLE_MIXED_PRECISION, 
    SMART_GPU_SELECTION, DEFAULT_CHUNK_LENGTH, MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH,
    MEMORY_SAFETY_FACTOR, MAX_BATCH_SIZE, PERFORMANCE_MODE
)
from tqdm import tqdm
import re
import psutil
import GPUtil
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError
import threading
import argparse # Added argparse

# Global shutdown event
shutdown_event = threading.Event()

def is_wsl():
    """Check if running in WSL (Windows Subsystem for Linux)."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower() or 'wsl' in f.read().lower()
    except:
        return False

def force_exit_wsl():
    """Force exit in WSL by killing the process tree."""
    if is_wsl():
        try:
            # In WSL, try to kill the entire process tree
            os.system(f"pkill -f {os.path.basename(__file__)}")
        except:
            pass

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

# Global variables for model caching
model_cache = {
    'model': None,
    'processor': None,
    'pipeline': None,
    'last_used': None,
    'device': None,
    'lock': threading.Lock()
}

MODEL_CACHE_TIMEOUT = 3600  # 60 minutes
MIN_MEMORY_THRESHOLD = 0.8  # 80% memory usage threshold

def should_unload_model():
    """Determine if the model should be unloaded based on system resources."""
    resources = get_system_resources()
    
    # Check memory usage
    if resources['memory_percent'] > MIN_MEMORY_THRESHOLD * 100:
        logger.info("Memory usage high, unloading model")
        return True
    
    # Check GPU memory if available
    if resources['gpu_info']:
        gpu = resources['gpu_info'][0]
        if gpu['memory_used'] / gpu['memory_total'] > MIN_MEMORY_THRESHOLD:
            logger.info("GPU memory usage high, unloading model")
            return True
    
    return False

def is_model_cache_valid():
    """Check if the cached model is still valid."""
    if model_cache['model'] is None:
        return False
    
    if model_cache['last_used'] is None:
        return False
    
    # Check if cache has expired
    if datetime.now() - model_cache['last_used'] > timedelta(seconds=MODEL_CACHE_TIMEOUT):
        logger.info("Model cache expired")
        return False
    
    # Check if we should unload due to resource constraints
    if should_unload_model():
        return False
    
    return True

def initialize_whisper():
    """Initialize Whisper model, processor and pipeline with caching and VRAM optimization."""
    global model_cache
    
    # Check if we can use the cached model
    with model_cache['lock']:
        if is_model_cache_valid():
            logger.info("Using cached model")
            model_cache['last_used'] = datetime.now()
            return
    
    # Get the current process ID for logging
    process_id = os.getpid()
    
    # Smart GPU selection based on available memory
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            # Use smart GPU selection for multiple GPUs
            best_gpu = select_best_gpu()
            device = f"cuda:{best_gpu}" if best_gpu is not None else "cuda:0"
            logger.info(f"Process {process_id} using smart-selected GPU {best_gpu} of {gpu_count} available GPUs")
        else:
            # Single GPU case
            device = "cuda:0"
            logger.info(f"Process {process_id} using single available GPU")
    else:
        device = "cpu"
        logger.info(f"Process {process_id} using CPU")

    # Always use float16 for GPU, float32 for CPU
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    try:
        with model_cache['lock']:
            # Check if another process has loaded the model while we were waiting
            if is_model_cache_valid():
                logger.info("Using model loaded by another process")
                model_cache['last_used'] = datetime.now()
                return
            
            logger.info(f"Process {process_id}: Loading model {WHISPER_MODEL}...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True if device != "cpu" else False,
                use_safetensors=True
            )
            model.to(device)
            logger.info(f"Process {process_id}: Model loaded successfully")

            logger.info(f"Process {process_id}: Loading processor...")
            processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
            logger.info(f"Process {process_id}: Processor loaded successfully")

            # Get current resource info
            resources = get_system_resources()
            # Use optimized batch size calculation based on performance mode
            optimal_batch_size = calculate_conservative_batch_size(device)
            
            # Calculate optimal chunk length based on performance mode and available memory
            chunk_length_s = DEFAULT_CHUNK_LENGTH
            if device != "cpu" and resources['gpu_info']:
                gpu = resources['gpu_info'][0]
                available_memory_gb = (gpu['memory_total'] - gpu['memory_used']) / 1024
                
                if PERFORMANCE_MODE == "aggressive":
                    if available_memory_gb > 8:  # More than 8GB available
                        chunk_length_s = min(MAX_CHUNK_LENGTH, 90)  # Very long chunks for maximum context
                    elif available_memory_gb > 6:  # More than 6GB available
                        chunk_length_s = min(MAX_CHUNK_LENGTH, 75)
                    elif available_memory_gb > 4:  # More than 4GB available
                        chunk_length_s = min(MAX_CHUNK_LENGTH, 60)
                    else:
                        chunk_length_s = min(MAX_CHUNK_LENGTH, 50)
                elif PERFORMANCE_MODE == "balanced":
                    if available_memory_gb > 6:  # More than 6GB available
                        chunk_length_s = min(MAX_CHUNK_LENGTH, 60)
                    elif available_memory_gb > 4:  # More than 4GB available
                        chunk_length_s = min(MAX_CHUNK_LENGTH, 45)
                    else:
                        chunk_length_s = min(MAX_CHUNK_LENGTH, 35)
                else:  # conservative mode
                    if available_memory_gb > 6:  # More than 6GB available
                        chunk_length_s = 45  # Longer chunks for better context
                    elif available_memory_gb > 4:  # More than 4GB available
                        chunk_length_s = 35
            
            logger.info(f"Process {process_id}: Performance mode: {PERFORMANCE_MODE} | "
                       f"Using batch_size={optimal_batch_size}, chunk_length_s={chunk_length_s}")

            # Create the pipeline with optimized settings
            logger.info(f"Process {process_id}: Creating optimized Whisper pipeline...")
            whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=chunk_length_s,
                batch_size=optimal_batch_size,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=device,
            )
            logger.info(f"Process {process_id}: Optimized pipeline created successfully")
            
            # Update cache
            model_cache.update({
                'model': model,
                'processor': processor,
                'pipeline': whisper_pipeline,
                'last_used': datetime.now(),
                'device': device
            })
            
    except Exception as e:
        logger.error(f"Process {process_id}: Error initializing Whisper: {e}")
        cleanup_whisper()  # Clean up on error
        raise  # Re-raise the exception to be handled by the caller

def cleanup_whisper():
    """Clean up Whisper model resources with improved memory management."""
    global model_cache
    
    with model_cache['lock']:
        if model_cache['model'] is None:
            return
            
        process_id = os.getpid()
        device = model_cache.get('device', 'cpu')
        
        logger.info(f"Process {process_id}: Cleaning up Whisper model resources...")
        
        try:
            # Move model to CPU first to free GPU memory
            if hasattr(model_cache['model'], 'to'):
                model_cache['model'].to('cpu')
            
            # Clear all references
            del model_cache['model']
            del model_cache['processor']
            del model_cache['pipeline']
            
            model_cache.update({
                'model': None,
                'processor': None,
                'pipeline': None,
                'last_used': None,
                'device': None
            })
                
            # Enhanced CUDA cleanup
            if torch.cuda.is_available() and "cuda" in device:
                gpu_id = int(device.split(":")[1]) if ":" in device else 0
                
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info(f"Process {process_id}: Enhanced cleanup completed for {device}")
                
        except Exception as e:
            logger.error(f"Process {process_id}: Error during Whisper cleanup: {e}")

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
    """Clean up any remaining .lock or .correction.lock files by scanning directories."""
    logger.info("Scanning for and removing orphaned lock files...")
    found_locks = 0

    # Scan for transcription locks (.lock) associated with files in AUDIO_DIR
    if os.path.exists(AUDIO_DIR):
        for root, _, files in os.walk(AUDIO_DIR):
            for file_name in files:
                # Check for .lock files that are siblings to non-lock files
                if not file_name.endswith(".lock"):
                    lock_file_path = os.path.join(root, file_name + ".lock")
                    if os.path.exists(lock_file_path):
                        try:
                            os.remove(lock_file_path)
                            logger.info(f"Cleaned up orphaned transcription lock file: {lock_file_path}")
                            found_locks += 1
                        except Exception as e:
                            logger.error(f"Error cleaning up lock file {lock_file_path}: {e}")
    
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
                            logger.info(f"Cleaned up orphaned correction lock file: {lock_file_path}")
                            found_locks += 1
                        except Exception as e:
                            logger.error(f"Error cleaning up lock file {lock_file_path}: {e}")
    
    if found_locks == 0:
        logger.info("No orphaned lock files found during scan.")

# Global flag to prevent recursive signal handling
_signal_received = False

def signal_handler(signum, frame):
    """Handle termination signals."""
    global _signal_received
    
    # Prevent recursive signal handling
    if _signal_received:
        logger.info(f"Signal {signum} already being handled, forcing immediate exit...")
        os._exit(1)
    
    _signal_received = True
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()
    
    # Force terminate all child processes
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                logger.info(f"Terminating child process {child.pid}")
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Wait a bit for graceful termination
        time.sleep(1)  # Reduced from 2 seconds
        
        # Force kill any remaining children
        for child in children:
            try:
                if child.is_running():
                    logger.info(f"Force killing child process {child.pid}")
                    child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
    except Exception as e:
        logger.error(f"Error in signal handler: {e}")
    
    logger.info("Signal handler complete, forcing exit...")
    # Use os._exit() for immediate termination without cleanup
    os._exit(0)

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
    except Exception as e:
        logger.error(f"Error creating lock file for {audio_path}: {e}")
        return
    
    retries = 0
    success = False
    try:
        while retries < MAX_RETRIES:
            logger.info(f"Starting transcription of {audio_path} (attempt {retries + 1}/{MAX_RETRIES})")
            initialize_whisper()
            pbar = tqdm(total=100, desc=f"Transcribing {os.path.basename(audio_path)}", leave=False, position=1)
            with model_cache['lock']:
                result = model_cache['pipeline'](audio_path, generate_kwargs={"max_new_tokens": 256})
            transcription = result["text"]
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
                logger.warning(f"Transcription too short for {audio_path}, retrying...")
                retries += 1
            time.sleep(5)
    except Exception as e:
        logger.error(f"Unexpected error during transcription of {audio_path} in try: {e}")
    finally:
        if not success:
            logger.error(f"Max retries reached or critical error for {audio_path}. Transcription failed or was too short.")
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception as e:
            logger.error(f"Error removing lock file for {audio_path}: {e}")

def correct_file(args):
    """Correct a transcription using Ollama API."""
    trans_path, corrected_path = args
    lock_file = trans_path + ".correction.lock"

    if os.path.exists(lock_file):
        logger.info(f"Skipping correction for {trans_path}: lock file exists.")
        return

    if os.path.exists(corrected_path):
        logger.info(f"Skipping correction for {trans_path}: corrected file already exists.")
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
        logger.info(f"Starting correction of {trans_path} using Ollama model {OLLAMA_MODEL}")
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
        logger.info(f"Successfully corrected {trans_path} and saved to {corrected_path}")

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

def get_system_resources():
    """Get current system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    gpu_info = []
    
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
        except Exception as e:
            logger.warning(f"Error getting GPU info: {e}")
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available': memory.available,
        'gpu_info': gpu_info
    }

def calculate_conservative_batch_size(device: str) -> int:
    """Calculate batch size to optimize VRAM utilization based on performance mode."""
    if "cuda" not in device:
        return 4
        
    try:
        gpu_id = int(device.split(":")[1]) if ":" in device else 0
        gpus = GPUtil.getGPUs()
        if gpu_id < len(gpus):
            gpu = gpus[gpu_id]
            available_memory_mb = gpu.memoryTotal - gpu.memoryUsed
            
            # Use performance mode settings from config
            from config import MEMORY_SAFETY_FACTOR, MAX_BATCH_SIZE, PERFORMANCE_MODE
            
            # Apply memory safety factor based on performance mode
            safe_memory = available_memory_mb * MEMORY_SAFETY_FACTOR
            
            # Aggressive batch sizing based on available memory
            if safe_memory > 10000:   # 10GB+ available
                batch_size = min(MAX_BATCH_SIZE, 24)
            elif safe_memory > 8000:  # 8GB+ available  
                batch_size = min(MAX_BATCH_SIZE, 20)
            elif safe_memory > 6000:  # 6GB+ available
                batch_size = min(MAX_BATCH_SIZE, 16)
            elif safe_memory > 4000:  # 4GB+ available
                batch_size = min(MAX_BATCH_SIZE, 12)
            elif safe_memory > 2000:  # 2GB+ available
                batch_size = min(MAX_BATCH_SIZE, 8)
            else:
                batch_size = min(MAX_BATCH_SIZE, 4)
            
            logger.info(f"Performance mode: {PERFORMANCE_MODE} | Available: {available_memory_mb}MB | "
                       f"Safe memory: {safe_memory:.0f}MB | Batch size: {batch_size}")
            
            return batch_size
            
    except Exception as e:
        logger.warning(f"Error calculating batch size: {e}")
        
    return 4  # Safe default

def select_best_gpu():
    """Select GPU with most available memory to prevent OOM errors."""
    try:
        if not torch.cuda.is_available():
            return None
            
        gpus = GPUtil.getGPUs()
        if not gpus:
            return 0
            
        # Find GPU with most available memory
        best_gpu = 0
        max_available = 0
        
        for gpu in gpus:
            available = gpu.memoryTotal - gpu.memoryUsed
            logger.info(f"GPU {gpu.id}: {available}MB available out of {gpu.memoryTotal}MB total")
            if available > max_available:
                max_available = available
                best_gpu = gpu.id
                
        logger.info(f"Selected GPU {best_gpu} with {max_available}MB available memory")
        return best_gpu
    except Exception as e:
        logger.warning(f"Error selecting best GPU: {e}, using GPU 0")
        return 0

def process_transcription_batch(tasks):
    """Process a batch of transcription tasks in parallel with optimized VRAM utilization."""
    process_id = os.getpid()
    try:
        # Get initial resources
        initial_resources = get_system_resources()
        logger.info(f"Process {process_id}: Initial resources - CPU: {initial_resources['cpu_percent']}%, "
                   f"Memory: {initial_resources['memory_percent']}%")
        if initial_resources['gpu_info']:
            gpu = initial_resources['gpu_info'][0]
            logger.info(f"Process {process_id}: GPU Memory: {gpu['memory_used']}MB/{gpu['memory_total']}MB used")
        
        # Process the entire batch using optimized batch transcription
        start_time = time.time()
        successful, failed = transcribe_files_batch(tasks)
        end_time = time.time()
        
        # Log performance metrics
        current_resources = get_system_resources()
        batch_time = end_time - start_time
        files_per_second = len(tasks) / batch_time if batch_time > 0 else 0
        
        logger.info(f"Process {process_id}: Batch of {len(tasks)} files completed in {batch_time:.2f}s "
                   f"({files_per_second:.2f} files/sec). Success: {successful}, Failed: {failed}")
        logger.info(f"Process {process_id}: Current resources - CPU: {current_resources['cpu_percent']}%, "
                   f"Memory: {current_resources['memory_percent']}%")
        
        if current_resources['gpu_info']:
            gpu = current_resources['gpu_info'][0]
            logger.info(f"Process {process_id}: GPU Memory: {gpu['memory_used']}MB/{gpu['memory_total']}MB used")
            
    except Exception as e:
        logger.error(f"Process {process_id}: Error in transcription batch: {e}")
    finally:
        # Clean up Whisper resources for this process
        cleanup_whisper()
        
        # Log final resources
        final_resources = get_system_resources()
        logger.info(f"Process {process_id}: Final resources - CPU: {final_resources['cpu_percent']}%, "
                   f"Memory: {final_resources['memory_percent']}%")
        if final_resources['gpu_info']:
            gpu = final_resources['gpu_info'][0]
            logger.info(f"Process {process_id}: Final GPU Memory: {gpu['memory_used']}MB/{gpu['memory_total']}MB used")

def split_tasks_into_batches(tasks, batch_size=4):
    """Split tasks into batches for parallel processing."""
    return [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

def transcribe_files_batch(file_batch):
    """Transcribe multiple audio files in a single batch operation for better VRAM utilization."""
    process_id = os.getpid()
    logger.info(f"Process {process_id}: Starting batch transcription of {len(file_batch)} files")
    
    # Initialize Whisper for this process if not already done
    initialize_whisper()
    
    successful_transcriptions = 0
    failed_transcriptions = 0
    
    try:
        with model_cache['lock']:
            pipeline = model_cache['pipeline']
            device = model_cache['device']  # Get device from cache
            
            for audio_path, trans_path in file_batch:
                lock_file = audio_path + ".lock"
                
                # Skip if already processing or completed
                if os.path.exists(lock_file):
                    logger.info(f"Skipping {audio_path}: lock file exists.")
                    continue
                
                if os.path.exists(trans_path) and is_valid_transcription(trans_path):
                    logger.info(f"Skipping {audio_path}: valid transcription already exists.")
                    continue
                
                try:
                    Path(lock_file).touch()
                except Exception as e:
                    logger.error(f"Error creating lock file for {audio_path}: {e}")
                    continue
                
                try:
                    logger.info(f"Transcribing {os.path.basename(audio_path)}")
                    
                    # Use optimized generation parameters for better VRAM utilization
                    if ENABLE_MIXED_PRECISION and "cuda" in device:
                        # Use automatic mixed precision for GPU inference
                        with autocast():
                            result = pipeline(
                                audio_path, 
                                generate_kwargs={
                                    "max_new_tokens": 256,  # Reduced to stay within model limits
                                    "do_sample": False,     # Deterministic output
                                    "num_beams": 1,         # Faster inference
                                }
                            )
                    else:
                        # Standard precision for CPU or when AMP is disabled
                        result = pipeline(
                        audio_path, 
                        generate_kwargs={
                            "max_new_tokens": 256,  # Reduced to stay within model limits
                            "do_sample": False,     # Deterministic output
                            "num_beams": 1,         # Faster inference
                        }
                    )
                    
                    transcription = result["text"]
                    
                    # Ensure directory exists and write transcription
                    ensure_dir(os.path.dirname(trans_path))
                    with open(trans_path, "w", encoding="utf-8") as f:
                        f.write(transcription)
                    
                    if is_valid_transcription(trans_path):
                        logger.info(f"Successfully transcribed {audio_path}")
                        successful_transcriptions += 1
                    else:
                        logger.warning(f"Transcription too short for {audio_path}")
                        failed_transcriptions += 1
                        
                except Exception as e:
                    logger.error(f"Error transcribing {audio_path}: {e}")
                    failed_transcriptions += 1
                finally:
                    # Always remove lock file
                    try:
                        if os.path.exists(lock_file):
                            os.remove(lock_file)
                    except Exception as e:
                        logger.error(f"Error removing lock file for {audio_path}: {e}")
            
    except Exception as e:
        logger.error(f"Process {process_id}: Critical error in batch transcription: {e}")
        raise
    
    logger.info(f"Process {process_id}: Batch completed - {successful_transcriptions} successful, {failed_transcriptions} failed")
    return successful_transcriptions, failed_transcriptions

def preload_and_optimize_model():
    """Preload and optimize the Whisper model for maximum performance."""
    logger.info("Preloading and optimizing Whisper model for maximum performance...")
    
    try:
        # Initialize the model
        initialize_whisper()
        
        # Get current resource info
        resources = get_system_resources()
        if resources['gpu_info']:
            gpu = resources['gpu_info'][0]
            logger.info(f"Model loaded. GPU Memory usage: {gpu['memory_used']}MB/{gpu['memory_total']}MB "
                       f"({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
            
            # Check if we can enable additional optimizations
            if torch.cuda.is_available():
                # Enable optimized attention if available (for newer PyTorch versions)
                try:
                    with model_cache['lock']:
                        if model_cache['model'] is not None:
                            # Try to enable flash attention or other optimizations
                            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                                logger.info("Scaled dot product attention available - model should use optimized attention")
                            
                            # Enable torch compile if available (PyTorch 2.0+)
                            if hasattr(torch, 'compile'):
                                logger.info("PyTorch compile available but skipping for compatibility")
                                # Uncomment next line if you want to try torch.compile (experimental)
                                # model_cache['model'] = torch.compile(model_cache['model'])
                                
                except Exception as e:
                    logger.warning(f"Could not apply additional optimizations: {e}")
        
        logger.info("Model preloading and optimization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during model preloading and optimization: {e}")
        return False

def main(num_workers_arg, batch_size_arg):
    logger.info("Initializing directories...")
    ensure_dir(AUDIO_DIR)
    ensure_dir(TRANSCRIPTIONS_DIR)
    ensure_dir(CORRECTED_DIR)
    ensure_dir("backup")
    
    if not OLLAMA_MODEL or not CORRECTION_PROMPT:
        logger.error("OLLAMA_MODEL and CORRECTION_PROMPT must be set in the configuration.")
        sys.exit(1)

    logger.info(f"Using Ollama model: {OLLAMA_MODEL} via {OLLAMA_API_URL}")
    
    # Log optimization settings
    logger.info("="*60)
    logger.info("TRANSFIXER OPTIMIZATIONS ENABLED:")
    logger.info("="*60)
    
    # Display performance mode
    from config import get_performance_summary
    perf_summary = get_performance_summary()
    logger.info(f"🚀 Performance Mode: {perf_summary['mode'].upper()}")
    logger.info(f"🎮 GPU Memory Usage: {perf_summary['gpu_memory_fraction']} (Safety Factor: {perf_summary['memory_safety_factor']})")
    logger.info(f"📦 Max Batch Size: {perf_summary['max_batch_size']}")
    logger.info(f"⏱️  Audio Chunk Length: {perf_summary['default_chunk_length']} (Range: {perf_summary['chunk_length_range']})")
    
    if ENABLE_MIXED_PRECISION:
        logger.info("✓ Automatic Mixed Precision (AMP) enabled for faster inference")
    if SMART_GPU_SELECTION:
        logger.info("✓ Smart GPU selection based on available memory")
    if perf_summary['aggressive_batching']:
        logger.info("✓ Aggressive batching enabled for maximum throughput")
    else:
        logger.info("✓ Conservative batch sizing to prevent OOM errors")
    logger.info("="*60)
    
    # Preload and optimize the Whisper model for maximum performance
    logger.info("Preloading Whisper model with memory optimizations...")
    if not preload_and_optimize_model():
        logger.warning("Model preloading failed, but continuing with standard initialization")
    else:
        resources = get_system_resources()
        if resources['gpu_info']:
            gpu = resources['gpu_info'][0]
            vram_usage_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            logger.info(f"Memory optimized! Using {gpu['memory_used']}MB/{gpu['memory_total']}MB ({vram_usage_percent:.1f}%) of GPU memory")

    try:
        cycle_count = 0
        while not shutdown_event.is_set():
            cycle_count += 1
            logger.info("="*50)
            logger.info(f"Starting processing cycle #{cycle_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*50)
            
            if shutdown_event.is_set(): break
            create_backup()
            if shutdown_event.is_set(): break
            cleanup_old_backups()
            if shutdown_event.is_set(): break

            # Phase 1: Transcribe audio files
            logger.info("-" * 10 + " Phase 1: Transcription " + "-" * 10)
            trans_tasks = collect_transcription_tasks()
            if trans_tasks and not shutdown_event.is_set():
                logger.info(f"Found {len(trans_tasks)} files to transcribe")
                resources = get_system_resources()
                num_processes = num_workers_arg
                batch_size = batch_size_arg
                logger.info(f"Using {num_processes} processes with batch size {batch_size}")
                task_batches = split_tasks_into_batches(trans_tasks, batch_size)
                
                try:
                    with ProcessPoolExecutor(max_workers=num_processes) as executor:
                        futures = {executor.submit(process_transcription_batch, batch) for batch in task_batches}
                        with tqdm(total=len(trans_tasks), desc="Overall Transcription Progress", position=0, leave=False) as pbar:
                            for future in as_completed(futures):
                                if shutdown_event.is_set():
                                    logger.info("Shutdown during transcription: cancelling pending tasks.")
                                    for f_cancel in futures:
                                        if not f_cancel.done(): 
                                            f_cancel.cancel()
                                    # Force shutdown the executor for WSL compatibility
                                    executor.shutdown(wait=False)
                                    break # Exit as_completed loop
                                try:
                                    future.result()
                                    pbar.update(len(task_batches[0]) if task_batches else 0) # Approx update based on first batch len or 0
                                except CancelledError:
                                    logger.info("A transcription batch was cancelled.")
                                    pbar.update(len(task_batches[0]) if task_batches else 0)
                                except Exception as e:
                                    logger.error(f"Transcription batch failed: {e}")
                        # End of tqdm block
                        if shutdown_event.is_set():
                            logger.info("Transcription phase interrupted by shutdown signal.")
                            # Ensure futures are cancelled before executor.__exit__ (which calls shutdown)
                            for f_cancel in futures:
                                if not f_cancel.done(): f_cancel.cancel()
                            # The 'with executor:' will call executor.shutdown(wait=True).
                            # If tasks don't respond to cancellation, this will wait.
                except Exception as e_exec:
                    logger.error(f"Error with transcription executor: {e_exec}")
                logger.info("Transcription phase completed or interrupted.")
            elif not trans_tasks:
                logger.info("No new audio files to transcribe")
            
            if shutdown_event.is_set(): break

            # Phase 2: Correct transcriptions
            logger.info("-" * 10 + " Phase 2: Correction " + "-" * 10)
            correct_tasks = collect_correction_tasks()
            if correct_tasks and not shutdown_event.is_set():
                logger.info(f"Found {len(correct_tasks)} transcriptions to correct")
                try:
                    with Pool(processes=1) as pool: # TODO: Parameterize correction workers
                        async_results = []
                        for task in correct_tasks:
                            if shutdown_event.is_set():
                                logger.info("Shutdown during correction task submission.")
                                break
                            async_results.append(pool.apply_async(correct_file, (task,)))
                        
                        if not shutdown_event.is_set():
                            with tqdm(total=len(async_results), desc="Correction Progress", leave=False) as pbar:
                                for res in async_results:
                                    if shutdown_event.is_set():
                                        logger.info("Shutdown while waiting for correction results. Terminating pool.")
                                        pool.terminate()
                                        break
                                    try:
                                        res.get(timeout=1) # Periodically check for shutdown
                                        pbar.update(1)
                                    except multiprocessing.TimeoutError:
                                        continue # Still waiting, check shutdown_event next
                                    except Exception as e:
                                        logger.error(f"Correction task failed: {e}")
                                        pbar.update(1) # Count as processed
                        
                        if shutdown_event.is_set():
                            logger.info("Terminating correction pool due to shutdown signal.")
                            pool.terminate()
                            pool.join(timeout=10) # Wait a bit for termination
                        else:
                            pool.close()
                            pool.join() # Normal shutdown
                except Exception as e_pool:
                    logger.error(f"Error with correction pool: {e_pool}")
                logger.info("Correction phase completed or interrupted.")
            elif not correct_tasks:
                logger.info("No new transcriptions to correct")

            if shutdown_event.is_set():
                logger.info("Shutdown requested, breaking main loop before sleep.")
                break
            
            logger.info(f"Cycle complete. Sleeping for {CHECK_INTERVAL} seconds.")
            # Sleep with frequent shutdown checks
            for i in range(CHECK_INTERVAL):
                if shutdown_event.is_set(): 
                    logger.info(f"Shutdown detected during sleep (after {i} seconds)")
                    break
                time.sleep(1)

        logger.info("Main processing loop finished or interrupted.")

    except Exception as e:
        logger.critical(f"Critical error in main loop: {e}", exc_info=True)
        shutdown_event.set() # Ensure shutdown is signalled on other critical errors
    finally:
        logger.info("Main loop finally block reached.")
        # Pools should be closed by their 'with' statements.
        # This finally block is a safeguard or for other main-level resources if any.


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Starting TransFixer (Ollama Edition)...")
        parser = argparse.ArgumentParser(description="TransFixer: Transcribe and correct audio files.")
        parser.add_argument("--num-workers", type=int, choices=[1, 2], default=1, help="Number of worker processes for transcription (1 or 2). Default is 1.")
        parser.add_argument("--batch-size", type=int, default=8, help="Batch size for transcription tasks. Default is 8.")
        args = parser.parse_args()
        main(args.num_workers, args.batch_size)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in __main__, ensuring shutdown event is set.")
        shutdown_event.set()
    except SystemExit as e:
        logger.info(f"SystemExit caught in __main__ ({e}), proceeding to final cleanup.")
    except Exception as e:
        logger.error(f"Unhandled exception in __main__: {e}", exc_info=True)
        shutdown_event.set()
    finally:
        logger.info("Performing final script cleanup...")
        if shutdown_event.is_set():
            logger.info("Shutdown event was set. Ensuring resources are released.")
        
        logger.info("Cleaning up lock files...")
        cleanup_lock_files()
        
        logger.info("Cleaning up Whisper model...")
        cleanup_whisper() # This might still be an issue if workers didn't exit cleanly
        
        logger.info("Exiting TransFixer.")