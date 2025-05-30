import os
import subprocess
import shutil
import logging
from multiprocessing import Pool, set_start_method
import multiprocessing
import requests
import time
from pathlib import Path
import signal
import sys
from datetime import datetime, timedelta
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from torch.cuda.amp import autocast
from config import (
    OLLAMA_MODEL, CORRECTION_PROMPT, OLLAMA_API_URL, OLLAMA_OPTIONS,
    GPU_MEMORY_FRACTION, ENABLE_MIXED_PRECISION, DEFAULT_CHUNK_LENGTH,
    MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH, MAX_BATCH_SIZE, PERFORMANCE_MODE,
    MEMORY_SAFETY_FACTOR, ENABLE_FLASH_ATTENTION, FLASH_ATTENTION_AVAILABLE,
    ATTENTION_IMPLEMENTATION, ENABLE_SDPA, ENABLE_TORCH_COMPILE,
    TORCH_COMPILE_MODE, TORCH_COMPILE_FULLGRAPH, PERFORMANCE_MONITORING_AVAILABLE
)
from tqdm import tqdm
import re
import psutil
import GPUtil
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import argparse

# Import new core components
from core import (
    ProcessManager, GPUManager, ModelCacheManager, SystemMonitor,
    TransFixerError, ProcessError, GPUError, ModelCacheError
)

# Global process manager instance
process_manager = ProcessManager()

# Global GPU manager instance
gpu_manager = GPUManager()

# Global model cache manager instance
model_cache_manager = ModelCacheManager()

# Global system monitor instance
system_monitor = SystemMonitor()

# Global shutdown event
shutdown_event = threading.Event()

# --- START: Configuration ---
AUDIO_DIR = "audio"
TRANSCRIPTIONS_DIR = "transcriptions"
CORRECTED_DIR = "corrected"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "transcription_errors.log")
MIN_CHARS = 50
MAX_RETRIES = 3
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
CHECK_INTERVAL = 300
# --- END: Configuration ---

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    set_start_method('spawn')
except RuntimeError:
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

# Global flag for verbose logging
verbose_logging = False

# Global flags for performance analysis
analyze_performance = False
performance_analysis_done = False

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

def cleanup_gpu_resources():
    """Enhanced GPU resource cleanup using GPU manager."""
    try:
        gpu_manager.cleanup_gpu_memory()
        logger.info("GPU resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up GPU resources: {e}")

def initialize_whisper():
    """Initialize Whisper model with improved caching and GPU management."""
    process_id = os.getpid()
    cache_key = f"whisper_{WHISPER_MODEL}"
    
    # Try to get from cache first
    cached_model = model_cache_manager.get(cache_key)
    if cached_model:
        log_verbose(f"Process {process_id}: Using cached Whisper model")
        return cached_model
    
    log_verbose(f"Process {process_id}: Initializing Whisper model")
    
    try:
        # Select best GPU using GPU manager
        best_gpu = gpu_manager.select_best_gpu()
        if best_gpu is not None and torch.cuda.is_available():
            device = f"cuda:{best_gpu}"
            log_verbose(f"Process {process_id}: Using GPU {best_gpu}")
        else:
            device = "cpu"
            log_verbose(f"Process {process_id}: Using CPU")

        # Always use float16 for GPU, float32 for CPU
        torch_dtype = torch.float16 if device != "cpu" else torch.float32

        log_verbose(f"Process {process_id}: Loading model {WHISPER_MODEL}...")
        
        # Model loading with optimizations
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True if device != "cpu" else False,
            "use_safetensors": True,
        }
        
        # Add attention implementation if Flash Attention is available
        if ENABLE_FLASH_ATTENTION and FLASH_ATTENTION_AVAILABLE and ATTENTION_IMPLEMENTATION == "flash_attention_2":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            log_verbose(f"Process {process_id}: Using Flash Attention 2")
        elif ENABLE_SDPA:
            model_kwargs["attn_implementation"] = "sdpa"
            log_verbose(f"Process {process_id}: Using SDPA (Scaled Dot Product Attention)")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL, **model_kwargs)
        model.to(device)
        log_verbose(f"Process {process_id}: Model loaded successfully")

        log_verbose(f"Process {process_id}: Loading processor...")
        processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
        log_verbose(f"Process {process_id}: Processor loaded successfully")

        # Calculate optimal batch size using GPU manager
        if device != "cpu":
            gpu_id = int(device.split(":")[1]) if ":" in device else 0
            optimal_batch_size = gpu_manager.calculate_optimal_batch_size(gpu_id)
        else:
            optimal_batch_size = 4
        
        # Calculate optimal chunk length
        chunk_length_s = DEFAULT_CHUNK_LENGTH
        if device != "cpu":
            gpu_info = gpu_manager.get_gpu_info(gpu_id)
            available_memory_gb = gpu_info['memory_available'] / 1024
            
            if PERFORMANCE_MODE == "aggressive":
                if available_memory_gb > 8:
                    chunk_length_s = min(MAX_CHUNK_LENGTH, 90)
                elif available_memory_gb > 6:
                    chunk_length_s = min(MAX_CHUNK_LENGTH, 75)
                elif available_memory_gb > 4:
                    chunk_length_s = min(MAX_CHUNK_LENGTH, 60)
                else:
                    chunk_length_s = min(MAX_CHUNK_LENGTH, 50)
            elif PERFORMANCE_MODE == "balanced":
                if available_memory_gb > 6:
                    chunk_length_s = min(MAX_CHUNK_LENGTH, 60)
                elif available_memory_gb > 4:
                    chunk_length_s = min(MAX_CHUNK_LENGTH, 45)
                else:
                    chunk_length_s = min(MAX_CHUNK_LENGTH, 35)
            else:  # conservative mode
                if available_memory_gb > 6:
                    chunk_length_s = 45
                elif available_memory_gb > 4:
                    chunk_length_s = 35
        
        log_verbose(f"Process {process_id}: Performance mode: {PERFORMANCE_MODE} | "
                   f"Using batch_size={optimal_batch_size}, chunk_length_s={chunk_length_s}")

        # Create the pipeline with optimized settings
        log_verbose(f"Process {process_id}: Creating optimized Whisper pipeline...")
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
        log_verbose(f"Process {process_id}: Optimized pipeline created successfully")
        
        # Enable torch compile if available (PyTorch 2.0+)
        if ENABLE_TORCH_COMPILE and hasattr(torch, 'compile'):
            log_verbose("Enabling PyTorch compile for maximum performance")
            try:
                # Apply torch.compile with optimal settings
                model = torch.compile(
                    model, 
                    mode=TORCH_COMPILE_MODE, 
                    fullgraph=TORCH_COMPILE_FULLGRAPH
                )
                log_verbose("✓ Torch compile enabled - expect 4.5x speed improvement")
            except Exception as e:
                logger.warning(f"Torch compile failed: {e}")
        
        # Store in cache with cleanup callback
        def cleanup_callback():
            try:
                # Move model to CPU first to free GPU memory
                if hasattr(model, 'to'):
                    model.to('cpu')
                
                # Enhanced CUDA cleanup
                if torch.cuda.is_available() and "cuda" in device:
                    gpu_id = int(device.split(":")[1]) if ":" in device else 0
                    gpu_manager.cleanup_gpu_memory(gpu_id)
                    
                log_verbose(f"Model cleanup completed for {cache_key}")
            except Exception as e:
                logger.error(f"Error in model cleanup: {e}")
        
        model_cache_manager.put(
            cache_key,
            model=model,
            processor=processor,
            pipeline=whisper_pipeline,
            device=device,
            cleanup_callback=cleanup_callback
        )
        
        return {
            'model': model,
            'processor': processor,
            'pipeline': whisper_pipeline,
            'device': device
        }
        
    except Exception as e:
        logger.error(f"Process {process_id}: Error initializing Whisper: {e}")
        cleanup_gpu_resources()
        raise

def cleanup_whisper():
    """Clean up Whisper model resources using cache manager."""
    try:
        cache_key = f"whisper_{WHISPER_MODEL}"
        model_cache_manager.remove(cache_key)
        log_verbose("Whisper model resources cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up Whisper: {e}")

def create_backup():
    """Create a backup of transcriptions and corrected files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("backup", timestamp)
    os.makedirs(backup_dir, exist_ok=True)
    
    log_verbose(f"Creating backup in {backup_dir}")
    
    # Backup transcriptions
    trans_backup = os.path.join(backup_dir, "transcriptions")
    if os.path.exists(TRANSCRIPTIONS_DIR):
        shutil.copytree(TRANSCRIPTIONS_DIR, trans_backup, dirs_exist_ok=True)
        log_verbose("Transcriptions backed up successfully")
    
    # Backup corrected files
    corr_backup = os.path.join(backup_dir, "corrected")
    if os.path.exists(CORRECTED_DIR):
        shutil.copytree(CORRECTED_DIR, corr_backup, dirs_exist_ok=True)
        log_verbose("Corrected files backed up successfully")
    
    return backup_dir

def cleanup_old_backups(max_backups=5):
    """Remove old backups keeping only the most recent ones."""
    backup_base_dir = "backup"
    if not os.path.exists(backup_base_dir):
        return
    backup_dirs = sorted([d for d in os.listdir(backup_base_dir) if os.path.isdir(os.path.join(backup_base_dir, d))])
    if len(backup_dirs) > max_backups:
        log_verbose(f"Cleaning up old backups. Keeping {max_backups} most recent backups.")
        for old_dir in backup_dirs[:-max_backups]:
            shutil.rmtree(os.path.join(backup_base_dir, old_dir))
            log_verbose(f"Removed old backup: {old_dir}")

def cleanup_lock_files():
    """Clean up any remaining .lock or .correction.lock files by scanning directories."""
    log_verbose("Scanning for and removing orphaned lock files...")
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
                            log_verbose(f"Cleaned up orphaned transcription lock file: {lock_file_path}")
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
                            log_verbose(f"Cleaned up orphaned correction lock file: {lock_file_path}")
                            found_locks += 1
                        except Exception as e:
                            logger.error(f"Error cleaning up lock file {lock_file_path}: {e}")
    
    if found_locks == 0:
        logger.info("No orphaned lock files found during scan.")

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
            model_components = initialize_whisper()
            pbar = tqdm(total=100, desc=f"Transcribing {os.path.basename(audio_path)}", leave=False, position=1)
            result = model_components['pipeline'](audio_path, generate_kwargs={"max_new_tokens": 256})
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
    """Get current system resource usage using system monitor."""
    try:
        return system_monitor.get_system_resources()
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        # Return minimal fallback data
        return {
            'cpu': {'percent': 0},
            'memory': {'percent': 0, 'available': 0},
            'gpu': {'devices': []}
        }

def calculate_conservative_batch_size(device: str) -> int:
    """Calculate batch size using GPU manager."""
    if "cuda" not in device:
        return 4
    
    try:
        gpu_id = int(device.split(":")[1]) if ":" in device else 0
        return gpu_manager.calculate_optimal_batch_size(gpu_id)
    except Exception as e:
        logger.warning(f"Error calculating batch size: {e}")
        return 4

def select_best_gpu():
    """Select GPU using GPU manager."""
    try:
        return gpu_manager.select_best_gpu()
    except Exception as e:
        logger.warning(f"Error selecting best GPU: {e}")
        return 0

def process_transcription_batch(tasks_and_config):
    """Process a batch of transcription tasks in parallel with optimized VRAM utilization."""
    tasks, verbose_flag = tasks_and_config
    global verbose_logging
    verbose_logging = verbose_flag
    
    process_id = os.getpid()
    log_verbose(f"Process {process_id}: Starting batch processing")
    
    try:
        # Check CUDA availability and initialize
        if torch.cuda.is_available():
            try:
                # Select best GPU for this process
                best_gpu = select_best_gpu()
                if best_gpu is not None:
                    torch.cuda.set_device(best_gpu)
                
                # Test CUDA initialization
                test_tensor = torch.randn(1, 1, 4, 4, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                log_verbose(f"Process {process_id}: CUDA initialized successfully")
            except Exception as cuda_error:
                logger.error(f"Process {process_id}: CUDA initialization failed: {cuda_error}")
                # Fall back to CPU
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                log_verbose(f"Process {process_id}: Falling back to CPU mode")
        else:
            log_verbose(f"Process {process_id}: CUDA not available, using CPU")
        
        # Get initial resources
        initial_resources = get_system_resources()
        log_verbose(f"Process {process_id}: Initial resources - CPU: {initial_resources['cpu']['percent']}%, "
                   f"Memory: {initial_resources['memory']['percent']}%")
        if initial_resources['gpu']['devices']:
            gpu = initial_resources['gpu']['devices'][0]
            log_verbose(f"Process {process_id}: GPU Memory: {gpu['memory_used']}MB/{gpu['memory_total']}MB used")
        
        # Process the entire batch using optimized batch transcription
        start_time = time.time()
        successful, failed = transcribe_files_batch(tasks)
        end_time = time.time()
        
        # Log performance metrics
        current_resources = get_system_resources()
        batch_time = end_time - start_time
        files_per_second = len(tasks) / batch_time if batch_time > 0 else 0
        
        log_verbose(f"Process {process_id}: Batch of {len(tasks)} files completed in {batch_time:.2f}s "
                   f"({files_per_second:.2f} files/sec). Success: {successful}, Failed: {failed}")
        log_verbose(f"Process {process_id}: Current resources - CPU: {current_resources['cpu']['percent']}%, "
                   f"Memory: {current_resources['memory']['percent']}%")
        
        if current_resources['gpu']['devices']:
            gpu = current_resources['gpu']['devices'][0]
            log_verbose(f"Process {process_id}: GPU Memory: {gpu['memory_used']}MB/{gpu['memory_total']}MB used")
            
    except Exception as e:
        logger.error(f"Process {process_id}: Critical error in transcription batch: {e}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Process {process_id}: Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # Clean up Whisper resources for this process
        try:
            cleanup_whisper()
        except Exception as cleanup_error:
            logger.error(f"Process {process_id}: Error during cleanup: {cleanup_error}")
        
        # Log final resources
        try:
            final_resources = get_system_resources()
            log_verbose(f"Process {process_id}: Final resources - CPU: {final_resources['cpu']['percent']}%, "
                       f"Memory: {final_resources['memory']['percent']}%")
            if final_resources['gpu']['devices']:
                gpu = final_resources['gpu']['devices'][0]
                log_verbose(f"Process {process_id}: Final GPU Memory: {gpu['memory_used']}MB/{gpu['memory_total']}MB used")
        except Exception as resource_error:
            logger.error(f"Process {process_id}: Error getting final resources: {resource_error}")

def split_tasks_into_batches(tasks, batch_size=4):
    """Split tasks into batches for parallel processing."""
    return [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

def transcribe_files_batch(file_batch):
    """Optimized batch transcription using transformers pipeline."""
    process_id = os.getpid()
    log_verbose(f"Process {process_id}: Starting batch transcription of {len(file_batch)} files")
    
    # Initialize Whisper for this process if not already done
    model_components = initialize_whisper()
    
    successful_transcriptions = 0
    failed_transcriptions = 0
    
    try:
        pipeline = model_components['pipeline']
        device = model_components['device']
        
        log_verbose(f"Process {process_id}: Using transformers pipeline for transcription")
        
        successful, failed = _transcribe_batch_transformers(file_batch, pipeline, device)
        successful_transcriptions += successful
        failed_transcriptions += failed
        
    except Exception as e:
        logger.error(f"Process {process_id}: Critical error in batch transcription: {e}")
        raise
    
    log_verbose(f"Process {process_id}: Batch completed - {successful_transcriptions} successful, {failed_transcriptions} failed")
    return successful_transcriptions, failed_transcriptions

def _transcribe_batch_transformers(file_batch, pipeline, device):
    """Optimized transcription using transformers backend with mixed precision"""
    process_id = os.getpid()
    successful = 0
    failed = 0
    
    for audio_path, trans_path in file_batch:
        lock_file = audio_path + ".lock"
        
        if os.path.exists(lock_file) or (os.path.exists(trans_path) and is_valid_transcription(trans_path)):
            continue
        
        try:
            Path(lock_file).touch()
        except Exception as e:
            logger.error(f"Error creating lock file for {audio_path}: {e}")
            continue
            
        try:
            log_essential(f"🎧 Transcribing: {os.path.basename(audio_path)}")
            
            # Use optimized generation parameters
            generate_kwargs = {
                "max_new_tokens": 256,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
                "pad_token_id": pipeline.tokenizer.eos_token_id,
                "input_features": None  # This will be set by the pipeline
            }
            
            # Use automatic mixed precision for GPU inference
            if ENABLE_MIXED_PRECISION and "cuda" in device:
                with torch.amp.autocast('cuda'):  # Updated to use new autocast syntax
                    result = pipeline(audio_path, generate_kwargs=generate_kwargs)
            else:
                result = pipeline(audio_path, generate_kwargs=generate_kwargs)
            
            transcription = result["text"]
            
            # Ensure directory exists and write transcription
            ensure_dir(os.path.dirname(trans_path))
            with open(trans_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            if is_valid_transcription(trans_path):
                log_verbose(f"✅ Successfully transcribed {os.path.basename(audio_path)}")
                successful += 1
            else:
                logger.warning(f"⚠️ Transcription too short for {os.path.basename(audio_path)}")
                failed += 1
                
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            failed += 1
        finally:
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except Exception as e:
                logger.error(f"Error removing lock file for {audio_path}: {e}")
    
    return successful, failed

def preload_and_optimize_model():
    """Preload and optimize the Whisper model for maximum performance."""
    log_verbose("Preloading and optimizing Whisper model for maximum performance...")
    
    try:
        # Initialize the model
        model_components = initialize_whisper()
        
        # Get current resource info
        try:
            resources = system_monitor.get_system_resources()
            gpu_devices = resources.get('gpu', {}).get('devices', [])
            
            if gpu_devices:
                gpu = gpu_devices[0]
                log_verbose(f"Model loaded. GPU Memory usage: {gpu['memory_used']}MB/{gpu['memory_total']}MB "
                           f"({gpu['memory_percent']:.1f}%)")
                
                # Check if we can enable additional optimizations
                if torch.cuda.is_available():
                    # Try to enable flash attention or other optimizations
                    try:
                        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                            log_verbose("Scaled dot product attention available - model should use optimized attention")
                        
                        # Enable torch compile if available (PyTorch 2.0+)
                        if ENABLE_TORCH_COMPILE and hasattr(torch, 'compile'):
                            log_verbose("Enabling PyTorch compile for maximum performance")
                            try:
                                model = model_components['model']
                                # Enable static cache for torch.compile compatibility
                                if hasattr(model, 'generation_config'):
                                    model.generation_config.cache_implementation = "static"
                                
                                # Apply torch.compile with optimal settings
                                compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                                
                                # Update cache with compiled model
                                cache_key = f"whisper_{WHISPER_MODEL}"
                                cached_entry = model_cache_manager.get(cache_key)
                                if cached_entry:
                                    model_cache_manager.put(
                                        cache_key,
                                        model=compiled_model,
                                        processor=cached_entry['processor'],
                                        pipeline=cached_entry['pipeline'],
                                        device=cached_entry['device']
                                    )
                                
                                log_verbose("✓ Torch compile enabled - expect 4.5x speed improvement")
                            except Exception as e:
                                logger.warning(f"Torch compile failed: {e}")
                            
                    except Exception as e:
                        logger.warning(f"Could not apply additional optimizations: {e}")
        except Exception as e:
            logger.debug(f"Error getting resource info: {e}")
        
        log_verbose("Model preloading and optimization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during model preloading and optimization: {e}")
        return False

def main(num_workers_arg, batch_size_arg, analyze_performance_arg=False, verbose_logging_arg=False, model_arg=None, force_cpu_arg=False):
    global analyze_performance, verbose_logging, WHISPER_MODEL
    analyze_performance = analyze_performance_arg
    verbose_logging = verbose_logging_arg
    
    # Override configuration with command line arguments
    if model_arg:
        # Handle both short names and full HF names
        if "/" not in model_arg:
            # Convert short names to full HF names
            model_mapping = {
                "tiny": "openai/whisper-tiny",
                "base": "openai/whisper-base", 
                "small": "openai/whisper-small",
                "medium": "openai/whisper-medium",
                "large": "openai/whisper-large",
                "large-v2": "openai/whisper-large-v2",
                "large-v3": "openai/whisper-large-v3",
                "turbo": "openai/whisper-large-v3-turbo",
                "distil-large-v3": "distil-whisper/distil-large-v3",
                "distil-medium.en": "distil-whisper/distil-medium.en"
            }
            WHISPER_MODEL = model_mapping.get(model_arg, model_arg)
        else:
            WHISPER_MODEL = model_arg
        log_verbose(f"Model overridden via command line: {WHISPER_MODEL}")
    
    # Handle force CPU option
    if force_cpu_arg:
        log_essential("🔧 Force CPU mode enabled - disabling CUDA operations")
        # Temporarily disable CUDA for this run
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # Also disable cuDNN
        os.environ['CUDNN_ENABLED'] = '0'

    # Register cleanup handlers with process manager
    process_manager.register_cleanup_handler(cleanup_whisper)
    process_manager.register_cleanup_handler(cleanup_gpu_resources)
    process_manager.register_cleanup_handler(cleanup_lock_files)
    process_manager.register_cleanup_handler(lambda: model_cache_manager.clear())
    process_manager.register_cleanup_handler(lambda: gpu_manager.stop_monitoring())
    process_manager.register_cleanup_handler(lambda: system_monitor.stop_monitoring())

    # Create required directories
    ensure_dir(AUDIO_DIR)
    ensure_dir(TRANSCRIPTIONS_DIR)
    ensure_dir(CORRECTED_DIR)
    ensure_dir(LOG_DIR)

    # Start monitoring systems
    try:
        gpu_manager.start_monitoring()
        system_monitor.start_monitoring()
        log_verbose("System monitoring started")
    except Exception as e:
        logger.warning(f"Error starting monitoring systems: {e}")

    # Preload and optimize the model
    if not preload_and_optimize_model():
        logger.error("Failed to preload and optimize model. Exiting.")
        return

    # Main processing loop
    while not process_manager.is_shutdown_requested():
        try:
            # Optimize cache before processing
            model_cache_manager.optimize_cache()
            
            # Collect tasks
            transcription_tasks = collect_transcription_tasks()
            correction_tasks = collect_correction_tasks()

            if not transcription_tasks and not correction_tasks:
                log_essential("No new tasks found. Waiting for new files...")
                # Check system health during idle time
                try:
                    health_status = system_monitor.get_health_status()
                    if health_status['status'] == 'critical':
                        logger.error("System health critical, consider reducing load")
                    elif health_status['status'] == 'warning':
                        logger.warning("System health warning detected")
                except Exception as e:
                    logger.debug(f"Error checking system health: {e}")
                
                # Wait with ability to respond to shutdown
                process_manager.wait_for_shutdown(timeout=CHECK_INTERVAL)
                continue

            # Process transcription tasks
            if transcription_tasks:
                log_essential(f"Found {len(transcription_tasks)} files to transcribe")
                
                # Check GPU health before processing
                try:
                    if torch.cuda.is_available():
                        for gpu_id in range(torch.cuda.device_count()):
                            if not gpu_manager.monitor_temperature(gpu_id):
                                logger.warning(f"GPU {gpu_id} temperature too high, reducing batch size")
                                batch_size_arg = max(1, batch_size_arg // 2)
                except Exception as e:
                    logger.debug(f"Error checking GPU health: {e}")
                
                # Split tasks into batches
                task_batches = split_tasks_into_batches(transcription_tasks, batch_size_arg)
                
                # Process batches with multiprocessing
                with ProcessPoolExecutor(max_workers=num_workers_arg) as executor:
                    # Submit batches to workers
                    futures = []
                    for batch in task_batches:
                        future = executor.submit(process_transcription_batch, (batch, verbose_logging))
                        futures.append(future)
                    
                    # Wait for completion with retry logic
                    failed_batches = []
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Error in transcription batch: {e}")
                            # Get the batch that failed
                            batch_index = futures.index(future)
                            failed_batches.append(task_batches[batch_index])
                    
                    # Retry failed batches with reduced batch size
                    if failed_batches:
                        logger.warning(f"Retrying {len(failed_batches)} failed batches with reduced batch size...")
                        for batch in failed_batches:
                            # Split failed batch into smaller chunks
                            smaller_batches = split_tasks_into_batches(batch, max(1, batch_size_arg // 2))
                            for small_batch in smaller_batches:
                                try:
                                    future = executor.submit(process_transcription_batch, (small_batch, verbose_logging))
                                    future.result()
                                except Exception as retry_error:
                                    logger.error(f"Retry failed for batch: {retry_error}")

            # Process correction tasks
            if correction_tasks:
                log_essential(f"Found {len(correction_tasks)} files to correct")
                
                # Process corrections sequentially to avoid overwhelming Ollama
                for task in correction_tasks:
                    if process_manager.is_shutdown_requested():
                        break
                    correct_file(task)

            # Create backup after processing
            try:
                backup_dir = create_backup()
                log_verbose(f"Created backup in {backup_dir}")
                
                # Clean up old backups
                cleanup_old_backups()
            except Exception as e:
                logger.error(f"Error creating backup: {e}")

            # Log performance summary
            try:
                if verbose_logging:
                    performance_summary = system_monitor.get_performance_summary(duration_minutes=30)
                    if performance_summary:
                        log_verbose("📊 Performance Summary (last 30 minutes):")
                        log_verbose(f"   CPU: avg {performance_summary['cpu']['avg']:.1f}%, max {performance_summary['cpu']['max']:.1f}%")
                        log_verbose(f"   Memory: avg {performance_summary['memory']['avg']:.1f}%, max {performance_summary['memory']['max']:.1f}%")
                        
                        # GPU performance summary
                        for gpu_id, gpu_stats in performance_summary.get('gpu_summary', {}).items():
                            log_verbose(f"   GPU {gpu_id}: avg {gpu_stats['memory_percent']['avg']:.1f}% memory, "
                                       f"max temp {gpu_stats['temperature']['max']:.0f}°C")
            except Exception as e:
                logger.debug(f"Error logging performance summary: {e}")

            # Wait before next check
            process_manager.wait_for_shutdown(timeout=CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, initiating shutdown...")
            process_manager.request_shutdown()
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            # Check if this is a critical error that should cause shutdown
            if isinstance(e, (GPUError, ModelCacheError)):
                logger.error("Critical system error detected, initiating shutdown...")
                process_manager.request_shutdown()
                break
            else:
                # Wait before retrying
                process_manager.wait_for_shutdown(timeout=CHECK_INTERVAL)

    # Final cleanup is handled by process manager cleanup handlers
    logger.info("Main processing loop completed")

if __name__ == "__main__":
    try:
        log_essential("🚀 Starting TransFixer (Enhanced Edition)...")
        parser = argparse.ArgumentParser(description="TransFixer: Transcribe and correct audio files.")
        parser.add_argument("--num-workers", type=int, choices=[1, 2], default=1, help="Number of worker processes for transcription (1 or 2). Default is 1.")
        parser.add_argument("--batch-size", type=int, default=8, help="Batch size for transcription tasks. Default is 8.")
        parser.add_argument("--cleanup-locks", action="store_true", help="Remove all lock files and exit. Use this if transcription was interrupted.")
        parser.add_argument("--analyze-performance", action="store_true", help="Enable detailed performance analysis during transcription.")
        parser.add_argument("--verbose-logging", action="store_true", help="Enable verbose logging output. Default is minimal logging (only progress and current transcription).")
        
        # Model selection arguments
        parser.add_argument("--model", type=str, default=None, 
                          help="Whisper model to use. Options: 'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'turbo', 'distil-large-v3', 'distil-medium.en' or full HuggingFace model names like 'openai/whisper-large-v3'")
        parser.add_argument("--list-models", action="store_true", help="List available models and exit.")
        parser.add_argument("--force-cpu", action="store_true", help="Force CPU-only mode (disable CUDA) - useful when CUDA/cuDNN has issues.")
        
        # System management arguments
        parser.add_argument("--system-info", action="store_true", help="Show system information and exit.")
        parser.add_argument("--cache-info", action="store_true", help="Show model cache information and exit.")
        
        args = parser.parse_args()
        
        # Handle system info option
        if args.system_info:
            try:
                resources = system_monitor.get_system_resources()
                health = system_monitor.get_health_status()
                
                print("🖥️  System Information:")
                print(f"   Platform: {resources['system']['platform']}")
                print(f"   WSL: {resources['system']['is_wsl']}")
                print(f"   CPU: {resources['cpu']['count']} cores, {resources['cpu']['percent']:.1f}% usage")
                print(f"   Memory: {resources['memory']['percent']:.1f}% used ({resources['memory']['used']/(1024**3):.1f}GB/{resources['memory']['total']/(1024**3):.1f}GB)")
                
                if resources['gpu']['cuda_available']:
                    print(f"   CUDA: Available ({len(resources['gpu']['devices'])} devices)")
                    for gpu in resources['gpu']['devices']:
                        print(f"     GPU {gpu['id']}: {gpu['name']} - {gpu['memory_percent']:.1f}% memory used, {gpu['temperature']}°C")
                else:
                    print("   CUDA: Not available")
                
                print(f"\n📊 Health Status: {health['status'].upper()}")
                if health.get('alerts'):
                    print("   Alerts:")
                    for alert in health['alerts']:
                        print(f"     - {alert['message']}")
                        
            except Exception as e:
                print(f"Error getting system info: {e}")
            sys.exit(0)
        
        # Handle cache info option
        if args.cache_info:
            try:
                cache_info = model_cache_manager.get_cache_info()
                gpu_stats = gpu_manager.get_performance_stats()
                
                print("💾 Model Cache Information:")
                print(f"   Total entries: {cache_info['total_entries']}")
                print(f"   Total memory usage: {cache_info['total_memory_mb']:.1f} MB")
                
                if cache_info['entries']:
                    print("   Cache entries:")
                    for key, entry in cache_info['entries'].items():
                        status = "EXPIRED" if entry['is_expired'] else "ACTIVE"
                        print(f"     {key}: {entry['memory_size_mb']:.1f}MB, used {entry['use_count']} times, {status}")
                
                if gpu_stats:
                    print(f"\n🎮 GPU Performance Stats:")
                    for gpu_id, stats in gpu_stats.get('gpus', {}).items():
                        print(f"   GPU {gpu_id}: avg {stats['avg_memory_usage']:.1f}% memory, max temp {stats['max_temperature']:.0f}°C")
                        print(f"     Warnings: {stats['memory_warnings']} memory, {stats['temperature_warnings']} temperature")
                        
            except Exception as e:
                print(f"Error getting cache info: {e}")
            sys.exit(0)
        
        # Handle list models option
        if args.list_models:
            print("🎯 Available Whisper Models:")
            print("\n📊 TRANSFORMERS MODELS (HuggingFace):")
            print("   Performance Models:")
            print("   • tiny       - Fastest, lowest accuracy (39 MB)")
            print("   • base       - Fast, basic accuracy (74 MB)")
            print("   • small      - Balanced speed/accuracy (244 MB)")
            print("   • medium     - Good accuracy, moderate speed (769 MB)")
            print("   • large      - High accuracy (1550 MB)")
            print("   • large-v2   - Improved accuracy (1550 MB)")
            print("   • large-v3   - Best accuracy (1550 MB)")
            print("   • turbo      - 8x faster than large-v3, similar to large-v2 accuracy")
            print("   \n   Specialized Models:")
            print("   • distil-large-v3    - 6x faster than large-v3, minimal accuracy loss")
            print("   • distil-medium.en   - English-only, very fast")
            print("\n🤗 Full HuggingFace Model Names:")
            print("   • openai/whisper-tiny")
            print("   • openai/whisper-base") 
            print("   • openai/whisper-small")
            print("   • openai/whisper-medium")
            print("   • openai/whisper-large")
            print("   • openai/whisper-large-v2")
            print("   • openai/whisper-large-v3")
            print("   • openai/whisper-large-v3-turbo")
            print("   • distil-whisper/distil-large-v3")
            print("   • distil-whisper/distil-medium.en")
            print("\n💡 Usage Examples:")
            print("   python transfixer.py --model large-v3")
            print("   python transfixer.py --model turbo")
            print("   python transfixer.py --model openai/whisper-large-v3")
            print("   python transfixer.py --model distil-large-v3")
            print("   python transfixer.py --force-cpu  # Use this if you have CUDA/cuDNN issues")
            sys.exit(0)
        
        # Handle cleanup locks option
        if args.cleanup_locks:
            log_essential("🧹 Cleaning up lock files...")
            cleanup_lock_files()
            log_essential("✅ Lock file cleanup completed. You can now run TransFixer normally.")
            sys.exit(0)
            
        # Set shutdown timeout for process manager
        process_manager.set_shutdown_timeout(30)
        
        # Run main application
        main(args.num_workers, args.batch_size, args.analyze_performance, args.verbose_logging, args.model, args.force_cpu)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in __main__, requesting graceful shutdown...")
        process_manager.request_shutdown()
    except SystemExit as e:
        logger.info(f"SystemExit caught in __main__ ({e}), proceeding to final cleanup.")
    except Exception as e:
        logger.error(f"Unhandled exception in __main__: {e}", exc_info=True)
        process_manager.request_shutdown()
    finally:
        logger.info("Performing final script cleanup...")
        
        # The process manager will handle all cleanup through registered handlers
        # No need for manual cleanup here as it's handled automatically
        
        logger.info("Enhanced TransFixer shutdown complete.")