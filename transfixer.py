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
    MEMORY_SAFETY_FACTOR, MAX_BATCH_SIZE, PERFORMANCE_MODE,
    ENABLE_TORCH_COMPILE, TORCH_COMPILE_MODE, TORCH_COMPILE_FULLGRAPH, 
    ATTENTION_IMPLEMENTATION, ENABLE_FLASH_ATTENTION, ENABLE_SDPA
)
from tqdm import tqdm
import re
import psutil
import GPUtil
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError
import threading
import argparse # Added argparse

# Try to import flash-attn
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

# Import advanced performance monitoring
try:
    from advanced_performance_monitor import (
        start_performance_monitoring, 
        stop_performance_monitoring,
        log_transcription_start, 
        log_transcription_complete,
        get_performance_summary,
        export_performance_metrics
    )
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False

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
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
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

# Global flag for verbose logging (will be set by command line argument)
verbose_logging = False

def log_verbose(message, level=logging.INFO):
    """Log a message only if verbose logging is enabled."""
    if verbose_logging:
        logger.log(level, message)

def log_essential(message, level=logging.INFO):
    """Log essential messages that should always be shown."""
    # Always log at INFO level, but use print for non-verbose mode to avoid timestamp clutter
    if verbose_logging:
        logger.log(level, message)
    else:
        if level >= logging.WARNING:
            logger.log(level, message)  # Always show warnings and errors
        else:
            print(message)  # Essential info without timestamp

# Log import status after logger is configured (verbose only)
if FLASH_ATTENTION_AVAILABLE:
    log_verbose("✓ Flash Attention 2 available")
else:
    log_verbose("⚠ Flash Attention not available. Install with: pip install flash-attn", logging.WARNING)

if PERFORMANCE_MONITORING_AVAILABLE:
    log_verbose("✓ Advanced performance monitoring available")
else:
    log_verbose("⚠ Advanced performance monitoring not available", logging.WARNING)

# Global variables for model caching
model_cache = {
    'model': None,
    'processor': None,
    'pipeline': None,
    'last_used': None,
    'device': None,
    'lock': threading.Lock()
}

# Global flag to track if performance analysis has been done
performance_analysis_done = False

# Global flag for performance analysis (set by command line argument)
analyze_performance = False

MODEL_CACHE_TIMEOUT = 3600  # 60 minutes
MIN_MEMORY_THRESHOLD = 0.8  # 80% memory usage threshold

def log_performance_analysis_during_transcription():
    """Log performance analysis during active transcription when GPU is actually being used."""
    global performance_analysis_done
    
    if performance_analysis_done or not PERFORMANCE_MONITORING_AVAILABLE:
        return
    
    performance_analysis_done = True
    
    log_verbose("=" * 60)
    log_verbose("📊 PERFORMANCE ANALYSIS DURING ACTIVE TRANSCRIPTION:")
    log_verbose("=" * 60)
    
    try:
        from advanced_performance_monitor import PerformanceOptimizer, PerformanceMetrics
        from datetime import datetime
        
        # Wait a moment for GPU utilization to stabilize
        time.sleep(2)
        
        # Get current metrics during active transcription
        resources = get_system_resources()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=resources['cpu_percent'],
            memory_percent=resources['memory_percent'],
            memory_used_gb=resources['memory_available'] / (1024**3),
            gpu_memory_used_mb=resources['gpu_info'][0]['memory_used'] if resources['gpu_info'] else None,
            gpu_memory_total_mb=resources['gpu_info'][0]['memory_total'] if resources['gpu_info'] else None,
            gpu_utilization=None,
            gpu_temperature=resources['gpu_info'][0]['temperature'] if resources['gpu_info'] else None
        )
        
        optimizer = PerformanceOptimizer()
        suggestions = optimizer.analyze_performance(metrics)
        optimal_settings = optimizer.suggest_optimal_settings(metrics)
        
        # Log current system state during transcription
        log_verbose(f"💻 System Under Load (during transcription):")
        log_verbose(f"   CPU: {metrics.cpu_percent:.1f}% | Memory: {metrics.memory_percent:.1f}%")
        if metrics.gpu_memory_used_mb:
            gpu_usage = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            log_verbose(f"   GPU Memory: {gpu_usage:.1f}% ({metrics.gpu_memory_used_mb:.0f}MB/{metrics.gpu_memory_total_mb:.0f}MB)")
            if metrics.gpu_temperature:
                log_verbose(f"   GPU Temperature: {metrics.gpu_temperature}°C")
        
        # Log suggestions based on real working load
        if suggestions:
            log_verbose(f"🔧 Optimization Suggestions (based on real workload):")
            for suggestion in suggestions:
                log_verbose(f"   {suggestion}")
        else:
            log_verbose(f"🎯 System performance looks optimal during transcription!")
        
        # Log optimal settings based on actual usage
        if optimal_settings:
            log_verbose(f"⚙️  Recommended Settings (based on actual GPU usage):")
            for key, value in optimal_settings.items():
                log_verbose(f"   {key}: {value}")
                
            # Compare with current config settings
            from config import PERFORMANCE_MODE, MAX_BATCH_SIZE, DEFAULT_CHUNK_LENGTH
            log_verbose(f"📋 Current Config vs Recommended:")
            log_verbose(f"   Performance Mode: {PERFORMANCE_MODE} → {optimal_settings.get('performance_mode', 'current is fine')}")
            if 'batch_size' in optimal_settings:
                log_verbose(f"   Batch Size: {MAX_BATCH_SIZE} → {optimal_settings['batch_size']}")
            if 'chunk_length' in optimal_settings:
                log_verbose(f"   Chunk Length: {DEFAULT_CHUNK_LENGTH} → {optimal_settings['chunk_length']}")
        
        log_verbose(f"💡 Note: These recommendations are based on actual GPU utilization during transcription.")
        log_verbose(f"💡 You can adjust settings in config.py or use --batch-size parameter.")
                
    except Exception as e:
        logger.warning(f"Could not generate performance analysis: {e}")
    
    log_verbose("=" * 60)

def should_unload_model():
    """Determine if the model should be unloaded based on system resources."""
    resources = get_system_resources()
    
    # Check memory usage
    if resources['memory_percent'] > MIN_MEMORY_THRESHOLD * 100:
        log_verbose("Memory usage high, unloading model")
        return True
    
    # Check GPU memory if available
    if resources['gpu_info']:
        gpu = resources['gpu_info'][0]
        if gpu['memory_used'] / gpu['memory_total'] > MIN_MEMORY_THRESHOLD:
            log_verbose("GPU memory usage high, unloading model")
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
        log_verbose("Model cache expired")
        return False
    
    # Check if we should unload due to resource constraints
    if should_unload_model():
        return False
    
    return True

def initialize_whisper():
    """Initialize Whisper model with optimized settings."""
    global model_cache
    
    # Check if we can use the cached model
    with model_cache['lock']:
        if is_model_cache_valid():
            log_verbose("Using cached model")
            model_cache['last_used'] = datetime.now()
            return
    
    process_id = os.getpid()
    log_verbose(f"Process {process_id}: Initializing Whisper model")
    
    # Smart GPU selection based on available memory
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            best_gpu = select_best_gpu()
            device = f"cuda:{best_gpu}" if best_gpu is not None else "cuda:0"
            log_verbose(f"Process {process_id} using smart-selected GPU {best_gpu} of {gpu_count} available GPUs")
        else:
            device = "cuda:0"
            log_verbose(f"Process {process_id} using single available GPU")
    else:
        device = "cpu"
        log_verbose(f"Process {process_id} using CPU")

    # Always use float16 for GPU, float32 for CPU
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    try:
        with model_cache['lock']:
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

            # Get current resource info and calculate optimal settings
            resources = get_system_resources()
            optimal_batch_size = calculate_conservative_batch_size(device)
            
            # Calculate optimal chunk length based on performance mode and available memory
            chunk_length_s = DEFAULT_CHUNK_LENGTH
            if device != "cpu" and resources['gpu_info']:
                gpu = resources['gpu_info'][0]
                available_memory_gb = (gpu['memory_total'] - gpu['memory_used']) / 1024
                
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
        cleanup_whisper()
        raise

def cleanup_whisper():
    """Clean up Whisper model resources with improved memory management."""
    global model_cache
    
    with model_cache['lock']:
        if model_cache['model'] is None:
            return
            
        process_id = os.getpid()
        device = model_cache.get('device', 'cpu')
        
        log_verbose(f"Process {process_id}: Cleaning up Whisper model resources...")
        
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
                
                log_verbose(f"Process {process_id}: Enhanced cleanup completed for {device}")
                
        except Exception as e:
            logger.error(f"Process {process_id}: Error during Whisper cleanup: {e}")

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
            
            log_verbose(f"Performance mode: {PERFORMANCE_MODE} | Available: {available_memory_mb}MB | "
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
            log_verbose(f"GPU {gpu.id}: {available}MB available out of {gpu.memoryTotal}MB total")
            if available > max_available:
                max_available = available
                best_gpu = gpu.id
                
        log_verbose(f"Selected GPU {best_gpu} with {max_available}MB available memory")
        return best_gpu
    except Exception as e:
        logger.warning(f"Error selecting best GPU: {e}, using GPU 0")
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
        log_verbose(f"Process {process_id}: Initial resources - CPU: {initial_resources['cpu_percent']}%, "
                   f"Memory: {initial_resources['memory_percent']}%")
        if initial_resources['gpu_info']:
            gpu = initial_resources['gpu_info'][0]
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
        log_verbose(f"Process {process_id}: Current resources - CPU: {current_resources['cpu_percent']}%, "
                   f"Memory: {current_resources['memory_percent']}%")
        
        if current_resources['gpu_info']:
            gpu = current_resources['gpu_info'][0]
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
            log_verbose(f"Process {process_id}: Final resources - CPU: {final_resources['cpu_percent']}%, "
                       f"Memory: {final_resources['memory_percent']}%")
            if final_resources['gpu_info']:
                gpu = final_resources['gpu_info'][0]
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
    initialize_whisper()
    
    # Log performance analysis after model is loaded and about to start transcription
    # This will capture the real GPU usage during active transcription
    if not performance_analysis_done and analyze_performance:
        # Start transcription in a separate thread to analyze while it's running
        def delayed_analysis():
            time.sleep(3)  # Wait for transcription to start using GPU
            log_performance_analysis_during_transcription()
        
        analysis_thread = threading.Thread(target=delayed_analysis, daemon=True)
        analysis_thread.start()
    
    successful_transcriptions = 0
    failed_transcriptions = 0
    
    try:
        with model_cache['lock']:
            pipeline = model_cache['pipeline']
            device = model_cache['device']
            
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
        initialize_whisper()
        
        # Get current resource info
        resources = get_system_resources()
        if resources['gpu_info']:
            gpu = resources['gpu_info'][0]
            log_verbose(f"Model loaded. GPU Memory usage: {gpu['memory_used']}MB/{gpu['memory_total']}MB "
                       f"({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
            
            # Check if we can enable additional optimizations
            if torch.cuda.is_available():
                # Enable optimized attention if available (for newer PyTorch versions)
                try:
                    with model_cache['lock']:
                        if model_cache['model'] is not None:
                            # Try to enable flash attention or other optimizations
                            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                                log_verbose("Scaled dot product attention available - model should use optimized attention")
                            
                            # Enable torch compile if available (PyTorch 2.0+)
                            if ENABLE_TORCH_COMPILE and hasattr(torch, 'compile'):
                                log_verbose("Enabling PyTorch compile for maximum performance")
                                try:
                                    model = model_cache['model']  # Get model from cache
                                    # Enable static cache for torch.compile compatibility
                                    if hasattr(model, 'generation_config'):
                                        model.generation_config.cache_implementation = "static"
                                    
                                    # Apply torch.compile with optimal settings
                                    compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                                    model_cache['model'] = compiled_model  # Update cache with compiled model
                                    log_verbose("✓ Torch compile enabled - expect 4.5x speed improvement")
                                except Exception as e:
                                    logger.warning(f"Torch compile failed: {e}")
                                
                except Exception as e:
                    logger.warning(f"Could not apply additional optimizations: {e}")
        
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

    # Create required directories
    ensure_dir(AUDIO_DIR)
    ensure_dir(TRANSCRIPTIONS_DIR)
    ensure_dir(CORRECTED_DIR)
    ensure_dir(LOG_DIR)

    # Preload and optimize the model
    if not preload_and_optimize_model():
        logger.error("Failed to preload and optimize model. Exiting.")
        return

    # Main processing loop
    while not shutdown_event.is_set():
        try:
            # Collect tasks
            transcription_tasks = collect_transcription_tasks()
            correction_tasks = collect_correction_tasks()

            if not transcription_tasks and not correction_tasks:
                log_essential("No new tasks found. Waiting for new files...")
                time.sleep(CHECK_INTERVAL)
                continue

            # Process transcription tasks
            if transcription_tasks:
                log_essential(f"Found {len(transcription_tasks)} files to transcribe")
                
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
                    if shutdown_event.is_set():
                        break
                    correct_file(task)

            # Create backup after processing
            backup_dir = create_backup()
            log_verbose(f"Created backup in {backup_dir}")
            
            # Clean up old backups
            cleanup_old_backups()

            # Wait before next check
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, initiating shutdown...")
            shutdown_event.set()
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(CHECK_INTERVAL)  # Wait before retrying

    # Final cleanup
    cleanup_whisper()
    cleanup_lock_files()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        log_essential("🚀 Starting TransFixer (Ollama Edition)...")
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
        
        args = parser.parse_args()
        
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
        main(args.num_workers, args.batch_size, args.analyze_performance, args.verbose_logging, args.model, args.force_cpu)

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