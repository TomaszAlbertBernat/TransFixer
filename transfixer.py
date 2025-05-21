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
from config import OLLAMA_MODEL, CORRECTION_PROMPT, OLLAMA_API_URL, OLLAMA_OPTIONS, NUM_PARALLEL_WHISPER
from tqdm import tqdm
import re
import argparse
import psutil
from whisper_transcriber import WhisperTranscriber
from ollama_corrector import OllamaCorrector
from task_manager import TaskManager
from path_manager import PathManager

# Global variable for persistent transcriber in worker processes
_worker_transcriber = None

def cleanup_worker():
    """
    Clean up resources when the worker process is shutting down.
    This helps ensure the worker transcriber is properly disposed.
    """
    global _worker_transcriber
    
    worker_id = os.getpid()
    logger.info(f"Worker {worker_id}: Shutting down and cleaning up resources")
    
    if _worker_transcriber is not None:
        try:
            # Call any cleanup methods on the transcriber
            _worker_transcriber.cleanup()
            _worker_transcriber = None
            logger.info(f"Worker {worker_id}: Transcriber cleanup successful")
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error during transcriber cleanup: {e}")
    
    # Final GPU memory cleanup
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info(f"Worker {worker_id}: Final GPU memory cleared")
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error during final GPU cleanup: {e}")

def init_worker(model_name, use_torch_compile, generate_kwargs, batch_size=1):
    """
    Initialize the worker process with a persistent WhisperTranscriber instance.
    This function is called once per worker when the Pool is created.
    
    Args:
        model_name: Name of the Whisper model to use
        use_torch_compile: Whether to use torch.compile
        generate_kwargs: Additional kwargs for generation
        batch_size: Number of audio files to process in a batch
    """
    global _worker_transcriber
    
    # Set nice priority for worker processes to prevent system lockup
    try:
        os.nice(10)  # Lower priority (higher nice value)
    except Exception as e:
        pass  # Ignore if not supported on this platform
    
    # Log worker initialization
    worker_id = os.getpid()
    logger.info(f"Initializing worker process {worker_id} with persistent transcriber (batch_size: {batch_size})")
    
    # Register cleanup function to run on process exit
    import atexit
    atexit.register(cleanup_worker)
    
    # Check GPU memory status
    if torch.cuda.is_available():
        try:
            device_id = torch.cuda.current_device()
            allocated_gb = torch.cuda.memory_allocated(device_id) / (1024**3)
            total_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            logger.info(f"Worker {worker_id} GPU {device_id} memory before init: {allocated_gb:.2f}GB / {total_gb:.2f}GB")
            
            # Clear GPU memory before initialization
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Worker {worker_id} error checking GPU memory: {e}")
    
    # Initialize the transcriber
    try:
        _worker_transcriber = WhisperTranscriber(
            model_name=model_name,
            use_torch_compile=use_torch_compile,
            generate_kwargs=generate_kwargs,
            batch_size=batch_size
        )
        logger.info(f"Worker {worker_id} successfully initialized transcriber with batch_size={batch_size}")
    except Exception as e:
        logger.error(f"Worker {worker_id} failed to initialize transcriber: {e}", exc_info=True)
        _worker_transcriber = None

# --- START: Resource Management ---
def get_available_memory():
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)

def get_gpu_memory_info():
    """Get GPU memory usage information if CUDA is available."""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {}
    for i in range(torch.cuda.device_count()):
        gpu_info[i] = {
            'total': torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
            'reserved': torch.cuda.memory_reserved(i) / (1024**3),  # GB
            'allocated': torch.cuda.memory_allocated(i) / (1024**3),  # GB
            'free': (torch.cuda.get_device_properties(i).total_memory - 
                    torch.cuda.memory_reserved(i)) / (1024**3)  # GB
        }
    return gpu_info

def clear_gpu_memory():
    """Clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        return True
    return False

def calculate_optimal_workers(args, min_workers=1):
    """
    Calculate the optimal number of workers based on available resources.
    
    Args:
        args: Command line arguments containing workers setting
        min_workers: Minimum number of workers to use
        
    Returns:
        int: Optimal number of workers
    """
    requested_workers = args.workers
    
    # If no CUDA available, just return the requested number
    if not torch.cuda.is_available():
        return requested_workers
    
    # Check GPU memory
    gpu_info = get_gpu_memory_info()
    if not gpu_info:
        return requested_workers
    
    # Estimate memory required per worker (GB)
    # Whisper large model is ~3GB in FP16
    estimated_memory_per_worker = 2.0  # Less conservative estimate (reduced from 4.0)
    
    # Calculate maximum workers based on GPU memory
    total_free_memory = sum(info['free'] for info in gpu_info.values())
    max_workers_by_gpu = max(min_workers, int(total_free_memory / estimated_memory_per_worker))
    
    # Check system memory too
    system_memory_gb = get_available_memory()
    max_workers_by_ram = max(min_workers, int(system_memory_gb / 2.0))  # Conservative estimate
    
    # Take the minimum of all constraints
    optimal_workers = min(requested_workers, max_workers_by_gpu, max_workers_by_ram)
    
    logger.info(f"Resource analysis: Requested={requested_workers}, GPU allows={max_workers_by_gpu}, " 
                f"RAM allows={max_workers_by_ram}, Using={optimal_workers}")
    
    return optimal_workers

def calculate_optimal_batch_size(requested_batch_size, min_batch_size=1):
    """
    Calculate optimal batch size based on available GPU and system memory.
    
    Args:
        requested_batch_size: User requested batch size
        min_batch_size: Minimum batch size to use (default: 1)
        
    Returns:
        int: Optimal batch size
    """
    # Start with requested batch size
    optimal_batch_size = requested_batch_size
    
    # Check GPU memory if available
    if torch.cuda.is_available():
        try:
            # Get GPU info
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                # Calculate free memory across all GPUs
                total_free_gpu_memory = sum(info['free'] for info in gpu_info.values())
                
                # Estimate memory needed per audio sample in batch (GB)
                # This is approximate and will vary with audio length
                estimated_memory_per_sample = 0.25  # Less conservative estimate (reduced from 0.5)
                
                # Calculate max batch size based on GPU memory, leaving 20% headroom
                max_by_gpu = max(min_batch_size, int((total_free_gpu_memory * 0.8) / estimated_memory_per_sample))
                optimal_batch_size = min(optimal_batch_size, max_by_gpu)
                
                logger.info(f"GPU memory-based batch size calculation: {max_by_gpu} (from {total_free_gpu_memory:.2f}GB free)")
        except Exception as e:
            logger.warning(f"Error calculating GPU-based batch size: {e}")
    
    # Also check system RAM
    try:
        available_ram_gb = get_available_memory()
        # Leave at least 2GB free and assume we need ~250MB per sample for intermediate processing
        ram_based_batch_size = max(min_batch_size, int((available_ram_gb - 2) / 0.25))
        optimal_batch_size = min(optimal_batch_size, ram_based_batch_size)
        
        logger.info(f"RAM-based batch size calculation: {ram_based_batch_size} (from {available_ram_gb:.2f}GB available)")
    except Exception as e:
        logger.warning(f"Error calculating RAM-based batch size: {e}")
    
    # Apply sanity limits
    if optimal_batch_size < min_batch_size:
        logger.warning(f"Calculated batch size {optimal_batch_size} is below minimum {min_batch_size}, using minimum")
        optimal_batch_size = min_batch_size
    
    if optimal_batch_size != requested_batch_size:
        logger.info(f"Dynamically adjusted batch size from {requested_batch_size} to {optimal_batch_size} based on available memory")
    
    return optimal_batch_size
# --- END: Resource Management ---

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
    parser.add_argument('--use-torch-compile', action='store_true',
                      help='Enable torch.compile for Whisper model (requires PyTorch 2.0+).')
    parser.add_argument('--hybrid-mode', action='store_true',
                      help='Use both multi-worker and batching modes together (each worker processes batches)')
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
WHISPER_MODEL = "openai/whisper-large-v3-turbo"  # Using v3-turbo for better performance
WHISPER_TEMPERATURE = 0.0  # Temperature for transcription generation
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

def signal_handler(signum, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, cleaning up...")
    # Lock file cleanup will be handled by the TaskManager instance if it's accessible here
    # or by the main script's finally block.
    # For now, direct call to a global TaskManager instance if we decide to make one, or handled by main.
    # This will be refined once TaskManager is integrated into main.
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

def transcribe_file_pooled(args_tuple):
    """
    Transcribes an audio file. Designed to be called by a multiprocessing Pool.
    Uses the persistent transcriber initialized in the worker process.
    """
    global _worker_transcriber
    
    # Unpack arguments including the TaskManager instance
    task_info, whisper_model_name, use_torch_compile_arg, generate_kwargs_arg, task_manager_instance_for_pool = args_tuple
    audio_path = task_info['audio_path']
    trans_path = task_info['trans_path']
    worker_id = os.getpid()

    if not task_manager_instance_for_pool:
        logger.error(f"Worker {worker_id}: TaskManager instance not provided to pooled transcription for {audio_path}. Skipping.")
        return

    # Use TaskManager for locking
    if task_manager_instance_for_pool.is_locked(audio_path, lock_type="transcription") or \
       task_manager_instance_for_pool.is_locked(audio_path, lock_type="process"):
        logger.info(f"Worker {worker_id}: Skipping transcription for {audio_path} (pooled): file is locked.")
        return
    
    # Check if transcription already exists
    if os.path.exists(trans_path):
        # Already processed, no need to continue
        return

    if not task_manager_instance_for_pool.acquire_lock(audio_path, lock_type="transcription"):
        logger.error(f"Worker {worker_id}: Could not acquire transcription lock for {audio_path} (pooled). Skipping.")
        return
    
    success = False
    try:
        logger.info(f"Worker {worker_id}: Starting transcription of {audio_path}")
        
        # Check if we have a valid transcriber
        if _worker_transcriber is None:
            logger.warning(f"Worker {worker_id}: No persistent transcriber available, initializing one-time transcriber")
            # Fall back to creating a temporary transcriber
            with WhisperTranscriber(model_name=whisper_model_name, use_torch_compile=use_torch_compile_arg, generate_kwargs=generate_kwargs_arg) as temp_transcriber:
                result = process_with_transcriber(temp_transcriber, audio_path, trans_path, task_manager_instance_for_pool)
                success = result["success"]
        else:
            # Use the persistent transcriber
            logger.info(f"Worker {worker_id}: Using persistent transcriber for {audio_path}")
            result = process_with_transcriber(_worker_transcriber, audio_path, trans_path, task_manager_instance_for_pool)
            success = result["success"]
                
    except Exception as e:
        logger.error(f"Worker {worker_id}: Unexpected error during pooled transcription of {audio_path}: {e}", exc_info=True)
    finally:
        # Check for GPU memory issues
        if torch.cuda.is_available():
            # Only do a light cleanup here since we're keeping the transcriber loaded
            try:
                # Just report memory usage without clearing cache
                allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"Worker {worker_id}: GPU memory after transcription: {allocated_gb:.2f}GB allocated")
                
                # Only clear cache if memory usage is very high (80% of what's reported as available)
                if allocated_gb > 10.0:  # Arbitrary high threshold
                    logger.warning(f"Worker {worker_id}: High GPU memory usage detected, clearing cache")
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Worker {worker_id}: Error checking GPU memory: {e}")
                
        task_manager_instance_for_pool.release_lock(audio_path, lock_type="transcription")
        if success:
            logger.info(f"Worker {worker_id}: Successfully completed transcription of {audio_path}")
        else:
            logger.error(f"Worker {worker_id}: Failed to transcribe {audio_path}")

def process_with_transcriber(transcriber, audio_path, trans_path, task_manager):
    """
    Process a single audio file with the provided transcriber instance.
    Extracted as a separate function to avoid code duplication.
    
    Args:
        transcriber: WhisperTranscriber instance
        audio_path: Path to the audio file
        trans_path: Path where transcription should be saved
        task_manager: TaskManager instance
        
    Returns:
        dict: Result with success status
    """
    success = False
    for attempt in range(MAX_RETRIES):
        logger.info(f"Transcription attempt {attempt + 1}/{MAX_RETRIES} for {audio_path}")
        
        # Check if we need to clear memory between retries
        if attempt > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        result = transcriber.transcribe_file(audio_path)
        transcription = result.get("transcription")

        if transcription is not None:
            ensure_dir(os.path.dirname(trans_path)) 
            with open(trans_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            if len(transcription.strip()) >= task_manager.min_chars:
                logger.info(f"Successfully transcribed {audio_path}")
                success = True
                break
            else:
                logger.warning(f"Transcription too short for {audio_path} (length {len(transcription.strip()) if transcription else 0}), retrying...")
        else:
            logger.error(f"Transcription failed for {audio_path}: {result.get('error')}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(5)
    
    if not success:
         logger.error(f"Max retries reached for {audio_path}. Transcription failed or was too short.")
    
    return {"success": success}

def correct_file(args, corrector_instance: OllamaCorrector = None, task_manager_instance: TaskManager = None):
    """Correct a transcription using Ollama API and TaskManager for locks."""
    # Handle args as a dictionary, not as a tuple
    trans_path = args['trans_path']
    corrected_path = args['corrected_path']
    
    if not task_manager_instance:
        logger.error(f"TaskManager instance not provided to correct_file for {trans_path}. Skipping lock management and correction.")
        return

    # Use TaskManager for locking
    if task_manager_instance.is_locked(trans_path, lock_type="correction") or \
       task_manager_instance.is_locked(trans_path, lock_type="process"):
        logger.info(f"Skipping correction for {trans_path}: file is locked.")
        return

    if os.path.exists(corrected_path):
        logger.info(f"Skipping correction for {trans_path}: corrected file already exists.")
        return
    
    # Validate source transcription using TaskManager's internal method if accessible or its min_chars
    # For now, assume task collection (which uses TaskManager) already did this.
    # Re-validating here can be redundant if task list is fresh.
    # if not task_manager_instance._is_valid_transcription(trans_path): # _is_valid_transcription is protected
    # For robustness, we can read and check length against task_manager_instance.min_chars
    try:
        with open(trans_path, "r", encoding="utf-8") as f_val:
            if len(f_val.read().strip()) < task_manager_instance.min_chars:
                logger.warning(f"Skipping correction for {trans_path}: source transcription is invalid or too short based on TaskManager's min_chars.")
                return
    except Exception as e_val:
        logger.error(f"Error reading source transcription {trans_path} for validation in correct_file: {e_val}")
        return

    if not task_manager_instance.acquire_lock(trans_path, lock_type="correction"):
        logger.error(f"Could not acquire correction lock for {trans_path}. Skipping.")
        return
    
    local_corrector_instance = corrector_instance
    if local_corrector_instance is None:
        logger.info("No OllamaCorrector instance provided to correct_file, attempting to create a default one.")
        if not all([OLLAMA_API_URL, OLLAMA_MODEL, CORRECTION_PROMPT]):
            logger.error("Cannot create default OllamaCorrector: Missing global config vars.")
            task_manager_instance.release_lock(trans_path, lock_type="correction") # Release lock before returning
            return
        local_corrector_instance = OllamaCorrector(
            api_url=OLLAMA_API_URL,
            model=OLLAMA_MODEL,
            default_correction_prompt=CORRECTION_PROMPT,
            ollama_options=OLLAMA_OPTIONS
        )

    try:
        logger.info(f"Starting correction of {trans_path} using Ollama model {local_corrector_instance.model}")
        with open(trans_path, "r", encoding="utf-8") as f:
            text_to_correct = f.read()
        
        correction_result = local_corrector_instance.correct_text(
            text_to_correct=text_to_correct,
            original_filename=os.path.basename(trans_path)
        )
        corrected_text = correction_result["corrected_text"]
        error_message = correction_result["error"]

        if error_message:
            logger.error(f"Correction for {trans_path} failed: {error_message}")
            return
        if not corrected_text:
            logger.warning(f"Ollama returned empty correction for {trans_path}.")
            return

        # task_manager_instance._ensure_output_dir_exists(corrected_path) # Done by collector
        ensure_dir(os.path.dirname(corrected_path))
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(corrected_text)
        logger.info(f"Successfully corrected {trans_path} and saved to {corrected_path}")

    except Exception as e:
        logger.error(f"Unexpected error in correct_file function for {trans_path}: {e}", exc_info=True)
    finally:
        task_manager_instance.release_lock(trans_path, lock_type="correction")

def handle_transcription_phase(args, path_manager: PathManager, task_manager: TaskManager, 
                             whisper_transcriber_instance: WhisperTranscriber | None = None) -> WhisperTranscriber | None:
    """
    Handles the transcription phase of the processing cycle.
    
    Args:
        args: Command line arguments
        path_manager: PathManager instance
        task_manager: TaskManager instance
        whisper_transcriber_instance: Optional existing WhisperTranscriber instance
        
    Returns:
        WhisperTranscriber | None: The WhisperTranscriber instance if created, None otherwise
    """
    logger.info("-" * 10 + " Phase 1: Transcription " + "-" * 10)
    
    # Check and log current resource status
    logger.info(f"Available system memory: {get_available_memory():.2f} GB")
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            for gpu_id, mem_info in gpu_info.items():
                logger.info(f"GPU {gpu_id}: {mem_info['free']:.2f} GB free / {mem_info['total']:.2f} GB total")
    
    # Clear GPU memory before starting transcription
    clear_gpu_memory()
    
    all_trans_tasks_structured = task_manager.collect_transcription_tasks()

    if not all_trans_tasks_structured:
        logger.info("No new audio files to transcribe.")
        return whisper_transcriber_instance

    logger.info(f"Found {len(all_trans_tasks_structured)} files potentially needing transcription.")
    tasks_to_process_this_run = all_trans_tasks_structured  # TaskManager already filtered these

    if not tasks_to_process_this_run:
        logger.info("No new audio files to transcribe in this run after TaskManager collection.")
        return whisper_transcriber_instance

    logger.info(f"Processing {len(tasks_to_process_this_run)} files for transcription this run.")
    
    if args.hybrid_mode:
        # Hybrid mode: Multiple workers, each processing batches
        logger.info("Using hybrid mode (multi-worker + batching)")
        
        # Calculate optimal workers and batch size
        optimal_workers = calculate_optimal_workers(args, min_workers=1)
        optimal_batch_size = calculate_optimal_batch_size(args.batch_size, min_batch_size=1)
        
        logger.info(f"Using {optimal_workers} workers with batch size {optimal_batch_size}")
        
        # Split tasks into batches
        all_batches = []
        for i in range(0, len(tasks_to_process_this_run), optimal_batch_size):
            batch = tasks_to_process_this_run[i:i+optimal_batch_size]
            all_batches.append(batch)
        
        logger.info(f"Split {len(tasks_to_process_this_run)} tasks into {len(all_batches)} batches")
        
        # Create pool with worker initialization
        pool = None
        try:
            with tqdm(total=len(tasks_to_process_this_run), desc=f"Hybrid Transcription ({optimal_workers} workers, batch size {optimal_batch_size})", position=0, leave=True) as pbar_overall:
                # Create pool with worker initialization
                pool = Pool(
                    processes=optimal_workers, 
                    initializer=init_worker, 
                    initargs=(WHISPER_MODEL, args.use_torch_compile, None, optimal_batch_size),
                    maxtasksperchild=None  # Keep workers alive with their initialized transcriber
                )
                
                # Process the batches
                batch_args = [(batch, WHISPER_MODEL, args.use_torch_compile, optimal_batch_size, task_manager) for batch in all_batches]
                
                # Process batches and update progress bar
                for _ in pool.imap_unordered(transcribe_batch_pooled, batch_args):
                    pbar_overall.update(optimal_batch_size)  # This is approximate since batches may not be full
            
            # Gracefully close the pool and wait for workers to finish
            logger.info("All transcription tasks completed, closing worker pool...")
            if pool:
                pool.close()
                pool.join()
            logger.info("Worker pool closed successfully")
                
        except Exception as e:
            logger.error(f"Error during hybrid pool processing: {e}", exc_info=True)
            # Ensure pool is terminated even if an exception occurs
            if pool:
                logger.warning("Terminating pool due to error")
                pool.terminate()
                pool.join()
        finally:
            # Clean up regardless of how we exit
            if pool:
                logger.info("Ensuring pool is closed and joined")
                try:
                    pool.close()
                    pool.join()
                except Exception as pool_cleanup_error:
                    logger.error(f"Error during pool cleanup: {pool_cleanup_error}")
            
            # Always clear GPU memory after all workers are done
            clear_gpu_memory()
    elif args.use_batching:
        # Original batching mode code
        # ... existing code for batching mode ...
        
        # Calculate optimal batch size based on available memory
        optimal_batch_size = calculate_optimal_batch_size(args.batch_size, min_batch_size=1)
        
        # Initialize or update WhisperTranscriber with new batch size if needed
        if whisper_transcriber_instance is None:
            logger.info(f"Initializing WhisperTranscriber for batch mode with batch size {optimal_batch_size}...")
            whisper_transcriber_instance = WhisperTranscriber(
                model_name=WHISPER_MODEL,
                batch_size=optimal_batch_size,
                use_torch_compile=args.use_torch_compile
            )
        elif whisper_transcriber_instance.batch_size != optimal_batch_size:
            logger.info(f"Updating WhisperTranscriber batch size from {whisper_transcriber_instance.batch_size} to {optimal_batch_size}")
            # If batch size property is accessible and updateable, update it
            try:
                whisper_transcriber_instance.batch_size = optimal_batch_size
            except Exception as e:
                logger.warning(f"Could not update batch size: {e}. Creating new instance.")
                # If we can't update, cleanup old and create new
                whisper_transcriber_instance.cleanup()
                whisper_transcriber_instance = WhisperTranscriber(
                    model_name=WHISPER_MODEL,
                    batch_size=optimal_batch_size,
                    use_torch_compile=args.use_torch_compile
                )
        
        # Get all files to process
        audio_paths_for_batch = []
        task_map_for_batch = {}

        for task_info in tasks_to_process_this_run:
            audio_path = task_info['audio_path']
            trans_path = task_info['trans_path']
            if task_manager.acquire_lock(audio_path, lock_type="transcription"):
                audio_paths_for_batch.append(audio_path)
                task_map_for_batch[audio_path] = trans_path
            else:
                logger.info(f"Could not acquire transcription lock for {audio_path} via TaskManager. Skipping.")

        if audio_paths_for_batch:
            # Process in chunks if we have many files, to allow periodic memory cleanup
            MAX_FILES_PER_CHUNK = max(optimal_batch_size * 5, 20)  # Process chunks of files
            audio_path_chunks = [audio_paths_for_batch[i:i + MAX_FILES_PER_CHUNK] 
                                for i in range(0, len(audio_paths_for_batch), MAX_FILES_PER_CHUNK)]
            
            logger.info(f"Processing {len(audio_paths_for_batch)} files in {len(audio_path_chunks)} chunks of up to {MAX_FILES_PER_CHUNK} files each")
            
            # Track progress across all chunks
            with tqdm(total=len(audio_paths_for_batch), desc=f"Overall Transcription (Batch Sz: {optimal_batch_size})", position=0, leave=True) as pbar_overall:
                for chunk_idx, audio_paths_chunk in enumerate(audio_path_chunks):
                    logger.info(f"Processing chunk {chunk_idx+1}/{len(audio_path_chunks)} with {len(audio_paths_chunk)} files")
                    
                    # Check memory before processing chunk
                    if chunk_idx > 0:
                        # Recalculate optimal batch size before each chunk
                        new_batch_size = calculate_optimal_batch_size(args.batch_size, min_batch_size=1)
                        if new_batch_size < optimal_batch_size:
                            logger.warning(f"Memory conditions changed. Reducing batch size from {optimal_batch_size} to {new_batch_size}")
                            optimal_batch_size = new_batch_size
                            try:
                                whisper_transcriber_instance.batch_size = optimal_batch_size
                            except:
                                pass  # Ignore if we can't update directly
                        
                        # Force memory cleanup between chunks
                        clear_gpu_memory()
                    
                    # Process this chunk
                    try:
                        transcription_results = whisper_transcriber_instance.transcribe_batch(
                            audio_paths_chunk, 
                            global_tqdm_instance=pbar_overall
                        )
                    
                        for result in transcription_results:
                            audio_path = result['audio_path']
                            transcription_text = result['transcription']
                            error = result['error']
                            trans_path = task_map_for_batch[audio_path]
    
                            if error:
                                logger.error(f"Transcription failed for {audio_path}: {error}")
                            elif transcription_text is not None:
                                ensure_dir(os.path.dirname(trans_path))
                                with open(trans_path, "w", encoding="utf-8") as f:
                                    f.write(transcription_text)
                                if task_manager._is_valid_transcription(trans_path):
                                    logger.info(f"Successfully transcribed {audio_path} to {trans_path}")
                                else:
                                    logger.warning(f"Transcription for {audio_path} was too short (length {len(transcription_text.strip())}).")
                            else:
                                logger.warning(f"No transcription returned for {audio_path}, and no explicit error.")
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx+1}: {e}", exc_info=True)
                        # Continue with next chunk rather than failing completely

                    # Always clear GPU memory after completing a chunk
                    clear_gpu_memory()
        else:
            logger.info("No files to transcribe in this batch after lock acquisition attempts.")

        # Release locks for this run's batch
        for audio_path_processed in audio_paths_for_batch:
            task_manager.release_lock(audio_path_processed, lock_type="transcription")

    else:
        # Original multi-worker mode code
        # ... existing code for multi-worker mode ...
        
        # Handle parallel processing mode with dynamic worker count based on available resources
        optimal_workers = calculate_optimal_workers(args)
        logger.info(f"Using multiprocessing pool with {optimal_workers} workers for transcription (optimized from requested {args.workers}).")
        
        pooled_tasks_args_final = []
        for task_info in tasks_to_process_this_run:
            pooled_tasks_args_final.append((
                task_info,
                WHISPER_MODEL,
                args.use_torch_compile,
                None,  # generate_kwargs placeholder
                task_manager
            ))

        if pooled_tasks_args_final:
            pool = None
            try:
                with tqdm(total=len(pooled_tasks_args_final), desc=f"Overall Transcription ({optimal_workers} workers)", position=0, leave=True) as pbar_overall:
                    # Create pool with worker initialization to set up persistent transcribers
                    pool = Pool(
                        processes=optimal_workers, 
                        initializer=init_worker, 
                        initargs=(WHISPER_MODEL, args.use_torch_compile, None, 1),  # Use batch_size=1 for non-batching mode
                        # Set maxtasksperchild=None to keep workers alive with their initialized transcriber
                        maxtasksperchild=None
                    )
                    
                    # Process the files, collecting results as they complete
                    for _ in pool.imap_unordered(transcribe_file_pooled, pooled_tasks_args_final):
                        pbar_overall.update(1)
                
                # Gracefully close the pool and wait for workers to finish
                logger.info("All transcription tasks completed, closing worker pool...")
                if pool:
                    pool.close()
                    pool.join()
                logger.info("Worker pool closed successfully")
                    
            except Exception as e:
                logger.error(f"Error during pool processing: {e}", exc_info=True)
                # Ensure pool is terminated even if an exception occurs
                if pool:
                    logger.warning("Terminating pool due to error")
                    pool.terminate()
                    pool.join()
            finally:
                # Clean up regardless of how we exit
                if pool:
                    logger.info("Ensuring pool is closed and joined")
                    try:
                        pool.close()
                        pool.join()
                    except Exception as pool_cleanup_error:
                        logger.error(f"Error during pool cleanup: {pool_cleanup_error}")
                
                # Always clear GPU memory after all parallel workers are done
                clear_gpu_memory()
        else:
            logger.info("No files for pooled transcription after checks.")

    return whisper_transcriber_instance

def handle_correction_phase(task_manager: TaskManager, ollama_corrector_instance: OllamaCorrector | None) -> None:
    """
    Handles the correction phase of the processing cycle.
    
    Args:
        task_manager: TaskManager instance
        ollama_corrector_instance: Optional OllamaCorrector instance
    """
    logger.info("-" * 10 + " Phase 2: Correction " + "-" * 10)
    correct_tasks = task_manager.collect_correction_tasks()
    
    if not correct_tasks:
        logger.info("No new transcriptions to correct")
        return

    logger.info(f"Found {len(correct_tasks)} transcriptions to correct")
    
    if ollama_corrector_instance:
        from functools import partial
        correct_file_with_dependencies = partial(
            correct_file,
            corrector_instance=ollama_corrector_instance,
            task_manager_instance=task_manager
        )
        with Pool(processes=1) as pool:
            pool.map(correct_file_with_dependencies, correct_tasks)
    else:
        logger.warning("OllamaCorrector instance not available. Correction tasks will be problematic.")

def transcribe_batch_pooled(args_tuple):
    """
    Transcribes a batch of audio files using a worker process.
    Uses the persistent transcriber initialized in the worker process.
    
    Args:
        args_tuple: Tuple containing (batch_tasks, model_name, use_torch_compile, batch_size, task_manager)
            batch_tasks: List of task dictionaries with audio_path and trans_path
            model_name: Name of the Whisper model
            use_torch_compile: Whether to use torch.compile
            batch_size: Number of audio files to process in a batch
            task_manager: TaskManager instance
    """
    global _worker_transcriber
    
    # Unpack arguments
    batch_tasks, model_name, use_torch_compile, batch_size, task_manager = args_tuple
    worker_id = os.getpid()
    
    if not task_manager:
        logger.error(f"Worker {worker_id}: TaskManager instance not provided for batch processing. Skipping.")
        return
    
    if len(batch_tasks) == 0:
        logger.warning(f"Worker {worker_id}: Empty batch received. Skipping.")
        return
    
    logger.info(f"Worker {worker_id}: Processing batch of {len(batch_tasks)} files")
    
    # Acquire locks for all files in the batch
    locked_tasks = []
    for task in batch_tasks:
        audio_path = task['audio_path']
        if task_manager.is_locked(audio_path, lock_type="transcription") or \
           task_manager.is_locked(audio_path, lock_type="process"):
            logger.info(f"Worker {worker_id}: Skipping {audio_path} - file is locked.")
            continue
        
        if os.path.exists(task['trans_path']):
            # Already processed
            continue
            
        if task_manager.acquire_lock(audio_path, lock_type="transcription"):
            locked_tasks.append(task)
        else:
            logger.error(f"Worker {worker_id}: Could not acquire transcription lock for {audio_path}. Skipping.")
    
    if not locked_tasks:
        logger.info(f"Worker {worker_id}: No files could be locked in this batch. Skipping.")
        return
    
    try:
        # Prepare batch
        audio_paths = [task['audio_path'] for task in locked_tasks]
        trans_paths = [task['trans_path'] for task in locked_tasks]
        
        logger.info(f"Worker {worker_id}: Starting batch transcription for {len(locked_tasks)} files")
        
        # Check if we have a valid transcriber
        if _worker_transcriber is None:
            logger.warning(f"Worker {worker_id}: No persistent transcriber available, initializing one-time transcriber")
            # Fall back to creating a temporary transcriber
            with WhisperTranscriber(
                model_name=model_name, 
                use_torch_compile=use_torch_compile,
                batch_size=batch_size
            ) as temp_transcriber:
                results = temp_transcriber.transcribe_batch(audio_paths)
        else:
            # Use the persistent transcriber
            logger.info(f"Worker {worker_id}: Using persistent transcriber for batch processing")
            results = _worker_transcriber.transcribe_batch(audio_paths)
        
        # Process results
        for task, result in zip(locked_tasks, results):
            audio_path = task['audio_path']
            trans_path = task['trans_path']
            transcription = result.get("transcription")
            error = result.get("error")
            
            if error:
                logger.error(f"Worker {worker_id}: Transcription failed for {audio_path}: {error}")
                continue
                
            if transcription is not None:
                ensure_dir(os.path.dirname(trans_path))
                with open(trans_path, "w", encoding="utf-8") as f:
                    f.write(transcription)
                
                if len(transcription.strip()) >= task_manager.min_chars:
                    logger.info(f"Worker {worker_id}: Successfully transcribed {audio_path}")
                else:
                    logger.warning(f"Worker {worker_id}: Transcription too short for {audio_path} (length {len(transcription.strip())})")
            else:
                logger.error(f"Worker {worker_id}: Transcription failed for {audio_path}: No transcription returned")
                
    except Exception as e:
        logger.error(f"Worker {worker_id}: Unexpected error during batch transcription: {e}", exc_info=True)
    finally:
        # Clean up memory if needed
        if torch.cuda.is_available():
            # Only do a light cleanup here since we're keeping the transcriber loaded
            try:
                allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"Worker {worker_id}: GPU memory after batch transcription: {allocated_gb:.2f}GB allocated")
                
                # Only clear cache if memory usage is very high
                if allocated_gb > 10.0:  # Arbitrary high threshold
                    logger.warning(f"Worker {worker_id}: High GPU memory usage detected, clearing cache")
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Worker {worker_id}: Error checking GPU memory: {e}")
        
        # Release all locks
        for task in locked_tasks:
            task_manager.release_lock(task['audio_path'], lock_type="transcription")
        
        logger.info(f"Worker {worker_id}: Batch processing completed for {len(locked_tasks)} files")

def main():
    """Main loop to process audio files in phases."""
    args = parse_arguments()
    
    # Validate command line arguments for mode conflicts
    if args.hybrid_mode and args.use_batching:
        logger.warning("Both --hybrid-mode and --use-batching are specified. Hybrid mode will take precedence.")
        args.use_batching = False
    
    # Log selected processing mode
    if args.hybrid_mode:
        logger.info("Using hybrid mode (multi-worker + batching)")
    elif args.use_batching:
        logger.info("Using batch mode with dynamic batch sizing (max batch size: %d)", args.batch_size)
    else:
        logger.info("Using parallel processing mode with dynamic worker allocation (max workers: %d)", args.workers)
    
    # Log memory management strategy based on args
    if args.hybrid_mode:
        logger.info(f"Using hybrid mode with dynamic worker count and batch sizing (max batch size: {args.batch_size}, max workers: {args.workers})")
        if torch.cuda.is_available():
            logger.info(f"GPU is available: memory-optimized worker count and batch sizing will be used")
        else:
            logger.info(f"GPU is not available: worker count and batch sizing will be based on system memory only")
    elif args.use_batching:
        logger.info(f"Using batch mode with dynamic batch sizing (max batch size: {args.batch_size})")
        if torch.cuda.is_available():
            logger.info(f"GPU is available: memory-optimized batch sizing will be used")
        else:
            logger.info(f"GPU is not available: batch sizing will be based on system memory only")
    else:
        logger.info(f"Using parallel processing mode with dynamic worker allocation (max workers: {args.workers})")
    
    # Initialize core components
    try:
        path_manager_instance = PathManager(
            audio_dir=AUDIO_DIR,
            transcriptions_dir=TRANSCRIPTIONS_DIR,
            corrected_dir=CORRECTED_DIR
        )
        logger.info("PathManager instance created.")
    except Exception as e:
        logger.error(f"Failed to initialize PathManager: {e}. The script cannot continue without it.", exc_info=True)
        sys.exit(1)

    try:
        task_manager_instance = TaskManager(
            path_manager=path_manager_instance,
            min_chars_for_valid_transcription=MIN_CHARS
        )
        task_manager_instance.cleanup_all_locks()
        logger.info("TaskManager instance created and initial lock cleanup performed.")
    except Exception as e:
        logger.error(f"Failed to initialize TaskManager: {e}. The script cannot continue without it.", exc_info=True)
        sys.exit(1)

    # Initialize OllamaCorrector if needed
    ollama_corrector_instance = None
    if not args.skip_transcribing or True:  # Always initialize for now, could be made conditional
        if not all([OLLAMA_API_URL, OLLAMA_MODEL, CORRECTION_PROMPT, OLLAMA_OPTIONS is not None]):
            logger.error("OLLAMA_MODEL, CORRECTION_PROMPT, OLLAMA_API_URL, and OLLAMA_OPTIONS must be set in the configuration for correction.")
        else:
            try:
                ollama_corrector_instance = OllamaCorrector(
                    api_url=OLLAMA_API_URL,
                    model=OLLAMA_MODEL,
                    default_correction_prompt=CORRECTION_PROMPT,
                    ollama_options=OLLAMA_OPTIONS
                )
                logger.info("OllamaCorrector instance created for the main loop.")
            except ValueError as e:
                logger.error(f"Failed to initialize OllamaCorrector: {e}")

    # Main processing loop
    whisper_transcriber_instance = None
    while True:
        try:
            logger.info("="*50)
            logger.info(f"Starting new processing cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Log current memory status
            system_memory = get_available_memory()
            logger.info(f"System memory available: {system_memory:.2f} GB")
            if torch.cuda.is_available():
                gpu_info = get_gpu_memory_info()
                if gpu_info:
                    for gpu_id, mem_info in gpu_info.items():
                        logger.info(f"GPU {gpu_id}: {mem_info['free']:.2f} GB free / {mem_info['total']:.2f} GB total")
            
            logger.info("="*50)
            
            create_backup()
            cleanup_old_backups()
            
            if not args.skip_transcribing:
                whisper_transcriber_instance = handle_transcription_phase(
                    args, 
                    path_manager_instance, 
                    task_manager_instance,
                    whisper_transcriber_instance
                )
            
            handle_correction_phase(task_manager_instance, ollama_corrector_instance)
            
            logger.info(f"Cycle complete. Sleeping for {CHECK_INTERVAL} seconds.")
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.critical(f"Critical error in main loop: {e}", exc_info=True)
            logger.info("Performing emergency cleanup...")
            
            if whisper_transcriber_instance:
                logger.info("Performing emergency cleanup of WhisperTranscriber instance.")
                whisper_transcriber_instance.cleanup()
                whisper_transcriber_instance = None
            
            if task_manager_instance:
                logger.info("Performing emergency cleanup of TaskManager locks.")
                task_manager_instance.cleanup_all_locks()
            
            logger.info("Sleeping for 60 seconds before attempting to restart loop...")
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
        logger.info("Exiting TransFixer.")
        sys.exit(0)