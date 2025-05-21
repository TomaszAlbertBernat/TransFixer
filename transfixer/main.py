import os
import sys
import logging
import argparse
import time
import concurrent.futures
from datetime import datetime
from typing import Optional, List, Dict, Any

from config.settings import (
    AUDIO_DIR, TRANSCRIPTIONS_DIR, CORRECTED_DIR,
    LOG_DIR, CHECK_INTERVAL, MIN_CHARS
)
from core.path_manager import PathManager
from core.task_manager import TaskManager
from services.transcription.whisper_service import WhisperService
from services.transcription.batch_processor import BatchProcessor
from services.correction.ollama_service import OllamaService
from utils.gpu_utils import get_gpu_info, clear_gpu_memory
from services.transcription.whisper_transcriber import WhisperTranscriber
from tqdm import tqdm

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "transfixer.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TransFixer - Audio Transcription and Correction System')
    parser.add_argument('--workers', type=int, default=2,
                      help='Number of parallel Whisper instances')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Number of audio files to process in a single batch')
    parser.add_argument('--skip-transcribing', action='store_true',
                      help='Skip the transcription phase and proceed directly to correction')
    parser.add_argument('--use-batching', action='store_true',
                      help='Use batched processing instead of multiple workers')
    parser.add_argument('--use-torch-compile', action='store_true',
                      help='Enable torch.compile for Whisper model')
    return parser.parse_args()

def initialize_components():
    """Initialize core components."""
    try:
        # Initialize path manager
        path_manager = PathManager(
            audio_dir=AUDIO_DIR,
            transcriptions_dir=TRANSCRIPTIONS_DIR,
            corrected_dir=CORRECTED_DIR
        )
        logger.info("PathManager initialized successfully")
        
        # Initialize task manager
        task_manager = TaskManager(
            path_manager=path_manager,
            min_chars_for_valid_transcription=MIN_CHARS
        )
        task_manager.cleanup_all_locks()
        logger.info("TaskManager initialized successfully")
        
        return path_manager, task_manager
    
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)

def process_audio_file(whisper_service: WhisperService, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single audio file using the provided WhisperService instance.
    
    Args:
        whisper_service: Initialized WhisperService instance
        task: Task dictionary containing audio_path and trans_path
        
    Returns:
        Dict containing processing result
    """
    try:
        result = whisper_service.transcribe_file(task['audio_path'])
        
        if result.get("transcription"):
            with open(task['trans_path'], "w", encoding="utf-8") as f:
                f.write(result["transcription"])
            logger.info(f"Transcribed {task['audio_path']}")
        else:
            logger.error(f"Failed to transcribe {task['audio_path']}: {result.get('error')}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing {task['audio_path']}: {e}")
        return {
            "transcription": None,
            "error": str(e),
            "metadata": {"task": task}
        }

def handle_transcription_phase(args, task_manager: TaskManager) -> Optional[WhisperTranscriber]:
    """Handle the transcription phase."""
    logger.info("-" * 10 + " Phase 1: Transcription " + "-" * 10)
    
    # Collect transcription tasks
    tasks = task_manager.collect_transcription_tasks()
    if not tasks:
        logger.info("No new audio files to transcribe")
        return None
    
    logger.info(f"Found {len(tasks)} files to transcribe")
    
    try:
        # Initialize WhisperTranscriber
        whisper_transcriber = WhisperTranscriber(
            model_name="openai/whisper-large-v3-turbo",
            batch_size=args.batch_size,
            use_torch_compile=args.use_torch_compile
        )
        
        # Process files in batches
        audio_paths = [task['audio_path'] for task in tasks]
        
        with tqdm(total=len(audio_paths), desc="Transcribing files") as pbar:
            results = whisper_transcriber.transcribe_batch(audio_paths, global_tqdm_instance=pbar)
            
            # Save results
            for task, result in zip(tasks, results):
                if result.get("transcription"):
                    with open(task['trans_path'], "w", encoding="utf-8") as f:
                        f.write(result["transcription"])
                    logger.info(f"Transcribed {task['audio_path']}")
                else:
                    logger.error(f"Failed to transcribe {task['audio_path']}: {result.get('error')}")
        
        return whisper_transcriber
    
    except Exception as e:
        logger.error(f"Error in transcription phase: {e}")
        return None

def handle_correction_phase(task_manager: TaskManager):
    """Handle the correction phase."""
    logger.info("-" * 10 + " Phase 2: Correction " + "-" * 10)
    
    # Collect correction tasks
    correction_tasks = task_manager.collect_correction_tasks()
    if not correction_tasks:
        logger.info("No new transcriptions to correct")
        return
    
    logger.info(f"Found {len(correction_tasks)} transcriptions to correct")
    
    try:
        # Initialize Ollama service
        ollama_service = OllamaService()
        
        # Process correction tasks one by one
        for trans_path, corrected_path in correction_tasks:
            try:
                # Skip if file is locked or already corrected
                if task_manager.is_locked(trans_path, lock_type="correction") or os.path.exists(corrected_path):
                    logger.info(f"Skipping {trans_path}: locked or already corrected")
                    continue
                
                # Acquire lock
                if not task_manager.acquire_lock(trans_path, lock_type="correction"):
                    logger.error(f"Could not acquire lock for {trans_path}")
                    continue
                
                try:
                    # Read the transcription
                    with open(trans_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    # Correct the text (OllamaService returns tuple of (text, stats))
                    corrected_text, _ = ollama_service.correct_text(text)
                    
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(corrected_path), exist_ok=True)
                    
                    # Write corrected text to file
                    with open(corrected_path, "w", encoding="utf-8") as f:
                        f.write(corrected_text)
                    
                    logger.info(f"Successfully corrected {trans_path} -> {corrected_path}")
                
                finally:
                    # Always release the lock
                    task_manager.release_lock(trans_path, lock_type="correction")
            
            except Exception as e:
                logger.error(f"Error correcting {trans_path}: {e}")
                # Make sure we release any lock we might have acquired
                if task_manager.is_locked(trans_path, lock_type="correction"):
                    task_manager.release_lock(trans_path, lock_type="correction")
    
    except Exception as e:
        logger.error(f"Error in correction phase: {e}")
        # Clean up any remaining locks
        for trans_path, _ in correction_tasks:
            if task_manager.is_locked(trans_path, lock_type="correction"):
                task_manager.release_lock(trans_path, lock_type="correction")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Log GPU information
    gpu_info = get_gpu_info()
    logger.info(f"GPU Information: {gpu_info}")
    
    # Initialize components
    path_manager, task_manager = initialize_components()
    
    # Main processing loop
    while True:
        try:
            logger.info("="*50)
            logger.info(f"Starting new processing cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*50)
            
            # Handle transcription phase
            whisper_transcriber = None
            if not args.skip_transcribing:
                whisper_transcriber = handle_transcription_phase(args, task_manager)
            
            # Handle correction phase
            handle_correction_phase(task_manager)
            
            # Cleanup
            if whisper_transcriber:
                whisper_transcriber.cleanup()
            clear_gpu_memory()
            
            logger.info(f"Cycle complete. Sleeping for {CHECK_INTERVAL} seconds.")
            time.sleep(CHECK_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            break
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    main() 