import os
import json
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import requests
import signal
import sys

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api"
EMBEDDINGS_DIR = "embeddings"
DEFAULT_MODEL = "mxbai-embed-large:335m"  # Default embedding model
BATCH_SIZE = 10  # Process files in batches to avoid overwhelming Ollama
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 2  # Delay between retries in seconds
CHECK_INTERVAL = 300  # Seconds to wait between checking for new files (5 minutes)

# Global shutdown flag
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print("\nShutdown requested. Finishing current batch and exiting...")
    shutdown_requested = True

# Register signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def get_embedding_from_ollama(text, model):
    """Get embeddings from Ollama API."""
    url = f"{OLLAMA_API_URL}/embeddings"
    payload = {
        "model": model,
        "prompt": text,
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["embedding"]
        except (requests.exceptions.RequestException, KeyError) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Error getting embeddings, retrying ({attempt+1}/{MAX_RETRIES}): {str(e)}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to get embeddings after {MAX_RETRIES} attempts: {str(e)}")
                return None

def process_transcription_file(file_path, model, save_dir):
    """Process a single transcription file and save its embeddings."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Skip empty files
        if not text:
            print(f"Skipping empty file: {file_path}")
            return False
            
        # Get embeddings
        embedding = get_embedding_from_ollama(text, model)
        
        if embedding is None:
            print(f"Failed to get embedding for: {file_path}")
            return False
            
        # Create metadata for the embedding
        metadata = {
            "source_file": file_path,
            "model": model,
            "timestamp": time.time(),
            "file_size": os.path.getsize(file_path),
            "vector_dimension": len(embedding)
        }
        
        # Determine output path
        rel_path = os.path.relpath(file_path, "transcriptions")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_path = os.path.join(save_dir, os.path.dirname(rel_path))
        ensure_directory_exists(dir_path)
        
        # Save embedding and metadata
        output_path = os.path.join(dir_path, f"{base_name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump({
                "embedding": np.array(embedding, dtype=np.float32),
                "metadata": metadata
            }, f)
            
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def find_transcription_files(base_dir):
    """Find all transcription files recursively."""
    files = []
    for root, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                files.append(os.path.join(root, filename))
    return files

def check_ollama_available():
    """Check if Ollama is running and available."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            available_models = response.json().get("models", [])
            model_names = [model.get("name") for model in available_models]
            return True, model_names
        return False, []
    except requests.exceptions.RequestException:
        return False, []

def find_new_transcription_files(base_dir, processed_files):
    """Find transcription files that haven't been vectorized yet."""
    files = []
    for root, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                
                # Calculate the expected output path for this file
                rel_path = os.path.relpath(file_path, base_dir)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                dir_path = os.path.dirname(rel_path)
                output_path = os.path.join(EMBEDDINGS_DIR, dir_path, f"{base_name}.pkl")
                
                # Check if this file has already been processed
                if file_path not in processed_files and not os.path.exists(output_path):
                    files.append(file_path)
    return files

def track_processed_files(embeddings_dir):
    """Build a set of already processed files based on existing embeddings."""
    processed = set()
    for root, _, files in os.walk(embeddings_dir):
        for file in files:
            if file.endswith('.pkl') and not file == "processing_summary.json":
                try:
                    # Load the embedding file to get the source file path
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        if 'metadata' in data and 'source_file' in data['metadata']:
                            processed.add(data['metadata']['source_file'])
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    return processed

def main():
    global EMBEDDINGS_DIR
    
    parser = argparse.ArgumentParser(description="Vectorize transcription files using Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use for embeddings (default: {DEFAULT_MODEL})")
    parser.add_argument("--transcriptions_dir", default="transcriptions", help="Directory containing transcription files")
    parser.add_argument("--output_dir", default=EMBEDDINGS_DIR, help=f"Directory to save embeddings (default: {EMBEDDINGS_DIR})")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous mode, periodically checking for new files")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL, help=f"Seconds to wait between checks in continuous mode (default: {CHECK_INTERVAL})")
    args = parser.parse_args()
    
    # Update global variables from arguments
    EMBEDDINGS_DIR = args.output_dir
    
    # Check if Ollama is available
    ollama_available, available_models = check_ollama_available()
    if not ollama_available:
        print("Error: Ollama is not running. Please start Ollama service.")
        print("You can start it by running 'ollama serve' in a terminal.")
        return
    
    if available_models and args.model not in available_models:
        print(f"Warning: Model '{args.model}' is not found in available models.")
        print(f"Available models: {', '.join(available_models)}")
        response = input(f"Do you want to pull the '{args.model}' model? (y/n): ")
        if response.lower() == 'y':
            print(f"Pulling model '{args.model}'...")
            try:
                requests.post(f"{OLLAMA_API_URL}/pull", json={"name": args.model})
                print(f"Successfully pulled model '{args.model}'")
            except requests.exceptions.RequestException as e:
                print(f"Failed to pull model: {str(e)}")
                return
        else:
            print("Exiting. Please specify a valid model.")
            return
    
    # Create output directory
    ensure_directory_exists(args.output_dir)
    
    # For tracking which files have been processed
    processed_files = set()
    if os.path.exists(args.output_dir):
        processed_files = track_processed_files(args.output_dir)
        print(f"Found {len(processed_files)} already processed files")
    
    if args.continuous:
        print(f"Running in continuous mode. Checking for new files every {args.interval} seconds.")
        print("Press Ctrl+C to exit.")
        
        while not shutdown_requested:
            # Find all new transcription files
            transcription_files = find_new_transcription_files(args.transcriptions_dir, processed_files)
            
            if transcription_files:
                print(f"Found {len(transcription_files)} new transcription files to process")
                
                # Process files in batches
                successful_count = 0
                failed_count = 0
                
                for i in range(0, len(transcription_files), BATCH_SIZE):
                    if shutdown_requested:
                        break
                        
                    batch = transcription_files[i:i+BATCH_SIZE]
                    print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(transcription_files)-1)//BATCH_SIZE + 1}...")
                    
                    for file_path in tqdm(batch):
                        # Calculate output path for this file
                        rel_path = os.path.relpath(file_path, args.transcriptions_dir)
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        dir_path = os.path.dirname(rel_path)
                        output_dir = os.path.join(args.output_dir, dir_path)
                        
                        success = process_transcription_file(file_path, args.model, args.output_dir)
                        if success:
                            successful_count += 1
                            processed_files.add(file_path)
                        else:
                            failed_count += 1
                
                print(f"Vectorization batch complete! Successfully processed {successful_count} files.")
                if failed_count > 0:
                    print(f"Failed to process {failed_count} files. Check the logs for details.")
                
                # Save processing summary
                summary = {
                    "timestamp": time.time(),
                    "model": args.model,
                    "total_files": len(transcription_files),
                    "successful": successful_count,
                    "failed": failed_count
                }
                
                with open(os.path.join(args.output_dir, "processing_summary.json"), 'w') as f:
                    json.dump(summary, f, indent=2)
            else:
                print(f"No new files found. Waiting {args.interval} seconds before next check...")
            
            # Sleep until next check, but check for shutdown every second
            for _ in range(args.interval):
                if shutdown_requested:
                    break
                time.sleep(1)
    else:
        # Single run mode (original behavior)
        # Find all transcription files
        transcription_files = find_transcription_files(args.transcriptions_dir)
        print(f"Found {len(transcription_files)} transcription files to process")
        
        # Process files in batches
        successful_count = 0
        failed_count = 0
        
        for i in range(0, len(transcription_files), BATCH_SIZE):
            batch = transcription_files[i:i+BATCH_SIZE]
            print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(transcription_files)-1)//BATCH_SIZE + 1}...")
            
            for file_path in tqdm(batch):
                success = process_transcription_file(file_path, args.model, args.output_dir)
                if success:
                    successful_count += 1
                else:
                    failed_count += 1
        
        print(f"Vectorization complete! Successfully processed {successful_count} files.")
        if failed_count > 0:
            print(f"Failed to process {failed_count} files. Check the logs for details.")
        
        # Save processing summary
        summary = {
            "timestamp": time.time(),
            "model": args.model,
            "total_files": len(transcription_files),
            "successful": successful_count,
            "failed": failed_count
        }
        
        with open(os.path.join(args.output_dir, "processing_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main() 