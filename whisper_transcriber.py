import os
import logging
import time
from pathlib import Path
import gc
from typing import Optional, Dict, Any, Tuple

import torch
import numpy as np
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

# It's good practice to get the logger by name, though using root logger might be fine for this script.
logger = logging.getLogger(__name__) # Or use logging.getLogger() if you want the root logger directly

class WhisperTranscriber:
    def __init__(self, model_name="openai/whisper-large-v3-turbo",
                 batch_size=4, use_torch_compile=False,
                 hf_token=None, generate_kwargs=None):
        """
        Initialize Whisper model, processor, and pipeline.

        Args:
            model_name (str): Name of the Whisper model to load from Hugging Face.
            batch_size (int): Batch size for the transcription pipeline.
            use_torch_compile (bool): Whether to apply torch.compile to the model.
            hf_token (str, optional): Hugging Face API token for gated models. Defaults to None.
            generate_kwargs (dict, optional): Additional keyword arguments for the pipeline's generate method.
        """
        self.model_name = model_name
        self.effective_batch_size = batch_size
        self.use_torch_compile = use_torch_compile
        self.hf_token = hf_token
        self.generate_kwargs = generate_kwargs if generate_kwargs else {
            "max_new_tokens": 256,
            "do_sample": False,
            "use_cache": True,
            "temperature": 0.0  # Default temperature for deterministic output
        }

        self.model = None
        self.processor = None
        self.pipeline = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() and self.device != "cpu" else torch.float32

        self._initialize_resources()

    def _initialize_resources(self):
        logger.info(f"Initializing WhisperTranscriber with model: {self.model_name}")
        logger.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")

        if self.model is not None:
            logger.info("Whisper resources already initialized.")
            return

        try:
            # Check for v3-turbo specific requirements
            if "v3-turbo" in self.model_name:
                if self.use_torch_compile:
                    logger.warning("torch.compile is not recommended for v3-turbo. Disabling.")
                    self.use_torch_compile = False

            logger.info(f"Loading model {self.model_name}...")
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "low_cpu_mem_usage": True if self.device != "cpu" else False,
                "use_safetensors": True
            }
            if self.hf_token:
                model_kwargs["token"] = self.hf_token

            # Configure attention implementation
            if torch.cuda.is_available():
                try:
                    import flash_attn
                    logger.info("Flash Attention 2 available. Configuring model to use it.")
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    logger.info("Flash Attention 2 not found. Using SDPA if available.")
                    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                        model_kwargs["attn_implementation"] = "sdpa"

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, **model_kwargs)
            self.model.to(self.device)

            # Apply torch.compile only if not v3-turbo and enabled
            if self.use_torch_compile and hasattr(torch, 'compile') and self.device != "cpu" and "v3-turbo" not in self.model_name:
                try:
                    logger.info("Applying torch.compile() to model for faster inference...")
                    self.model = torch.compile(self.model)
                    logger.info("Model compilation successful.")
                except Exception as e:
                    logger.warning(f"Could not compile model with torch.compile(): {e}. Proceeding without compilation.")

            logger.info("Model loaded successfully.")

            logger.info(f"Loading processor for {self.model_name}...")
            processor_kwargs = {}
            if self.hf_token:
                processor_kwargs["token"] = self.hf_token
            self.processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)
            logger.info("Processor loaded successfully.")

            # Configure chunk length based on model
            chunk_length_s = 20 if "v3-turbo" in self.model_name else 30

            logger.info(f"Creating Whisper pipeline with batch_size={self.effective_batch_size}...")
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=chunk_length_s,
                batch_size=self.effective_batch_size,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
                device=self.device,
                generate_kwargs=self.generate_kwargs
            )

            if self.device != "cpu":
                try:
                    total_mem = torch.cuda.get_device_properties(self.device).total_memory
                    current_allocated_mem = torch.cuda.memory_allocated(self.device)
                    available_mem = total_mem - current_allocated_mem

                    logger.info(f"Total GPU memory: {total_mem / (1024**3):.2f}GB")
                    logger.info(f"Current available GPU memory (approx): {available_mem / (1024**3):.2f}GB")

                    # Adjust batch size based on available memory for v3-turbo
                    if "v3-turbo" in self.model_name and available_mem < 6 * 1024 * 1024 * 1024:  # < 6GB
                        logger.warning("Low available GPU memory detected. Reducing batch size for v3-turbo.")
                        self.effective_batch_size = max(1, self.effective_batch_size // 2)
                        self.pipeline.batch_size = self.effective_batch_size

                except Exception as e:
                    logger.warning(f"Error during GPU memory assessment: {e}")

            logger.info("WhisperTranscriber initialized successfully.")

        except Exception as e:
            logger.error(f"Error initializing Whisper resources: {e}", exc_info=True)
            raise

    def detect_language(self, audio_path: str) -> Optional[str]:
        """
        Detect the language of the audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Optional[str]: Detected language code or None if detection fails
        """
        try:
            if not self.pipeline:
                logger.error("Pipeline not initialized.")
                return None

            # Load audio using librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Detect language using the pipeline
            result = self.pipeline(
                audio,
                return_timestamps=False,
                generate_kwargs={"max_new_tokens": 1}  # Minimal tokens for language detection
            )
            
            return result.get("language")
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return None

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe a single audio file with optional language specification.
        
        Args:
            audio_path (str): Path to the audio file
            language (Optional[str]): Language code to use for transcription
            
        Returns:
            dict: Transcription result with metadata
        """
        if not self.pipeline:
            logger.error("Pipeline not initialized.")
            return {"audio_path": audio_path, "transcription": None, "error": "Pipeline not initialized"}

        try:
            # Load audio using librosa
            audio, sr = librosa.load(audio_path, sr=16000)

            # Transcribe with detected or specified language
            result = self.pipeline(
                audio,
                return_timestamps=True,
                generate_kwargs=self.generate_kwargs
            )

            return {
                "audio_path": audio_path,
                "transcription": result.get("text"),
                "language": result.get("language"),
                "timestamps": result.get("timestamps"),
                "error": None
            }

        except Exception as e:
            logger.error(f"Error transcribing file {audio_path}: {e}")
            return {
                "audio_path": audio_path,
                "transcription": None,
                "language": language,
                "timestamps": None,
                "error": str(e)
            }

    def transcribe_batch(self, audio_paths: list[str], global_tqdm_instance=None) -> list[dict]:
        """
        Transcribe a batch of audio files.

        Args:
            audio_paths (list[str]): List of paths to audio files.
            global_tqdm_instance (tqdm, optional): An external tqdm instance to update for overall progress.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains:
                        'audio_path': path to the audio file,
                        'transcription': transcribed text (str) or None if error,
                        'language': detected language code,
                        'timestamps': transcription timestamps,
                        'error': error message (str) or None if successful.
        """
        if not audio_paths:
            return []
        if not self.pipeline:
            logger.error("Pipeline not initialized. Call _initialize_resources first or check for errors.")
            return [{"audio_path": p, "transcription": None, "error": "Pipeline not initialized"} for p in audio_paths]

        results = []
        
        logger.info(f"Processing batch of {len(audio_paths)} files with effective_batch_size={self.effective_batch_size}")
        
        # Log memory usage before batch
        if torch.cuda.is_available() and self.device != "cpu":
            before_mem = torch.cuda.memory_allocated(self.device) / (1024**2)
            logger.info(f"GPU memory before batch processing: {before_mem:.2f}MB")

        try:
            # Process files in batches
            for i in range(0, len(audio_paths), self.effective_batch_size):
                batch_paths = audio_paths[i:i + self.effective_batch_size]
                
                # Detect languages for the batch
                batch_languages = {}
                for audio_path in batch_paths:
                    try:
                        language = self.detect_language(audio_path)
                        if language:
                            batch_languages[audio_path] = language
                            logger.info(f"Detected language for {audio_path}: {language}")
                    except Exception as e:
                        logger.warning(f"Error detecting language for {audio_path}: {e}")

                # Transcribe batch
                try:
                    batch_results = self.pipeline(
                        batch_paths,
                        batch_size=self.effective_batch_size,
                        return_timestamps=True,
                        generate_kwargs=self.generate_kwargs
                    )

                    # Process results
                    for audio_path, result in zip(batch_paths, batch_results):
                        results.append({
                            "audio_path": audio_path,
                            "transcription": result.get("text"),
                            "language": batch_languages.get(audio_path),
                            "timestamps": result.get("timestamps"),
                            "error": None
                        })

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # Add error results for the failed batch
                    for audio_path in batch_paths:
                        results.append({
                            "audio_path": audio_path,
                            "transcription": None,
                            "language": batch_languages.get(audio_path),
                            "timestamps": None,
                            "error": str(e)
                        })

                if global_tqdm_instance:
                    global_tqdm_instance.update(len(batch_paths))

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Add error results for remaining files
            for audio_path in audio_paths[len(results):]:
                results.append({
                    "audio_path": audio_path,
                    "transcription": None,
                    "language": None,
                    "timestamps": None,
                    "error": str(e)
                })

        # Log memory usage after batch
        if torch.cuda.is_available() and self.device != "cpu":
            after_mem = torch.cuda.memory_allocated(self.device) / (1024**2)
            logger.info(f"GPU memory after batch processing: {after_mem:.2f}MB")
            logger.info(f"Batch memory delta: {after_mem - before_mem:.2f}MB")

        return results

    def cleanup(self):
        """Clean up Whisper model resources."""
        logger.info("Cleaning up WhisperTranscriber resources...")
        try:
            if hasattr(self.model, 'to') and torch.cuda.is_available() and self.device != "cpu":
                self.model.to('cpu')
            
            del self.model
            del self.processor
            del self.pipeline
            self.model = None
            self.processor = None
            self.pipeline = None
            
            if torch.cuda.is_available() and self.device != "cpu":
                torch.cuda.empty_cache()
            
            gc.collect() # Python's garbage collector
            
            if torch.cuda.is_available() and self.device != "cpu":
                allocated_mem = torch.cuda.memory_allocated(self.device)
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
                available_mem = total_mem - allocated_mem
                logger.info(f"GPU memory after WhisperTranscriber cleanup: {allocated_mem / (1024**3):.2f}GB used, {available_mem / (1024**3):.2f}GB available")
                
            logger.info("WhisperTranscriber resources cleaned up successfully.")
        except Exception as e:
            logger.error(f"Error during WhisperTranscriber cleanup: {e}", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 