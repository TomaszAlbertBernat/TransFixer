import os
import torch
from typing import Dict, Any, Optional
from ..utils.gpu_utils import get_gpu_info

# Default configuration optimized for Whisper-v3-turbo
WHISPER_CONFIG = {
    # Model settings
    "model_id": "openai/whisper-large-v3-turbo",
    "model_dir": os.path.join(os.path.expanduser("~"), ".cache", "whisper"),
    "language": None,  # Auto-detect language
    "task": "transcribe",
    
    # Processing settings
    "sample_rate": 16000,  # Required by Whisper
    "chunk_length_s": 30,  # Optimal chunk length for large-v3-turbo
    "chunk_overlap": 0.1,  # Minimal overlap for better performance
    
    # Transcription settings
    "fp16": True,  # Use FP16 for better performance
    "beam_size": 1,  # Minimal beam size for speed
    "best_of": 1,  # Minimal candidates for speed
    "temperature": 0.0,  # Deterministic output
    "condition_on_previous_text": False,  # Disable for better performance
    "initial_prompt": None,
    
    # Batch processing
    "batch_size": 16,  # Process multiple chunks in parallel
    "max_batch_size": 32,  # Maximum parallel chunks
    "min_batch_size": 1,
    
    # Performance settings
    "use_torch_compile": False,  # Disabled as it's not compatible with chunked processing
    "use_flash_attention_2": True,  # Enable Flash Attention 2
    "use_safetensors": True,  # Use safetensors for faster loading
    "low_cpu_mem_usage": True,  # Optimize CPU memory usage
    "num_workers": 0,  # No additional workers needed
    
    # Generation settings
    "max_new_tokens": 448,
    "compression_ratio_threshold": 1.35,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
    
    # Advanced settings
    "use_sdpa": True,  # Use PyTorch's scaled dot-product attention
    "use_bettertransformer": True,  # Use BetterTransformer for optimization
    
    # Memory optimizations
    "use_cpu_offload": False,  # Offload to CPU when GPU memory is low
    "use_8bit": False,  # Use 8-bit quantization (experimental)
    "use_4bit": False,  # Use 4-bit quantization (experimental)
    "use_dynamic_quantization": True,  # Use dynamic quantization for better memory efficiency
    
    # CUDA optimizations
    "use_cudnn_benchmark": True,  # Enable cuDNN benchmarking
    "use_tf32": True,  # Use TF32 for faster computation on Ampere GPUs
    "use_cuda_graphs": True,  # Use CUDA graphs for faster inference
    
    # Pipeline optimizations
    "use_streaming": False,  # Enable streaming for real-time transcription
    "use_parallel_processing": True,  # Enable parallel processing of chunks
    "use_memory_efficient_attention": True,  # Use memory-efficient attention
    "use_attention_slicing": True,  # Slice attention for better memory usage
    
    # Cache settings
    "use_cache": True,  # Enable KV cache for faster inference
    "cache_precision": "fp16",  # Use FP16 for cache
    "cache_size": 1024,  # Maximum cache size in MB
}

def validate_whisper_config(config: Dict[str, Any]) -> bool:
    """Validate Whisper configuration."""
    required_keys = {
        "model_id", "model_dir", "task", "sample_rate",
        "beam_size", "best_of", "temperature", "chunk_length_s"
    }
    
    if not all(key in config for key in required_keys):
        return False
    
    if config["task"] not in {"transcribe", "translate"}:
        return False
    
    if not (0 <= config["chunk_overlap"] <= 1):
        return False
    
    if not (0 <= config["temperature"] <= 1):
        return False
    
    if config["beam_size"] < 1 or config["best_of"] < 1:
        return False
    
    if config["batch_size"] < config["min_batch_size"] or config["batch_size"] > config["max_batch_size"]:
        return False
    
    # Validate turbo-specific settings
    if config["model_id"] == "openai/whisper-large-v3-turbo":
        if config["use_torch_compile"] and config["use_flash_attention_2"]:
            return False  # Cannot use both torch.compile and Flash Attention 2
    
    return True

def get_optimal_batch_size(audio_length: int) -> int:
    """Calculate optimal batch size based on audio length and available resources."""
    gpu_info = get_gpu_info()
    if not gpu_info:
        return WHISPER_CONFIG["min_batch_size"]
    
    # Calculate based on available memory
    available_memory = gpu_info.get("free_memory", 0)
    memory_per_chunk = audio_length * 4  # 4 bytes per sample
    
    # Calculate maximum possible batch size
    max_batch = min(
        WHISPER_CONFIG["max_batch_size"],
        available_memory // memory_per_chunk
    )
    
    return max(WHISPER_CONFIG["min_batch_size"], max_batch)

def get_optimal_chunk_size(audio_length: int) -> int:
    """Calculate optimal chunk size for audio processing."""
    # Base chunk size (30 seconds at 16kHz)
    base_chunk = WHISPER_CONFIG["chunk_length_s"] * WHISPER_CONFIG["sample_rate"]
    
    # For short audio, process as one chunk
    if audio_length <= base_chunk:
        return audio_length
    
    # For longer audio, use multiple chunks
    num_chunks = min(4, max(2, audio_length // base_chunk))
    return audio_length // num_chunks

def get_model_config(model_id: str) -> Dict[str, Any]:
    """Get model-specific configuration optimized for performance."""
    configs = {
        "openai/whisper-large-v3-turbo": {
            "beam_size": 1,
            "best_of": 1,
            "max_batch_size": 32,
            "chunk_length_s": 30,
            "use_flash_attention_2": True,
            "use_sdpa": True,
            "use_bettertransformer": True,
            "use_torch_compile": False,  # Not compatible with chunked processing
            "use_safetensors": True,
            "low_cpu_mem_usage": True
        }
    }
    
    return configs.get(model_id, configs["openai/whisper-large-v3-turbo"]) 