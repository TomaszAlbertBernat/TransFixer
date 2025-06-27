#!/usr/bin/env python3
"""
GPU Utilization Optimization Script for TransFixer
Based on faster-whisper community research and best practices.

This script implements several optimizations to improve GPU utilization
beyond the typical 50% limit commonly seen with Whisper models.
"""

import os
import torch
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_capabilities():
    """Check GPU capabilities and suggest optimizations."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        return None
        
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPU(s)")
    
    gpu_info = {}
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpu_info[i] = {
            'name': props.name,
            'total_memory': props.total_memory // (1024**3),  # GB
            'compute_capability': f"{props.major}.{props.minor}",
            'multiprocessor_count': props.multi_processor_count
        }
        logger.info(f"GPU {i}: {props.name} ({props.total_memory // (1024**3)}GB)")
        
    return gpu_info

def optimize_environment_variables():
    """Set environment variables for better GPU utilization."""
    optimizations = {
        # CUDA optimizations
        'CUDA_LAUNCH_BLOCKING': '0',  # Enable async kernel launches
        'CUDA_CACHE_DISABLE': '0',    # Enable CUDA cache
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',  # Consistent GPU ordering
        
        # PyTorch optimizations
        'TORCH_CUDA_ARCH_LIST': 'Auto',  # Auto-detect architecture
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,roundup_power2_divisions:16',
        
        # For better memory management
        'PYTORCH_NO_CUDA_MEMORY_CACHING': '0',  # Enable memory caching
        
        # Faster-whisper specific
        'OMP_NUM_THREADS': str(min(os.cpu_count() or 8, 8)),  # Optimal thread count
        'MKL_NUM_THREADS': str(min(os.cpu_count() or 8, 8)),
        
        # Triton optimizations (used by torch.compile)
        'TRITON_CACHE_DIR': '/tmp/triton_cache',
    }
    
    for key, value in optimizations.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")

def enable_advanced_torch_settings():
    """Enable advanced PyTorch settings for better GPU utilization."""
    try:
        # Enable TensorFloat-32 (TF32) for Ampere GPUs
        if torch.cuda.is_available():
            # TF32 can significantly improve performance on A100/RTX 30xx/RTX 40xx
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for better performance")
            
            # Enable CUDNN benchmarking
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled CUDNN benchmarking")
            
            # Optimize memory allocation
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
            
    except Exception as e:
        logger.warning(f"Could not enable some advanced settings: {e}")

def create_optimized_transcription_config():
    """Create an optimized configuration for transcription."""
    config = {
        # Model settings optimized for GPU utilization
        'model_settings': {
            'compute_type': 'float16',  # Use FP16 for better GPU utilization
            'device': 'cuda',
            'device_index': 0,
            # Try different numbers of workers based on your GPU
            'num_workers': 1,  # Start with 1, can experiment with 2-4
        },
        
        # Batch processing for better GPU utilization
        'batch_settings': {
            'batch_size': 1,  # faster-whisper doesn't support batch_size > 1 natively
            'chunk_length': 30,  # Optimal for most content
            'overlap': 0,  # No overlap for efficiency
        },
        
        # Transcription parameters optimized for speed
        'transcription_params': {
            'beam_size': 1,  # Reduce from 5 to 1 for better GPU utilization
            'best_of': 1,    # Single candidate for speed
            'temperature': 0.0,  # Deterministic output
            'condition_on_previous_text': False,  # Faster processing
            'compression_ratio_threshold': 2.4,
            'log_prob_threshold': -1.0,
            'no_speech_threshold': 0.6,
            'initial_prompt': None,
        }
    }
    
    return config

def monitor_gpu_utilization(duration_seconds=60):
    """Monitor GPU utilization for a specified duration."""
    logger.info(f"Monitoring GPU utilization for {duration_seconds} seconds...")
    
    try:
        # Use nvidia-smi to monitor GPU utilization
        cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu', 
               '--format=csv,noheader,nounits', f'--loop-ms=1000']
        
        utilization_data = []
        start_time = time.time()
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        while time.time() - start_time < duration_seconds:
            try:
                if process.stdout is None:
                    break
                line = process.stdout.readline()
                if line:
                    values = [float(x.strip()) for x in line.strip().split(',')]
                    utilization_data.append({
                        'gpu_util': values[0],
                        'mem_util': values[1], 
                        'power_draw': values[2],
                        'temperature': values[3],
                        'timestamp': time.time()
                    })
                    
                if process.poll() is not None:
                    break
                    
            except (ValueError, IndexError):
                continue
                
        process.terminate()
        
        if utilization_data:
            avg_gpu_util = sum(d['gpu_util'] for d in utilization_data) / len(utilization_data)
            avg_power = sum(d['power_draw'] for d in utilization_data) / len(utilization_data)
            max_gpu_util = max(d['gpu_util'] for d in utilization_data)
            
            logger.info(f"Average GPU Utilization: {avg_gpu_util:.1f}%")
            logger.info(f"Maximum GPU Utilization: {max_gpu_util:.1f}%")
            logger.info(f"Average Power Draw: {avg_power:.1f}W")
            
            return utilization_data
        else:
            logger.warning("No utilization data collected")
            return []
            
    except FileNotFoundError:
        logger.error("nvidia-smi not found. Please install NVIDIA drivers.")
        return []
    except Exception as e:
        logger.error(f"Error monitoring GPU: {e}")
        return []

def optimize_for_multiple_files():
    """Suggestions for optimizing transcription of multiple files."""
    optimizations = [
        "1. Use multiprocessing with 2-4 processes (not more than GPU memory allows)",
        "2. Consider using ThreadPoolExecutor instead of ProcessPoolExecutor for some workloads",
        "3. Pre-load audio files in batches to reduce I/O overhead",
        "4. Use num_workers parameter in faster-whisper model initialization",
        "5. Consider using int8 quantization for very fast transcription with minimal quality loss",
        "6. For very large files, split them into smaller chunks (15-30 second segments)",
        "7. Use VAD (Voice Activity Detection) to skip silent parts",
    ]
    
    logger.info("Optimizations for multiple files:")
    for opt in optimizations:
        logger.info(f"  {opt}")

def test_transcription_performance():
    """Test different configurations to find optimal settings."""
    logger.info("Testing transcription performance with different configurations...")
    
    # This would require implementing actual transcription tests
    # For now, we'll just provide guidance
    
    suggestions = [
        "Test with beam_size=1 vs beam_size=5 (1 usually gives better GPU utilization)",
        "Compare float16 vs int8 quantization",
        "Try different chunk_length values (15, 30, 45 seconds)",
        "Test with and without torch.compile enabled",
        "Monitor with different num_workers values (1, 2, 4)",
        "Compare performance with different WHISPER_MODEL sizes",
    ]
    
    logger.info("Performance testing suggestions:")
    for suggestion in suggestions:
        logger.info(f"  - {suggestion}")

def main():
    """Main optimization function."""
    logger.info("=== TransFixer GPU Optimization Tool ===")
    
    # Check GPU capabilities
    gpu_info = check_gpu_capabilities()
    if not gpu_info:
        return
    
    # Apply environment optimizations
    logger.info("\n=== Applying Environment Optimizations ===")
    optimize_environment_variables()
    
    # Enable advanced PyTorch settings
    logger.info("\n=== Enabling Advanced PyTorch Settings ===")
    enable_advanced_torch_settings()
    
    # Show optimized configuration
    logger.info("\n=== Optimized Configuration ===")
    config = create_optimized_transcription_config()
    for section, settings in config.items():
        logger.info(f"{section}:")
        for key, value in settings.items():
            logger.info(f"  {key}: {value}")
    
    # Provide optimization tips
    logger.info("\n=== Optimization Tips ===")
    optimize_for_multiple_files()
    
    # Performance testing guidance
    logger.info("\n=== Performance Testing ===")
    test_transcription_performance()
    
    logger.info("\n=== Important Notes ===")
    logger.info("1. 50% GPU utilization is NORMAL for Whisper models")
    logger.info("2. Focus on overall throughput rather than GPU percentage")
    logger.info("3. Monitor power draw - 60-150W is reasonable for transcription")
    logger.info("4. CPU can become the bottleneck in many cases")
    logger.info("5. Memory bandwidth may limit GPU utilization")
    
    # Offer to monitor current utilization
    response = input("\nWould you like to monitor GPU utilization for 30 seconds? (y/n): ")
    if response.lower() == 'y':
        monitor_gpu_utilization(30)

if __name__ == "__main__":
    main() 