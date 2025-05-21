import torch
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPU and its memory.
    
    Returns:
        Dict containing GPU information and memory stats
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "device_count": 0,
            "device_name": None,
            "total_memory": 0,
            "allocated_memory": 0,
            "free_memory": 0
        }
    
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = total_memory - allocated_memory
    
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": device_name,
        "total_memory": total_memory,
        "allocated_memory": allocated_memory,
        "free_memory": free_memory
    }

def optimize_batch_size(config: Dict[str, Any]) -> int:
    """
    Optimize batch size based on available GPU memory.
    
    Args:
        config: Current configuration dictionary
        
    Returns:
        Optimized batch size
    """
    gpu_info = get_gpu_info()
    if not gpu_info["available"]:
        return 1  # CPU mode, use minimal batch size
    
    free_memory = gpu_info["free_memory"]
    
    # Memory thresholds in bytes
    HIGH_MEMORY = 6 * 1024 * 1024 * 1024  # 6GB
    LOW_MEMORY = 2 * 1024 * 1024 * 1024   # 2GB
    
    if free_memory >= HIGH_MEMORY:
        return min(16, config.get("batch_size", 16))
    elif free_memory >= LOW_MEMORY:
        return min(8, config.get("batch_size", 16))
    else:
        return min(4, config.get("batch_size", 16))

def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")

def check_gpu_compatibility(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if the current GPU configuration is compatible with the requested settings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_compatible, reason)
    """
    if not torch.cuda.is_available():
        if config.get("device") == "cuda:0":
            return False, "CUDA not available but GPU device requested"
        return True, "CPU mode"
    
    gpu_info = get_gpu_info()
    free_memory = gpu_info["free_memory"]
    
    # Check if we have enough memory for the requested batch size
    batch_size = config.get("batch_size", 16)
    if batch_size > 8 and free_memory < 4 * 1024 * 1024 * 1024:  # 4GB
        return False, f"Insufficient GPU memory for batch size {batch_size}"
    
    # Check Flash Attention compatibility
    if config.get("use_flash_attention", False):
        try:
            import flash_attn
        except ImportError:
            return False, "Flash Attention requested but not installed"
    
    return True, "Compatible"

def get_optimal_device_config() -> Dict[str, Any]:
    """
    Get optimal device configuration based on available hardware.
    
    Returns:
        Dictionary with optimal device settings
    """
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "torch_dtype": torch.float32,
            "use_torch_compile": False,
            "use_flash_attention": False
        }
    
    gpu_info = get_gpu_info()
    device_name = gpu_info["device_name"]
    
    # Basic configuration
    config = {
        "device": "cuda:0",
        "torch_dtype": torch.float16,
        "use_torch_compile": False,
        "use_flash_attention": False
    }
    
    # Check for Flash Attention support
    try:
        import flash_attn
        config["use_flash_attention"] = True
    except ImportError:
        pass
    
    # Check for torch.compile support
    if hasattr(torch, 'compile'):
        config["use_torch_compile"] = True
    
    return config 