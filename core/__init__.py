"""
Core module for TransFixer providing essential system components.
"""

# Import only the classes, not the modules
from .exceptions import TransFixerError, ProcessError, GPUError, ModelCacheError

# Lazy imports to avoid dependency issues
def _get_process_manager():
    from .process_manager import ProcessManager
    return ProcessManager

def _get_gpu_manager():
    from .gpu_manager import GPUManager
    return GPUManager

def _get_model_cache_manager():
    from .model_cache import ModelCacheManager
    return ModelCacheManager

def _get_system_monitor():
    from .system_monitor import SystemMonitor
    return SystemMonitor

# Create lazy-loaded classes
class ProcessManager:
    def __new__(cls, *args, **kwargs):
        return _get_process_manager()(*args, **kwargs)

class GPUManager:
    def __new__(cls, *args, **kwargs):
        return _get_gpu_manager()(*args, **kwargs)

class ModelCacheManager:
    def __new__(cls, *args, **kwargs):
        return _get_model_cache_manager()(*args, **kwargs)

class SystemMonitor:
    def __new__(cls, *args, **kwargs):
        return _get_system_monitor()(*args, **kwargs)

__all__ = [
    'TransFixerError',
    'ProcessError', 
    'GPUError',
    'ModelCacheError',
    'ProcessManager',
    'GPUManager',
    'ModelCacheManager',
    'SystemMonitor'
] 