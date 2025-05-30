"""
Custom exceptions for TransFixer application.
"""


class TransFixerError(Exception):
    """Base exception class for TransFixer."""
    pass


class ProcessError(TransFixerError):
    """Exception raised for process management errors."""
    pass


class GPUError(TransFixerError):
    """Exception raised for GPU-related errors."""
    pass


class ModelCacheError(TransFixerError):
    """Exception raised for model cache errors."""
    pass


class ResourceError(TransFixerError):
    """Exception raised for system resource errors."""
    pass


class ConfigurationError(TransFixerError):
    """Exception raised for configuration errors."""
    pass 