"""
GPU manager for comprehensive GPU resource management and monitoring.
"""

import time
import logging
import threading
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import torch
import GPUtil
from .exceptions import GPUError

logger = logging.getLogger(__name__)


class GPUManager:
    """Manages GPU resources, memory allocation, and temperature monitoring."""
    
    def __init__(self, temperature_threshold: float = 85.0, memory_threshold: float = 0.95):
        self.temperature_threshold = temperature_threshold
        self.memory_threshold = memory_threshold  # Increased to 95% to reduce false alarms
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 60  # Increased to 60 seconds to reduce noise
        self._shutdown_event = threading.Event()
        
        # GPU state tracking
        self.gpu_states: Dict[int, Dict] = {}
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)
        
        # Performance tracking
        self.memory_usage_history: List[Tuple[datetime, Dict]] = []
        self.max_history_size = 100
        
        # Warning suppression for external processes
        self.warning_cooldown = timedelta(minutes=5)  # Only warn every 5 minutes
        self.last_warning_time: Dict[str, datetime] = {}
        
        self._initialize_gpu_states()
    
    def _initialize_gpu_states(self):
        """Initialize GPU state tracking."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU manager will have limited functionality")
            return
            
        gpu_count = torch.cuda.device_count()
        logger.info(f"Initializing GPU manager for {gpu_count} GPU(s)")
        
        for gpu_id in range(gpu_count):
            self.gpu_states[gpu_id] = {
                'device_name': torch.cuda.get_device_name(gpu_id),
                'total_memory': torch.cuda.get_device_properties(gpu_id).total_memory,
                'last_used': None,
                'active_processes': 0,
                'temperature_warnings': 0,
                'memory_warnings': 0
            }
            logger.debug(f"GPU {gpu_id}: {self.gpu_states[gpu_id]['device_name']}")
    
    def select_best_gpu(self) -> Optional[int]:
        """Select the GPU with the most available memory."""
        if not torch.cuda.is_available():
            return None
            
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0
            
            best_gpu = 0
            max_available = 0
            
            for gpu in gpus:
                available = gpu.memoryTotal - gpu.memoryUsed
                logger.debug(f"GPU {gpu.id}: {available}MB available out of {gpu.memoryTotal}MB total")
                
                if available > max_available:
                    max_available = available
                    best_gpu = gpu.id
            
            logger.info(f"Selected GPU {best_gpu} with {max_available}MB available memory")
            return best_gpu
            
        except Exception as e:
            logger.error(f"Error selecting best GPU: {e}")
            return 0
    
    def get_gpu_info(self, gpu_id: Optional[int] = None) -> Dict:
        """Get comprehensive GPU information."""
        if not torch.cuda.is_available():
            return {'cuda_available': False}
        
        if gpu_id is None:
            gpu_id = 0
            
        try:
            gpus = GPUtil.getGPUs()
            if gpu_id >= len(gpus):
                raise GPUError(f"GPU {gpu_id} not found")
            
            gpu = gpus[gpu_id]
            
            # Get PyTorch memory info
            torch_memory = {}
            if torch.cuda.is_initialized():
                torch_memory = {
                    'allocated': torch.cuda.memory_allocated(gpu_id) / 1024**2,  # MB
                    'cached': torch.cuda.memory_reserved(gpu_id) / 1024**2,  # MB
                    'max_allocated': torch.cuda.max_memory_allocated(gpu_id) / 1024**2,  # MB
                }
            
            gpu_info = {
                'cuda_available': True,
                'id': gpu.id,
                'name': gpu.name,
                'driver': gpu.driver,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'memory_available': gpu.memoryTotal - gpu.memoryUsed,
                'temperature': gpu.temperature,
                'load': gpu.load * 100,
                'torch_memory': torch_memory,
                'state': self.gpu_states.get(gpu_id, {})
            }
            
            return gpu_info
            
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            raise GPUError(f"Failed to get GPU info: {e}")
    
    def _should_warn(self, warning_type: str) -> bool:
        """Check if we should issue a warning based on cooldown period."""
        now = datetime.now()
        last_warning = self.last_warning_time.get(warning_type)
        
        if last_warning is None or (now - last_warning) > self.warning_cooldown:
            self.last_warning_time[warning_type] = now
            return True
        return False

    def monitor_temperature(self, gpu_id: int, suppress_warnings: bool = False) -> bool:
        """Monitor GPU temperature and return True if safe to continue."""
        try:
            gpu_info = self.get_gpu_info(gpu_id)
            temperature = gpu_info['temperature']
            
            if temperature > self.temperature_threshold:
                self.gpu_states[gpu_id]['temperature_warnings'] += 1
                
                # Only warn if not suppressed and cooldown period has passed
                if not suppress_warnings and self._should_warn(f"temp_{gpu_id}"):
                    logger.warning(f"GPU {gpu_id} temperature high: {temperature}°C (threshold: {self.temperature_threshold}°C)")
                
                if temperature > self.temperature_threshold + 15:  # Increased critical threshold
                    if not suppress_warnings:
                        logger.error(f"GPU {gpu_id} temperature critical: {temperature}°C, requesting load reduction")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error monitoring GPU temperature: {e}")
            return True  # Assume safe if we can't monitor
    
    def check_memory_usage(self, gpu_id: int, suppress_warnings: bool = False) -> Tuple[bool, float]:
        """Check GPU memory usage and return (is_safe, usage_percent)."""
        try:
            gpu_info = self.get_gpu_info(gpu_id)
            usage_percent = gpu_info['memory_percent'] / 100
            
            # Only warn if memory is critically high (95%+) and we haven't warned recently
            if usage_percent > self.memory_threshold:
                self.gpu_states[gpu_id]['memory_warnings'] += 1
                
                if not suppress_warnings and self._should_warn(f"memory_{gpu_id}"):
                    logger.warning(f"GPU {gpu_id} memory usage critically high: {usage_percent:.1%} (threshold: {self.memory_threshold:.1%})")
                
                # Only return False if memory is extremely high (98%+)
                if usage_percent > 0.98:
                    return False, usage_percent
                
            return True, usage_percent
            
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return True, 0.0  # Assume safe if we can't check
    
    def cleanup_gpu_memory(self, gpu_id: Optional[int] = None):
        """Perform comprehensive GPU memory cleanup."""
        if not torch.cuda.is_available():
            return
        
        try:
            if gpu_id is not None:
                devices = [gpu_id]
            else:
                devices = list(range(torch.cuda.device_count()))
            
            for device in devices:
                logger.debug(f"Cleaning up GPU {device} memory...")
                
                with torch.cuda.device(device):
                    # Clear PyTorch cache
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Synchronize to ensure operations complete
                    torch.cuda.synchronize()
                
                logger.debug(f"GPU {device} memory cleanup completed")
            
            self.last_cleanup = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during GPU memory cleanup: {e}")
            raise GPUError(f"GPU memory cleanup failed: {e}")
    
    def defragment_memory(self, gpu_id: int):
        """Attempt to defragment GPU memory."""
        if not torch.cuda.is_available():
            return
        
        try:
            logger.info(f"Defragmenting GPU {gpu_id} memory...")
            
            # Get memory info before
            before_info = self.get_gpu_info(gpu_id)
            before_available = before_info['memory_available']
            
            with torch.cuda.device(gpu_id):
                # Multiple cleanup passes for thorough defragmentation
                for i in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    time.sleep(0.1)  # Small delay between passes
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Final synchronization
                torch.cuda.synchronize()
            
            # Get memory info after
            after_info = self.get_gpu_info(gpu_id)
            after_available = after_info['memory_available']
            
            freed_mb = after_available - before_available
            logger.info(f"GPU {gpu_id} defragmentation completed, freed {freed_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error during GPU memory defragmentation: {e}")
    
    def calculate_optimal_batch_size(self, gpu_id: int, safety_factor: float = 0.85) -> int:
        """Calculate optimal batch size based on available GPU memory."""
        try:
            gpu_info = self.get_gpu_info(gpu_id)
            available_memory_mb = gpu_info['memory_available']
            
            # Apply safety factor - slightly more aggressive for better GPU utilization
            safe_memory = available_memory_mb * safety_factor
            
            # More aggressive batch sizing for better GPU utilization
            # These values are optimized for transcription workloads
            if safe_memory > 12000:   # 12GB+
                return min(32, 40)
            elif safe_memory > 10000:   # 10GB+
                return min(28, 32)
            elif safe_memory > 8000:  # 8GB+
                return min(24, 28)
            elif safe_memory > 6000:  # 6GB+
                return min(20, 24)
            elif safe_memory > 4000:  # 4GB+
                return min(16, 20)
            elif safe_memory > 2000:  # 2GB+
                return min(12, 16)
            else:
                return min(8, 12)
                
        except Exception as e:
            logger.error(f"Error calculating optimal batch size: {e}")
            return 8  # Slightly higher safe default
    
    def start_monitoring(self):
        """Start background GPU monitoring."""
        if self.monitoring_enabled:
            logger.warning("GPU monitoring already enabled")
            return
        
        self.monitoring_enabled = True
        self._shutdown_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="GPUMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop background GPU monitoring."""
        if not self.monitoring_enabled:
            return
        
        logger.info("Stopping GPU monitoring...")
        self.monitoring_enabled = False
        self._shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("GPU monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.debug("GPU monitoring loop started")
        
        while self.monitoring_enabled and not self._shutdown_event.is_set():
            try:
                self._collect_metrics()
                
                # Check if cleanup is needed
                if datetime.now() - self.last_cleanup > self.cleanup_interval:
                    self._auto_cleanup()
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
            
            # Wait for next monitoring cycle
            self._shutdown_event.wait(timeout=self.monitoring_interval)
        
        logger.debug("GPU monitoring loop ended")
    
    def _collect_metrics(self):
        """Collect GPU metrics for monitoring."""
        if not torch.cuda.is_available():
            return
        
        timestamp = datetime.now()
        metrics = {}
        
        for gpu_id in range(torch.cuda.device_count()):
            try:
                gpu_info = self.get_gpu_info(gpu_id)
                metrics[gpu_id] = {
                    'memory_used': gpu_info['memory_used'],
                    'memory_percent': gpu_info['memory_percent'],
                    'temperature': gpu_info['temperature'],
                    'load': gpu_info['load']
                }
                
                # Check for warnings but suppress them during background monitoring
                # to avoid spam from external processes like Ollama
                self.monitor_temperature(gpu_id, suppress_warnings=True)
                self.check_memory_usage(gpu_id, suppress_warnings=True)
                
            except Exception as e:
                logger.debug(f"Error collecting metrics for GPU {gpu_id}: {e}")
        
        # Store metrics history
        self.memory_usage_history.append((timestamp, metrics))
        
        # Trim history if too large
        if len(self.memory_usage_history) > self.max_history_size:
            self.memory_usage_history = self.memory_usage_history[-self.max_history_size:]
    
    def _auto_cleanup(self):
        """Perform automatic cleanup based on usage patterns."""
        try:
            for gpu_id in range(torch.cuda.device_count()):
                is_safe, usage_percent = self.check_memory_usage(gpu_id, suppress_warnings=True)
                
                # Only trigger cleanup if memory is extremely high (98%+)
                if not is_safe or usage_percent > 0.98:
                    logger.info(f"Auto-cleanup triggered for GPU {gpu_id} (usage: {usage_percent:.1%})")
                    self.cleanup_gpu_memory(gpu_id)
                    
        except Exception as e:
            logger.error(f"Error during auto-cleanup: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from monitoring history."""
        if not self.memory_usage_history:
            return {}
        
        stats = {
            'monitoring_duration': len(self.memory_usage_history) * self.monitoring_interval,
            'gpus': {}
        }
        
        for gpu_id in range(torch.cuda.device_count()):
            gpu_metrics = []
            for timestamp, metrics in self.memory_usage_history:
                if gpu_id in metrics:
                    gpu_metrics.append(metrics[gpu_id])
            
            if gpu_metrics:
                memory_usage = [m['memory_percent'] for m in gpu_metrics]
                temperatures = [m['temperature'] for m in gpu_metrics if m['temperature'] > 0]
                
                stats['gpus'][gpu_id] = {
                    'avg_memory_usage': sum(memory_usage) / len(memory_usage),
                    'max_memory_usage': max(memory_usage),
                    'avg_temperature': sum(temperatures) / len(temperatures) if temperatures else 0,
                    'max_temperature': max(temperatures) if temperatures else 0,
                    'temperature_warnings': self.gpu_states[gpu_id]['temperature_warnings'],
                    'memory_warnings': self.gpu_states[gpu_id]['memory_warnings']
                }
        
        return stats 