import json
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import threading
import psutil
import GPUtil
import torch
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    files_processed: int = 0
    total_duration_seconds: float = 0.0
    total_processing_time: float = 0.0
    average_real_time_factor: float = 0.0
    gpu_utilization: List[float] = None
    memory_usage: List[float] = None
    errors: int = 0
    retries: int = 0
    
    def __post_init__(self):
        if self.gpu_utilization is None:
            self.gpu_utilization = []
        if self.memory_usage is None:
            self.memory_usage = []

class ConfigManager:
    """Advanced configuration management with validation and hot reloading"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.callbacks = []
        self._lock = threading.Lock()
        self._last_modified = None
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        self.default_config = {
            "whisper": {
                "model_name": "openai/whisper-large-v3-turbo",
                "torch_dtype": "float16",
                "chunk_length_s": 30,
                "batch_size_auto": True,
                "max_batch_size": 32,
                "use_torch_compile": True,
                "enable_flash_attention": True
            },
            "processing": {
                "max_workers": 2,
                "max_retries": 3,
                "timeout_seconds": 300,
                "priority_processing": True,
                "streaming_threshold_minutes": 5,
                "min_file_duration": 1.0
            },
            "resources": {
                "memory_threshold": 0.85,
                "gpu_memory_threshold": 0.9,
                "cpu_threshold": 0.9,
                "monitor_interval": 10
            },
            "quality": {
                "min_chars": 50,
                "validate_transcriptions": True,
                "quality_check_interval": 100,
                "auto_retry_poor_quality": True
            },
            "paths": {
                "audio_dir": "audio",
                "transcriptions_dir": "transcriptions",
                "corrected_dir": "corrected",
                "temp_dir": "/tmp/transfixer",
                "log_dir": "logs"
            }
        }
        
        self.load_config()
        
    def load_config(self):
        """Load configuration with validation"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                self.config = self._merge_configs(self.default_config, loaded_config)
                self._last_modified = self.config_path.stat().st_mtime
            else:
                self.config = self.default_config.copy()
                self.save_config()
                
            self._validate_config()
            logging.info("Configuration loaded successfully")
            
        except Exception as e:
            logging.error(f"Error loading config: {e}, using defaults")
            self.config = self.default_config.copy()
            
    def _merge_configs(self, default: dict, loaded: dict) -> dict:
        """Recursively merge configurations"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
        
    def _validate_config(self):
        """Validate configuration values"""
        whisper_config = self.config.get("whisper", {})
        processing_config = self.config.get("processing", {})
        
        # Validate whisper settings
        if whisper_config.get("chunk_length_s", 0) <= 0:
            raise ValueError("chunk_length_s must be positive")
            
        if processing_config.get("max_workers", 0) <= 0:
            raise ValueError("max_workers must be positive")
            
        # Validate paths exist or can be created
        for path_key, path_value in self.config.get("paths", {}).items():
            if path_key != "temp_dir":  # temp_dir might not exist yet
                Path(path_value).mkdir(parents=True, exist_ok=True)
                
    def save_config(self):
        """Save current configuration"""
        with self._lock:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                
    def get(self, path: str, default=None):
        """Get configuration value using dot notation"""
        keys = path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
        
    def set(self, path: str, value):
        """Set configuration value using dot notation"""
        with self._lock:
            keys = path.split('.')
            config = self.config
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            config[keys[-1]] = value
            self.save_config()
            
        # Notify callbacks
        for callback in self.callbacks:
            callback(path, value)
            
    def add_change_callback(self, callback):
        """Add callback for configuration changes"""
        self.callbacks.append(callback)
        
    def start_monitoring(self):
        """Start monitoring config file for changes"""
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(target=self._monitor_config, daemon=True)
            self._monitor_thread.start()
            
    def _monitor_config(self):
        """Monitor configuration file for changes"""
        while not self._stop_monitoring.is_set():
            try:
                if self.config_path.exists():
                    current_modified = self.config_path.stat().st_mtime
                    if current_modified != self._last_modified:
                        logging.info("Configuration file changed, reloading...")
                        self.load_config()
                        
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error monitoring config: {e}")
                time.sleep(5)

class PerformanceMonitor:
    """Advanced performance monitoring with metrics collection"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.metrics = PerformanceMetrics()
        self.session_start = datetime.now()
        self.file_metrics = {}  # file_path -> metrics
        self._lock = threading.Lock()
        
        # Performance tracking
        self.processing_times = []
        self.queue_sizes = []
        self.resource_history = []
        
        # Start resource monitoring
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        
    def record_file_start(self, file_path: str, file_duration: float):
        """Record start of file processing"""
        with self._lock:
            self.file_metrics[file_path] = {
                "start_time": time.time(),
                "duration": file_duration,
                "device": None,
                "retry_count": 0
            }
            
    def record_file_completion(self, file_path: str, success: bool, device: str = None, retry_count: int = 0):
        """Record completion of file processing"""
        with self._lock:
            if file_path in self.file_metrics:
                file_metric = self.file_metrics[file_path]
                processing_time = time.time() - file_metric["start_time"]
                file_duration = file_metric["duration"]
                
                self.metrics.files_processed += 1
                self.metrics.total_duration_seconds += file_duration
                self.metrics.total_processing_time += processing_time
                
                if file_duration > 0:
                    rtf = processing_time / file_duration
                    self.processing_times.append(rtf)
                    
                    # Update running average
                    self.metrics.average_real_time_factor = sum(self.processing_times) / len(self.processing_times)
                
                if not success:
                    self.metrics.errors += 1
                    
                self.metrics.retries += retry_count
                
                # Clean up
                del self.file_metrics[file_path]
                
    def _monitor_resources(self):
        """Monitor system resources"""
        interval = self.config.get("resources.monitor_interval", 10)
        
        while True:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                resource_data = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3)
                }
                
                # GPU metrics
                if torch.cuda.is_available():
                    try:
                        gpus = GPUtil.getGPUs()
                        gpu_data = []
                        for gpu in gpus:
                            gpu_data.append({
                                "id": gpu.id,
                                "utilization": gpu.load * 100,
                                "memory_used": gpu.memoryUsed,
                                "memory_total": gpu.memoryTotal,
                                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                                "temperature": gpu.temperature
                            })
                            
                        resource_data["gpus"] = gpu_data
                        
                        # Update metrics
                        with self._lock:
                            for gpu in gpu_data:
                                self.metrics.gpu_utilization.append(gpu["utilization"])
                                self.metrics.memory_usage.append(gpu["memory_percent"])
                                
                    except Exception as e:
                        logging.warning(f"Error getting GPU metrics: {e}")
                
                self.resource_history.append(resource_data)
                
                # Keep only recent history
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-500:]
                    
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Error in resource monitoring: {e}")
                time.sleep(interval)
                
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        with self._lock:
            session_duration = (datetime.now() - self.session_start).total_seconds()
            
            recent_resources = self.resource_history[-10:] if self.resource_history else []
            
            report = {
                "session": {
                    "start_time": self.session_start.isoformat(),
                    "duration_hours": session_duration / 3600,
                    "files_processed": self.metrics.files_processed,
                    "total_audio_hours": self.metrics.total_duration_seconds / 3600,
                    "processing_hours": self.metrics.total_processing_time / 3600
                },
                "performance": {
                    "average_rtf": self.metrics.average_real_time_factor,
                    "files_per_hour": self.metrics.files_processed / (session_duration / 3600) if session_duration > 0 else 0,
                    "error_rate": self.metrics.errors / max(self.metrics.files_processed, 1),
                    "retry_rate": self.metrics.retries / max(self.metrics.files_processed, 1)
                },
                "resources": {
                    "recent_cpu_avg": sum(r.get("cpu_percent", 0) for r in recent_resources) / max(len(recent_resources), 1),
                    "recent_memory_avg": sum(r.get("memory_percent", 0) for r in recent_resources) / max(len(recent_resources), 1),
                    "gpu_utilization_avg": sum(self.metrics.gpu_utilization[-100:]) / max(len(self.metrics.gpu_utilization[-100:]), 1) if self.metrics.gpu_utilization else 0
                },
                "queue": {
                    "active_files": len(self.file_metrics),
                    "average_queue_size": sum(self.queue_sizes[-100:]) / max(len(self.queue_sizes[-100:]), 1) if self.queue_sizes else 0
                }
            }
            
            return report
            
    def save_report(self, file_path: str):
        """Save performance report to file"""
        report = self.get_performance_report()
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)

class QualityAssurance:
    """Quality assurance for transcriptions"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.quality_metrics = {
            "total_checked": 0,
            "poor_quality_count": 0,
            "auto_retries": 0
        }
        
    def check_transcription_quality(self, text: str, audio_duration: float) -> Dict[str, any]:
        """Check transcription quality using multiple metrics"""
        min_chars = self.config.get("quality.min_chars", 50)
        
        quality_score = 100.0
        issues = []
        
        # Length check
        if len(text.strip()) < min_chars:
            quality_score -= 30
            issues.append("Too short")
            
        # Character rate check (reasonable speaking rate)
        chars_per_second = len(text) / audio_duration if audio_duration > 0 else 0
        if chars_per_second < 5:  # Very slow
            quality_score -= 20
            issues.append("Possibly incomplete")
        elif chars_per_second > 25:  # Very fast
            quality_score -= 15
            issues.append("Possibly garbled")
            
        # Repetition check
        words = text.lower().split()
        if len(words) > 5:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.5:
                quality_score -= 25
                issues.append("High repetition")
                
        # Special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_chars / max(len(text), 1) > 0.1:
            quality_score -= 10
            issues.append("High special character ratio")
            
        self.quality_metrics["total_checked"] += 1
        if quality_score < 70:
            self.quality_metrics["poor_quality_count"] += 1
            
        return {
            "quality_score": max(0, quality_score),
            "issues": issues,
            "should_retry": quality_score < 50 and self.config.get("quality.auto_retry_poor_quality", False)
        }

# Usage example with improved integration
class EnhancedTranscriptionSystem:
    """Enhanced system with all improvements integrated"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = ConfigManager(config_path)
        self.monitor = PerformanceMonitor(self.config)
        self.qa = QualityAssurance(self.config)
        
        # Start configuration monitoring
        self.config.start_monitoring()
        self.config.add_change_callback(self._on_config_change)
        
    def _on_config_change(self, path: str, value):
        """Handle configuration changes"""
        logging.info(f"Configuration changed: {path} = {value}")
        # Implement hot reloading logic here
        
    def get_current_status(self) -> Dict:
        """Get current system status"""
        return {
            "performance": self.monitor.get_performance_report(),
            "quality": self.qa.quality_metrics,
            "config": {
                "model": self.config.get("whisper.model_name"),
                "workers": self.config.get("processing.max_workers"),
                "batch_size": self.config.get("whisper.max_batch_size")
            }
        } 