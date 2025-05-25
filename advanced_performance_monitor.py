#!/usr/bin/env python3
"""
Advanced Performance Monitor for TransFixer
Real-time monitoring of Whisper transcription performance with optimization suggestions
"""

import time
import threading
import psutil
import GPUtil
import torch
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_temperature: Optional[float] = None
    transcription_rate: Optional[float] = None  # files per second
    avg_file_duration: Optional[float] = None  # seconds
    batch_size: Optional[int] = None
    chunk_length: Optional[int] = None
    backend: Optional[str] = None
    model_name: Optional[str] = None

class PerformanceOptimizer:
    """Real-time performance optimization suggestions"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_log = []
        
    def analyze_performance(self, metrics: PerformanceMetrics) -> List[str]:
        """Analyze current performance and suggest optimizations"""
        suggestions = []
        
        # GPU Memory Analysis
        if metrics.gpu_memory_used_mb and metrics.gpu_memory_total_mb:
            gpu_usage_percent = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            
            if gpu_usage_percent > 90:
                suggestions.append("🔴 GPU memory critical (>90%). Consider reducing batch size or chunk length.")
            elif gpu_usage_percent > 80:
                suggestions.append("🟡 GPU memory high (>80%). Monitor for OOM errors.")
            elif gpu_usage_percent < 50:
                suggestions.append("🟢 GPU memory low (<50%). Consider increasing batch size for better throughput.")
        
        # CPU Usage Analysis
        if metrics.cpu_percent > 90:
            suggestions.append("🔴 CPU usage critical (>90%). Consider reducing worker processes.")
        elif metrics.cpu_percent < 30:
            suggestions.append("🟢 CPU usage low (<30%). Consider increasing worker processes.")
        
        # Memory Analysis
        if metrics.memory_percent > 85:
            suggestions.append("🔴 System memory critical (>85%). Enable model cleanup between batches.")
        elif metrics.memory_percent < 40:
            suggestions.append("🟢 System memory low (<40%). Safe to increase batch sizes.")
        
        # Transcription Rate Analysis
        if metrics.transcription_rate and metrics.transcription_rate < 0.1:
            suggestions.append("🔴 Low transcription rate (<0.1 files/sec). Check model optimizations.")
        elif metrics.transcription_rate and metrics.transcription_rate > 1.0:
            suggestions.append("🟢 Excellent transcription rate (>1.0 files/sec)!")
        
        return suggestions
    
    def suggest_optimal_settings(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Suggest optimal configuration settings based on current performance"""
        optimal_settings = {}
        
        if metrics.gpu_memory_used_mb and metrics.gpu_memory_total_mb:
            available_memory = metrics.gpu_memory_total_mb - metrics.gpu_memory_used_mb
            
            # Suggest batch size based on available memory
            if available_memory > 8000:  # 8GB+
                optimal_settings["batch_size"] = 24
                optimal_settings["chunk_length"] = 90
            elif available_memory > 6000:  # 6GB+
                optimal_settings["batch_size"] = 16
                optimal_settings["chunk_length"] = 60
            elif available_memory > 4000:  # 4GB+
                optimal_settings["batch_size"] = 12
                optimal_settings["chunk_length"] = 45
            else:
                optimal_settings["batch_size"] = 8
                optimal_settings["chunk_length"] = 30
        
        # Suggest performance mode
        if metrics.memory_percent < 60 and metrics.gpu_memory_used_mb:
            gpu_usage = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            if gpu_usage < 60:
                optimal_settings["performance_mode"] = "aggressive"
        
        return optimal_settings

class AdvancedPerformanceMonitor:
    """Advanced real-time performance monitoring system"""
    
    def __init__(self, log_file: str = "logs/performance_monitor.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("PerformanceMonitor")
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimizer = PerformanceOptimizer()
        self.monitoring = False
        self.monitor_thread = None
        
        # Performance tracking
        self.transcription_start_times = {}
        self.completed_transcriptions = 0
        self.total_audio_duration = 0
        
    def start_monitoring(self, interval: float = 5.0):
        """Start real-time performance monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 metrics to prevent memory bloat
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Log performance and suggestions
                self._log_performance(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None
        gpu_temperature = None
        
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_memory_used = gpu.memoryUsed
                    gpu_memory_total = gpu.memoryTotal
                    gpu_utilization = gpu.load * 100
                    gpu_temperature = gpu.temperature
            except Exception:
                pass
        
        # Calculate transcription rate
        transcription_rate = None
        if hasattr(self, 'last_rate_calculation'):
            time_diff = time.time() - self.last_rate_calculation
            if time_diff >= 10:  # Calculate rate every 10 seconds
                transcription_rate = self.completed_transcriptions / time_diff if time_diff > 0 else 0
                self.completed_transcriptions = 0
                self.last_rate_calculation = time.time()
        else:
            self.last_rate_calculation = time.time()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature,
            transcription_rate=transcription_rate
        )
    
    def _log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics and suggestions"""
        # Get optimization suggestions
        suggestions = self.optimizer.analyze_performance(metrics)
        optimal_settings = self.optimizer.suggest_optimal_settings(metrics)
        
        # Log basic metrics
        self.logger.info(f"CPU: {metrics.cpu_percent:.1f}% | "
                        f"Memory: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}GB)")
        
        if metrics.gpu_memory_used_mb:
            gpu_usage = (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb) * 100
            self.logger.info(f"GPU: {gpu_usage:.1f}% ({metrics.gpu_memory_used_mb:.0f}MB/{metrics.gpu_memory_total_mb:.0f}MB) | "
                           f"Temp: {metrics.gpu_temperature}°C")
        
        if metrics.transcription_rate:
            self.logger.info(f"Transcription rate: {metrics.transcription_rate:.2f} files/sec")
        
        # Log suggestions
        for suggestion in suggestions:
            self.logger.info(f"SUGGESTION: {suggestion}")
        
        # Log optimal settings if different from current
        if optimal_settings:
            self.logger.info(f"OPTIMAL SETTINGS: {optimal_settings}")
    
    def log_transcription_start(self, file_path: str, audio_duration: float = None):
        """Log the start of a transcription"""
        self.transcription_start_times[file_path] = time.time()
        if audio_duration:
            self.total_audio_duration += audio_duration
    
    def log_transcription_complete(self, file_path: str, success: bool = True):
        """Log the completion of a transcription"""
        if file_path in self.transcription_start_times:
            duration = time.time() - self.transcription_start_times[file_path]
            del self.transcription_start_times[file_path]
            
            if success:
                self.completed_transcriptions += 1
                self.logger.info(f"Transcription completed: {file_path} in {duration:.2f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_duration_minutes": len(self.metrics_history) * 5 / 60,  # Assuming 5s intervals
            "average_cpu_percent": round(avg_cpu, 2),
            "average_memory_percent": round(avg_memory, 2),
            "total_transcriptions": self.completed_transcriptions,
            "active_transcriptions": len(self.transcription_start_times)
        }
        
        if recent_metrics[0].gpu_memory_used_mb:
            avg_gpu_memory = sum(m.gpu_memory_used_mb for m in recent_metrics if m.gpu_memory_used_mb) / len(recent_metrics)
            avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics if m.gpu_utilization) / len(recent_metrics)
            
            summary.update({
                "average_gpu_memory_mb": round(avg_gpu_memory, 2),
                "average_gpu_utilization": round(avg_gpu_util, 2),
                "gpu_memory_total_mb": recent_metrics[0].gpu_memory_total_mb
            })
        
        return summary
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": self.get_performance_summary(),
            "metrics": [asdict(m) for m in self.metrics_history],
            "optimization_suggestions": self.optimizer.optimization_log
        }
        
        # Convert datetime objects to strings for JSON serialization
        for metric in export_data["metrics"]:
            if isinstance(metric["timestamp"], datetime):
                metric["timestamp"] = metric["timestamp"].isoformat()
        
        export_path = Path("logs") / filename
        export_path.parent.mkdir(exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics exported to {export_path}")
        return str(export_path)

# Global performance monitor instance
performance_monitor = AdvancedPerformanceMonitor()

def start_performance_monitoring():
    """Start the global performance monitor"""
    performance_monitor.start_monitoring()

def stop_performance_monitoring():
    """Stop the global performance monitor"""
    performance_monitor.stop_monitoring()

def log_transcription_start(file_path: str, audio_duration: float = None):
    """Log transcription start to performance monitor"""
    performance_monitor.log_transcription_start(file_path, audio_duration)

def log_transcription_complete(file_path: str, success: bool = True):
    """Log transcription completion to performance monitor"""
    performance_monitor.log_transcription_complete(file_path, success)

def get_performance_summary():
    """Get current performance summary"""
    return performance_monitor.get_performance_summary()

def export_performance_metrics(filename: str = None):
    """Export performance metrics to file"""
    return performance_monitor.export_metrics(filename)

if __name__ == "__main__":
    # Test the performance monitor
    monitor = AdvancedPerformanceMonitor()
    monitor.start_monitoring(interval=2.0)
    
    try:
        print("Performance monitoring active. Press Ctrl+C to stop...")
        while True:
            time.sleep(5)
            summary = monitor.get_performance_summary()
            print(f"Summary: {summary}")
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop_monitoring()
        monitor.export_metrics("test_metrics.json") 