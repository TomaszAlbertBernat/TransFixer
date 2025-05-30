"""
System monitor for tracking overall system health and resource usage.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import psutil
import torch
import GPUtil
from .exceptions import ResourceError

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources and health."""
    
    def __init__(self, monitoring_interval: int = 30):
        self.monitoring_interval = monitoring_interval
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Resource thresholds
        self.cpu_threshold = 90.0  # %
        self.memory_threshold = 85.0  # %
        self.gpu_memory_threshold = 95.0  # % - Increased to reduce false alarms from external processes
        self.gpu_temperature_threshold = 85.0  # °C
        
        # Metrics history
        self.metrics_history: List[Tuple[datetime, Dict]] = []
        self.max_history_size = 200  # Keep ~1.5 hours at 30s intervals
        
        # Alert tracking
        self.alert_counts = {
            'cpu_high': 0,
            'memory_high': 0,
            'gpu_memory_high': 0,
            'gpu_temperature_high': 0
        }
        
        # WSL detection
        self.is_wsl = self._detect_wsl()
        
    def _detect_wsl(self) -> bool:
        """Detect if running in WSL."""
        try:
            with open('/proc/version', 'r') as f:
                content = f.read().lower()
                return 'microsoft' in content or 'wsl' in content
        except Exception:
            return False
    
    def get_system_resources(self) -> Dict:
        """Get comprehensive system resource information."""
        try:
            # CPU information
            cpu_info = {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'free': memory.free
            }
            
            # Swap information
            swap = psutil.swap_memory()
            swap_info = {
                'total': swap.total,
                'used': swap.used,
                'free': swap.free,
                'percent': swap.percent
            }
            
            # Disk information
            disk_info = {}
            try:
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disk_info[partition.device] = {
                            'mountpoint': partition.mountpoint,
                            'fstype': partition.fstype,
                            'total': usage.total,
                            'used': usage.used,
                            'free': usage.free,
                            'percent': (usage.used / usage.total) * 100
                        }
                    except PermissionError:
                        # Skip inaccessible partitions
                        continue
            except Exception as e:
                logger.debug(f"Error getting disk info: {e}")
            
            # GPU information
            gpu_info = self._get_gpu_info()
            
            # Process information
            process_info = self._get_process_info()
            
            return {
                'timestamp': datetime.now(),
                'cpu': cpu_info,
                'memory': memory_info,
                'swap': swap_info,
                'disk': disk_info,
                'gpu': gpu_info,
                'processes': process_info,
                'system': {
                    'platform': psutil.WINDOWS if psutil.WINDOWS else 'unix',
                    'is_wsl': self.is_wsl,
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()),
                    'uptime': datetime.now() - datetime.fromtimestamp(psutil.boot_time())
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            raise ResourceError(f"Failed to get system resources: {e}")
    
    def _get_gpu_info(self) -> Dict:
        """Get GPU information."""
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'devices': []
        }
        
        if not torch.cuda.is_available():
            return gpu_info
        
        try:
            # Get GPU information using GPUtil
            gpus = GPUtil.getGPUs()
            
            for gpu in gpus:
                device_info = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'driver': gpu.driver,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature,
                    'load': gpu.load * 100
                }
                
                # Add PyTorch-specific information
                if torch.cuda.is_initialized():
                    try:
                        device_info.update({
                            'torch_allocated': torch.cuda.memory_allocated(gpu.id) / 1024**2,  # MB
                            'torch_cached': torch.cuda.memory_reserved(gpu.id) / 1024**2,  # MB
                            'torch_max_allocated': torch.cuda.max_memory_allocated(gpu.id) / 1024**2,  # MB
                        })
                    except Exception as e:
                        logger.debug(f"Error getting PyTorch memory info for GPU {gpu.id}: {e}")
                
                gpu_info['devices'].append(device_info)
                
        except Exception as e:
            logger.debug(f"Error getting GPU info: {e}")
        
        return gpu_info
    
    def _get_process_info(self) -> Dict:
        """Get process information."""
        try:
            current_process = psutil.Process()
            
            # Current process info
            process_info = {
                'current': {
                    'pid': current_process.pid,
                    'cpu_percent': current_process.cpu_percent(),
                    'memory_info': current_process.memory_info()._asdict(),
                    'memory_percent': current_process.memory_percent(),
                    'num_threads': current_process.num_threads(),
                    'create_time': datetime.fromtimestamp(current_process.create_time()),
                    'status': current_process.status()
                },
                'children': []
            }
            
            # Children processes
            try:
                children = current_process.children(recursive=True)
                for child in children:
                    try:
                        child_info = {
                            'pid': child.pid,
                            'name': child.name(),
                            'cpu_percent': child.cpu_percent(),
                            'memory_percent': child.memory_percent(),
                            'status': child.status()
                        }
                        process_info['children'].append(child_info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                logger.debug(f"Error getting children processes: {e}")
            
            return process_info
            
        except Exception as e:
            logger.debug(f"Error getting process info: {e}")
            return {}
    
    def check_resource_alerts(self, resources: Dict) -> List[Dict]:
        """Check for resource alerts and return list of alerts."""
        alerts = []
        
        # CPU alert
        cpu_percent = resources.get('cpu', {}).get('percent', 0)
        if cpu_percent > self.cpu_threshold:
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning' if cpu_percent < 95 else 'critical',
                'message': f"High CPU usage: {cpu_percent:.1f}%",
                'value': cpu_percent,
                'threshold': self.cpu_threshold
            })
            self.alert_counts['cpu_high'] += 1
        
        # Memory alert
        memory_percent = resources.get('memory', {}).get('percent', 0)
        if memory_percent > self.memory_threshold:
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning' if memory_percent < 95 else 'critical',
                'message': f"High memory usage: {memory_percent:.1f}%",
                'value': memory_percent,
                'threshold': self.memory_threshold
            })
            self.alert_counts['memory_high'] += 1
        
        # GPU alerts
        gpu_devices = resources.get('gpu', {}).get('devices', [])
        for gpu in gpu_devices:
            # GPU memory alert
            gpu_memory_percent = gpu.get('memory_percent', 0)
            if gpu_memory_percent > self.gpu_memory_threshold:
                alerts.append({
                    'type': 'gpu_memory_high',
                    'severity': 'warning' if gpu_memory_percent < 95 else 'critical',
                    'message': f"High GPU {gpu['id']} memory usage: {gpu_memory_percent:.1f}%",
                    'value': gpu_memory_percent,
                    'threshold': self.gpu_memory_threshold,
                    'gpu_id': gpu['id']
                })
                self.alert_counts['gpu_memory_high'] += 1
            
            # GPU temperature alert
            gpu_temp = gpu.get('temperature', 0)
            if gpu_temp > self.gpu_temperature_threshold:
                alerts.append({
                    'type': 'gpu_temperature_high',
                    'severity': 'warning' if gpu_temp < 90 else 'critical',
                    'message': f"High GPU {gpu['id']} temperature: {gpu_temp}°C",
                    'value': gpu_temp,
                    'threshold': self.gpu_temperature_threshold,
                    'gpu_id': gpu['id']
                })
                self.alert_counts['gpu_temperature_high'] += 1
        
        return alerts
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_enabled:
            logger.warning("System monitoring already enabled")
            return
        
        self.monitoring_enabled = True
        self._shutdown_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="SystemMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.monitoring_enabled:
            return
        
        logger.info("Stopping system monitoring...")
        self.monitoring_enabled = False
        self._shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.debug("System monitoring loop started")
        
        while self.monitoring_enabled and not self._shutdown_event.is_set():
            try:
                # Collect system resources
                resources = self.get_system_resources()
                
                # Check for alerts
                alerts = self.check_resource_alerts(resources)
                
                # Log only critical alerts to reduce noise from external processes
                for alert in alerts:
                    if alert['severity'] == 'critical':
                        logger.error(alert['message'])
                    elif alert['severity'] == 'warning':
                        # Only log non-GPU warnings to avoid spam from external GPU usage
                        if not alert['type'].startswith('gpu_'):
                            logger.warning(alert['message'])
                
                # Store metrics history
                self.metrics_history.append((resources['timestamp'], {
                    'cpu_percent': resources['cpu']['percent'],
                    'memory_percent': resources['memory']['percent'],
                    'gpu_devices': resources['gpu']['devices'],
                    'alerts': alerts
                }))
                
                # Trim history if too large
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
            
            # Wait for next monitoring cycle
            self._shutdown_event.wait(timeout=self.monitoring_interval)
        
        logger.debug("System monitoring loop ended")
    
    def get_performance_summary(self, duration_minutes: int = 60) -> Dict:
        """Get performance summary for the specified duration."""
        if not self.metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [
            (timestamp, metrics) for timestamp, metrics in self.metrics_history
            if timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [metrics['cpu_percent'] for _, metrics in recent_metrics]
        memory_values = [metrics['memory_percent'] for _, metrics in recent_metrics]
        
        summary = {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'alert_counts': self.alert_counts.copy(),
            'gpu_summary': {}
        }
        
        # GPU summary
        gpu_data = {}
        for _, metrics in recent_metrics:
            for gpu in metrics.get('gpu_devices', []):
                gpu_id = gpu['id']
                if gpu_id not in gpu_data:
                    gpu_data[gpu_id] = {
                        'memory_percent': [],
                        'temperature': [],
                        'load': []
                    }
                gpu_data[gpu_id]['memory_percent'].append(gpu['memory_percent'])
                gpu_data[gpu_id]['temperature'].append(gpu['temperature'])
                gpu_data[gpu_id]['load'].append(gpu['load'])
        
        for gpu_id, data in gpu_data.items():
            summary['gpu_summary'][gpu_id] = {
                'memory_percent': {
                    'avg': sum(data['memory_percent']) / len(data['memory_percent']),
                    'max': max(data['memory_percent']),
                    'min': min(data['memory_percent'])
                },
                'temperature': {
                    'avg': sum(data['temperature']) / len(data['temperature']),
                    'max': max(data['temperature']),
                    'min': min(data['temperature'])
                },
                'load': {
                    'avg': sum(data['load']) / len(data['load']),
                    'max': max(data['load']),
                    'min': min(data['load'])
                }
            }
        
        return summary
    
    def get_health_status(self) -> Dict:
        """Get overall system health status."""
        try:
            resources = self.get_system_resources()
            alerts = self.check_resource_alerts(resources)
            
            # Determine overall health
            critical_alerts = [a for a in alerts if a['severity'] == 'critical']
            warning_alerts = [a for a in alerts if a['severity'] == 'warning']
            
            if critical_alerts:
                health_status = 'critical'
            elif warning_alerts:
                health_status = 'warning'
            else:
                health_status = 'healthy'
            
            return {
                'status': health_status,
                'timestamp': resources['timestamp'],
                'alerts': alerts,
                'summary': {
                    'cpu_percent': resources['cpu']['percent'],
                    'memory_percent': resources['memory']['percent'],
                    'gpu_count': len(resources['gpu']['devices']),
                    'total_alerts': len(alerts),
                    'critical_alerts': len(critical_alerts),
                    'warning_alerts': len(warning_alerts)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                'status': 'unknown',
                'timestamp': datetime.now(),
                'error': str(e)
            } 