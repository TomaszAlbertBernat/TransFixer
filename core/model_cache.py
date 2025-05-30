"""
Intelligent model cache manager with adaptive caching and memory management.
"""

import threading
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import torch
from .exceptions import ModelCacheError

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """Intelligent caching system for ML models with adaptive management."""
    
    def __init__(self, 
                 default_timeout: int = 3600,  # 1 hour
                 memory_threshold: float = 0.8,
                 adaptive_timeout: bool = True):
        self.default_timeout = default_timeout
        self.memory_threshold = memory_threshold
        self.adaptive_timeout = adaptive_timeout
        
        # Cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        
        # Usage tracking for adaptive management
        self._usage_stats: Dict[str, Dict[str, Any]] = {}
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(minutes=10)
        
        # Callbacks for cache events
        self._cleanup_callbacks: Dict[str, Callable] = {}
        
    def put(self, key: str, 
            model: Any, 
            processor: Any = None, 
            pipeline: Any = None,
            device: str = 'cpu',
            cleanup_callback: Optional[Callable] = None,
            timeout_override: Optional[int] = None) -> None:
        """Store model components in cache."""
        with self._cache_lock:
            # Calculate adaptive timeout
            timeout = self._calculate_timeout(key, timeout_override)
            
            cache_entry = {
                'model': model,
                'processor': processor,
                'pipeline': pipeline,
                'device': device,
                'created_at': datetime.now(),
                'last_used': datetime.now(),
                'timeout': timeout,
                'use_count': 0,
                'total_time_used': timedelta(),
                'memory_size': self._estimate_memory_size(model, processor, pipeline)
            }
            
            # Store cleanup callback if provided
            if cleanup_callback:
                self._cleanup_callbacks[key] = cleanup_callback
            
            # Check if we need to evict existing entries
            self._maybe_evict_entries(cache_entry['memory_size'])
            
            self._cache[key] = cache_entry
            self._initialize_usage_stats(key)
            
            logger.debug(f"Cached model '{key}' with timeout {timeout}s, estimated size: {cache_entry['memory_size']:.1f}MB")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve model components from cache."""
        with self._cache_lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            now = datetime.now()
            
            # Check if entry has expired
            if self._is_expired(entry, now):
                logger.debug(f"Cache entry '{key}' expired, removing")
                self._remove_entry(key)
                return None
            
            # Update usage statistics
            entry['last_used'] = now
            entry['use_count'] += 1
            self._update_usage_stats(key, now)
            
            logger.debug(f"Cache hit for '{key}' (use count: {entry['use_count']})")
            
            return {
                'model': entry['model'],
                'processor': entry['processor'],
                'pipeline': entry['pipeline'],
                'device': entry['device']
            }
    
    def remove(self, key: str) -> bool:
        """Manually remove an entry from cache."""
        with self._cache_lock:
            if key in self._cache:
                self._remove_entry(key)
                logger.debug(f"Manually removed cache entry '{key}'")
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._cache_lock:
            keys_to_remove = list(self._cache.keys())
            for key in keys_to_remove:
                self._remove_entry(key)
            logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries and return count of removed entries."""
        with self._cache_lock:
            now = datetime.now()
            expired_keys = []
            
            for key, entry in self._cache.items():
                if self._is_expired(entry, now):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            self._last_cleanup = now
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def force_cleanup_by_memory(self, target_memory_mb: float) -> int:
        """Force cleanup to free up specified amount of memory."""
        with self._cache_lock:
            removed_count = 0
            freed_memory = 0.0
            
            # Sort entries by priority (least recently used first)
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1]['last_used'], x[1]['use_count'])
            )
            
            for key, entry in sorted_entries:
                if freed_memory >= target_memory_mb:
                    break
                
                freed_memory += entry['memory_size']
                self._remove_entry(key)
                removed_count += 1
            
            logger.info(f"Force cleanup freed {freed_memory:.1f}MB by removing {removed_count} entries")
            return removed_count
    
    def get_memory_usage(self) -> float:
        """Get total estimated memory usage in MB."""
        with self._cache_lock:
            return sum(entry['memory_size'] for entry in self._cache.values())
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        with self._cache_lock:
            now = datetime.now()
            
            info = {
                'total_entries': len(self._cache),
                'total_memory_mb': self.get_memory_usage(),
                'entries': {}
            }
            
            for key, entry in self._cache.items():
                age = (now - entry['created_at']).total_seconds()
                time_since_use = (now - entry['last_used']).total_seconds()
                
                info['entries'][key] = {
                    'device': entry['device'],
                    'age_seconds': age,
                    'time_since_use_seconds': time_since_use,
                    'use_count': entry['use_count'],
                    'memory_size_mb': entry['memory_size'],
                    'is_expired': self._is_expired(entry, now)
                }
            
            return info
    
    def optimize_cache(self):
        """Optimize cache based on usage patterns and memory constraints."""
        with self._cache_lock:
            # Clean up expired entries first
            self.cleanup_expired()
            
            # Check memory usage
            total_memory = self.get_memory_usage()
            
            # If memory usage is high, perform adaptive cleanup
            if self._is_memory_pressure():
                logger.info(f"Memory pressure detected (usage: {total_memory:.1f}MB), optimizing cache")
                self._adaptive_cleanup()
            
            # Update timeouts based on usage patterns
            if self.adaptive_timeout:
                self._update_adaptive_timeouts()
    
    def _calculate_timeout(self, key: str, timeout_override: Optional[int]) -> int:
        """Calculate timeout for cache entry."""
        if timeout_override is not None:
            return timeout_override
        
        if not self.adaptive_timeout or key not in self._usage_stats:
            return self.default_timeout
        
        # Adaptive timeout based on usage patterns
        stats = self._usage_stats[key]
        usage_frequency = stats.get('usage_frequency', 0)
        
        if usage_frequency > 10:  # High usage
            return self.default_timeout * 2
        elif usage_frequency > 5:  # Medium usage
            return self.default_timeout
        else:  # Low usage
            return self.default_timeout // 2
    
    def _estimate_memory_size(self, model: Any, processor: Any = None, pipeline: Any = None) -> float:
        """Estimate memory size of cached components in MB."""
        size_mb = 0.0
        
        try:
            # Estimate model size
            if model is not None and hasattr(model, 'parameters'):
                params = sum(p.numel() for p in model.parameters())
                # Assume float32 parameters (4 bytes each)
                size_mb += (params * 4) / (1024 * 1024)
            
            # Add overhead for processor and pipeline (rough estimates)
            if processor is not None:
                size_mb += 10  # Rough estimate
            
            if pipeline is not None:
                size_mb += 50  # Rough estimate for pipeline overhead
            
            # Add 20% overhead for Python objects and CUDA memory alignment
            size_mb *= 1.2
            
        except Exception as e:
            logger.debug(f"Error estimating memory size: {e}")
            size_mb = 500  # Conservative default estimate
        
        return size_mb
    
    def _is_expired(self, entry: Dict[str, Any], now: datetime) -> bool:
        """Check if cache entry has expired."""
        age = (now - entry['created_at']).total_seconds()
        return age > entry['timeout']
    
    def _is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            if torch.cuda.is_available():
                # Check GPU memory
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id)
                    reserved = torch.cuda.memory_reserved(gpu_id)
                    total = torch.cuda.get_device_properties(gpu_id).total_memory
                    
                    usage_ratio = max(allocated, reserved) / total
                    if usage_ratio > self.memory_threshold:
                        return True
            
            # Could add system RAM check here if needed
            return False
            
        except Exception as e:
            logger.debug(f"Error checking memory pressure: {e}")
            return False
    
    def _maybe_evict_entries(self, new_entry_size: float):
        """Evict entries if necessary to make room for new entry."""
        current_memory = self.get_memory_usage()
        
        # If adding new entry would exceed memory threshold, evict some entries
        if self._is_memory_pressure() or (current_memory + new_entry_size) > 2000:  # 2GB limit
            target_free = max(new_entry_size * 1.5, 200)  # Free at least 200MB or 1.5x new entry size
            self.force_cleanup_by_memory(target_free)
    
    def _adaptive_cleanup(self):
        """Perform adaptive cleanup based on usage patterns."""
        # Remove least recently used entries with low usage counts
        candidates = []
        
        for key, entry in self._cache.items():
            score = self._calculate_eviction_score(entry)
            candidates.append((score, key, entry))
        
        # Sort by eviction score (higher score = more likely to evict)
        candidates.sort(reverse=True)
        
        # Remove up to 50% of entries or until memory pressure is relieved
        max_removals = max(1, len(candidates) // 2)
        removed = 0
        
        for score, key, entry in candidates:
            if removed >= max_removals or not self._is_memory_pressure():
                break
            
            self._remove_entry(key)
            removed += 1
    
    def _calculate_eviction_score(self, entry: Dict[str, Any]) -> float:
        """Calculate eviction score (higher = more likely to evict)."""
        now = datetime.now()
        
        # Time since last use (normalized to 0-1)
        time_since_use = (now - entry['last_used']).total_seconds()
        time_factor = min(time_since_use / 3600, 1.0)  # Normalize to 1 hour
        
        # Inverse of use count (normalized)
        use_factor = 1.0 / max(entry['use_count'], 1)
        
        # Memory size factor (larger = higher score)
        memory_factor = entry['memory_size'] / 1000  # Normalize to 1GB
        
        # Combine factors
        return time_factor * 0.4 + use_factor * 0.3 + memory_factor * 0.3
    
    def _remove_entry(self, key: str):
        """Remove entry and call cleanup callback."""
        if key not in self._cache:
            return
        
        # Call cleanup callback if exists
        if key in self._cleanup_callbacks:
            try:
                self._cleanup_callbacks[key]()
            except Exception as e:
                logger.error(f"Error in cleanup callback for '{key}': {e}")
            finally:
                del self._cleanup_callbacks[key]
        
        # Remove from cache
        del self._cache[key]
        
        # Clean up usage stats
        if key in self._usage_stats:
            del self._usage_stats[key]
    
    def _initialize_usage_stats(self, key: str):
        """Initialize usage statistics for a cache key."""
        self._usage_stats[key] = {
            'creation_time': datetime.now(),
            'total_uses': 0,
            'usage_frequency': 0,
            'last_usage_time': datetime.now()
        }
    
    def _update_usage_stats(self, key: str, now: datetime):
        """Update usage statistics."""
        if key not in self._usage_stats:
            self._initialize_usage_stats(key)
        
        stats = self._usage_stats[key]
        stats['total_uses'] += 1
        stats['last_usage_time'] = now
        
        # Calculate usage frequency (uses per hour)
        age_hours = (now - stats['creation_time']).total_seconds() / 3600
        stats['usage_frequency'] = stats['total_uses'] / max(age_hours, 0.1)
    
    def _update_adaptive_timeouts(self):
        """Update timeouts for existing entries based on usage patterns."""
        for key, entry in self._cache.items():
            if key in self._usage_stats:
                new_timeout = self._calculate_timeout(key, None)
                if new_timeout != entry['timeout']:
                    entry['timeout'] = new_timeout
                    logger.debug(f"Updated timeout for '{key}' to {new_timeout}s") 