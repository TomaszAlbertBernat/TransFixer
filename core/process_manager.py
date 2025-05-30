"""
Process manager for handling process lifecycle and graceful shutdown.
"""

import os
import signal
import sys
import time
import threading
import logging
import psutil
from typing import List, Optional, Callable
from .exceptions import ProcessError

logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages process lifecycle, signal handling, and graceful shutdown."""
    
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.cleanup_handlers: List[Callable] = []
        self.child_processes: List[psutil.Process] = []
        self._signal_received = False
        self._shutdown_timeout = 30  # seconds
        self._is_wsl = self._detect_wsl()
        
        # Register signal handlers
        self._register_signal_handlers()
        
    def _detect_wsl(self) -> bool:
        """Detect if running in WSL (Windows Subsystem for Linux)."""
        try:
            with open('/proc/version', 'r') as f:
                content = f.read().lower()
                return 'microsoft' in content or 'wsl' in content
        except Exception:
            return False
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]
        
        # Add SIGQUIT for Unix systems
        if hasattr(signal, 'SIGQUIT'):
            signals_to_handle.append(signal.SIGQUIT)
            
        for sig in signals_to_handle:
            signal.signal(sig, self._signal_handler)
            
        logger.debug(f"Registered signal handlers for: {[s.name for s in signals_to_handle]}")
    
    def _signal_handler(self, signum: int, frame):
        """Handle termination signals gracefully."""
        if self._signal_received:
            logger.warning(f"Signal {signum} already being handled, forcing immediate exit...")
            self._force_exit()
            return
            
        self._signal_received = True
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name} ({signum}), initiating graceful shutdown...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Start shutdown process in a separate thread to avoid blocking signal handler
        shutdown_thread = threading.Thread(target=self._perform_shutdown, daemon=True)
        shutdown_thread.start()
        
        # Wait for shutdown with timeout
        shutdown_thread.join(timeout=self._shutdown_timeout)
        
        if shutdown_thread.is_alive():
            logger.error("Graceful shutdown timed out, forcing exit...")
            self._force_exit()
    
    def _perform_shutdown(self):
        """Perform graceful shutdown sequence."""
        try:
            logger.info("Starting graceful shutdown sequence...")
            
            # Run cleanup handlers in reverse order
            for i, handler in enumerate(reversed(self.cleanup_handlers)):
                try:
                    logger.debug(f"Running cleanup handler {len(self.cleanup_handlers) - i}")
                    handler()
                except Exception as e:
                    logger.error(f"Error in cleanup handler: {e}")
            
            # Terminate child processes
            self._terminate_children()
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
        finally:
            # Exit cleanly
            sys.exit(0)
    
    def _terminate_children(self):
        """Terminate all child processes gracefully."""
        if not self.child_processes:
            # Auto-discover child processes
            try:
                current_process = psutil.Process()
                self.child_processes = current_process.children(recursive=True)
            except Exception as e:
                logger.error(f"Error discovering child processes: {e}")
                return
        
        if not self.child_processes:
            logger.debug("No child processes to terminate")
            return
            
        logger.info(f"Terminating {len(self.child_processes)} child processes...")
        
        # First attempt: SIGTERM
        active_children = []
        for child in self.child_processes:
            try:
                if child.is_running():
                    logger.debug(f"Sending SIGTERM to process {child.pid}")
                    child.terminate()
                    active_children.append(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Process {child.pid} already terminated or access denied: {e}")
        
        # Wait for graceful termination
        if active_children:
            logger.debug("Waiting for processes to terminate gracefully...")
            gone, alive = psutil.wait_procs(active_children, timeout=5)
            
            for proc in gone:
                logger.debug(f"Process {proc.pid} terminated gracefully")
            
            # Force kill remaining processes
            if alive:
                logger.warning(f"Force killing {len(alive)} remaining processes...")
                for proc in alive:
                    try:
                        logger.debug(f"Force killing process {proc.pid}")
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.debug(f"Process {proc.pid} already gone or access denied: {e}")
    
    def _force_exit(self):
        """Force immediate exit for WSL compatibility."""
        if self._is_wsl:
            try:
                # In WSL, try to kill the entire process tree
                os.system(f"pkill -f {os.path.basename(sys.argv[0])}")
            except Exception:
                pass
        
        # Force exit
        os._exit(1)
    
    def register_cleanup_handler(self, handler: Callable):
        """Register a cleanup handler to be called during shutdown."""
        if not callable(handler):
            raise ProcessError("Cleanup handler must be callable")
        
        self.cleanup_handlers.append(handler)
        logger.debug(f"Registered cleanup handler: {handler.__name__}")
    
    def add_child_process(self, process: psutil.Process):
        """Add a child process to be managed."""
        self.child_processes.append(process)
        logger.debug(f"Added child process {process.pid} to management")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_event.is_set()
    
    def wait_for_shutdown(self, timeout: Optional[float] = None):
        """Wait for shutdown event."""
        return self.shutdown_event.wait(timeout)
    
    def request_shutdown(self):
        """Request shutdown programmatically."""
        logger.info("Shutdown requested programmatically")
        self.shutdown_event.set()
    
    def set_shutdown_timeout(self, timeout: int):
        """Set the timeout for graceful shutdown."""
        self._shutdown_timeout = max(5, timeout)  # Minimum 5 seconds
        logger.debug(f"Shutdown timeout set to {self._shutdown_timeout} seconds") 