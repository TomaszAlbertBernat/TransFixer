import torch
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging
from contextlib import contextmanager
import psutil
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time

@dataclass
class TranscriptionTask:
    audio_path: str
    output_path: str
    priority: int = 0
    retry_count: int = 0

class SharedModelManager:
    """Improved shared model manager with better memory utilization"""
    
    def __init__(self, model_name: str, max_workers: int = 2):
        self.model_name = model_name
        self.max_workers = max_workers
        self.device_models = {}  # device -> model mapping
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.shutdown_event = threading.Event()
        self._lock = threading.Lock()
        
    def start_workers(self):
        """Start worker threads for each GPU"""
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        for gpu_id in range(min(gpu_count, self.max_workers)):
            worker = threading.Thread(
                target=self._worker_loop, 
                args=(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
    def _worker_loop(self, device: str):
        """Worker loop for processing transcription tasks"""
        # Load model once per worker/GPU
        model, processor, pipe = self._load_model_for_device(device)
        
        while not self.shutdown_event.is_set():
            try:
                # Get task with timeout to check shutdown
                priority, task = self.task_queue.get(timeout=1.0)
                
                # Process the task
                result = self._transcribe_single(pipe, task, device)
                self.result_queue.put((task, result))
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker {device} error: {e}")
                if 'task' in locals():
                    self.result_queue.put((task, {"error": str(e)}))
                    
    def _load_model_for_device(self, device: str):
        """Load model optimized for specific device"""
        with self._lock:
            if device in self.device_models:
                return self.device_models[device]
                
        logging.info(f"Loading Whisper model on {device}")
        
        # Optimized model loading
        torch_dtype = torch.float16 if "cuda" in device else torch.float32
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map={"": device} if "cuda" in device else None
        )
        
        # Enable optimizations
        if "cuda" in device:
            model = model.half()  # Ensure fp16
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logging.info(f"Torch compile enabled for {device}")
                except Exception as e:
                    logging.warning(f"Torch compile failed for {device}: {e}")
        
        processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Optimized pipeline settings
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
            batch_size=self._calculate_optimal_batch_size(device),
            return_timestamps=True
        )
        
        with self._lock:
            self.device_models[device] = (model, processor, pipe)
            
        return model, processor, pipe
    
    def _calculate_optimal_batch_size(self, device: str) -> int:
        """Calculate optimal batch size for device"""
        if "cuda" not in device:
            return 4
            
        try:
            gpu_id = int(device.split(":")[1])
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                gpu = gpus[gpu_id]
                total_memory_gb = gpu.memoryTotal / 1024
                
                # More conservative but reliable batch sizing
                if total_memory_gb >= 24:  # RTX 4090, A6000, etc.
                    return 16
                elif total_memory_gb >= 16:  # RTX 4080, etc.
                    return 12
                elif total_memory_gb >= 12:  # RTX 4070 Ti, etc.
                    return 8
                elif total_memory_gb >= 8:   # RTX 4070, etc.
                    return 6
                else:
                    return 4
        except:
            pass
            
        return 6  # Safe default

    @contextmanager
    def memory_efficient_processing(self, device: str):
        """Context manager for memory-efficient processing"""
        if "cuda" in device:
            # Clear cache before processing
            torch.cuda.empty_cache()
            
        try:
            yield
        finally:
            if "cuda" in device:
                # Clear cache after processing
                torch.cuda.empty_cache()

    def _transcribe_single(self, pipe, task: TranscriptionTask, device: str) -> Dict[str, Any]:
        """Transcribe single file with optimizations"""
        try:
            with self.memory_efficient_processing(device):
                # Use autocast for automatic mixed precision
                if "cuda" in device:
                    with autocast():
                        result = pipe(
                            task.audio_path,
                            generate_kwargs={
                                "max_new_tokens": 512,
                                "do_sample": False,
                                "num_beams": 1,
                                "use_cache": True
                            }
                        )
                else:
                    result = pipe(
                        task.audio_path,
                        generate_kwargs={
                            "max_new_tokens": 512,
                            "do_sample": False,
                            "num_beams": 1
                        }
                    )
                
                return {
                    "text": result["text"],
                    "chunks": result.get("chunks", []),
                    "device": device,
                    "success": True
                }
                
        except Exception as e:
            logging.error(f"Transcription failed for {task.audio_path}: {e}")
            return {"error": str(e), "success": False}

class StreamingTranscriber:
    """Streaming transcriber for large files"""
    
    def __init__(self, model_manager: SharedModelManager):
        self.model_manager = model_manager
        
    def transcribe_streaming(self, audio_path: str, chunk_duration: int = 30) -> List[Dict]:
        """Transcribe large files in streaming chunks"""
        import librosa
        
        # Load audio and split into chunks
        y, sr = librosa.load(audio_path, sr=16000)
        chunk_samples = chunk_duration * sr
        chunks = []
        
        for i in range(0, len(y), chunk_samples):
            chunk = y[i:i + chunk_samples]
            if len(chunk) < sr:  # Skip chunks shorter than 1 second
                continue
                
            # Create temporary chunk file
            chunk_path = f"/tmp/chunk_{i}_{int(time.time())}.wav"
            import soundfile as sf
            sf.write(chunk_path, chunk, sr)
            
            chunks.append({
                "path": chunk_path,
                "start_time": i / sr,
                "end_time": min((i + chunk_samples) / sr, len(y) / sr)
            })
            
        return chunks

class OptimizedBatchProcessor:
    """Optimized batch processor with better resource management"""
    
    def __init__(self, model_manager: SharedModelManager):
        self.model_manager = model_manager
        
    def process_batch_with_priority(self, tasks: List[TranscriptionTask]) -> List[Dict]:
        """Process batch with priority queue and load balancing"""
        # Sort tasks by priority and file size
        tasks.sort(key=lambda x: (x.priority, self._get_file_size(x.audio_path)))
        
        # Add tasks to queue
        for i, task in enumerate(tasks):
            self.model_manager.task_queue.put((task.priority, task))
            
        # Collect results
        results = []
        for _ in tasks:
            task, result = self.model_manager.result_queue.get()
            results.append((task, result))
            
        return results
    
    def _get_file_size(self, path: str) -> int:
        """Get file size for sorting"""
        try:
            return os.path.getsize(path)
        except:
            return 0

# Example usage improvements
class ImprovedTranscriptionEngine:
    """Main engine with all improvements"""
    
    def __init__(self, model_name: str = "openai/whisper-large-v3-turbo"):
        self.model_name = model_name
        self.model_manager = SharedModelManager(model_name, max_workers=2)
        self.batch_processor = OptimizedBatchProcessor(self.model_manager)
        self.streaming_transcriber = StreamingTranscriber(self.model_manager)
        
    def start(self):
        """Start the transcription engine"""
        self.model_manager.start_workers()
        
    def transcribe_files(self, file_paths: List[str], use_streaming: bool = False) -> List[Dict]:
        """Main transcription method with optimizations"""
        tasks = []
        
        for path in file_paths:
            # Determine if file should use streaming
            if use_streaming or self._should_use_streaming(path):
                chunks = self.streaming_transcriber.transcribe_streaming(path)
                for chunk in chunks:
                    task = TranscriptionTask(
                        audio_path=chunk["path"],
                        output_path=path.replace(".wav", f"_chunk_{chunk['start_time']:.2f}.txt"),
                        priority=1 if self._is_large_file(path) else 0
                    )
                    tasks.append(task)
            else:
                task = TranscriptionTask(
                    audio_path=path,
                    output_path=path.replace(".wav", ".txt"),
                    priority=0
                )
                tasks.append(task)
                
        return self.batch_processor.process_batch_with_priority(tasks)
    
    def _should_use_streaming(self, path: str) -> bool:
        """Determine if file should use streaming based on size/duration"""
        try:
            import librosa
            duration = librosa.get_duration(filename=path)
            return duration > 300  # 5 minutes
        except:
            return False
            
    def _is_large_file(self, path: str) -> bool:
        """Check if file is large and should get priority"""
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            return size_mb > 50  # 50MB threshold
        except:
            return False
            
    def shutdown(self):
        """Graceful shutdown"""
        self.model_manager.shutdown_event.set()
        for worker in self.model_manager.workers:
            worker.join(timeout=5) 