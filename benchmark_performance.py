#!/usr/bin/env python3
"""
TransFixer Performance Benchmark Script
Compares different Whisper optimization strategies for maximum performance
"""

import os
import time
import logging
import argparse
import torch
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# Import TransFixer components
from config import (
    USE_FASTER_WHISPER_BACKEND, FASTER_WHISPER_MODEL, WHISPER_MODEL,
    PERFORMANCE_MODE, ENABLE_TORCH_COMPILE, ENABLE_MIXED_PRECISION
)

# Try to import optimized backends
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for different Whisper optimizations"""
    
    def __init__(self, test_audio_path: str, output_dir: str = "benchmark_results"):
        self.test_audio_path = test_audio_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Test configurations
        self.test_configs = [
            {
                "name": "OpenAI Whisper Large-v3 (Baseline)",
                "backend": "transformers", 
                "model": "openai/whisper-large-v3",
                "optimizations": {"compile": False, "flash_attn": False, "amp": False}
            },
            {
                "name": "OpenAI Whisper Large-v3 + AMP",
                "backend": "transformers",
                "model": "openai/whisper-large-v3", 
                "optimizations": {"compile": False, "flash_attn": False, "amp": True}
            },
            {
                "name": "OpenAI Whisper Large-v3 + Torch Compile",
                "backend": "transformers",
                "model": "openai/whisper-large-v3",
                "optimizations": {"compile": True, "flash_attn": False, "amp": True}
            },
            {
                "name": "OpenAI Whisper Large-v3 + Flash Attention",
                "backend": "transformers",
                "model": "openai/whisper-large-v3",
                "optimizations": {"compile": True, "flash_attn": True, "amp": True}
            },
            {
                "name": "OpenAI Whisper Large-v3-Turbo (8x faster)",
                "backend": "transformers",
                "model": "openai/whisper-large-v3-turbo",
                "optimizations": {"compile": True, "flash_attn": True, "amp": True}
            },
            {
                "name": "Distil-Whisper Large-v3 (6.3x faster)",
                "backend": "transformers",
                "model": "distil-whisper/distil-large-v3",
                "optimizations": {"compile": True, "flash_attn": True, "amp": True}
            },
            {
                "name": "faster-whisper Large-v3 (CTranslate2)",
                "backend": "faster-whisper",
                "model": "large-v3",
                "optimizations": {"compute_type": "float16"}
            },
            {
                "name": "faster-whisper Large-v3-Turbo (CTranslate2)",
                "backend": "faster-whisper", 
                "model": "large-v3-turbo",
                "optimizations": {"compute_type": "float16"}
            },
            {
                "name": "faster-whisper Distil-Large-v3 (CTranslate2)",
                "backend": "faster-whisper",
                "model": "distil-large-v3", 
                "optimizations": {"compute_type": "float16"}
            },
            {
                "name": "faster-whisper Large-v3 INT8 (CTranslate2)",
                "backend": "faster-whisper",
                "model": "large-v3",
                "optimizations": {"compute_type": "int8"}
            },
        ]
        
        self.results = []

    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        info = {
            "cpu": {
                "model": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown",
                "cores": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / 1024**3, 2)
            },
            "gpu": []
        }
        
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    info["gpu"].append({
                        "name": gpu.name,
                        "memory_gb": round(gpu.memoryTotal / 1024, 2),
                        "cuda_version": torch.version.cuda
                    })
            except Exception as e:
                self.logger.warning(f"Could not get GPU info: {e}")
                
        return info

    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage"""
        memory_info = {
            "cpu_memory_mb": psutil.virtual_memory().used / 1024**2,
            "cpu_memory_percent": psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    memory_info.update({
                        "gpu_memory_mb": gpu.memoryUsed,
                        "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100
                    })
            except Exception:
                pass
                
        return memory_info

    def load_transformers_model(self, config: Dict[str, Any]) -> Tuple[Any, Any, float]:
        """Load transformers model with specified optimizations"""
        start_time = time.time()
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True if device != "cpu" else False,
            "use_safetensors": True,
        }
        
        # Add attention optimizations
        if config["optimizations"].get("flash_attn", False):
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.logger.info("Using Flash Attention 2")
            except ImportError:
                self.logger.warning("Flash Attention not available")
                model_kwargs["attn_implementation"] = "sdpa"
        else:
            model_kwargs["attn_implementation"] = "sdpa"
        
        # Load model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(config["model"], **model_kwargs)
        model.to(device)
        
        # Apply torch compile
        if config["optimizations"].get("compile", False) and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                self.logger.info("Applied torch.compile")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}")
        
        # Create processor and pipeline
        processor = AutoProcessor.from_pretrained(config["model"])
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=True
        )
        
        load_time = time.time() - start_time
        return pipe, device, load_time

    def load_faster_whisper_model(self, config: Dict[str, Any]) -> Tuple[Any, str, float]:
        """Load faster-whisper model"""
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError("faster-whisper not available")
            
        start_time = time.time()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = config["optimizations"].get("compute_type", "float16")
        
        model = FasterWhisperModel(
            config["model"],
            device=device,
            compute_type=compute_type
        )
        
        load_time = time.time() - start_time
        return model, device, load_time

    def transcribe_with_transformers(self, pipe: Any, device: str, use_amp: bool = False) -> Tuple[str, float]:
        """Transcribe using transformers pipeline"""
        start_time = time.time()
        
        generate_kwargs = {
            "max_new_tokens": 256,
            "do_sample": False,
            "num_beams": 1,
        }
        
        if use_amp and "cuda" in device:
            from torch.cuda.amp import autocast
            with autocast():
                result = pipe(self.test_audio_path, generate_kwargs=generate_kwargs)
        else:
            result = pipe(self.test_audio_path, generate_kwargs=generate_kwargs)
        
        transcription_time = time.time() - start_time
        return result["text"], transcription_time

    def transcribe_with_faster_whisper(self, model: Any) -> Tuple[str, float]:
        """Transcribe using faster-whisper"""
        start_time = time.time()
        
        segments, info = model.transcribe(
            self.test_audio_path,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False
        )
        
        transcription = " ".join([segment.text for segment in segments])
        transcription_time = time.time() - start_time
        
        return transcription, transcription_time

    def run_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run benchmark for a single configuration"""
        self.logger.info(f"Testing: {config['name']}")
        
        # Measure initial memory
        initial_memory = self.measure_memory_usage()
        
        try:
            if config["backend"] == "faster-whisper":
                model, device, load_time = self.load_faster_whisper_model(config)
                
                # Measure memory after loading
                loaded_memory = self.measure_memory_usage()
                
                # Run transcription
                transcription, transcription_time = self.transcribe_with_faster_whisper(model)
                
            else:  # transformers
                pipe, device, load_time = self.load_transformers_model(config)
                
                # Measure memory after loading
                loaded_memory = self.measure_memory_usage()
                
                # Run transcription
                use_amp = config["optimizations"].get("amp", False)
                transcription, transcription_time = self.transcribe_with_transformers(pipe, device, use_amp)
            
            # Final memory measurement
            final_memory = self.measure_memory_usage()
            
            # Calculate metrics
            result = {
                "config": config,
                "success": True,
                "load_time": load_time,
                "transcription_time": transcription_time,
                "total_time": load_time + transcription_time,
                "transcription": transcription[:100] + "..." if len(transcription) > 100 else transcription,
                "memory": {
                    "initial": initial_memory,
                    "loaded": loaded_memory, 
                    "final": final_memory,
                    "peak_gpu_mb": loaded_memory.get("gpu_memory_mb", 0),
                    "peak_cpu_mb": max(loaded_memory.get("cpu_memory_mb", 0), final_memory.get("cpu_memory_mb", 0))
                },
                "device": device
            }
            
            self.logger.info(f"✓ {config['name']}: {transcription_time:.2f}s transcription, {load_time:.2f}s load")
            
        except Exception as e:
            self.logger.error(f"✗ {config['name']}: {str(e)}")
            result = {
                "config": config,
                "success": False,
                "error": str(e),
                "load_time": 0,
                "transcription_time": 0,
                "total_time": 0
            }
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result

    def run_all_benchmarks(self) -> None:
        """Run all benchmark configurations"""
        self.logger.info("Starting comprehensive Whisper performance benchmark")
        self.logger.info(f"Test audio: {self.test_audio_path}")
        
        # Get system info
        system_info = self.get_system_info()
        self.logger.info(f"System: {system_info}")
        
        # Run benchmarks
        for config in self.test_configs:
            result = self.run_benchmark(config)
            self.results.append(result)
            
            # Wait a bit between tests
            time.sleep(2)
        
        # Generate report
        self.generate_report(system_info)

    def generate_report(self, system_info: Dict[str, Any]) -> None:
        """Generate comprehensive performance report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate baseline (first successful test)
        baseline_time = None
        for result in self.results:
            if result.get("success", False):
                baseline_time = result["transcription_time"]
                break
        
        # Create summary
        summary = {
            "timestamp": timestamp,
            "system_info": system_info,
            "test_audio": str(self.test_audio_path),
            "baseline_time": baseline_time,
            "results": []
        }
        
        print("\n" + "="*100)
        print("TRANSFIXER WHISPER PERFORMANCE BENCHMARK RESULTS")
        print("="*100)
        print(f"Test Audio: {self.test_audio_path}")
        print(f"System: {system_info['cpu']['cores']} cores, {system_info['cpu']['memory_gb']}GB RAM")
        
        if system_info["gpu"]:
            for gpu in system_info["gpu"]:
                print(f"GPU: {gpu['name']} ({gpu['memory_gb']}GB)")
        
        print("\n" + "-"*100)
        print(f"{'Configuration':<50} {'Status':<10} {'Load(s)':<10} {'Trans(s)':<10} {'Speedup':<10} {'GPU(MB)':<10}")
        print("-"*100)
        
        for result in self.results:
            if result.get("success", False):
                speedup = f"{baseline_time/result['transcription_time']:.1f}x" if baseline_time else "N/A"
                gpu_mem = result.get("memory", {}).get("peak_gpu_mb", 0)
                
                print(f"{result['config']['name']:<50} {'✓':<10} {result['load_time']:<10.2f} "
                      f"{result['transcription_time']:<10.2f} {speedup:<10} {gpu_mem:<10.0f}")
                
                # Add to summary
                summary["results"].append({
                    "name": result['config']['name'],
                    "backend": result['config']['backend'],
                    "model": result['config']['model'],
                    "load_time": result['load_time'],
                    "transcription_time": result['transcription_time'],
                    "speedup": baseline_time/result['transcription_time'] if baseline_time else 1.0,
                    "memory_usage_mb": gpu_mem,
                    "transcription_preview": result.get("transcription", "")
                })
            else:
                print(f"{result['config']['name']:<50} {'✗':<10} {'ERROR':<10} {result.get('error', '')}")
        
        print("-"*100)
        
        # Save detailed report
        report_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Print recommendations
        self.print_recommendations()

    def print_recommendations(self) -> None:
        """Print performance optimization recommendations"""
        print("\n" + "="*100)
        print("PERFORMANCE OPTIMIZATION RECOMMENDATIONS")
        print("="*100)
        
        # Find best performers
        successful_results = [r for r in self.results if r.get("success", False)]
        if not successful_results:
            print("No successful benchmarks to analyze")
            return
        
        # Sort by transcription time
        successful_results.sort(key=lambda x: x["transcription_time"])
        
        print("\n🏆 TOP PERFORMERS:")
        for i, result in enumerate(successful_results[:3]):
            print(f"{i+1}. {result['config']['name']}")
            print(f"   Transcription time: {result['transcription_time']:.2f}s")
            print(f"   Backend: {result['config']['backend']}")
            print(f"   Model: {result['config']['model']}")
            print()
        
        print("📋 RECOMMENDATIONS:")
        print("1. Use faster-whisper with CTranslate2 for maximum performance")
        print("2. Consider large-v3-turbo for 8x speed improvement with minimal accuracy loss")
        print("3. Use distil-whisper for 6.3x speed improvement with <1% WER increase")
        print("4. Enable torch.compile for transformers backend (4.5x improvement)")
        print("5. Use INT8 quantization for memory-constrained environments")
        print("6. Enable Flash Attention 2 for memory efficiency")

def main():
    parser = argparse.ArgumentParser(description="TransFixer Whisper Performance Benchmark")
    parser.add_argument("audio_file", help="Path to test audio file")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer configurations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1
    
    # Create benchmark
    benchmark = PerformanceBenchmark(args.audio_file, args.output_dir)
    
    if args.quick:
        # Quick benchmark - only test key configurations
        benchmark.test_configs = [
            config for config in benchmark.test_configs 
            if any(keyword in config["name"] for keyword in ["Baseline", "Turbo", "faster-whisper Large-v3 (CTranslate2)"])
        ]
    
    # Run benchmarks
    benchmark.run_all_benchmarks()
    
    return 0

if __name__ == "__main__":
    exit(main()) 