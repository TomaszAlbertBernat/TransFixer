#!/usr/bin/env python3
"""
TransFixer Performance Summary & Optimization Recommendations
Analyzes current system configuration and provides optimization recommendations
"""

import os
import sys
import json
import torch
import psutil
import GPUtil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

def check_system_capabilities() -> Dict[str, Any]:
    """Check current system capabilities and installed optimizations"""
    capabilities = {
        "system": {
            "cpu_cores": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": sys.version,
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "compile_available": hasattr(torch, 'compile'),
        },
        "gpu": [],
        "optimizations": {
            "faster_whisper": False,
            "flash_attention": False,
            "ctranslate2": False,
            "transformers_version": None,
        }
    }
    
    # Check GPU information
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                capabilities["gpu"].append({
                    "name": gpu.name,
                    "memory_gb": round(gpu.memoryTotal / 1024, 2),
                    "available_memory_gb": round((gpu.memoryTotal - gpu.memoryUsed) / 1024, 2),
                    "utilization": gpu.load * 100,
                    "temperature": gpu.temperature
                })
            capabilities["pytorch"]["cuda_version"] = torch.version.cuda
        except Exception as e:
            print(f"Warning: Could not get GPU info: {e}")
    
    # Check for optimization libraries
    try:
        import faster_whisper
        capabilities["optimizations"]["faster_whisper"] = faster_whisper.__version__
    except ImportError:
        pass
    
    try:
        import flash_attn
        capabilities["optimizations"]["flash_attention"] = flash_attn.__version__
    except ImportError:
        pass
    
    try:
        import ctranslate2
        capabilities["optimizations"]["ctranslate2"] = ctranslate2.__version__
    except ImportError:
        pass
    
    try:
        import transformers
        capabilities["optimizations"]["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    
    return capabilities

def analyze_current_config() -> Dict[str, Any]:
    """Analyze current TransFixer configuration"""
    config_analysis = {
        "config_file_exists": os.path.exists("config.py"),
        "settings": {},
        "recommendations": []
    }
    
    if config_analysis["config_file_exists"]:
        try:
            # Import config settings
            sys.path.insert(0, ".")
            import config
            
            config_analysis["settings"] = {
                "performance_mode": getattr(config, 'PERFORMANCE_MODE', 'unknown'),
                "use_faster_whisper": getattr(config, 'USE_FASTER_WHISPER_BACKEND', False),
                "whisper_model": getattr(config, 'WHISPER_MODEL', 'unknown'),
                "faster_whisper_model": getattr(config, 'FASTER_WHISPER_MODEL', 'unknown'),
                "enable_torch_compile": getattr(config, 'ENABLE_TORCH_COMPILE', False),
                "enable_flash_attention": getattr(config, 'ENABLE_FLASH_ATTENTION', False),
                "enable_mixed_precision": getattr(config, 'ENABLE_MIXED_PRECISION', False),
                "max_batch_size": getattr(config, 'MAX_BATCH_SIZE', 'unknown'),
                "ctranslate2_compute_type": getattr(config, 'CTRANSLATE2_COMPUTE_TYPE', 'unknown'),
            }
        except ImportError as e:
            config_analysis["error"] = f"Could not import config: {e}"
    
    return config_analysis

def generate_recommendations(capabilities: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate optimization recommendations based on system capabilities"""
    recommendations = []
    
    # Check if faster-whisper is available but not enabled
    if capabilities["optimizations"]["faster_whisper"] and not config["settings"].get("use_faster_whisper", False):
        recommendations.append({
            "priority": "HIGH",
            "category": "Backend Optimization",
            "title": "Enable faster-whisper Backend",
            "description": "faster-whisper (CTranslate2) is installed but not enabled",
            "expected_improvement": "4-8x speed improvement",
            "action": "Set USE_FASTER_WHISPER_BACKEND = True in config.py"
        })
    
    # Check if faster-whisper is not installed
    if not capabilities["optimizations"]["faster_whisper"]:
        recommendations.append({
            "priority": "CRITICAL",
            "category": "Backend Optimization", 
            "title": "Install faster-whisper",
            "description": "faster-whisper provides the highest performance improvement",
            "expected_improvement": "4-8x speed improvement",
            "action": "pip install faster-whisper>=1.1.0 ctranslate2>=4.5.0"
        })
    
    # Check torch.compile availability
    if capabilities["pytorch"]["compile_available"] and not config["settings"].get("enable_torch_compile", False):
        recommendations.append({
            "priority": "HIGH",
            "category": "PyTorch Optimization",
            "title": "Enable torch.compile",
            "description": "PyTorch 2.0+ compile optimization available but not enabled",
            "expected_improvement": "4.5x speed improvement",
            "action": "Set ENABLE_TORCH_COMPILE = True in config.py"
        })
    
    # Check Flash Attention
    if not capabilities["optimizations"]["flash_attention"]:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Memory Optimization",
            "title": "Install Flash Attention 2",
            "description": "Flash Attention 2 reduces memory usage by 50% and increases speed 2-3x",
            "expected_improvement": "2-3x speed improvement, 50% less memory",
            "action": "pip install flash-attn>=2.0.0"
        })
    
    # GPU-specific recommendations
    if capabilities["gpu"]:
        for gpu in capabilities["gpu"]:
            gpu_memory = gpu["memory_gb"]
            
            if gpu_memory >= 24:  # RTX 4090 class
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Hardware Optimization",
                    "title": f"Optimize for {gpu['name']} (24GB+)",
                    "description": "High-end GPU detected - use aggressive performance mode",
                    "expected_improvement": "Maximum throughput",
                    "action": "Set PERFORMANCE_MODE = 'aggressive', MAX_BATCH_SIZE = 32"
                })
            elif gpu_memory >= 16:  # RTX 3080 Ti class
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Hardware Optimization", 
                    "title": f"Optimize for {gpu['name']} (16GB+)",
                    "description": "High-memory GPU - use balanced aggressive settings",
                    "expected_improvement": "High throughput",
                    "action": "Set PERFORMANCE_MODE = 'balanced', MAX_BATCH_SIZE = 20"
                })
            elif gpu_memory >= 8:  # RTX 3060 class
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Hardware Optimization",
                    "title": f"Optimize for {gpu['name']} (8GB+)",
                    "description": "Mid-range GPU - use balanced settings",
                    "expected_improvement": "Good throughput",
                    "action": "Set PERFORMANCE_MODE = 'balanced', MAX_BATCH_SIZE = 12"
                })
            else:  # Lower memory GPUs
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Hardware Optimization",
                    "title": f"Optimize for {gpu['name']} (<8GB)",
                    "description": "Lower-memory GPU - use conservative settings and INT8",
                    "expected_improvement": "Stable performance",
                    "action": "Set PERFORMANCE_MODE = 'conservative', CTRANSLATE2_COMPUTE_TYPE = 'int8'"
                })
    
    # Model recommendations
    current_model = config["settings"].get("whisper_model", "").lower()
    if "large-v3" in current_model and "turbo" not in current_model:
        recommendations.append({
            "priority": "HIGH",
            "category": "Model Optimization",
            "title": "Use Whisper Large-v3-Turbo",
            "description": "Turbo model provides 8x speed improvement with minimal accuracy loss",
            "expected_improvement": "8x speed improvement",
            "action": "Set WHISPER_MODEL = 'openai/whisper-large-v3-turbo'"
        })
    
    # Performance mode recommendations
    current_mode = config["settings"].get("performance_mode", "").lower()
    if current_mode in ["", "unknown", "conservative"] and capabilities["gpu"] and capabilities["gpu"][0]["memory_gb"] >= 8:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Performance Configuration",
            "title": "Increase Performance Mode",
            "description": "Your GPU has sufficient memory for higher performance settings",
            "expected_improvement": "2-4x throughput improvement",
            "action": "Set PERFORMANCE_MODE = 'balanced' or 'aggressive'"
        })
    
    return recommendations

def calculate_estimated_performance(capabilities: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate estimated performance improvements"""
    baseline_speed = 1.0
    current_speed = baseline_speed
    optimized_speed = baseline_speed
    
    # Current optimizations
    if config["settings"].get("use_faster_whisper", False):
        current_speed *= 5.0
    if config["settings"].get("enable_torch_compile", False):
        current_speed *= 4.5
    if capabilities["optimizations"]["flash_attention"]:
        current_speed *= 2.5
    
    # Potential optimizations
    optimized_speed = baseline_speed
    if capabilities["optimizations"]["faster_whisper"] or config["settings"].get("use_faster_whisper", False):
        optimized_speed *= 5.0
    if capabilities["pytorch"]["compile_available"]:
        optimized_speed *= 4.5
    if capabilities["optimizations"]["flash_attention"]:
        optimized_speed *= 2.5
    
    # Model improvements
    current_model = config["settings"].get("whisper_model", "").lower()
    if "turbo" in current_model:
        optimized_speed *= 8.0
    elif "distil" in current_model:
        optimized_speed *= 6.3
    
    return {
        "baseline_files_per_hour": 20,
        "current_estimated_files_per_hour": int(20 * current_speed),
        "optimized_estimated_files_per_hour": int(20 * optimized_speed),
        "current_speedup": f"{current_speed:.1f}x",
        "potential_speedup": f"{optimized_speed:.1f}x",
        "improvement_potential": f"{optimized_speed/current_speed:.1f}x" if current_speed > 0 else "∞"
    }

def generate_optimal_config(capabilities: Dict[str, Any]) -> str:
    """Generate optimal configuration based on system capabilities"""
    config_lines = [
        "# TransFixer Optimal Configuration",
        "# Generated based on your system capabilities",
        f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "# Backend Selection (HIGHEST PRIORITY)",
    ]
    
    if capabilities["optimizations"]["faster_whisper"]:
        config_lines.extend([
            "USE_FASTER_WHISPER_BACKEND = True",
            "FASTER_WHISPER_MODEL = 'large-v3-turbo'  # 8x faster than large-v3",
        ])
    else:
        config_lines.extend([
            "# Install faster-whisper for maximum performance:",
            "# pip install faster-whisper>=1.1.0 ctranslate2>=4.5.0",
            "USE_FASTER_WHISPER_BACKEND = False",
            "WHISPER_MODEL = 'openai/whisper-large-v3-turbo'",
        ])
    
    config_lines.extend([
        "",
        "# Performance Mode"
    ])
    
    if capabilities["gpu"] and capabilities["gpu"][0]["memory_gb"] >= 16:
        config_lines.append("PERFORMANCE_MODE = 'aggressive'")
    elif capabilities["gpu"] and capabilities["gpu"][0]["memory_gb"] >= 8:
        config_lines.append("PERFORMANCE_MODE = 'balanced'")
    else:
        config_lines.append("PERFORMANCE_MODE = 'conservative'")
    
    config_lines.extend([
        "",
        "# PyTorch Optimizations"
    ])
    
    if capabilities["pytorch"]["compile_available"]:
        config_lines.extend([
            "ENABLE_TORCH_COMPILE = True",
            "TORCH_COMPILE_MODE = 'reduce-overhead'",
            "TORCH_COMPILE_FULLGRAPH = True",
        ])
    else:
        config_lines.extend([
            "# Upgrade to PyTorch 2.0+ for torch.compile support",
            "ENABLE_TORCH_COMPILE = False",
        ])
    
    config_lines.extend([
        "",
        "# Memory Optimizations",
        "ENABLE_MIXED_PRECISION = True",
    ])
    
    if capabilities["optimizations"]["flash_attention"]:
        config_lines.extend([
            "ENABLE_FLASH_ATTENTION = True",
            "ATTENTION_IMPLEMENTATION = 'flash_attention_2'",
        ])
    else:
        config_lines.extend([
            "# Install Flash Attention 2 for memory optimization:",
            "# pip install flash-attn>=2.0.0",
            "ENABLE_FLASH_ATTENTION = False",
        ])
    
    if capabilities["optimizations"]["faster_whisper"]:
        config_lines.extend([
            "",
            "# CTranslate2 Optimizations",
        ])
        
        if capabilities["gpu"] and capabilities["gpu"][0]["memory_gb"] >= 8:
            config_lines.append("CTRANSLATE2_COMPUTE_TYPE = 'float16'")
        else:
            config_lines.append("CTRANSLATE2_COMPUTE_TYPE = 'int8'  # For memory-constrained systems")
    
    return "\n".join(config_lines)

def main():
    print("="*80)
    print("TRANSFIXER PERFORMANCE ANALYSIS & OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    print()
    
    # Check system capabilities
    print("🔍 Analyzing system capabilities...")
    capabilities = check_system_capabilities()
    
    # Analyze current configuration
    print("⚙️  Analyzing current configuration...")
    config = analyze_current_config()
    
    # System Summary
    print("📊 SYSTEM SUMMARY")
    print("-" * 40)
    print(f"CPU: {capabilities['system']['cpu_cores']} cores")
    print(f"Memory: {capabilities['system']['memory_gb']} GB")
    print(f"PyTorch: {capabilities['pytorch']['version']}")
    print(f"CUDA Available: {capabilities['pytorch']['cuda_available']}")
    print(f"torch.compile Available: {capabilities['pytorch']['compile_available']}")
    
    if capabilities["gpu"]:
        print("\n🎮 GPU INFORMATION")
        print("-" * 40)
        for i, gpu in enumerate(capabilities["gpu"]):
            print(f"GPU {i}: {gpu['name']}")
            print(f"  Memory: {gpu['memory_gb']} GB (Available: {gpu['available_memory_gb']} GB)")
            print(f"  Utilization: {gpu['utilization']:.1f}%")
            print(f"  Temperature: {gpu['temperature']}°C")
    else:
        print("\n⚠️  No GPU detected - CPU-only mode")
    
    # Optimization Libraries
    print("\n🚀 OPTIMIZATION LIBRARIES")
    print("-" * 40)
    optimizations = capabilities["optimizations"]
    print(f"faster-whisper: {'✓ ' + str(optimizations['faster_whisper']) if optimizations['faster_whisper'] else '✗ Not installed'}")
    print(f"Flash Attention: {'✓ ' + str(optimizations['flash_attention']) if optimizations['flash_attention'] else '✗ Not installed'}")
    print(f"CTranslate2: {'✓ ' + str(optimizations['ctranslate2']) if optimizations['ctranslate2'] else '✗ Not installed'}")
    print(f"transformers: {'✓ ' + str(optimizations['transformers_version']) if optimizations['transformers_version'] else '✗ Not installed'}")
    
    # Current Configuration
    if config["config_file_exists"]:
        print("\n⚙️  CURRENT CONFIGURATION")
        print("-" * 40)
        settings = config["settings"]
        print(f"Performance Mode: {settings.get('performance_mode', 'unknown')}")
        print(f"Use faster-whisper: {settings.get('use_faster_whisper', 'unknown')}")
        print(f"Whisper Model: {settings.get('whisper_model', 'unknown')}")
        print(f"torch.compile: {settings.get('enable_torch_compile', 'unknown')}")
        print(f"Flash Attention: {settings.get('enable_flash_attention', 'unknown')}")
        print(f"Mixed Precision: {settings.get('enable_mixed_precision', 'unknown')}")
        print(f"Max Batch Size: {settings.get('max_batch_size', 'unknown')}")
    else:
        print("\n⚠️  config.py not found")
    
    # Performance Estimation
    performance = calculate_estimated_performance(capabilities, config)
    print("\n📈 PERFORMANCE ESTIMATION")
    print("-" * 40)
    print(f"Baseline Performance: {performance['baseline_files_per_hour']} files/hour")
    print(f"Current Estimated: {performance['current_estimated_files_per_hour']} files/hour ({performance['current_speedup']})")
    print(f"Optimized Potential: {performance['optimized_estimated_files_per_hour']} files/hour ({performance['potential_speedup']})")
    print(f"Improvement Potential: {performance['improvement_potential']} faster")
    
    # Recommendations
    recommendations = generate_recommendations(capabilities, config)
    if recommendations:
        print("\n🎯 OPTIMIZATION RECOMMENDATIONS")
        print("-" * 40)
        
        # Group by priority
        critical = [r for r in recommendations if r["priority"] == "CRITICAL"]
        high = [r for r in recommendations if r["priority"] == "HIGH"]
        medium = [r for r in recommendations if r["priority"] == "MEDIUM"]
        
        for priority, recs in [("CRITICAL", critical), ("HIGH", high), ("MEDIUM", medium)]:
            if recs:
                print(f"\n{priority} PRIORITY:")
                for i, rec in enumerate(recs, 1):
                    print(f"\n{i}. {rec['title']}")
                    print(f"   Category: {rec['category']}")
                    print(f"   Description: {rec['description']}")
                    print(f"   Expected Improvement: {rec['expected_improvement']}")
                    print(f"   Action: {rec['action']}")
    else:
        print("\n✅ No optimization recommendations - your system is well configured!")
    
    # Generate optimal configuration
    print("\n📝 OPTIMAL CONFIGURATION")
    print("-" * 40)
    optimal_config = generate_optimal_config(capabilities)
    print("Optimal config.py settings for your system:")
    print()
    print(optimal_config)
    
    # Save to file
    config_file = Path("optimal_config.py")
    with open(config_file, 'w') as f:
        f.write(optimal_config)
    print(f"\n💾 Optimal configuration saved to: {config_file}")
    
    # Quick Start Instructions
    print("\n🚀 QUICK START INSTRUCTIONS")
    print("-" * 40)
    print("1. Install missing optimization libraries:")
    
    install_commands = []
    if not optimizations["faster_whisper"]:
        install_commands.append("pip install faster-whisper>=1.1.0 ctranslate2>=4.5.0")
    if not optimizations["flash_attention"]:
        install_commands.append("pip install flash-attn>=2.0.0")
    
    if install_commands:
        for cmd in install_commands:
            print(f"   {cmd}")
    else:
        print("   ✓ All optimization libraries are installed!")
    
    print("\n2. Update your config.py with optimal settings (see optimal_config.py)")
    print("\n3. Run benchmark to verify improvements:")
    print("   python benchmark_performance.py test_audio.wav")
    
    print("\n4. Start TransFixer with optimized settings:")
    print("   python transfixer.py")
    
    print("\n" + "="*80)
    print("Analysis complete! Follow the recommendations above for maximum performance.")
    print("="*80)

if __name__ == "__main__":
    main() 