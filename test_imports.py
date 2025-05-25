#!/usr/bin/env python3

import sys
import traceback

def test_import(module_name, description):
    """Test importing a module and report status."""
    try:
        if module_name == "faster_whisper":
            from faster_whisper import WhisperModel
            print(f"✓ {description}: Available")
            return True
        elif module_name == "flash_attn":
            import flash_attn
            print(f"✓ {description}: Available")
            return True
        elif module_name == "transformers":
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            print(f"✓ {description}: Available")
            return True
        elif module_name == "torch":
            import torch
            print(f"✓ {description}: Available")
            if torch.cuda.is_available():
                print(f"   - CUDA devices: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print("   - CUDA: Not available")
            return True
        elif module_name == "config":
            from config import USE_FASTER_WHISPER_BACKEND, PERFORMANCE_MODE
            print(f"✓ {description}: Available")
            print(f"   - Performance Mode: {PERFORMANCE_MODE}")
            print(f"   - Use Faster-Whisper: {USE_FASTER_WHISPER_BACKEND}")
            return True
        else:
            __import__(module_name)
            print(f"✓ {description}: Available")
            return True
    except ImportError as e:
        print(f"✗ {description}: Not available - {e}")
        return False
    except Exception as e:
        print(f"✗ {description}: Error - {e}")
        return False

def main():
    """Test all imports required by TransFixer."""
    print("🔍 Testing TransFixer Dependencies")
    print("=" * 50)
    
    tests = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers (Hugging Face)"),
        ("faster_whisper", "Faster-Whisper (CTranslate2)"),
        ("flash_attn", "Flash Attention 2"),
        ("config", "TransFixer Configuration"),
        ("psutil", "System Monitoring"),
        ("GPUtil", "GPU Monitoring"),
        ("tqdm", "Progress Bars"),
        ("requests", "HTTP Requests"),
    ]
    
    passed = 0
    total = len(tests)
    
    for module, description in tests:
        if test_import(module, description):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Import Test Summary: {passed}/{total} modules available")
    
    if passed >= total - 2:  # Allow 2 optional modules to fail
        print("🎉 TransFixer should work correctly!")
        if passed < total:
            print("⚠ Some optional performance features may not be available.")
    else:
        print("❌ Critical dependencies missing. Please install requirements:")
        print("   pip install -r requirements.txt")
        print("   python3 install_dependencies.py")
    
    # Test the main TransFixer imports
    print("\n🧪 Testing TransFixer Main Script...")
    try:
        # Import just the core modules that transfixer.py uses
        import os, logging, multiprocessing, time, datetime, pathlib, signal, sys
        print("✓ Core Python modules: OK")
        
        from config import PERFORMANCE_MODE, USE_FASTER_WHISPER_BACKEND
        print("✓ TransFixer config: OK")
        
        print("✓ TransFixer should start successfully!")
        
    except Exception as e:
        print(f"✗ TransFixer import test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 