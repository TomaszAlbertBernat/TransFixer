#!/usr/bin/env python3
"""
Test script for TransFixer optimizations
Verifies that all low-risk improvements are working correctly
"""

import torch
import GPUtil
import logging
from config import (
    GPU_MEMORY_FRACTION, CONSERVATIVE_BATCH_SIZING, ENABLE_MIXED_PRECISION, 
    SMART_GPU_SELECTION, DEFAULT_CHUNK_LENGTH
)

def test_gpu_availability():
    """Test GPU availability and CUDA setup"""
    print("🔍 Testing GPU Setup...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA is available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        return True
    else:
        print("⚠️  CUDA not available, will use CPU mode")
        return False

def test_mixed_precision_support():
    """Test automatic mixed precision support"""
    print("\n🔍 Testing Mixed Precision Support...")
    
    if not torch.cuda.is_available():
        print("⚠️  Mixed precision requires CUDA")
        return False
    
    try:
        from torch.cuda.amp import autocast
        
        # Test if autocast works
        with autocast():
            x = torch.randn(10, 10, device='cuda')
            y = torch.mm(x, x)
        
        print("✅ Automatic Mixed Precision (AMP) is supported")
        return True
    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        return False

def test_gpu_memory_info():
    """Test GPU memory detection"""
    print("\n🔍 Testing GPU Memory Detection...")
    
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            print("❌ No GPUs detected by GPUtil")
            return False
        
        for gpu in gpus:
            available = gpu.memoryTotal - gpu.memoryUsed
            utilization = (gpu.memoryUsed / gpu.memoryTotal) * 100
            print(f"✅ GPU {gpu.id}: {available}MB available, {utilization:.1f}% used")
        
        return True
    except Exception as e:
        print(f"❌ GPU memory detection failed: {e}")
        return False

def test_smart_gpu_selection():
    """Test the smart GPU selection function"""
    print("\n🔍 Testing Smart GPU Selection...")
    
    try:
        # Import the function from our main module
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        
        # We'll need to import after adding to path
        from transfixer import select_best_gpu
        
        best_gpu = select_best_gpu()
        if best_gpu is not None:
            print(f"✅ Smart GPU selection chose GPU {best_gpu}")
            return True
        else:
            print("⚠️  No GPU selected (likely CPU-only mode)")
            return False
    except Exception as e:
        print(f"❌ Smart GPU selection test failed: {e}")
        return False

def test_configuration_loading():
    """Test that configuration values are loaded correctly"""
    print("\n🔍 Testing Configuration Loading...")
    
    config_tests = [
        ("GPU Memory Fraction", GPU_MEMORY_FRACTION, float, 0.1, 1.0),
        ("Conservative Batch Sizing", CONSERVATIVE_BATCH_SIZING, bool),
        ("Mixed Precision", ENABLE_MIXED_PRECISION, bool),
        ("Smart GPU Selection", SMART_GPU_SELECTION, bool),
        ("Default Chunk Length", DEFAULT_CHUNK_LENGTH, int, 10, 60),
    ]
    
    all_passed = True
    for name, value, expected_type, *bounds in config_tests:
        if not isinstance(value, expected_type):
            print(f"❌ {name}: Expected {expected_type.__name__}, got {type(value).__name__}")
            all_passed = False
        elif bounds and not (bounds[0] <= value <= bounds[1]):
            print(f"❌ {name}: Value {value} out of range [{bounds[0]}, {bounds[1]}]")
            all_passed = False
        else:
            print(f"✅ {name}: {value}")
    
    return all_passed

def test_memory_calculation():
    """Test conservative batch size calculation"""
    print("\n🔍 Testing Memory Calculation...")
    
    try:
        from transfixer import calculate_conservative_batch_size
        
        # Test with different mock devices
        test_cases = [
            ("cpu", 4),
            ("cuda:0", None),  # Will depend on actual GPU memory
        ]
        
        for device, expected in test_cases:
            batch_size = calculate_conservative_batch_size(device)
            if device == "cpu":
                if batch_size == expected:
                    print(f"✅ CPU batch size: {batch_size}")
                else:
                    print(f"❌ CPU batch size: Expected {expected}, got {batch_size}")
                    return False
            else:
                if isinstance(batch_size, int) and batch_size > 0:
                    print(f"✅ GPU batch size: {batch_size}")
                else:
                    print(f"❌ Invalid GPU batch size: {batch_size}")
                    return False
        
        return True
    except Exception as e:
        print(f"❌ Memory calculation test failed: {e}")
        return False

def main():
    """Run all optimization tests"""
    print("🚀 TransFixer Optimization Test Suite")
    print("=" * 50)
    
    tests = [
        test_gpu_availability,
        test_mixed_precision_support,
        test_gpu_memory_info,
        test_configuration_loading,
        test_memory_calculation,
        test_smart_gpu_selection,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optimizations are working correctly!")
        print("\n💡 Recommendations:")
        if ENABLE_MIXED_PRECISION and torch.cuda.is_available():
            print("   • Mixed precision is enabled - expect ~1.5-2x speed improvement")
        if CONSERVATIVE_BATCH_SIZING:
            print("   • Conservative batch sizing is enabled - expect fewer OOM errors")
        if SMART_GPU_SELECTION and torch.cuda.device_count() > 1:
            print("   • Smart GPU selection is enabled - will use best available GPU")
    else:
        print("⚠️  Some optimizations may not work correctly.")
        print("   Check the error messages above and adjust config.py if needed.")
    
    print("\n📋 Current Configuration:")
    print(f"   GPU Memory Fraction: {GPU_MEMORY_FRACTION}")
    print(f"   Conservative Batch Sizing: {CONSERVATIVE_BATCH_SIZING}")
    print(f"   Mixed Precision: {ENABLE_MIXED_PRECISION}")
    print(f"   Smart GPU Selection: {SMART_GPU_SELECTION}")
    print(f"   Default Chunk Length: {DEFAULT_CHUNK_LENGTH}s")

if __name__ == "__main__":
    main() 