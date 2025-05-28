#!/usr/bin/env python3
"""
GTX 1070 cuDNN Diagnostic Tool for TransFixer
==============================================

This script diagnoses cuDNN compatibility issues specifically for GTX 1070 (Pascal architecture)
and provides tailored solutions for the most common problems.

Usage:
    python gtx1070_diagnostic.py

The script will:
1. Detect your GPU and CUDA setup
2. Test cuDNN compatibility with different compute types
3. Provide specific recommendations for GTX 1070
4. Test TransFixer components
"""

import os
import sys
import subprocess
import platform
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section."""
    print(f"\n🔍 {title}")
    print("-" * 50)

def run_command(cmd, capture_output=True):
    """Run a command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_system_info():
    """Check basic system information."""
    print_section("System Information")
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"Architecture: {platform.machine()}")
    
    # Check if running in WSL
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            if 'microsoft' in version_info.lower() or 'wsl' in version_info.lower():
                print("🐧 WSL Environment: Detected")
                print("💡 Note: WSL may have additional CUDA/cuDNN considerations")
            else:
                print("🐧 Native Linux Environment")
    except:
        print("🪟 Windows Environment")

def check_nvidia_driver():
    """Check NVIDIA driver installation."""
    print_section("NVIDIA Driver Check")
    
    success, output, error = run_command("nvidia-smi")
    if success:
        print("✅ NVIDIA driver is installed")
        lines = output.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"   {line.strip()}")
            elif 'GTX 1070' in line or 'GeForce' in line:
                print(f"   GPU: {line.strip()}")
        return True
    else:
        print("❌ NVIDIA driver not found or not working")
        print("💡 Install latest NVIDIA drivers from nvidia.com")
        return False

def check_cuda_installation():
    """Check CUDA installation."""
    print_section("CUDA Installation Check")
    
    # Check nvcc
    success, output, error = run_command("nvcc --version")
    if success:
        lines = output.split('\n')
        for line in lines:
            if 'release' in line.lower():
                print(f"✅ CUDA Compiler: {line.strip()}")
                cuda_version = line.split('release')[1].split(',')[0].strip()
                return True, cuda_version
    
    # Check if CUDA is in PATH
    cuda_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\*/bin\\nvcc.exe"
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✅ Found CUDA at: {path}")
            return True, "unknown"
    
    print("❌ CUDA not found")
    print("💡 Install CUDA 11.8 for best GTX 1070 compatibility:")
    print("   https://developer.nvidia.com/cuda-11-8-0-download-archive")
    return False, None

def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    print_section("PyTorch CUDA Check")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            
            device_count = torch.cuda.device_count()
            print(f"   GPU count: {device_count}")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"           Compute Capability: {props.major}.{props.minor}")
                print(f"           Memory: {props.total_memory / 1024**3:.1f} GB")
                
                # Check if this is a GTX 1070
                if "gtx 1070" in props.name.lower():
                    print("   🎯 GTX 1070 detected!")
                    if props.major == 6 and props.minor == 1:
                        print("   ✅ Compute Capability 6.1 - supports int8")
                    return True, "gtx1070"
                elif props.major == 6:
                    print("   🎯 Pascal architecture detected")
                    return True, "pascal"
            
            return True, "other"
        else:
            print("❌ CUDA not available in PyTorch")
            print("💡 Install CUDA-enabled PyTorch:")
            print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
            return False, None
            
    except ImportError:
        print("❌ PyTorch not installed")
        print("💡 Install PyTorch with CUDA support:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return False, None

def test_cudnn_operations(gpu_type):
    """Test cuDNN operations with different compute types."""
    print_section("cuDNN Operations Test")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available for testing")
            return False
        
        # Test basic CUDA operation
        print("Testing basic CUDA operation...")
        try:
            test_tensor = torch.randn(100, 100, device='cuda')
            result = torch.matmul(test_tensor, test_tensor)
            print("✅ Basic CUDA operations work")
        except Exception as e:
            print(f"❌ Basic CUDA operations failed: {e}")
            return False
        
        # Test cuDNN operations
        print("Testing cuDNN operations...")
        try:
            test_tensor = torch.randn(1, 1, 10, 10, device='cuda')
            conv = torch.nn.Conv2d(1, 1, 3, padding=1).cuda()
            with torch.no_grad():
                result = conv(test_tensor)
            print("✅ cuDNN operations work")
        except Exception as e:
            print(f"❌ cuDNN operations failed: {e}")
            print("💡 This indicates a cuDNN installation issue")
            return False
        
        # Test different precision types for GTX 1070
        if gpu_type in ["gtx1070", "pascal"]:
            print("\nTesting precision types for Pascal GPU...")
            
            # Test float32 (should always work)
            try:
                with torch.cuda.device(0):
                    test_tensor = torch.randn(32, 32, device='cuda', dtype=torch.float32)
                    result = torch.matmul(test_tensor, test_tensor)
                print("✅ float32 operations work")
            except Exception as e:
                print(f"❌ float32 operations failed: {e}")
            
            # Test float16 (may not work efficiently on GTX 1070)
            try:
                with torch.cuda.device(0):
                    test_tensor = torch.randn(32, 32, device='cuda', dtype=torch.float16)
                    result = torch.matmul(test_tensor, test_tensor)
                print("⚠️  float16 operations work (but may be slow on GTX 1070)")
            except Exception as e:
                print(f"❌ float16 operations failed: {e}")
                print("💡 This is expected on GTX 1070 - use int8 or float32 instead")
        
        return True
        
    except Exception as e:
        print(f"❌ cuDNN test failed: {e}")
        return False

def test_faster_whisper():
    """Test faster-whisper installation and compatibility."""
    print_section("faster-whisper Test")
    
    try:
        from faster_whisper import WhisperModel
        print("✅ faster-whisper is installed")
        
        # Test model loading with different compute types
        test_model = "tiny"  # Use tiny model for quick testing
        
        compute_types = ["int8", "float32"]
        
        for compute_type in compute_types:
            print(f"Testing {compute_type} compute type...")
            try:
                model = WhisperModel(test_model, device="cuda", compute_type=compute_type)
                print(f"✅ {compute_type} model loading works")
                
                # Test transcription with a short audio (dummy test)
                # Note: This would need actual audio file to test fully
                print(f"   Model loaded successfully with {compute_type}")
                
                # Clean up
                del model
                
            except Exception as e:
                print(f"❌ {compute_type} failed: {e}")
                if "float16" in str(e).lower() and compute_type == "float16":
                    print("   💡 Expected on GTX 1070 - use int8 instead")
        
        return True
        
    except ImportError:
        print("❌ faster-whisper not installed")
        print("💡 Install with: pip install faster-whisper")
        return False
    except Exception as e:
        print(f"❌ faster-whisper test failed: {e}")
        return False

def provide_gtx1070_recommendations(issues_found):
    """Provide specific recommendations for GTX 1070 setup."""
    print_header("GTX 1070 Specific Recommendations")
    
    print("🎯 For optimal GTX 1070 (Pascal) performance:")
    print()
    
    print("1. 📦 RECOMMENDED SOFTWARE VERSIONS:")
    print("   • CUDA 11.8 (not 12.x)")
    print("   • cuDNN 8.6.0 or 8.8.0")
    print("   • PyTorch with CUDA 11.8 support:")
    print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print()
    
    print("2. ⚙️  OPTIMAL TRANSFIXER SETTINGS:")
    print("   • Use int8 compute type (best for GTX 1070):")
    print("     python transfixer.py --compute-type int8")
    print("   • Alternative with float32:")
    print("     python transfixer.py --compute-type float32")
    print("   • Force CPU if CUDA issues persist:")
    print("     python transfixer.py --force-cpu")
    print()
    
    print("3. 🔧 ENVIRONMENT VARIABLES (if needed):")
    print("   • For float16 support (not recommended):")
    print("     export CT2_CUDA_ALLOW_FP16=1")
    print("   • Disable cuDNN for debugging:")
    print("     export CUDNN_ENABLED=0")
    print()
    
    print("4. 🚀 PERFORMANCE OPTIMIZATION:")
    print("   • GTX 1070 works best with:")
    print("     - Smaller batch sizes (4-8)")
    print("     - int8 quantization")
    print("     - faster-whisper backend")
    print("   • Expected performance: ~2-3x realtime for speech")
    print()
    
    if issues_found:
        print("5. 🔧 FIXING DETECTED ISSUES:")
        for issue in issues_found:
            print(f"   • {issue}")
    
    print("\n6. 📋 QUICK SETUP COMMANDS:")
    print("   # Install compatible PyTorch")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("   ")
    print("   # Install faster-whisper")
    print("   pip install faster-whisper")
    print("   ")
    print("   # Test with optimal settings")
    print("   python transfixer.py --compute-type int8 --verbose-logging")

def main():
    """Main diagnostic function."""
    print_header("GTX 1070 cuDNN Diagnostic Tool for TransFixer")
    print("This tool will diagnose your GTX 1070 setup and provide specific recommendations.")
    print()
    
    issues_found = []
    
    # Check system info
    check_system_info()
    
    # Check NVIDIA driver
    if not check_nvidia_driver():
        issues_found.append("Install/update NVIDIA drivers")
    
    # Check CUDA
    cuda_available, cuda_version = check_cuda_installation()
    if not cuda_available:
        issues_found.append("Install CUDA 11.8")
    
    # Check PyTorch
    pytorch_cuda, gpu_type = check_pytorch_cuda()
    if not pytorch_cuda:
        issues_found.append("Install CUDA-enabled PyTorch")
    
    # Test cuDNN operations
    if pytorch_cuda:
        if not test_cudnn_operations(gpu_type):
            issues_found.append("Fix cuDNN installation")
    
    # Test faster-whisper
    if not test_faster_whisper():
        issues_found.append("Install faster-whisper")
    
    # Provide recommendations
    provide_gtx1070_recommendations(issues_found)
    
    print_header("Diagnostic Complete")
    if not issues_found:
        print("🎉 No major issues detected! Your GTX 1070 should work well with TransFixer.")
        print("💡 Run TransFixer with: python transfixer.py --compute-type int8")
    else:
        print(f"⚠️  Found {len(issues_found)} issues to address:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        print("\n💡 Follow the recommendations above to fix these issues.")

if __name__ == "__main__":
    main() 