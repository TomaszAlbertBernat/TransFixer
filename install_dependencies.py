#!/usr/bin/env python3
"""
TransFixer Dependency Installation Script
Handles proper installation order for all dependencies including PyTorch and flash-attn
"""

import subprocess
import sys
import platform
import os

def run_command(command, description="", check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print("✅ Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        if check:
            return False
        return True

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_core_dependencies():
    """Install core dependencies first"""
    print("\n🚀 Installing core dependencies...")
    
    # Install core PyTorch and basic dependencies first
    core_deps = [
        "torch>=2.3.0",
        "numpy>=2.2.5", 
        "tqdm>=4.67.1",
        "requests>=2.32.3",
        "psutil>=7.0.0"
    ]
    
    for dep in core_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep}"):
            return False
    
    return True

def install_ml_dependencies():
    """Install ML and transformers dependencies"""
    print("\n🤖 Installing ML dependencies...")
    
    ml_deps = [
        "transformers>=4.46.0",
        "accelerate>=1.6.0", 
        "datasets>=3.6.0"
    ]
    
    for dep in ml_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep}"):
            return False
    
    return True

def install_audio_dependencies():
    """Install audio processing dependencies"""
    print("\n🎵 Installing audio processing dependencies...")
    
    audio_deps = [
        "librosa>=0.11.0",
        "soundfile>=0.13.1", 
        "audioread>=3.0.1"
    ]
    
    for dep in audio_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep}"):
            return False
    
    return True

def install_gpu_dependencies():
    """Install GPU utility dependencies"""
    print("\n🖥️  Installing GPU monitoring dependencies...")
    
    if not run_command("pip install gputil>=1.4.0", "Installing GPUtil"):
        return False
    
    return True

def install_performance_optimizations():
    """Install optional performance optimizations"""
    print("\n⚡ Installing performance optimizations...")
    
    # Try to install flash-attn (may fail on some systems)
    print("\n🔥 Attempting to install Flash Attention 2...")
    print("⚠️  Note: This may take several minutes and requires a compatible GPU")
    print("⚠️  If this fails, the system will work without it (just slower)")
    
    success = run_command(
        "pip install flash-attn>=2.0.0 --no-build-isolation", 
        "Installing Flash Attention 2",
        check=False
    )
    
    if not success:
        print("⚠️  Flash Attention installation failed - this is optional")
        print("   The system will work without it, just with less memory optimization")
    else:
        print("✅ Flash Attention 2 installed successfully!")
    
    # Try to install faster-whisper
    print("\n🚀 Installing faster-whisper...")
    if not run_command("pip install faster-whisper>=1.1.0", "Installing faster-whisper"):
        print("⚠️  faster-whisper installation failed")
        return False
    
    # Try to install ctranslate2
    if not run_command("pip install ctranslate2>=4.5.0", "Installing CTranslate2"):
        print("⚠️  ctranslate2 installation failed")
        return False
    
    return True

def verify_installation():
    """Verify that key dependencies are installed"""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        "torch",
        "transformers", 
        "numpy",
        "librosa",
        "faster_whisper"
    ]
    
    failed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed successfully")
        except ImportError:
            print(f"❌ {package} not found")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        return False
    
    print("\n✅ All core dependencies installed successfully!")
    return True

def main():
    """Main installation process"""
    print("🚀 TransFixer Dependency Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in a virtual environment (recommended)
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in a virtual environment - this is still okay")
    
    # Install dependencies in order
    steps = [
        ("Core Dependencies", install_core_dependencies),
        ("ML Dependencies", install_ml_dependencies), 
        ("Audio Dependencies", install_audio_dependencies),
        ("GPU Dependencies", install_gpu_dependencies),
        ("Performance Optimizations", install_performance_optimizations)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*60}")
        print(f"📦 {step_name}")
        print(f"{'='*60}")
        
        if not step_func():
            print(f"❌ Failed during: {step_name}")
            print("🔧 You can continue anyway - some features may not be available")
        else:
            print(f"✅ {step_name} completed successfully")
    
    # Verify installation
    verify_installation()
    
    print("\n" + "="*60)
    print("🎉 Installation process complete!")
    print("="*60)
    print("\n📝 Next steps:")
    print("1. Place audio files in the 'audio/' directory")
    print("2. Run: python3 transfixer.py")
    print("\n✨ Happy transcribing!")

if __name__ == "__main__":
    main() 