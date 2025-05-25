#!/usr/bin/env python3

import subprocess
import sys
import os

def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"Installing {package_name}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                              capture_output=True, text=True, check=True)
        print(f"✓ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package_name}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error installing {package_name}: {e}")
        return False

def main():
    """Install all required dependencies for optimal TransFixer performance."""
    print("🚀 Installing TransFixer performance dependencies...")
    print("=" * 50)
    
    # List of packages to install
    packages = [
        "faster-whisper",      # CTranslate2 backend for 4-8x speed improvement
        "ctranslate2",         # Core CTranslate2 library
        # "flash-attn",        # Commented out as it requires specific CUDA setup
    ]
    
    success_count = 0
    
    for package in packages:
        if install_package(package):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"Installation Summary: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("🎉 All performance dependencies installed successfully!")
        print("\nYou can now run TransFixer with maximum performance:")
        print("   python3 transfixer.py")
    else:
        print("⚠ Some dependencies failed to install.")
        print("TransFixer will still work with the transformers backend.")
        print("\nTo install manually:")
        for package in packages:
            print(f"   pip install {package}")
    
    print("\nNote: Flash Attention 2 requires specific CUDA setup.")
    print("If you have a compatible GPU, install it manually:")
    print("   pip install flash-attn --no-build-isolation")

if __name__ == "__main__":
    main() 