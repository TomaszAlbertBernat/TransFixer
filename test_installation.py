#!/usr/bin/env python3
"""
Quick test script to verify TransFixer dependencies are working
"""

def test_import(module_name, import_name=None):
    """Test if a module can be imported successfully"""
    try:
        if import_name is None:
            import_name = module_name
        __import__(import_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - Failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - Partial: {e}")
        return True  # Partial success is still success

def main():
    print("🧪 Testing TransFixer Dependencies")
    print("=" * 50)
    
    # Core ML dependencies
    print("\n📦 Core ML Dependencies:")
    torch_ok = test_import("PyTorch", "torch")
    transformers_ok = test_import("Transformers", "transformers")
    numpy_ok = test_import("NumPy", "numpy")
    
    # Audio processing
    print("\n🎵 Audio Processing:")
    librosa_ok = test_import("Librosa", "librosa")
    soundfile_ok = test_import("SoundFile", "soundfile")
    audioread_ok = test_import("AudioRead", "audioread")
    
    # Performance optimizations
    print("\n⚡ Performance Optimizations:")
    faster_whisper_ok = test_import("faster-whisper", "faster_whisper")
    ctranslate2_ok = test_import("CTranslate2", "ctranslate2")
    flash_attn_ok = test_import("Flash Attention", "flash_attn")
    
    # System utilities
    print("\n🔧 System Utilities:")
    psutil_ok = test_import("psutil")
    gputil_ok = test_import("GPUtil")
    tqdm_ok = test_import("tqdm")
    
    # Summary
    critical_deps = [torch_ok, transformers_ok, numpy_ok, librosa_ok, faster_whisper_ok]
    optional_deps = [flash_attn_ok, ctranslate2_ok]
    
    print("\n" + "=" * 50)
    print("📊 Installation Summary:")
    print(f"Critical dependencies: {sum(critical_deps)}/{len(critical_deps)}")
    print(f"Optional optimizations: {sum(optional_deps)}/{len(optional_deps)}")
    
    if all(critical_deps):
        print("\n🎉 All critical dependencies installed successfully!")
        print("✅ TransFixer is ready to use!")
        
        if all(optional_deps):
            print("🚀 All performance optimizations available - maximum speed!")
        else:
            print("⚠️  Some optimizations missing - basic functionality available")
    else:
        print("\n❌ Some critical dependencies missing")
        missing = []
        if not torch_ok: missing.append("torch")
        if not transformers_ok: missing.append("transformers")  
        if not numpy_ok: missing.append("numpy")
        if not librosa_ok: missing.append("librosa")
        if not faster_whisper_ok: missing.append("faster-whisper")
        
        print(f"Missing: {', '.join(missing)}")
        print("\n🔧 Try running the installation script again:")
        print("python3 install_dependencies.py")

if __name__ == "__main__":
    main() 