#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import from transfixer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transfixer import initialize_whisper, model_cache, ensure_dir

def test_single_transcription():
    """Test transcription of a single audio file."""
    print("Starting transcription test...")
    
    # Find the first audio file
    audio_file = None
    for root, dirs, files in os.walk("audio"):
        for file in files:
            if file.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
                audio_file = os.path.join(root, file)
                break
        if audio_file:
            break
    
    if not audio_file:
        print("No audio files found!")
        return False
    
    print(f"Testing transcription of: {audio_file}")
    
    # Initialize Whisper
    try:
        initialize_whisper()
        print("Whisper model initialized successfully")
    except Exception as e:
        print(f"Error initializing Whisper: {e}")
        return False
    
    # Create transcription path
    base_filename = os.path.splitext(os.path.basename(audio_file))[0]
    trans_path = f"test_transcription_{base_filename}.txt"
    
    try:
        print("Starting transcription...")
        with model_cache['lock']:
            pipeline = model_cache['pipeline']
            device = model_cache['device']
            
            result = pipeline(
                audio_file, 
                generate_kwargs={
                    "max_new_tokens": 256,
                    "do_sample": False,
                    "num_beams": 1,
                }
            )
        
        transcription = result["text"]
        print(f"Transcription completed. Length: {len(transcription)} characters")
        
        # Write transcription
        with open(trans_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        print(f"Transcription saved to: {trans_path}")
        print(f"First 200 characters: {transcription[:200]}")
        
        return True
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return False

if __name__ == "__main__":
    success = test_single_transcription()
    if success:
        print("✅ Transcription test passed!")
    else:
        print("❌ Transcription test failed!")
    sys.exit(0 if success else 1) 