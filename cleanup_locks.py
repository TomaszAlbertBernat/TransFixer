#!/usr/bin/env python3
"""
Lock File Cleanup Script for TransFixer
Removes orphaned .lock and .correction.lock files to allow transcription to continue
"""

import os
import sys
from pathlib import Path

def find_and_remove_lock_files():
    """Find and remove all lock files in the TransFixer directory structure."""
    
    lock_files_found = []
    lock_files_removed = []
    errors = []
    
    # Define directories to scan
    directories_to_scan = [
        "audio",
        "transcriptions", 
        "corrected",
        "."  # Current directory
    ]
    
    print("🔍 Scanning for lock files...")
    print("=" * 50)
    
    for directory in directories_to_scan:
        if not os.path.exists(directory):
            print(f"⚠️  Directory {directory} does not exist, skipping...")
            continue
            
        print(f"📁 Scanning {directory}/...")
        
        # Scan recursively for lock files
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.lock') or file.endswith('.correction.lock'):
                    lock_file_path = os.path.join(root, file)
                    lock_files_found.append(lock_file_path)
                    
                    # Try to remove the lock file
                    try:
                        os.remove(lock_file_path)
                        lock_files_removed.append(lock_file_path)
                        print(f"✅ Removed: {lock_file_path}")
                    except Exception as e:
                        errors.append(f"❌ Failed to remove {lock_file_path}: {e}")
                        print(f"❌ Failed to remove: {lock_file_path} - {e}")
    
    print("\n" + "=" * 50)
    print("🧹 CLEANUP SUMMARY:")
    print("=" * 50)
    
    print(f"📊 Lock files found: {len(lock_files_found)}")
    print(f"✅ Lock files removed: {len(lock_files_removed)}")
    print(f"❌ Errors: {len(errors)}")
    
    if lock_files_found:
        print(f"\n📋 Files that were found:")
        for lock_file in lock_files_found:
            status = "✅ REMOVED" if lock_file in lock_files_removed else "❌ FAILED"
            print(f"   {status}: {lock_file}")
    else:
        print(f"\n🎉 No lock files found! All clear.")
    
    if errors:
        print(f"\n⚠️  Errors encountered:")
        for error in errors:
            print(f"   {error}")
    
    if lock_files_removed:
        print(f"\n🚀 SUCCESS! Removed {len(lock_files_removed)} lock files.")
        print(f"📝 The following files can now be processed:")
        
        # Try to identify which audio files can now be processed
        for lock_file in lock_files_removed:
            if lock_file.endswith('.lock'):
                # Remove .lock to get original file
                original_file = lock_file[:-5]  # Remove '.lock'
                if os.path.exists(original_file):
                    print(f"   🎵 {original_file}")
            elif lock_file.endswith('.correction.lock'):
                # Remove .correction.lock to get original file
                original_file = lock_file[:-16]  # Remove '.correction.lock'
                if os.path.exists(original_file):
                    print(f"   📝 {original_file}")
        
        print(f"\n🎯 You can now run TransFixer to process these files!")
        print(f"   python3 transfixer.py")
    
    return len(lock_files_removed), len(errors)

if __name__ == "__main__":
    print("🧹 TransFixer Lock File Cleanup Tool")
    print("=" * 50)
    print("This script will remove all .lock and .correction.lock files")
    print("to allow interrupted transcriptions to continue.")
    print()
    
    try:
        removed_count, error_count = find_and_remove_lock_files()
        
        if removed_count > 0:
            print(f"\n✅ SUCCESS: Cleaned up {removed_count} lock files!")
            sys.exit(0)
        elif error_count > 0:
            print(f"\n⚠️  Some errors occurred during cleanup.")
            sys.exit(1)
        else:
            print(f"\n🎉 No cleanup needed - no lock files found!")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        sys.exit(1) 