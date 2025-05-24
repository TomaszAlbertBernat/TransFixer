#!/usr/bin/env python3

import time
import psutil
import subprocess
import os
from datetime import datetime
import json

def get_transfixer_process():
    """Find the running TransFixer process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'transfixer.py' in ' '.join(proc.info['cmdline']):
                return psutil.Process(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def get_gpu_memory():
    """Get GPU memory usage using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu'], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:  # Header + data
            data = lines[1].split(', ')
            return {
                'memory_used': int(data[0].split()[0]),  # Remove 'MiB'
                'memory_total': int(data[1].split()[0]), 
                'gpu_util': int(data[2].split()[0]),     # Remove '%'
                'temperature': int(data[3].split()[0])   # Remove 'C'
            }
    except (subprocess.CalledProcessError, IndexError, ValueError):
        pass
    return None

def count_transcriptions():
    """Count completed transcription files."""
    try:
        result = subprocess.run(['find', 'transcriptions', '-name', '*.txt'], 
                              capture_output=True, text=True, check=True)
        return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except subprocess.CalledProcessError:
        return 0

def monitor_transfixer():
    """Monitor TransFixer performance."""
    print("🔍 TransFixer Performance Monitor")
    print("=" * 50)
    
    start_time = datetime.now()
    last_transcription_count = 0
    performance_data = []
    
    while True:
        current_time = datetime.now()
        elapsed = (current_time - start_time).total_seconds()
        
        # Get TransFixer process info
        proc = get_transfixer_process()
        if not proc:
            print(f"⚠️  TransFixer process not found at {current_time.strftime('%H:%M:%S')}")
            time.sleep(10)
            continue
        
        # Get system metrics
        try:
            cpu_percent = proc.cpu_percent()
            memory_info = proc.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
        except psutil.NoSuchProcess:
            print(f"⚠️  Process ended at {current_time.strftime('%H:%M:%S')}")
            break
        
        # Get GPU metrics
        gpu_info = get_gpu_memory()
        
        # Get transcription progress
        transcription_count = count_transcriptions()
        new_transcriptions = transcription_count - last_transcription_count
        
        # Calculate rates
        files_per_hour = (transcription_count / elapsed * 3600) if elapsed > 0 else 0
        
        # Store performance data
        data_point = {
            'timestamp': current_time.isoformat(),
            'elapsed_seconds': elapsed,
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'transcriptions_completed': transcription_count,
            'files_per_hour': files_per_hour,
            'gpu_info': gpu_info
        }
        performance_data.append(data_point)
        
        # Display current status
        print(f"\n⏰ {current_time.strftime('%H:%M:%S')} | Runtime: {elapsed/60:.1f}m")
        print(f"📊 CPU: {cpu_percent:.1f}% | RAM: {memory_mb:.0f}MB")
        
        if gpu_info:
            gpu_usage_percent = (gpu_info['memory_used'] / gpu_info['memory_total']) * 100
            print(f"🎮 GPU: {gpu_info['gpu_util']}% | VRAM: {gpu_info['memory_used']}MB/{gpu_info['memory_total']}MB ({gpu_usage_percent:.1f}%) | Temp: {gpu_info['temperature']}°C")
        
        print(f"📝 Transcriptions: {transcription_count} total ({files_per_hour:.1f}/hour)")
        
        if new_transcriptions > 0:
            print(f"✅ {new_transcriptions} new transcription(s) completed!")
        
        # Save performance data every 5 minutes
        if len(performance_data) % 10 == 0:  # Every 10 cycles (5 minutes)
            with open('performance_data.json', 'w') as f:
                json.dump(performance_data, f, indent=2)
            print(f"💾 Performance data saved ({len(performance_data)} data points)")
        
        last_transcription_count = transcription_count
        time.sleep(30)  # Update every 30 seconds

if __name__ == "__main__":
    try:
        monitor_transfixer()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        
        # Save final performance data
        if 'performance_data' in locals():
            with open('performance_data.json', 'w') as f:
                json.dump(performance_data, f, indent=2)
            print(f"💾 Final performance data saved ({len(performance_data)} data points)") 