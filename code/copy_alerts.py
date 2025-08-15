#!/usr/bin/env python3
"""
Alert Copier for Backtesting

Copies existing VERB alert files from historical_data to runs directory
at a rate of 2 per second to simulate real-time alert generation.
"""

import sys
import shutil
import time
import glob
import os
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 code/copy_alerts.py <date> <run_dir>")
        sys.exit(1)
    
    date = sys.argv[1]
    run_dir = sys.argv[2]
    
    src_dir = f"historical_data/{date}/alerts/bullish/"
    dst_dir = f"{run_dir}/historical_data/{date}/alerts/bullish/"
    
    # Create destination directory
    os.makedirs(dst_dir, exist_ok=True)
    
    # Find all VERB alert files
    files = sorted(glob.glob(src_dir + "alert_VERB_*.json"))
    print(f"Found {len(files)} VERB alert files to copy")
    
    if not files:
        print("No VERB alert files found to copy")
        return
    
    # Copy files at 2 per second (0.5 second delay)
    for i, file_path in enumerate(files):
        shutil.copy2(file_path, dst_dir)
        filename = os.path.basename(file_path)
        print(f"Copied {filename} ({i+1}/{len(files)})")
        
        # Sleep for 0.5 seconds (2 files per second)
        if i < len(files) - 1:  # Don't sleep after the last file
            time.sleep(0.5)
    
    print(f"Completed copying {len(files)} VERB alert files")

if __name__ == "__main__":
    main()