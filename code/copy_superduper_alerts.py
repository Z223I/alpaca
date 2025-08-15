#!/usr/bin/env python3
"""
Superduper Alert Copier for Backtesting

Copies existing VERB superduper alert files from historical_data to runs directory
at a rate of 1 per 3 seconds to test trade processor.
"""

import sys
import shutil
import time
import glob
import os
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 code/copy_superduper_alerts.py <date> <run_dir>")
        sys.exit(1)

    date = sys.argv[1]
    run_dir = sys.argv[2]

    src_dir = f"historical_data/{date}/superduper_alerts/bullish/"
    dst_dir = f"{run_dir}/historical_data/{date}/superduper_alerts/bullish/green/"

    # Create destination directory
    os.makedirs(dst_dir, exist_ok=True)

    # Find all VERB superduper alert files
    files = sorted(glob.glob(src_dir + "*VERB*.json"))
    print(f"Found {len(files)} VERB superduper alert files to copy")

    if not files:
        print("No VERB superduper alert files found to copy")
        return

    # Copy files at 1 per 3 seconds
    for i, file_path in enumerate(files):
        shutil.copy2(file_path, dst_dir)
        filename = os.path.basename(file_path)
        print(f"Copied {filename} ({i+1}/{len(files)})")

        # Sleep for 3 seconds (1 file per 3 seconds)
        if i < len(files) - 1:  # Don't sleep after the last file
            time.sleep(3.0)

    print(f"Completed copying {len(files)} VERB superduper alert files")

if __name__ == "__main__":
    main()