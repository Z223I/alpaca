#!/usr/bin/env python3
"""
Debug script to verify config.py directory settings
"""

import sys
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')

from atoms.alerts.config import (
    DEFAULT_PLOTS_ROOT_DIR, DEFAULT_DATA_ROOT_DIR, DEFAULT_LOGS_ROOT_DIR,
    DEFAULT_HISTORICAL_ROOT_DIR, DEFAULT_PRICE_MOMENTUM_CONFIG
)

print("🔍 CONFIG.PY VERIFICATION:")
print("=" * 50)
print(f"DEFAULT_PLOTS_ROOT_DIR: {DEFAULT_PLOTS_ROOT_DIR}")
print(f"DEFAULT_DATA_ROOT_DIR: {DEFAULT_DATA_ROOT_DIR}")
print(f"DEFAULT_LOGS_ROOT_DIR: {DEFAULT_LOGS_ROOT_DIR}")
print(f"DEFAULT_HISTORICAL_ROOT_DIR: {DEFAULT_HISTORICAL_ROOT_DIR}")
print(f"DEFAULT_PRICE_MOMENTUM_CONFIG: {DEFAULT_PRICE_MOMENTUM_CONFIG}")
print()

# Test specific paths
try:
    historical_root = DEFAULT_HISTORICAL_ROOT_DIR
    superduper_dir = historical_root.get_superduper_alerts_sent_dir("2025-08-04")
    print(f"Superduper alerts sent dir: {superduper_dir}")
    print(f"Superduper dir exists: {superduper_dir.exists()}")
    
    # List contents if exists
    if superduper_dir.exists():
        files = list(superduper_dir.glob("*.json"))
        print(f"Found {len(files)} files in superduper directory")
    
except Exception as e:
    print(f"Error accessing superduper directory: {e}")

print()
print("🔍 CURRENT WORKING DIRECTORY:")
import os
print(f"CWD: {os.getcwd()}")