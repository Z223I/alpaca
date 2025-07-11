"""
Atom for extracting timestamps from alert JSON files.
"""

import os
import json
import glob
from datetime import datetime
from typing import List, Optional


def extract_alert_timestamps(date: str, symbol: str, signal: str = 'bullish') -> List[datetime]:
    """
    Extract timestamps from alert JSON files for a specific date, symbol, and signal type.
    
    Args:
        date: Date in YYYY-MM-DD format (e.g., '2025-07-10')
        symbol: Stock symbol (e.g., 'PROK')
        signal: Signal type - 'bullish', 'bearish', or 'both' (default: 'bullish')
        
    Returns:
        List of datetime objects extracted from alert timestamps
    """
    timestamps = []
    
    try:
        # Base directory for historical data
        base_dir = 'historical_data'
        date_dir = os.path.join(base_dir, date, 'alerts')
        
        if not os.path.exists(date_dir):
            return timestamps
        
        # Determine which signal directories to search
        signal_dirs = []
        if signal == 'both':
            signal_dirs = ['bullish', 'bearish']
        elif signal in ['bullish', 'bearish']:
            signal_dirs = [signal]
        else:
            return timestamps
        
        # Process each signal directory
        for signal_type in signal_dirs:
            signal_dir = os.path.join(date_dir, signal_type)
            
            if not os.path.exists(signal_dir):
                continue
            
            # Find all alert files for the symbol
            pattern = os.path.join(signal_dir, f'alert_{symbol}_*.json')
            alert_files = glob.glob(pattern)
            
            # Extract timestamps from each file
            for alert_file in alert_files:
                try:
                    with open(alert_file, 'r') as f:
                        alert_data = json.load(f)
                    
                    # Extract timestamp from the JSON data
                    if 'timestamp' in alert_data:
                        timestamp_str = alert_data['timestamp']
                        # Parse the timestamp (format: "2025-07-10T11:28:00")
                        timestamp = datetime.fromisoformat(timestamp_str)
                        timestamps.append(timestamp)
                        
                except Exception as e:
                    print(f"Warning: Could not parse alert file {alert_file}: {e}")
                    continue
        
        # Sort timestamps chronologically
        timestamps.sort()
        
    except Exception as e:
        print(f"Error extracting alert timestamps: {e}")
        return []
    
    return timestamps