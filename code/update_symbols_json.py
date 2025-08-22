#!/usr/bin/env python3
"""
Update Symbols JSON Script

This script scans historical_data directories for symbols that have superduper alerts
in the superduper_alerts_sent/bullish/green directories and creates/replaces
the data/backtesting/symbols.json file.

Scans starting from August 2025 dates.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


def find_superduper_alert_symbols() -> Dict[str, str]:
    """
    Scan historical_data directories for symbols with superduper alerts.
    
    Returns:
        Dictionary mapping symbol -> earliest date found
    """
    symbols_dates = {}
    historical_data_dir = Path("historical_data")
    
    if not historical_data_dir.exists():
        print(f"❌ Historical data directory not found: {historical_data_dir}")
        return symbols_dates
    
    # Pattern to match date directories (YYYY-MM-DD format)
    date_pattern = re.compile(r"2025-\d{2}-\d{2}")
    
    # Get all date directories starting from August 2025
    date_dirs = []
    for item in historical_data_dir.iterdir():
        if item.is_dir() and date_pattern.match(item.name):
            # Only include August 2025 and later
            if item.name >= "2025-08-01":
                date_dirs.append(item)
    
    # Sort by date
    date_dirs.sort(key=lambda x: x.name)
    
    print(f"📅 Scanning {len(date_dirs)} date directories from August 2025...")
    
    for date_dir in date_dirs:
        superduper_alerts_path = date_dir / "superduper_alerts_sent" / "bullish" / "green"
        
        if superduper_alerts_path.exists():
            print(f"🔍 Checking {date_dir.name}...")
            
            # Get all superduper alert files
            for alert_file in superduper_alerts_path.glob("superduper_alert_*.json"):
                # Extract symbol from filename
                # Pattern: superduper_alert_SYMBOL_YYYYMMDD_HHMMSS.json
                match = re.match(r"superduper_alert_([A-Z]+)_\d{8}_\d{6}\.json", alert_file.name)
                if match:
                    symbol = match.group(1)
                    
                    # Skip test symbols
                    if symbol in ['TEST', 'TSLA']:  # TSLA seems to be test data based on the directory listing
                        continue
                    
                    # Store the earliest date for each symbol
                    if symbol not in symbols_dates or date_dir.name < symbols_dates[symbol]:
                        symbols_dates[symbol] = date_dir.name
                        print(f"  ✅ Found {symbol} on {date_dir.name}")
    
    return symbols_dates


def create_symbols_json(symbols_dates: Dict[str, str]) -> None:
    """
    Create the symbols.json file with the found symbols.
    
    Args:
        symbols_dates: Dictionary mapping symbol -> date
    """
    # Create the data structure
    symbols_list = []
    for symbol, date in sorted(symbols_dates.items()):
        symbols_list.append({
            "symbol": symbol,
            "date": date,
            "active": "yes"
        })
    
    symbols_data = {
        "symbols": symbols_list
    }
    
    # Ensure the output directory exists
    output_dir = Path("data/backtesting")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the JSON file
    output_file = output_dir / "symbols.json"
    with open(output_file, 'w') as f:
        json.dump(symbols_data, f, indent=2)
    
    print(f"\n📄 Created {output_file} with {len(symbols_list)} symbols:")
    for symbol_data in symbols_list:
        print(f"  - {symbol_data['symbol']} (first seen: {symbol_data['date']})")


def main():
    """Main function to update symbols.json"""
    print("🚀 Updating symbols.json with superduper alert symbols...")
    print("=" * 60)
    
    # Find all symbols with superduper alerts
    symbols_dates = find_superduper_alert_symbols()
    
    if not symbols_dates:
        print("❌ No symbols found with superduper alerts")
        return
    
    print(f"\n✅ Found {len(symbols_dates)} unique symbols with superduper alerts")
    
    # Create the symbols.json file
    create_symbols_json(symbols_dates)
    
    print("\n🎉 symbols.json update completed!")


if __name__ == "__main__":
    main()