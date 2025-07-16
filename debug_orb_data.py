#!/usr/bin/env python3
"""
Debug script to trace exactly what happens to the data in orb.py when processing OPEN symbol.
"""

import os
import sys
import pandas as pd
import pytz
from datetime import datetime, date, time
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.api.init_alpaca_client import init_alpaca_client
from atoms.display.plot_candle_chart import plot_candle_chart

def debug_orb_data_processing():
    """Debug the exact data processing steps for multiple symbols."""
    
    # Initialize API client
    try:
        api = init_alpaca_client()
        print(f"✓ API client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize API client: {e}")
        return False
    
    # Test multiple symbols to see if this is symbol-specific
    symbols = ["AAPL", "OPEN", "IXHL", "TSLA"]
    target_date = date(2025, 7, 16)
    
    # Set up time range (exactly as orb.py does)
    et_tz = pytz.timezone('America/New_York')
    start_time = datetime.combine(target_date, time(9, 30), tzinfo=et_tz)
    end_time = datetime.combine(target_date, time(16, 0), tzinfo=et_tz)
    
    print(f"\n=== Debugging ORB Data Processing for Multiple Symbols ===")
    print(f"Target date: {target_date}")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    
    # Determine feed (exactly as orb.py does)
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    feed = 'iex' if "paper" in base_url else 'sip'
    print(f"Using {feed.upper()} data feed")
    
    results = {}
    
    for symbol in symbols:
        print(f"\n--- Testing {symbol} ---")
        try:
            # Fetch data exactly as orb.py does
            print(f"Fetching data from API...")
            bars = api.get_bars(
                symbol,
                tradeapi.TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                limit=10000,
                feed=feed
            )
        
            if not bars:
                print(f"No data returned for {symbol}")
                results[symbol] = {'status': 'no_data'}
                continue
            
            print(f"Raw bars returned: {len(bars)} bars")
            
            if len(bars) > 0:
                first_bar = bars[0]
                last_bar = bars[-1]
                print(f"First bar: {first_bar.t} (O: {first_bar.o}, H: {first_bar.h}, L: {first_bar.l}, C: {first_bar.c})")
                print(f"Last bar: {last_bar.t} (O: {last_bar.o}, H: {last_bar.h}, L: {last_bar.l}, C: {last_bar.c})")
                
                # Check timing
                expected_start = start_time.replace(tzinfo=pytz.timezone('America/New_York'))
                expected_end = end_time.replace(tzinfo=pytz.timezone('America/New_York'))
                
                actual_start = first_bar.t
                actual_end = last_bar.t
                
                if actual_start > expected_start:
                    minutes_late = (actual_start - expected_start).total_seconds() / 60
                    print(f"⚠️  Data starts {minutes_late:.1f} minutes late!")
                else:
                    print(f"✓ Data starts on time")
                
                if actual_end < expected_end:
                    minutes_early = (expected_end - actual_end).total_seconds() / 60
                    print(f"⚠️  Data ends {minutes_early:.1f} minutes early!")
                else:
                    print(f"✓ Data ends on time")
                
                # Calculate coverage
                session_duration = (expected_end - expected_start).total_seconds() / 60
                actual_duration = (actual_end - actual_start).total_seconds() / 60
                coverage = (actual_duration / session_duration) * 100 if session_duration > 0 else 0
                print(f"Session coverage: {coverage:.1f}% ({len(bars)} bars)")
                
                # Check for opening range data
                opening_range_end = expected_start.replace(hour=9, minute=45)
                opening_bars = [bar for bar in bars if bar.t <= opening_range_end]
                print(f"Opening range bars (9:30-9:45): {len(opening_bars)}")
                
                results[symbol] = {
                    'status': 'success',
                    'total_bars': len(bars),
                    'first_bar_time': actual_start,
                    'last_bar_time': actual_end,
                    'coverage': coverage,
                    'opening_range_bars': len(opening_bars),
                    'minutes_late': minutes_late if actual_start > expected_start else 0
                }
            else:
                results[symbol] = {'status': 'empty'}
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            results[symbol] = {'status': 'error', 'error': str(e)}
            continue
    
    # Summary
    print(f"\n=== SUMMARY ===")
    for symbol, result in results.items():
        if result['status'] == 'success':
            status = "✓" if result['coverage'] > 95 and result['opening_range_bars'] > 0 else "⚠️"
            print(f"{status} {symbol}: {result['coverage']:.1f}% coverage, {result['opening_range_bars']} opening range bars")
            if result['minutes_late'] > 1:
                print(f"    ⚠️  Data starts {result['minutes_late']:.1f} minutes late")
        else:
            print(f"✗ {symbol}: {result['status']}")
    
    return True

if __name__ == "__main__":
    debug_orb_data_processing()