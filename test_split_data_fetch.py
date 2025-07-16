#!/usr/bin/env python3
"""
Test script to fetch data in two halves and combine them.
This may help get around IEX data feed limitations.
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

def test_split_data_fetch():
    """Test fetching data in two halves and combining them."""
    
    # Initialize API client
    api = init_alpaca_client()
    
    # Set parameters
    symbol = "OPEN"
    target_date = date(2025, 7, 16)
    
    # Set up time ranges
    et_tz = pytz.timezone('America/New_York')
    
    # First half: 9:30 AM to 12:30 PM (3 hours)
    first_half_start = datetime.combine(target_date, time(9, 30), tzinfo=et_tz)
    first_half_end = datetime.combine(target_date, time(12, 30), tzinfo=et_tz)
    
    # Second half: 12:30 PM to 4:00 PM (3.5 hours)
    second_half_start = datetime.combine(target_date, time(12, 30), tzinfo=et_tz)
    second_half_end = datetime.combine(target_date, time(16, 0), tzinfo=et_tz)
    
    print(f"=== Split Data Fetch Test for {symbol} ===")
    print(f"First half: {first_half_start} to {first_half_end}")
    print(f"Second half: {second_half_start} to {second_half_end}")
    
    # Determine feed
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    feed = 'iex' if "paper" in base_url else 'sip'
    print(f"Using {feed.upper()} data feed")
    
    all_bars = []
    
    try:
        # Fetch first half
        print(f"\n--- Fetching first half (9:30 AM - 12:30 PM) ---")
        first_half_bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Minute,
            start=first_half_start.isoformat(),
            end=first_half_end.isoformat(),
            limit=10000,
            feed=feed
        )
        
        print(f"First half returned: {len(first_half_bars)} bars")
        
        if len(first_half_bars) > 0:
            print(f"First half range: {first_half_bars[0].t} to {first_half_bars[-1].t}")
            
            # Check if we got opening range data
            opening_end = et_tz.localize(datetime.combine(target_date, time(9, 45)))
            opening_bars = [bar for bar in first_half_bars if bar.t <= opening_end]
            print(f"Opening range bars (9:30-9:45): {len(opening_bars)}")
            
            if len(opening_bars) > 0:
                print(f"✓ Got opening range data!")
                for i, bar in enumerate(opening_bars[:5]):
                    print(f"  {i+1}. {bar.t} | O:{bar.o:7.2f} H:{bar.h:7.2f} L:{bar.l:7.2f} C:{bar.c:7.2f}")
            else:
                print(f"✗ No opening range data in first half")
        
        all_bars.extend(first_half_bars)
        
        # Fetch second half
        print(f"\n--- Fetching second half (12:30 PM - 4:00 PM) ---")
        second_half_bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Minute,
            start=second_half_start.isoformat(),
            end=second_half_end.isoformat(),
            limit=10000,
            feed=feed
        )
        
        print(f"Second half returned: {len(second_half_bars)} bars")
        
        if len(second_half_bars) > 0:
            print(f"Second half range: {second_half_bars[0].t} to {second_half_bars[-1].t}")
        
        # Remove duplicate bar at 12:30 PM if present
        second_half_filtered = []
        for bar in second_half_bars:
            if bar.t > first_half_end:  # Only include bars after 12:30 PM
                second_half_filtered.append(bar)
        
        print(f"Second half after removing duplicates: {len(second_half_filtered)} bars")
        all_bars.extend(second_half_filtered)
        
        # Combine and analyze
        print(f"\n--- Combined Results ---")
        print(f"Total bars: {len(all_bars)}")
        
        if len(all_bars) > 0:
            # Sort by timestamp to ensure proper order
            all_bars.sort(key=lambda x: x.t)
            
            print(f"Combined range: {all_bars[0].t} to {all_bars[-1].t}")
            
            # Check for gaps
            print(f"\n--- Checking for gaps ---")
            for i in range(1, len(all_bars)):
                prev_time = all_bars[i-1].t
                curr_time = all_bars[i].t
                gap_minutes = (curr_time - prev_time).total_seconds() / 60
                
                if gap_minutes > 1.5:  # More than 1.5 minutes gap
                    print(f"Gap detected: {prev_time} to {curr_time} ({gap_minutes:.1f} minutes)")
            
            # Check opening range coverage
            opening_end = et_tz.localize(datetime.combine(target_date, time(9, 45)))
            opening_bars = [bar for bar in all_bars if bar.t <= opening_end]
            print(f"\nOpening range bars (9:30-9:45): {len(opening_bars)}")
            
            # Check full session coverage
            expected_start = first_half_start
            expected_end = second_half_end
            actual_start = all_bars[0].t
            actual_end = all_bars[-1].t
            
            start_delay = (actual_start - expected_start).total_seconds() / 60
            end_gap = (expected_end - actual_end).total_seconds() / 60
            
            print(f"\nSession coverage analysis:")
            print(f"Expected: {expected_start} to {expected_end}")
            print(f"Actual: {actual_start} to {actual_end}")
            print(f"Start delay: {start_delay:.1f} minutes")
            print(f"End gap: {end_gap:.1f} minutes")
            
            # Test conversion to DataFrame
            print(f"\n--- Testing DataFrame conversion ---")
            symbol_data = []
            for bar in all_bars:
                bar_data = {
                    'timestamp': bar.t.isoformat(),
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': int(bar.v),
                    'symbol': symbol
                }
                symbol_data.append(bar_data)
            
            df = pd.DataFrame(symbol_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            print(f"DataFrame created with {len(df)} rows")
            print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Show first few rows
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            return df
        else:
            print("No data returned from either half")
            return None
            
    except Exception as e:
        print(f"Error in split data fetch: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_split_data_fetch()