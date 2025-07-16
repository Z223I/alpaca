#!/usr/bin/env python3
"""
Test script to fetch data starting from premarket hours.
This may help get the opening range data.
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

def test_premarket_fetch():
    """Test fetching data starting from premarket hours."""
    
    # Initialize API client
    api = init_alpaca_client()
    
    # Set parameters
    symbol = "OPEN"
    target_date = date(2025, 7, 16)
    
    # Set up time ranges - start from premarket
    et_tz = pytz.timezone('America/New_York')
    
    # Start from 4:00 AM (premarket start) to 4:00 PM
    start_time = datetime.combine(target_date, time(4, 0), tzinfo=et_tz)
    end_time = datetime.combine(target_date, time(16, 0), tzinfo=et_tz)
    
    print(f"=== Premarket Data Fetch Test for {symbol} ===")
    print(f"Request range: {start_time} to {end_time}")
    
    # Determine feed
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    feed = 'iex' if "paper" in base_url else 'sip'
    print(f"Using {feed.upper()} data feed")
    
    try:
        # Fetch data from premarket
        print(f"\n--- Fetching data from 4:00 AM ---")
        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Minute,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            limit=10000,
            feed=feed
        )
        
        print(f"Returned: {len(bars)} bars")
        
        if len(bars) > 0:
            print(f"Data range: {bars[0].t} to {bars[-1].t}")
            
            # Check for opening range data
            opening_start = et_tz.localize(datetime.combine(target_date, time(9, 30)))
            opening_end = et_tz.localize(datetime.combine(target_date, time(9, 45)))
            
            opening_bars = [bar for bar in bars if opening_start <= bar.t <= opening_end]
            print(f"Opening range bars (9:30-9:45): {len(opening_bars)}")
            
            if len(opening_bars) > 0:
                print(f"✓ Got opening range data!")
                for i, bar in enumerate(opening_bars):
                    print(f"  {i+1}. {bar.t} | O:{bar.o:7.2f} H:{bar.h:7.2f} L:{bar.l:7.2f} C:{bar.c:7.2f}")
            else:
                print(f"✗ No opening range data")
            
            # Check for premarket data
            premarket_bars = [bar for bar in bars if bar.t < opening_start]
            print(f"Premarket bars (4:00-9:30): {len(premarket_bars)}")
            
            if len(premarket_bars) > 0:
                print(f"First premarket bar: {premarket_bars[0].t}")
                print(f"Last premarket bar: {premarket_bars[-1].t}")
            
            # Show first 10 bars
            print(f"\nFirst 10 bars:")
            for i, bar in enumerate(bars[:10]):
                print(f"  {i+1:2d}. {bar.t} | O:{bar.o:7.2f} H:{bar.h:7.2f} L:{bar.l:7.2f} C:{bar.c:7.2f}")
            
            # Check if we have any bars around 9:30 AM specifically
            market_open = et_tz.localize(datetime.combine(target_date, time(9, 30)))
            nearby_bars = [bar for bar in bars if abs((bar.t - market_open).total_seconds()) < 3600]  # Within 1 hour
            
            print(f"\nBars within 1 hour of 9:30 AM: {len(nearby_bars)}")
            for i, bar in enumerate(nearby_bars[:10]):
                print(f"  {i+1:2d}. {bar.t} | O:{bar.o:7.2f} H:{bar.h:7.2f} L:{bar.l:7.2f} C:{bar.c:7.2f}")
            
        else:
            print("No data returned")
            
    except Exception as e:
        print(f"Error in premarket fetch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_premarket_fetch()