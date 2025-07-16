#!/usr/bin/env python3
"""
Simple test to see exactly what the API returns without any processing.
"""

import os
import sys
import pytz
from datetime import datetime, date, time
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.api.init_alpaca_client import init_alpaca_client

def simple_api_test():
    """Test the API directly without complex processing."""
    
    # Initialize API client
    api = init_alpaca_client()
    
    # Set parameters
    symbol = "OPEN"
    target_date = date(2025, 7, 16)
    
    # Set up time range
    et_tz = pytz.timezone('America/New_York')
    start_time = datetime.combine(target_date, time(9, 30), tzinfo=et_tz)
    end_time = datetime.combine(target_date, time(16, 0), tzinfo=et_tz)
    
    print(f"=== Simple API Test ===")
    print(f"Symbol: {symbol}")
    print(f"Date: {target_date}")
    print(f"Start: {start_time}")
    print(f"End: {end_time}")
    print(f"Start ISO: {start_time.isoformat()}")
    print(f"End ISO: {end_time.isoformat()}")
    
    # Test different data feeds
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Test both feeds
    feeds = ['iex', 'sip']
    
    for feed in feeds:
        print(f"\n=== Testing {feed.upper()} feed ===")
        
        try:
            # Get data
            bars = api.get_bars(
                symbol,
                tradeapi.TimeFrame.Minute,
                start=start_time.isoformat(),
                end=end_time.isoformat(),
                limit=10000,
                feed=feed
            )
        
        print(f"\nAPI returned {len(bars)} bars")
        
        if len(bars) > 0:
            print(f"\nFirst 10 bars:")
            for i, bar in enumerate(bars[:10]):
                print(f"  {i+1:2d}. {bar.t} | O:{bar.o:7.2f} H:{bar.h:7.2f} L:{bar.l:7.2f} C:{bar.c:7.2f} V:{bar.v:8.0f}")
            
            print(f"\nLast 5 bars:")
            for i, bar in enumerate(bars[-5:]):
                print(f"  {i+1:2d}. {bar.t} | O:{bar.o:7.2f} H:{bar.h:7.2f} L:{bar.l:7.2f} C:{bar.c:7.2f} V:{bar.v:8.0f}")
            
            # Check timing
            first_bar = bars[0]
            last_bar = bars[-1]
            
            print(f"\nTiming Analysis:")
            print(f"Expected start: {start_time}")
            print(f"Actual start:   {first_bar.t}")
            print(f"Expected end:   {end_time}")
            print(f"Actual end:     {last_bar.t}")
            
            # Calculate delays
            start_delay = (first_bar.t - start_time).total_seconds() / 60
            end_gap = (end_time - last_bar.t).total_seconds() / 60
            
            print(f"\nStart delay: {start_delay:.1f} minutes")
            print(f"End gap: {end_gap:.1f} minutes")
            
            # Show opening range specific
            opening_end = et_tz.localize(datetime.combine(target_date, time(9, 45)))
            opening_bars = [bar for bar in bars if bar.t <= opening_end]
            print(f"\nOpening range bars (≤ 9:45 AM): {len(opening_bars)}")
            
            if len(opening_bars) > 0:
                print(f"Opening range bars:")
                for i, bar in enumerate(opening_bars[:5]):
                    print(f"  {i+1:2d}. {bar.t} | O:{bar.o:7.2f} H:{bar.h:7.2f} L:{bar.l:7.2f} C:{bar.c:7.2f} V:{bar.v:8.0f}")
            else:
                print("No opening range bars found!")
                
        else:
            print("No bars returned!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_api_test()