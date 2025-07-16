#!/usr/bin/env python3
"""
Test the fixed ORB data fetching to verify we get opening range data.
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

def test_fixed_orb_fetching():
    """Test the fixed ORB data fetching approach."""
    
    # Initialize API client
    api = init_alpaca_client()
    
    # Test symbol
    symbol = "OPEN"
    target_date = date(2025, 7, 16)
    
    # Use the fixed approach - start from premarket
    et_tz = pytz.timezone('America/New_York')
    start_time = datetime.combine(target_date, time(4, 0), tzinfo=et_tz)  # Start from premarket
    end_time = datetime.combine(target_date, time(16, 0), tzinfo=et_tz)
    
    print(f"=== Testing Fixed ORB Data Fetching for {symbol} ===")
    print(f"Date: {target_date}")
    print(f"Fetching from {start_time} to {end_time}")
    print(f"  (Starting from premarket to ensure opening range data is included)")
    
    # Determine feed
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    feed = 'iex' if "paper" in base_url else 'sip'
    print(f"Using {feed.upper()} data feed")
    
    try:
        # Fetch data exactly as the fixed orb.py does
        print(f"\n--- Fetching market data ---")
        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Minute,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            limit=10000,
            feed=feed
        )
        
        if not bars:
            print(f"No market data available for {symbol}")
            return False
        
        print(f"Retrieved {len(bars)} bars")
        
        # Convert to DataFrame format exactly as orb.py does
        symbol_data = []
        for bar in bars:
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
        
        symbol_df = pd.DataFrame(symbol_data)
        symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
        
        print(f"DataFrame created with {len(symbol_df)} rows")
        
        if len(symbol_df) > 0:
            min_time = symbol_df['timestamp'].min()
            max_time = symbol_df['timestamp'].max()
            print(f"Data range: {min_time} to {max_time}")
            
            # Check for opening range data specifically
            opening_start = et_tz.localize(datetime.combine(target_date, time(9, 30)))
            opening_end = et_tz.localize(datetime.combine(target_date, time(9, 45)))
            
            # Filter for opening range
            opening_mask = (symbol_df['timestamp'] >= opening_start) & (symbol_df['timestamp'] <= opening_end)
            opening_data = symbol_df[opening_mask]
            
            print(f"\n--- Opening Range Analysis ---")
            print(f"Opening range data points (9:30-9:45 AM): {len(opening_data)}")
            
            if len(opening_data) > 0:
                print(f"✓ SUCCESS: Got complete opening range data!")
                print(f"Opening range: {opening_data['timestamp'].min()} to {opening_data['timestamp'].max()}")
                
                # Show opening range data
                print(f"\nOpening range bars:")
                for i, (idx, row) in enumerate(opening_data.iterrows()):
                    print(f"  {i+1:2d}. {row['timestamp']} | O:{row['open']:7.2f} H:{row['high']:7.2f} L:{row['low']:7.2f} C:{row['close']:7.2f}")
                
                # Calculate ORB levels
                orb_high = opening_data['high'].max()
                orb_low = opening_data['low'].min()
                print(f"\nORB Levels:")
                print(f"  ORB High: ${orb_high:.2f}")
                print(f"  ORB Low: ${orb_low:.2f}")
                print(f"  ORB Range: ${orb_high - orb_low:.2f}")
                
            else:
                print(f"✗ FAILED: No opening range data found")
                return False
            
            # Test chart generation
            print(f"\n--- Testing Chart Generation ---")
            test_plots_dir = 'test_plots'
            os.makedirs(test_plots_dir, exist_ok=True)
            
            success = plot_candle_chart(symbol_df, symbol, test_plots_dir, alerts=[])
            
            if success:
                print(f"✓ Chart generated successfully")
                
                # Check what files were created
                import glob
                chart_files = glob.glob(f"{test_plots_dir}/**/*{symbol}*", recursive=True)
                for chart_file in chart_files:
                    print(f"  Created: {chart_file}")
                    
                return True
            else:
                print(f"✗ Chart generation failed")
                return False
        else:
            print(f"No data in DataFrame")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_orb_fetching()
    if success:
        print(f"\n🎉 FIX SUCCESSFUL! Opening range data is now available.")
    else:
        print(f"\n❌ Fix failed.")