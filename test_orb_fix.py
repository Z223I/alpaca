#!/usr/bin/env python3
"""
Test script to verify ORB calculation fix with BSLK data
"""

import sys
import pandas as pd
from datetime import datetime, time
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.indicators.orb_calculator import ORBCalculator
from atoms.alerts.breakout_detector import BreakoutDetector

def test_orb_calculation():
    """Test ORB calculation with BSLK data."""
    
    # Load BSLK data
    csv_file = "historical_data/2025-07-08/market_data/BSLK_20250708_090100.csv"
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("🔍 Testing ORB Calculation Fix")
    print("=" * 60)
    print(f"Data: {len(df)} points from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    # Initialize ORB calculator
    orb_calc = ORBCalculator()
    
    # Calculate ORB levels for BSLK
    print("\n📊 Calculating ORB levels...")
    orb_level = orb_calc.calculate_orb_levels('BSLK', df)
    
    if orb_level:
        print(f"✅ ORB calculation successful!")
        print(f"   ORB High: ${orb_level.orb_high:.3f}")
        print(f"   ORB Low: ${orb_level.orb_low:.3f}")
        print(f"   ORB Range: ${orb_level.orb_range:.3f}")
        print(f"   Sample Count: {orb_level.sample_count}")
        
        # Test breakout detection
        print("\n🎯 Testing breakout detection...")
        breakout_detector = BreakoutDetector()
        
        # Test with a price that should trigger breakout
        test_price = 4.0  # Should be above ORB high of ~3.88
        test_timestamp = datetime(2025, 7, 8, 14, 0, 0, tzinfo=pytz.UTC)  # 10:00 ET
        
        print(f"   Testing price: ${test_price:.3f}")
        print(f"   Testing time: {test_timestamp} UTC")
        
        # Check if within alert window
        in_window = breakout_detector.is_within_alert_window(test_timestamp)
        print(f"   Within alert window: {in_window}")
        
        # Check breakout
        breakout = breakout_detector.detect_breakout(
            symbol='BSLK',
            current_price=test_price,
            volume_ratio=2.0,  # 2x average volume
            timestamp=test_timestamp
        )
        
        if breakout:
            print(f"   ✅ Breakout detected: {breakout.direction} at ${breakout.price:.3f}")
            print(f"   Breakout percentage: {breakout.breakout_percentage:.2f}%")
        else:
            print(f"   ❌ No breakout detected")
            
    else:
        print("❌ ORB calculation failed - no levels calculated")
        
        # Debug: Check filtered data
        print("\n🔍 Debugging ORB filtering...")
        price_data = df[['timestamp', 'high', 'low', 'close', 'volume']].copy()
        filtered_data = orb_calc._filter_opening_range(price_data)
        print(f"   Original data points: {len(price_data)}")
        print(f"   Filtered data points: {len(filtered_data)}")
        
        if len(filtered_data) > 0:
            print(f"   Filtered time range: {filtered_data['timestamp_et'].iloc[0]} to {filtered_data['timestamp_et'].iloc[-1]}")
        else:
            print("   No data found in opening range period")
            
            # Show timezone conversion for first few points
            print("\n   Sample timezone conversions:")
            for i in range(min(5, len(price_data))):
                utc_time = price_data['timestamp'].iloc[i]
                et_time = utc_time.tz_localize('UTC').tz_convert('US/Eastern')
                print(f"   UTC: {utc_time} -> ET: {et_time}")

if __name__ == "__main__":
    test_orb_calculation()