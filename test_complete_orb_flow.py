#!/usr/bin/env python3
"""
Test complete ORB alert flow with the fix for late start
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.config.alert_config import config
from atoms.websocket.alpaca_stream import MarketData
from atoms.websocket.data_buffer import DataBuffer
from atoms.indicators.orb_calculator import ORBCalculator
from atoms.alerts.breakout_detector import BreakoutDetector

def test_complete_orb_flow():
    """Test complete ORB flow with historical opening range data."""
    
    print("🔍 Testing Complete ORB Alert Flow")
    print("=" * 60)
    
    # Step 1: Load historical opening range data (from tmp file that was fetched)
    print("📊 Step 1: Loading historical opening range data...")
    orb_file = "tmp/ZVSA_opening_range_20250708.csv"
    
    if not os.path.exists(orb_file):
        print(f"❌ Opening range file not found: {orb_file}")
        return
    
    orb_df = pd.read_csv(orb_file)
    print(f"✅ Loaded {len(orb_df)} opening range data points")
    
    # Step 2: Load post-ORB data from the original file
    print("\n📊 Step 2: Loading post-ORB market data...")
    market_file = "historical_data/2025-07-08/market_data/ZVSA_20250708_100227.csv"
    
    if not os.path.exists(market_file):
        print(f"❌ Market data file not found: {market_file}")
        return
    
    market_df = pd.read_csv(market_file)
    print(f"✅ Loaded {len(market_df)} market data points")
    
    # Step 3: Combine the data (simulating what the system should do)
    print("\n📊 Step 3: Combining opening range and market data...")
    
    # Convert timestamps and normalize timezones
    orb_df['timestamp'] = pd.to_datetime(orb_df['timestamp'])
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
    
    # Normalize both to UTC timezone-naive
    if orb_df['timestamp'].dt.tz is not None:
        orb_df['timestamp'] = orb_df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    elif orb_df['timestamp'].dt.tz is None:
        orb_df['timestamp'] = orb_df['timestamp'].dt.tz_localize('UTC').dt.tz_localize(None)
        
    if market_df['timestamp'].dt.tz is not None:
        market_df['timestamp'] = market_df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    elif market_df['timestamp'].dt.tz is None:
        market_df['timestamp'] = market_df['timestamp'].dt.tz_localize('UTC').dt.tz_localize(None)
    
    # Ensure both have the same columns
    if 'open' not in market_df.columns and 'open' in orb_df.columns:
        market_df['open'] = market_df['close']  # Approximate
    if 'open' not in orb_df.columns and 'open' in market_df.columns:
        orb_df['open'] = orb_df['close']  # Approximate
        
    # Combine dataframes
    combined_df = pd.concat([orb_df, market_df], ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Combined data: {len(combined_df)} total data points")
    print(f"   Time range: {combined_df['timestamp'].iloc[0]} to {combined_df['timestamp'].iloc[-1]}")
    
    # Step 4: Populate data buffer
    print("\n📊 Step 4: Populating data buffer...")
    data_buffer = DataBuffer()
    
    for _, row in combined_df.iterrows():
        # Convert timezone-aware timestamps to timezone-naive UTC
        timestamp = row['timestamp']
        if hasattr(timestamp, 'tz') and timestamp.tz is not None:
            timestamp = timestamp.astimezone(pytz.UTC).replace(tzinfo=None)
        elif hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(pytz.UTC).replace(tzinfo=None)
        elif timestamp.tzinfo is None:
            # Assume UTC if timezone-naive
            pass
            
        market_data = MarketData(
            symbol="ZVSA",
            timestamp=timestamp,
            price=row['close'],
            volume=row['volume'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            trade_count=row.get('trade_count', 1),
            vwap=row.get('vwap', row['close'])
        )
        
        data_buffer.add_market_data(market_data)
    
    print(f"✅ Added {len(combined_df)} data points to buffer")
    
    # Step 5: Test ORB calculation
    print("\n📊 Step 5: Testing ORB calculation...")
    orb_calc = ORBCalculator()
    symbol_data = data_buffer.get_symbol_data("ZVSA")
    
    if symbol_data is not None and not symbol_data.empty:
        print(f"📊 Symbol data shape: {symbol_data.shape}")
        print(f"📊 Symbol data time range: {symbol_data['timestamp'].iloc[0]} to {symbol_data['timestamp'].iloc[-1]}")
        
        # Check what data we have around market open
        et_tz = pytz.timezone('US/Eastern')
        symbol_data_copy = symbol_data.copy()
        symbol_data_copy['timestamp'] = pd.to_datetime(symbol_data_copy['timestamp'])
        symbol_data_copy['timestamp'] = symbol_data_copy['timestamp'].dt.tz_localize('UTC')
        symbol_data_copy['timestamp_et'] = symbol_data_copy['timestamp'].dt.tz_convert(et_tz)
        symbol_data_copy['time_et'] = symbol_data_copy['timestamp_et'].dt.time
        
        # Check opening range data
        from datetime import time
        market_open = time(9, 30)
        orb_end = time(9, 45)
        
        orb_mask = (symbol_data_copy['time_et'] >= market_open) & (symbol_data_copy['time_et'] <= orb_end)
        orb_data = symbol_data_copy[orb_mask]
        
        print(f"📊 Opening range data points: {len(orb_data)}")
        if len(orb_data) > 0:
            print(f"📊 ORB period: {orb_data['time_et'].min()} - {orb_data['time_et'].max()} ET")
            print(f"📊 ORB data preview:")
            print(orb_data[['timestamp_et', 'time_et', 'high', 'low', 'close']].head())
        
        orb_level = orb_calc.calculate_orb_levels("ZVSA", symbol_data)
        
        if orb_level:
            print(f"✅ ORB Level calculated successfully!")
            print(f"   ORB High: ${orb_level.orb_high:.3f}")
            print(f"   ORB Low: ${orb_level.orb_low:.3f}")
            print(f"   ORB Range: ${orb_level.orb_range:.3f}")
            print(f"   Sample Count: {orb_level.sample_count}")
            
            # Step 6: Test breakout detection
            print("\n📊 Step 6: Testing breakout detection...")
            breakout_detector = BreakoutDetector(orb_calc)
            
            alerts_found = 0
            
            # Test each data point after the opening range
            et_tz = pytz.timezone('US/Eastern')
            market_open = datetime(2025, 7, 8, 9, 30, 0, tzinfo=et_tz)
            orb_end = market_open + timedelta(minutes=15)
            
            for _, row in symbol_data.iterrows():
                # Convert timestamp to check if it's after ORB period
                timestamp = row['timestamp']
                if hasattr(timestamp, 'tz') and timestamp.tz is None:
                    timestamp = pd.to_datetime(timestamp).tz_localize('UTC')
                
                timestamp_et = timestamp.astimezone(et_tz)
                
                if timestamp_et > orb_end:
                    # Test for breakouts using current high/low prices
                    for test_price in [row['high'], row['low'], row['close']]:
                        volume_ratio = 1.5  # Assume sufficient volume
                        
                        breakout_signal = breakout_detector.detect_breakout(
                            "ZVSA", test_price, volume_ratio, timestamp
                        )
                        
                        if breakout_signal:
                            alerts_found += 1
                            print(f"🚨 BREAKOUT DETECTED!")
                            print(f"   Time: {timestamp_et.strftime('%H:%M:%S')} ET")
                            print(f"   Type: {breakout_signal.breakout_type.value}")
                            print(f"   Price: ${breakout_signal.current_price:.3f}")
                            print(f"   Breakout %: {breakout_signal.breakout_percentage:.2f}%")
                            break  # Only report first breakout per time period
            
            if alerts_found == 0:
                print("❌ No breakouts detected")
                
                # Debug: Check thresholds
                print("\n🔍 Debug: Checking breakout thresholds...")
                high_threshold = orb_level.get_breakout_threshold(config.breakout_threshold)
                low_threshold = orb_level.get_breakdown_threshold(config.breakout_threshold)
                
                print(f"   ORB High: ${orb_level.orb_high:.3f}")
                print(f"   High Threshold: ${high_threshold:.3f} (+{config.breakout_threshold*100:.1f}%)")
                print(f"   ORB Low: ${orb_level.orb_low:.3f}")
                print(f"   Low Threshold: ${low_threshold:.3f} (-{config.breakout_threshold*100:.1f}%)")
                
                # Check prices in post-ORB data
                post_orb_data = symbol_data[pd.to_datetime(symbol_data['timestamp']).dt.tz_localize('UTC').dt.tz_convert(et_tz) > orb_end]
                
                max_high = post_orb_data['high'].max()
                min_low = post_orb_data['low'].min()
                
                print(f"   Post-ORB Max High: ${max_high:.3f}")
                print(f"   Post-ORB Min Low: ${min_low:.3f}")
                
                if max_high > high_threshold:
                    print(f"   🚨 High breach detected: ${max_high:.3f} > ${high_threshold:.3f}")
                if min_low < low_threshold:
                    print(f"   🚨 Low breach detected: ${min_low:.3f} < ${low_threshold:.3f}")
            else:
                print(f"✅ Found {alerts_found} breakout alerts!")
                
        else:
            print("❌ ORB Level calculation failed")
    else:
        print("❌ No symbol data found")

if __name__ == "__main__":
    test_complete_orb_flow()