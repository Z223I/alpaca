#!/usr/bin/env python3
"""
Test ZVSA ORB alert processing to debug why no alerts are generated
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
import asyncio

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.config.alert_config import config
from molecules.orb_alert_engine import ORBAlertEngine
from atoms.websocket.alpaca_stream import MarketData
from atoms.websocket.data_buffer import DataBuffer
from atoms.indicators.orb_calculator import ORBCalculator
from atoms.alerts.breakout_detector import BreakoutDetector
from atoms.alerts.confidence_scorer import ConfidenceScorer

def test_zvsa_orb_alert():
    """Test ZVSA ORB alert processing step by step."""
    
    print("🔍 Testing ZVSA ORB Alert Processing")
    print("=" * 60)
    
    # Load the ZVSA data
    csv_file = "historical_data/2025-07-08/market_data/ZVSA_20250708_100227.csv"
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} data points from {csv_file}")
        
        # Convert timestamps and handle timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Filter for ZVSA
        df = df[df['symbol'] == 'ZVSA']
        print(f"📊 ZVSA data points: {len(df)}")
        print(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        # Create data buffer and populate with historical data
        data_buffer = DataBuffer()
        
        for _, row in df.iterrows():
            # Convert timezone-aware timestamp to timezone-naive UTC for compatibility
            timestamp = row['timestamp'].astimezone(pytz.UTC).replace(tzinfo=None)
            
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
        
        print(f"✅ Added {len(df)} data points to buffer")
        
        # Test ORB calculation
        print("\n🔍 Testing ORB Calculation...")
        orb_calc = ORBCalculator()
        symbol_data = data_buffer.get_symbol_data("ZVSA")
        
        if symbol_data is not None:
            print(f"📊 Symbol data shape: {symbol_data.shape}")
            print(f"📊 Symbol data columns: {symbol_data.columns.tolist()}")
            print(f"📊 First few timestamps: {symbol_data['timestamp'].head().tolist()}")
            
            try:
                orb_level = orb_calc.calculate_orb_levels("ZVSA", symbol_data)
                
                if orb_level:
                    print(f"✅ ORB Level calculated successfully")
                    print(f"   ORB High: ${orb_level.orb_high:.3f}")
                    print(f"   ORB Low: ${orb_level.orb_low:.3f}")
                    print(f"   ORB Range: ${orb_level.orb_range:.3f}")
                    print(f"   ORB Midpoint: ${orb_level.orb_midpoint:.3f}")
                    print(f"   Data Points: {len(orb_level.orb_data)}")
                    
                    # Test breakout detection
                    print("\n🔍 Testing Breakout Detection...")
                    breakout_detector = BreakoutDetector(orb_calc)
                    
                    # Test with data points that should trigger breakout
                    post_orb_data = symbol_data[symbol_data['timestamp'] > orb_level.orb_end_time]
                
                alerts_found = 0
                for _, row in post_orb_data.iterrows():
                    current_price = row['close']
                    current_high = row['high']
                    current_low = row['low']
                    volume_ratio = 1.5  # Assume sufficient volume
                    
                    # Check for breakout using high/low prices
                    for test_price in [current_high, current_low, current_price]:
                        timestamp = row['timestamp']
                        # Convert to timezone-aware for breakout detector
                        if hasattr(timestamp, 'tz') and timestamp.tz is None:
                            timestamp = timestamp.replace(tzinfo=pytz.UTC)
                        
                        breakout_signal = breakout_detector.detect_breakout(
                            "ZVSA", test_price, volume_ratio, timestamp
                        )
                        
                        if breakout_signal:
                            alerts_found += 1
                            et_tz = pytz.timezone('US/Eastern')
                            timestamp_et = breakout_signal.timestamp.astimezone(et_tz)
                            print(f"🚨 BREAKOUT DETECTED:")
                            print(f"   Time: {timestamp_et.strftime('%H:%M:%S')} ET")
                            print(f"   Type: {breakout_signal.breakout_type.value}")
                            print(f"   Price: ${breakout_signal.current_price:.3f}")
                            print(f"   Breakout %: {breakout_signal.breakout_percentage:.2f}%")
                            print(f"   Volume Ratio: {breakout_signal.volume_ratio:.1f}")
                            
                            # Test confidence scoring
                            print("\n🔍 Testing Confidence Scoring...")
                            confidence_scorer = ConfidenceScorer()
                            confidence_score = confidence_scorer.score_breakout(
                                breakout_signal, symbol_data
                            )
                            print(f"   Confidence Score: {confidence_score:.3f}")
                            
                            # Test alert window
                            print("\n🔍 Testing Alert Window...")
                            is_within_window = breakout_detector.is_within_alert_window(breakout_signal.timestamp)
                            print(f"   Within alert window: {is_within_window}")
                            
                            print()
                
                    if alerts_found == 0:
                        print("❌ No breakouts detected by breakout detector")
                        
                        # Debug: Check thresholds
                        print("\n🔍 Debug: Checking breakout thresholds...")
                        high_threshold = orb_level.get_breakout_threshold(config.breakout_threshold)
                        low_threshold = orb_level.get_breakdown_threshold(config.breakout_threshold)
                        
                        print(f"   ORB High: ${orb_level.orb_high:.3f}")
                        print(f"   High Threshold: ${high_threshold:.3f} (+{config.breakout_threshold*100:.1f}%)")
                        print(f"   ORB Low: ${orb_level.orb_low:.3f}")
                        print(f"   Low Threshold: ${low_threshold:.3f} (-{config.breakout_threshold*100:.1f}%)")
                        
                        # Check if any prices breach thresholds
                        for _, row in post_orb_data.iterrows():
                            if row['high'] > high_threshold:
                                print(f"   🚨 High breach at {row['timestamp']}: ${row['high']:.3f}")
                            if row['low'] < low_threshold:
                                print(f"   🚨 Low breach at {row['timestamp']}: ${row['low']:.3f}")
                    
                    else:
                        print(f"✅ Found {alerts_found} potential breakouts")
                        
                else:
                    print("❌ ORB Level calculation failed - returned None")
                    
            except Exception as orb_error:
                print(f"❌ ORB Level calculation error: {orb_error}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ No symbol data found in buffer")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zvsa_orb_alert()