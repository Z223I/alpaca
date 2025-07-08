#!/usr/bin/env python3
"""
Simple test to check why ZVSA ORB calculation fails
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

def test_zvsa_orb_simple():
    """Simple test to check ZVSA ORB calculation."""
    
    print("🔍 Simple ZVSA ORB Test")
    print("=" * 40)
    
    # Load the BSLK data
    csv_file = "historical_data/2025-07-08/market_data/BSLK_20250708_090100.csv"
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} data points")
        
        # Convert timestamps 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Filter for BSLK
        df = df[df['symbol'] == 'BSLK']
        print(f"📊 BSLK data points: {len(df)}")
        
        # Check first few rows
        print("\n📊 First 5 rows:")
        print(df.head())
        
        # Create data buffer and populate
        data_buffer = DataBuffer()
        
        for _, row in df.iterrows():
            # Convert to timezone-naive UTC
            timestamp = row['timestamp'].astimezone(pytz.UTC).replace(tzinfo=None)
            
            market_data = MarketData(
                symbol="BSLK",
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
        
        # Get symbol data
        symbol_data = data_buffer.get_symbol_data("BSLK")
        print(f"📊 Symbol data retrieved: {symbol_data is not None}")
        
        if symbol_data is not None:
            print(f"📊 Symbol data shape: {symbol_data.shape}")
            print(f"📊 Symbol data columns: {symbol_data.columns.tolist()}")
            print(f"📊 Timestamp dtype: {symbol_data['timestamp'].dtype}")
            print(f"📊 First timestamp: {symbol_data['timestamp'].iloc[0]}")
            print(f"📊 Last timestamp: {symbol_data['timestamp'].iloc[-1]}")
            
            # Try ORB calculation
            orb_calc = ORBCalculator()
            
            try:
                print("\n🔍 Attempting ORB calculation...")
                orb_level = orb_calc.calculate_orb_levels("BSLK", symbol_data)
                
                if orb_level:
                    print(f"✅ ORB Level calculated successfully!")
                    print(f"   ORB High: ${orb_level.orb_high:.3f}")
                    print(f"   ORB Low: ${orb_level.orb_low:.3f}")
                    print(f"   ORB Range: ${orb_level.orb_range:.3f}")
                    print(f"   Sample Count: {orb_level.sample_count}")
                    print(f"   Calculation Time: {orb_level.calculation_time}")
                else:
                    print("❌ ORB calculation returned None")
                    
            except Exception as e:
                print(f"❌ ORB calculation error: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zvsa_orb_simple()