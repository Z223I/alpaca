#!/usr/bin/env python3
"""
Test timezone consistency across all data sources
"""

import sys
from datetime import datetime, timedelta
import pytz
import pandas as pd

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.websocket.alpaca_stream import MarketData
from atoms.websocket.data_buffer import DataBuffer

def test_timezone_consistency():
    """Test that all data uses consistent Eastern Time format."""
    
    print("🔍 Testing Timezone Consistency")
    print("=" * 50)
    
    # 1. Test websocket data processing
    print("📊 Testing websocket data processing...")
    
    # Simulate websocket data with UTC timestamp
    test_bar_data = {
        "T": "b",
        "S": "TEST",
        "t": "2025-07-08T13:30:00.000Z",  # 9:30 AM ET in UTC
        "c": 100.0,
        "h": 101.0,
        "l": 99.0,
        "v": 1000,
        "n": 10,
        "vw": 100.5
    }
    
    # Process timestamp like the websocket handler does
    timestamp_str = test_bar_data.get("t", "").replace("Z", "+00:00")
    timestamp_aware = datetime.fromisoformat(timestamp_str)
    et_tz = pytz.timezone('US/Eastern')
    timestamp_et_naive = timestamp_aware.astimezone(et_tz).replace(tzinfo=None)
    
    print(f"   Original UTC: {timestamp_str}")
    print(f"   Converted ET: {timestamp_et_naive}")
    print(f"   Expected: 09:30:00 ET")
    
    if timestamp_et_naive.hour == 9 and timestamp_et_naive.minute == 30:
        print("   ✅ Websocket timestamp conversion correct")
    else:
        print("   ❌ Websocket timestamp conversion incorrect")
    
    # 2. Test data buffer storage
    print("\n📊 Testing data buffer storage...")
    
    data_buffer = DataBuffer()
    market_data = MarketData(
        symbol="TEST",
        timestamp=timestamp_et_naive,
        price=100.0,
        volume=1000,
        high=101.0,
        low=99.0,
        close=100.0,
        trade_count=10,
        vwap=100.5
    )
    
    data_buffer.add_market_data(market_data)
    symbol_data = data_buffer.get_symbol_data("TEST")
    
    stored_timestamp = symbol_data['timestamp'].iloc[0]
    print(f"   Stored timestamp: {stored_timestamp}")
    print(f"   Timezone info: {getattr(stored_timestamp, 'tzinfo', 'None')}")
    
    if stored_timestamp.hour == 9 and stored_timestamp.minute == 30 and stored_timestamp.tzinfo is None:
        print("   ✅ Data buffer storage correct (timezone-naive ET)")
    else:
        print("   ❌ Data buffer storage incorrect")
    
    # 3. Test opening range data conversion
    print("\n📊 Testing opening range data conversion...")
    
    # Simulate opening range API data (timezone-aware)
    # Create properly localized Eastern Time
    naive_dt = datetime(2025, 7, 8, 9, 31, 0)
    et_aware_dt = et_tz.localize(naive_dt)
    test_api_data = [{
        'timestamp': et_aware_dt,
        'symbol': 'TEST',
        'open': 100.0,
        'high': 101.0,
        'low': 99.0,
        'close': 100.5,
        'volume': 1000,
        'trade_count': 10,
        'vwap': 100.25
    }]
    
    df = pd.DataFrame(test_api_data)
    print(f"   Original API timestamp: {df['timestamp'].iloc[0]}")
    
    # Apply the conversion logic from orb_alerts.py
    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is not None:
            # Convert timezone-aware to timezone-naive Eastern Time
            df['timestamp'] = df['timestamp'].dt.tz_convert(et_tz).dt.tz_localize(None)
    
    converted_timestamp = df['timestamp'].iloc[0]
    print(f"   Converted timestamp: {converted_timestamp}")
    print(f"   Timezone info: {getattr(converted_timestamp, 'tzinfo', 'None')}")
    
    if converted_timestamp.hour == 9 and converted_timestamp.minute == 31 and converted_timestamp.tzinfo is None:
        print("   ✅ Opening range data conversion correct")
    else:
        print("   ❌ Opening range data conversion incorrect")
    
    # 4. Summary
    print("\n📋 Summary:")
    print("   All data should now be stored as timezone-naive Eastern Time")
    print("   This ensures consistency between websocket and API data")
    print("   Files will show times like '09:30:00' instead of '13:30:00' or '09:30:00-04:00'")

if __name__ == "__main__":
    test_timezone_consistency()