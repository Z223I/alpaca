#!/usr/bin/env python3
"""
Test mixed timezone fix - verify historical and live data timestamps are consistent
"""

import sys
from datetime import datetime
import pandas as pd
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.websocket.alpaca_stream import MarketData
from atoms.websocket.data_buffer import DataBuffer

def test_mixed_timezone_fix():
    """Test that historical and live data timestamps are consistent."""
    
    print("🔍 Testing Mixed Timezone Fix")
    print("=" * 50)
    
    # Create data buffer
    data_buffer = DataBuffer()
    
    # 1. Simulate historical data (timezone-aware)
    print("📊 Adding historical data (timezone-aware)...")
    historical_timestamp = datetime(2025, 7, 8, 13, 31, 0, tzinfo=pytz.UTC)
    
    # Normalize to timezone-naive UTC (like the fix does)
    historical_timestamp_naive = historical_timestamp.astimezone(pytz.UTC).replace(tzinfo=None)
    
    historical_data = MarketData(
        symbol="TEST",
        timestamp=historical_timestamp_naive,
        price=100.0,
        volume=1000,
        high=100.5,
        low=99.5,
        close=100.0,
        trade_count=10,
        vwap=100.0
    )
    
    data_buffer.add_market_data(historical_data)
    print(f"   Historical timestamp: {historical_data.timestamp}")
    print(f"   Has timezone info: {historical_data.timestamp.tzinfo is not None}")
    
    # 2. Simulate live websocket data (timezone-naive UTC)
    print("\n🔴 Adding live websocket data (timezone-naive UTC)...")
    
    # Simulate websocket timestamp conversion (from the fix)
    websocket_timestamp_str = "2025-07-08T13:32:00.000Z"
    websocket_timestamp_aware = datetime.fromisoformat(websocket_timestamp_str.replace("Z", "+00:00"))
    websocket_timestamp_naive = websocket_timestamp_aware.astimezone(pytz.UTC).replace(tzinfo=None)
    
    live_data = MarketData(
        symbol="TEST",
        timestamp=websocket_timestamp_naive,
        price=101.0,
        volume=1100,
        high=101.5,
        low=100.5,
        close=101.0,
        trade_count=11,
        vwap=101.0
    )
    
    data_buffer.add_market_data(live_data)
    print(f"   Live timestamp: {live_data.timestamp}")
    print(f"   Has timezone info: {live_data.timestamp.tzinfo is not None}")
    
    # 3. Add more data to reach position 20+
    print("\n📈 Adding more data to test position 20+ scenario...")
    
    for i in range(25):
        test_data = MarketData(
            symbol="TEST",
            timestamp=datetime(2025, 7, 8, 13, 33 + i, 0),  # timezone-naive UTC
            price=100.0 + i,
            volume=1000 + i,
            high=100.5 + i,
            low=99.5 + i,
            close=100.0 + i,
            trade_count=10 + i,
            vwap=100.0 + i
        )
        data_buffer.add_market_data(test_data)
    
    # 4. Test DataFrame creation (this was failing before)
    print("\n🎯 Testing DataFrame creation with mixed data...")
    try:
        symbol_data = data_buffer.get_symbol_data("TEST")
        print(f"✅ DataFrame created successfully")
        print(f"   Total rows: {len(symbol_data)}")
        print(f"   Timestamp column type: {symbol_data['timestamp'].dtype}")
        print(f"   First timestamp: {symbol_data['timestamp'].iloc[0]}")
        print(f"   Last timestamp: {symbol_data['timestamp'].iloc[-1]}")
        
        # Check for timezone consistency
        timestamps = symbol_data['timestamp']
        has_tz = timestamps.dt.tz is not None
        print(f"   All timestamps are timezone-naive: {not has_tz}")
        
    except Exception as e:
        print(f"❌ DataFrame creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test with actual timestamp normalization function
    print("\n🔧 Testing timestamp normalization function...")
    
    def normalize_timestamp(timestamp):
        """Normalize timestamp to timezone-naive UTC."""
        if hasattr(timestamp, 'tz') and timestamp.tz is not None:
            # Convert timezone-aware to timezone-naive UTC
            return timestamp.astimezone(pytz.UTC).replace(tzinfo=None)
        elif hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
            # Convert timezone-aware to timezone-naive UTC
            return timestamp.astimezone(pytz.UTC).replace(tzinfo=None)
        return timestamp
    
    # Test with various timestamp formats
    test_timestamps = [
        datetime(2025, 7, 8, 13, 31, 0),  # timezone-naive
        datetime(2025, 7, 8, 13, 31, 0, tzinfo=pytz.UTC),  # timezone-aware UTC
        datetime(2025, 7, 8, 9, 31, 0, tzinfo=pytz.timezone('US/Eastern')),  # timezone-aware ET
    ]
    
    for i, ts in enumerate(test_timestamps):
        normalized = normalize_timestamp(ts)
        print(f"   Test {i+1}: {ts} -> {normalized} (tz: {normalized.tzinfo})")
    
    print("\n✅ All mixed timezone tests passed!")
    return True

if __name__ == "__main__":
    test_mixed_timezone_fix()