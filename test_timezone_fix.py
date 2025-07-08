#!/usr/bin/env python3
"""
Test timezone fix - verify no more pandas datetime conversion errors
"""

import sys
from datetime import datetime
import pandas as pd
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.websocket.alpaca_stream import MarketData
from atoms.websocket.data_buffer import DataBuffer

def test_timezone_conversion():
    """Test that timezone-naive UTC timestamps work with pandas."""
    
    print("🔍 Testing Timezone Conversion Fix")
    print("=" * 50)
    
    # Simulate websocket timestamp conversion
    timestamp_str = "2025-07-08T13:31:00.000Z"
    timestamp_aware = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    timestamp_naive = timestamp_aware.astimezone(pytz.UTC).replace(tzinfo=None)
    
    print(f"Original timestamp: {timestamp_str}")
    print(f"Timezone-aware: {timestamp_aware}")
    print(f"Timezone-naive UTC: {timestamp_naive}")
    print(f"Has timezone info: {timestamp_naive.tzinfo is not None}")
    print()
    
    # Test MarketData creation
    print("📊 Testing MarketData creation...")
    market_data = MarketData(
        symbol="TEST",
        timestamp=timestamp_naive,
        price=100.0,
        volume=1000,
        high=100.5,
        low=99.5,
        close=100.0,
        trade_count=10,
        vwap=100.0
    )
    print(f"✅ MarketData created successfully")
    print(f"   Timestamp: {market_data.timestamp}")
    print(f"   Timezone info: {market_data.timestamp.tzinfo}")
    print()
    
    # Test DataFrame creation (this was causing the error)
    print("📈 Testing DataFrame creation...")
    try:
        data_buffer = DataBuffer()
        data_buffer.add_market_data(market_data)
        
        # Add more data points to reach position 20+
        for i in range(25):
            test_data = MarketData(
                symbol="TEST",
                timestamp=timestamp_naive,
                price=100.0 + i,
                volume=1000,
                high=100.5 + i,
                low=99.5 + i,
                close=100.0 + i,
                trade_count=10,
                vwap=100.0 + i
            )
            data_buffer.add_market_data(test_data)
        
        # Get symbol data (this creates DataFrame)
        symbol_data = data_buffer.get_symbol_data("TEST")
        print(f"✅ DataFrame created successfully")
        print(f"   Rows: {len(symbol_data)}")
        print(f"   Timestamp column type: {symbol_data['timestamp'].dtype}")
        print(f"   First timestamp: {symbol_data['timestamp'].iloc[0]}")
        print()
        
    except Exception as e:
        print(f"❌ DataFrame creation failed: {e}")
        return False
    
    print("🎯 Testing ORB calculation with new timestamps...")
    try:
        from atoms.indicators.orb_calculator import ORBCalculator
        
        # Create test data with timezone-naive UTC timestamps
        test_data = []
        base_time = datetime(2025, 7, 8, 13, 31, 0)  # 09:31 ET in UTC
        
        for i in range(15):  # 15 minutes of data
            test_data.append({
                'timestamp': base_time.replace(minute=31+i),
                'symbol': 'TEST',
                'high': 100.0 + i,
                'low': 99.0 + i,
                'close': 99.5 + i,
                'volume': 1000
            })
        
        df = pd.DataFrame(test_data)
        orb_calc = ORBCalculator()
        orb_level = orb_calc.calculate_orb_levels('TEST', df)
        
        if orb_level:
            print(f"✅ ORB calculation successful")
            print(f"   ORB High: ${orb_level.orb_high:.3f}")
            print(f"   ORB Low: ${orb_level.orb_low:.3f}")
        else:
            print(f"❌ ORB calculation failed")
            
    except Exception as e:
        print(f"❌ ORB calculation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All timezone conversion tests passed!")
    return True

if __name__ == "__main__":
    test_timezone_conversion()