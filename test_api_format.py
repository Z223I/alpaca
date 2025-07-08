#!/usr/bin/env python3
"""
Test if the API format works with a minimal test
"""

import sys
from datetime import datetime, timedelta
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.config.alert_config import config

def test_api_format():
    """Test if the new API format works."""
    
    print("🔍 Testing API Format")
    print("=" * 40)
    
    # Test the same logic as in orb_alerts.py
    try:
        from alpaca_trade_api.rest import REST
        
        # Initialize client
        historical_client = REST(
            key_id=config.api_key,
            secret_key=config.secret_key,
            base_url=config.base_url
        )
        
        # Create test time range
        et_tz = pytz.timezone('US/Eastern')
        test_date_naive = datetime(2025, 7, 8, 9, 30, 0)
        start_time = et_tz.localize(test_date_naive)
        end_time = start_time + timedelta(minutes=15)
        
        # Convert to UTC and format
        start_utc = start_time.astimezone(pytz.UTC)
        end_utc = end_time.astimezone(pytz.UTC)
        start_str = start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        print(f"📊 Testing format with symbol ZVSA")
        print(f"   Start: {start_str}")
        print(f"   End: {end_str}")
        
        # Test API call
        feed = 'iex' if "paper" in config.base_url else 'sip'
        print(f"   Feed: {feed}")
        
        bars = historical_client.get_bars(
            'ZVSA',
            '1Min',
            start=start_str,
            end=end_str,
            limit=20,
            feed=feed
        )
        
        if bars:
            print(f"✅ API call successful!")
            print(f"   Received {len(bars)} bars")
            
            # Show first few bars
            for i, bar in enumerate(bars[:3]):
                print(f"   Bar {i+1}: {bar.t} - High: ${bar.h:.3f}, Low: ${bar.l:.3f}, Close: ${bar.c:.3f}")
        else:
            print(f"⚠️  API call returned no data")
            
    except Exception as e:
        print(f"❌ API call failed: {e}")

if __name__ == "__main__":
    test_api_format()