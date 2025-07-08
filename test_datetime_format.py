#!/usr/bin/env python3
"""
Test the new datetime format for legacy API
"""

import sys
from datetime import datetime, timedelta
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

def test_datetime_format():
    """Test the new datetime format."""
    
    print("🔍 Testing New Datetime Format")
    print("=" * 50)
    
    # Create test time range (9:30-9:45 AM ET on July 8, 2025)
    et_tz = pytz.timezone('US/Eastern')
    # Create naive datetime first, then localize
    test_date_naive = datetime(2025, 7, 8, 9, 30, 0)
    test_date = et_tz.localize(test_date_naive)
    start_time = test_date
    end_time = test_date + timedelta(minutes=15)
    
    print(f"📊 Original times (ET):")
    print(f"   Start: {start_time}")
    print(f"   End: {end_time}")
    
    # Test the new UTC conversion and formatting
    start_utc = start_time.astimezone(pytz.UTC)
    end_utc = end_time.astimezone(pytz.UTC)
    start_str = start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    print(f"\n📊 Converted to UTC:")
    print(f"   Start UTC: {start_utc}")
    print(f"   End UTC: {end_utc}")
    
    print(f"\n📊 API format strings:")
    print(f"   Start string: {start_str}")
    print(f"   End string: {end_str}")
    
    # Verify the conversion
    if "13:30:00Z" in start_str and "13:45:00Z" in end_str:
        print(f"\n✅ UTC conversion is correct!")
        print(f"   9:30 AM ET = 13:30 UTC ✓")
        print(f"   9:45 AM ET = 13:45 UTC ✓")
    else:
        print(f"\n❌ UTC conversion may be incorrect")
        
    print(f"\nThis format should be compatible with the legacy Alpaca API.")

if __name__ == "__main__":
    test_datetime_format()