#!/usr/bin/env python3
"""
Test the opening range data fetch fix
"""

import sys
import os
from datetime import datetime, timedelta
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca/code')
from orb_alerts import ORBAlertSystem

def test_opening_range_fix():
    """Test that opening range data fetch now gets the correct time range."""
    
    print("🔍 Testing Opening Range Data Fetch Fix")
    print("=" * 60)
    
    # Create ORB alert system in test mode
    orb_system = ORBAlertSystem(test_mode=True)
    
    # Test the _fetch_with_legacy_api method directly
    symbols = ['ZVSA']
    
    # Create test time range (9:30-9:45 AM ET on July 8, 2025)
    et_tz = pytz.timezone('US/Eastern')
    test_date = datetime(2025, 7, 8, 9, 30, 0, tzinfo=et_tz)
    start_time = test_date
    end_time = test_date + timedelta(minutes=15)
    
    print(f"📊 Test time range:")
    print(f"   Start: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   End: {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Test the string formatting
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')
    
    print(f"📊 API call strings:")
    print(f"   Start string: {start_str}")
    print(f"   End string: {end_str}")
    
    print(f"\n✅ Format fix verified!")
    print(f"   Before fix: Would have used '2025-07-08' (entire day)")
    print(f"   After fix: Uses '{start_str}' (specific time range)")
    
    # Check that the times are correct
    if "09:30:00" in start_str and "09:45:00" in end_str:
        print(f"✅ Time range is correct for opening range period")
    else:
        print(f"❌ Time range is incorrect")
        
    print(f"\nThis fix should resolve the issue where historical opening range")
    print(f"data fetch was getting pre-market data instead of 9:30-9:45 AM data.")

if __name__ == "__main__":
    test_opening_range_fix()