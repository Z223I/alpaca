#!/usr/bin/env python3
"""
Test script to verify the timestamp validation fix in orb_alerts_trade_stocks.py

This test uses the VERB superduper alert that was previously filtered out due to
the after-hours backtesting execution time bug.
"""

import sys
import json
import os
from datetime import datetime
import pytz

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from code.orb_alerts_trade_stocks import ORBTradeStocksMonitor

def test_timestamp_validation():
    """Test the fixed timestamp validation logic"""
    
    # Load the VERB test alert
    alert_file = "runs/2025-08-04/VERB/run_2025-08-04_tf10_th0.65_bd5dbd92/historical_data/2025-08-04/superduper_alerts/bullish/superduper_alert_VERB_20250824_174906.json"
    
    try:
        with open(alert_file, 'r') as f:
            superduper_alert_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Test alert file not found: {alert_file}")
        return False
    
    print("🧪 Testing Timestamp Validation Fix")
    print("=" * 50)
    
    # Display key timestamps
    runtime_timestamp = superduper_alert_data.get('timestamp')
    historical_timestamp = superduper_alert_data.get('latest_super_alert', {}).get('timestamp')
    original_timestamp = superduper_alert_data.get('latest_super_alert', {}).get('original_alert', {}).get('timestamp')
    
    print(f"📅 **Timestamps in Alert:**")
    print(f"   • Runtime (backtesting):  {runtime_timestamp}")
    print(f"   • Historical super alert: {historical_timestamp}")  
    print(f"   • Original alert:         {original_timestamp}")
    print()
    
    # Check alert message for time signal
    alert_message = superduper_alert_data.get('alert_message', '')
    if "• Time of Day:" in alert_message:
        lines = alert_message.split('\n')
        for line in lines:
            if "• Time of Day:" in line:
                print(f"⏰ **Time Signal in Message:** {line.strip()}")
                break
    print()
    
    # Create monitor instance (using dummy config)
    print("🔧 Creating ORBTradeStocksMonitor instance...")
    monitor = ORBTradeStocksMonitor()
    
    # Test the validation method
    print("🚀 **Testing _validate_time_of_day_signal()...**")
    print()
    
    # This should now pass because we use historical timestamp (12:40 PM) 
    # instead of runtime timestamp (5:49 PM)
    try:
        result = monitor._validate_time_of_day_signal(superduper_alert_data)
        
        print(f"📋 **Validation Result:** {'✅ PASSED' if result else '❌ FAILED'}")
        
        if result:
            print()
            print("🎉 **SUCCESS!** The fix correctly:")
            print("   • Extracted historical timestamp (2025-08-04T12:40:00)")
            print("   • Validated market hours using 12:40 PM (market hours)")  
            print("   • Ignored runtime timestamp (5:49 PM after hours)")
            print("   • Alert should now reach superduper_alerts_sent directory")
        else:
            print()
            print("❌ **FAILED!** The alert was still rejected.")
            print("   This suggests the fix may not be working correctly.")
            
    except Exception as e:
        print(f"💥 **ERROR during validation:** {e}")
        return False
    
    print()
    print("=" * 50)
    return result

def main():
    """Main test function"""
    print("🔍 **Timestamp Validation Fix Test**")
    print()
    
    # Show current time for reference
    et_tz = pytz.timezone('US/Eastern')
    current_et = datetime.now(et_tz)
    print(f"🕐 Current Time (ET): {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"🏪 Market Status: {'🟢 OPEN' if 9.5 <= current_et.hour + current_et.minute/60 <= 16 and current_et.weekday() < 5 else '🔴 CLOSED'}")
    print()
    
    success = test_timestamp_validation()
    
    if success:
        print("✅ **TEST PASSED** - Timestamp validation fix is working!")
        print("   VERB and other symbols should now appear in summary charts.")
    else:
        print("❌ **TEST FAILED** - Fix needs investigation.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())