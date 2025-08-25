#!/usr/bin/env python3
"""
Test script to validate timestamp fix against multiple sample alerts
to understand why no alerts are reaching the superduper_alerts_sent directory.
"""

import sys
import json
import os
import glob
from datetime import datetime
import pytz

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from code.orb_alerts_trade_stocks import ORBTradeStocksMonitor

def test_sample_alerts():
    """Test the timestamp validation fix against multiple sample alerts"""
    
    print("🔍 **Multiple Alert Validation Test**")
    print("=" * 60)
    
    # Get sample alerts from VERB and BTAI runs
    verb_alerts = glob.glob("runs/*/VERB/*/historical_data/*/superduper_alerts/bullish/*.json")[:3]
    btai_alerts = glob.glob("runs/*/BTAI/*/historical_data/*/superduper_alerts/bullish/*.json")[:3]
    
    sample_alerts = verb_alerts + btai_alerts
    
    if not sample_alerts:
        print("❌ No sample alerts found to test")
        return False
    
    print(f"📋 Testing {len(sample_alerts)} sample alerts...")
    print()
    
    # Create monitor instance
    monitor = ORBTradeStocksMonitor()
    
    results = []
    
    for i, alert_file in enumerate(sample_alerts, 1):
        symbol = "VERB" if "VERB" in alert_file else "BTAI" if "BTAI" in alert_file else "UNKNOWN"
        
        print(f"🧪 **Test {i}/{len(sample_alerts)}: {symbol}**")
        print(f"📁 File: {os.path.basename(alert_file)}")
        
        try:
            # Load alert data
            with open(alert_file, 'r') as f:
                superduper_alert_data = json.load(f)
            
            # Extract key timestamps
            runtime_timestamp = superduper_alert_data.get('timestamp')
            historical_timestamp = superduper_alert_data.get('latest_super_alert', {}).get('timestamp')
            original_timestamp = superduper_alert_data.get('latest_super_alert', {}).get('original_alert', {}).get('timestamp')
            
            print(f"   • Runtime:    {runtime_timestamp}")
            print(f"   • Historical: {historical_timestamp}")
            print(f"   • Original:   {original_timestamp}")
            
            # Check for time signal in message
            alert_message = superduper_alert_data.get('alert_message', '')
            time_signal = "NOT FOUND"
            if "• Time of Day:" in alert_message:
                lines = alert_message.split('\\n')
                for line in lines:
                    if "• Time of Day:" in line:
                        time_signal = line.strip().replace("• Time of Day: ", "")
                        break
            
            print(f"   • Time Signal: {time_signal}")
            
            # Test validation
            result = monitor._validate_time_of_day_signal(superduper_alert_data)
            
            print(f"   • **Result: {'✅ PASS' if result else '❌ FAIL'}**")
            print()
            
            results.append({
                'file': os.path.basename(alert_file),
                'symbol': symbol,
                'runtime': runtime_timestamp,
                'historical': historical_timestamp,
                'original': original_timestamp,
                'time_signal': time_signal,
                'passed': result
            })
            
        except Exception as e:
            print(f"   • **Error: {e}**")
            print()
            results.append({
                'file': os.path.basename(alert_file),
                'symbol': symbol,
                'error': str(e),
                'passed': False
            })
    
    # Summary
    print("=" * 60)
    print("📊 **SUMMARY**")
    print("=" * 60)
    
    passed_count = sum(1 for r in results if r.get('passed', False))
    total_count = len(results)
    
    print(f"✅ **Passed:** {passed_count}/{total_count}")
    print(f"❌ **Failed:** {total_count - passed_count}/{total_count}")
    print()
    
    if passed_count > 0:
        print("🎉 **PASSING ALERTS:**")
        for result in results:
            if result.get('passed', False):
                print(f"   • {result['symbol']}: {result['file']}")
        print()
    
    if passed_count < total_count:
        print("💥 **FAILING ALERTS:**")
        for result in results:
            if not result.get('passed', False):
                error_msg = f" (Error: {result.get('error', 'Unknown')})" if 'error' in result else ""
                print(f"   • {result['symbol']}: {result['file']}{error_msg}")
                if 'time_signal' in result:
                    print(f"     Time Signal: {result['time_signal']}")
        print()
    
    # Recommendations
    if passed_count == 0:
        print("🚨 **CRITICAL: All alerts failed validation!**")
        print("   The timestamp fix may not be working as expected.")
    elif passed_count < total_count:
        print("⚠️  **PARTIAL SUCCESS:** Some alerts are still being filtered.")
        print("   May need additional investigation for specific cases.")
    else:
        print("🏆 **PERFECT:** All alerts passed validation!")
        print("   The timestamp fix appears to be working correctly.")
    
    return passed_count > 0

def main():
    """Main test function"""
    et_tz = pytz.timezone('US/Eastern')
    current_et = datetime.now(et_tz)
    print(f"🕐 Current Time (ET): {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()
    
    success = test_sample_alerts()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())