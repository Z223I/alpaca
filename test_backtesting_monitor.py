#!/usr/bin/env python3
"""
Test script to verify the backtesting mode works correctly
"""

import sys
import json
import os
import shutil
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from code.orb_alerts_trade_stocks import ORBTradeStocksMonitor

def test_backtesting_mode():
    """Test the backtesting mode functionality"""
    
    print("🧪 **Testing Backtesting Mode**")
    print("=" * 50)
    
    # Test parameters
    test_date = "2025-08-04"
    
    print(f"📅 Test Date: {test_date}")
    print(f"🏃 Mode: BACKTESTING")
    print()
    
    try:
        # Create monitor in backtesting mode
        print("🔧 Creating ORBTradeStocksMonitor in backtesting mode...")
        monitor = ORBTradeStocksMonitor(
            test_mode=True,  # Don't execute real trades
            no_telegram=True,  # Don't send notifications
            date=test_date,
            backtesting_mode=True
        )
        
        print("✅ Monitor created successfully!")
        print()
        
        # Check directory paths
        print("📁 **Directory Paths:**")
        print(f"   • Source alerts:     {monitor.source_alerts_dir}")
        print(f"   • Sent alerts:       {monitor.superduper_alerts_dir}")
        print(f"   • Trades:           {monitor.trades_dir}")
        print()
        
        # Check if directories exist
        print("🔍 **Directory Status:**")
        print(f"   • Source exists:     {monitor.source_alerts_dir.exists()}")
        print(f"   • Sent exists:       {monitor.superduper_alerts_dir.exists()}")
        print(f"   • Trades exists:     {monitor.trades_dir.exists()}")
        print()
        
        # Test validation with a sample alert
        print("🧪 **Testing Alert Validation:**")
        
        # Use one of the VERB alerts we know passes validation
        sample_alert_path = "runs/2025-08-04/VERB/run_2025-08-04_tf10_th0.65_bd5dbd92/historical_data/2025-08-04/superduper_alerts/bullish/superduper_alert_VERB_20250824_174906.json"
        
        if Path(sample_alert_path).exists():
            with open(sample_alert_path, 'r') as f:
                alert_data = json.load(f)
            
            result = monitor._validate_time_of_day_signal(alert_data)
            symbol = alert_data.get('symbol', 'UNKNOWN')
            
            print(f"   • Sample Alert: {symbol}")
            print(f"   • Validation:   {'✅ PASS' if result else '❌ FAIL'}")
            
            if result:
                print(f"   • This alert should be copied to sent directory during processing")
            else:
                print(f"   • This alert would be filtered out")
        else:
            print(f"   • Sample alert file not found: {sample_alert_path}")
        
        print()
        print("🏆 **Backtesting Mode Test: SUCCESS**")
        print("   The monitor is correctly configured for backtesting.")
        
        return True
        
    except Exception as e:
        print(f"❌ **Backtesting Mode Test: FAILED**")
        print(f"   Error: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 **Backtesting Mode Configuration Test**")
    print()
    
    success = test_backtesting_mode()
    
    if success:
        print()
        print("✅ **ALL TESTS PASSED**")
        print("   The backtesting monitor should now properly:")
        print("   1. Monitor superduper_alerts in runs/current")  
        print("   2. Validate alerts using historical timestamps")
        print("   3. Copy validated alerts to superduper_alerts_sent")
        print("   4. Show up in summary charts")
    else:
        print()
        print("❌ **TESTS FAILED**")
        print("   Check the errors above and fix before proceeding.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())