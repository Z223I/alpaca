#!/usr/bin/env python3
"""
Test the opening range data fetching functionality.
"""

import sys
import asyncio
from datetime import datetime, timedelta, time

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.config.alert_config import config

# Mock the historical data fetching to test the logic without API calls
class MockORBAlertSystem:
    """Mock ORB Alert System for testing opening range logic."""
    
    def __init__(self):
        self.historical_client = "mock_client"  # Simulate client exists
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _check_opening_range_status(self) -> dict:
        """Check if we need to fetch opening range data."""
        now = datetime.now()
        
        # Parse market open time for today
        market_open_hour, market_open_minute = map(int, config.market_open_time.split(':'))
        market_open_today = now.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
        
        # Calculate opening range end time (market open + 15 minutes)
        orb_end_time = market_open_today + timedelta(minutes=config.orb_period_minutes)
        
        status = {
            "current_time": now,
            "market_open_today": market_open_today,
            "orb_end_time": orb_end_time,
            "market_has_opened": now >= market_open_today,
            "orb_period_ended": now > orb_end_time,
            "need_historical_fetch": now > orb_end_time,
            "within_orb_period": market_open_today <= now <= orb_end_time
        }
        
        return status

def test_opening_range_logic():
    """Test the opening range data fetching logic."""
    
    print("=== Opening Range Data Fetch Test ===")
    
    mock_system = MockORBAlertSystem()
    status = mock_system._check_opening_range_status()
    
    print(f"Current time: {status['current_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Market open today: {status['market_open_today'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ORB period end: {status['orb_end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("=== Status Analysis ===")
    print(f"Market has opened: {'✅ Yes' if status['market_has_opened'] else '❌ No'}")
    print(f"ORB period ended: {'✅ Yes' if status['orb_period_ended'] else '❌ No'}")
    print(f"Within ORB period: {'✅ Yes' if status['within_orb_period'] else '❌ No'}")
    print(f"Need historical fetch: {'✅ Yes' if status['need_historical_fetch'] else '❌ No'}")
    print()
    
    print("=== Expected Behavior ===")
    if not status['market_has_opened']:
        print("• System would wait until market open")
        print("• No historical data fetch needed")
    elif status['within_orb_period']:
        print("• System starts immediately (market just opened)")
        print("• No historical data fetch needed - still collecting opening range")
    elif status['need_historical_fetch']:
        print("• System starts immediately")
        print("• 🔄 Historical data fetch REQUIRED for opening range (9:30-9:45)")
        print(f"• Would fetch data from {status['market_open_today'].strftime('%H:%M')} to {status['orb_end_time'].strftime('%H:%M')}")
    
    print()
    print("=== Configuration ===")
    print(f"ORB period: {config.orb_period_minutes} minutes")
    print(f"Market open time: {config.market_open_time} ET")
    print(f"Start at market open: {config.start_collection_at_open}")
    
    # Test different scenarios
    print()
    print("=== Scenario Testing ===")
    
    scenarios = [
        ("Pre-market (8:00)", time(8, 0)),
        ("Market open (9:30)", time(9, 30)),
        ("During ORB (9:40)", time(9, 40)),
        ("Just after ORB (9:46)", time(9, 46)),
        ("Mid-day (12:00)", time(12, 0)),
        ("After hours (17:00)", time(17, 0))
    ]
    
    for scenario_name, test_time in scenarios:
        # Create a test datetime for today at the specified time
        test_datetime = datetime.now().replace(
            hour=test_time.hour,
            minute=test_time.minute,
            second=0,
            microsecond=0
        )
        
        market_open = test_datetime.replace(hour=9, minute=30)
        orb_end = market_open + timedelta(minutes=15)
        
        need_fetch = test_datetime > orb_end
        
        print(f"{scenario_name:20} | Need historical fetch: {'Yes' if need_fetch else 'No'}")

if __name__ == "__main__":
    test_opening_range_logic()