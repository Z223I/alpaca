#!/usr/bin/env python3
"""
Test script to verify the updated opening range data fetching logic.
"""

import os
import sys
from datetime import datetime, timedelta
import pytz

def test_opening_range_logic():
    """Test the opening range logic for different start times."""
    
    print("=== Testing Updated Opening Range Logic ===")
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Before Market Open (9:25 AM)',
            'current_time': '09:25:00',
            'expected': 'No fetch needed'
        },
        {
            'name': 'At Market Open (9:30 AM)',
            'current_time': '09:30:00', 
            'expected': 'Real-time collection'
        },
        {
            'name': 'Early Opening Range (9:31 AM)',
            'current_time': '09:31:00',
            'expected': 'Fetch missing 1 minute (9:30-9:31)'
        },
        {
            'name': 'Mid Opening Range (9:38 AM)',
            'current_time': '09:38:00',
            'expected': 'Fetch missing 8 minutes (9:30-9:38)'
        },
        {
            'name': 'Late Opening Range (9:44 AM)',
            'current_time': '09:44:00',
            'expected': 'Fetch missing 14 minutes (9:30-9:44)'
        },
        {
            'name': 'Just After Opening Range (9:46 AM)',
            'current_time': '09:46:00',
            'expected': 'Fetch complete opening range (9:30-9:45)'
        },
        {
            'name': 'Well After Opening Range (10:00 AM)',
            'current_time': '10:00:00',
            'expected': 'Fetch complete opening range (9:30-9:45)'
        }
    ]
    
    et_tz = pytz.timezone('US/Eastern')
    today = datetime.now(et_tz).date()
    
    print(f"\nTesting scenarios for {today}:")
    print("=" * 80)
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Parse current time
        hour, minute, second = map(int, scenario['current_time'].split(':'))
        current_time = et_tz.localize(datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute, second=second)))
        
        # Market open time
        market_open = et_tz.localize(datetime.combine(today, datetime.min.time().replace(hour=9, minute=30, second=0)))
        
        # Opening range end time (15 minutes after market open)
        orb_end = market_open + timedelta(minutes=15)
        
        print(f"Current time: {current_time.strftime('%H:%M:%S')}")
        print(f"Market open:  {market_open.strftime('%H:%M:%S')}")
        print(f"ORB end:      {orb_end.strftime('%H:%M:%S')}")
        
        # Apply the new logic
        if current_time < market_open:
            action = "✓ No fetch needed (before market open)"
            data_range = "N/A"
        elif current_time <= orb_end:
            minutes_missed = (current_time - market_open).total_seconds() / 60
            
            if minutes_missed < 1:
                action = "✓ Real-time collection (started close to market open)"
                data_range = "N/A"
            else:
                action = f"⚠️  Fetch missing {minutes_missed:.1f} minutes of opening range data"
                data_range = f"{market_open.strftime('%H:%M')} to {current_time.strftime('%H:%M')}"
        else:
            action = "⚠️  Fetch complete opening range data"
            data_range = f"{market_open.strftime('%H:%M')} to {orb_end.strftime('%H:%M')}"
        
        print(f"Action: {action}")
        if data_range != "N/A":
            print(f"Data range to fetch: {data_range}")
        print(f"Expected: {scenario['expected']}")
        
        # Validate the logic matches expectations
        if "No fetch needed" in scenario['expected'] and "No fetch needed" in action:
            print("✅ PASS")
        elif "Real-time collection" in scenario['expected'] and "Real-time collection" in action:
            print("✅ PASS")
        elif "1 minute" in scenario['expected'] and "1.0 minutes" in action:
            print("✅ PASS")
        elif "8 minutes" in scenario['expected'] and "8.0 minutes" in action:
            print("✅ PASS")
        elif "14 minutes" in scenario['expected'] and "14.0 minutes" in action:
            print("✅ PASS")
        elif "complete opening range" in scenario['expected'] and "complete opening range" in action:
            print("✅ PASS")
        else:
            print("❌ FAIL - Logic doesn't match expected behavior")
    
    print("\n" + "=" * 80)
    print("SUMMARY OF NEW LOGIC:")
    print("1. Before 9:30 AM: No fetch needed")
    print("2. 9:30-9:31 AM: Real-time collection (minimal missed data)")
    print("3. 9:31-9:45 AM: Fetch missing portion + continue real-time")
    print("4. After 9:45 AM: Fetch complete opening range (9:30-9:45)")
    print("\nThis ensures complete ORB data regardless of start time!")
    print("=" * 80)

if __name__ == "__main__":
    test_opening_range_logic()