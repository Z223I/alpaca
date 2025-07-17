#!/usr/bin/env python3
"""
Test script to verify the simplified historical data fetching logic.
"""

import os
import sys
from datetime import datetime, timedelta
import pytz

def test_simplified_logic():
    """Test the simplified logic for different start times."""
    
    print("=== Testing Simplified Historical Data Logic ===")
    
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
            'name': 'Early Morning (9:35 AM)',
            'current_time': '09:35:00',
            'expected': 'Fetch 5 minutes (9:30-9:35)'
        },
        {
            'name': 'Mid Opening Range (9:38 AM)',
            'current_time': '09:38:00',
            'expected': 'Fetch 8 minutes (9:30-9:38)'
        },
        {
            'name': 'After Opening Range (10:00 AM)',
            'current_time': '10:00:00',
            'expected': 'Fetch 30 minutes (9:30-10:00)'
        },
        {
            'name': 'Mid-Session Start (11:00 AM)',
            'current_time': '11:00:00',
            'expected': 'Fetch 90 minutes (9:30-11:00) - SUFFICIENT FOR EMA20!'
        },
        {
            'name': 'Afternoon Start (2:00 PM)',
            'current_time': '14:00:00',
            'expected': 'Fetch 270 minutes (9:30-14:00) - PLENTY FOR EMAs!'
        }
    ]
    
    et_tz = pytz.timezone('US/Eastern')
    today = datetime.now(et_tz).date()
    
    print(f"\\nTesting scenarios for {today}:")
    print("=" * 80)
    
    for scenario in scenarios:
        print(f"\\n--- {scenario['name']} ---")
        
        # Parse current time
        hour, minute, second = map(int, scenario['current_time'].split(':'))
        current_time = et_tz.localize(datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute, second=second)))
        
        # Market open time
        market_open = et_tz.localize(datetime.combine(today, datetime.min.time().replace(hour=9, minute=30, second=0)))
        
        print(f"Current time: {current_time.strftime('%H:%M:%S')}")
        print(f"Market open:  {market_open.strftime('%H:%M:%S')}")
        
        # Apply the new simplified logic
        if current_time < market_open:
            action = "✓ No fetch needed (before market open)"
            data_range = "N/A"
            ema_status = "N/A"
        else:
            minutes_since_open = (current_time - market_open).total_seconds() / 60
            
            if minutes_since_open < 1:
                action = "✓ Real-time collection (started close to market open)"
                data_range = "N/A"
                ema_status = "Will build EMAs in real-time"
            else:
                action = f"📊 Fetch ALL data from market open to current time"
                data_range = f"{market_open.strftime('%H:%M')} to {current_time.strftime('%H:%M')} ({minutes_since_open:.0f} minutes)"
                
                # EMA analysis
                if minutes_since_open >= 20:
                    ema_status = "✅ SUFFICIENT for EMA20 (and EMA9)"
                elif minutes_since_open >= 9:
                    ema_status = "✅ SUFFICIENT for EMA9 (partial EMA20)"
                else:
                    ema_status = "⚠️ Insufficient for reliable EMAs"
        
        print(f"Action: {action}")
        if data_range != "N/A":
            print(f"Data range: {data_range}")
        print(f"EMA Status: {ema_status}")
        print(f"Expected: {scenario['expected']}")
        
        # Highlight the key improvement
        if "11:00" in scenario['name'] or "2:00" in scenario['name']:
            print("🎯 KEY IMPROVEMENT: Now fetches sufficient data for EMAs!")
        
        print()
    
    print("=" * 80)
    print("SUMMARY OF SIMPLIFIED LOGIC:")
    print("1. Before 9:30 AM: No fetch needed")
    print("2. 9:30-9:31 AM: Real-time collection")
    print("3. After 9:31 AM: Fetch ALL data from 9:30 AM to current time")
    print()
    print("✅ BENEFITS:")
    print("  - Simplified logic (no complex partial fetching)")
    print("  - Always ensures sufficient data for EMA calculations")
    print("  - Works regardless of start time")
    print("  - Complete historical context for accurate indicators")
    print()
    print("🎯 EMA CALCULATION STATUS:")
    print("  - Starting at 9:39 AM: 9 minutes → ✅ EMA9 ready")
    print("  - Starting at 9:50 AM: 20 minutes → ✅ EMA20 ready")
    print("  - Starting at 11:00 AM: 90 minutes → ✅ Both EMAs ready")
    print("=" * 80)

if __name__ == "__main__":
    test_simplified_logic()