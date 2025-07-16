#!/usr/bin/env python3
"""
Test script to verify API data coverage for complete trading session (9:30 AM - 4:00 PM ET).
This script will test the Alpaca API to ensure we're getting data for the full trading session.
"""

import os
import sys
import pandas as pd
import pytz
from datetime import datetime, date, time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.api.init_alpaca_client import init_alpaca_client

def test_api_data_coverage(symbols=['AAPL', 'OPEN', 'IXHL'], test_date=None):
    """
    Test API data coverage for specified symbols and date.
    
    Args:
        symbols: List of stock symbols to test
        test_date: Date to test (defaults to today)
    """
    
    # Initialize API client
    try:
        api = init_alpaca_client()
        print(f"✓ Successfully initialized Alpaca API client")
        print(f"  Base URL: {api._base_url}")
    except Exception as e:
        print(f"✗ Failed to initialize API client: {e}")
        return False
    
    # Use today's date if not specified
    if test_date is None:
        test_date = date.today()
    
    # Convert to string format for API
    date_str = test_date.strftime('%Y-%m-%d')
    
    # Define trading session times in ET
    et_tz = pytz.timezone('America/New_York')
    session_start = et_tz.localize(datetime.combine(test_date, time(9, 30)))
    session_end = et_tz.localize(datetime.combine(test_date, time(16, 0)))
    
    print(f"\n=== Testing API Data Coverage for {date_str} ===")
    print(f"Expected trading session: {session_start} to {session_end}")
    print(f"Session duration: {(session_end - session_start).total_seconds() / 60:.0f} minutes")
    
    results = {}
    
    for symbol in symbols:
        print(f"\n--- Testing {symbol} ---")
        
        try:
            # Get market data using the API
            print(f"Requesting data for {symbol} from {session_start} to {session_end}")
            
            # Use the same parameters as in the actual code
            bars = api.get_bars(
                symbol,
                '1Min',
                start=session_start.isoformat(),
                end=session_end.isoformat(),
                adjustment='raw',
                limit=10000
            )
            
            # Convert to DataFrame
            df = bars.df
            
            if df.empty:
                print(f"✗ No data returned for {symbol}")
                results[symbol] = {'status': 'no_data', 'coverage': 0.0}
                continue
            
            # Convert index to ET timezone for analysis
            df.index = df.index.tz_convert(et_tz)
            
            # Analyze data coverage
            data_start = df.index.min()
            data_end = df.index.max()
            data_points = len(df)
            
            print(f"  Data range: {data_start} to {data_end}")
            print(f"  Data points: {data_points}")
            
            # Check if we have opening range data (9:30-9:45 AM)
            opening_range_start = session_start
            opening_range_end = et_tz.localize(datetime.combine(test_date, time(9, 45)))
            
            opening_data = df[(df.index >= opening_range_start) & (df.index <= opening_range_end)]
            opening_data_points = len(opening_data)
            
            print(f"  Opening range data (9:30-9:45): {opening_data_points} points")
            
            if opening_data_points > 0:
                first_opening_point = opening_data.index.min()
                last_opening_point = opening_data.index.max()
                print(f"    First opening point: {first_opening_point}")
                print(f"    Last opening point: {last_opening_point}")
            else:
                print(f"    ✗ NO OPENING RANGE DATA FOUND!")
            
            # Check first hour coverage (9:30-10:30 AM)
            first_hour_end = et_tz.localize(datetime.combine(test_date, time(10, 30)))
            first_hour_data = df[(df.index >= session_start) & (df.index <= first_hour_end)]
            first_hour_points = len(first_hour_data)
            
            print(f"  First hour data (9:30-10:30): {first_hour_points} points")
            
            # Calculate overall session coverage
            expected_minutes = (session_end - session_start).total_seconds() / 60
            actual_minutes = (data_end - data_start).total_seconds() / 60
            coverage_percentage = (actual_minutes / expected_minutes) * 100
            
            print(f"  Session coverage: {coverage_percentage:.1f}% ({actual_minutes:.0f}/{expected_minutes:.0f} minutes)")
            
            # Check for gaps in the first hour
            if first_hour_points > 0:
                first_hour_start_actual = first_hour_data.index.min()
                minutes_late = (first_hour_start_actual - session_start).total_seconds() / 60
                print(f"  Minutes late from 9:30 AM: {minutes_late:.0f}")
                
                if minutes_late > 1:
                    print(f"    ⚠️  WARNING: Data starts {minutes_late:.0f} minutes late!")
            
            # Store results
            results[symbol] = {
                'status': 'success',
                'data_points': data_points,
                'data_start': data_start,
                'data_end': data_end,
                'coverage': coverage_percentage,
                'opening_range_points': opening_data_points,
                'first_hour_points': first_hour_points,
                'minutes_late': minutes_late if first_hour_points > 0 else None
            }
            
            # Show sample of first few data points
            print(f"  First 5 data points:")
            for i, (timestamp, row) in enumerate(df.head().iterrows()):
                print(f"    {timestamp}: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} V={row['volume']}")
                
        except Exception as e:
            print(f"✗ Error getting data for {symbol}: {e}")
            results[symbol] = {'status': 'error', 'error': str(e)}
    
    # Summary
    print(f"\n=== SUMMARY ===")
    for symbol, result in results.items():
        if result['status'] == 'success':
            status = "✓" if result['coverage'] > 95 and result['opening_range_points'] > 0 else "⚠️"
            print(f"{status} {symbol}: {result['coverage']:.1f}% coverage, {result['opening_range_points']} opening range points")
            if result['minutes_late'] and result['minutes_late'] > 1:
                print(f"    ⚠️  Data starts {result['minutes_late']:.0f} minutes late")
        else:
            print(f"✗ {symbol}: {result['status']}")
    
    return results

if __name__ == "__main__":
    # Test with today's date
    test_date = date(2025, 7, 16)  # Use the date from the chart
    results = test_api_data_coverage(test_date=test_date)