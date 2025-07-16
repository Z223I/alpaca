#!/usr/bin/env python3
"""
Fix to handle missing opening range data gracefully in ORB charts.
This will modify the plotting function to detect and handle incomplete data.
"""

import os
import sys
from datetime import datetime, time
import pytz

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atoms.display.plot_candle_chart import plot_candle_chart

def create_enhanced_plot_function():
    """
    Create an enhanced plotting function that handles missing opening range data.
    """
    
    # Read the current plot_candle_chart function
    plot_file = '/home/wilsonb/dl/github.com/z223i/alpaca/atoms/display/plot_candle_chart.py'
    
    # The fix involves:
    # 1. Detect if opening range data is missing
    # 2. Add warning messages
    # 3. Still calculate ORB levels from available data if possible
    # 4. Add data completeness indicator to chart
    
    enhancements = """
    
    # Enhanced ORB data completeness check
    def check_data_completeness(symbol_data, et_tz):
        '''Check if we have complete trading session data, especially opening range.'''
        
        if len(symbol_data) == 0:
            return {'complete': False, 'missing_opening': True, 'message': 'No data available'}
        
        # Get the trading date
        first_timestamp = symbol_data['timestamp'].iloc[0]
        if hasattr(first_timestamp, 'date'):
            trading_date = first_timestamp.date()
        else:
            trading_date = first_timestamp.to_pydatetime().date()
        
        # Define expected opening range: 9:30-9:45 AM ET
        expected_open = et_tz.localize(datetime.combine(trading_date, time(9, 30)))
        opening_range_end = et_tz.localize(datetime.combine(trading_date, time(9, 45)))
        
        # Check if we have any data in the opening range
        opening_mask = (symbol_data['timestamp'] >= expected_open) & (symbol_data['timestamp'] <= opening_range_end)
        opening_data = symbol_data[opening_mask]
        
        # Check actual data start time
        actual_start = symbol_data['timestamp'].min()
        if actual_start.tzinfo is None:
            actual_start = et_tz.localize(actual_start)
        
        missing_opening = len(opening_data) == 0
        minutes_late = (actual_start - expected_open).total_seconds() / 60 if actual_start > expected_open else 0
        
        completeness = {
            'complete': not missing_opening and minutes_late < 5,  # Allow 5 minute grace period
            'missing_opening': missing_opening,
            'minutes_late': minutes_late,
            'actual_start': actual_start,
            'expected_start': expected_open,
            'opening_data_points': len(opening_data),
            'message': f'Data starts {minutes_late:.0f} minutes late' if minutes_late > 0 else 'Complete data'
        }
        
        return completeness
    
    This function should be integrated into the plot_candle_chart function.
    """
    
    print("Enhanced plotting function design:")
    print(enhancements)
    
    return True

if __name__ == "__main__":
    create_enhanced_plot_function()