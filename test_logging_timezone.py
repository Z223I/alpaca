#!/usr/bin/env python3
"""
Test the Eastern Time logging formatter
"""

import sys
import logging
from datetime import datetime
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca/code')

def test_logging_timezone():
    """Test the Eastern Time logging formatter."""
    
    print("🔍 Testing Eastern Time Logging")
    print("=" * 50)
    
    # Test the custom formatter
    class EasternFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            et_tz = pytz.timezone('US/Eastern')
            et_time = datetime.fromtimestamp(record.created, et_tz)
            if datefmt:
                return et_time.strftime(datefmt)
            return et_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3] + ' ET'
    
    # Create test logger
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add Eastern Time handler
    handler = logging.StreamHandler()
    formatter = EasternFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Test log messages
    print("Current system time for comparison:")
    print(f"  System UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    et_tz = pytz.timezone('US/Eastern')
    print(f"  System ET:  {datetime.now(et_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    print("\nTesting Eastern Time logger:")
    logger.info("Test log message - this should show Eastern Time")
    logger.info("Another test message for verification")
    
    print("\n✅ Eastern Time logging test completed")

if __name__ == "__main__":
    test_logging_timezone()