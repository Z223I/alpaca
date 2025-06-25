# Alpaca API Data Collection Guide

## Overview

This guide explains how to use the Alpaca API to collect historical market data for multiple stock symbols at specified time intervals. The Alpaca API provides comprehensive market data through their Python SDK with support for various timeframes and data types.

## Prerequisites

### Installation
```bash
pip install alpaca-py
# or
pip install alpaca-trade-api  # Legacy version
```

### API Credentials
Create a `.env` file with your Alpaca API credentials:
```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
```

**Note**: Crypto data doesn't require API keys, but stock data requires authentication.

## Basic Setup

### Import Required Libraries
```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
```

### Initialize the Client
```python
# For stock data (requires API keys)
client = StockHistoricalDataClient(
    api_key=os.getenv('ALPACA_API_KEY'), 
    secret_key=os.getenv('ALPACA_SECRET_KEY')
)

# For crypto data (no API keys required)
from alpaca.data.historical import CryptoHistoricalDataClient
crypto_client = CryptoHistoricalDataClient()
```

## Time Intervals and Timeframes

Alpaca supports the following timeframes:
- **Minute intervals**: `1Min`, `5Min`, `15Min`, `30Min`
- **Daily intervals**: `1Day`
- **Weekly intervals**: `1Week`
- **Monthly intervals**: `1Month`

### Using TimeFrame Class
```python
from alpaca.data.timeframe import TimeFrame

# Available timeframes
TimeFrame.Minute      # 1 minute bars
TimeFrame.Hour        # 1 hour bars  
TimeFrame.Day         # 1 day bars
TimeFrame.Week        # 1 week bars
TimeFrame.Month       # 1 month bars

# Custom intervals
TimeFrame(5, TimeFrameUnit.Minute)   # 5-minute bars
TimeFrame(15, TimeFrameUnit.Minute)  # 15-minute bars
```

## Collecting Data for Multiple Stock Symbols

### Basic Example - Multiple Symbols
```python
def get_stock_data(symbols, timeframe, start_date, end_date):
    """
    Collect historical data for multiple stock symbols
    
    Args:
        symbols (list): List of stock symbols ['AAPL', 'GOOGL', 'MSFT']
        timeframe (TimeFrame): Data interval
        start_date (datetime): Start date for data collection
        end_date (datetime): End date for data collection
    
    Returns:
        dict: Market data organized by symbol
    """
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        limit=1000  # Max 1000 bars per request
    )
    
    bars = client.get_stock_bars(request_params)
    return bars

# Example usage
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

# Get daily data
daily_data = get_stock_data(symbols, TimeFrame.Day, start_date, end_date)

# Get 5-minute data
minute_data = get_stock_data(symbols, TimeFrame(5, TimeFrameUnit.Minute), start_date, end_date)
```

### Converting to Pandas DataFrame
```python
# Convert to pandas DataFrame for easier analysis
df = daily_data.df

# Access data for specific symbol
aapl_data = daily_data['AAPL']
aapl_df = aapl_data.df

print(f"AAPL data shape: {aapl_df.shape}")
print(aapl_df.head())
```

## Scheduled Data Collection from Market Open

### Market Hours Detection
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

def is_market_open():
    """Check if market is currently open"""
    trading_client = TradingClient(
        api_key=os.getenv('ALPACA_API_KEY'), 
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper=True
    )
    
    clock = trading_client.get_clock()
    return clock.is_open

def get_market_hours(date):
    """Get market open/close times for a specific date"""
    trading_client = TradingClient(
        api_key=os.getenv('ALPACA_API_KEY'), 
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        paper=True
    )
    
    calendar_request = GetCalendarRequest(
        start=date,
        end=date
    )
    
    calendar = trading_client.get_calendar(calendar_request)
    if calendar:
        return calendar[0].open, calendar[0].close
    return None, None
```

### Scheduled Data Collection
```python
import time
import schedule
from datetime import datetime, timedelta

class MarketDataCollector:
    def __init__(self, symbols, interval_minutes=5):
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self.client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'), 
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )
    
    def collect_data_from_open(self):
        """Collect data from market open at specified intervals"""
        if not is_market_open():
            print("Market is closed")
            return
        
        # Get data from market open to now
        market_open, market_close = get_market_hours(datetime.now().date())
        
        if market_open:
            request_params = StockBarsRequest(
                symbol_or_symbols=self.symbols,
                timeframe=TimeFrame(self.interval_minutes, TimeFrameUnit.Minute),
                start=market_open,
                end=datetime.now(),
                limit=1000
            )
            
            try:
                bars = self.client.get_stock_bars(request_params)
                self.process_data(bars)
                print(f"Data collected at {datetime.now()}")
            except Exception as e:
                print(f"Error collecting data: {e}")
    
    def process_data(self, bars):
        """Process and store collected data"""
        df = bars.df
        
        # Save to CSV with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_data_{timestamp}.csv"
        df.to_csv(f"data/{filename}")
        
        # Or store in database, send to API, etc.
        print(f"Saved {len(df)} records to {filename}")

# Usage example
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
collector = MarketDataCollector(symbols, interval_minutes=5)

# Schedule data collection every 5 minutes during market hours
schedule.every(5).minutes.do(collector.collect_data_from_open)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
```

## Advanced Data Collection Examples

### Real-time Data Collection with WebSocket
```python
from alpaca.data.live import StockDataStream

def setup_live_data_stream(symbols):
    """Setup live data stream for real-time updates"""
    stream = StockDataStream(
        api_key=os.getenv('ALPACA_API_KEY'), 
        secret_key=os.getenv('ALPACA_SECRET_KEY')
    )
    
    @stream.on_bar(*symbols)
    async def on_bar_update(bar):
        print(f"Bar update for {bar.symbol}: {bar.close}")
        # Process real-time data here
    
    @stream.on_quote(*symbols)
    async def on_quote_update(quote):
        print(f"Quote update for {quote.symbol}: bid={quote.bid_price}, ask={quote.ask_price}")
    
    return stream

# Usage
symbols = ['AAPL', 'GOOGL', 'MSFT']
stream = setup_live_data_stream(symbols)
stream.run()
```

### Batch Data Collection for Historical Analysis
```python
def collect_historical_batch(symbols, start_date, end_date, timeframe=TimeFrame.Day):
    """
    Collect historical data in batches to handle large datasets
    """
    all_data = {}
    
    # Split symbols into batches of 100 (API limit)
    batch_size = 100
    symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    for batch in symbol_batches:
        request_params = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            limit=1000
        )
        
        try:
            bars = client.get_stock_bars(request_params)
            
            # Merge with existing data
            for symbol in batch:
                if symbol in bars:
                    all_data[symbol] = bars[symbol]
            
            # Rate limiting - avoid hitting API limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error collecting batch {batch}: {e}")
            continue
    
    return all_data

# Collect data for S&P 500 stocks
sp500_symbols = ['AAPL', 'GOOGL', 'MSFT']  # Add full S&P 500 list
historical_data = collect_historical_batch(
    sp500_symbols, 
    datetime(2023, 1, 1), 
    datetime(2024, 1, 1)
)
```

## Data Structure and Fields

The Alpaca API returns bar data with the following fields:
- **timestamp**: Time of the bar
- **open**: Opening price
- **high**: Highest price during the period
- **low**: Lowest price during the period
- **close**: Closing price
- **volume**: Number of shares traded
- **trade_count**: Number of trades
- **vwap**: Volume-weighted average price

## Error Handling and Best Practices

### Rate Limiting
```python
import time
from functools import wraps

def rate_limit(calls_per_minute=200):
    """Decorator to enforce rate limiting"""
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 60.0 / calls_per_minute - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(calls_per_minute=200)
def get_data_with_rate_limit(symbols, timeframe, start, end):
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end
    )
    return client.get_stock_bars(request_params)
```

### Retry Logic
```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    """Retry decorator for API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1)
def robust_data_collection(symbols, timeframe, start, end):
    return get_stock_data(symbols, timeframe, start, end)
```

## Data Storage Options

### CSV Storage
```python
def save_to_csv(data, filename):
    """Save market data to CSV file"""
    df = data.df
    df.to_csv(f"data/{filename}")
    print(f"Saved to {filename}")

# Usage
daily_data = get_stock_data(['AAPL', 'GOOGL'], TimeFrame.Day, start_date, end_date)
save_to_csv(daily_data, "daily_data_20240101.csv")
```

### Database Storage
```python
import sqlite3
import pandas as pd

def save_to_database(data, db_path="market_data.db"):
    """Save market data to SQLite database"""
    conn = sqlite3.connect(db_path)
    
    df = data.df
    df.to_sql('stock_bars', conn, if_exists='append', index=True)
    
    conn.close()
    print(f"Saved to database: {db_path}")
```

## Summary

The Alpaca API provides a powerful and flexible way to collect historical market data for multiple stock symbols at various time intervals. Key features include:

- Support for minute, daily, weekly, and monthly intervals
- Batch processing for multiple symbols
- Real-time data streaming capabilities  
- Built-in pandas integration
- Free access to crypto data, paid access for stock data
- Comprehensive error handling and rate limiting support

For production use, implement proper error handling, rate limiting, and data storage strategies based on your specific requirements.