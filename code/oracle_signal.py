#!/usr/bin/env python3

import os
import csv
import glob
from datetime import datetime


def get_current_date_str():
    """Get current date in YYYYMMDD format"""
    return datetime.now().strftime("%Y%m%d")


def get_current_date_dash():
    """Get current date in YYYY-MM-DD format"""
    return datetime.now().strftime("%Y-%m-%d")


def read_signal_data(date_str):
    """Read the data/YYYYMMDD.csv file and return symbol data with signal and resistance"""
    data_file = f"data/{date_str}.csv"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    signals = {}
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row['Symbol']
            try:
                signal = float(row['Signal'])
                resistance = float(row['Resistance'])
                signals[symbol] = {
                    'signal': signal,
                    'resistance': resistance
                }
            except (ValueError, TypeError):
                continue

    return signals


def get_most_recent_market_data_file(symbol, date_dash):
    """Get the most recent historical market data file for a symbol"""
    pattern = f"historical_data/{date_dash}/market_data/{symbol}_*_*.csv"
    files = glob.glob(pattern)

    if not files:
        return None

    # Sort by filename (which includes timestamp) to get most recent
    files.sort()
    return files[-1]


def get_most_recent_low(market_data_file):
    """Get the most recent 'low' value from the market data file"""
    if not market_data_file or not os.path.exists(market_data_file):
        return None

    try:
        with open(market_data_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                return None

            # Get the last row (most recent data)
            last_row = rows[-1]
            return float(last_row['low'])

    except (ValueError, TypeError, KeyError):
        return None



def main():
    """Main function to compare signals with recent lows"""
    date_str = get_current_date_str()
    date_dash = get_current_date_dash()

    try:
        # Read signal data
        signals = read_signal_data(date_str)

        # Results list
        results = []

        # Process each symbol
        for symbol, data in signals.items():
            signal = data['signal']
            resistance = data['resistance']

            # Get most recent market data file
            market_file = get_most_recent_market_data_file(symbol, date_dash)

            if market_file:
                # Get most recent low value
                recent_low = get_most_recent_low(market_file)

                if recent_low is not None:
                    # Check if low is at 95% or greater of signal value
                    threshold = signal * 0.95
                    if recent_low >= threshold:
                        results.append({
                            'symbol': symbol,
                            'signal': signal,
                            'resistance': resistance,
                            'low': recent_low
                        })

        # Sort results by symbol name
        results.sort(key=lambda x: x['symbol'])

        # Print results in CSV format
        print("Symbol,Signal,Resistance,Low")
        for result in results:
            print(f"{result['symbol']},{result['signal']:.2f},{result['resistance']:.2f},{result['low']:.2f}")

        # Print summary
        if results:
            print(f"\n📊 Found {len(results)} symbols approaching signal threshold")
        else:
            print("No symbols found approaching signal threshold (95%+)")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
