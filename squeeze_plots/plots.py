#!/usr/bin/env python3
"""
Generate charts for randomly selected squeeze alerts from historical data.

This script:
1. Randomly selects 10 squeeze alerts from historical_data/<YYYY-MM-DD>/squeeze_alerts_sent/
   between the hours 09:45 and 16:00
2. Uses code/alpaca.py to retrieve 1-minute candlesticks for the date, symbol, and up to the timestamp
3. Uses atoms/display/generate_chart_from_df.py to create the charts
4. Saves the charts to ./tmp/squeeze
"""

import sys
import os
import json
import random
import glob
import shutil
from datetime import datetime, time as dt_time
from pathlib import Path
import pytz
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_trade_api as tradeapi
from atoms.api.init_alpaca_client import init_alpaca_client
from atoms.display.generate_chart_from_df import generate_chart_from_dataframe


class SqueezeAlertPlotter:
    """Generate charts for historical squeeze alerts."""

    def __init__(self, output_dir: str = './tmp/squeeze'):
        """
        Initialize the plotter.

        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = output_dir
        self.et_tz = pytz.timezone('America/New_York')

        # Initialize Alpaca API client
        print("Initializing Alpaca API client...")
        self.api = init_alpaca_client("alpaca")

        # Clean up old output directory and create fresh one
        if os.path.exists(self.output_dir):
            print(f"Cleaning up old charts in {self.output_dir}...")
            shutil.rmtree(self.output_dir)

        os.makedirs(self.output_dir, exist_ok=True)

    def find_squeeze_alerts(self, start_hour: int = 9, start_minute: int = 45,
                           end_hour: int = 16, end_minute: int = 0,
                           min_price: float = 2.0, max_price: float = 10.0) -> list:
        """
        Find all squeeze alert JSON files within specified time range and price range.

        Args:
            start_hour: Start hour filter (default: 9)
            start_minute: Start minute filter (default: 45)
            end_hour: End hour filter (default: 16)
            end_minute: End minute filter (default: 0)
            min_price: Minimum stock price (default: 2.0)
            max_price: Maximum stock price (default: 10.0)

        Returns:
            List of tuples: (file_path, alert_data)
        """
        print(f"\nSearching for squeeze alerts:")
        print(f"  Time range: {start_hour:02d}:{start_minute:02d} to {end_hour:02d}:{end_minute:02d}")
        print(f"  Price range: ${min_price:.2f} to ${max_price:.2f}")

        # Find all alert files
        alert_pattern = "historical_data/*/squeeze_alerts_sent/alert_*.json"
        alert_files = glob.glob(alert_pattern)

        print(f"Found {len(alert_files)} total squeeze alert files")

        # Filter by time range and price range
        valid_alerts = []
        start_time = dt_time(start_hour, start_minute)
        end_time = dt_time(end_hour, end_minute)

        for alert_file in alert_files:
            try:
                with open(alert_file, 'r') as f:
                    alert_data = json.load(f)

                # Parse timestamp
                timestamp_str = alert_data.get('timestamp')
                if not timestamp_str:
                    continue

                # Parse the timestamp (format: "2025-12-16T10:01:35.820779-05:00")
                alert_time = datetime.fromisoformat(timestamp_str)
                alert_time_only = alert_time.time()

                # Check if within time range
                if not (start_time <= alert_time_only <= end_time):
                    continue

                # Check if within price range
                # Squeeze alerts use 'last_price' instead of 'current_price'
                current_price = alert_data.get('current_price') or alert_data.get('last_price')
                if current_price is None:
                    continue

                if not (min_price <= current_price <= max_price):
                    continue

                valid_alerts.append((alert_file, alert_data))

            except Exception as e:
                print(f"Warning: Could not process {alert_file}: {e}")
                continue

        print(f"Found {len(valid_alerts)} alerts matching filters (time + price)")
        return valid_alerts

    def check_volume_criteria(self, symbol: str, alert_time: datetime, min_avg_volume: float = 80000) -> tuple:
        """
        Check if the average volume in the 10 minutes before the alert meets the minimum.

        Args:
            symbol: Stock symbol
            alert_time: Time of the alert
            min_avg_volume: Minimum average volume required (default: 80000)

        Returns:
            Tuple of (meets_criteria: bool, avg_volume: float or None)
        """
        try:
            # Get data for 10 minutes before the alert
            end_time = alert_time
            start_time = alert_time - pd.Timedelta(minutes=10)

            # Format as RFC3339 with proper timezone format
            start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S') + start_time.strftime('%z')[:3] + ':' + start_time.strftime('%z')[3:]
            end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S') + end_time.strftime('%z')[:3] + ':' + end_time.strftime('%z')[3:]

            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Minute,
                start=start_str,
                end=end_str,
                limit=10,
                feed='sip'
            )

            if not bars or len(bars) == 0:
                return (False, None)

            # Calculate average volume
            volumes = [int(bar.v) for bar in bars]
            avg_volume = sum(volumes) / len(volumes)

            return (avg_volume >= min_avg_volume, avg_volume)

        except Exception as e:
            # If we can't get the data, exclude this alert
            return (False, None)

    def filter_alerts_by_volume(self, alerts: list, min_avg_volume: float = 80000) -> list:
        """
        Filter alerts based on average volume in the 10 minutes before the alert.

        Args:
            alerts: List of (file_path, alert_data) tuples
            min_avg_volume: Minimum average volume required (default: 80000)

        Returns:
            Filtered list of alerts that meet volume criteria
        """
        print(f"\nFiltering alerts by volume (avg > {min_avg_volume:,.0f} in 10 min before alert)...")

        filtered_alerts = []
        checked_count = 0

        for alert_file, alert_data in alerts:
            checked_count += 1

            # Show progress every 50 alerts
            if checked_count % 50 == 0:
                print(f"  Checked {checked_count}/{len(alerts)} alerts... (found {len(filtered_alerts)} matching)")

            # Parse alert time
            timestamp_str = alert_data.get('timestamp')
            if not timestamp_str:
                continue

            alert_time = datetime.fromisoformat(timestamp_str)
            symbol = alert_data.get('symbol')

            # Check volume criteria
            meets_criteria, avg_volume = self.check_volume_criteria(symbol, alert_time, min_avg_volume)

            if meets_criteria:
                filtered_alerts.append((alert_file, alert_data))

        print(f"  Volume filter complete: {len(filtered_alerts)} alerts passed (from {len(alerts)})")
        return filtered_alerts

    def select_random_alerts(self, alerts: list, count: int = 10) -> list:
        """
        Randomly select alerts from the list.

        Args:
            alerts: List of (file_path, alert_data) tuples
            count: Number of alerts to select

        Returns:
            List of randomly selected alerts
        """
        if len(alerts) <= count:
            print(f"Selecting all {len(alerts)} available alerts")
            return alerts

        selected = random.sample(alerts, count)
        print(f"Randomly selected {count} alerts from {len(alerts)} available")
        return selected

    def get_market_data(self, symbol: str, alert_date: datetime, alert_time: datetime) -> pd.DataFrame:
        """
        Retrieve 1-minute candlestick data from market open up to alert time + 10 minutes.

        Args:
            symbol: Stock symbol
            alert_date: Date of the alert
            alert_time: Time of the alert

        Returns:
            DataFrame with OHLCV data
        """
        # Set up time range (4:00 AM to alert time + 10 minutes, Eastern Time)
        start_time = datetime.combine(alert_date.date(), dt_time(4, 0), tzinfo=self.et_tz)
        end_time = alert_time + pd.Timedelta(minutes=10)

        print(f"  Fetching market data from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} (alert + 10 min)...")

        try:
            # Format as RFC3339 with proper timezone format
            start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S') + start_time.strftime('%z')[:3] + ':' + start_time.strftime('%z')[3:]
            end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S') + end_time.strftime('%z')[:3] + ':' + end_time.strftime('%z')[3:]

            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Minute,
                start=start_str,
                end=end_str,
                limit=10000,
                feed='sip'  # Use SIP feed for consolidated data
            )
        except Exception as e:
            print(f"  Error fetching market data: {e}")
            return None

        if not bars:
            print(f"  No market data available for {symbol}")
            return None

        # Convert bars to DataFrame
        market_data = []
        for bar in bars:
            bar_data = {
                'timestamp': bar.t.isoformat(),
                'open': float(bar.o),
                'high': float(bar.h),
                'low': float(bar.l),
                'close': float(bar.c),
                'volume': int(bar.v),
                'symbol': symbol
            }
            market_data.append(bar_data)

        df = pd.DataFrame(market_data)
        print(f"  Retrieved {len(df)} minutes of data")
        return df

    def generate_chart(self, df: pd.DataFrame, symbol: str, alert_data: dict, alert_file: str) -> bool:
        """
        Generate chart for the alert.

        Args:
            df: DataFrame with market data
            symbol: Stock symbol
            alert_data: Alert data dictionary
            alert_file: Path to alert file (for naming)

        Returns:
            True if successful, False otherwise
        """
        if df is None or df.empty:
            print(f"  Skipping chart generation - no data available")
            return False

        # Parse the squeeze alert timestamp
        timestamp = datetime.fromisoformat(alert_data['timestamp'])

        # Create alert info for chart - add timestamp_dt field for plotting
        alert_for_chart = alert_data.copy()
        alert_for_chart['timestamp_dt'] = timestamp
        alert_for_chart['alert_type'] = 'squeeze'  # Mark as squeeze alert
        alert_for_chart['alert_level'] = 'squeeze'  # Set level for color coding

        alerts = [alert_for_chart]

        print(f"  Generating chart with squeeze alert at {timestamp.strftime('%H:%M:%S')}...")

        try:
            success = generate_chart_from_dataframe(
                df=df,
                symbol=symbol,
                output_dir=self.output_dir,
                alerts=alerts,
                verbose=False
            )

            if success:
                print(f"  Chart generated successfully")
                return True
            else:
                print(f"  Chart generation failed")
                return False

        except Exception as e:
            print(f"  Error generating chart: {e}")
            return False

    def process_alert(self, alert_file: str, alert_data: dict) -> bool:
        """
        Process a single alert: fetch data and generate chart.

        Args:
            alert_file: Path to alert JSON file
            alert_data: Parsed alert data

        Returns:
            True if successful, False otherwise
        """
        symbol = alert_data.get('symbol')
        timestamp_str = alert_data.get('timestamp')

        if not symbol or not timestamp_str:
            print(f"Skipping invalid alert: missing symbol or timestamp")
            return False

        # Parse timestamp
        alert_time = datetime.fromisoformat(timestamp_str)

        # Get price from either current_price or last_price field
        price = alert_data.get('current_price') or alert_data.get('last_price', 'N/A')
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)

        print(f"\nProcessing alert: {symbol} at {alert_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  Alert file: {alert_file}")
        print(f"  Price: {price_str}")

        # Show additional info if available
        if 'percent_gain_since_market_open' in alert_data:
            print(f"  Gain since open: {alert_data.get('percent_gain_since_market_open', 'N/A')}%")
        elif 'percent_change' in alert_data:
            print(f"  Percent change: {alert_data.get('percent_change', 'N/A'):.2f}%")

        # Get market data
        df = self.get_market_data(symbol, alert_time, alert_time)

        if df is None or df.empty:
            return False

        # Generate chart
        return self.generate_chart(df, symbol, alert_data, alert_file)

    def run(self, num_alerts: int = 10, min_price: float = 2.0, max_price: float = 10.0,
            min_avg_volume: float = 80000) -> None:
        """
        Main execution: find, select, and process alerts.

        Args:
            num_alerts: Number of alerts to process
            min_price: Minimum stock price filter (default: 2.0)
            max_price: Maximum stock price filter (default: 10.0)
            min_avg_volume: Minimum average volume in 10 min before alert (default: 80000)
        """
        print("="*80)
        print("SQUEEZE ALERT CHART GENERATOR")
        print("="*80)

        # Find all valid alerts (09:45 - 16:00, price range $2-$10)
        alerts = self.find_squeeze_alerts(
            start_hour=9, start_minute=45,
            end_hour=16, end_minute=0,
            min_price=min_price,
            max_price=max_price
        )

        if not alerts:
            print("\nNo alerts found matching time and price filters!")
            return

        # Filter by volume criteria
        alerts = self.filter_alerts_by_volume(alerts, min_avg_volume)

        if not alerts:
            print("\nNo alerts found matching all filters (time + price + volume)!")
            return

        # Randomly select alerts
        selected_alerts = self.select_random_alerts(alerts, num_alerts)

        # Process each alert
        print(f"\n{'='*80}")
        print(f"PROCESSING {len(selected_alerts)} ALERTS")
        print(f"{'='*80}")

        success_count = 0
        for i, (alert_file, alert_data) in enumerate(selected_alerts, 1):
            print(f"\n[{i}/{len(selected_alerts)}] ", end="")

            if self.process_alert(alert_file, alert_data):
                success_count += 1

        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total alerts processed: {len(selected_alerts)}")
        print(f"Charts generated successfully: {success_count}")
        print(f"Failed: {len(selected_alerts) - success_count}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nDone!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate charts for randomly selected squeeze alerts'
    )
    parser.add_argument(
        '-n', '--num-alerts',
        type=int,
        default=10,
        help='Number of alerts to process (default: 10)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./tmp/squeeze',
        help='Output directory for charts (default: ./tmp/squeeze)'
    )
    parser.add_argument(
        '--min-price',
        type=float,
        default=2.0,
        help='Minimum stock price filter (default: 2.0)'
    )
    parser.add_argument(
        '--max-price',
        type=float,
        default=10.0,
        help='Maximum stock price filter (default: 10.0)'
    )
    parser.add_argument(
        '--min-avg-volume',
        type=float,
        default=80000,
        help='Minimum average volume in 10 min before alert (default: 80000)'
    )

    args = parser.parse_args()

    # Create plotter and run
    plotter = SqueezeAlertPlotter(output_dir=args.output_dir)
    plotter.run(num_alerts=args.num_alerts, min_price=args.min_price, max_price=args.max_price,
                min_avg_volume=args.min_avg_volume)


if __name__ == "__main__":
    main()
