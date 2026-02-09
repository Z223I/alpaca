#!/home/wilsonb/miniconda3/envs/alpaca/bin/python3
"""
Max Gainer - Find the symbol with the highest gain for a given date.

Uses momentum alert data to calculate max gains per symbol and returns
the top gainer.

Usage:
    python3 max_gainer.py                           # Today's max gainer
    python3 max_gainer.py --date 2025-01-15         # Specific date
    python3 max_gainer.py --period 30               # Last 30 minutes
    python3 max_gainer.py --date 2025-01-15 --period 60
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

# Add cgi-bin to path for atoms imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from atoms.alpaca_api.momentum_alerts_atoms import (
    collect_alerts_by_symbol,
    calculate_max_gains
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Find the symbol with the highest gain for a given date'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Date in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--period',
        type=int,
        default=None,
        help='Time window in minutes to examine (from now, going back)'
    )
    return parser.parse_args()


def filter_alerts_by_period(symbol_alerts, period_minutes):
    """Filter alerts to only those within the specified time window.

    Args:
        symbol_alerts: dict from collect_alerts_by_symbol()
        period_minutes: Number of minutes to look back from now

    Returns:
        dict: Filtered symbol_alerts with only recent alerts
    """
    cutoff_time = datetime.now() - timedelta(minutes=period_minutes)
    filtered = {}

    for symbol, alerts in symbol_alerts.items():
        recent_alerts = [
            alert for alert in alerts
            if alert['timestamp'] >= cutoff_time
        ]
        if recent_alerts:
            filtered[symbol] = recent_alerts

    return filtered


def main():
    """Main entry point."""
    args = parse_args()

    # Get the base path (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    # Collect alerts for the date
    symbol_alerts = collect_alerts_by_symbol(args.date, base_path)

    if not symbol_alerts:
        print(f"No alerts found for {args.date}")
        sys.exit(1)

    # Filter by period if specified
    if args.period:
        symbol_alerts = filter_alerts_by_period(symbol_alerts, args.period)
        if not symbol_alerts:
            print(f"No alerts found in the last {args.period} minutes")
            sys.exit(1)

    # Calculate max gains
    max_gains = calculate_max_gains(symbol_alerts)

    if not max_gains:
        print(f"No symbols with multiple alerts found for {args.date}")
        sys.exit(1)

    # Sort by gain descending
    sorted_gains = sorted(max_gains, key=lambda x: x['max_gain'], reverse=True)

    # Print symbols with gain >= 10% in descending order
    for entry in sorted_gains:
        gain_pct = (entry['max_gain'] - 1) * 100
        if gain_pct >= 10:
            print(f"{entry['symbol']} {gain_pct:.1f}% "
                  f"(${entry['first_price']:.2f} -> ${entry['max_price']:.2f})")


if __name__ == '__main__':
    main()
