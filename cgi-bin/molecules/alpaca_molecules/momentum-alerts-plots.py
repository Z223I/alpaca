#!/home/wilsonb/miniconda3/envs/alpaca/bin/python3
"""
Momentum Alerts Plots - Analyze max gains from momentum alerts.

Reads momentum alert JSON files, calculates max gain per symbol per day,
and generates multiple charts.

Usage:
    python3 momentum-alerts-plots.py
    python3 momentum-alerts-plots.py --start-date 2025-12-12 --end-date 2025-12-31
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.ndimage import gaussian_filter1d


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze max gains from momentum alerts'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-12-12',
        help='Start date in YYYY-MM-DD format (default: 2025-12-12)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date in YYYY-MM-DD format (default: today)'
    )
    return parser.parse_args()


def get_date_range(start_date_str, end_date_str):
    """Generate list of dates between start and end."""
    start = datetime.strptime(start_date_str, '%Y-%m-%d')
    end = datetime.strptime(end_date_str, '%Y-%m-%d')
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return dates


def collect_alerts_by_symbol(date_str, base_path):
    """Collect all alerts with prices and timestamps for each symbol on a given date."""
    symbol_alerts = defaultdict(list)
    alerts_path = os.path.join(
        base_path,
        'historical_data',
        date_str,
        'momentum_alerts_sent',
        'bullish',
        'alert_*.json'
    )

    for json_file in glob(alerts_path):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                symbol = data.get('symbol')
                price = data.get('current_price')
                timestamp_str = data.get('timestamp')
                if symbol and price is not None and timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    symbol_alerts[symbol].append({
                        'price': price,
                        'timestamp': timestamp
                    })
        except (json.JSONDecodeError, IOError, ValueError) as e:
            print(f"Warning: Could not read {json_file}: {e}")

    return symbol_alerts


def calculate_max_gains(symbol_alerts):
    """Calculate max gain for each symbol with timing info.

    Uses first alert (earliest timestamp) as starting price,
    and max price alert as ending price.
    """
    max_gains = []
    for symbol, alerts in symbol_alerts.items():
        if len(alerts) >= 2:
            # Use first alert (earliest timestamp) as starting point
            first_alert = min(alerts, key=lambda x: x['timestamp'])
            max_alert = max(alerts, key=lambda x: x['price'])
            first_price = first_alert['price']
            max_price = max_alert['price']

            if first_price > 0:
                max_gain = max_price / first_price
                first_time = first_alert['timestamp']
                max_time = max_alert['timestamp']

                max_gains.append({
                    'symbol': symbol,
                    'max_gain': max_gain,
                    'max_price': max_price,
                    'first_price': first_price,
                    'first_time': first_time,
                    'max_time': max_time,
                    'num_alerts': len(alerts)
                })
    return max_gains


def get_output_dir(base_path):
    """Get the output directory for plots."""
    output_dir = os.path.join(
        base_path,
        'historical_data',
        'momentum_alerts_sent'
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_histogram(all_max_gains, start_date, end_date, base_path):
    """Plot histogram of max gains binned by 10%."""
    if not all_max_gains:
        print("No data to plot.")
        return

    gains = [g['max_gain'] for g in all_max_gains]
    percent_gains = [(g - 1) * 100 for g in gains]

    max_percent = max(percent_gains)
    bins = list(range(0, int(max_percent) + 20, 10))

    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(percent_gains, bins=bins, edgecolor='black', alpha=0.7)

    # Add quantity above each column
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            plt.text(x, count, f'{int(count)}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Max Gain (%)')
    plt.ylabel('Quantity')
    plt.title(f'Distribution of Max Gains from Momentum Alerts\n{start_date} to {end_date}')
    plt.grid(axis='y', alpha=0.3)

    stats_text = f'Count: {len(percent_gains)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(get_output_dir(base_path), 'momo_max_gains_histogram.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histogram to {output_path}")


def plot_max_gain_per_day(all_max_gains, start_date, end_date, base_path):
    """Plot max gain per day (weekends excluded)."""
    if not all_max_gains:
        return

    # Group by date and find max gain per day
    daily_max = {}
    for g in all_max_gains:
        date = g['date']
        gain_pct = (g['max_gain'] - 1) * 100
        if date not in daily_max or gain_pct > daily_max[date]['gain']:
            daily_max[date] = {'gain': gain_pct, 'symbol': g['symbol']}

    dates = sorted(daily_max.keys())
    gains = [daily_max[d]['gain'] for d in dates]
    symbols = [daily_max[d]['symbol'] for d in dates]

    # Use date labels as categorical (removes weekend gaps)
    date_labels = [d[2:4] + '-' + d[5:] for d in dates]  # YY-MM-DD format

    plt.figure(figsize=(14, 6))
    x_pos = range(len(dates))
    bars = plt.bar(x_pos, gains, edgecolor='black', alpha=0.7)

    # Add quantity above each column
    for i, (bar, gain, symbol) in enumerate(zip(bars, gains, symbols)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{gain:.0f}%\n{symbol}', ha='center', va='bottom', fontsize=8)

    plt.xlabel('Date')
    plt.ylabel('Percent')
    plt.title(f'Maximum Gain Per Day\n{start_date} to {end_date}')
    plt.xticks(x_pos, date_labels, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(get_output_dir(base_path), 'momo_max_gain_per_day.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved max gain per day to {output_path}")


def plot_time_of_day(all_max_gains, start_date, end_date, base_path):
    """Plot time of day between first alert and max price for each stock-day (weekends excluded)."""
    if not all_max_gains:
        return

    plt.figure(figsize=(14, 10))

    # Sort by date for y-axis ordering
    sorted_gains = sorted(all_max_gains, key=lambda x: x['date'])

    # Get unique dates (trading days only) for categorical y-axis
    unique_dates = sorted(set(g['date'] for g in sorted_gains))
    date_to_y = {date: i for i, date in enumerate(unique_dates)}

    for g in sorted_gains:
        y_pos = date_to_y[g['date']]
        first_time = g['first_time']
        max_time = g['max_time']

        # Convert times to hours since midnight for x-axis
        first_hour = first_time.hour + first_time.minute / 60
        max_hour = max_time.hour + max_time.minute / 60

        # Draw line with dots at ends
        plt.plot([first_hour, max_hour], [y_pos, y_pos], 'b-', linewidth=1.5, alpha=0.7)
        plt.plot(first_hour, y_pos, 'go', markersize=6)  # Green dot for first alert time
        plt.plot(max_hour, y_pos, 'ro', markersize=6)  # Red dot for max price time

    plt.xlabel('Time of Day (Hour)')
    plt.ylabel('Date')
    plt.title(f'Time Range Between First Alert and Max Price\n{start_date} to {end_date}\n(Green=First Alert, Red=Max Price)')
    plt.yticks(range(len(unique_dates)), unique_dates)
    plt.xlim(4, 16)  # Market hours roughly 4 AM to 4 PM ET
    plt.xticks(range(4, 17), [f'{h}:00' for h in range(4, 17)], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(get_output_dir(base_path), 'momo_time_of_day.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved time of day chart to {output_path}")


def plot_duration(all_max_gains, start_date, end_date, base_path):
    """Plot histogram of duration between min and max price."""
    if not all_max_gains:
        return

    # Calculate durations in minutes
    durations = []
    for g in all_max_gains:
        duration = abs((g['max_time'] - g['first_time']).total_seconds() / 60)
        durations.append(duration)

    # Bin by 30 minute intervals
    max_duration = max(durations)
    bins = list(range(0, int(max_duration) + 60, 30))

    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(durations, bins=bins, edgecolor='black', alpha=0.7)

    # Add quantity above each column
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            plt.text(x, count, f'{int(count)}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Duration (minutes)')
    plt.ylabel('Quantity')
    plt.title(f'Duration Between Start and Max Price\n{start_date} to {end_date}')
    plt.xticks(bins, rotation=45)
    plt.grid(axis='y', alpha=0.3)

    stats_text = f'Count: {len(durations)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(get_output_dir(base_path), 'momo_duration.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved duration chart to {output_path}")


def plot_duration_15min(all_max_gains, start_date, end_date, base_path):
    """Plot histogram of duration between min and max price (15-minute bins, first 10 bins only)."""
    if not all_max_gains:
        return

    # Calculate durations in minutes
    durations = []
    for g in all_max_gains:
        duration = abs((g['max_time'] - g['first_time']).total_seconds() / 60)
        durations.append(duration)

    # Bin by 15 minute intervals, only first 10 bins (0-150 minutes)
    bins = list(range(0, 165, 15))  # 0, 15, 30, ..., 150

    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(durations, bins=bins, edgecolor='black', alpha=0.7)

    # Add quantity above each column
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            plt.text(x, count, f'{int(count)}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Duration (minutes)')
    plt.ylabel('Quantity')
    plt.title(f'Duration Between Start and Max Price (15-min bins)\n{start_date} to {end_date}')
    plt.xticks(bins, rotation=45)
    plt.grid(axis='y', alpha=0.3)

    stats_text = f'Count: {len(durations)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(get_output_dir(base_path), 'momo_duration_15min.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved duration (15-min bins) chart to {output_path}")


def plot_duration_5min(all_max_gains, start_date, end_date, base_path):
    """Plot histogram of duration between min and max price (5-minute bins, first 10 bins only)."""
    if not all_max_gains:
        return

    # Calculate durations in minutes
    durations = []
    for g in all_max_gains:
        duration = abs((g['max_time'] - g['first_time']).total_seconds() / 60)
        durations.append(duration)

    # Bin by 5 minute intervals, only first 10 bins (0-50 minutes)
    bins = list(range(0, 55, 5))  # 0, 5, 10, ..., 50

    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(durations, bins=bins, edgecolor='black', alpha=0.7)

    # Add quantity above each column
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            plt.text(x, count, f'{int(count)}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Duration (minutes)')
    plt.ylabel('Quantity')
    plt.title(f'Duration Between Start and Max Price (5-min bins)\n{start_date} to {end_date}')
    plt.xticks(bins, rotation=45)
    plt.grid(axis='y', alpha=0.3)

    stats_text = f'Count: {len(durations)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(get_output_dir(base_path), 'momo_duration_5min.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved duration (5-min bins) chart to {output_path}")


def plot_duration_5min_top_gainers(all_max_gains, start_date, end_date, base_path):
    """Plot histogram of duration (5-min bins, first 10 bins) for top gainer per day only."""
    if not all_max_gains:
        return

    # Find the top gainer for each day
    daily_top = {}
    for g in all_max_gains:
        date = g['date']
        if date not in daily_top or g['max_gain'] > daily_top[date]['max_gain']:
            daily_top[date] = g

    top_gainers = list(daily_top.values())

    # Calculate durations in minutes
    durations = []
    for g in top_gainers:
        duration = abs((g['max_time'] - g['first_time']).total_seconds() / 60)
        durations.append(duration)

    # Bin by 5 minute intervals, only first 10 bins (0-50 minutes)
    bins = list(range(0, 55, 5))  # 0, 5, 10, ..., 50

    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(durations, bins=bins, edgecolor='black', alpha=0.7)

    # Add quantity above each column
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            plt.text(x, count, f'{int(count)}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Duration (minutes)')
    plt.ylabel('Quantity')
    plt.title(f'Duration Between Start and Max Price (5-min bins, Top Gainer Per Day)\n{start_date} to {end_date}')
    plt.xticks(bins, rotation=45)
    plt.grid(axis='y', alpha=0.3)

    stats_text = f'Count: {len(durations)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(get_output_dir(base_path), 'momo_duration_5min_top_gainers.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved duration (5-min bins, top gainers) chart to {output_path}")


def plot_duration_30min_top_gainers(all_max_gains, start_date, end_date, base_path):
    """Plot histogram of duration (30-min bins, first 10 bins) for top gainer per day only."""
    if not all_max_gains:
        return

    # Find the top gainer for each day
    daily_top = {}
    for g in all_max_gains:
        date = g['date']
        if date not in daily_top or g['max_gain'] > daily_top[date]['max_gain']:
            daily_top[date] = g

    top_gainers = list(daily_top.values())

    # Calculate durations in minutes
    durations = []
    for g in top_gainers:
        duration = abs((g['max_time'] - g['first_time']).total_seconds() / 60)
        durations.append(duration)

    # Bin by 30 minute intervals, only first 10 bins (0-300 minutes)
    bins = list(range(0, 330, 30))  # 0, 30, 60, ..., 300

    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(durations, bins=bins, edgecolor='black', alpha=0.7)

    # Add quantity above each column
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            plt.text(x, count, f'{int(count)}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Duration (minutes)')
    plt.ylabel('Quantity')
    plt.title(f'Duration Between Start and Max Price (30-min bins, Top Gainer Per Day)\n{start_date} to {end_date}')
    plt.xticks(bins, rotation=45)
    plt.grid(axis='y', alpha=0.3)

    stats_text = f'Count: {len(durations)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(get_output_dir(base_path), 'momo_duration_30min_top_gainers.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved duration (30-min bins, top gainers) chart to {output_path}")


def plot_duration_vs_gain_scatter(all_max_gains, start_date, end_date, base_path):
    """Plot scatter chart of duration vs gain (duration 0-50 minutes)."""
    if not all_max_gains:
        return

    # Calculate durations and gains, filter to 0-50 minutes
    durations = []
    gains = []
    for g in all_max_gains:
        duration = abs((g['max_time'] - g['first_time']).total_seconds() / 60)
        if duration <= 50:
            durations.append(duration)
            gains.append((g['max_gain'] - 1) * 100)  # Convert to percentage

    plt.figure(figsize=(12, 6))
    plt.scatter(durations, gains, alpha=0.6, edgecolors='black', linewidths=0.5)

    plt.xlabel('Duration (minutes)')
    plt.ylabel('Gain (%)')
    plt.title(f'Duration vs Gain (0-50 minutes)\n{start_date} to {end_date}')
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3)

    stats_text = f'Count: {len(durations)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(get_output_dir(base_path), 'momo_duration_vs_gain_scatter.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved duration vs gain scatter chart to {output_path}")


def plot_duration_vs_gain_scatter_full(all_max_gains, start_date, end_date, base_path):
    """Plot scatter chart of duration vs gain (full duration range)."""
    if not all_max_gains:
        return

    # Calculate durations and gains
    durations = []
    gains = []
    for g in all_max_gains:
        duration = abs((g['max_time'] - g['first_time']).total_seconds() / 60)
        durations.append(duration)
        gains.append((g['max_gain'] - 1) * 100)  # Convert to percentage

    plt.figure(figsize=(12, 6))
    plt.scatter(durations, gains, alpha=0.6, edgecolors='black', linewidths=0.5)

    plt.xlabel('Duration (minutes)')
    plt.ylabel('Gain (%)')
    plt.title(f'Duration vs Gain (Full Range)\n{start_date} to {end_date}')
    plt.xlim(0, max(durations) + 10)
    plt.grid(True, alpha=0.3)

    stats_text = f'Count: {len(durations)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(get_output_dir(base_path), 'momo_duration_vs_gain_scatter_full.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved duration vs gain scatter chart (full range) to {output_path}")


def plot_time_distribution(all_max_gains, start_date, end_date, base_path):
    """Plot distribution of start (first alert) and end (max price) times with smoothed curves."""
    if not all_max_gains:
        return

    # Collect start times (first alert) and end times (max price)
    start_times = []
    end_times = []
    for g in all_max_gains:
        first_time = g['first_time']
        max_time = g['max_time']
        start_times.append(first_time.hour + first_time.minute / 60)
        end_times.append(max_time.hour + max_time.minute / 60)

    plt.figure(figsize=(12, 6))

    # Create histogram bins (30-minute intervals from 4 AM to 4 PM)
    bins = np.arange(4, 16.5, 0.5)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate histograms
    start_counts, _ = np.histogram(start_times, bins=bins)
    end_counts, _ = np.histogram(end_times, bins=bins)

    # Smooth the curves
    sigma = 1.5
    start_smooth = gaussian_filter1d(start_counts.astype(float), sigma)
    end_smooth = gaussian_filter1d(end_counts.astype(float), sigma)

    # Plot smoothed curves
    plt.plot(bin_centers, start_smooth, 'g-', linewidth=2.5, label='First Alert Time (Start)')
    plt.plot(bin_centers, end_smooth, 'r-', linewidth=2.5, label='Max Price Time (End)')

    # Add shaded area under curves
    plt.fill_between(bin_centers, start_smooth, alpha=0.3, color='green')
    plt.fill_between(bin_centers, end_smooth, alpha=0.3, color='red')

    plt.xlabel('Time of Day (Hour)')
    plt.ylabel('Quantity')
    plt.title(f'Distribution of First Alert/Max Price Times\n{start_date} to {end_date}')
    plt.xlim(4, 16)
    plt.xticks(range(4, 17), [f'{h}:00' for h in range(4, 17)], rotation=45)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    stats_text = f'Count: {len(all_max_gains)}'
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(get_output_dir(base_path), 'momo_time_distribution.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved time distribution chart to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Get the base path (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    print(f"Analyzing momentum alerts from {args.start_date} to {args.end_date}")
    print(f"Base path: {base_path}")

    dates = get_date_range(args.start_date, args.end_date)
    all_max_gains = []

    for date_str in dates:
        symbol_alerts = collect_alerts_by_symbol(date_str, base_path)
        if symbol_alerts:
            max_gains = calculate_max_gains(symbol_alerts)
            for gain in max_gains:
                gain['date'] = date_str
            all_max_gains.extend(max_gains)
            print(f"{date_str}: {len(max_gains)} symbols with multiple alerts")

    print(f"\nTotal symbol-days analyzed: {len(all_max_gains)}")

    if all_max_gains:
        # Print top 10 max gains
        sorted_gains = sorted(all_max_gains, key=lambda x: x['max_gain'], reverse=True)
        print("\nTop 10 Max Gains:")
        print("-" * 60)
        for i, g in enumerate(sorted_gains[:10], 1):
            gain_pct = (g['max_gain'] - 1) * 100
            print(f"{i:2}. {g['symbol']:6} on {g['date']}: {gain_pct:6.1f}% "
                  f"(${g['first_price']:.2f} -> ${g['max_price']:.2f})")

        # Generate all plots
        plot_histogram(all_max_gains, args.start_date, args.end_date, base_path)
        plot_max_gain_per_day(all_max_gains, args.start_date, args.end_date, base_path)
        plot_time_of_day(all_max_gains, args.start_date, args.end_date, base_path)
        plot_duration(all_max_gains, args.start_date, args.end_date, base_path)
        plot_duration_15min(all_max_gains, args.start_date, args.end_date, base_path)
        plot_duration_5min(all_max_gains, args.start_date, args.end_date, base_path)
        plot_duration_5min_top_gainers(all_max_gains, args.start_date, args.end_date, base_path)
        plot_duration_30min_top_gainers(all_max_gains, args.start_date, args.end_date, base_path)
        plot_duration_vs_gain_scatter(all_max_gains, args.start_date, args.end_date, base_path)
        plot_duration_vs_gain_scatter_full(all_max_gains, args.start_date, args.end_date, base_path)
        plot_time_distribution(all_max_gains, args.start_date, args.end_date, base_path)


if __name__ == '__main__':
    main()
