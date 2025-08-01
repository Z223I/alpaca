#!/usr/bin/env python3
"""
Count superduper alerts with high momentum scores for the latest available date.
Includes binning and plotting of momentum distribution.
"""
import json
import glob
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np

def find_latest_date():
    """Find the latest date directory with superduper alerts"""
    # Get script directory and construct relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    historical_data_dir = os.path.join(repo_root, "historical_data")
    
    if not os.path.exists(historical_data_dir):
        return None
    
    # Get all date directories
    date_dirs = []
    for item in os.listdir(historical_data_dir):
        item_path = os.path.join(historical_data_dir, item)
        if os.path.isdir(item_path) and item.startswith('2025-'):
            # Check if it has superduper alerts
            alerts_path = os.path.join(item_path, 'superduper_alerts', 'bullish')
            if os.path.exists(alerts_path) and os.listdir(alerts_path):
                date_dirs.append(item)
    
    if not date_dirs:
        return None
    
    # Return the latest date
    return sorted(date_dirs)[-1]

def analyze_momentum_alerts(date=None, threshold=0.5, verbose=False, plot=False, bins=20):
    """Analyze superduper alerts momentum scores with binning and optional plotting"""
    
    # Use latest date if none specified
    if date is None:
        date = find_latest_date()
        if date is None:
            print("No superduper alerts found in historical data")
            return 0, []
    
    # Get script directory and construct relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    alerts_dir = os.path.join(repo_root, "historical_data", date, "superduper_alerts", "bullish")
    
    if not os.path.exists(alerts_dir):
        print(f"No alerts directory found for {date}: {alerts_dir}")
        return 0, []
    
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(alerts_dir, "*.json"))
    
    total_alerts = len(json_files)
    high_momentum_count = 0
    momentum_scores = []
    alert_details = []
    
    print(f"Found {total_alerts} superduper alerts for {date}")
    print(f"Analyzing momentum scores (threshold >= {threshold})...")
    
    # Process each alert file
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                alert_data = json.load(f)
            
            # Extract momentum score from enhanced_metrics
            momentum_score = alert_data.get('enhanced_metrics', {}).get('momentum_score', 0)
            momentum_scores.append(momentum_score)
            
            symbol = alert_data.get('symbol', 'UNKNOWN')
            timestamp = alert_data.get('timestamp', 'UNKNOWN')
            alert_details.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'momentum': momentum_score
            })
            
            if momentum_score >= threshold:
                high_momentum_count += 1
                if verbose:
                    print(f"  ✓ {symbol} at {timestamp}: momentum = {momentum_score:.3f}")
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {file_path}: {e}")
    
    # Print statistics
    if momentum_scores:
        print(f"\nMomentum Score Statistics:")
        print(f"  Total alerts: {total_alerts}")
        print(f"  High momentum alerts (>= {threshold}): {high_momentum_count}")
        print(f"  Min momentum: {min(momentum_scores):.3f}")
        print(f"  Max momentum: {max(momentum_scores):.3f}")
        print(f"  Mean momentum: {np.mean(momentum_scores):.3f}")
        print(f"  Median momentum: {np.median(momentum_scores):.3f}")
        print(f"  Std deviation: {np.std(momentum_scores):.3f}")
        
        # Create binned distribution
        hist, bin_edges = np.histogram(momentum_scores, bins=bins)
        print(f"\nMomentum Distribution ({bins} bins):")
        for i, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
            next_edge = bin_edges[i + 1]
            percentage = (count / total_alerts) * 100
            print(f"  {edge:.2f} - {next_edge:.2f}: {count:2d} alerts ({percentage:4.1f}%)")
        
        # Plot if requested
        if plot:
            create_momentum_plot(momentum_scores, date, threshold, bins)
    
    return high_momentum_count, momentum_scores

def create_momentum_plot(momentum_scores, date, threshold, bins):
    """Create and save momentum distribution plot"""
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    n, bins_array, patches = plt.hist(momentum_scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Color bars above threshold differently
    for i, (patch, bin_edge) in enumerate(zip(patches, bins_array[:-1])):
        if bin_edge >= threshold:
            patch.set_facecolor('lightcoral')
    
    # Add threshold line
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold}')
    
    # Add statistics text
    mean_momentum = np.mean(momentum_scores)
    median_momentum = np.median(momentum_scores)
    high_count = sum(1 for score in momentum_scores if score >= threshold)
    
    stats_text = f'Total Alerts: {len(momentum_scores)}\n'
    stats_text += f'High Momentum (≥{threshold}): {high_count}\n'
    stats_text += f'Mean: {mean_momentum:.3f}\n'
    stats_text += f'Median: {median_momentum:.3f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    plt.xlabel('Momentum Score', fontsize=12)
    plt.ylabel('Number of Alerts', fontsize=12)
    plt.title(f'Superduper Alerts Momentum Distribution - {date}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create output directory and save plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    output_dir = os.path.join(repo_root, "historical_data", date, "alerts", "summary")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = os.path.join(output_dir, f'momentum_distribution_{date.replace("-", "")}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    plt.show()

def count_high_momentum_alerts(date=None, threshold=0.5, verbose=False):
    """Legacy function for backward compatibility"""
    count, _ = analyze_momentum_alerts(date, threshold, verbose, plot=False)
    return count

def main():
    parser = argparse.ArgumentParser(description='Analyze superduper alerts momentum distribution')
    parser.add_argument('--date', help='Date to analyze (YYYY-MM-DD format, defaults to latest available)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Minimum momentum threshold (default: 0.5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed alert information')
    parser.add_argument('--plot', '-p', action='store_true', help='Generate and save momentum distribution plot')
    parser.add_argument('--bins', type=int, default=20, help='Number of bins for histogram (default: 20)')
    
    args = parser.parse_args()
    
    analyze_momentum_alerts(args.date, args.threshold, args.verbose, args.plot, args.bins)

if __name__ == "__main__":
    main()