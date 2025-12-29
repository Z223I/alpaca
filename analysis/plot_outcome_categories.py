#!/usr/bin/env python3
"""
Plot Outcome Categories from Model Training

Visualizes the distribution of trade outcome categories from trained models.
Categories include:
  1. Target hit first (successful trades)
  2. Stop hit first (stopped out trades)
  3. Both hit (OCO decision via timestamps)
  4. Neither hit (ended without trigger)
  5. Fallback (no interval data)

Usage:
    # Plot single threshold
    python analysis/plot_outcome_categories.py --threshold 6.0

    # Plot all available thresholds
    python analysis/plot_outcome_categories.py --all

    # Plot specific trailing stop
    python analysis/plot_outcome_categories.py --threshold 6.0 --trailing-stop 2.0

    # Compare multiple thresholds
    python analysis/plot_outcome_categories.py --compare-thresholds
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =============================================================================
# Data Loading
# =============================================================================

def load_model_info(threshold: float, trailing_stop_pct: float = 2.0) -> Optional[Dict]:
    """
    Load model info JSON file for a given threshold.

    Args:
        threshold: Gain threshold percentage
        trailing_stop_pct: Trailing stop percentage

    Returns:
        Model info dictionary or None if not found
    """
    models_dir = Path("analysis/tuned_models")
    info_file = models_dir / f"xgboost_tuned_{threshold}pct_stop{trailing_stop_pct}_info.json"

    if not info_file.exists():
        print(f"⚠️  Model info not found: {info_file}")
        return None

    with open(info_file, 'r') as f:
        return json.load(f)


def find_all_model_infos() -> List[Tuple[float, float, Dict]]:
    """
    Find all model info files in the tuned_models directory.

    Returns:
        List of tuples (threshold, trailing_stop, info_dict)
    """
    models_dir = Path("analysis/tuned_models")
    if not models_dir.exists():
        print(f"⚠️  Models directory not found: {models_dir}")
        return []

    model_infos = []
    for info_file in models_dir.glob("xgboost_tuned_*_info.json"):
        # Parse filename: xgboost_tuned_6.0pct_stop2.0_info.json
        filename = info_file.stem  # Remove .json
        parts = filename.split('_')

        try:
            # Extract threshold
            threshold_str = [p for p in parts if 'pct' in p and 'stop' not in p][0]
            threshold = float(threshold_str.replace('pct', ''))

            # Extract trailing stop
            stop_str = [p for p in parts if 'stop' in p][0]
            trailing_stop = float(stop_str.replace('stop', ''))

            with open(info_file, 'r') as f:
                info = json.load(f)

            model_infos.append((threshold, trailing_stop, info))
        except (IndexError, ValueError) as e:
            print(f"⚠️  Could not parse filename {info_file.name}: {e}")
            continue

    return sorted(model_infos, key=lambda x: (x[0], x[1]))  # Sort by threshold, then trailing_stop


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_single_pie_chart(
    category_stats: Dict,
    threshold: float,
    trailing_stop_pct: float,
    output_dir: Path = Path("analysis/plots")
) -> Path:
    """
    Create pie chart showing distribution of outcome categories.

    Args:
        category_stats: Category statistics dictionary
        threshold: Gain threshold
        trailing_stop_pct: Trailing stop percentage
        output_dir: Output directory for plot

    Returns:
        Path to saved plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    categories = ['Target Hit First', 'Stop Hit First', 'Both Hit\n(OCO)', 'Neither Hit', 'Fallback']
    counts = [
        category_stats['target_hit_first']['count'],
        category_stats['stop_hit_first']['count'],
        category_stats['both_hit_oco_decision']['count'],
        category_stats['neither_hit']['count'],
        category_stats['fallback_no_interval_data']['count'],
    ]
    percentages = [
        category_stats['target_hit_first']['percentage'],
        category_stats['stop_hit_first']['percentage'],
        category_stats['both_hit_oco_decision']['percentage'],
        category_stats['neither_hit']['percentage'],
        category_stats['fallback_no_interval_data']['percentage'],
    ]

    # Colors: green for success, red for failures, yellow for mixed, gray for fallback
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6', '#bdc3c7']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=categories,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )

    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_weight('bold')

    # Add title
    title = f'Trade Outcome Categories\n{threshold}% Threshold, {trailing_stop_pct}% Trailing Stop'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add legend with counts
    legend_labels = [
        f'{cat}: {count:,} ({pct:.1f}%)'
        for cat, count, pct in zip(categories, counts, percentages)
    ]
    ax.legend(
        legend_labels,
        loc='center left',
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10
    )

    # Add total in center
    total = category_stats['total_trades']
    ax.text(
        0, 0, f'Total\n{total:,}\ntrades',
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / f"outcome_categories_pie_{threshold}pct_stop{trailing_stop_pct}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_stacked_bar(
    category_stats: Dict,
    threshold: float,
    trailing_stop_pct: float,
    output_dir: Path = Path("analysis/plots")
) -> Path:
    """
    Create stacked bar chart showing outcome categories.

    Args:
        category_stats: Category statistics dictionary
        threshold: Gain threshold
        trailing_stop_pct: Trailing stop percentage
        output_dir: Output directory for plot

    Returns:
        Path to saved plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    categories = ['Target Hit\nFirst', 'Stop Hit\nFirst', 'Both Hit\n(OCO)', 'Neither\nHit', 'Fallback']
    counts = [
        category_stats['target_hit_first']['count'],
        category_stats['stop_hit_first']['count'],
        category_stats['both_hit_oco_decision']['count'],
        category_stats['neither_hit']['count'],
        category_stats['fallback_no_interval_data']['count'],
    ]

    # Colors
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6', '#bdc3c7']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create stacked bar
    bottom = 0
    for i, (cat, count, color) in enumerate(zip(categories, counts, colors)):
        pct = count / category_stats['total_trades'] * 100
        ax.barh(0, count, left=bottom, color=color, label=cat, height=0.5)

        # Add text label
        if count > 0:
            ax.text(
                bottom + count/2, 0,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white'
            )
        bottom += count

    # Formatting
    ax.set_xlim(0, category_stats['total_trades'])
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Number of Trades', fontsize=12, fontweight='bold')
    title = f'Trade Outcome Categories - {threshold}% Threshold, {trailing_stop_pct}% Trailing Stop'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=10)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_file = output_dir / f"outcome_categories_bar_{threshold}pct_stop{trailing_stop_pct}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_comparison_across_thresholds(
    model_infos: List[Tuple[float, float, Dict]],
    output_dir: Path = Path("analysis/plots")
) -> Path:
    """
    Create comparison plot across multiple thresholds.

    Args:
        model_infos: List of (threshold, trailing_stop, info_dict) tuples
        output_dir: Output directory for plot

    Returns:
        Path to saved plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    thresholds = []
    target_hit = []
    stop_hit = []
    both_hit = []
    neither_hit = []
    fallback = []

    for threshold, trailing_stop, info in model_infos:
        if 'outcome_categories' not in info or info['outcome_categories'] is None:
            continue

        cat_stats = info['outcome_categories']
        thresholds.append(f"{threshold}%\n({trailing_stop}% stop)")
        target_hit.append(cat_stats['target_hit_first']['percentage'])
        stop_hit.append(cat_stats['stop_hit_first']['percentage'])
        both_hit.append(cat_stats['both_hit_oco_decision']['percentage'])
        neither_hit.append(cat_stats['neither_hit']['percentage'])
        fallback.append(cat_stats['fallback_no_interval_data']['percentage'])

    if not thresholds:
        print("⚠️  No outcome categories data found in model info files")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Bar positions
    x = np.arange(len(thresholds))
    width = 0.15

    # Create grouped bars
    ax.bar(x - 2*width, target_hit, width, label='Target Hit First', color='#2ecc71')
    ax.bar(x - width, stop_hit, width, label='Stop Hit First', color='#e74c3c')
    ax.bar(x, both_hit, width, label='Both Hit (OCO)', color='#f39c12')
    ax.bar(x + width, neither_hit, width, label='Neither Hit', color='#95a5a6')
    ax.bar(x + 2*width, fallback, width, label='Fallback', color='#bdc3c7')

    # Formatting
    ax.set_xlabel('Threshold (Trailing Stop)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Trades (%)', fontsize=12, fontweight='bold')
    ax.set_title('Outcome Categories Comparison Across Thresholds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    plt.tight_layout()

    # Save
    output_file = output_dir / "outcome_categories_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_detailed_breakdown(
    category_stats: Dict,
    threshold: float,
    trailing_stop_pct: float,
    output_dir: Path = Path("analysis/plots")
) -> Path:
    """
    Create detailed breakdown with multiple subplots.

    Args:
        category_stats: Category statistics dictionary
        threshold: Gain threshold
        trailing_stop_pct: Trailing stop percentage
        output_dir: Output directory for plot

    Returns:
        Path to saved plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Pie chart (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Target Hit\nFirst', 'Stop Hit\nFirst', 'Both Hit\n(OCO)', 'Neither\nHit', 'Fallback']
    counts = [
        category_stats['target_hit_first']['count'],
        category_stats['stop_hit_first']['count'],
        category_stats['both_hit_oco_decision']['count'],
        category_stats['neither_hit']['count'],
        category_stats['fallback_no_interval_data']['count'],
    ]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6', '#bdc3c7']
    wedges, texts, autotexts = ax1.pie(counts, labels=categories, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    ax1.set_title('Distribution', fontweight='bold', fontsize=12)

    # 2. Bar chart (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(categories, counts, color=colors)
    ax2.set_xlabel('Count', fontweight='bold')
    ax2.set_title('Trade Counts by Category', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    for i, (count, color) in enumerate(zip(counts, colors)):
        ax2.text(count, i, f' {count:,}', va='center', fontweight='bold')

    # 3. Success vs Failure (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    success = category_stats['target_hit_first']['count']
    failure = category_stats['stop_hit_first']['count']
    other = (category_stats['both_hit_oco_decision']['count'] +
             category_stats['neither_hit']['count'] +
             category_stats['fallback_no_interval_data']['count'])
    ax3.bar(['Success\n(Target Hit)', 'Failure\n(Stop Hit)', 'Other'],
            [success, failure, other],
            color=['#2ecc71', '#e74c3c', '#95a5a6'])
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Success vs Failure', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for i, (val, color) in enumerate(zip([success, failure, other], ['#2ecc71', '#e74c3c', '#95a5a6'])):
        pct = val / category_stats['total_trades'] * 100
        ax3.text(i, val, f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

    # 4. Summary text (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*40}

    Total Trades: {category_stats['total_trades']:,}

    Threshold: {threshold}%
    Trailing Stop: {trailing_stop_pct}%

    OUTCOME BREAKDOWN:

    ✅ Target Hit First: {success:,} ({success/category_stats['total_trades']*100:.1f}%)
       → Successful trades that reached target

    ❌ Stop Hit First: {failure:,} ({failure/category_stats['total_trades']*100:.1f}%)
       → Trades stopped out before target

    ⚖️  Both Hit (OCO): {category_stats['both_hit_oco_decision']['count']:,} ({category_stats['both_hit_oco_decision']['percentage']:.1f}%)
       → Timestamp-based decision

    ⏸️  Neither Hit: {category_stats['neither_hit']['count']:,} ({category_stats['neither_hit']['percentage']:.1f}%)
       → Ended without trigger

    📋 Fallback: {category_stats['fallback_no_interval_data']['count']:,} ({category_stats['fallback_no_interval_data']['percentage']:.1f}%)
       → No interval data available
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Main title
    fig.suptitle(f'Outcome Categories Analysis - {threshold}% Threshold, {trailing_stop_pct}% Trailing Stop',
                 fontsize=16, fontweight='bold')

    # Save
    output_file = output_dir / f"outcome_categories_detailed_{threshold}pct_stop{trailing_stop_pct}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot outcome categories from trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single threshold
  python analysis/plot_outcome_categories.py --threshold 6.0

  # Plot all available thresholds
  python analysis/plot_outcome_categories.py --all

  # Compare across thresholds
  python analysis/plot_outcome_categories.py --compare-thresholds
        """
    )

    parser.add_argument('--threshold', type=float, help='Gain threshold percentage')
    parser.add_argument('--trailing-stop', type=float, default=2.0,
                        help='Trailing stop percentage (default: 2.0)')
    parser.add_argument('--all', action='store_true',
                        help='Plot all available thresholds')
    parser.add_argument('--compare-thresholds', action='store_true',
                        help='Create comparison plot across thresholds')
    parser.add_argument('--output-dir', type=str, default='analysis/plots',
                        help='Output directory for plots')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    output_dir = Path(args.output_dir)

    if args.compare_thresholds:
        # Load all models and create comparison
        print("Loading all model info files...")
        model_infos = find_all_model_infos()

        if not model_infos:
            print("❌ No model info files found")
            return

        print(f"Found {len(model_infos)} model(s)")

        print("\nCreating comparison plot...")
        plot_file = plot_comparison_across_thresholds(model_infos, output_dir)
        if plot_file:
            print(f"✓ Saved: {plot_file}")

    elif args.all:
        # Plot all available thresholds
        print("Loading all model info files...")
        model_infos = find_all_model_infos()

        if not model_infos:
            print("❌ No model info files found")
            return

        print(f"Found {len(model_infos)} model(s)")

        for threshold, trailing_stop, info in model_infos:
            if 'outcome_categories' not in info or info['outcome_categories'] is None:
                print(f"⚠️  No outcome categories for {threshold}% (stop {trailing_stop}%)")
                continue

            print(f"\nPlotting {threshold}% threshold (stop {trailing_stop}%)...")
            category_stats = info['outcome_categories']

            # Create all plot types
            pie_file = plot_single_pie_chart(category_stats, threshold, trailing_stop, output_dir)
            print(f"  ✓ Pie chart: {pie_file}")

            bar_file = plot_stacked_bar(category_stats, threshold, trailing_stop, output_dir)
            print(f"  ✓ Bar chart: {bar_file}")

            detailed_file = plot_detailed_breakdown(category_stats, threshold, trailing_stop, output_dir)
            print(f"  ✓ Detailed: {detailed_file}")

    elif args.threshold:
        # Plot single threshold
        print(f"Loading model info for {args.threshold}% threshold...")
        info = load_model_info(args.threshold, args.trailing_stop)

        if not info:
            print("❌ Model info not found")
            return

        if 'outcome_categories' not in info or info['outcome_categories'] is None:
            print("❌ No outcome categories data in model info")
            return

        category_stats = info['outcome_categories']

        print("\nCreating plots...")

        # Create all plot types
        pie_file = plot_single_pie_chart(category_stats, args.threshold, args.trailing_stop, output_dir)
        print(f"✓ Pie chart: {pie_file}")

        bar_file = plot_stacked_bar(category_stats, args.threshold, args.trailing_stop, output_dir)
        print(f"✓ Bar chart: {bar_file}")

        detailed_file = plot_detailed_breakdown(category_stats, args.threshold, args.trailing_stop, output_dir)
        print(f"✓ Detailed breakdown: {detailed_file}")

    else:
        print("❌ Please specify --threshold, --all, or --compare-thresholds")
        print("Run with --help for usage information")


if __name__ == '__main__':
    main()
