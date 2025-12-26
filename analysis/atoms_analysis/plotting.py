"""
Plotting functions for squeeze alert outcome analysis.

This module contains all visualization and plotting routines extracted from
predict_squeeze_outcomes.py for better code organization and reusability.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import roc_curve


def plot_feature_importance(feature_names: list, feature_importances: np.ndarray,
                            model_name: str = 'Random Forest',
                            output_path: str = 'analysis/plots/feature_importance.png') -> pd.DataFrame:
    """
    Plot feature importance for tree-based models.

    Args:
        feature_names: List of feature names
        feature_importances: Array of feature importance values
        model_name: Name of model to analyze
        output_path: Where to save plot

    Returns:
        DataFrame with feature importance sorted by value
    """
    print("\n" + "="*80)
    print(f"STEP 8: FEATURE IMPORTANCE ANALYSIS - {model_name}")
    print("="*80)

    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    # Print top features
    print("\nTop 10 Most Important Features:")
    print("="*60)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:35s} {row['importance']:.4f}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_file = Path(output_path)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved feature importance plot to: {output_file}")
    plt.close()

    return importance_df


def plot_roc_curves(results: Dict, y_test: pd.Series,
                   output_path: str = 'analysis/plots/roc_curves.png'):
    """
    Plot ROC curves for all models.

    Args:
        results: Dictionary of model results with y_test_proba and roc_auc
        y_test: Test labels
        output_path: Where to save plot
    """
    print("\n" + "="*80)
    print("PLOTTING ROC CURVES")
    print("="*80)

    plt.figure(figsize=(10, 8))

    for name, result in results.items():
        if result['y_test_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_test_proba'])
            auc = result['roc_auc']
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = Path(output_path)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curves to: {output_file}")
    plt.close()


def plot_class_distribution(stats: Dict, gain_threshold: float = 5.0,
                            output_path: str = 'analysis/plots/class_distribution_5pct.png',
                            y_train_balanced: Optional[pd.Series] = None):
    """
    Create pie charts showing both original and SMOTE-balanced class distributions.

    Args:
        stats: Dictionary with class distribution statistics
        gain_threshold: Gain percentage threshold for success
        output_path: Where to save the plot
        y_train_balanced: Balanced training labels after SMOTE (optional)
    """
    if not stats:
        return

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Common properties
    labels = ['Success', 'Failure']
    colors = ['#2ecc71', '#e74c3c']  # Green for success, red for failure
    explode = (0.05, 0)  # Slightly separate the success slice

    # ===== LEFT PLOT: Original Distribution =====
    sizes_original = [stats['successes'], stats['failures']]

    wedges1, texts1, autotexts1 = ax1.pie(
        sizes_original,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
        textprops={'fontsize': 14, 'weight': 'bold'}
    )

    # Make percentage text more visible
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontsize(16)

    # Add title with statistics
    ax1.set_title(
        f'Original Distribution\n{int(gain_threshold)}% Target\n\n'
        f'Total Alerts: {stats["total"]:,}\n'
        f'Success: {stats["successes"]:,} ({stats["success_pct"]:.1f}%) | '
        f'Failure: {stats["failures"]:,} ({stats["failure_pct"]:.1f}%)\n'
        f'Imbalance Ratio: 1 : {stats["imbalance_ratio"]:.2f}',
        fontsize=12,
        fontweight='bold',
        pad=20
    )
    ax1.axis('equal')

    # ===== RIGHT PLOT: Balanced Distribution =====
    if y_train_balanced is not None:
        balanced_successes = int(y_train_balanced.sum())
        balanced_total = len(y_train_balanced)
        balanced_failures = balanced_total - balanced_successes
        balanced_success_pct = (balanced_successes / balanced_total * 100) if balanced_total > 0 else 0
        balanced_failure_pct = (balanced_failures / balanced_total * 100) if balanced_total > 0 else 0
        balanced_imbalance_ratio = balanced_failures / balanced_successes if balanced_successes > 0 else 0

        sizes_balanced = [balanced_successes, balanced_failures]

        wedges2, texts2, autotexts2 = ax2.pie(
            sizes_balanced,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90,
            textprops={'fontsize': 14, 'weight': 'bold'}
        )

        # Make percentage text more visible
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontsize(16)

        # Add title with statistics
        ax2.set_title(
            f'SMOTE-Balanced Distribution\n{int(gain_threshold)}% Target\n\n'
            f'Total Alerts: {balanced_total:,}\n'
            f'Success: {balanced_successes:,} ({balanced_success_pct:.1f}%) | '
            f'Failure: {balanced_failures:,} ({balanced_failure_pct:.1f}%)\n'
            f'Imbalance Ratio: 1 : {balanced_imbalance_ratio:.2f}',
            fontsize=12,
            fontweight='bold',
            pad=20
        )
        ax2.axis('equal')

        # Add assessment text for balanced data
        if 45 <= balanced_success_pct <= 55:
            assessment = "✓ Excellent Balance (Model Training Data)"
            assessment_color = 'green'
        elif 40 <= balanced_success_pct <= 60:
            assessment = "~ Good Balance (Model Training Data)"
            assessment_color = 'orange'
        elif 35 <= balanced_success_pct <= 65:
            assessment = "⚠ Moderate Imbalance (Model Training Data)"
            assessment_color = 'darkorange'
        else:
            assessment = "❌ Severe Imbalance (Model Training Data)"
            assessment_color = 'red'
    else:
        # If no balanced data provided, show a message
        ax2.text(0.5, 0.5, 'No Balanced Data Provided\n(SMOTE not applied)',
                ha='center', va='center', fontsize=14, weight='bold')
        ax2.axis('off')
        assessment = "Original Distribution Only"
        assessment_color = 'gray'

    plt.figtext(
        0.5, 0.02,
        assessment,
        ha='center',
        fontsize=12,
        weight='bold',
        color=assessment_color
    )

    plt.tight_layout()

    # Save plot
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved class distribution plot to: {output_file}")


def plot_price_category_analysis(grouped: pd.DataFrame, gain_threshold: float,
                                 output_path: str):
    """
    Create visualization of price category performance.

    Args:
        grouped: DataFrame with metrics by price category
        gain_threshold: Gain percentage threshold
        output_path: Where to save plot
    """
    # Sort by price category order for better visualization
    price_order = ['<$2', '$2-5', '$5-10', '$10-20', '$20-40', '>$40']
    grouped_sorted = grouped.reindex([p for p in price_order if p in grouped.index])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Squeeze Performance by Price Category - {gain_threshold}% Target',
                 fontsize=16, fontweight='bold')

    # 1. Win Rate comparison
    ax1 = axes[0, 0]
    x_pos = range(len(grouped_sorted))
    win_col = f'Win_Rate_{int(gain_threshold)}pct'
    bars = ax1.bar(x_pos, grouped_sorted[win_col] * 100, color='green', alpha=0.7)
    ax1.axhline(y=grouped_sorted[win_col].mean() * 100, color='red', linestyle='--',
               label=f'Average ({grouped_sorted[win_col].mean()*100:.1f}%)')
    ax1.set_xlabel('Price Category', fontweight='bold')
    ax1.set_ylabel(f'{int(gain_threshold)}% Win Rate (%)', fontweight='bold')
    ax1.set_title(f'{int(gain_threshold)}% Win Rate by Price Category', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(grouped_sorted.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, grouped_sorted[win_col] * 100)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2. Stop Loss Rate
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, grouped_sorted['Stop_Loss_Rate'] * 100, color='red', alpha=0.7)
    ax2.axhline(y=grouped_sorted['Stop_Loss_Rate'].mean() * 100, color='blue', linestyle='--',
               label=f'Average ({grouped_sorted["Stop_Loss_Rate"].mean()*100:.1f}%)')
    ax2.set_xlabel('Price Category', fontweight='bold')
    ax2.set_ylabel('Stop Loss Rate (%)', fontweight='bold')
    ax2.set_title('Stop Loss Rate by Price Category', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(grouped_sorted.index, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, grouped_sorted['Stop_Loss_Rate'] * 100)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 3. Profitability Score (Win Rate - Stop Loss Rate)
    ax3 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in grouped_sorted['Profitability_Score']]
    bars = ax3.bar(x_pos, grouped_sorted['Profitability_Score'] * 100, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Price Category', fontweight='bold')
    ax3.set_ylabel('Profitability Score (%)', fontweight='bold')
    ax3.set_title('Profitability Score (Win Rate - Stop Loss Rate)', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(grouped_sorted.index, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, grouped_sorted['Profitability_Score'] * 100)):
        y_pos = bar.get_height() + 1 if val > 0 else bar.get_height() - 3
        ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    # 4. Average Final Gain
    ax4 = axes[1, 1]
    colors = ['green' if x > 0 else 'red' for x in grouped_sorted['Avg_Final_Gain']]
    bars = ax4.bar(x_pos, grouped_sorted['Avg_Final_Gain'], color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('Price Category', fontweight='bold')
    ax4.set_ylabel('Average Final Gain (%)', fontweight='bold')
    ax4.set_title('Average Final Gain by Price Category', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(grouped_sorted.index, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, grouped_sorted['Avg_Final_Gain'])):
        y_pos = bar.get_height() + 0.3 if val > 0 else bar.get_height() - 0.5
        ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    plt.tight_layout()

    output_file = Path(output_path)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved price category analysis plot to: {output_file}")
    plt.close()


def plot_time_of_day_analysis(grouped: pd.DataFrame, gain_threshold: float,
                              output_path: str):
    """
    Create visualization of time of day performance.

    Args:
        grouped: DataFrame with metrics by time bin
        gain_threshold: Gain percentage threshold
        output_path: Where to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Squeeze Performance by Time of Day - {gain_threshold}% Target',
                 fontsize=16, fontweight='bold')

    x_pos = range(len(grouped))
    time_labels = grouped.index.tolist()

    # 1. Win Rate by time
    ax1 = axes[0, 0]
    win_col = f'Win_Rate_{int(gain_threshold)}pct'
    bars = ax1.bar(x_pos, grouped[win_col] * 100, color='green', alpha=0.7)
    ax1.axhline(y=grouped[win_col].mean() * 100, color='red', linestyle='--',
               label=f'Average ({grouped[win_col].mean()*100:.1f}%)')

    # Highlight first hour (9:30-10:30)
    morning_bins = ['09:30-10:00', '10:00-10:30']
    for i, time_bin in enumerate(time_labels):
        if time_bin in morning_bins:
            bars[i].set_color('darkgreen')
            bars[i].set_alpha(0.9)

    ax1.set_xlabel('Time of Day', fontweight='bold')
    ax1.set_ylabel(f'{int(gain_threshold)}% Win Rate (%)', fontweight='bold')
    ax1.set_title(f'{int(gain_threshold)}% Win Rate by Time of Day (First hour highlighted)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Stop Loss Rate
    ax2 = axes[0, 1]
    bars = ax2.bar(x_pos, grouped['Stop_Loss_Rate'] * 100, color='red', alpha=0.7)
    ax2.axhline(y=grouped['Stop_Loss_Rate'].mean() * 100, color='blue', linestyle='--',
               label=f'Average ({grouped["Stop_Loss_Rate"].mean()*100:.1f}%)')
    ax2.set_xlabel('Time of Day', fontweight='bold')
    ax2.set_ylabel('Stop Loss Rate (%)', fontweight='bold')
    ax2.set_title('Stop Loss Rate by Time of Day', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Profitability Score
    ax3 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in grouped['Profitability_Score']]
    bars = ax3.bar(x_pos, grouped['Profitability_Score'] * 100, color=colors, alpha=0.7)

    # Highlight first hour
    for i, time_bin in enumerate(time_labels):
        if time_bin in morning_bins:
            if grouped.iloc[i]['Profitability_Score'] > 0:
                bars[i].set_color('darkgreen')
                bars[i].set_alpha(0.9)

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Time of Day', fontweight='bold')
    ax3.set_ylabel('Profitability Score (%)', fontweight='bold')
    ax3.set_title('Profitability Score by Time of Day (First hour highlighted)', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Alert Count (volume throughout day)
    ax4 = axes[1, 1]
    bars = ax4.bar(x_pos, grouped['Count'], color='blue', alpha=0.7)
    ax4.set_xlabel('Time of Day', fontweight='bold')
    ax4.set_ylabel('Number of Alerts', fontweight='bold')
    ax4.set_title('Alert Volume by Time of Day', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels on volume bars
    for i, (bar, val) in enumerate(zip(bars, grouped['Count'])):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(grouped['Count'])*0.01,
                f'{int(val)}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    output_file = Path(output_path)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved time of day analysis plot to: {output_file}")
    plt.close()


def generate_prediction_plots(predictions_df: pd.DataFrame, model_trades: pd.DataFrame,
                              threshold_suffix: str, gain_threshold: float):
    """Generate plots for prediction analysis."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = Path('analysis/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Plot 1: Profit Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Prediction Analysis - {gain_threshold}% Target', fontsize=16, fontweight='bold')

    # 1a: Model trades profit distribution
    ax1 = axes[0, 0]
    if len(model_trades) > 0:
        ax1.hist(model_trades['realistic_profit'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=model_trades['realistic_profit'].mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Avg: {model_trades["realistic_profit"].mean():.2f}%')
        ax1.set_xlabel('Profit %', fontweight='bold')
        ax1.set_ylabel('Number of Trades', fontweight='bold')
        ax1.set_title(f'Model Trades Profit Distribution ({len(model_trades)} trades)', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

    # 1b: All trades profit distribution
    ax2 = axes[0, 1]
    ax2.hist(predictions_df['realistic_profit'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax2.axvline(x=predictions_df['realistic_profit'].mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Avg: {predictions_df["realistic_profit"].mean():.2f}%')
    ax2.set_xlabel('Profit %', fontweight='bold')
    ax2.set_ylabel('Number of Trades', fontweight='bold')
    ax2.set_title(f'All Opportunities Profit Distribution ({len(predictions_df)} trades)', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 1c: Cumulative profit comparison (using compounding multiplication, not addition)
    ax3 = axes[1, 0]
    if len(model_trades) > 0:
        # Correct compounding: convert % to decimal, multiply returns, convert back to %
        # Example: 10% then 5% = 1.10 × 1.05 - 1 = 15.5% (not 15%)
        model_cumulative = ((1 + model_trades['realistic_profit']/100).cumprod() - 1) * 100
        all_cumulative = ((1 + predictions_df['realistic_profit']/100).cumprod() - 1) * 100
        ax3.plot(model_cumulative.values, label=f'Model ({len(model_trades)} trades)', linewidth=2, color='green')
        ax3.plot(all_cumulative.values, label=f'Take-All ({len(predictions_df)} trades)', linewidth=2, color='orange')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax3.set_xlabel('Trade Number', fontweight='bold')
        ax3.set_ylabel('Cumulative Profit % (Compounded)', fontweight='bold')
        ax3.set_title('Cumulative Profit Comparison', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

    # 1d: Probability distribution of selected vs missed trades
    ax4 = axes[1, 1]
    selected = predictions_df[predictions_df['predicted_outcome'] == 1]['prediction_probability']
    missed = predictions_df[predictions_df['predicted_outcome'] == 0]['prediction_probability']
    ax4.hist([selected, missed], bins=30, alpha=0.7, label=['Selected', 'Missed'], color=['green', 'red'])
    ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax4.set_xlabel('Prediction Probability', fontweight='bold')
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('Probability Distribution: Selected vs Missed', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plot_file = plots_dir / f'prediction_analysis{threshold_suffix}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved analysis plots to: {plot_file}")
    plt.close()

    # Plot 2: Win/Loss Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Win/Loss Analysis - {gain_threshold}% Target', fontsize=16, fontweight='bold')

    # 2a: Model trades pie chart
    ax1 = axes[0]
    if len(model_trades) > 0:
        model_wins = len(model_trades[model_trades['realistic_profit'] > 0])
        model_losses = len(model_trades[model_trades['realistic_profit'] <= 0])
        ax1.pie([model_wins, model_losses], labels=['Wins', 'Losses'],
               autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
        ax1.set_title(f'Model Trades\n({model_wins} wins, {model_losses} losses)', fontweight='bold')

    # 2b: Comparison bar chart
    ax2 = axes[1]
    if len(model_trades) > 0:
        # Use compounding multiplication for total profit, not addition
        model_total = ((1 + model_trades['realistic_profit']/100).prod() - 1) * 100
        all_total = ((1 + predictions_df['realistic_profit']/100).prod() - 1) * 100
        model_avg = model_trades['realistic_profit'].mean()
        all_avg = predictions_df['realistic_profit'].mean()

        x = np.arange(2)
        width = 0.35
        ax2.bar(x - width/2, [model_total, model_avg], width, label='Model', color='green', alpha=0.7)
        ax2.bar(x + width/2, [all_total, all_avg], width, label='Take-All', color='orange', alpha=0.7)
        ax2.set_ylabel('Profit %', fontweight='bold')
        ax2.set_title('Profit Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Total Profit\n(Compounded)', 'Avg Per Trade'])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bars1, bars2) in enumerate(zip(ax2.containers[0], ax2.containers[1])):
            ax2.text(bars1.get_x() + bars1.get_width()/2, bars1.get_height(),
                    f'{bars1.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
            ax2.text(bars2.get_x() + bars2.get_width()/2, bars2.get_height(),
                    f'{bars2.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_file = plots_dir / f'win_loss_analysis{threshold_suffix}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved win/loss plots to: {plot_file}")
    plt.close()


def generate_aligned_cumulative_profit_plot(predictions_df: pd.DataFrame, model_trades: pd.DataFrame,
                                            threshold_suffix: str, gain_threshold: float):
    """
    Generate aligned cumulative profit comparison chart.

    This chart shows both Model and Take-All strategies on the same timeline,
    with model trades aligned vertically with their corresponding opportunities.

    Args:
        predictions_df: All predictions with realistic_profit
        model_trades: Subset of predictions where model predicted to trade
        threshold_suffix: Suffix for output filename
        gain_threshold: Target gain threshold percentage
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = Path('analysis/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    if len(model_trades) == 0:
        print("⚠️  No model trades to plot - skipping aligned cumulative profit chart")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(f'Aligned Cumulative Profit Comparison - {gain_threshold}% Target',
                 fontsize=16, fontweight='bold')

    # Calculate cumulative profit for Take-All strategy
    all_cumulative = ((1 + predictions_df['realistic_profit']/100).cumprod() - 1) * 100

    # Calculate cumulative profit for Model strategy, aligned with all opportunities
    # Start with all zeros, then fill in model trades
    model_cumulative_aligned = pd.Series(0.0, index=predictions_df.index)

    # Track cumulative multiplier for model
    cumulative_multiplier = 1.0

    # For each row in predictions_df, check if model traded
    for idx in predictions_df.index:
        if idx in model_trades.index:
            # Model took this trade
            profit_pct = predictions_df.loc[idx, 'realistic_profit']
            cumulative_multiplier *= (1 + profit_pct/100)
        # Store current cumulative profit (whether traded or not)
        model_cumulative_aligned.loc[idx] = (cumulative_multiplier - 1) * 100

    # Plot both strategies
    x_positions = range(len(predictions_df))

    # Take-All strategy
    ax.plot(x_positions, all_cumulative.values,
            label=f'Take-All ({len(predictions_df)} trades)',
            linewidth=2, color='orange', alpha=0.8)

    # Model strategy (aligned)
    ax.plot(x_positions, model_cumulative_aligned.values,
            label=f'Model ({len(model_trades)} trades selected)',
            linewidth=2, color='green', alpha=0.8)

    # Add horizontal line at 0
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Labels and formatting
    ax.set_xlabel('Opportunity Number (chronological)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cumulative Profit % (Compounded)', fontweight='bold', fontsize=12)
    ax.set_title('Model vs Take-All: Same Timeline Comparison', fontweight='bold', fontsize=13, pad=10)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Enable grid with more vertical lines
    ax.grid(alpha=0.3, linestyle='--', which='both')
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.15, linestyle=':', axis='x')

    # Add summary statistics box
    final_model_profit = model_cumulative_aligned.iloc[-1]
    final_all_profit = all_cumulative.iloc[-1]
    model_avg = model_trades['realistic_profit'].mean()
    all_avg = predictions_df['realistic_profit'].mean()

    stats_text = f'Final Results:\n'
    stats_text += f'Model: {final_model_profit:+.2f}% total ({model_avg:+.2f}% avg/trade)\n'
    stats_text += f'Take-All: {final_all_profit:+.2f}% total ({all_avg:+.2f}% avg/trade)\n'
    stats_text += f'Difference: {final_model_profit - final_all_profit:+.2f}% total'

    # Position text box
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plot_file = plots_dir / f'aligned_cumulative_profit{threshold_suffix}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved aligned cumulative profit plot to: {plot_file}")
    plt.close()


def generate_time_binned_outcomes_chart(predictions_df: pd.DataFrame, threshold_suffix: str,
                                        gain_threshold: float):
    """
    Generate a bar chart showing wins and losses by 30-minute time bins.

    Args:
        predictions_df: DataFrame with predictions and timestamps
        threshold_suffix: Suffix for filename (e.g., '_2pct')
        gain_threshold: Gain threshold percentage
    """
    print("\n" + "="*80)
    print("GENERATING TIME-BINNED OUTCOMES CHART")
    print("="*80)

    # Make a copy to avoid modifying original
    df = predictions_df.copy()

    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract time of day
    df['time_of_day'] = df['timestamp'].dt.time

    # Create 30-minute bins
    # Convert time to minutes since midnight for easier binning
    df['minutes_since_midnight'] = (
        df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    )

    # Create 30-minute bins (0, 30, 60, 90, ...)
    bin_size = 30
    df['time_bin'] = (df['minutes_since_midnight'] // bin_size) * bin_size

    # Convert bin back to time format for display (HH:MM)
    df['time_bin_label'] = df['time_bin'].apply(
        lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}"
    )

    # Count wins and losses per bin using actual_outcome
    # actual_outcome == 1 means the trade achieved the gain threshold
    bins_data = df.groupby('time_bin_label').agg({
        'actual_outcome': ['sum', 'count']
    }).reset_index()

    # Flatten column names
    bins_data.columns = ['time_bin', 'wins', 'total']
    bins_data['losses'] = bins_data['total'] - bins_data['wins']

    # Sort by time
    bins_data['sort_key'] = bins_data['time_bin'].apply(
        lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    )
    bins_data = bins_data.sort_values('sort_key').reset_index(drop=True)

    print(f"\nTime-binned outcomes (30-minute bins):")
    print(f"{'Time Bin':<12} {'Wins':<8} {'Losses':<8} {'Total':<8} {'Win Rate':<10}")
    print("-" * 50)
    for _, row in bins_data.iterrows():
        win_rate = row['wins'] / row['total'] * 100 if row['total'] > 0 else 0
        print(f"{row['time_bin']:<12} {int(row['wins']):<8} {int(row['losses']):<8} "
              f"{int(row['total']):<8} {win_rate:>6.1f}%")

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(bins_data))
    width = 0.35

    # Plot wins and losses side by side
    bars1 = ax.bar(x - width/2, bins_data['wins'], width, label='Wins',
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, bins_data['losses'], width, label='Losses',
                   color='#e74c3c', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('Time of Day (30-minute bins)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
    ax.set_title(f'Wins vs Losses by Time of Day ({gain_threshold}% Target)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bins_data['time_bin'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Add statistics text box
    total_wins = bins_data['wins'].sum()
    total_losses = bins_data['losses'].sum()
    total_trades = bins_data['total'].sum()
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    stats_text = f'Overall Statistics:\n'
    stats_text += f'Total Trades: {int(total_trades)}\n'
    stats_text += f'Wins: {int(total_wins)} ({overall_win_rate:.1f}%)\n'
    stats_text += f'Losses: {int(total_losses)} ({100-overall_win_rate:.1f}%)'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plots_dir = Path('analysis/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plots_dir / f'time_binned_outcomes{threshold_suffix}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved time-binned outcomes chart to: {plot_file}")
    plt.close()


def generate_price_binned_outcomes_chart(predictions_df: pd.DataFrame, threshold_suffix: str,
                                        gain_threshold: float):
    """
    Generate a bar chart showing wins and losses by price bins.

    Args:
        predictions_df: DataFrame with predictions and squeeze_entry_price
        threshold_suffix: Suffix for filename (e.g., '_2pct')
        gain_threshold: Gain threshold percentage
    """
    print("\n" + "="*80)
    print("GENERATING PRICE-BINNED OUTCOMES CHART")
    print("="*80)

    # Make a copy to avoid modifying original
    df = predictions_df.copy()

    # Remove rows with missing prices
    df = df.dropna(subset=['squeeze_entry_price'])

    if len(df) == 0:
        print("⚠ No valid price data available for price binning")
        return

    # Determine price bins based on data distribution
    min_price = df['squeeze_entry_price'].min()
    max_price = df['squeeze_entry_price'].max()

    # Create bins - doubled number of bins (half the width)
    # Using finer granularity for better analysis
    price_range = max_price - min_price

    if price_range < 50:
        # For lower-priced stocks, use $2.50 bins
        bin_width = 2.5
    elif price_range < 200:
        # For mid-priced stocks, use $5 bins
        bin_width = 5
    elif price_range < 500:
        # For higher-priced stocks, use $12.50 bins
        bin_width = 12.5
    else:
        # For very high-priced stocks, use $25 bins
        bin_width = 25

    # Create bin edges using float bins
    bin_start = (int(min_price / bin_width)) * bin_width
    bin_end = ((int(max_price / bin_width)) + 1) * bin_width
    bins = list(np.arange(bin_start, bin_end + bin_width, bin_width))

    # Create price bins
    df['price_bin'] = pd.cut(df['squeeze_entry_price'], bins=bins,
                              include_lowest=True, right=False)

    # Create bin labels - format based on whether we have fractional dollars
    # e.g., "$2.50-$5.00" or "$100-$105"
    def format_price_label(interval):
        if pd.notna(interval):
            left = interval.left
            right = interval.right
            # Use .2f for bins with decimals, otherwise use int
            if bin_width % 1 == 0:
                return f"${int(left)}-${int(right)}"
            else:
                return f"${left:.2f}-${right:.2f}"
        return "Unknown"

    df['price_bin_label'] = df['price_bin'].apply(format_price_label)

    # Count wins and losses per bin using actual_outcome
    # actual_outcome == 1 means the trade achieved the gain threshold
    bins_data = df.groupby('price_bin_label').agg({
        'actual_outcome': ['sum', 'count']
    }).reset_index()

    # Flatten column names
    bins_data.columns = ['price_bin', 'wins', 'total']
    bins_data['losses'] = bins_data['total'] - bins_data['wins']

    # Sort by price (extract lower bound from label, handle floats)
    bins_data['sort_key'] = bins_data['price_bin'].apply(
        lambda x: float(x.split('-')[0].replace('$', '')) if '-' in x else 0
    )
    bins_data = bins_data.sort_values('sort_key').reset_index(drop=True)

    # Remove bins with no trades
    bins_data = bins_data[bins_data['total'] > 0]

    if len(bins_data) == 0:
        print("⚠ No valid data after binning")
        return

    # Format bin width display
    bin_width_str = f"{int(bin_width)}" if bin_width % 1 == 0 else f"{bin_width:.2f}"
    print(f"\nPrice-binned outcomes (${bin_width_str} bins):")
    print(f"{'Price Range':<15} {'Wins':<8} {'Losses':<8} {'Total':<8} {'Win Rate':<10}")
    print("-" * 55)
    for _, row in bins_data.iterrows():
        win_rate = row['wins'] / row['total'] * 100 if row['total'] > 0 else 0
        print(f"{row['price_bin']:<15} {int(row['wins']):<8} {int(row['losses']):<8} "
              f"{int(row['total']):<8} {win_rate:>6.1f}%")

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(bins_data))
    width = 0.35

    # Plot wins and losses side by side
    bars1 = ax.bar(x - width/2, bins_data['wins'], width, label='Wins',
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, bins_data['losses'], width, label='Losses',
                   color='#e74c3c', alpha=0.8)

    # Customize the plot
    ax.set_xlabel(f'Stock Price at Entry (${bin_width_str} bins)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
    ax.set_title(f'Wins vs Losses by Entry Price ({gain_threshold}% Target)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bins_data['price_bin'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Add statistics text box
    total_wins = bins_data['wins'].sum()
    total_losses = bins_data['losses'].sum()
    total_trades = bins_data['total'].sum()
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    stats_text = f'Overall Statistics:\n'
    stats_text += f'Total Trades: {int(total_trades)}\n'
    stats_text += f'Wins: {int(total_wins)} ({overall_win_rate:.1f}%)\n'
    stats_text += f'Losses: {int(total_losses)} ({100-overall_win_rate:.1f}%)\n'
    stats_text += f'Price Range: ${min_price:.2f} - ${max_price:.2f}'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save the plot
    plots_dir = Path('analysis/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plots_dir / f'price_binned_outcomes{threshold_suffix}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved price-binned outcomes chart to: {plot_file}")
    plt.close()
