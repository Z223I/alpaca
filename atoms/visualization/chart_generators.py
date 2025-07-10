"""
Chart generation utilities for alert performance visualization.

This atom provides functions to create various charts and plots for
analyzing and presenting alert performance data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, Optional, List, Tuple, Union
import seaborn as sns
from datetime import datetime
import warnings

# Set default style
plt.style.use('default')
sns.set_palette("husl")


def create_performance_summary_chart(
    alerts_df: pd.DataFrame,
    return_col: str = "return_pct",
    priority_col: str = "priority",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive performance summary chart.
    
    Args:
        alerts_df: DataFrame containing alert data
        return_col: Column name for returns
        priority_col: Column name for priority levels
        figsize: Figure size tuple
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Alert Performance Summary', fontsize=16, fontweight='bold')
    
    # 1. Return Distribution Histogram
    if return_col in alerts_df.columns:
        returns = alerts_df[return_col].dropna()
        axes[0, 0].hist(returns, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(returns.mean(), color='red', linestyle='--', 
                          label=f'Mean: {returns.mean():.2f}%')
        axes[0, 0].axvline(returns.median(), color='orange', linestyle='--', 
                          label=f'Median: {returns.median():.2f}%')
        axes[0, 0].set_xlabel('Return (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Return Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box Plot by Priority
    if priority_col in alerts_df.columns and return_col in alerts_df.columns:
        priority_data = []
        priority_labels = []
        for priority in alerts_df[priority_col].unique():
            priority_returns = alerts_df[alerts_df[priority_col] == priority][return_col].dropna()
            if not priority_returns.empty:
                priority_data.append(priority_returns)
                priority_labels.append(priority)
        
        if priority_data:
            axes[0, 1].boxplot(priority_data, labels=priority_labels)
            axes[0, 1].set_xlabel('Priority Level')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].set_title('Returns by Priority')
            axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Success Rate by Priority
    if priority_col in alerts_df.columns and 'status' in alerts_df.columns:
        success_rates = []
        priorities = []
        
        for priority in alerts_df[priority_col].unique():
            priority_alerts = alerts_df[alerts_df[priority_col] == priority]
            success_rate = (priority_alerts['status'] == 'SUCCESS').mean() * 100
            success_rates.append(success_rate)
            priorities.append(priority)
        
        bars = axes[0, 2].bar(priorities, success_rates, alpha=0.7)
        axes[0, 2].set_xlabel('Priority Level')
        axes[0, 2].set_ylabel('Success Rate (%)')
        axes[0, 2].set_title('Success Rate by Priority')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom')
    
    # 4. Cumulative Returns
    if return_col in alerts_df.columns and 'timestamp' in alerts_df.columns:
        df_sorted = alerts_df.sort_values('timestamp').copy()
        df_sorted['cumulative_return'] = (1 + df_sorted[return_col]/100).cumprod() - 1
        
        axes[1, 0].plot(df_sorted.index, df_sorted['cumulative_return'] * 100, 
                       linewidth=2, color='blue')
        axes[1, 0].set_xlabel('Alert Number')
        axes[1, 0].set_ylabel('Cumulative Return (%)')
        axes[1, 0].set_title('Cumulative Performance')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 5. Monthly Performance (if timestamp available)
    if 'timestamp' in alerts_df.columns and return_col in alerts_df.columns:
        df_time = alerts_df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        df_time['month'] = df_time['timestamp'].dt.to_period('M')
        
        monthly_returns = df_time.groupby('month')[return_col].mean()
        monthly_counts = df_time.groupby('month').size()
        
        if not monthly_returns.empty:
            months = [str(m) for m in monthly_returns.index]
            bars = axes[1, 1].bar(range(len(months)), monthly_returns.values, alpha=0.7)
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Average Return (%)')
            axes[1, 1].set_title('Monthly Average Returns')
            axes[1, 1].set_xticks(range(len(months)))
            axes[1, 1].set_xticklabels(months, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 6. Performance Heatmap by Hour and Day
    if 'timestamp' in alerts_df.columns and return_col in alerts_df.columns:
        df_time = alerts_df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        df_time['hour'] = df_time['timestamp'].dt.hour
        df_time['day'] = df_time['timestamp'].dt.day_name()
        
        pivot_data = df_time.pivot_table(
            values=return_col, 
            index='day', 
            columns='hour', 
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            im = axes[1, 2].imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
            axes[1, 2].set_xticks(range(len(pivot_data.columns)))
            axes[1, 2].set_xticklabels(pivot_data.columns)
            axes[1, 2].set_yticks(range(len(pivot_data.index)))
            axes[1, 2].set_yticklabels(pivot_data.index)
            axes[1, 2].set_xlabel('Hour of Day')
            axes[1, 2].set_ylabel('Day of Week')
            axes[1, 2].set_title('Performance Heatmap')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 2])
            cbar.set_label('Average Return (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_risk_analysis_chart(
    returns: pd.Series,
    risk_metrics: Dict[str, Any],
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive risk analysis visualization.
    
    Args:
        returns: Series of return values
        risk_metrics: Dictionary with risk metrics
        figsize: Figure size tuple
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Risk Analysis Dashboard', fontsize=16, fontweight='bold')
    
    if returns.empty:
        fig.text(0.5, 0.5, 'No data available for risk analysis', 
                ha='center', va='center', fontsize=14)
        return fig
    
    # 1. Return vs Risk Scatter
    rolling_returns = returns.rolling(window=10).mean()
    rolling_vol = returns.rolling(window=10).std()
    
    scatter = axes[0, 0].scatter(rolling_vol, rolling_returns, alpha=0.6, s=30)
    axes[0, 0].set_xlabel('Volatility (10-period)')
    axes[0, 0].set_ylabel('Average Return (10-period)')
    axes[0, 0].set_title('Risk-Return Scatter')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Drawdown Chart
    cumulative_returns = (1 + returns/100).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max * 100
    
    axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, 
                           alpha=0.7, color='red', label='Drawdown')
    axes[0, 1].set_xlabel('Trade Number')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].set_title('Drawdown Analysis')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Rolling Volatility
    rolling_vol_30 = returns.rolling(window=30).std()
    axes[0, 2].plot(rolling_vol_30, linewidth=2, color='orange')
    axes[0, 2].set_xlabel('Trade Number')
    axes[0, 2].set_ylabel('Rolling Volatility (30-period)')
    axes[0, 2].set_title('Volatility Over Time')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. VaR Visualization
    if 'var_metrics' in risk_metrics:
        var_levels = ['var_1', 'var_5', 'var_10']
        var_values = [risk_metrics['var_metrics'].get(level, 0) for level in var_levels]
        confidence_levels = ['99%', '95%', '90%']
        
        bars = axes[1, 0].bar(confidence_levels, var_values, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Confidence Level')
        axes[1, 0].set_ylabel('Value at Risk (%)')
        axes[1, 0].set_title('Value at Risk')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, var_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height - 0.1,
                           f'{value:.2f}%', ha='center', va='top', color='white', 
                           fontweight='bold')
    
    # 5. Return Distribution with Normal Overlay
    axes[1, 1].hist(returns, bins=30, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black', label='Actual')
    
    # Overlay normal distribution
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = stats.norm.pdf(x, returns.mean(), returns.std())
    axes[1, 1].plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
    
    axes[1, 1].set_xlabel('Return (%)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Return Distribution vs Normal')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Risk Metrics Summary
    axes[1, 2].axis('off')
    
    # Create text summary of risk metrics
    text_content = []
    
    if 'sharpe_ratio' in risk_metrics:
        text_content.append(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
    if 'sortino_ratio' in risk_metrics:
        text_content.append(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.3f}")
    if 'calmar_ratio' in risk_metrics:
        text_content.append(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.3f}")
    if 'max_drawdown' in risk_metrics:
        text_content.append(f"Max Drawdown: {risk_metrics['max_drawdown']:.2f}%")
    if 'volatility' in risk_metrics:
        text_content.append(f"Volatility: {risk_metrics['volatility']:.2f}%")
    if 'skewness' in risk_metrics:
        text_content.append(f"Skewness: {risk_metrics['skewness']:.3f}")
    if 'kurtosis' in risk_metrics:
        text_content.append(f"Kurtosis: {risk_metrics['kurtosis']:.3f}")
    
    # Display metrics
    y_pos = 0.9
    for line in text_content:
        axes[1, 2].text(0.1, y_pos, line, fontsize=12, transform=axes[1, 2].transAxes)
        y_pos -= 0.12
    
    axes[1, 2].set_title('Risk Metrics Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    p_value_matrix: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create correlation heatmap with significance indicators.
    
    Args:
        correlation_matrix: DataFrame with correlation coefficients
        p_value_matrix: Optional DataFrame with p-values
        figsize: Figure size tuple
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if correlation_matrix.empty:
        ax.text(0.5, 0.5, 'No correlation data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    # Create mask for significance if p-values provided
    mask = None
    if p_value_matrix is not None:
        # Mask non-significant correlations (p > 0.05)
        mask = p_value_matrix > 0.05
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                mask=mask,
                cbar_kws={'label': 'Correlation Coefficient'},
                fmt='.3f',
                ax=ax)
    
    ax.set_title('Correlation Matrix\n(Only significant correlations shown if p-values provided)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_time_series_chart(
    alerts_df: pd.DataFrame,
    value_col: str,
    timestamp_col: str = "timestamp",
    title: str = "Time Series Analysis",
    figsize: Tuple[int, int] = (15, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create time series chart with trend analysis.
    
    Args:
        alerts_df: DataFrame containing time series data
        value_col: Column name for values to plot
        timestamp_col: Column name for timestamps
        title: Chart title
        figsize: Figure size tuple
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if alerts_df.empty or value_col not in alerts_df.columns:
        ax.text(0.5, 0.5, 'No time series data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    # Prepare data
    df = alerts_df[[timestamp_col, value_col]].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    
    # Plot main time series
    ax.plot(df[timestamp_col], df[value_col], linewidth=2, alpha=0.7, label='Value')
    
    # Add moving average
    if len(df) >= 10:
        df['ma_10'] = df[value_col].rolling(window=10).mean()
        ax.plot(df[timestamp_col], df['ma_10'], 
               linewidth=2, color='red', alpha=0.8, label='10-Period MA')
    
    # Add trend line
    if len(df) >= 5:
        x_numeric = mdates.date2num(df[timestamp_col])
        z = np.polyfit(x_numeric, df[value_col], 1)
        p = np.poly1d(z)
        ax.plot(df[timestamp_col], p(x_numeric), 
               linestyle='--', color='orange', linewidth=2, alpha=0.8, label='Trend')
    
    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col.replace('_', ' ').title())
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df)//10)))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_performance_comparison_chart(
    comparison_data: Dict[str, pd.Series],
    title: str = "Performance Comparison",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comparison chart for multiple performance series.
    
    Args:
        comparison_data: Dictionary with series names as keys and return series as values
        title: Chart title
        figsize: Figure size tuple
        save_path: Optional path to save the chart
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if not comparison_data:
        fig.text(0.5, 0.5, 'No comparison data available', 
                ha='center', va='center', fontsize=14)
        return fig
    
    # 1. Cumulative Returns Comparison
    for name, returns in comparison_data.items():
        if not returns.empty:
            cumulative = (1 + returns/100).cumprod() - 1
            axes[0, 0].plot(range(len(cumulative)), cumulative * 100, 
                           linewidth=2, label=name, alpha=0.8)
    
    axes[0, 0].set_xlabel('Trade Number')
    axes[0, 0].set_ylabel('Cumulative Return (%)')
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 2. Return Distribution Comparison
    for name, returns in comparison_data.items():
        if not returns.empty:
            axes[0, 1].hist(returns, bins=20, alpha=0.6, label=name, density=True)
    
    axes[0, 1].set_xlabel('Return (%)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Return Distributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Summary Statistics Comparison
    metrics = ['Mean', 'Std', 'Sharpe', 'Max Drawdown']
    series_names = list(comparison_data.keys())
    
    summary_data = []
    for name, returns in comparison_data.items():
        if not returns.empty:
            from atoms.metrics.calculate_returns import calculate_cumulative_returns, calculate_maximum_drawdown
            cumulative = calculate_cumulative_returns(returns, compound=True)
            max_dd = abs(calculate_maximum_drawdown(cumulative))
            sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0
            
            summary_data.append([
                returns.mean(),
                returns.std(),
                sharpe,
                max_dd
            ])
    
    if summary_data:
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (name, data) in enumerate(zip(series_names, summary_data)):
            offset = width * (i - len(series_names)/2 + 0.5)
            axes[1, 0].bar(x + offset, data, width, label=name, alpha=0.8)
        
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Summary Statistics')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Rolling Sharpe Ratio Comparison
    for name, returns in comparison_data.items():
        if not returns.empty and len(returns) >= 20:
            rolling_sharpe = returns.rolling(window=20).apply(
                lambda x: x.mean() / x.std() if x.std() != 0 else 0
            )
            axes[1, 1].plot(range(len(rolling_sharpe)), rolling_sharpe, 
                           linewidth=2, label=name, alpha=0.8)
    
    axes[1, 1].set_xlabel('Trade Number')
    axes[1, 1].set_ylabel('Rolling Sharpe Ratio (20-period)')
    axes[1, 1].set_title('Rolling Risk-Adjusted Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_chart_collection(
    charts: Dict[str, plt.Figure],
    output_dir: str,
    format: str = 'png'
) -> Dict[str, str]:
    """
    Save a collection of charts to files.
    
    Args:
        charts: Dictionary with chart names as keys and figure objects as values
        output_dir: Directory to save charts
        format: File format ('png', 'pdf', 'svg')
        
    Returns:
        Dictionary with chart names and saved file paths
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    for chart_name, figure in charts.items():
        filename = f"{chart_name}.{format}"
        filepath = os.path.join(output_dir, filename)
        
        try:
            figure.savefig(filepath, dpi=300, bbox_inches='tight', format=format)
            saved_files[chart_name] = filepath
        except Exception as e:
            print(f"Error saving chart {chart_name}: {e}")
            continue
        finally:
            plt.close(figure)  # Close figure to free memory
    
    return saved_files