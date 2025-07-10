"""
Performance Reporting Molecule.

This molecule combines analysis, visualization, and export atoms to provide
comprehensive performance reporting and visualization capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import atoms
from atoms.metrics.risk_metrics import (
    calculate_value_at_risk, calculate_advanced_sharpe_metrics,
    calculate_downside_risk_metrics, calculate_tail_risk_metrics,
    calculate_rolling_risk_metrics
)
from atoms.analysis.statistical_analysis import (
    calculate_alert_performance_statistics, analyze_return_distribution,
    analyze_priority_performance, calculate_correlation_matrix
)
from atoms.visualization.chart_generators import (
    create_performance_summary_chart, create_risk_analysis_chart,
    create_correlation_heatmap, create_time_series_chart,
    create_performance_comparison_chart, save_chart_collection
)
from atoms.metrics.success_rate import calculate_alert_performance_summary
from atoms.metrics.calculate_returns import calculate_cumulative_returns, calculate_maximum_drawdown


class PerformanceReporter:
    """
    Advanced performance reporting and visualization molecule.
    
    This molecule combines various analysis and visualization atoms to generate
    comprehensive performance reports with charts and statistical analysis.
    """
    
    def __init__(self, 
                 output_dir: str = "reports",
                 chart_format: str = "png",
                 include_charts: bool = True):
        """
        Initialize the PerformanceReporter.
        
        Args:
            output_dir: Directory for saving reports and charts
            chart_format: Format for saving charts ('png', 'pdf', 'svg')
            include_charts: Whether to generate charts in reports
        """
        self.output_dir = Path(output_dir)
        self.chart_format = chart_format
        self.include_charts = include_charts
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        if self.include_charts:
            (self.output_dir / "charts").mkdir(exist_ok=True)
        
        # Storage for report data
        self.report_data = {}
        self.charts = {}
        
    def generate_comprehensive_report(self,
                                    alerts_df: pd.DataFrame,
                                    return_col: str = "return_pct",
                                    status_col: str = "status",
                                    priority_col: str = "priority",
                                    symbol_col: str = "symbol",
                                    timestamp_col: str = "timestamp") -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            alerts_df: DataFrame containing alert performance data
            return_col: Column name for returns
            status_col: Column name for trade status
            priority_col: Column name for priority levels
            symbol_col: Column name for symbols
            timestamp_col: Column name for timestamps
            
        Returns:
            Dictionary containing complete report data
        """
        if alerts_df.empty:
            return {'error': 'No data provided for report generation'}
        
        print("Generating comprehensive performance report...")
        
        # Initialize report structure
        self.report_data = {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'total_alerts': len(alerts_df),
                'date_range': self._get_date_range(alerts_df, timestamp_col),
                'symbols_analyzed': alerts_df[symbol_col].nunique() if symbol_col in alerts_df.columns else 0
            },
            'summary_metrics': {},
            'risk_analysis': {},
            'statistical_analysis': {},
            'performance_breakdown': {},
            'charts': {}
        }
        
        # 1. Generate summary metrics
        self.report_data['summary_metrics'] = self._generate_summary_metrics(
            alerts_df, return_col, status_col
        )
        
        # 2. Perform risk analysis
        if return_col in alerts_df.columns:
            returns = alerts_df[return_col].dropna()
            self.report_data['risk_analysis'] = self._generate_risk_analysis(returns)
        
        # 3. Statistical analysis
        self.report_data['statistical_analysis'] = self._generate_statistical_analysis(
            alerts_df, return_col, status_col, priority_col
        )
        
        # 4. Performance breakdown
        self.report_data['performance_breakdown'] = self._generate_performance_breakdown(
            alerts_df, return_col, status_col, priority_col, symbol_col, timestamp_col
        )
        
        # 5. Generate charts if requested
        if self.include_charts:
            self.charts = self._generate_all_charts(alerts_df, return_col, priority_col, timestamp_col)
            self.report_data['charts'] = self._save_charts()
        
        print("Report generation completed successfully.")
        return self.report_data
    
    def _generate_summary_metrics(self,
                                alerts_df: pd.DataFrame,
                                return_col: str,
                                status_col: str) -> Dict[str, Any]:
        """Generate high-level summary metrics."""
        summary = {}
        
        # Basic performance summary
        basic_summary = calculate_alert_performance_summary(
            alerts_df, return_col, status_col
        )
        summary.update(basic_summary)
        
        # Additional calculations
        if return_col in alerts_df.columns:
            returns = alerts_df[return_col].dropna()
            if not returns.empty:
                # Cumulative performance
                cumulative_returns = calculate_cumulative_returns(returns, compound=True)
                summary['total_return'] = cumulative_returns.iloc[-1] if not cumulative_returns.empty else 0
                summary['best_trade'] = returns.max()
                summary['worst_trade'] = returns.min()
                
                # Consistency metrics
                positive_returns = returns[returns > 0]
                summary['win_percentage'] = len(positive_returns) / len(returns) * 100
                summary['consistency_score'] = (returns > 0).rolling(window=10).mean().mean() * 100
        
        return summary
    
    def _generate_risk_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """Generate comprehensive risk analysis."""
        risk_analysis = {}
        
        # Advanced Sharpe metrics
        risk_analysis['sharpe_metrics'] = calculate_advanced_sharpe_metrics(returns)
        
        # Value at Risk analysis
        risk_analysis['var_analysis'] = {
            'var_1': calculate_value_at_risk(returns, 0.01),
            'var_5': calculate_value_at_risk(returns, 0.05),
            'var_10': calculate_value_at_risk(returns, 0.10)
        }
        
        # Downside risk metrics
        risk_analysis['downside_risk'] = calculate_downside_risk_metrics(returns)
        
        # Tail risk analysis
        risk_analysis['tail_risk'] = calculate_tail_risk_metrics(returns)
        
        # Rolling risk metrics (if enough data)
        if len(returns) >= 30:
            rolling_metrics = calculate_rolling_risk_metrics(returns, window=30)
            if not rolling_metrics.empty:
                risk_analysis['rolling_metrics'] = {
                    'volatility_trend': rolling_metrics['volatility'].tolist()[-10:] if 'volatility' in rolling_metrics else [],
                    'sharpe_trend': rolling_metrics['sharpe_ratio'].tolist()[-10:] if 'sharpe_ratio' in rolling_metrics else []
                }
        
        return risk_analysis
    
    def _generate_statistical_analysis(self,
                                     alerts_df: pd.DataFrame,
                                     return_col: str,
                                     status_col: str,
                                     priority_col: str) -> Dict[str, Any]:
        """Generate statistical analysis."""
        return calculate_alert_performance_statistics(
            alerts_df, return_col, status_col, priority_col
        )
    
    def _generate_performance_breakdown(self,
                                      alerts_df: pd.DataFrame,
                                      return_col: str,
                                      status_col: str,
                                      priority_col: str,
                                      symbol_col: str,
                                      timestamp_col: str) -> Dict[str, Any]:
        """Generate detailed performance breakdown."""
        breakdown = {}
        
        # Performance by priority
        if priority_col in alerts_df.columns:
            breakdown['by_priority'] = {}
            for priority in alerts_df[priority_col].unique():
                priority_data = alerts_df[alerts_df[priority_col] == priority]
                if not priority_data.empty:
                    breakdown['by_priority'][priority] = self._calculate_subset_metrics(
                        priority_data, return_col, status_col
                    )
        
        # Performance by symbol (top 10)
        if symbol_col in alerts_df.columns:
            breakdown['by_symbol'] = {}
            symbol_counts = alerts_df[symbol_col].value_counts()
            top_symbols = symbol_counts.head(10).index
            
            for symbol in top_symbols:
                symbol_data = alerts_df[alerts_df[symbol_col] == symbol]
                breakdown['by_symbol'][symbol] = self._calculate_subset_metrics(
                    symbol_data, return_col, status_col
                )
        
        # Temporal performance
        if timestamp_col in alerts_df.columns:
            breakdown['temporal'] = self._analyze_temporal_performance(
                alerts_df, return_col, timestamp_col
            )
        
        return breakdown
    
    def _calculate_subset_metrics(self,
                                subset_df: pd.DataFrame,
                                return_col: str,
                                status_col: str) -> Dict[str, Any]:
        """Calculate metrics for a subset of data."""
        metrics = {
            'count': len(subset_df),
            'success_rate': 0.0,
            'avg_return': 0.0,
            'total_return': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'volatility': 0.0
        }
        
        if status_col in subset_df.columns:
            metrics['success_rate'] = (subset_df[status_col] == 'SUCCESS').mean() * 100
        
        if return_col in subset_df.columns:
            returns = subset_df[return_col].dropna()
            if not returns.empty:
                metrics['avg_return'] = returns.mean()
                metrics['total_return'] = (1 + returns/100).prod() - 1
                metrics['best_trade'] = returns.max()
                metrics['worst_trade'] = returns.min()
                metrics['volatility'] = returns.std()
        
        return metrics
    
    def _analyze_temporal_performance(self,
                                    alerts_df: pd.DataFrame,
                                    return_col: str,
                                    timestamp_col: str) -> Dict[str, Any]:
        """Analyze performance over time."""
        temporal_analysis = {}
        
        df = alerts_df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Monthly performance
        df['month'] = df[timestamp_col].dt.to_period('M')
        monthly_performance = df.groupby('month')[return_col].agg(['mean', 'count', 'std'])
        temporal_analysis['monthly'] = monthly_performance.to_dict('index')
        
        # Daily performance
        df['day_of_week'] = df[timestamp_col].dt.day_name()
        daily_performance = df.groupby('day_of_week')[return_col].agg(['mean', 'count'])
        temporal_analysis['daily'] = daily_performance.to_dict('index')
        
        # Hourly performance
        df['hour'] = df[timestamp_col].dt.hour
        hourly_performance = df.groupby('hour')[return_col].agg(['mean', 'count'])
        temporal_analysis['hourly'] = hourly_performance.to_dict('index')
        
        return temporal_analysis
    
    def _generate_all_charts(self,
                           alerts_df: pd.DataFrame,
                           return_col: str,
                           priority_col: str,
                           timestamp_col: str) -> Dict[str, plt.Figure]:
        """Generate all visualization charts."""
        charts = {}
        
        try:
            # 1. Performance Summary Chart
            charts['performance_summary'] = create_performance_summary_chart(
                alerts_df, return_col, priority_col
            )
            
            # 2. Risk Analysis Chart
            if return_col in alerts_df.columns:
                returns = alerts_df[return_col].dropna()
                if not returns.empty:
                    risk_metrics = self.report_data.get('risk_analysis', {})
                    charts['risk_analysis'] = create_risk_analysis_chart(returns, risk_metrics)
            
            # 3. Time Series Chart
            if timestamp_col in alerts_df.columns and return_col in alerts_df.columns:
                charts['time_series'] = create_time_series_chart(
                    alerts_df, return_col, timestamp_col, "Alert Performance Over Time"
                )
            
            # 4. Correlation Heatmap (if enough numeric columns)
            numeric_cols = alerts_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                corr_matrix, p_matrix = calculate_correlation_matrix(alerts_df, numeric_cols)
                if not corr_matrix.empty:
                    charts['correlation_heatmap'] = create_correlation_heatmap(corr_matrix, p_matrix)
            
        except Exception as e:
            print(f"Warning: Error generating charts: {e}")
        
        return charts
    
    def _save_charts(self) -> Dict[str, str]:
        """Save all generated charts."""
        if not self.charts:
            return {}
        
        chart_dir = self.output_dir / "charts"
        return save_chart_collection(self.charts, str(chart_dir), self.chart_format)
    
    def _get_date_range(self, alerts_df: pd.DataFrame, timestamp_col: str) -> Dict[str, str]:
        """Get date range from alerts data."""
        if timestamp_col not in alerts_df.columns:
            return {'start': 'N/A', 'end': 'N/A'}
        
        timestamps = pd.to_datetime(alerts_df[timestamp_col])
        return {
            'start': timestamps.min().strftime('%Y-%m-%d %H:%M:%S'),
            'end': timestamps.max().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def export_report(self,
                     filename: Optional[str] = None,
                     format: str = "json") -> str:
        """
        Export the generated report to file.
        
        Args:
            filename: Optional filename (auto-generated if not provided)
            format: Export format ('json', 'txt')
            
        Returns:
            Path to exported file
        """
        if not self.report_data:
            raise ValueError("No report data available. Generate report first.")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.{format}"
        
        filepath = self.output_dir / filename
        
        if format == "json":
            # Custom serializer for numpy types and complex objects
            def clean_for_json(obj):
                """Recursively clean object for JSON serialization."""
                if isinstance(obj, dict):
                    # Convert tuple keys to strings
                    cleaned = {}
                    for k, v in obj.items():
                        if isinstance(k, tuple):
                            k_str = str(k)
                        else:
                            k_str = str(k) if not isinstance(k, (str, int, float, bool)) else k
                        cleaned[k_str] = clean_for_json(v)
                    return cleaned
                elif isinstance(obj, (list, tuple)):
                    return [clean_for_json(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                else:
                    return obj
            
            def numpy_encoder(obj):
                return str(obj)
            
            with open(filepath, 'w') as f:
                cleaned_data = clean_for_json(self.report_data)
                json.dump(cleaned_data, f, indent=2, default=numpy_encoder)
        
        elif format == "txt":
            self._export_text_report(filepath)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return str(filepath)
    
    def _export_text_report(self, filepath: Path):
        """Export report in text format."""
        with open(filepath, 'w') as f:
            f.write("ALERT PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            metadata = self.report_data.get('metadata', {})
            f.write("REPORT METADATA\n")
            f.write("-" * 20 + "\n")
            f.write(f"Generated: {metadata.get('generation_timestamp', 'N/A')}\n")
            f.write(f"Total Alerts: {metadata.get('total_alerts', 0):,}\n")
            f.write(f"Symbols Analyzed: {metadata.get('symbols_analyzed', 0):,}\n")
            f.write(f"Date Range: {metadata.get('date_range', {}).get('start', 'N/A')} to {metadata.get('date_range', {}).get('end', 'N/A')}\n\n")
            
            # Summary Metrics
            summary = self.report_data.get('summary_metrics', {})
            f.write("SUMMARY METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Success Rate: {summary.get('success_rate', 0):.2f}%\n")
            f.write(f"Average Return: {summary.get('average_return', 0):.2f}%\n")
            f.write(f"Total Return: {summary.get('total_return', 0):.2f}%\n")
            f.write(f"Best Trade: {summary.get('best_trade', 0):.2f}%\n")
            f.write(f"Worst Trade: {summary.get('worst_trade', 0):.2f}%\n")
            f.write(f"Win Percentage: {summary.get('win_percentage', 0):.2f}%\n")
            f.write(f"Profit Factor: {summary.get('profit_factor', 0):.2f}\n\n")
            
            # Risk Analysis
            risk_analysis = self.report_data.get('risk_analysis', {})
            if risk_analysis:
                f.write("RISK ANALYSIS\n")
                f.write("-" * 20 + "\n")
                
                sharpe_metrics = risk_analysis.get('sharpe_metrics', {})
                f.write(f"Sharpe Ratio: {sharpe_metrics.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"Sortino Ratio: {sharpe_metrics.get('sortino_ratio', 0):.3f}\n")
                f.write(f"Calmar Ratio: {sharpe_metrics.get('calmar_ratio', 0):.3f}\n")
                
                var_5 = risk_analysis.get('var_analysis', {}).get('var_5', {})
                f.write(f"VaR (5%): {var_5.get('var', 0):.2f}%\n")
                f.write(f"CVaR (5%): {var_5.get('cvar', 0):.2f}%\n\n")
            
            # Performance Breakdown
            breakdown = self.report_data.get('performance_breakdown', {})
            
            # By Priority
            if 'by_priority' in breakdown:
                f.write("PERFORMANCE BY PRIORITY\n")
                f.write("-" * 30 + "\n")
                for priority, metrics in breakdown['by_priority'].items():
                    f.write(f"{priority}: {metrics.get('success_rate', 0):.1f}% success, "
                           f"{metrics.get('avg_return', 0):.2f}% avg return "
                           f"({metrics.get('count', 0)} alerts)\n")
                f.write("\n")
            
            # Top Symbols
            if 'by_symbol' in breakdown:
                f.write("TOP PERFORMING SYMBOLS\n")
                f.write("-" * 30 + "\n")
                symbol_items = list(breakdown['by_symbol'].items())
                # Sort by success rate
                symbol_items.sort(key=lambda x: x[1].get('success_rate', 0), reverse=True)
                
                for symbol, metrics in symbol_items[:10]:
                    f.write(f"{symbol}: {metrics.get('success_rate', 0):.1f}% success, "
                           f"{metrics.get('avg_return', 0):.2f}% avg return "
                           f"({metrics.get('count', 0)} alerts)\n")
                f.write("\n")
            
            # Charts
            if self.report_data.get('charts'):
                f.write("GENERATED CHARTS\n")
                f.write("-" * 20 + "\n")
                for chart_name, chart_path in self.report_data['charts'].items():
                    f.write(f"{chart_name}: {chart_path}\n")
    
    def create_executive_summary(self) -> Dict[str, Any]:
        """
        Create a concise executive summary of the performance report.
        
        Returns:
            Dictionary with executive summary
        """
        if not self.report_data:
            return {'error': 'No report data available'}
        
        summary = self.report_data.get('summary_metrics', {})
        risk_analysis = self.report_data.get('risk_analysis', {})
        metadata = self.report_data.get('metadata', {})
        
        executive_summary = {
            'key_metrics': {
                'total_alerts': metadata.get('total_alerts', 0),
                'success_rate': summary.get('success_rate', 0),
                'total_return': summary.get('total_return', 0),
                'average_return': summary.get('average_return', 0),
                'profit_factor': summary.get('profit_factor', 0)
            },
            'risk_profile': {
                'sharpe_ratio': risk_analysis.get('sharpe_metrics', {}).get('sharpe_ratio', 0),
                'max_drawdown': risk_analysis.get('sharpe_metrics', {}).get('max_drawdown', 0),
                'var_5_percent': risk_analysis.get('var_analysis', {}).get('var_5', {}).get('var', 0)
            },
            'recommendations': self._generate_recommendations()
        }
        
        return executive_summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        summary = self.report_data.get('summary_metrics', {})
        risk_analysis = self.report_data.get('risk_analysis', {})
        
        # Performance recommendations
        success_rate = summary.get('success_rate', 0)
        if success_rate < 50:
            recommendations.append("Consider reviewing alert criteria - success rate below 50%")
        elif success_rate > 70:
            recommendations.append("Strong alert performance - consider increasing position sizes")
        
        # Risk recommendations
        sharpe_ratio = risk_analysis.get('sharpe_metrics', {}).get('sharpe_ratio', 0)
        if sharpe_ratio < 1.0:
            recommendations.append("Risk-adjusted returns could be improved - review stop-loss levels")
        elif sharpe_ratio > 2.0:
            recommendations.append("Excellent risk-adjusted performance - strategy is working well")
        
        # Consistency recommendations
        consistency_score = summary.get('consistency_score', 0)
        if consistency_score < 40:
            recommendations.append("Performance consistency is low - consider smoothing parameters")
        
        return recommendations