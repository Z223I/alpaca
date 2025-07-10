"""
Alert Performance Analysis Molecule.

This molecule combines analysis, metrics, and simulation atoms to provide
comprehensive alert performance analysis functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
from datetime import datetime

# Import atoms
from atoms.analysis.filter_trading_hours import filter_trading_hours
from atoms.analysis.filter_alert_hours import filter_alert_hours
from atoms.analysis.align_timestamps import align_alerts_to_market_data
from atoms.analysis.validate_data import validate_market_data, validate_alert_data
from atoms.analysis.statistical_analysis import (
    calculate_alert_performance_statistics, analyze_return_distribution,
    analyze_priority_performance, calculate_correlation_matrix
)
from atoms.metrics.success_rate import (
    calculate_success_rate, calculate_success_rate_by_group,
    calculate_alert_performance_summary
)
from atoms.metrics.calculate_returns import (
    calculate_trade_returns, calculate_risk_adjusted_return,
    calculate_maximum_drawdown, calculate_cumulative_returns
)
from atoms.metrics.risk_metrics import (
    calculate_value_at_risk, calculate_advanced_sharpe_metrics,
    calculate_downside_risk_metrics, calculate_tail_risk_metrics,
    calculate_rolling_risk_metrics
)
from atoms.simulation.trade_executor import simulate_multiple_trades
from molecules.performance_reporter import PerformanceReporter


class AlertAnalyzer:
    """
    Main analyzer class for alert performance analysis.
    
    This molecule combines various atoms to provide comprehensive
    alert performance analysis capabilities.
    """
    
    def __init__(self, 
                 trading_hours_start: str = "09:30",
                 trading_hours_end: str = "16:00",
                 alert_hours_start: str = "09:30", 
                 alert_hours_end: str = "15:30",
                 timezone: str = "US/Eastern"):
        """
        Initialize the AlertAnalyzer.
        
        Args:
            trading_hours_start: Start time for trading hours (HH:MM format)
            trading_hours_end: End time for trading hours (HH:MM format)
            alert_hours_start: Start time for valid alerts (HH:MM format)
            alert_hours_end: End time for valid alerts (HH:MM format)
            timezone: Timezone for time filtering
        """
        self.trading_hours_start = trading_hours_start
        self.trading_hours_end = trading_hours_end
        self.alert_hours_start = alert_hours_start
        self.alert_hours_end = alert_hours_end
        self.timezone = timezone
        
        # Data storage
        self.market_data = pd.DataFrame()
        self.alerts_data = pd.DataFrame()
        self.simulation_results = pd.DataFrame()
        self.analysis_results = {}
        
    def load_data(self, 
                  market_data_path: str,
                  alerts_data_path: str,
                  date: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and validate market data and alerts data.
        
        Args:
            market_data_path: Path to market data directory
            alerts_data_path: Path to alerts data directory
            date: Specific date to analyze (YYYY-MM-DD format)
            
        Returns:
            Dictionary with load results and validation status
        """
        try:
            # Load market data
            market_data_files = self._find_data_files(market_data_path, date)
            self.market_data = self._load_market_data_files(market_data_files)
            
            # Load alerts data
            alerts_data_files = self._find_data_files(alerts_data_path, date)
            self.alerts_data = self._load_alerts_data_files(alerts_data_files)
            
            # Validate data
            market_validation = validate_market_data(self.market_data)
            alerts_validation = validate_alert_data(self.alerts_data)
            
            # Filter to trading/alert hours
            self.market_data = filter_trading_hours(
                self.market_data, 
                self.trading_hours_start, 
                self.trading_hours_end,
                self.timezone
            )
            
            self.alerts_data = filter_alert_hours(
                self.alerts_data,
                self.alert_hours_start,
                self.alert_hours_end, 
                self.timezone
            )
            
            return {
                'status': 'success',
                'market_data_count': len(self.market_data),
                'alerts_count': len(self.alerts_data),
                'market_validation': market_validation,
                'alerts_validation': alerts_validation
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'market_data_count': 0,
                'alerts_count': 0
            }
    
    def run_simulation(self, 
                      stop_loss_pct: float = 7.5,
                      take_profit_pct: float = 15.0,
                      max_duration_hours: int = 24) -> Dict[str, Any]:
        """
        Run trade simulation for all alerts.
        
        Args:
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
            max_duration_hours: Maximum trade duration
            
        Returns:
            Dictionary with simulation results
        """
        if self.market_data.empty or self.alerts_data.empty:
            return {
                'status': 'error',
                'error': 'No data loaded for simulation'
            }
        
        try:
            # Align alerts with market data
            aligned_data = align_alerts_to_market_data(
                self.alerts_data, 
                self.market_data
            )
            
            # Run simulation
            self.simulation_results = simulate_multiple_trades(
                aligned_data,
                self.market_data
            )
            
            return {
                'status': 'success',
                'simulated_trades': len(self.simulation_results),
                'successful_trades': len(self.simulation_results[
                    self.simulation_results['status'] == 'SUCCESS'
                ]),
                'failed_trades': len(self.simulation_results[
                    self.simulation_results['status'] == 'FAILED'
                ])
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with all performance metrics
        """
        if self.simulation_results.empty:
            return {
                'status': 'error',
                'error': 'No simulation results available'
            }
        
        try:
            # Basic performance summary
            performance_summary = calculate_alert_performance_summary(
                self.simulation_results
            )
            
            # Success rates by group
            symbol_performance = calculate_success_rate_by_group(
                self.simulation_results, 'symbol'
            )
            
            priority_performance = calculate_success_rate_by_group(
                self.simulation_results, 'priority'
            ) if 'priority' in self.simulation_results.columns else pd.DataFrame()
            
            # Calculate returns
            returns = calculate_trade_returns(self.simulation_results)
            cumulative_returns = calculate_cumulative_returns(returns)
            
            # Risk-adjusted metrics
            risk_metrics = calculate_risk_adjusted_return(returns)
            
            # Maximum drawdown
            max_drawdown = calculate_maximum_drawdown(cumulative_returns)
            
            self.analysis_results = {
                'status': 'success',
                'summary': performance_summary,
                'symbol_performance': symbol_performance.to_dict('records'),
                'priority_performance': priority_performance.to_dict('records'),
                'returns_metrics': {
                    'total_return': cumulative_returns.iloc[-1] if not cumulative_returns.empty else 0,
                    'average_return': returns.mean() if not returns.empty else 0,
                    'return_std': returns.std() if not returns.empty else 0,
                    'max_drawdown': max_drawdown
                },
                'risk_metrics': risk_metrics,
                'trade_count': len(self.simulation_results)
            }
            
            return self.analysis_results
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_alerts(self, 
                      market_data_path: str,
                      alerts_data_path: str,
                      date: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete end-to-end alert analysis.
        
        Args:
            market_data_path: Path to market data directory
            alerts_data_path: Path to alerts data directory
            date: Specific date to analyze
            
        Returns:
            Dictionary with complete analysis results
        """
        # Load data
        load_result = self.load_data(market_data_path, alerts_data_path, date)
        if load_result['status'] != 'success':
            return load_result
        
        # Run simulation
        simulation_result = self.run_simulation()
        if simulation_result['status'] != 'success':
            return simulation_result
        
        # Calculate metrics
        metrics_result = self.calculate_performance_metrics()
        if metrics_result['status'] != 'success':
            return metrics_result
        
        # Combine results
        complete_results = {
            'status': 'success',
            'data_load': load_result,
            'simulation': simulation_result,
            'performance': metrics_result
        }
        
        return complete_results
    
    def export_results(self, output_path: str) -> bool:
        """
        Export analysis results to JSON file.
        
        Args:
            output_path: Path to output JSON file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            output_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'configuration': {
                    'trading_hours': f"{self.trading_hours_start}-{self.trading_hours_end}",
                    'alert_hours': f"{self.alert_hours_start}-{self.alert_hours_end}",
                    'timezone': self.timezone
                },
                'results': self.analysis_results
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def _find_data_files(self, data_path: str, date: Optional[str] = None) -> List[str]:
        """
        Find data files for the specified date or most recent date.
        
        Args:
            data_path: Base path to data directory
            date: Specific date (YYYY-MM-DD) or None for most recent
            
        Returns:
            List of data file paths
        """
        base_path = Path(data_path)
        
        if date:
            # Look for specific date
            date_path = base_path / date
            if date_path.exists():
                return [str(f) for f in date_path.glob("*.json")]
        else:
            # Find most recent date
            if not base_path.exists():
                return []
            
            date_dirs = [d for d in base_path.iterdir() if d.is_dir()]
            if not date_dirs:
                return []
            
            # Sort by name (assuming YYYY-MM-DD format)
            date_dirs.sort(key=lambda x: x.name, reverse=True)
            most_recent = date_dirs[0]
            return [str(f) for f in most_recent.glob("*.json")]
        
        return []
    
    def _load_market_data_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Load market data from multiple JSON files."""
        all_data = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # Standardize column names and types
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _load_alerts_data_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Load alerts data from multiple JSON files."""
        all_data = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        # Standardize column names and types
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def calculate_advanced_risk_metrics(self, 
                                      confidence_levels: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Calculate advanced risk metrics for the simulation results.
        
        Args:
            confidence_levels: List of confidence levels for VaR calculation
            
        Returns:
            Dictionary with advanced risk metrics
        """
        if self.simulation_results.empty or 'return_pct' not in self.simulation_results.columns:
            return {'error': 'No simulation results available for risk analysis'}
        
        returns = self.simulation_results['return_pct'].dropna()
        
        if returns.empty:
            return {'error': 'No valid returns for risk analysis'}
        
        risk_metrics = {}
        
        # Advanced Sharpe metrics
        risk_metrics['sharpe_metrics'] = calculate_advanced_sharpe_metrics(returns)
        
        # Value at Risk analysis
        if confidence_levels is None:
            confidence_levels = [0.01, 0.05, 0.10]
        
        risk_metrics['var_analysis'] = {}
        for confidence in confidence_levels:
            var_result = calculate_value_at_risk(returns, confidence)
            risk_metrics['var_analysis'][f'var_{int(confidence*100)}'] = var_result
        
        # Downside risk metrics
        risk_metrics['downside_risk'] = calculate_downside_risk_metrics(returns)
        
        # Tail risk analysis
        risk_metrics['tail_risk'] = calculate_tail_risk_metrics(returns)
        
        # Rolling risk metrics (if enough data)
        if len(returns) >= 30:
            rolling_metrics = calculate_rolling_risk_metrics(returns, window=30)
            if not rolling_metrics.empty:
                risk_metrics['rolling_metrics'] = {
                    'latest_volatility': rolling_metrics['volatility'].iloc[-1] if 'volatility' in rolling_metrics else None,
                    'latest_sharpe': rolling_metrics['sharpe_ratio'].iloc[-1] if 'sharpe_ratio' in rolling_metrics else None,
                    'volatility_trend': rolling_metrics['volatility'].tolist()[-10:] if 'volatility' in rolling_metrics else []
                }
        
        return risk_metrics
    
    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on simulation results.
        
        Returns:
            Dictionary with statistical analysis results
        """
        if self.simulation_results.empty:
            return {'error': 'No simulation results available for statistical analysis'}
        
        # Comprehensive performance statistics
        stats_result = calculate_alert_performance_statistics(
            self.simulation_results,
            return_col='return_pct',
            status_col='status',
            priority_col='priority'
        )
        
        # Return distribution analysis
        if 'return_pct' in self.simulation_results.columns:
            returns = self.simulation_results['return_pct'].dropna()
            if not returns.empty:
                stats_result['distribution_analysis'] = analyze_return_distribution(returns)
        
        # Priority performance analysis
        if 'priority' in self.simulation_results.columns:
            stats_result['priority_comparison'] = analyze_priority_performance(
                self.simulation_results,
                'return_pct',
                'status',
                'priority'
            )
        
        # Correlation analysis
        numeric_cols = self.simulation_results.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            corr_matrix, p_matrix = calculate_correlation_matrix(
                self.simulation_results,
                numeric_cols
            )
            if not corr_matrix.empty:
                stats_result['correlation_analysis'] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'p_value_matrix': p_matrix.to_dict(),
                    'significant_correlations': self._identify_significant_correlations(corr_matrix, p_matrix)
                }
        
        return stats_result
    
    def generate_comprehensive_report(self, 
                                    output_dir: str = "reports",
                                    include_charts: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report with visualizations.
        
        Args:
            output_dir: Directory to save reports and charts
            include_charts: Whether to generate visualization charts
            
        Returns:
            Dictionary with report generation results
        """
        if self.simulation_results.empty:
            return {'error': 'No simulation results available for report generation'}
        
        # Initialize performance reporter
        reporter = PerformanceReporter(
            output_dir=output_dir,
            include_charts=include_charts
        )
        
        # Generate comprehensive report
        report_data = reporter.generate_comprehensive_report(
            self.simulation_results,
            return_col='return_pct',
            status_col='status',
            priority_col='priority',
            symbol_col='symbol',
            timestamp_col='alert_time'
        )
        
        # Export report
        if report_data.get('status') != 'error':
            json_path = reporter.export_report(format='json')
            txt_path = reporter.export_report(format='txt')
            
            # Create executive summary
            executive_summary = reporter.create_executive_summary()
            
            return {
                'status': 'success',
                'report_data': report_data,
                'executive_summary': executive_summary,
                'exports': {
                    'json_report': json_path,
                    'text_report': txt_path,
                    'charts': report_data.get('charts', {})
                }
            }
        
        return report_data
    
    def _identify_significant_correlations(self, 
                                         corr_matrix: pd.DataFrame,
                                         p_matrix: pd.DataFrame,
                                         significance_level: float = 0.05) -> List[Dict[str, Any]]:
        """Identify statistically significant correlations."""
        significant_correlations = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlations
                    corr_value = corr_matrix.loc[col1, col2]
                    p_value = p_matrix.loc[col1, col2]
                    
                    if p_value < significance_level and abs(corr_value) > 0.3:
                        significant_correlations.append({
                            'variable_1': col1,
                            'variable_2': col2,
                            'correlation': corr_value,
                            'p_value': p_value,
                            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                        })
        
        # Sort by absolute correlation strength
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return significant_correlations