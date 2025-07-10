"""
Dashboard generator atoms for real-time performance visualization.

This atom provides HTML dashboard generation capabilities for monitoring
system performance, alerts, and trading metrics in real-time.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Template
import base64
import io


class DashboardGenerator:
    """
    Generate HTML dashboards for real-time monitoring.
    
    Creates interactive dashboards with charts, metrics, and alerts
    for system performance and trading analysis.
    """
    
    def __init__(self, 
                 template_dir: str = "templates",
                 output_dir: str = "dashboard"):
        """
        Initialize the dashboard generator.
        
        Args:
            template_dir: Directory containing HTML templates
            output_dir: Directory for output dashboard files
        """
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Default color scheme
        self.color_scheme = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'danger': '#DC3545',
            'info': '#17A2B8',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
        
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'responsive': True
        }
    
    def generate_performance_dashboard(self, 
                                    performance_data: Dict[str, Any],
                                    output_filename: str = "performance_dashboard.html") -> str:
        """
        Generate a comprehensive performance dashboard.
        
        Args:
            performance_data: Dictionary containing performance metrics
            output_filename: Name of the output HTML file
            
        Returns:
            Path to the generated dashboard file
        """
        # Create charts
        charts = self._create_performance_charts(performance_data)
        
        # Prepare dashboard data
        dashboard_data = {
            'title': 'Alert Performance Dashboard',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'charts': charts,
            'metrics': self._format_key_metrics(performance_data),
            'alerts': self._format_recent_alerts(performance_data),
            'system_health': self._format_system_health(performance_data)
        }
        
        # Generate HTML
        html_content = self._generate_html_dashboard(dashboard_data)
        
        # Save dashboard
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def generate_trading_dashboard(self, 
                                 trading_data: Dict[str, Any],
                                 output_filename: str = "trading_dashboard.html") -> str:
        """
        Generate a trading-focused dashboard.
        
        Args:
            trading_data: Dictionary containing trading metrics
            output_filename: Name of the output HTML file
            
        Returns:
            Path to the generated dashboard file
        """
        # Create trading-specific charts
        charts = self._create_trading_charts(trading_data)
        
        # Prepare dashboard data
        dashboard_data = {
            'title': 'Trading Performance Dashboard',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'charts': charts,
            'metrics': self._format_trading_metrics(trading_data),
            'positions': self._format_positions(trading_data),
            'risk_metrics': self._format_risk_metrics(trading_data)
        }
        
        # Generate HTML
        html_content = self._generate_html_dashboard(dashboard_data, template_type='trading')
        
        # Save dashboard
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def generate_real_time_dashboard(self, 
                                   data_source: callable,
                                   refresh_interval: int = 30,
                                   output_filename: str = "realtime_dashboard.html") -> str:
        """
        Generate a real-time dashboard with auto-refresh capability.
        
        Args:
            data_source: Callable that returns current data
            refresh_interval: Refresh interval in seconds
            output_filename: Name of the output HTML file
            
        Returns:
            Path to the generated dashboard file
        """
        # Get initial data
        current_data = data_source()
        
        # Create charts
        charts = self._create_realtime_charts(current_data)
        
        # Prepare dashboard data
        dashboard_data = {
            'title': 'Real-time Monitoring Dashboard',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'charts': charts,
            'metrics': self._format_realtime_metrics(current_data),
            'refresh_interval': refresh_interval
        }
        
        # Generate HTML with auto-refresh
        html_content = self._generate_html_dashboard(dashboard_data, template_type='realtime')
        
        # Save dashboard
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _create_performance_charts(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create performance monitoring charts."""
        charts = {}
        
        # Success rate over time
        if 'success_rate_history' in data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['success_rate_history']['timestamps'],
                y=data['success_rate_history']['values'],
                mode='lines+markers',
                name='Success Rate',
                line=dict(color=self.color_scheme['success'])
            ))
            fig.update_layout(
                title='Success Rate Over Time',
                xaxis_title='Time',
                yaxis_title='Success Rate (%)',
                height=300
            )
            charts['success_rate'] = fig.to_html(include_plotlyjs='cdn', div_id='success-rate-chart')
        
        # Return distribution
        if 'returns' in data:
            returns = data['returns']
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns,
                name='Returns',
                nbinsx=50,
                marker_color=self.color_scheme['primary']
            ))
            fig.update_layout(
                title='Return Distribution',
                xaxis_title='Return (%)',
                yaxis_title='Frequency',
                height=300
            )
            charts['return_distribution'] = fig.to_html(include_plotlyjs='cdn', div_id='return-dist-chart')
        
        # Performance by symbol
        if 'symbol_performance' in data:
            symbols = list(data['symbol_performance'].keys())
            success_rates = [data['symbol_performance'][s]['success_rate'] for s in symbols]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=symbols,
                y=success_rates,
                name='Success Rate by Symbol',
                marker_color=self.color_scheme['info']
            ))
            fig.update_layout(
                title='Performance by Symbol',
                xaxis_title='Symbol',
                yaxis_title='Success Rate (%)',
                height=300
            )
            charts['symbol_performance'] = fig.to_html(include_plotlyjs='cdn', div_id='symbol-perf-chart')
        
        return charts
    
    def _create_trading_charts(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create trading-specific charts."""
        charts = {}
        
        # Equity curve
        if 'equity_curve' in data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['equity_curve']['dates'],
                y=data['equity_curve']['values'],
                mode='lines',
                name='Equity',
                line=dict(color=self.color_scheme['primary'])
            ))
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Equity ($)',
                height=300
            )
            charts['equity_curve'] = fig.to_html(include_plotlyjs='cdn', div_id='equity-curve-chart')
        
        # Drawdown chart
        if 'drawdown' in data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['drawdown']['dates'],
                y=data['drawdown']['values'],
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color=self.color_scheme['danger'])
            ))
            fig.update_layout(
                title='Drawdown Analysis',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                height=300
            )
            charts['drawdown'] = fig.to_html(include_plotlyjs='cdn', div_id='drawdown-chart')
        
        # Risk metrics radar
        if 'risk_metrics' in data:
            metrics = data['risk_metrics']
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=[metrics.get('sharpe_ratio', 0) * 10,
                   metrics.get('sortino_ratio', 0) * 10,
                   100 - abs(metrics.get('max_drawdown', 0)),
                   metrics.get('win_rate', 0),
                   metrics.get('profit_factor', 0) * 20],
                theta=['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor'],
                fill='toself',
                name='Risk Metrics'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title='Risk Metrics Overview',
                height=400
            )
            charts['risk_radar'] = fig.to_html(include_plotlyjs='cdn', div_id='risk-radar-chart')
        
        return charts
    
    def _create_realtime_charts(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create real-time monitoring charts."""
        charts = {}
        
        # System health gauge
        if 'system_health' in data:
            health = data['system_health']
            
            # CPU usage gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health.get('cpu_usage', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': self.color_scheme['primary']},
                       'steps': [
                           {'range': [0, 50], 'color': self.color_scheme['success']},
                           {'range': [50, 80], 'color': self.color_scheme['warning']},
                           {'range': [80, 100], 'color': self.color_scheme['danger']}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            fig.update_layout(height=250)
            charts['cpu_gauge'] = fig.to_html(include_plotlyjs='cdn', div_id='cpu-gauge-chart')
            
            # Memory usage gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health.get('memory_usage', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Memory Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': self.color_scheme['info']},
                       'steps': [
                           {'range': [0, 50], 'color': self.color_scheme['success']},
                           {'range': [50, 80], 'color': self.color_scheme['warning']},
                           {'range': [80, 100], 'color': self.color_scheme['danger']}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}))
            fig.update_layout(height=250)
            charts['memory_gauge'] = fig.to_html(include_plotlyjs='cdn', div_id='memory-gauge-chart')
        
        # Recent alerts timeline
        if 'recent_alerts' in data:
            alerts = data['recent_alerts']
            if alerts:
                fig = go.Figure()
                
                severity_colors = {
                    'critical': self.color_scheme['danger'],
                    'warning': self.color_scheme['warning'],
                    'info': self.color_scheme['info']
                }
                
                for severity in ['critical', 'warning', 'info']:
                    severity_alerts = [a for a in alerts if a.get('severity') == severity]
                    if severity_alerts:
                        fig.add_trace(go.Scatter(
                            x=[a['timestamp'] for a in severity_alerts],
                            y=[severity] * len(severity_alerts),
                            mode='markers',
                            name=severity.capitalize(),
                            marker=dict(
                                color=severity_colors[severity],
                                size=10
                            ),
                            text=[a['title'] for a in severity_alerts],
                            hovertemplate='<b>%{text}</b><br>Time: %{x}<extra></extra>'
                        ))
                
                fig.update_layout(
                    title='Recent Alerts Timeline',
                    xaxis_title='Time',
                    yaxis_title='Severity',
                    height=200
                )
                charts['alerts_timeline'] = fig.to_html(include_plotlyjs='cdn', div_id='alerts-timeline-chart')
        
        return charts
    
    def _format_key_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format key metrics for display."""
        metrics = []
        
        if 'summary_metrics' in data:
            summary = data['summary_metrics']
            
            metrics.append({
                'title': 'Success Rate',
                'value': f"{summary.get('success_rate', 0):.1f}%",
                'icon': 'fas fa-check-circle',
                'color': 'success' if summary.get('success_rate', 0) > 60 else 'warning'
            })
            
            metrics.append({
                'title': 'Total Return',
                'value': f"{summary.get('total_return', 0):.2f}%",
                'icon': 'fas fa-chart-line',
                'color': 'success' if summary.get('total_return', 0) > 0 else 'danger'
            })
            
            metrics.append({
                'title': 'Sharpe Ratio',
                'value': f"{summary.get('sharpe_ratio', 0):.2f}",
                'icon': 'fas fa-balance-scale',
                'color': 'success' if summary.get('sharpe_ratio', 0) > 1 else 'info'
            })
            
            metrics.append({
                'title': 'Max Drawdown',
                'value': f"{summary.get('max_drawdown', 0):.1f}%",
                'icon': 'fas fa-arrow-down',
                'color': 'danger' if summary.get('max_drawdown', 0) > 20 else 'warning'
            })
        
        return metrics
    
    def _format_recent_alerts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format recent alerts for display."""
        alerts = []
        
        if 'recent_alerts' in data:
            for alert in data['recent_alerts'][:10]:  # Show last 10 alerts
                alerts.append({
                    'title': alert.get('title', 'Unknown Alert'),
                    'severity': alert.get('severity', 'info'),
                    'timestamp': alert.get('timestamp', ''),
                    'content': alert.get('content', '')
                })
        
        return alerts
    
    def _format_system_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format system health metrics."""
        health = {
            'status': 'healthy',
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'uptime': '0 days'
        }
        
        if 'system_health' in data:
            sys_health = data['system_health']
            health.update({
                'cpu_usage': sys_health.get('cpu_usage', 0),
                'memory_usage': sys_health.get('memory_usage', 0),
                'disk_usage': sys_health.get('disk_usage', 0),
                'uptime': sys_health.get('uptime', '0 days')
            })
            
            # Determine overall status
            if any(usage > 90 for usage in [health['cpu_usage'], health['memory_usage'], health['disk_usage']]):
                health['status'] = 'critical'
            elif any(usage > 80 for usage in [health['cpu_usage'], health['memory_usage'], health['disk_usage']]):
                health['status'] = 'warning'
        
        return health
    
    def _format_trading_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format trading metrics for display."""
        metrics = []
        
        if 'trading_metrics' in data:
            trading = data['trading_metrics']
            
            metrics.append({
                'title': 'Total Trades',
                'value': str(trading.get('total_trades', 0)),
                'icon': 'fas fa-exchange-alt',
                'color': 'info'
            })
            
            metrics.append({
                'title': 'Win Rate',
                'value': f"{trading.get('win_rate', 0):.1f}%",
                'icon': 'fas fa-trophy',
                'color': 'success' if trading.get('win_rate', 0) > 50 else 'warning'
            })
            
            metrics.append({
                'title': 'Profit Factor',
                'value': f"{trading.get('profit_factor', 0):.2f}",
                'icon': 'fas fa-calculator',
                'color': 'success' if trading.get('profit_factor', 0) > 1 else 'danger'
            })
            
            metrics.append({
                'title': 'Average Trade',
                'value': f"${trading.get('avg_trade', 0):.2f}",
                'icon': 'fas fa-dollar-sign',
                'color': 'success' if trading.get('avg_trade', 0) > 0 else 'danger'
            })
        
        return metrics
    
    def _format_positions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format current positions for display."""
        positions = []
        
        if 'positions' in data:
            for position in data['positions']:
                positions.append({
                    'symbol': position.get('symbol', ''),
                    'quantity': position.get('quantity', 0),
                    'entry_price': position.get('entry_price', 0),
                    'current_price': position.get('current_price', 0),
                    'unrealized_pnl': position.get('unrealized_pnl', 0),
                    'unrealized_pnl_pct': position.get('unrealized_pnl_pct', 0)
                })
        
        return positions
    
    def _format_risk_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format risk metrics for display."""
        risk = {
            'var_5': 0,
            'cvar_5': 0,
            'beta': 0,
            'correlation': 0
        }
        
        if 'risk_metrics' in data:
            risk.update(data['risk_metrics'])
        
        return risk
    
    def _format_realtime_metrics(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format real-time metrics for display."""
        metrics = []
        
        if 'realtime_metrics' in data:
            rt_metrics = data['realtime_metrics']
            
            metrics.append({
                'title': 'Alerts/Hour',
                'value': f"{rt_metrics.get('alerts_per_hour', 0):.1f}",
                'icon': 'fas fa-bell',
                'color': 'info'
            })
            
            metrics.append({
                'title': 'Active Connections',
                'value': str(rt_metrics.get('active_connections', 0)),
                'icon': 'fas fa-link',
                'color': 'success'
            })
            
            metrics.append({
                'title': 'Processing Time',
                'value': f"{rt_metrics.get('avg_processing_time', 0):.1f}ms",
                'icon': 'fas fa-clock',
                'color': 'success' if rt_metrics.get('avg_processing_time', 0) < 100 else 'warning'
            })
        
        return metrics
    
    def _generate_html_dashboard(self, 
                               data: Dict[str, Any], 
                               template_type: str = 'performance') -> str:
        """Generate HTML dashboard from template."""
        
        # Default HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .alert-item {
            border-left: 4px solid;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .alert-critical { border-left-color: #dc3545; background-color: #f8d7da; }
        .alert-warning { border-left-color: #ffc107; background-color: #fff3cd; }
        .alert-info { border-left-color: #17a2b8; background-color: #d1ecf1; }
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        {% if template_type == 'realtime' %}
        .auto-refresh {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        {% endif %}
    </style>
    {% if template_type == 'realtime' %}
    <meta http-equiv="refresh" content="{{ refresh_interval }}">
    {% endif %}
</head>
<body class="bg-light">
    <div class="container-fluid">
        <div class="dashboard-header">
            <h1><i class="fas fa-chart-line me-2"></i>{{ title }}</h1>
            <p class="mb-0">Last updated: {{ timestamp }}
            {% if template_type == 'realtime' %}
            <span class="auto-refresh"><i class="fas fa-sync-alt ms-2"></i></span>
            {% endif %}
            </p>
        </div>
        
        <!-- Key Metrics Row -->
        <div class="row mb-4">
            {% for metric in metrics %}
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <i class="{{ metric.icon }} fa-2x mb-2"></i>
                    <h3>{{ metric.value }}</h3>
                    <p class="mb-0">{{ metric.title }}</p>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Charts Row -->
        <div class="row">
            {% for chart_name, chart_html in charts.items() %}
            <div class="col-md-6">
                <div class="chart-container">
                    {{ chart_html|safe }}
                </div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Alerts and System Health -->
        <div class="row">
            {% if alerts %}
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-bell me-2"></i>Recent Alerts</h5>
                    </div>
                    <div class="card-body">
                        {% for alert in alerts %}
                        <div class="alert-item alert-{{ alert.severity }}">
                            <strong>{{ alert.title }}</strong>
                            <small class="text-muted d-block">{{ alert.timestamp }}</small>
                            <p class="mb-0">{{ alert.content }}</p>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
            {% endif %}
            
            {% if system_health %}
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-heart me-2"></i>System Health</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <div class="mb-3">
                                    <label class="form-label">CPU Usage</label>
                                    <div class="progress">
                                        <div class="progress-bar" style="width: {{ system_health.cpu_usage }}%">
                                            {{ system_health.cpu_usage }}%
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="mb-3">
                                    <label class="form-label">Memory Usage</label>
                                    <div class="progress">
                                        <div class="progress-bar bg-info" style="width: {{ system_health.memory_usage }}%">
                                            {{ system_health.memory_usage }}%
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <p class="mb-0">
                            <strong>Status:</strong> 
                            <span class="badge bg-{{ 'success' if system_health.status == 'healthy' else 'warning' if system_health.status == 'warning' else 'danger' }}">
                                {{ system_health.status|title }}
                            </span>
                        </p>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(**data, template_type=template_type)
    
    def update_dashboard_data(self, 
                            dashboard_path: str, 
                            new_data: Dict[str, Any]) -> bool:
        """
        Update dashboard with new data (for real-time dashboards).
        
        Args:
            dashboard_path: Path to the dashboard file
            new_data: New data to update the dashboard with
            
        Returns:
            True if update successful
        """
        try:
            # For real-time dashboards, we would typically use WebSocket or AJAX
            # For now, regenerate the entire dashboard
            dashboard_type = 'performance'  # Could be inferred from path
            
            if 'trading_metrics' in new_data:
                return self.generate_trading_dashboard(new_data, Path(dashboard_path).name) is not None
            else:
                return self.generate_performance_dashboard(new_data, Path(dashboard_path).name) is not None
                
        except Exception as e:
            print(f"Error updating dashboard: {e}")
            return False


def create_sample_dashboard_data() -> Dict[str, Any]:
    """
    Create sample data for dashboard testing.
    
    Returns:
        Dictionary with sample dashboard data
    """
    import random
    from datetime import datetime, timedelta
    
    # Generate sample time series data
    dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    
    return {
        'summary_metrics': {
            'success_rate': 65.5,
            'total_return': 12.34,
            'sharpe_ratio': 1.45,
            'max_drawdown': 8.2
        },
        'success_rate_history': {
            'timestamps': dates,
            'values': [random.uniform(50, 80) for _ in dates]
        },
        'returns': [random.gauss(0.5, 2.0) for _ in range(1000)],
        'symbol_performance': {
            'AAPL': {'success_rate': 72.3},
            'GOOGL': {'success_rate': 68.9},
            'MSFT': {'success_rate': 71.2},
            'TSLA': {'success_rate': 59.8}
        },
        'recent_alerts': [
            {
                'title': 'High volatility detected',
                'severity': 'warning',
                'timestamp': '2025-01-15 14:30:00',
                'content': 'Volatility spike in TSLA'
            },
            {
                'title': 'Success rate below threshold',
                'severity': 'critical',
                'timestamp': '2025-01-15 13:45:00',
                'content': 'Success rate dropped to 45%'
            }
        ],
        'system_health': {
            'status': 'healthy',
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'disk_usage': 32.1,
            'uptime': '5 days'
        }
    }