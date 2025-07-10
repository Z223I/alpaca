# Alert Performance Analysis System

A comprehensive, production-ready alert performance analysis system built with atoms-molecules architecture for institutional-grade trading alert evaluation and monitoring.

## Overview

This system provides advanced analytics, monitoring, and alerting capabilities for trading alert systems, designed to meet institutional performance requirements with sub-500ms latency targets.

## Architecture

The system follows a modular atoms-molecules architecture:

### Atoms (Basic Components)
- **Analysis Atoms**: Core performance calculation functions
- **Metrics Atoms**: Advanced risk and statistical analysis
- **Visualization Atoms**: Professional chart generation
- **Monitoring Atoms**: Real-time performance tracking
- **Alerting Atoms**: Threshold-based notification system
- **Dashboard Atoms**: HTML dashboard generation

### Molecules (Composite Components)
- **AlertAnalyzer**: Comprehensive alert performance analysis
- **PerformanceReporter**: Multi-format reporting system
- **SystemMonitor**: Real-time system health monitoring
- **AlertManager**: Complete alert lifecycle management

## Features

### Phase 1: Core Alert Analysis
- ✅ Basic performance metrics (success rate, win rate, profit factor)
- ✅ Risk assessment (drawdown, volatility, Sharpe ratio)
- ✅ Time-based analysis with trading hours filtering
- ✅ Multi-format reporting (JSON, CSV, HTML)
- ✅ Comprehensive CLI interface

### Phase 2: Advanced Analytics
- ✅ Advanced risk metrics (VaR, CVaR, tail risk, downside risk)
- ✅ Statistical analysis (ANOVA, t-tests, distribution analysis)
- ✅ Professional visualizations (performance charts, risk heatmaps)
- ✅ Enhanced reporting with institutional-grade analytics
- ✅ Cross-sectional and time-series analysis

### Phase 3: Monitoring & Alerting
- ✅ Real-time performance monitoring
- ✅ Threshold-based alerting system
- ✅ Multi-channel notifications (console, file, email, Slack, webhook)
- ✅ System health monitoring
- ✅ Interactive HTML dashboards
- ✅ Alert lifecycle management with escalation

## Installation

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, matplotlib, seaborn, plotly, scipy, jinja2, psutil, requests

### Setup
```bash
# Clone repository
git clone [repository-url]
cd alpaca

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly scipy jinja2 psutil requests pytest

# Verify installation
python -c "from molecules.alert_analyzer import AlertAnalyzer; print('Installation successful')"
```

## Quick Start

### Basic Alert Analysis
```python
from molecules.alert_analyzer import AlertAnalyzer
import pandas as pd

# Load your alert data
alerts_df = pd.read_csv('your_alerts.csv')

# Initialize analyzer
analyzer = AlertAnalyzer()

# Run comprehensive analysis
results = analyzer.analyze_alerts(alerts_df)

# Generate reports
analyzer.generate_performance_report(results, 'performance_report.html')
```

### System Monitoring
```python
from molecules.system_monitor import SystemMonitor

# Initialize monitor
monitor = SystemMonitor(
    monitoring_interval=30,  # Check every 30 seconds
    enable_notifications=True,
    enable_dashboard=True
)

# Start monitoring
monitor.start_monitoring()

# Record trading metrics
monitor.record_trading_metrics({
    'success_rate': 72.5,
    'avg_return': 2.3,
    'max_drawdown': 5.1
})

# Generate dashboard
dashboard_path = monitor.generate_dashboard()
```

### Alert Management
```python
from molecules.alert_manager import AlertManager, AlertRule, AlertSeverity, AlertPriority

# Initialize alert manager
alert_manager = AlertManager()

# Create alert rule
rule = AlertRule(
    name="low_success_rate",
    description="Trading success rate below threshold",
    condition="success_rate < 50",
    severity=AlertSeverity.WARNING,
    priority=AlertPriority.P2_HIGH
)

# Add rule and trigger alert
alert_manager.add_alert_rule(rule)
alert = alert_manager.trigger_alert(
    rule_name="low_success_rate",
    metric_name="success_rate",
    current_value=45.0,
    threshold_value=50.0
)
```

## CLI Usage

### Basic Analysis
```bash
# Analyze alerts from CSV file
python -m molecules.alert_analyzer --input alerts.csv --output results/

# Generate specific report type
python -m molecules.alert_analyzer --input alerts.csv --report-type html --output report.html

# Filter by date range
python -m molecules.alert_analyzer --input alerts.csv --start-date 2024-01-01 --end-date 2024-12-31
```

### Advanced Analysis
```bash
# Run with advanced analytics
python -m molecules.alert_analyzer --input alerts.csv --enable-advanced-analytics

# Generate risk analysis
python -m molecules.alert_analyzer --input alerts.csv --analysis-type risk --output risk_report.html

# Statistical analysis
python -m molecules.alert_analyzer --input alerts.csv --analysis-type statistical --output stats.json
```

## Data Format

### Required Alert Data Columns
- `symbol`: Trading symbol (e.g., 'AAPL', 'GOOGL')
- `timestamp`: Alert timestamp (ISO format)
- `signal`: Alert signal ('BUY', 'SELL', 'HOLD')
- `confidence`: Confidence score (0.0 to 1.0)
- `entry_price`: Entry price for the trade
- `exit_price`: Exit price for the trade (optional)
- `return_pct`: Return percentage (calculated if not provided)
- `status`: Alert status ('SUCCESS', 'FAILURE', 'PENDING')

### Optional Columns
- `priority`: Alert priority ('HIGH', 'MEDIUM', 'LOW')
- `stop_loss`: Stop loss price
- `take_profit`: Take profit price
- `volume`: Trading volume
- `market_cap`: Market capitalization
- `sector`: Stock sector

## Performance Metrics

### Basic Metrics
- **Success Rate**: Percentage of profitable alerts
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Return**: Mean return per alert
- **Total Return**: Cumulative return
- **Sharpe Ratio**: Risk-adjusted return measure

### Advanced Risk Metrics
- **Value at Risk (VaR)**: Maximum expected loss at confidence level
- **Conditional VaR (CVaR)**: Expected loss beyond VaR
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Downside Risk**: Volatility of negative returns
- **Tail Risk**: Risk of extreme negative outcomes
- **Sortino Ratio**: Downside risk-adjusted return

### Statistical Analysis
- **ANOVA**: Analysis of variance across groups
- **T-Tests**: Statistical significance testing
- **Distribution Analysis**: Return distribution characteristics
- **Correlation Analysis**: Cross-asset correlation patterns
- **Regression Analysis**: Performance factor analysis

## Monitoring & Alerting

### Performance Targets
- Data Ingestion: < 100ms
- Processing: < 50ms
- Alert Generation: < 200ms
- Total Latency: < 500ms

### Alert Conditions
- Success rate below threshold
- High system resource usage
- Large trading losses detected
- Performance degradation trends
- System health issues

### Notification Channels
- Console output
- File logging
- Email notifications
- Slack integration
- Custom webhooks

## Testing

### Run All Tests
```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_phase1_integration.py -v
python -m pytest tests/test_phase2_integration.py -v
python -m pytest tests/test_phase3_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=atoms --cov=molecules --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual atom functionality
- **Integration Tests**: Molecule-level integration
- **Performance Tests**: Latency and throughput validation
- **End-to-End Tests**: Complete workflow testing

## Configuration

### Environment Variables
```bash
# Optional configuration
export ALERT_SYSTEM_OUTPUT_DIR="./output"
export ALERT_SYSTEM_LOG_LEVEL="INFO"
export ALERT_SYSTEM_TRADING_HOURS="09:30-16:00"
export ALERT_SYSTEM_TIMEZONE="America/New_York"
```

### Configuration Files
- `monitoring_config.json`: System monitoring configuration
- `alert_rules.json`: Alert threshold rules
- `notification_config.json`: Notification channel settings

## Development

### Adding New Atoms
1. Create atom in appropriate `atoms/` subdirectory
2. Implement core functionality with comprehensive docstrings
3. Add unit tests in `tests/`
4. Update integration tests if needed

### Adding New Molecules
1. Create molecule in `molecules/` directory
2. Compose existing atoms into higher-level functionality
3. Add comprehensive integration tests
4. Update documentation and examples

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Comprehensive docstrings with Args/Returns
- Error handling with appropriate logging
- Performance optimization for production use

## Performance Optimization

### Memory Management
- Efficient pandas operations
- Streaming data processing for large datasets
- Configurable history retention limits
- Memory profiling and optimization

### Computational Efficiency
- Vectorized operations using NumPy
- Parallel processing for independent calculations
- Caching of expensive computations
- Optimized statistical algorithms

## Troubleshooting

### Common Issues
1. **ImportError**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce data size or increase system memory
3. **Performance Slow**: Check data size and enable optimizations
4. **Visualization Errors**: Verify matplotlib/plotly installation

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode
analyzer = AlertAnalyzer(debug=True)
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki

---

**Note**: This system is designed for institutional-grade trading analysis and should be used with appropriate risk management practices. Always validate results against known benchmarks and consult with quantitative analysts before making trading decisions.