# ORB Alert System Documentation

## Overview

The Opening Range Breakout (ORB) Alert System is a sophisticated trading alerts platform that identifies high-probability breakout opportunities based on the first 15 minutes of market trading. The system uses Principal Component Analysis (PCA) showing 82.31% variance explained by ORB patterns to generate confident trading signals.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technical Indicators](#technical-indicators)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Alert Types](#alert-types)
- [Risk Management](#risk-management)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)

## Features

### Core Capabilities
- **Real-time ORB Detection**: Monitors 15-minute opening range breakouts
- **PCA-Based Confidence Scoring**: 82.31% variance explained statistical model
- **Multi-Priority Alerts**: HIGH, MEDIUM, LOW, VERY_LOW priority classification
- **Volume Analysis**: Tracks volume spikes and anomalies
- **Technical Validation**: EMA and VWAP confirmation signals
- **Risk Management**: Automatic stop-loss and take-profit calculations
- **Production Monitoring**: Phase 3 performance tracking and optimization

### Alert Types
- **Bullish Breakouts**: Price breaks above ORB high with volume confirmation
- **Bearish Breakdowns**: Price breaks below ORB low with volume confirmation
- **High-Volume Scenarios**: 2x+ average volume breakout confirmation
- **Multi-Timeframe**: Real-time intraday alerts throughout market hours

## Architecture

### Component Structure

```
ORB Alert System
├── code/
│   └── orb_alerts.py              # Main entry point
├── molecules/
│   └── orb_alert_engine.py        # Core orchestration engine
├── atoms/
│   ├── alerts/
│   │   ├── breakout_detector.py   # ORB breakout detection
│   │   ├── confidence_scorer.py   # PCA-based scoring
│   │   └── alert_formatter.py     # Alert generation & formatting
│   ├── indicators/
│   │   └── orb_calculator.py      # ORB level calculations
│   ├── websocket/
│   │   ├── alpaca_stream.py       # Real-time data streaming
│   │   └── data_buffer.py         # Market data buffering
│   ├── config/
│   │   └── alert_config.py        # System configuration
│   └── utils/
│       ├── calculate_vwap.py      # VWAP calculations
│       └── calculate_ema.py       # EMA calculations
└── tests/
    ├── test_bmnr_orb_alerts.py    # Bullish scenario tests
    └── test_artl_orb_alerts.py    # Bearish scenario tests
```

### Data Flow

```
Market Data → Data Buffer → ORB Calculator → Breakout Detector
                                                      ↓
Alert Formatter ← Confidence Scorer ← Technical Indicators
       ↓
[Alert Output: Console, File, Callbacks]
```

## Technical Indicators

### 📈 Technical Indicator Workflow

The system calculates focused technical indicators as part of its PCA-based confidence scoring:

```python
# 1. Historical Data → BreakoutDetector.calculate_technical_indicators()
historical_data = self.data_buffer.get_symbol_data(symbol)

# 2. Calculate EMA(9) from close prices
ema_9 = symbol_data['close'].ewm(span=9).mean().iloc[-1]

# 3. Calculate VWAP using typical price and volume
typical_price = (high + low + close) / 3
vwap = (typical_price * volume).sum() / volume.sum()

# 4. Calculate Deviations from current price
ema_deviation = abs(current_price - ema_9) / current_price
vwap_deviation = abs(current_price - vwap) / current_price

# 5. Feed to Confidence Scorer → PC3 component (3.78% of total score)
pc3_score = (ema_score + vwap_score) / 2.0

# 6. Combine with PC1 (ORB, 82.31%) and PC2 (Volume, 8.54%)
total_confidence = pc1_weight * pc1_score + pc2_weight * pc2_score + pc3_weight * pc3_score

# 7. Generate Final Confidence Score
confidence_level = "VERY_HIGH" if total_confidence >= 0.85 else "HIGH" if >= 0.70 else "MEDIUM"
```

### Calculated Indicators

| Indicator | Purpose | Weight | Calculation |
|-----------|---------|--------|-------------|
| **EMA(9)** | Short-term trend | PC3 (3.78%) | `close.ewm(span=9).mean()` |
| **VWAP** | Volume-weighted benchmark | PC3 (3.78%) | `Σ(typical_price × volume) / Σ(volume)` |
| **Price Deviations** | Momentum strength | PC3 (3.78%) | `abs(current_price - indicator) / current_price` |

### PCA Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **PC1** | 82.31% | ORB momentum (breakout strength, direction, range quality) |
| **PC2** | 8.54% | Volume dynamics (volume ratio, spikes, consistency) |
| **PC3** | 3.78% | Technical alignment (EMA/VWAP deviations) |

### 🚫 What's NOT Calculated

The system **does not** calculate:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- Multiple timeframe indicators

### 💡 Design Philosophy

**Minimalist Approach:**
- **Primary Focus**: ORB momentum (82.31% weight)
- **Secondary**: Volume dynamics (8.54% weight)  
- **Tertiary**: Technical alignment (3.78% weight)
- **Philosophy**: Simple, fast indicators that confirm momentum rather than complex oscillators

**Utility Functions Available:**
- `atoms/utils/calculate_vwap.py` - Multiple VWAP calculation methods
- `atoms/utils/calculate_ema.py` - EMA calculation with manual and pandas methods

The technical indicators serve as **momentum confirmation** rather than primary signals, keeping the system focused on the core ORB breakout methodology while adding technical validation.

## Installation

### Prerequisites
- Python 3.10+
- Conda (Miniconda or Anaconda)
- Alpaca API account (paper or live trading)
- Required Python packages

### Setup

1. **Clone Repository**
```bash
git clone https://github.com/Z223I/alpaca.git
cd alpaca
```

2. **Create Conda Environment**
```bash
# Create new conda environment
conda create -n alpaca python=3.10
conda activate alpaca
```

3. **Install Dependencies**
```bash
# Install required packages in conda environment
pip install alpaca-trade-api python-dotenv matplotlib pytest pytest-cov pandas numpy psutil pytz websockets scikit-learn
```

4. **Environment Configuration**
```bash
# Create .env file
cp .env.example .env
```

5. **Configure API Credentials**
```env
# .env file
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
PORTFOLIO_RISK=0.10  # Optional: default is 0.10 (10%)
```

6. **Set Python Path**
```bash
# Set the Python path for module imports
export PYTHONPATH=/home/wilsonb/dl/github.com/z223i/alpaca
```
   *Note: This command may be placed in your `.bashrc` file for permanent setup.*

7. **Verify Installation**
```bash
# Activate environment before running
conda activate alpaca
python3 code/orb_alerts.py --test
```

## Configuration

### ORB Alert Configuration (`atoms/config/alert_config.py`)

```python
@dataclass
class ORBAlertConfig:
    # Primary ORB Settings
    orb_period_minutes: int = 15          # Opening range period
    breakout_threshold: float = 0.002     # 0.2% above ORB high
    volume_multiplier: float = 1.5        # 1.5x average volume required
    
    # Statistical Confidence (PCA weights)
    pc1_weight: float = 0.8231           # PC1 variance weight
    pc2_weight: float = 0.0854           # PC2 variance weight  
    pc3_weight: float = 0.0378           # PC3 variance weight
    min_confidence_score: float = 0.70   # Minimum confidence threshold
    
    # Alert Timing
    alert_window_start: str = "09:45"    # Post-ORB period
    alert_window_end: str = "15:30"      # Before close
    
    # Risk Management  
    min_price: float = 0.01              # Minimum stock price
    max_price: float = 10.00             # Maximum stock price
    min_volume: int = 1000000            # Minimum daily volume
```

### Symbol Configuration
Create `data/symbols.csv` with target symbols:
```csv
symbol,enabled
AAPL,true
TSLA,true
BMNR,true
ARTL,true
```

## Usage

### Command Line Interface

```bash
# Start monitoring all symbols
python3 code/orb_alerts.py

# Monitor specific symbols
python3 code/orb_alerts.py --symbols-file custom_symbols.csv

# Test mode (dry run)
python3 code/orb_alerts.py --test

# Verbose logging
python3 code/orb_alerts.py --verbose

# Show daily summary
python3 code/orb_alerts.py --summary
```

### Programmatic Usage

```python
import asyncio
from code.orb_alerts import ORBAlertSystem

async def main():
    # Initialize system
    system = ORBAlertSystem(
        symbols_file="data/symbols.csv",
        test_mode=False
    )
    
    # Add custom alert callback
    def handle_alert(alert):
        print(f"Alert: {alert.symbol} - {alert.priority.value}")
        # Custom processing logic here
    
    system.alert_engine.add_alert_callback(handle_alert)
    
    # Start monitoring
    await system.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## Alert Types

### Priority Classification

| Priority | Conditions | Confidence | Volume | Use Case |
|----------|------------|------------|--------|----------|
| **HIGH** | confidence ≥ 0.85 AND volume ≥ 2.0x | ≥ 85% | ≥ 2x avg | Strong conviction trades |
| **MEDIUM** | confidence ≥ 0.70 OR volume ≥ 2.0x | ≥ 70% | Any | Moderate conviction |
| **LOW** | confidence ≥ 0.60 | ≥ 60% | Any | Low conviction |
| **VERY_LOW** | confidence < 0.60 | < 60% | Any | Monitoring only |

### Sample Alert Output

```
[11:15:30] ORB ALERT: BMNR ↑ $35.50 (+49.16% vs ORB High)
Volume: 3.2x avg | Confidence: 0.87 | Priority: HIGH
Stop: $21.85 | Target: $63.80
```

### Alert Components

- **Symbol & Direction**: BMNR ↑ (bullish) or ↓ (bearish)
- **Current Price**: Real-time breakout price
- **Breakout Percentage**: % above ORB high or below ORB low
- **Volume Ratio**: Current volume vs average volume
- **Confidence Score**: PCA-based confidence (0.0-1.0)
- **Priority Level**: HIGH/MEDIUM/LOW/VERY_LOW
- **Risk Management**: Stop-loss and take-profit levels

## Risk Management

### Automatic Calculations

**For Bullish Breakouts:**
```python
# Stop loss below ORB low with 7.5% buffer
stop_loss = orb_low * 0.925

# Take profit with 2:1 reward/risk ratio
stop_distance = current_price - stop_loss
take_profit = current_price + (stop_distance * 2.0)
```

**For Bearish Breakdowns:**
```python
# Stop loss above ORB high with 7.5% buffer  
stop_loss = orb_high * 1.075

# Take profit with 2:1 reward/risk ratio
stop_distance = stop_loss - current_price
take_profit = current_price - (stop_distance * 2.0)
```

### Position Sizing
- **First Position**: Uses `PORTFOLIO_RISK` percentage of cash (default 10%)
- **Subsequent Positions**: Uses remaining available cash
- **Risk Per Trade**: 7.5% stop-loss from entry point

## Testing

### Test Suites

The system includes comprehensive test coverage with real market data:

```bash
# Run all tests
./test.sh

# Run specific test suites
python -m pytest tests/test_bmnr_orb_alerts.py -v  # Bullish scenarios
python -m pytest tests/test_artl_orb_alerts.py -v  # Bearish scenarios

# Run with coverage
./test.sh coverage
```

### Test Coverage

- **24 total tests** (11 BMNR bullish + 13 ARTL bearish)
- **Real market data** from 2025-06-30
- **High/very high alert scenarios** 
- **Volume spike validation**
- **Risk management verification**
- **Alert message formatting**

### Example Test Scenarios

**BMNR (Bullish):**
- ORB High: $23.80, ORB Low: $17.64
- Max breakout: $45.97 (93.19% above ORB high)
- High volume scenarios with 3x+ average volume

**ARTL (Bearish):**
- ORB High: $25.10, ORB Low: $19.04  
- Max breakdown: $11.99 (37.02% below ORB low)
- Massive -34.5% decline day with volume confirmation

## API Reference

### Core Classes

#### `ORBAlertSystem`
Main orchestrator class.

```python
class ORBAlertSystem:
    def __init__(self, symbols_file: str = None, test_mode: bool = False)
    async def start(self) -> None
    async def stop(self) -> None
    def get_statistics(self) -> dict
    def print_daily_summary(self) -> None
```

#### `ORBAlertEngine`
Core alert generation engine.

```python
class ORBAlertEngine:
    def __init__(self, symbols_file: str = None, output_dir: str = "alerts")
    def add_alert_callback(self, callback: Callable[[ORBAlert], None]) -> None
    async def start(self) -> None
    def get_stats(self) -> AlertEngineStats
    def get_recent_alerts(self, limit: int = 10) -> List[ORBAlert]
```

#### `ORBAlert`
Alert data structure.

```python
@dataclass
class ORBAlert:
    symbol: str
    timestamp: datetime
    current_price: float
    orb_high: float
    orb_low: float
    breakout_type: BreakoutType
    breakout_percentage: float
    volume_ratio: float
    confidence_score: float
    priority: AlertPriority
    recommended_stop_loss: float
    recommended_take_profit: float
    alert_message: str
```

### Configuration Classes

#### `ORBAlertConfig`
System configuration.

```python
@dataclass
class ORBAlertConfig:
    orb_period_minutes: int = 15
    breakout_threshold: float = 0.002
    volume_multiplier: float = 1.5
    pc1_weight: float = 0.8231
    pc2_weight: float = 0.0854
    pc3_weight: float = 0.0378
```

## Examples

### Basic Monitoring

```python
import asyncio
from code.orb_alerts import ORBAlertSystem

async def basic_monitoring():
    system = ORBAlertSystem()
    await system.start()

asyncio.run(basic_monitoring())
```

### Custom Alert Handler

```python
import asyncio
from code.orb_alerts import ORBAlertSystem
from atoms.alerts.alert_formatter import AlertPriority

async def custom_alerts():
    system = ORBAlertSystem(test_mode=True)
    
    def high_priority_handler(alert):
        if alert.priority == AlertPriority.HIGH:
            print(f"🚨 HIGH PRIORITY: {alert.symbol}")
            print(f"💰 Entry: ${alert.current_price:.2f}")
            print(f"🛑 Stop: ${alert.recommended_stop_loss:.2f}")
            print(f"🎯 Target: ${alert.recommended_take_profit:.2f}")
            # Send to trading system, email, Slack, etc.
    
    system.alert_engine.add_alert_callback(high_priority_handler)
    await system.start()

asyncio.run(custom_alerts())
```

### Integration with Trading System

```python
from code.alpaca import alpaca_private

async def trading_integration():
    # Initialize ORB alerts
    orb_system = ORBAlertSystem()
    
    # Initialize trading system
    trader = alpaca_private()
    
    def execute_trade(alert):
        if alert.priority == AlertPriority.HIGH and alert.confidence_score >= 0.90:
            if alert.breakout_type == BreakoutType.BULLISH_BREAKOUT:
                # Execute buy order
                trader._buy(
                    symbol=alert.symbol,
                    submit=True,
                    stop_loss_price=alert.recommended_stop_loss,
                    take_profit_price=alert.recommended_take_profit
                )
            elif alert.breakout_type == BreakoutType.BEARISH_BREAKDOWN:
                # Execute short order (if supported)
                print(f"Short signal: {alert.symbol} at {alert.current_price}")
    
    orb_system.alert_engine.add_alert_callback(execute_trade)
    await orb_system.start()
```

## Performance Monitoring

### Phase 3 Production Features

The system includes comprehensive monitoring:

- **Performance Tracking**: Operation timing and throughput
- **Error Handling**: Retry logic and circuit breakers  
- **Data Validation**: Market data quality checks
- **Optimization**: Dynamic performance tuning
- **Health Monitoring**: System resource usage

### Monitoring Dashboard

```python
# Get system status
status = alert_engine.get_phase3_status()
print(f"Performance Tracker: {status['performance_tracker_running']}")
print(f"Dashboard: {status['dashboard_running']}")
print(f"Optimizer: {status['optimizer_running']}")
```

## Troubleshooting

### Common Issues

**No Alerts Generated:**
- Check market hours (9:30 AM - 4:00 PM ET)
- Verify symbols in configuration file
- Ensure sufficient volume (min 1.5x average)
- Check confidence threshold settings

**Connection Issues:**
- Verify Alpaca API credentials
- Check network connectivity
- Review websocket timeout settings

**Performance Issues:**
- Monitor system resources
- Adjust symbol count for available CPU/memory
- Review Phase 3 optimization settings

### Debugging

```bash
# Enable verbose logging
python3 code/orb_alerts.py --verbose

# Check system status
python3 code/orb_alerts.py --summary

# Run in test mode
python3 code/orb_alerts.py --test
```

### Log Analysis

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# System logs show:
# - ORB level calculations
# - Breakout detections  
# - Confidence scoring
# - Alert generation
# - Performance metrics
```

## Contributing

### Development Setup

1. **Fork Repository**
2. **Create Feature Branch**
```bash
git checkout -b feature/new-indicator
```

3. **Run Tests**
```bash
./test.sh
```

4. **Add Tests for New Features**
```bash
# Add tests to appropriate test file
tests/test_new_feature.py
```

5. **Submit Pull Request**

### Code Standards

- **PEP 8** compliance
- **Type hints** for all functions
- **Comprehensive docstrings**
- **Unit test coverage**
- **Performance considerations**

### Adding New Technical Indicators

```python
# atoms/indicators/new_indicator.py
def calculate_new_indicator(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate new technical indicator."""
    return {'new_indicator_value': result}

# atoms/alerts/breakout_detector.py
def calculate_technical_indicators(self, symbol_data: pd.DataFrame):
    indicators = {}
    # Add new indicator calculation
    indicators.update(calculate_new_indicator(symbol_data))
    return indicators
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading stocks involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## Support

For issues, questions, or contributions:

- **GitHub Issues**: Submit bug reports and feature requests
- **Documentation**: Review code comments and docstrings
- **Testing**: Run comprehensive test suite for validation

---

**Last Updated**: January 2025
**Version**: 1.0.0
**Python**: 3.10+
**Dependencies**: alpaca-trade-api, pandas, numpy, pytest