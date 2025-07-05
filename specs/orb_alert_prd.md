# ORB Trading Alerts System - Product Requirements Document

## Executive Summary

This document outlines the requirements for developing a real-time Opening Range Breakout (ORB) trading alerts system. Based on PCA analysis showing 82.31% variance explained by ORB patterns, this system will provide high-confidence buy signals for day trading operations using live market data from Alpaca.markets API.

## Background & Market Analysis

### Conda Environment

Use the 'alpaca' conda environment.

### Statistical Foundation
Recent PCA analysis on July 3, 2025 data demonstrates:
- **82.31% variance explained by ORB patterns** (PC1), validating ORB strategy effectiveness
- **8.54% variance from volume dynamics** (PC2), confirming volume as secondary signal
- **3.78% variance from technical divergences** (PC3), indicating reversal patterns

### Key Trading Insights
1. **ORB Dominance**: 82.31% of price behavior follows predictable ORB patterns
2. **Volume Confirmation**: Volume surges are critical for breakout validity
3. **Technical Divergence**: EMA/VWAP deviations signal potential reversals

## Product Vision

Create a Python-based real-time alert system that identifies high-probability ORB breakout opportunities with configurable sensitivity parameters, leveraging websocket data streams for sub-second latency.

## Core Requirements

### 1. Real-Time Data Architecture

#### 1.1 Websocket Integration
- **Primary**: Alpaca.markets websocket API for real-time quotes
- **Fallback**: REST API polling at 1-second intervals
- **Data Points**: OHLCV + timestamp for each symbol
- **Concurrency**: Support 50+ symbols simultaneously

#### 1.2 Data Processing Pipeline
- **Opening Range Calculation**: First 15 minutes (9:30-9:45 AM ET)
- **Breakout Detection**: Real-time price vs ORB high/low monitoring
- **Volume Analysis**: Current volume vs 20-day average
- **Technical Indicators**: EMA(9), VWAP calculation

### 2. Alert Configuration System

#### 2.1 Sensitivity Parameters
```python
class ORBAlertConfig:
    # Primary ORB Settings
    orb_period_minutes: int = 15          # Opening range period
    breakout_threshold: float = 0.002     # 0.2% above ORB high
    volume_multiplier: float = 1.5        # 1.5x average volume required

    # Statistical Confidence (based on PCA analysis)
    pc1_weight: float = 0.8231           # PC1 variance weight
    pc2_weight: float = 0.0854           # PC2 variance weight
    pc3_weight: float = 0.0378           # PC3 variance weight

    # Risk Management
    min_price: float = 0.01              # Minimum stock price
    max_price: float = 10.00             # Maximum stock price
    min_volume: int = 1000000            # Minimum daily volume

    # Alert Timing
    alert_window_start: str = "09:45"    # Post-ORB period
    alert_window_end: str = "15:30"      # Before close
```

#### 2.2 Symbol Selection
- **Watchlist Management**: CSV-based symbol configuration (data/symbols.csv)
- **Dynamic Filtering**: Market cap, sector, volatility criteria
- **Exclusion Rules**: Earnings announcements, low volume stocks

### 3. Alert Generation Logic

#### 3.1 ORB Breakout Detection
```python
def detect_orb_breakout(symbol_data: Dict) -> Optional[ORBAlert]:
    """
    Core breakout detection algorithm based on PCA findings
    """
    # Calculate ORB levels (9:30-9:45 AM ET)
    orb_high = max(symbol_data.high[:15])
    orb_low = min(symbol_data.low[:15])

    # Current price analysis
    current_price = symbol_data.close[-1]
    price_vs_orb_high = (current_price - orb_high) / orb_high

    # Volume confirmation (PC2 component)
    volume_ratio = symbol_data.volume[-1] / symbol_data.avg_volume_20d

    # Technical indicator divergence (PC3 component)
    ema_deviation = abs(current_price - symbol_data.ema_9) / current_price
    vwap_deviation = abs(current_price - symbol_data.vwap) / current_price

    # PCA-weighted confidence score
    confidence_score = (
        config.pc1_weight * price_vs_orb_high +
        config.pc2_weight * volume_ratio +
        config.pc3_weight * (ema_deviation + vwap_deviation)
    )

    # Generate alert if thresholds met
    if (price_vs_orb_high >= config.breakout_threshold and
        volume_ratio >= config.volume_multiplier and
        confidence_score >= config.min_confidence_score):
        return ORBAlert(symbol, current_price, confidence_score)

    return None
```

#### 3.2 Alert Prioritization
- **High Priority**: Confidence score > 0.85, volume > 2x average
- **Medium Priority**: Confidence score 0.70-0.85, volume > 1.5x average
- **Low Priority**: Confidence score 0.60-0.70, volume > 1.2x average

### 4. Output & Notification System

#### 4.1 Alert Format
```python
@dataclass
class ORBAlert:
    symbol: str
    timestamp: datetime
    current_price: float
    orb_high: float
    orb_low: float
    breakout_percentage: float
    volume_ratio: float
    confidence_score: float
    priority: AlertPriority
    recommended_stop_loss: float
    recommended_take_profit: float
```

#### 4.2 Delivery Methods
- **Console Output**: Real-time terminal display
- **JSON Log**: Structured logging for analysis
- **File Export**: CSV format for spreadsheet integration
- **Future**: Email/SMS notifications (Phase 2)

## Technical Architecture

### 5. Module Structure (Atom-Molecules Pattern)

```
atoms/
├── websocket/
│   ├── alpaca_stream.py      # Websocket client
│   ├── data_buffer.py        # Real-time data storage
│   └── connection_manager.py # Connection handling
├── indicators/
│   ├── orb_calculator.py     # ORB level computation
│   ├── volume_analyzer.py    # Volume analysis
│   └── technical_indicators.py # EMA, VWAP, etc.
├── alerts/
│   ├── breakout_detector.py  # Core detection logic
│   ├── confidence_scorer.py  # PCA-based scoring
│   └── alert_formatter.py    # Output formatting
└── config/
    ├── alert_config.py       # Configuration management
    └── symbol_manager.py     # Watchlist handling (data/symbols.csv)

molecules/
├── orb_alert_engine.py       # Main orchestration
├── real_time_processor.py    # Data processing pipeline
└── notification_manager.py   # Alert delivery

code/
└── orb_alerts.py            # Main entry point
```

### 6. Performance Requirements

#### 6.1 Latency Targets
- **Data Ingestion**: < 100ms from market to system
- **Processing**: < 50ms for breakout detection
- **Alert Generation**: < 200ms end-to-end
- **Total Latency**: < 500ms from market event to alert

#### 6.2 Throughput Requirements
- **Symbols**: Support 50+ concurrent symbols
- **Update Frequency**: Process 1-second intervals
- **Alert Volume**: Handle 100+ alerts per trading day
- **Memory Usage**: < 1GB for full system

### 7. Quality Assurance

#### 7.1 Testing Strategy
```python
# Test Coverage Requirements
tests/
├── test_orb_calculator.py       # ORB calculation accuracy
├── test_breakout_detector.py    # Detection algorithm
├── test_confidence_scorer.py    # PCA scoring logic
├── test_websocket_client.py     # Real-time data handling
├── test_alert_generation.py     # End-to-end alerts
└── test_performance.py          # Latency benchmarks
```

#### 7.2 Validation Criteria
- **Accuracy**: 95% correlation with manual ORB identification
- **False Positives**: < 10% of generated alerts
- **Performance**: Meet all latency targets under load
- **Reliability**: 99.9% uptime during market hours

#### 7.3 Compliance Checks
- **Linting**: flake8 compliance per setup.cfg
- **Type Checking**: mypy static analysis
- **Code Coverage**: >90% test coverage
- **VS Code Integration**: No import errors or warnings

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- Websocket client implementation
- Basic ORB calculation
- Configuration system
- Unit tests

### Phase 2: Alert Logic (Week 3-4)
- PCA-based confidence scoring
- Breakout detection algorithm
- Alert prioritization
- Integration tests

### Phase 3: Production Ready (Week 5-6)
- Performance optimization
- Error handling & recovery
- Monitoring & logging
- Documentation

## Risk Management

### 8. Technical Risks
- **API Rate Limits**: Implement exponential backoff
- **Connection Drops**: Automatic reconnection with state recovery
- **Data Quality**: Validation and anomaly detection
- **Memory Leaks**: Periodic garbage collection

### 8.1 Trading Risks
- **False Signals**: Multi-factor confirmation required
- **Market Conditions**: Suspend alerts during high volatility
- **Regulatory**: Ensure compliance with day trading rules
- **Position Sizing**: Recommend appropriate risk levels

## Success Metrics

### 9. Key Performance Indicators
- **Alert Accuracy**: >90% successful breakout follow-through
- **Latency**: <500ms average alert generation time
- **Uptime**: >99.9% during market hours
- **Coverage**: Monitor 50+ symbols simultaneously

### 9.1 Business Metrics
- **Daily Alerts**: 20-50 quality alerts per trading day
- **Win Rate**: >60% of alerts result in profitable trades
- **Risk-Adjusted Returns**: Positive Sharpe ratio over 30 days
- **User Satisfaction**: Subjective feedback from day traders

## Appendix

### A. PCA Analysis Reference
Based on July 3, 2025 analysis of 22 symbols with 1,980 feature rows:
- PC1 (82.31%): ORB momentum patterns
- PC2 (8.54%): Volume dynamics
- PC3 (3.78%): Technical divergences

### B. Configuration Examples
```python
# Conservative Settings
conservative_config = ORBAlertConfig(
    breakout_threshold=0.005,    # 0.5% breakout
    volume_multiplier=2.0,       # 2x volume required
    min_confidence_score=0.80    # High confidence only
)

# Aggressive Settings
aggressive_config = ORBAlertConfig(
    breakout_threshold=0.001,    # 0.1% breakout
    volume_multiplier=1.2,       # 1.2x volume required
    min_confidence_score=0.60    # Lower confidence threshold
)
```

### C. Expected Output Format
```
[10:15:32] ORB ALERT: AAPL @ $150.25 (+0.34% vs ORB High)
Volume: 2.3x average | Confidence: 0.87 | Priority: HIGH
Stop Loss: $148.50 | Take Profit: $153.00
```

---

*This PRD serves as the foundation for developing a statistically-driven ORB trading alerts system with proven 82.31% variance explanation from PCA analysis.*