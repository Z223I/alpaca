# Alpaca Stock Screener Specification

## Overview

This specification defines the requirements and implementation approach for building a comprehensive stock screener using the Alpaca Trading API v2, based on traditional stock screening metrics and enhanced with volume surge detection capabilities.

**Target Implementation**: `code/alpaca_screener.py`

## Existing Infrastructure Integration

### atoms/api/init_alpaca_client.py

The screener will leverage the existing Alpaca client initialization infrastructure:

**Key Capabilities:**
- **Multi-environment support**: paper, live, cash trading environments
- **Multi-account support**: Bruce, Dale Wilson, Janice account configurations  
- **Flexible credential management**: Config file + environment variable fallback
- **Pre-configured API clients**: Ready-to-use `tradeapi.REST` instances

**Integration Pattern:**
```python
from atoms.api.init_alpaca_client import init_alpaca_client

# Initialize client for screener
client = init_alpaca_client(provider="alpaca", account="Bruce", environment="paper")
```

**Configuration Structure** (from `code/alpaca_config.py`):
```python
CONFIG = AlpacaConfig(
    providers={
        "alpaca": ProviderConfig(
            accounts={
                "Bruce": AccountConfig(paper=..., live=..., cash=...),
                "Dale Wilson": AccountConfig(...),
                "Janice": AccountConfig(...)
            }
        )
    }
)
```

**Benefits for Screener Implementation:**
- No manual credential management required
- Consistent authentication across all tools
- Easy account type switching (paper ↔ live ↔ cash)
- Account-specific configurations already defined
- Error handling and fallback mechanisms built-in

## API Foundation

### Alpaca API v2 Market Data Endpoints

#### Historical Stock Bars API
- **Endpoint**: `/v2/stocks/bars`
- **Base URL**: `https://data.alpaca.markets/v2`
- **Authentication**: Required via `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY` headers

#### Key Parameters
```python
StockBarsRequest(
    symbol_or_symbols: Union[str, List[str]], 
    timeframe: TimeFrame,  # 1Min, 5Min, 15Min, 1Hour, 1Day, etc.
    start: Optional[datetime] = None,
    end: Optional[datetime] = None, 
    limit: Optional[int] = None,
    feed: Optional[DataFeed] = None,  # iex, sip, boats, overnight
    adjustment: Optional[Adjustment] = None,
    sort: Optional[Sort] = None
)
```

#### Data Sources
1. **IEX Exchange** (`feed="iex"`)
   - ~2.5% market coverage
   - Free tier
   - Ideal for testing

2. **Securities Information Processors** (`feed="sip"`)
   - 100% market coverage
   - Ultra-low latency
   - Recommended for production

3. **Blue Ocean ATS** (`feed="boats"`)
   - Extended hours trading
   - Evening market data

#### Available Screener Endpoints
1. **Most Actives**: `get_most_actives()` - Returns most active stocks
2. **Market Movers**: `get_market_movers()` - Returns market movers

## Screening Metrics Implementation

### Core Price and Volume Metrics

| Metric | Unit | Alpaca Implementation | Data Source |
|--------|------|---------------------|-------------|
| **Price** | USD | `bars.close` from latest bar | Real-time/Historical bars |
| **Volume** | Shares | `bars.volume` from latest bar | Real-time/Historical bars |
| **% Change** | Percent | `((current_price - prev_close) / prev_close) * 100` | Calculated from bars |
| **$ Volume** | USD | `bars.volume * bars.vwap` | Calculated from bars |
| **Day Range** | USD | `bars.high - bars.low` | Daily bars |

### Advanced Metrics

| Metric | Unit | Implementation Approach | Calculation Method |
|--------|------|----------------------|-------------------|
| **Market Cap** | USD | `price * shares_outstanding` | External data source required |
| **Float** | Shares | Third-party fundamental data | External API integration |
| **Trades** | Count | `bars.trade_count` | Available in bar data |
| **EPS** | USD | Fundamental data | External API integration |
| **P/E Ratio** | Ratio | `price / eps` | Calculated from price + EPS |
| **SMA** | Days | Rolling average calculation | Historical bars analysis |

### Volume Analysis Metrics

| Metric | Unit | Implementation | Lookback Period |
|--------|------|----------------|----------------|
| **Avg Daily Volume** | Shares (5 Days) | `sum(volume[-5:]) / 5` | 5 trading days |
| **Avg Daily Range** | USD (5 Days) | `sum(high-low[-5:]) / 5` | 5 trading days |

## Volume Surge Detection System

### Core Feature: N Times Volume Over M Days

#### Implementation Specification
```python
def detect_volume_surge(symbol: str, n_multiplier: float, m_days: int) -> bool:
    """
    Detect if current volume is N times higher than M-day average
    
    Args:
        symbol: Stock ticker
        n_multiplier: Volume multiplier threshold (e.g., 2.0 for 2x volume)
        m_days: Lookback period for average calculation
    
    Returns:
        bool: True if volume surge detected
    """
```

#### Configuration Parameters
- **N (Multiplier)**: 1.5x, 2x, 3x, 5x, 10x
- **M (Days)**: 5, 10, 20, 30, 60 trading days
- **Default**: 2x volume over 5 days

#### Calculation Logic
```python
# Get M days of historical volume data
historical_bars = get_bars(symbol, timeframe="1Day", limit=m_days + 1)
current_volume = historical_bars[-1].volume
avg_volume = sum([bar.volume for bar in historical_bars[:-1]]) / m_days

# Check surge condition
volume_surge = current_volume >= (avg_volume * n_multiplier)
```

## Architecture Design

### Class Structure

#### 1. AlpacaScreener (Main Class)
```python
from atoms.api.init_alpaca_client import init_alpaca_client

class AlpacaScreener:
    def __init__(self, provider: str = "alpaca", account: str = "Bruce", environment: str = "paper"):
        # Use existing infrastructure for client initialization
        self.client = init_alpaca_client(provider, account, environment)
        self.screener_client = ScreenerClient(api_key, secret_key)  # From alpaca-py SDK
        
        # Store configuration for reference
        self.provider = provider
        self.account = account 
        self.environment = environment
    
    def screen_stocks(self, criteria: ScreeningCriteria) -> List[StockResult]:
        """Main screening method"""
    
    def detect_volume_surges(self, symbols: List[str], n: float, m: int) -> List[VolumeSurge]:
        """Volume surge detection"""
```

#### 2. ScreeningCriteria (Configuration)
```python
@dataclass
class ScreeningCriteria:
    # Price filters
    min_price: Optional[float] = 0.75
    max_price: Optional[float] = None
    
    # Volume filters  
    min_volume: Optional[int] = 1_000_000
    min_avg_volume_5d: Optional[int] = None
    
    # Change filters
    min_percent_change: Optional[float] = None
    max_percent_change: Optional[float] = None
    
    # Market cap filters
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    
    # Technical filters
    min_trades: Optional[int] = None
    sma_periods: List[int] = field(default_factory=list)
    
    # Volume surge detection
    volume_surge_multiplier: Optional[float] = None
    volume_surge_days: Optional[int] = None
```

#### 3. StockResult (Output)
```python
@dataclass
class StockResult:
    symbol: str
    price: float
    volume: int
    percent_change: float
    dollar_volume: float
    day_range: float
    market_cap: Optional[float] = None
    trades: Optional[int] = None
    avg_volume_5d: Optional[float] = None
    avg_range_5d: Optional[float] = None
    volume_surge_detected: bool = False
    volume_surge_ratio: Optional[float] = None
```

### Data Pipeline

#### 1. Symbol Universe
```python
def get_active_symbols() -> List[str]:
    """Get list of actively traded symbols"""
    # Use Alpaca's most_actives endpoint
    # Filter by volume/price criteria
    # Return symbol list
```

#### 2. Data Collection
```python
def collect_stock_data(symbols: List[str]) -> Dict[str, StockData]:
    """Parallel data collection for all symbols"""
    # Batch API calls for efficiency
    # Collect current + historical data
    # Handle rate limiting
```

#### 3. Filtering Engine
```python
def apply_filters(data: Dict[str, StockData], criteria: ScreeningCriteria) -> List[StockResult]:
    """Apply all screening criteria"""
    # Price filters
    # Volume filters
    # Technical indicators
    # Volume surge detection
```

## External Data Integration

### Fundamental Data Requirements
For complete screening functionality, integration with fundamental data providers is required:

#### Recommended Providers
1. **Alpha Vantage** - Fundamental data API
2. **Financial Modeling Prep** - Market cap, float, EPS data
3. **Quandl/Nasdaq Data Link** - Historical fundamentals

#### Integration Pattern
```python
class FundamentalDataProvider:
    def get_market_cap(self, symbol: str) -> float:
        """Get market capitalization"""
    
    def get_shares_outstanding(self, symbol: str) -> float:
        """Get shares outstanding for float calculation"""
    
    def get_eps(self, symbol: str) -> float:
        """Get earnings per share"""
```

## Performance Optimization

### Caching Strategy
- **Redis/Memory**: Cache fundamental data (daily refresh)
- **Local Storage**: Cache historical bars (hourly refresh)
- **API Rate Limiting**: Implement exponential backoff

### Batch Processing
- **Symbol Batching**: Process 100-200 symbols per API call
- **Parallel Processing**: Use asyncio for concurrent requests  
- **Data Chunking**: Split large symbol lists into manageable chunks

### Real-time Updates
- **WebSocket Integration**: Use Alpaca's streaming API for real-time price/volume
- **Delta Updates**: Only recalculate changed metrics
- **Scheduled Scans**: Run full scans during pre-market hours

## Configuration Files

### config/screener_config.py
```python
SCREENING_DEFAULTS = {
    'min_price': 0.75,
    'min_volume': 1_000_000,
    'volume_surge_multiplier': 2.0,
    'volume_surge_days': 5,
    'max_symbols_per_scan': 3000,
    'api_rate_limit': 200,  # requests per minute
}

TIMEFRAMES = {
    'realtime': '1Min',
    'intraday': '5Min', 
    'daily': '1Day',
    'historical': '1Day'
}

DATA_FEEDS = {
    'free': 'iex',
    'premium': 'sip',
    'extended_hours': 'boats'
}
```

## Command Line Interface

### Usage Examples
```bash
# Basic screening with price and volume filters  
python code/alpaca_screener.py --min-price 0.75 --min-volume 1000000

# Volume surge detection
python code/alpaca_screener.py --volume-surge 2.0 --surge-days 5

# Combined screening with account/environment selection
python code/alpaca_screener.py --account-name Bruce --account paper --min-price 1.0 --max-price 50.0 --volume-surge 3.0

# Multi-account screening
python code/alpaca_screener.py --account-name "Dale Wilson" --account live --min-volume 500000

# Export results
python code/alpaca_screener.py --min-volume 500000 --export-csv results.csv --export-json results.json
```

### Argument Specification
```python
# Account and environment configuration (leveraging existing infrastructure)
parser.add_argument('--provider', default='alpaca', help='API provider (default: alpaca)')
parser.add_argument('--account-name', default='Bruce', help='Account name (Bruce, Dale Wilson, Janice)')
parser.add_argument('--account', default='paper', help='Account type (paper, live, cash)')

# Screening criteria
parser.add_argument('--min-price', type=float, help='Minimum stock price (USD)')
parser.add_argument('--max-price', type=float, help='Maximum stock price (USD)')
parser.add_argument('--min-volume', type=int, help='Minimum daily volume (shares)')
parser.add_argument('--volume-surge', type=float, help='Volume surge multiplier (e.g., 2.0 for 2x)')
parser.add_argument('--surge-days', type=int, default=5, help='Days for volume surge calculation')
parser.add_argument('--min-percent-change', type=float, help='Minimum percent change')
parser.add_argument('--max-percent-change', type=float, help='Maximum percent change')

# Output options  
parser.add_argument('--export-csv', help='Export results to CSV file')
parser.add_argument('--export-json', help='Export results to JSON file')
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
```

## Testing Strategy

### Unit Tests
- Individual metric calculations
- Volume surge detection algorithm
- Data validation and filtering
- API error handling

### Integration Tests
- End-to-end screening pipeline
- External API integrations
- Performance benchmarks
- Rate limiting compliance

### Mock Data Tests
```python
def test_volume_surge_detection():
    # Test with known volume patterns
    # Verify N times calculation
    # Edge cases (zero volume, missing data)
```

## Deployment Considerations

### Environment Variables
```bash
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://data.alpaca.markets
FUNDAMENTAL_DATA_API_KEY=external_provider_key
REDIS_URL=redis://localhost:6379
```

### Monitoring and Logging
- API usage tracking
- Performance metrics
- Error rate monitoring  
- Data quality validation

### Scalability
- Containerization (Docker)
- Horizontal scaling capabilities
- Database optimization for large result sets
- CDN integration for web interface

## Security Best Practices

### API Key Management
- Environment-based configuration
- Secure credential storage
- API key rotation procedures
- Access logging and monitoring

### Data Protection  
- Input validation and sanitization
- Rate limiting protection
- Error message security (no sensitive data exposure)
- Secure logging practices

## Expected Output Format

### Console Output
```
Alpaca Stock Screener Results
=============================
Scan completed at: 2025-08-27 14:30:00 EST
Criteria applied: Min Price: $0.75, Min Volume: 1M shares, Volume Surge: 2x over 5 days

Results found: 23 stocks

Symbol  Price   Volume     %Change  $Volume      Day Range  Surge
------  -----   --------   -------  ----------   ---------  -----
AAPL    $175.50  45.2M     +2.35%   $7.94B       $3.21      No
NVDA    $892.15  89.7M     +8.42%   $80.0B       $45.80     Yes (3.2x)
TSLA    $234.80  112.5M    +12.18%  $26.4B       $15.60     Yes (4.1x)
```

### JSON Export Format
```json
{
  "scan_metadata": {
    "timestamp": "2025-08-27T14:30:00Z",
    "total_symbols_scanned": 2847,
    "results_count": 23,
    "criteria": {
      "min_price": 0.75,
      "min_volume": 1000000,
      "volume_surge_multiplier": 2.0,
      "volume_surge_days": 5
    }
  },
  "results": [
    {
      "symbol": "NVDA",
      "price": 892.15,
      "volume": 89700000,
      "percent_change": 8.42,
      "dollar_volume": 80000000000,
      "day_range": 45.80,
      "trades": 125000,
      "avg_volume_5d": 67200000,
      "avg_range_5d": 38.20,
      "volume_surge_detected": true,
      "volume_surge_ratio": 3.2,
      "market_cap": 2200000000000
    }
  ]
}
```

This specification provides a comprehensive framework for implementing a production-ready stock screener using Alpaca Trading API v2 with advanced volume surge detection capabilities.