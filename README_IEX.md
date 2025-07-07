# IEX Feed Explanation

**IEX (Investors Exchange)** is a stock exchange that provides market data through Alpaca's websocket service. Here's what you need to know:

## What is IEX?
- **IEX** = Investors Exchange, a U.S. stock exchange launched in 2016
- **Mission**: Designed to reduce high-frequency trading advantages and provide fairer markets
- **Regulatory**: SEC-registered stock exchange, fully legitimate and regulated

## IEX vs SIP Data Feeds

### IEX Feed (What we're using)
- ✅ **Free** with Alpaca paper trading accounts
- ✅ **Real-time** market data
- ✅ **Good coverage** of major stocks
- ⚠️ **Limited**: Only shows trades executed on IEX exchange (~2-3% of total volume)
- **URL**: `wss://stream.data.alpaca.markets/v2/iex`

### SIP Feed (What we bypassed)
- 💰 **Paid** subscription required (~$100+/month)
- ✅ **Complete** market data from all exchanges
- ✅ **Full volume** and comprehensive price discovery
- **URL**: `wss://stream.data.alpaca.markets/v2/sip`

## Why IEX Works for ORB Alerts

**ORB (Opening Range Breakout)** strategy benefits from IEX because:

1. **Price Discovery**: IEX prices are accurate for trend detection
2. **Volume Patterns**: Even partial volume data shows relative activity spikes
3. **Breakout Detection**: Price movements above/below ORB levels are valid regardless of exchange
4. **Cost Effective**: Free data for development and testing

## IEX Data Quality

**What You Get:**
- ✅ Real-time price updates
- ✅ Volume data (IEX portion)
- ✅ OHLC (Open, High, Low, Close) bars
- ✅ Trade count and VWAP

**Limitations:**
- ⚠️ ~2-3% of total market volume
- ⚠️ May miss some price movements in low-activity periods
- ⚠️ Less granular than full SIP data

## Our Implementation

In our system:
```python
# This determines which feed to use
def _get_websocket_url(self) -> str:
    if "paper" in self.base_url:
        return "wss://stream.data.alpaca.markets/v2/iex"  # Free IEX
    else:
        return "wss://stream.data.alpaca.markets/v2/sip"  # Paid SIP
```

## Configuration

The system automatically selects the appropriate feed based on your `.env` configuration:

```env
# For IEX feed (free)
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# For SIP feed (paid subscription required)
ALPACA_BASE_URL=https://api.alpaca.markets
```

## For Production Use

**Current Setup (IEX)**: Perfect for:
- ✅ Development and testing
- ✅ ORB strategy backtesting
- ✅ Proof of concept trading
- ✅ Small-scale automated trading

**Upgrade to SIP When**:
- 💰 Higher trading volumes (>$10k/day)
- 💰 Need complete market picture
- 💰 High-frequency strategies
- 💰 Professional trading operation

## Data Storage

The system stores IEX data in CSV format with the following structure:
```csv
timestamp,symbol,high,low,close,volume,vwap,trade_count
2025-07-07 14:36:00+00:00,SONY,25.455,25.455,25.455,100,25.455,1
```

Historical data is saved to `historical_data/YYYY-MM-DD/market_data/` every 1 minute.

## Bottom Line

IEX feed provides **sufficient quality data** for ORB alerts at **zero cost**, making it ideal for development and moderate-scale trading. The data is real, regulated, and timely - just not as comprehensive as paid alternatives.

For most ORB trading strategies, IEX data quality is more than adequate for identifying breakout patterns and generating profitable trading signals.