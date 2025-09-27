# Daily Instructions

Always running

```bash
python molecules/telegram_polling_watchdog.py
```

The watchdog automatically starts and monitors the telegram polling service. It will restart the service if errors are detected and creates timestamped logs in the logs/ directory.

Morning

## Collect Symbols

### Windows Machine

### Ubuntu Machine

## Execute Code

### New

```bash
python code/volume_profile_bulk.py
```

### Automated Alerts Watch (Recommended)

Fully automated system that manages all ORB alert processes during market hours (9:30 AM - 4:00 PM ET, Mon-Fri). Automatically starts all processes at market open, monitors them, and runs post-market analysis.

```bash
python code/alerts_watch.py
```

Features:
- Automatic market hours scheduling (Eastern Time)
- Starts/stops all 6 alert processes automatically (including orb_alerts.py, vwap_bounce_alerts.py, and momentum_alerts.py)
- Process monitoring and restart on failures
- Post-market analysis and Telegram summary to Bruce
- Comprehensive logging

This replaces the need to manually run orb_watchdog.py and individual commands below.

### Manual ORB Alert Commands (Alternative)

For manual control or testing, use these individual commands:

#### ORB Basic Alerts (DEPRECATED - Use alerts_watch.py instead)

**⚠️ DEPRECATED**: orb_watchdog.py is no longer recommended. Use `alerts_watch.py` which manages orb_alerts.py along with all other processes.

For manual testing only:

```bash
python3 code/orb_alerts.py --verbose
```

Note: alerts_watch.py now handles orb_alerts.py management automatically.

#### ORB Super Alerts

Produces ORB super alerts.

```bash
python3 code/orb_alerts_monitor.py --no-telegram --verbose
```

#### ORB Superduper Alerts

Produces superduper alerts.

```bash
python code/orb_alerts_monitor_superduper.py --verbose
```

#### ORB Trades

Generates trades. Configuration required.

```bash
python code/orb_alerts_trade_stocks.py --verbose
```

#### VWAP Bounce Alerts

Monitors historical market data files for VWAP bounce patterns and sends alerts to Bruce when detected. Automatically analyzes the last 10 1-minute candlesticks, combines them into two 5-minute candles, and alerts when both are green, the first is within 7% above VWAP, and the second is higher.

```bash
python3 code/vwap_bounce_alerts.py --verbose
```

**Features:**
- File watchdog monitoring of `historical_data/YYYY-MM-DD/market_data/` directory
- Real-time analysis of 5-minute green candle patterns near VWAP
- Telegram alerts sent directly to Bruce
- Operating window: 9:45 AM - 8:00 PM ET
- Test mode available: `--test` flag

**Manual Testing:**
```bash
# Test mode (no alerts sent)
python3 code/vwap_bounce_alerts.py --test --verbose

# Monitor specific date
python3 code/vwap_bounce_alerts.py --date 2025-07-18 --verbose
```

#### Momentum Alerts

Monitors top gaining stocks from market open and generates momentum alerts based on VWAP and EMA9 criteria. The system automatically runs the market_open_top_gainers.py script every hour for 4 hours, then monitors the generated CSV for stocks that meet momentum criteria.

```bash
python3 code/momentum_alerts.py --verbose
```

**Features:**
- Automated startup script execution (market_open_top_gainers.py)
- CSV file monitoring for gainers_nasdaq_amex.csv
- Every-minute stock monitoring with 30-minute data collection
- VWAP and EMA9 filtering with dual momentum urgency checks
- Telegram alerts sent directly to Bruce
- Comprehensive logging and error handling

**Manual Testing:**
```bash
# Test mode (no alerts sent)
python3 code/momentum_alerts.py --test --verbose

# Verbose mode for debugging
python3 code/momentum_alerts.py --verbose
```

**Process Flow:**
1. **Startup**: Runs market_open_top_gainers.py every hour for 4 hours with parameters:
   - Exchanges: NASDAQ, AMEX
   - Max symbols: 7000
   - Price range: $0.75 - $40.00
   - Min volume: 50,000
   - Top gainers: 20
   - Output: gainers_nasdaq_amex.csv

2. **Monitoring**: Watches for CSV file creation in `./historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv`

3. **Stock Analysis**: Every minute, collects 30 minutes of 1-minute candlesticks for each stock in the CSV

4. **Momentum Alerts**: Only generates alerts for stocks meeting ALL criteria:
   - Latest candlestick above VWAP
   - Latest candlestick above EMA9
   - Passes dual momentum urgency filter (both momentum and momentum_short must be green)

5. **Telegram Integration**: Sends formatted alerts to Bruce with price, VWAP, EMA9, momentum metrics, and timestamps

After market close

ORB Alert Summary

```bash
python code/orb_alerts_summary.py
```

ORB Analysis with Charts

```bash
python3 code/orb.py
```
