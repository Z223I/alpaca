# Market Sentinel - SIP Data Configuration

## Overview

Market Sentinel now uses **SIP (Securities Information Processor) data from your Alpaca LIVE account** for real-time market data.

**IMPORTANT: This is READ-ONLY market data access. NO TRADING OPERATIONS are performed.**

## Why Use Live Account for Market Data?

The Market Sentinel interface has been configured to use your **live account** instead of paper account for the following reasons:

1. **Real-time Data**: Live account provides real-time SIP data with minimal delay
2. **Accurate Trade Data**: Paper account has 15+ minute to hour delays on individual trade (Time & Sales) data
3. **Better Quality**: SIP data from consolidated tape is more accurate and complete
4. **No Trading Risk**: Market data API is completely separate from trading API - there is zero risk of accidental trades

## How to Configure Your Credentials

### Option 1: Environment Variables (Recommended)

Set these environment variables in your shell or `.env` file:

```bash
export ALPACA_LIVE_API_KEY="your_live_api_key_here"
export ALPACA_LIVE_SECRET_KEY="your_live_secret_key_here"
```

### Option 2: Edit Configuration File Directly

Edit the file: `cgi-bin/molecules/alpaca_molecules/alpaca_config.py`

Find lines 136-137 and update with your credentials:

```python
live=EnvironmentConfig(
    app_key=os.getenv("ALPACA_LIVE_API_KEY", "YOUR_LIVE_KEY_HERE"),
    app_secret=os.getenv("ALPACA_LIVE_SECRET_KEY", "YOUR_LIVE_SECRET_HERE"),
    url="https://api.alpaca.markets",
    # ... other settings
)
```

## Where to Get Your Live API Credentials

1. Log in to your Alpaca account at https://alpaca.markets
2. Go to **Paper Trading** → Switch to **Live Trading**
3. Navigate to **Your API Keys** section
4. Generate new API keys or use existing ones
5. Copy the **API Key ID** and **Secret Key**

## Current Configuration

The following files have been updated to use live account SIP data:

- `cgi-bin/molecules/alpaca_molecules/market_data.py` - Default changed from "paper" to "live"
- `cgi-bin/api/market_data_api.py` - Documentation added explaining SIP usage

## Security Notes

- Your secret keys should NEVER be committed to git
- Use environment variables when possible
- The `.env` file is already in `.gitignore` (if you use one)
- These credentials are only used for READ-ONLY market data access

## Verification

To verify your credentials are working, check the Market Sentinel interface:

1. Open a symbol (e.g., AAPL, MTC)
2. Check the Time & Sales panel on the right
3. If data is recent (within 1-2 minutes), SIP data is working correctly
4. Trades marked with exchange "BAR" are fallback data from 1-minute bars
5. Trades with actual exchange codes (Q, D, P, etc.) are real SIP trades

## Troubleshooting

**Issue**: "Authentication failed" or API errors
- **Solution**: Verify your API keys are correct and have market data permissions

**Issue**: Still seeing delayed data
- **Solution**: Make sure you're using live account keys, not paper account keys

**Issue**: Time & Sales showing "BAR" exchange
- **Solution**: This is normal fallback when recent tick data isn't available. The data is still accurate from 1-minute bars.

## Contact

If you have issues with configuration, check:
1. Your API keys are from the live account
2. Environment variables are set correctly
3. The alpaca_config.py file has the correct credentials
