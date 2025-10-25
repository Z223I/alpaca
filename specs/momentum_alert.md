# Momentum Alerts

## High Level Requirements

You are a highly skilled Python developer and software architect.

## Mid Level Requirements

Create code/momentum_alerts.py based on low level requirements.

### Background Information

#### Review

Review atoms/api/get_stock_data.py. You will be using it to retrieve stock data.  See code/orb_alerts.py for example usage.

Review atoms/alerts/breakout_detector.py for EMA9 calculation.


### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

### Startup

On startup, run the following script. It can take up to 20 minutes for the script to complete. Run this script every hour for four hours.

""" bash
python code/market_open_top_gainers.py  --exchanges NASDAQ AMEX  --max-symbols 7000  --min-price 0.75  --max-price 100.00  --min-volume 50000 --top-gainers 20 --export-csv gainers_nasdaq_amex.csv --verbose
"""

### Main Loop

1) Monitor the creation ./historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv which is the output of code/market_open_top_gainers.py.

#### Stock Monitoring sub-loop

1) Every minute, collect the last 30 minutes of 1-minute candlesticks for each stock in the CSV file.
2) Process each stock symbol for momentum alerts described separately.

### Momentum Alerts

Given the data for a stock:

Only generate momentum alerts for symbols meeting the following requirements:

- The latest candlestick must be above VWAP.  VWAP will be in the stock data.
- The latest candlestick must be above EMA9.
- Use atoms/alerts/config.py:get_urgency_level_dual() to determine if the stock is to be filter for momentum or momentum short.

For generated alerts, use atoms/telegram/telegram_post.py:send_message_to_user() to send it to 'Bruce'.


### Integration

- Integrate into code/alerts_watch.py to be ran with the other alerts.
- Document the integration into Daily_Instructions.md.