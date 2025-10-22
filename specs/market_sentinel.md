# Market Sentinel

## High Level Requirements

You are a highly skilled web designer, Python developer, and software architect.
You will be using Python and Flask to be compatible with GoDaddy websites.  I will have the appropriate hosting plan.

## Mid Level Requirements

Create ./public_html/index.html to be the "Market Sentinel" web interface.

Use ONLY standard GoDaddy directories:
  - public_html/
  - cgi-bin/ (Using atoms/molecules directory structure.)
  - logs/

Create README_market_sentinel.md and keep meaningful notes to yourself about this project.  We will be building this project in steps.

Any Alpaca related code that is an atom is to go into ./cgi-bin/atoms/api.  Existing code is in ./atoms/api.

Any Alpaca related code that is a molecule is to go into ./cgi-bin/alpaca.  Existing code is in ./code.

Only work in the standard GoDaddy directories.

You may use existing code in other directories as context but leave it unchanged.

### Background Information

#### Review

##### Symbol Polling

atoms/api/symbol_polling.py

##### Momentum Alerts

code/momentum_alerts.py

### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

### Create index.html

Create index.html

### Create "Sentinel" Panel

Create a "Sentinel" panel.

Columns:
Symbol, "% Gain from Close", "% Gain from Open", "% Vol Surge", "News"

#### Symbol

Use symbols from historical_data/YYYY-MM-DD/premarket/gainers_nasdaq_amex.csv. This file is static for the day.
Header:
symbol,current_price,previous_close,gain_percent,premarket_volume,premarket_high,premarket_low,premarket_range,dollar_volume,total_premarket_bars,current_timestamp

Also use the symbols from historical_data/YYYY-MM-DD/scanner/relative_volume_nasdaq_amex.csv. This file changes throughout the day.
Use a file watchdog to detect changes.
Header:
symbol,price,volume,percent_change,dollar_volume,day_range,timestamp,trades,avg_volume_5d,avg_range_5d,volume_surge_detected,volume_surge_ratio

If it exists, also use data/YYYYMMDD.csv.
Header:
Symbol,Signal,Long Delta,Resistance,Max,Last,%Chg,Volume,Float,Mkt Cap


#### News

This is a tri-state button or checkbox.  Values "Yes", "No", "Unk".  Defaults to "Unk" for unknown.


Top gainers since last close
Top gainers today since market open
Slow burners
Fast burners
Can click on the above symbols to open the chart.
Charts with tabs
In the top gainers panel, also have a volume surge column yes, or no, and a column for the amount of the surge. also have a news column. Defaults to unknown. Can check it to indicate that it does have news. Or mark it for no news. Is there a tri-state check box?
