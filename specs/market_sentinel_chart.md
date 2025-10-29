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

Any Alpaca related code that is an atom is to go into ./cgi-bin/atoms/alpaca.  Existing code is in ./atoms/api.

Any Alpaca related code that is a molecule is to go into ./cgi-bin/molecules/alpaca.  Existing code is in ./code.

Only work in the standard GoDaddy directories.

You may use existing code in other directories as context but leave it unchanged.

### Background Information

#### Review

##### Display Chart

atoms/display/generate_chart_from_df.py

##### Momentum Alerts

code/alpaca.py --plot

##### Live SIP Data

Review Alpaca Trading API v2 to determine retrieval of live SIP time and sales data.  Have a fallback if SIP is not available.

### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

## Create alpaca_config.py

Create ./cgi-bin/molecules/alpaca/alpaca_config.py as an exact copy of code/alpaca_config.py.

## Create alpaca.py

Create ./cgi-bin/molecules/alpaca/alpaca.py.  It is only to be used for collecting and displaying stock data.  Ignore trading related functionality.

### Create index.html

Create index.html

#### Create "Search" Feature

Use a magnifying glass to indicate the search feature.

When the user presses the magnifying glass or presses enter create a chart panel.

#### Create Chart Panel

Each chart panel has a tab for easy selection.

Each chart panel is to be continously updated using SIP data.  I do have access to SIP data for my Alpaca account.

Each chart panel is to contain a candlestick chart.  Allow the selection of candlestick time intervals or 10, 20, 30 seconds; 1, 5, 30 minutes; 1 hour; 1 day; 1 week; 1 month.  Allow the selection of the display range 1 day; 2 day; 5 day; 1 month; and 1 year.

The candlestick chart is to have options for 9, 20, 21, 50, 200 EMA; VWAP directly from the Alpaca data; volume display; MACD display.

Each chart panel is to contain and time and sales window.  It is to be located to the right of the candlestick chart.

This panel and subpanels should be resizable.
