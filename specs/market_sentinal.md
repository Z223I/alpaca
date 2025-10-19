# Market Sentinal

## High Level Requirements

You are a highly skilled web designer, Python developer, and software architect.
You will be using Python and Flask to be compatible with GoDaddy websites.  I will have the appropriate hosting plan.

## Mid Level Requirements

Create ./public_html/index.html to be the "Market Sentinal" web interface.

Use ONLY standard GoDaddy directories:
  - public_html/
  - cgi-bin/ (Using atoms/molecules directory structure.)
  - logs/

Create README_market_sentinal.md and keep meaningful notes to yourself about this project.  We will be building this project in steps.

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

### Create "Sentinal" Panel

Create a "Sentinal" panel.

Columns:
Symbol, "% Gain from Close", "% Gain from Open", "% Vol Surge", "News"


Top gainers since last close
Top gainers today since market open
Slow burners
Fast burners
Can click on the above symbols to open the chart.
Charts with tabs
In the top gainers panel, also have a volume surge column yes, or no, and a column for the amount of the surge. also have a news column. Defaults to unknown. Can check it to indicate that it does have news. Or mark it for no news. Is there a tri-state check box?
