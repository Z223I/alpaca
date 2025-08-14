# Trade Generator

## High Level Requirements

You are a highly skilled Python developer and software architect.

## Mid Level Requirements

Make a backup copy of atoms/alerts/config.py


Where to pass parms to code?  Which script(s)?

---Put these values into a config file, say data/backtesting/parameters.json.
---Use the values in data/backtesting/parameters.json to create a nested loop of values for atoms/alerts/config.py

--- must deal with all the dirs.

Parametric Testing using atoms/alerts/config.py
- trend_analysis_timeframe_minutes: int = timeframe where timeframe in (10, 15, 20, 25, 30)
- green_threshold: float = threshold where threshold in (.60, .65, .70, .75)

### Background Information

One historical run per day.
Each run will execute:
code/orb_pipeline_simulator.py --symbols-file [symbols file] --date YYYY-MM-DD --save-alerts --speed 10 --verbose
code/orb_alerts_monitor.py --date YYYY-MM-DD --no-telegram --verbose
code/orb_alerts_monitor_superduper.py --no-telegram --date YYYY-MM-DD --verbose
code/orb_alerts_trade_stocks.py --date YYYY-MM-DD --no-telegram --test --verbose

### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

Read data/backtesting/symbols.json and collect symbols by date where active = 'yes'.

Each date will be an historical run.

### Individual Runs

For each run:
1) Create a symbols file for the date.
  a) copy ./data/YYYYMMDD.csv
  b) keep only the symbols from symbols.json
2) Create processes for each script in Background Infomation
3) Stop when the data pipeline is complete
4) For each symbol
  a) python code/alpaca.py --plot --symbol [symbol]

### Summary

Summarize the individual runs for superduper alerts sent and test trades.
Create pie chart for superduper alerts sent by date.
Create pie chart for trades by date.
