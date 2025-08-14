# Trade Generator

## High Level Requirements

You are a highly skilled Python developer and software architect.

## Mid Level Requirements

You will be adding --monitor-positions for code/alpaca.py.
--account-name and --account are optional because they have defaults.  They are important though.

The new method, _monitorPositions(), will poll the Alpaca account for positions using from atoms.api.get_positions import get_positions.

The polling shall continue until the script instance is stopped.

The polling period is one minute.

### Background Information

One historical run per day.
Each run will execute:
??? data pipeline ??? code/orb_alerts.py --date YYYY-MM-DD --symbols-file <SYMBOLS_FILE> --verbose
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

Summary the individual runs for superduper alerts sent and test trades.