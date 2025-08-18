# Telegram Plots

## High Level Requirements

You are a highly skilled Telegram API expert, Python developer and software architect.

## Mid Level Requirements

Create code/backtesting.py based on low level requirements.

Parametric Testing using atoms/alerts/config.py
- trend_analysis_timeframe_minutes: int = timeframe where timeframe in (10, 15, 20, 25, 30)
- green_threshold: float = threshold where threshold in (.60, .65, .70, .75)

Put these values into a config file, say data/backtesting/parameters.json.
Use the values in data/backtesting/parameters.json to create a nested loop of values for atoms/alerts/config.py

### Background Information

One historical run per day.
Each run will execute:
code/orb_pipeline_simulator.py --symbols-file [symbols file] --date YYYY-MM-DD --save-alerts --speed 10 --verbose
code/orb_alerts_monitor.py --date YYYY-MM-DD --no-telegram --verbose
code/orb_alerts_monitor_superduper.py --no-telegram --date YYYY-MM-DD --verbose
code/orb_alerts_trade_stocks.py --date YYYY-MM-DD --no-telegram --test --verbose

These scripts are already aware of atoms/alerts/config.py and the parameters within.

### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

Read data/backtesting/symbols.json and collect symbols by date where active = 'yes'.

Each date will be an historical run.

### On Startup

Make a backup copy of atoms/alerts/config.py

something about the dirs

### Individual Runs

Use the values in data/backtesting/parameters.json for the nested for loop.

For each run:
Establish the run directory name using the format ./runs/run_YYYY-MM-DD_<UUID>

1) Update atoms/alerts/config.py to contain the run directory name.  Update these classes.
DEFAULT_PLOTS_ROOT_DIR = PlotsRootDir()
DEFAULT_DATA_ROOT_DIR = DataRootDir()
DEFAULT_LOGS_ROOT_DIR = LogsRootDir()
DEFAULT_HISTORICAL_ROOT_DIR = HistoricalRootDir()
DEFAULT_PRICE_MOMENTUM_CONFIG = PriceMomentumConfig()

You must set these values using the loop variables:
trend_analysis_timeframe_minutes
green_threshold


2) Create a symbols file for the date.
  a) copy ./data/YYYYMMDD.csv to DataRootDir() / data
  b) keep only the symbols from symbols.json and of course the current date
3) Create processes for each script in Background Infomation
4) Stop when the data pipeline is complete
5) For each symbol
  a) python code/alpaca.py --plot --symbol [symbol] --date YYYY-MM-DD
6) Plots
Create pie chart for superduper alerts sent by symbol.
Create pie chart for trades by symbol.


### After Run Complete

Restore the backup copy of atoms/alerts/config.py on completion.

Summarize the individual runs for superduper alerts sent and test trades.
Create pie chart for superduper alerts sent by date.
Create pie chart for trades by date.
