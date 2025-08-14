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


Note to create symbols file
ok code/orb_alerts.py --date YYYY-MM-DD --symbols-file <SYMBOLS_FILE> --verbose
ok code/orb_alerts_monitor.py --date YYYY-MM-DD --no-telegram --verbose

ok code/orb_alerts_monitor_superduper.py --no-telegram --date YYYY-MM-DD --verbose
ok code/orb_alerts_trade_stocks.py --date YYYY-MM-DD --no-telegram --verbose

### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

If there are positions, create a list of unique stock symbols.

For each unique symbol,
  - Collect sufficient data for calculate_macd
  - Calculate macd
  - Score macd
  - If the MACD score is a red, liquidate the position using the same method as used by the --liquidate arg.