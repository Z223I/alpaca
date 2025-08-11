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

Review atoms/utils/calculate_macd.py
Review atoms/utils/macd_alert_scorer.py

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