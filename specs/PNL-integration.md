# Trade Generator

## High Level Requirements

You are a highly skilled Python developer and software architect.

## Mid Level Requirements

Update ./code/alpaca.py to add new arg --PNL.

### Background Information

Review ./atoms/api/pnl.py to determine usage.
Review code/alpaca_config.py.  Use the "Primary":"paper" account info

### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

The new arg --PNL is to be used only by itself.

This arg is used to call AlpacaDailyPnL:create_pnl() in atoms/api/pnl.py. Use the "Primary":"paper" account info.