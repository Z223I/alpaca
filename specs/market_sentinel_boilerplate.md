# Market Sentinel

## High Level Requirements

You are a highly skilled web designer, Python developer, and software architect.
You will be using Python and Flask to be compatible with GoDaddy websites.  I will have the appropriate hosting plan.

## Mid Level Requirements

Use ONLY standard GoDaddy directories:

- public_html/
- cgi-bin/ (Using atoms/molecules directory structure.)
- logs/
- services/

Create README_market_sentinel.md and keep meaningful notes to yourself about this project.  We will be building this project in steps.

Any Alpaca related code that is an atom is to go into ./cgi-bin/atoms/alpaca_api.  Existing code is in ./atoms/api.

Any Alpaca related code that is a molecule is to go into ./cgi-bin/molecules/alpaca_molecules.  Existing code is in ./code.

Only work in the standard GoDaddy directories.

You may use existing code in other directories as context but leave it unchanged.

### Background Information

#### Review

##### Web Server

The Apache 2 web server is running.  The ./public_html files can be reached using http://localhost/market_sentinel/.

See APACHE_SETUP.md and keep it up to date.

##### Momentum Alerts

For context, see:
code/momentum_alerts.py
code/momentum_alerts_config.py

### Standards

Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

### Update public_html/index.html

#### Scanner Panel

Add "Scanner" panel below the Watch List panel

Columns:
Source
Time (Always ET)
Gain
Volume
Text

#### Use Momentum Alerts

Populate the Scanner Panel at the top for Momentum Alerts.  Check for the trigger of the popup window and use the trigger.  It is probably the creation of a momentum alert sent file.

Set Source = "Momentum"

Populate Time, Gain, and Volume from the Momentum alert.

Populate the "Text" field with "Momentum", "Momentum Short", "Squeezing" and "Halt Status"

