# Trade Generator

## High Level Requirements

You are a highly skilled Python developer and software architect.

## Mid Level Requirements

### Background Information

Review code/orb_alerts_monitor_superduper.py
Review atoms/alerts/superduper_alert_generator.py
Review code/alpaca_config.py
Review code/alpaca.py to see what object is returned in an order confirmation.  And then determine what fields are in the object.  Search the internet if necessary.

# Example Trailing sell order with custom trailing percentage
python3 code/alpaca.py --sell-trailing --symbol AAPL --amount 100 --trailing-percent 5.0 --submit


### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

### Task

Mirror atoms/alerts/superduper_alert_generator.py: create atoms/alerts/trade_generator.py.

## Low Level Requirements

### Symbol Extraction

Monitor the directory historical_data/YYYY-MM-DD/superduper_alerts_sent/bullish/green/

The filenames in that directory follow a specific format.  Extract the stock symbol from the file name.
Example:
Input:
superduper_alert_LOBO_20250802_135836
Output:
LOBO

### Issue Trades

The code that you are mirroring is saving alert information to a file.

This new code is going to save information about stock trades using alpaca.py.  You will store the trades in the directory
structure historical_data/YYYY-MM-DD/<account name>/<account> (account will be paper, live, or cash).  Use JSON.

code/alpaca.py will give the results of the trade.  Keep the results as a text string. Keep a success field with "yes" or "no". Have fields for symbol, quantity, average fill price.  You should have figured out what fields are in an order confirmation already.

Using the data in code/alpaca_config.py
trade_count = 0
for each <account name>
    for each <account>
        if auto_trade="yes"
            trade_count += 1
            if trade_count == 1: call code/alpaca.py --sell-trailing --symbol <symbol> --amount <auto_amount> --trailing-percent <trailing_percent>

Important. This is extremely powerful code.  The code above specificly allows a maximum of a single trade while the code is initially tested.  It will be updated later. Actually, I removed the --submit arg. So, set success = "no" when receiving the dry run message from alpaca.py and save that message to the results field.

