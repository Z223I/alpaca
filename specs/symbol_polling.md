# Momentum Alerts

## High Level Requirements

You are a highly skilled Python developer and software architect.

## Mid Level Requirements

Create atoms/api/symbol_polling.py based on low level requirements.

### Background Information

#### Review

code/alpaca_config.py to understand how to retrieve ALPACA_API_KEY,ALPACA_API_SECRET based on account-name and account.




### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

### Initial Script

Start with the following script:

"""python
import threading,time
from config import ALPACA_API_KEY,ALPACA_API_SECRET,SYMBOLS,PRICE_POLL_INTERVAL
from alpaca_trade_api import REST
from alert import send_alert
rest=REST(ALPACA_API_KEY,ALPACA_API_SECRET)

def poll_prices():
    while True:
        try:
            for sym in SYMBOLS:
                t=rest.get_last_trade(sym)
                send_alert("Price",f"{sym}: {t.price}","",[sym])
        except Exception as e:
            print("Price poll error:",e)
        time.sleep(PRICE_POLL_INTERVAL)

def run_price_poll_thread():
    threading.Thread(target=poll_prices,daemon=True).start()
"""

### Main Method

Add main method to script and have it call run_price_poll_thread();

### Create Args

Create CLI args --account-name and --account to be used to retrieve ALPACA_API_KEY,ALPACA_API_SECRET from the alpaca_config.py file. Use 'bruce' and 'paper' as the defaults.

--verbose

--test Uses the most recent data/{YYYYMMDD}.csv and historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv as static data.

### Symbols

Retrieve SYMBOLS from data/{YYYYMMDD}.csv.

Monitor for the creation of historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv. It is updated many times per day.

Each time historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv is updated, update symbols to be unique from both sources.

### Poll Interval

PRICE_POLL_INTERVAL = 5 seconds.

### Create send_alert method

Create a local send_alert() method that just prints the args on a single line.  Of course, get rid of the import.

### Create rest Object

Use atoms/api/init_alpaca_client.py:init_alpaca_client() to create the 'rest' object instead of the pre-existing assignment.
