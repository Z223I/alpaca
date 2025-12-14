# TODO

- [ ] Implement EMA divergence quality prediction system following specs/ema_divergence_prd.md (use only data from 2025-07-28 onwards)

The remaining tasks are lower priority documentation and deployment tools:
  - Production configuration and deployment tools
  - Comprehensive API documentation

Be sure to watch stocks minute-by-minute for buying oportunities Jdun Trades style but with 1 min candles.

## Big Stock List

- [ ] Compile and maintain.
- [ ] Perform volume analysis (5X) per Ross Cameron of Warrior Trading

## Multiple Accounts

- [ ] Update to handle multiple account types and people.


## BRB, Breakout(break above) Retest Breakout

## Alpaca MCP server using my implementation

new branch


## Trade

- [ ] Update telegram monitor to accept "trade" from Bruce only to trade the last alert.  Keep the symbol and price for the last sent alert in a variable.  Use that when "trade" is received to place an Alpaca trade.

## Yahoo
- [ ] Update for increase in volume over N months average.

## Mermaid Diagram



Training wheels. No more than X positions at a time. Put it in the configuration file that would be the alpaca configuration file.

Work on filtering. May have to pass candlestick info all the way through the pipeline. Require the candle be at least 50% green and candle close be at 25% or above (H-L).Include the last three candle sticks and the current one.  I think the chart has 1 minute candles.


- [ ] For orb_alerts, get next resistance from Oracle.  No timing out. Can press <Enter> to autofix.


- [ ] Max losses per day

- [ ] When loading stocks, create stocks.json file with every stock having trading='yes'
- [ ] stop <symbol>


- [ ] Finish Daily_Instructions.md



## MACD chart to Telegram

- [ ]


## Major Resistance

- [ ] Add a major resistance field and pass it through the alerts.
- [ ] Color code major resistance in the superduper alert generator.
- [ ] Add time of day signals with green and then yellow for the afternoon



## Test new trade placement

- [ ] ~/miniconda3/envs/alpaca/bin/python execute_aapl_trade.py


Filter out stocks below two dollars. ???


Incorporate major resistance into trading.

Send alert to Bruce the first time a symbol is at 20, 30, etc. percent above orb high in the 30 minutes.

Moving Google doc Trading into a .md file in the repo.

Why didn't I get alerts for THAR on 8/20????

Doc in phone notes close all positions and cancel all orders

## Strategies

- [ ] EMA9

## Webull

## Strategies VWAP Bounce

- [X] VERY IMPORTANT. Create code/vwap_bounce_alerts.py: It is to use a watchdog to monitor the creation of new files in historical_data/YYYY-MM-DD/market_data/[symbol]*.csv; For each symbol for the current date; Take the 10 most recent candlesticks (They are 1-minute each) and combine them into two 5-minute candlesticks. Check if both candlesticks are green.  If the are both green and one of them is within 4% above VWAP, send a VWAP Bounce alert Telegram post to Bruce. It is only to send alerts during the currently configured timeframe that starts close to market open and ends at 20:00 ET.

## Daily Trade Limit

- [ ] VERY IMPORTANT.  Update lsdkfj skdj: Daily trade counts need to be saved to a dated file if the script shuts down during market hours and then restored if restarted during market hours.

## Surge

- [ ] Incoporate surge detection in daily routine.

Find photo clicker
TCM



Filter out stocks that have halted.
Create turn off/on program that accepts multiple symbols. Change all symbols to uppercase.
Update telegram polling for turn off/on.  Change all symbols to uppercase.  Bruce is the only authorized user.
Create a check program for stocks approaching the signal.
Stop limit buy to purchase when the stop limit is hit


Check a penetration of 100% is required


Filter out stocks that have halted.
Create shut off program that accepts multiple symbols
Create a check program for stocks approaching the signal.
Stop limit buy to purchase when the stop limit is hit

Check a penetration of 100% is required

Create a program code/percentage_bin.Py. Have it go through all historical data and bin percentage gains by minute.

Can the data from volume surge be saved, and then used for top gainers? Or, can they be combined and written to two different databases?

Create program and connect to telegram polling to give a list of stocks that have any type of alert. Sort by alert type give the time stamp.


## Major Resistance

Top of Profile Range

```text
============================================================
VOLUME PROFILE ANALYSIS SUMMARY
============================================================
Configuration:
  Time Per Profile: DAY
  Price Mode: AUTOMATIC
  Value Area %: 70.0%

Results:
  Total Profiles: 1
  Time Range: 2025-09-29 04:00:00-04:00 to 2025-09-29 09:25:00-04:00
  Average POC: $0.69
  POC Range: $0.69 - 0.69
  Avg Value Area Width: $0.15

Latest Profile:
  📅 Period: 2025-09-29 04:00:00-04:00 to 2025-09-29 09:25:00-04:00
  🎯 Point of Control: $0.69
  📊 Value Area: $0.59 - $0.74
  📈 Profile Range: $0.45 - $0.77
  📦 Total Volume: 43,584,867
============================================================
```


## Squeeze

- [ ] Monitor 'premarket top gainers' for stocks that are in a squeeze.






Automatically send myself premarket top gainers and top gainers.

Momentum alerts: dial them down.



Telegram polling : integrate volume profile and volume profile bulk.




Momentum alerts: VWAP is wrong

Momentum alerts: BQ halted. On October 2. Momentum alert does not show it halted.  Look for a 90 second gap.




Use volume profile bulk to produce M/R for momentum alerts.


Need an alert for when a symbol hits a price. Maybe use symbol polling.  Use that in conjunction with volume profile to get the prices. Or do that one separately. Need alert when symbol is approaching M/R or do automatically.


Up the volume on the scanners

Real time charts. Add a bar for the ratio of selling versus buying. Buying to the right. have L2 and time and sales.


## MACD alerts

- [ ] Create MACD alerts.  Add that to momentum alerts and then use or.  Only use if in the top three or four top gainers. Otherwise there will be too much noise.  It’ll be necessary to store the number for the top gainer.  Look for the MACD atom. Include the number of consecutive green bars in the alert.  Once on the list always on the list until the program is restarted. Do not include symbols that end in a W.

## Hidden Buyer/Seller

- [ ] Watch alpaca-py L1 for best bid/ask; and Time & Sales to see if there is a hidden buyer or seller.  If there are more sales on the T&S at a specific price than on the L1, there is a hidden buyer/seller.

Use alpaca-py to get the latest stock quote with ask_price, bid_size, and ask_size.
Looking at your code in atoms/api/get_latest_quote.py:5 and atoms/display/print_quote.py:20-21, the current implementation only displays bid_price and ask_price, but the quote object also contains bid_size and ask_size that you can access.

## Why public_html/cgi-bin

- [X] Because it is a symbolic link.



Do you want to add top gainers to squeeze alerts (indirectly)?

## Ask - Bid

- [ ] LSKDJF Update LKDSJFSDLKJ add a panel for the spread

## Wycoff

- [ ] It is light volume during consolidation and then a burst of volume on the uptrend.  See https://youtu.be/woxIjeeS0yw?si=zmevajT60W3y36cZ&t=1276
- [ ] LSKDFJS Update lskdjf: Build volume profile. One second volume candles?

Install TraderVue

## Is VWAP Correct

- [ ] Hit session limit during update to display VWAP on squeeze alerts.  It may be necessary to manual calculate it.  Manully do it and compare the two.

## Volume Profile

- [ ] Create subplot or side plot for volume profile for the last five minutes.




## Reverse Splits

- [ ] Keep reverse splits out of Watchlist.

## News

Add news column to CSV. Manually update it. Pass it through the alerts system.

Add news column to signals. Also, volume surge if possible.

## Sanity Check

- [ ] Ensure data_master/master.csv is not being used to limit stock symbol selection.

## Float Rotation Scanner

- [ ] Create a float rotation scanner.  Can something be extended?

## BRB Alert

- [ ] Write script to review historical data.  Need to perfom analysis to find what works. Use usual suspects price > 9 EMA > 21 EMA; MACD open.
- [ ] Create BRB alerts. Trigger the watching based on an all green squeeze alert minus PM High. Stop monitoring after 10 minutes unless it is going well.

## Premarket Top Gainers Reverse Splits

- [ ] Update code/premarket_top_gainers.py to eliminate reverse splits using yfinance.

## Oracle

- [X] Update public_html/index.html: In the Watchlist panel, the Oracle column is populated with the current date's info and fallbacks to the last oracle data if necessary.  The fallback was for testing.  We are now past testing.  Eliminate the fallback info.

## Fake Trades

- [X] ULTRATHINK. The squeeze alerts we have created seem to work well.  It is now time to test them using fake trades and using the websocket already running in cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py.  Update cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py: create a fake_trade method: the method receives the args symbol, quantity (1 default), trailing_stop_loss (2% default), and take_profit (5% default); it then monitors the price until it hits the take_profit or the trailing_stop_loss (trailing stops follow the price up but never down); store fake trade data in ./historical_data/YYYYMMDD/fake_trades/*.json; there can be multiple instances of fake_trade. After a squeeze alert gets generated, call the fake_trade method.

- [X] ULTRATHINK. Update public_html/index.html: In the upper-righthand corner, keep a tally of the fake trades.  Also, have a button that when clicked creates a popup that lists the fake trades for the current date.  The columns shall be symbol, entry timestamp, entry price, quantity, profit_percent, profit_dollars, time held (exit_timestamp - entry_timestamp).  The columns are also to be sortable.  Have a second icon for a popup for fake trade statistics.

- [X] Review public_html/index.html: what script and socket is it using to subscribe to stock symbol trading data?  Document in docs/README_time_and_sales.md.

- [X] Review cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py: It processes fake trades.  Question: what are the correct words to describe what is being done?  Is it spawn or something else? [simulate, initiate, start, create, or track]

- [X] Review cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py and docs/README_time_and_sales.md.  Would it be better to subscribe the symbol to the websocket every time a fake track is created?
Key Methods (cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py):

  1. Line 2186-2242: _start_fake_trade() - Creates/initiates a simulated trade
  2. Line 2244-2321: _check_fake_trade_conditions() - Monitors active trades on each price update
  3. Line 2323-2366: _finalize_fake_trade() - Closes and saves completed trades
  4. Line 2703-2711 - Where fake trades are started (when a squeeze alert fires)

- [X] Review docs/README_time_and_sales.md to understand how to subscribe and unsubscribe to stock symbol trading data. Update cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py: when a fake trade is created, subscribe to the symbol. That way, something else doesn’t unsubscribe and mess things up.

## Analysis

Test 2, 3, 4, 5 percent with a 2% stop loss.

Why 2% Wins:

  1. Higher Win Rate: 75% (vs 68% for 5%)
    - More consistent returns
    - Builds trading confidence
    - Better risk management
  2. Catches More Opportunities: 79% recall
    - Captures 79% of all 2%+ moves
    - Only misses 21% of winners
    - vs 5% target misses 35% of winners
  3. Better Calibrated Model
    - 71% of squeezes achieve 2% (easier target)
    - 50% of squeezes achieve 5% (harder target)
    - Less noise in classification
  4. Faster Exits
    - Median time to 2%: 45 seconds
    - Median time to 5%: 2.5 minutes
    - Reduces exposure time


## Spread

Do this for data collection for data analysis.
High volume and short candlestick body is sign of a reversal.

- [ ] Update lksjdf: Use code/alpaca.py to retrieve the bid and ask.  Add the spread (ask - bid) to the squeeze alert.  Add spread percentage as (ask - bid) / latest price in squeeze alert.
- [ ] Is the data statistically independent?
- [ ] Is data after the squeeze alert saved to all performance analysis?
- [ ] Add 9/21 EMA, MACD (MACD, Signal, or histogram)

## Surge all stocks

- [X] Update cgi-bin/molecules/alpaca_molecules/momentum_alerts.py: "--symbols-file", "data_master/master.csv", -> "--exchanges", "NASDAQ", "AMEX",
            "--max-symbols", "7000",

```python
        script_path = Path("code") / "alpaca_screener.py"
        cmd = [
            "~/miniconda3/envs/alpaca/bin/python",
            str(script_path),
            "--symbols-file", "data_master/master.csv",
            "--min-volume", "250000",
            "--min-percent-change", "5.0",
            "--surge-days", "50",
            "--volume-surge", "5.0",
            "--export-csv", "relative_volume_nasdaq_amex.csv",
            "--verbose"
        ]
```

## Gain Calculation

- [X] Update the script with debugging statements because I am getting '!!! TRACKED SYMBOL TAOP - FILTERED: gain -19.62% < min 10.0%' but my Webull brokage account shows the gain above 40%. To speed up testing, run code/premarket_top_gainers.py  --symbols TAOP --verbose.  Fix it.

## Missing Top Gainers

- [X] THINK HARD.  Review public_html/index.html: Why hasn't AHMA and QTTB not shown up in the gain column in the Watchlist panel?  Question only.

## Green Background for Squeeze Alerts

- [X] Update public_html/index.html: When adding squeeze alerts to the scanner panel, if the HOD icon is green make the background green. That has highest precendence. If not, if premarket high is green, make the background lime.

## HOD

- [X] Update cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py: Eliminate the line in the squeeze alert that list number of shares traded.
- [X] Review cgi-bin/molecules/alpaca_molecules/market_data.py: Does it ever see or calculate HOD, high of day? [no]
- [X] Update cgi-bin/molecules/alpaca_molecules/market_data.py: To have a function that takes a symbol to calculate Premarket High and Regular hours HOD.
- [X] Update cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py: Update the squeeze alert to include HOD and premarket high light icon. Use cgi-bin/molecules/alpaca_molecules/market_data.py to collect the values.  Use green if price within 1.5% of either value. Red if 5% or more lower.  Otherwise yellow.

## Surge More than Once

- [X] How many time is Surge ran.  I want it more than once.

## Momo

- [X]Are manually added watch list items being sent to momentum_alerts.py?

## Company Names

- [X] Add company names to index.html

## Web Socket

- [ ] Create ./wss_test dir.  Create ./wss_test/quotes.py from the code below.  Use the keys from .env.

```python
from alpaca.data.live import StockDataStream

# Initialize the stream client (requires API keys)
wss_client = StockDataStream('api-key', 'secret-key')

# Define an async handler for quote data
async def quote_data_handler(data):
    print(data)  # Process quote data here

# Subscribe to quotes for specific symbols
wss_client.subscribe_quotes(quote_data_handler, "SPY")

# Start receiving data
wss_client.run()
```

## Top Pre-market Gainers is Wrong

- [X] Oops it is using top gainers instead.  This might be from the Nano.  I need to do a betting job documenting this when I figure out what I meant.

## Stock Symbol Name

- [X] THINK HARD.  Update public_html/index.html and cgi-bin/molecules/alpaca_molecules/momentum_alerts.py: When index.html is preparing a chart, have it also display the company name.  It is to use momentum_alerts.py to retrieve the company_name field from data_master/master.csv.
- [ ] THINK HARD.  Update cgi-bin/molecules/alpaca_molecules/momentum_alerts.py: if unable to retrieve the company_name field from data_master/master.csv, use the same mechanism to retrieve the company name as in molecules/add_company_names.py.

## Premarket Top Gainer to Accept File

- [X] THINK HARD.  Update code/premarket_top_gainers.py to accept a file of symbols like code/alpaca_screener.py does.
- [X] Update cgi-bin/molecules/alpaca_molecules/momentum_alerts.py to call "premarket_top_gainers.py" (See line 492) with "--symbols-file", "data_master/master.csv".

## Other

- [X] Pre-market is failing on server.  Check to see if failing on Orin Nano.  Nano is fine.  Fixed on server.

## Time and Sales

- [X] Off by one hour

## Momentum Alerts Timing

- [X] Review cgi-bin/molecules/alpaca_molecules/momentum_alerts.py. What scripts is it running? What are the timing of the script? and what are the args?
- [X] Update cgi-bin/molecules/alpaca_molecules/momentum_alerts.py. Run an additional script

```python
premarket_script = os.path.join(project_root, 'code', 'premarket_top_gainers.py')
```

Get the args and additional info from where it is executed in molecules/telegram_polling.py.

This script is to run from 0400 to 2000 ET every 20 minutes. Run it in a different CPU core.

The symbols from this run are to be monitored just like the others in this script.

## Index.html Update 1

- [X] ULTRATHINK. Follow the instructions in specs/market_sentinel_boilerplate.md;
- [X] Continue.
- [X] /commit
- [X] ULTRATHINK. Follow the instructions in specs/market_sentinel_boilerplate.md;
- [X] Get rid of ET on the tool bar.
- [X] The volume bars should match the color of the candlesticks.  They are always green.  Some should be red.
- [X] Create a legend in the top-lefthand corner of the candlestick chart.
- [X] THINK HARD.  check time and sales.  It should be continously updating and the trades scrolling down.
- [X] THINK HARD. Follow the instructions in specs/market_sentinel_boilerplate.md;
- [ ] ULTRATHINK. Follow the instructions in specs/market_sentinel_boilerplate.md;

## Index.html

### Config

- [X] claude --dangerously-skip-permissions
- [-] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] run git switch -c feature/market_sentinel_chart
- [X] /clear
- [X] /read...

### Spec File

- [-] ULTRATHINK. Implement the instructions in specs/market_sentinel_chart.md
All code is on branch feature/market_sentinel_chart and ready for the API compatibility updates in Phase 2!
- [X] Nudge.
- [X] /commit
- [X] Publish the branch
- [ ] run git switch master; merge current branch and verify that all changes are merged.
- [ ] Sync changes


## Update Momentum Alerts

- [X] ULTRATHINK. Update code/momentum_alerts.py: When collecting symbols from the two scripts that are called, leave the execution of the script the same but only keep the first five symbols from each list that do not end in a "W". When saving the symbols, have separate boolean fields indicating what list from which they came.  Also, have an addition boolean field called "Oracle" to use for the symbols that came from the ./data dir.  Add all the of these new fields to the momentum alerts.
- [X] ULTRATHINK. Update code/momentum_alerts.py: After each time collecting symbols, call
```bash
code/alpaca_screener.py --symbols {symbol list} --export-csv symbol_list.csv
```
Then read historical_data/YYYY-MM-DD/scanner/symbol_list.csv and obtain the fields volume_surge_detected,volume_surge_ratio per symbol.
Include these new fields in the momentum alerts in a new "Volume" section and call them "Surge Detected", and "Surge Ratio".
- [X] THINK HARD.  Update code/momentum_alerts.py: There is already "Momentum" and "Momentum Short" fields that use values from a config file to calculate percent gain per minute.  Add a new value to the config file for squeeze duration = 10. Add a new field "Squeezing" after "Momentum Short".  Display a green checkmark if the percent gain per minute is >= 1%. Pick another emoji if false.
- [X] ULTRATHINK.  Create an atom in ./atoms/api to use polygon.io to retrieve shares_outstanding, float_shares, market_cap.  If that fails, use Yahoo Finance to retrieve the information. Update code/momentum_alerts.py: When collecting the symbols, save those fields. Create a new section in the momentum alerts above the timestamp at the end of the alert and include the fields "Shares Outstanding", "Float Shares", and "Market Cap". Pick an appropriate name for the section.
- [X] THINK HARD.  Update code/momentum_alerts.py: Update
```bash
code/alpaca_screener.py --symbols {symbol list} --export-csv symbol_list.csv
```
to
```bash
code/alpaca_screener.py --symbols {symbol list} --surge-days 50  --volume-surge 5.0 --export-csv symbol_list.csv
```
- [X] ULTRATHINK. Research Alpaca Trading API v2 and see if there is a way to obtain float rotation for stocks.
- [X] ULTRATHINK. The float rotation is being calculated on the last 30 minutes of data.  Instead use code/alpaca.py to retrieve one hour candlesticks starting at 0400 ET and sum the volume from those.

## Update Top Gainers

- [X] Update code/momentum_alerts.py: Update '"--max-price", "40.00"' to use 100.00.  Update any related docs.


## EMA 21, 34, 50

- [X] ULTRATHINK. Update code/alpaca.py --plot to also plot EMA 21, 34, 50.  It already plots EMA 9.

## Momentum Alerts - % Gain

- [X] ULTRATHINK.  Review code/momentum_alerts.py:  When retrieving the symbols from market open top gainers, also retrieve the market open stock price for each symbol.  Add the market open price to the momentum alert.  Also, add percent gain since market open to the momentum alert.
- [X] ULTRATHINK.  For symbols that do not have market open price available, use code/alpaca.py.

## Publish Momentum Alerts

- [X] THINK HARD.  Update .telegram_users.csv: add a new field momentum_alerts=false. Set Bruce and "Lucille Wang" to true.
- [X] THINK HARD.  Update code/momentum_alerts.py: Instead of sending momentum alerts to Bruce, send them to all users in .telegram_users.csv where momentum_alerts=true.


## Top Gainers

- [X] top gainers as an atom
- [X] ULTRATHINK. Update molecules/telegram_polling.py:  Add a new command "Top Gainers" case insensitive.  Call code/alpaca.py --top-gainers --limit 40.  Add the new command to the /help documentation.
- [X] Update the atom to only display the top gainers.  I do not want to see the top losers.


## Symbol Polling Update II

- [X] THINK HARD. Update atoms/api/symbol_polling.py:  Add --save arg. When --save is present, save the stock data on a per symbol basis in a csv file
using the path historical_data/{symbol}/YYYY-MM-DD.csv.  You choose if it is best to save the data continuously or periodically.
- [X] Update the .md file and the file docstring.


## Symbol Polling Update

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] run git switch -c feature/symbol_polling_update
- [X] /clear
- [X] /read...

### Spec File

- [X] THINK HARD. Research Alpaca Trading API version 2: Is there a way to obtain Level 2 data for "time and sales"?
- [X] Review atoms/api/symbol_polling.py: Does it use the WebSocket subscription?
- [X] THINK HARD.  Please updat the script to use the WebSocket subscription.
- [X] Did you update the file docstring?  Make sure the README file is mentioned in the file docstring.
- [X] THINK HARD.  Time is always to be ET.  Not local.  Also, retrieve the time of the sale from the WebSocket if possible.  Is the date available?
- [X] THINK HARD. Add --symbol arg.  When parsing the arg convert to uppercase.  If --symbol, do not read symbols from the files.
- [X] What is the units for volume? 1 or 1000.
- [-] ULTRATHINK. Implement the instructions in specs/major_resistance.md.
- [X] Nudge.
- [-] Please put usage examples in the file docstring.
- [X] /commit
- [X] Publish the branch
- [X] run git switch master; merge current branch and verify that all changes are merged.
- [X] Sync changes


## Momentum Update

- [X] THINK HARD. Update code/momentum_alerts.py: The code is currently calling market open top gainers repeatedly.  Update the code to call market open top gainers starting at 9:40 ET instead of 10:00 ET.


## Chart Major Resistance

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] run git switch -c feature/major_resistance
- [X] /clear
- [X] /read...

### Spec File

- [X] ULTRATHINK. Implement the instructions in specs/major_resistance.md.
- [X] Update to only plot the highest major resistance.
- [-] Please put usage examples in the file docstring.
- [X] /commit
- [X] Publish the branch
- [X] run git switch master; merge current branch and verify that all changes are merged.
- [X] Sync changes


## Orb.py EOD Update

- [X] Where is code/orb.py getting the list of stock symbols to plot?
- [X] ULTRATHINK. Update code/orb.py.  Replace the current process collecting the stock symbols.  Update that process to collect the unique stock symbols from historical_data/YYYY-MM-DD/superduper_alerts_sent/bullish/green/superduper_alert_{symbol}_*.json and historical_data/YYYY-MM-DD/momentum_alerts_sent/bullish/alert_{symbol}_*.json.


## Plot Momentum Alerts Update

- [X] ULTRATHINK.  Update code/alpaca.py --plot: It occasionally fails with "Insufficient data for MACD calculation...". Fix the code so that the plot gets generated.  You may leave the MACD section empty if there is an error.


## EOD Charts

- [X] EOD charts are missing Momentum Alerts.


## Plot Momentum Alerts Update

- [X] ULTRATHINK.  code/alpaca.py --plot currently generates stock charts to include sent superduper alerts
and momentum alerts.  Momentum alerts are to use a different color than the sent superduper alerts.  Both colors are green.
Please fix. Show me the chart for ONMD for 2025-10-06.

## Overnight Gainers Integration

- [X] ULTRATHINK. Update code/momentum_alerts.py: The code is currently calling market open top gainers repeatedly.  Update the code to call volume surge scanner once at startup.  Use the args specified in molecules/telegram_polling.py /help command.  Add the stock symbols to the other stock symbols already being collected.


## Overnight Gainers

- [X] ULTRATHINK. Update molecules/telegram_polling.py: Update
```python
            cmd = [
                python_path, premarket_script,
                '--exchanges', 'NASDAQ', 'AMEX',
                '--max-symbols', '7000',
                '--min-price', '0.75',
                '--max-price', '40.00',
                '--min-volume', '50000',
                '--top-gainers', '20',
                '--export-csv', 'top_gainers_nasdaq_amex.csv',
                '--verbose'
            ]
```
min-volume 50000 -> 250000
top-gainers 20 -> 40
Also, update the /help command to match.


## Minimum Volume Increase

- [X] Raise minimum volume on market open top gainers


## Top Gainers Update

- [X] ULTRATHINK.  Update code/market_open_top_gainers.py: Before saving the CSV file or during, round all the float values to two decimal places.

## Momentum Alerts Filter Volume

- [X] ULTRATHINK. Update code/momentum_alerts.py to filter out low volume sent momentum alters.
Use code/momentum_alerts_config.py: volume_low_threshold


## Plot Momentum Alerts Also

- [X] ULTRATHINK.  code/alpaca.py --plot currently generates stock charts to include sent superduper alerts.
Please also plot momentum alerts but use a different color than the sent superduper alerts.
The file path format is as follows:
historical_data/YYYY-MM-DD/momentum_alerts_sent/bullish/alert_{symbol}_YYYY-MM-DD_*.json

## Plot Called Twice

- [X] THINK HARD.  Update molecules/telegram_polling.py:  When the 'plot' command gets ran on Telegram, I receive two plots for the same symbol.  There should only be one.

## Test Symbol Polling

- [X] Run symbol_polling.py after 10 AM ET to see it run.
- [X] THINK HARD.  Update atoms/api/symbol_polling.py: The configuration should be available.
python atoms/api/symbol_polling.py
Warning: Could not load config file ("Configuration not found for alpaca/bruce/paper: 'bruce'"), falling back to environment variables
Error initializing Alpaca client: ('Key ID must be given to access Alpaca trade API', ' (env: APCA_API_KEY_ID)')
- [X] THINK HARD.  Research Alpaca Trading API v2.  Then, fix
Loaded 12 symbols from /home/wilsonb/dl/github.com/z223i/alpaca/data/20251006.csv
Loaded 20 symbols from /home/wilsonb/dl/github.com/z223i/alpaca/historical_data/2025-10-03/market/gainers_nasdaq_amex.csv
Error polling HUMA: 'REST' object has no attribute 'get_last_trade'
Error polling FEMY: 'REST' object has no attribute 'get_last_trade'
Error polling CRML: 'REST' object has no attribute 'get_last_trade'
Error polling NIOBW: 'REST' object has no attribute 'get_last_trade'
Error polling STAK: 'REST' object has no attribute 'get_last_trade'
Error polling ACIU: 'REST' object has no attribute 'get_last_trade'
- [X] THINK HARD.  python atoms/api/symbol_polling.py --verbose should only load today's data.
It should be waiting to the historical data to appear.
Output:
Loaded 12 symbols from /home/wilsonb/dl/github.com/z223i/alpaca/data/20251006.csv
Loaded 20 symbols from /home/wilsonb/dl/github.com/z223i/alpaca/historical_data/2025-10-03/market/gainers_nasdaq_amex.csv




## Symbol Polling

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] run git switch -c feature/symbol_polling
- [X] /clear
- [X] /read...

### Spec File

- [X] ULTRATHINK. Implement the instructions in specs/symbol_polling.md.
- [X] Please put usage examples in the file docstring.
- [X] /commit
- [X] Publish the branch
- [X] run git switch master; merge current branch and verify that all changes are merged.
- [X] Sync changes


## Momentum Normalized

- [X] ULTRATHINK.  Update code/momentum_alerts.py: Update the Telegram message to display Momentum and Momentum Short as normalized per minute.  Show example.


## Send Top Gainers Frequency Update

- [X] THINK HARD.  Update code/momentum_alerts.py: The top gainers are currently generated once per hour for four hours.  Update the generate twice per hour for five hours.


## Send Top Gainers Update

- [X] THINK HARD.  Update code/momentum_alerts.py: After each time generating top gainers, the file name is sent to 'bruce' using atoms/telegram/telegram_post.py.  Update to send the file contents.


## Send Top Gainers

- [X] THINK HARD.  Update code/momentum_alerts.py: After each time generating top gainers, send the file to 'bruce' using atoms/telegram/telegram_post.py.


## Super Duper Alert Threshold

- [X] THINK HARD. Update atoms/alerts/config.py: trend_analysis_timeframe_minutes=20 -> 17.
- [X] 0.75 -> 0.65 in config.py

## Automatic trading

- [X] THINK HARD. Create code/configure_alpaca.py: it is to accepts args to modify code/alpaca_config.py.
Here are the args to control what account-name and account get modified:
  --account-name ACCOUNT_NAME
                        Account name to use (default: Bruce)
  --account ACCOUNT     Account environment to use: paper, live, cash (default: paper)
These are the fields that can be modified:
  auto_trade="no",
  auto_amount=100,
  trailing_percent=10.0,
  take_profit_percent=12.0,
  max_trades_per_day=1

- [X] ULTRATHINK.  Update molecules/telegram_polling.py: Only Bruce can send the command 'configure alpaca' case insensitive. It can accept the same args as code/configure_alpaca.py. It is to run code/configure_alpaca.py with the matching args. Add command to help but only for Bruce.
- [X] THINK HARD. Review and update the polling script.  Telegram changes double hyphens. Accept single hyphens for the cli args.

## Dial Down Momentum Alerts

- [X] Update Momentum thresholds 0.70 -> 0.60.


## Momentum Alerts Updates III

- [X] ULTRATHINK.  Update code/momentum_alerts.py: Momentum and Momentum Short have the same value in the momentum alert.  I doubt that that is correct.  The Momentum should be based on 20-minutes and Momentum Short based on 5-minutes.  Suggest a fix.
- [fail] THINK HARD. Update code/momentum_alerts.py: Momentum and Momentum Short are to be normalized within there 20 and 5 minute time frames and then checked.
- [X] THINK HARD. Update code/momentum_alerts.py: Momentum = Momentum / 20; Momentum Short /= 5; Then check them against the threshold.
- [X] THINK HARD. Create code/momentum_alerts_config.py: Move the constants 20 and 5 to this file in a dataclass; Find the Momentum threshold and create three constants Momentum Long, Momentum, and Momentum Short with that value. Use the dataclass in code/momentum_alerts.py.  Momentum long is not yet used but will be.
- [X] THINK HARD. Update code/momentum_alerts.py: Review the Momentum and Momentum Short calculations.  They are to be based on minutes from the new config file.  I think they are based on the number of candlesticks.  If a stock is halted, there is not a candlestick.  Use the time please.
- [X] THINK HARD. Using the knowledge that a halted stock is missing data, update the is halted logic.
- [X] THINK HARD. Update code/momentum_alerts.py: When sending the alerts on Telegram, do not mark them as urgent.
- [X] THINK HARD. Update code/momentum_alerts.py: Use the stock volume and add a volume field to the momentum alert. Greater than 80,000 = green; less then 60,000 is a red light icon; otherwise yellow light icon.
- [X] THINK HARD. Update code/momentum_alerts.py: The stock VWAP is being used.  It is to come from the stock data. No calculation required.
- [X] THINK HARD. Update code/momentum_alerts.py: When collecting stock symbols, do not save warrants.  The end in a 'W'. Research stock warrant naming practices.
- [X] THINK HARD. Update code/momentum_alerts.py: Do not include warrant stocks.


## Orb Alerts Check If Halted Stocks Being Filtered.

- [X] ULTRATHINK.  Review code/orb_alerts.py: It has code to check if a stock is halted but, it the code being used or is it dead code?


## Momentum Alerts Updates II

- [X] THINK HARD.  Update code/momentum_alerts.py: Currently the scripts is getting a list of stocks from ./historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv.  Add to that list the stocks in data/{YYYMMDD}.csv if it exists and keep only unique symbols.


## Momentum Alerts Updates

- [X] ULTRATHINK.  Update code/momentum_alerts.py:
1) Update momentum and momentum short fields in the alert to reflect the appropriate signal light icon as defined in atoms/alerts/config.py;
2) Mirror stock is halted logic in code/orb_alerts.py to create a Python script in ./atoms/api;
3) Use the new ./atoms/api script to update the alert in code/momentum_alerts.py;
3) After creating a momentum alert, save it to historical_data/{YYYY-MM-DD}/momentum_alerts/bullish/alert_{symbol}_{YYYY-MM-DD}_{hhmmss}.json
4) After sending a momentum alert, save it to historical_data/{YYYY-MM-DD}/momentum_alerts_sent/bullish/alert_{symbol}_{YYYY-MM-DD}_{hhmmss}.json


## Momentum Alerts

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] git switch -c feature/momentum_alert
- [X] /clear
- [X] /read...

### Spec File

- [X] VERY IMPORTANT. ULTRATHINK. Implement the instructions in specs/momentum_alert.md.
- [X] Nudge
- [X] /commit
- [X] Publish the branch
- [X] git switch master; merge current branch and verify that all changes are merged.
- [X] Sync changes


## Market Open Top Gainers

- [X] VERY IMPORTANT. THINK HARD.  Update molecules/telegram_polling.py: Any user can send the command 'market open top gainers' case insensitive. Respond to the command by running the Python script below.  The last line of the output will contain a filename of the form ./historical_data/{YYYY-MM-DD}/market/gainers_*.csv. Return the file contents to the user that sent the command. Add this to the /help command.  Use a timeout of 15 minutes.  This script takes a while to run.
```bash
python code/market_open_top_gainers.py  --exchanges NASDAQ AMEX  --max-symbols 7000  --min-price 0.75  --max-price 40.00  --min-volume 50000 --top-gainers 20 --export-csv gainers_nasdaq_amex.csv --verbose
```


## Market Open Top Gainer

- [X] VERY IMPORTANT. THINK HARD. Mirror code/premarket_top_gainers.py. Create code/market_open_top_gainers.py: Use 1-minute candles from the previous market open to no more than market close to calculate top gainers.  This script will be during market open and sometimes after market close.


## Premarket Top Gainers

- [X] VERY IMPORTANT. THINK HARD.  Update molecules/telegram_polling.py: Any user can send the command 'premarket top gainers' case insensitive. Respond to the command by running the Python script below.  The last line of the output will contain a filename of the form ./historical_data/{YYYY-MM-DD}/premarket/top_gainers_*.csv. Return the file contents to the user that sent the command. Add this to the /help command.  Use a timeout of 10 minutes.  This script takes a while to run.
```bash
python code/premarket_top_gainers.py  --exchanges NASDAQ AMEX  --max-symbols 7000  --min-price 0.75  --max-price 40.00  --min-volume 50000 --top-gainers 20 --export-csv top_gainers_nasdaq_amex.csv --verbose
```
- [X] /commit
- [X] Publish the branch
- [X] git switch master; merge current branch and verify that all changes are merged.
- [X] Sync changes


## Pre-market Top Gainer

- [X] VERY IMPORTANT. THINK HARD. Mirror code/alpaca_screener.py. Create code/premarket_top_gainers.py: Use 5-minute candles from the previous market close to determine top-gainers.  This must work during pre-market hours. Think about: you could collect data for the last 7 days and then use only the data from since the last market close.

## Run Volume Profile in Bulk

- [X] VERY IMPORTANT. THINK HARD. Create code/volume_profile_bulk.py: Read data/YYYYMMDD.csv for today's date;
Add a new column "POC" after the 'Signal' column;
For each stock symbol in that file run code/volume_profile.py --symbol [symbol] --days 1 --timeframe 5Min --time-per-profile DAY --chart;
From the .JSON output, read "summary":"avg_poc" and put it in the 'POC' field; and
Save the updated data as data/YYYYMMDD_POC.csv.


## Update Volume Profile

- [X] VERY IMPORTANT: Update code/volume_profile.py: Automatically save the ouput in a .JSON file using the same dir and naming convention as the charts. But with .JSON of course.


## Convert ToS to Python

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] git switch -c feature/tos_to_python
- [X] /clear
- [X] /read...

### Spec File

- [X] VERY IMPORTANT. Think hard. Implement the instructions in specs/tos_to_python.md.
- [X] Nudge
- [X] /commit
- [X] Publish the branch
- [X] git switch master; merge current branch and verify that all changes are merged.
- [X] Sync changes



## Webull Hot Keys

- [X] Set up, shortcut keys!


## /newbie

- [X] Update molecules/telegram_polling.py: Add a new command /newbie.  This command is to display:
"**Welcome!**

Daytrading is frequently very fast.  In and out of a stock in ten minutes is common. I once made almost 9% in less than 90 seconds.

**Hot Keys**

Hot keys are a way to buy and sell quickly.  These are supported by Lightning Trader, Charles Schwab (Tos), and Webull.  See https://www.youtube.com/watch?v=9hgpZMvCY08 for setup instructions.

**Scanner Commands**

'volume surge' and 'top gainers' function only after market open.  If before market open, the prior trading day results will be given.

**Best wishes!**"

## Top Gainers

- [X] VERY IMPORTANT:  Remember everything you did for 'volume surge'. And apply that to update molecules/telegram_polling.py: Any user can send the command 'top gainers' case insensitive. Respond to the command by running the Python script below.  The last line of the output will contain a filename of the form ./historical_data/YYYY-MM-DD/scanner/top_gainers_*.csv. Return the file contents to the user that sent the command. Add this to the /help command.  Use a timeout of 10 minutes.  This script takes a while to run.
```bash
python code/alpaca_screener.py  --exchanges NASDAQ AMEX  --max-symbols 7000  --min-price 0.75  --max-price 40.00  --min-volume 50000 --top-gainers 20 --export-csv top_gainers_nasdaq_amex.csv --verbose
```


## Relative Volume Surge

- [X] VERY IMPORTANT:  Update molecules/telegram_polling.py: Any user can send the command 'volume surge' case insensitive. Respond to the command by running the Python script below.  The last line of the output will contain a filename of the form ./historical_data/YYYY-MM-DD/scanner/relative_volume_*.csv. Return the file contents to the user that sent the command. Add this to the /help command.  Use a timeout of 10 minutes.  This script takes a while to run.
```bash
python code/alpaca_screener.py  --exchanges NASDAQ AMEX  --max-symbols 7000  --min-price 0.75  --max-price 40.00  --min-volume 50000 --min-percent-change 5.0  --surge-days 50  --volume-surge 5.0  --export-csv relative_volume_nasdaq_amex.csv  --verbose
```


## Halted Stocks Update

NOTE: This was done because false positive were being generated.
This give a person a chance to still trade false positive and even positives.
- [X] Stop filtering halted stocks at orb alert level.
- [X] Pass 'Halted' field through to trade_generator.py.
- [X] Update trade_generator.py to filter stock that have been halted.


## Halted Stocks

- [X] VERY IMPORTANT: Review code/orb_alert.py:  How is it filtering stocks?
- [X] VERY IMPORTANT: Update code/orb_alert.py:  Before the VWAP filter add a stock halted filter.  When a stock halts there is missing timestamps in the stock date.  Stock data is per minute.  So if there is more than a minute between timestamps, the stock halted.  Here is an example 2025-09-16 09:36:00 and 2025-09-16 09:42:00.


## Signal

- [X] VERY IMPORTANT: Create code/signal.py: 1) Read data/YYYYMMDD.csv using current date, 2) For each symbol in data/YYYYMMDD.csv, compare the signal field to the most recent 'low' value in historical_data/YYYY-MM-DD/market_data/[symbol]_YYYYMMDD_*.csv; where the historical data file is the most recent (there are at least two per symbol), A) If the 'low' value is at 95% or greater of the signal value, store the symbol, signal, and low value to a list. 3) Finally, print the list to std in csv format.
Example Files:  data/20250917.csv; historical_data/2025-09-17/market_data/ATCH_20250917_*.csv
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.
- [X] VERY IMPORTANT:  Update molecules/telegram_polling.py: Any user can send the command 'signal' case insensitive. Respond to the command by running code/oracle_signal.py.  Return the output to the user that sent the command. Add this to the /help command.



## Negative Long Delta

- [X] Currently getting an error then autofix for a negative long delta.  This should not be an error.  Investigate.  Changed * 1.10 to + 0.02. I would have radically altered the alerts if left negative.  That value is used for penetration range calculation.  "Resistance" is major resistance and is very important.  I just set it to $0.02 above signal.


## Bam

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [O] git switch -c feature/bam
- [X] /clear
- [X] /read...

### Spec File

- [X] VERY IMPORTANT. Think hard. Implement the instructions in specs/bam.md.
- [X] Nudge
- [X] /commit
- [O] Publish the branch
- [O] git switch master; merge current branch and verify that all changes are merged.
- [X] Sync changes


## Do not send on yellow or red momentum short

- [X] Do not send on yellow or red momentum short

## Major Oops

- [X] Filter sent superduper alerts on VWAP.


## Scanner

- [X] VERY IMPORTANT, think hard: Review the following CSV info. It is from a stock screen filter on my computer.  Review Alpaca Trading API version 2 for stock screening. Create specs/alpaca_screener.md with all the info necessary to write a stock screener. also add the ability to screen stocks for which the volume is up N times versus the last M days.
```text
Metric,From,To,Unit
Price,0.75,,USD
Volume,1000000,,Shares
% Change,,,Percent
Float,,,Shares
Trades,,,Count
Market Capitalization,,,USD
$ Volume,,,USD
Day Range,,,USD
EPS,,,USD
P/E Ratio,,,Ratio
SMA,,,Days
Avg Daily Volume,,,Shares (5 Days)
Avg Daily Range,,,USD (5 Days)
```

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] git switch -c feature/alpaca-scanner
- [X] /clear
- [X] /read...

### Spec File

- [X] VERY IMPORTANT. Think hard. Implement the instructions in specs/alpaca_screener.md.
- [X] Nudge
- [X] /commit
- [X] Publish the branch
- [X] git switch master; merge current branch and verify that all changes are merged.
- [X] Sync changes


# Basic screening
  python code/alpaca_screener.py --min-price 0.75 --min-volume 1000000

  # Volume surge detection
  python code/alpaca_screener.py --volume-surge 2.0 --surge-days 5

  # Export results
  python code/alpaca_screener.py --min-volume 500000 --export-csv results.csv --export-json results.json



## Test Monitor Positions

- [ ] Do this in the paper account
- [ ] Test on Sat. if possible with AAPL
- [ ] Update the monitor to send a Telegram post to the account holder every 15 minutes
- [ ] Buy AAPL
- [ ] Verify the monitor works for this simple case
- [ ] Expand testing to include closing positions

## Trades Exponential Decay

- [ ] VERY IMPORTANT. Update atoms/alerts/trade_generator.py: Do an exponential decay on trades.

## Stock Review

- [ ] 2025-08-26 Did $HKPD take off very early.  Why not detected.



Work M/r into alters and not yet trades.




## EOD

- [X] IMPORTANT.  What does code/alerts_watch.py do at the end of the day?
- [X] VERY IMPORTANT.  How is atoms/alerts/trade_generator.py keeping track of trades per account-name and account? If so, use that information to create a list of unique account-name/account combinations.  Then get the alpaca.py --account-name <account name> --account <account> --PNL for each unique combination.  Also, send a Telegram post to Bruce for the PNL for each unique combination.  Both of those can be put inside the same loop.

## BLIND FLIGHT

- [X] VERY IMPORTANT. Do a root cause analysis.  Review sent superduper alerts. Why does the "MACD Technical Analysis" "Status" show "No live data available"? There should be data availble. It should be retrieved via the Alpaca API that is instantiated with the account-name Janice and account paper. !Current SIP data is not allowed.!
- [X] IMPORTANT. Update code/orb_alerts.py: It currently saves historically data every ten minutes. Change that to every minute.  Also, once the data is saved, delete the previous data for the symbol.
- [X] VERY IMPORTANT. The historical data files do have current data. Use that data to perform the "MACD Technical Analysis".
- [X] VERY IMPORTANT. Use the same data file to conduct the "Momentum Short" analysis in the "Trend Analysis" section.

## Trades

- [X] VERY IMPORTANT. Update atoms/alerts/trade_generator.py: It is currently not using --account-name and --account.  Does it know those values?  Update the alpaca.py call with those values.  Review code/alpaca.py parsing mechanism to determine the order of the args. Example: --account-name Bruce --account paper.
- [X] Why 15 trades?

- [ ] My trades place two orders.  Provide feedback in the Telegram post for both orders.


## Max Results

Hardcoded root

    {
      "parameters": {
        "take_profit_pct": 12.5,
        "trailing_stop_pct": 12.5
      },
      "metrics": {
        "total_return": 481469.8460905292,
        "win_rate": 74.50980392156863,
        "avg_win": 15.840112785812831,
        "avg_loss": 9.60207755819928,
        "profit_factor": 4.822067710087149,
        "sharpe_ratio": 12.625597465750946,
        "max_drawdown": 68.02693468719997,
        "total_trades": 102,
        "winning_trades": 76,
        "losing_trades": 26,
        "mean_return": 9.354848580476416,
        "std_return": 11.76210613259913
      },
      "trade_count": 102,
      "valid_trades": 102
    },


## maximize_profit.py

- [X] Review code/maximize_profit.py: Is it printing all charts for trades with losses?  It is supposed to do that.
- [X] Verify

## Maximize

- [X] Run code/backtesting.py first.  Then
- [X] Run maximize_profit.py with 8/21 data

## Major Resistance

- [X] Update Claude to set m/r field to $0.00
- [ ] Review code/alpaca.py: Is there trade monitoring software. Is it used the various buy orders?
- [ ] Have it sell just below m/r?

## Broadcast Message

- [ ] Add a broadcast message to telegram_polling.

## Holy Cow!

Why so many trades today?  That must have been during testing.  They were great though!

## Trading 2025-08-21

- [X] Trade Live with $10 and limit of 3 trades


## Massive maximize_profit changes

- [X] git tag -a V1.0.7 -m "Maximize profit filter late day trades."

## Stuff

- [X] Change this back.  open -> o, etc.  How did this even get back into the code??
               df_data.append({
                    'timestamp': timestamp_et,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                    'symbol': self.symbol
                })
- [X] Remove changes to maximize profit
- [X] remove all log file usage from maximize_profit.py - JSON files only
- [X] Remove fake MACD


## MISC

- [X] Review README_MACD.md and also why am I getting Blind Flight in my alerts?
- [X] Update   Alert Timeframe: 20 minutes
- [X] Alert Green Threshold: 0.70




## Exit Strategy

- [X] VERY IMPORTANT. THINK HARD. I am collecting historical data on sent superduper alerts.  I plan to backtest my alerts under different         │
│   selling parameters. The ultimate goal is to maximize profit. My entrance strategy is based on the alerts.  I need to determine the optimal   │
│   exit strategy. When buying, I usually set a stop loss. I can set a take profit percent. I can set a trailing loss percent. I can let a MACD  │
│   criteria trigger the sell.  If the stock position does not sell by 15:40 ET, it is automatically closed.  The question is how to conduct     │
│   the statistic analysis to maximize profit while considering that some of the exit strategies overlap.  That is a take profit percent may or  │
│   may not prevent a MACD criteria from even being tested.  I am not concerned about when to liquidate at the end of the day.  And, what type   │
│   of statistical testing.                                                                                                                      │
╰───────────────────────────


### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] git switch -c feature/max-profits
- [X] /clear
- [X] /read...

### Spec File

- [X] VERY IMPORTANT. Think hard. Implement the instructions in specs/exit-strategy-testing.md to create code/maximize_profit.py.
- [X] Nudge
- [X] /commit
- [X] Publish the branch
- [X] git switch master; merge current branch
- [X] Sync changes


## EOD

- [ ] No trades after 15:30 ET.
- [X] Close all positions
- [X] Cancel all orders
- [X] Today's PNL - needs to be added to watchdog.
- [ ] TEST: Close all positions at the end of the day.  Close all orders. Calculate PNL




PCA analysis on exit strategy


## Plots

- [X] Send plots on Telegram.

## Oops

- [X] Shut off trades in the back testing. IT placed a bunch of orders during the middle of the night.  I was up and testing in the middle of the night.
- [X] Review ./atoms/alerts. Find the code that executes trades using alpaca.py.
- [X] Investigate why those trades happened. Use real time of day for executing trades.


## Plots

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] git switch -c feature/telegram-plots
- [X] /clear
- [X] /read...

### Spec File

- [X] VERY IMPORTANT. Think hard. Implement the instructions in specs/telegram-plots.md
- [X] Nudge
- [X] /commit
- [X] Update molecules/telegram_polling.py: When it receives a trigger word, "plot", and then -plot -symbol <symbol>, extract the plot location    │
│   from the output of the alpaca.py call. Here is an example "Chart generated successfully: plots/20250818/SNGX_chart.png". Then send the       │
│   image to the user using atoms/telegram/send_image.py. This command can be used by any Telegram user.
- [X] Publish the branch
- [X] git switch master; merge current branch
- [X] Sync changes


## Trades

- [X] Is VERB doubled again? Fix it.
- [O] Try to get trades into the analysis. No. Have PCA figure out what filters to apply.  Specifically TOD, Time of Day.

## errata

- [X] VERY IMPORTANT: the buy type for $BSLK was not correct. The trade execution responded with no but without detail.
- [X] Reduce the price momentum length. Reduce from 0.75 to 0.65 in atoms/alerts/config.py. Centralize the location of the price momentum in atoms/alerts/config.py to 20.

## Backtesting Update

- [X] VERY IMPORTANT. THINK HARD. We have been having difficults getting the processes in code/backtesting.py coordinated as to which directories to output their data. So, we are just going to keep them the same but with a twist.
Update code/backtesting.py to copy atoms/alerts/config_current_run.py to atoms/alerts/config.py at startup.  It is to copy atoms/alerts/config_orig.py to atoms/alerts/config.py at shutdown to include responding to a <CTRL+C>.
Every run will write to the same output directory.  At the end of the run use a Bash mv command to move the dir in the now updated config file to a dynamically named dir which is already in the code.
- [ ] Run verb by itself and see if 640 or 320.  Are the charts showing all three symbols? 2043 symbol pie chart shows three.

## Backtesting

- [X] Centralize the starting point for the historical_data dir in config.py. 2025-08-13 17:00
- [X] Create ./data/backtesting/symbols.json; populate with symbol/date/active='no' combos BSLK,2025-08-13; ATNF,2025-08-12; and STAI,2025-07-29. Then on 2025-08-04 the symbols OPEN, BTAI, and VERB.


- [X] Add --date arg to code/orb_alerts.py. Default to todays date.
- [X] Add --date arg to code/orb_alerts_monitor.py. Default to todays date.
- [X] VERY IMPORTANT. Centralize the starting point for the "logs" dir in atoms/alerts/config.py. It should be currently ./logs. Mirror historical root in atoms/alerts/config.py. Update the four scripts, code/orb_alerts.py, code/orb_alerts_monitor.py, code/orb_alerts_monitor_superduper.py, and code/orb_alerts_trade_stocks.py accordingly. For any script that does not use the logs dir, leave it unchanged.
- [X] VERY IMPORTANT. Centralize the starting point for the "data" dir in atoms/alerts/config.py. It should be currently ./data. Mirror historical root in atoms/alerts/config.py. Update code/orb_alerts.py.
- [X] VERY IMPORTANT. Centralize the starting point for the "plots" dir in atoms/alerts/config.py. It should be currently ./plots. Mirror historical root in atoms/alerts/config.py. Update code/alpaca.py.



- [X] VERY IMPORTANT. This is only a question.  Leave the code unchanged. How to create a data pipeline for code/orb_alerts.py? Perhaps read all the data for the symbol and date using pre-existing Alpaca scripts in this repo; then pipe in the data iteratively adding a minute (candlestick) at a time to replicate real-time processing.
- [X] Would you create a new CLI arg for alpaca.py or perhaps create a new script?
- [X] Create the new script.
- [X] /commit
- [O] Fall back commit, 2c28752b4c78e284e9a204d1d3c3f9588b87d0b, to recreate!!!

- [X] Are there any hardcode directory paths in:
code/orb_pipeline_simulator.py
code/orb_alerts_monitor.py
code/orb_alerts_monitor_superduper.py
code/orb_alerts_trade_stocks.py
- [X] Nudge: update the path

--plot --symbol and post to Telegram
Parametric testing on stop lost?


### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] git switch -c feature/backtesting
- [X] /clear
- [X] /read...

### Spec File

- [X] VERY IMPORTANT. Think hard. Implement the instructions in specs/backtesting.md
- [O] Nudge (None)
- [X] /commit
- [X] Publish the branch
- [X] git switch master; merge current branch
- [X] Sync changes


## MACD Analysis

- [X] Add MACD analysis to superduper alert.  Filter red emojis


## Monitor Positions

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] git switch -c feature/closing-trades
- [X] /clear
- [X] /read...

### Spec File

- [X] Important. Think hard. Implement the instructions in specs/closing-trades.md
- [X] If a position is liquidated, use Telegram to notify the account holder.  Use Bruce for Primary.
- [X] Also, notify the user if the liquidation failed.
- [X] Switch the default account from Primary to Bruce.  Get rid of the mapping.  Get rid of the Primary account in the config file.  All Telegram users will map directly to the account names in the config file.
- [X] Automatically start monitor positions when the first trade of the day is made. Does it need to be async?
- [X] /commit
- [X] Publish the branch
- [X] git switch master; merge current branch
- [X] Sync changes

## Time of Day

- [X] Add time of day to trend analysis.

## Alpaca CLI arg --plot

- [X] Require --symbol
- [X] Use atoms/display/generate_chart_from_df.py

## Trailing Sell

- [X] Change from 5 to 7.5.
- [X] Can I add a take profit at N perent? Yes.  Update.

## ORB Alerts Watchdog

- [X] Create a watchdog program in ./code to ensure code/orb_alerts.py is running.  If not "python3 code/orb_alerts.py --verbose". Can you have the watchdog launch orb_alerts on start-up and display its output to the console.  That would be better.


## Filter Red Candle

### Config

- [X] /clear
- [X] /read...

### Spec File

- [X] Important. Think hard. Implement the instructions in specs/red-candle.md
- [X] Nudge
- [X] /commit


## PNL Integration

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit
- [X] Sync changes
- [X] /clear
- [X] git switch -c feature/PNL-integration
- [X] /read...

### Spec File

- [X] Important. Think hard. Implement the instructions in specs/PNL-integration.md
- [X] Nudge
- [X] /commit
- [X] Publish the branch
- [X] git switch master; merge current branch
- [X] Sync changes

## PNL

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit /clear
- [X] git switch -c feature/PNL
- [X] /read...

### Spec File

- [X] Important. Think hard. Implement the instructions in specs/PNL.md
- [X] I forgot to give you valid Alpaca login info.  See code/alpaca_config.py.  Use the "Primary":"paper" account info for testing.
- [X] /commit
- [X] Publish the branch
- [X] git switch master; merge current branch

## Resistance Pricing

- [X] Think hard.  Update code/orb_alerts.py: The code is currently applying an automatic timeout for resistance inversion. Delete that. I will enter the info or press <Enter>.

## Cancel All Orders

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit /clear /read...
- [ ] git switch -c cancel-all-orders
- [X] /read...

### Spec File
- [X] /commit
- [X] Important. Think hard. Implement the instructions in specs/cancel_all_orders.md
- [X] Nudge.  Do not require --submit.
- [X] /commit
- [ ] Publish the branch
- [ ] git switch master; merge current branch

## No Yellow Alerts

- [X] Important.  Think hard.

## Trade Generator (autopilot) Update

- [X] Important.  Think hard.  Update atoms/alerts/trade_generator.py: There is only one trade to be allowed during my testing.
- [X] Add --submit and verify one trade.
- [X] Change from one trade to five.
- [X] Change five to twelve.

## Autopilot 2

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit /clear /read...
- [X] git switch -c trade-generator-2
- [X] /read...

### Spec File
- [X] /commit
- [X] Important. Think hard. Implement the instructions in specs/trade_generator_2a.md
- [X] Nudge Claude in the correct direction.
- [X] Important. Think hard. Mirror code/orb_alerts_monitor_superduper.py to create code/orb_alerts_trade_stocks.py.  Use the atom just created to monitor historical_data/YYYY-MM-DD/superduper_alerts_sent/bullish/green/
- [X] Important.  Think hard. Review .telegram_users.csv.  Update the notifications to only be sent to telegram user for which the trade was made.

## Update Green Light

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit /clear /read...
- [X] /read...
- [X] Review code/orb_alerts_monitor.py: Is this the file that designates if the trend analysis momentum is green and yellow?
- [X] It is in atoms/alerts/superduper_alert_generator.py.  Update 0.5 to 0.75 as the dividing line between yellow and green
- [X] /commit

## Autopilot

### Config

- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /commit /clear /read...
- [X] git switch -c autopilot
- [X] /read...
- [X] Update code/alpaca_config.py: update "CONFIG": for the accounts paper, live, and cash add the fields auto_trade="no"; auto_amount=10
- [X] Update code/alpaca_config.py: update "CONFIG": Duplicate the "Primary" account and use the names "Bruce" and "Dale".

### Spec File
- [ ] /commit
- [ ] Important. Think hard. Implement the instructions in specs/trade_generator.md

## Telegram Updates

- [X] /commit /clear /read...
- [X] git switch -c telegram
- [X] /exit
- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /read...
- [X] Important. Think hard. Update atoms/telegram/orb_alerts.py the method send_orb_alert at the very end of the method to store superduper alerts actually sent.  Use the directory structure historical_data/2025-08-01/superduper_alerts_sent. There is to be bullish and bearish directories below that. Then have another level of sub directories for yellow and green superduper alerts sent.
- [X] Important. Think hard. Now update code/orb_alerts_summary.py to generate plots for the sent superduper alerts.
- [X] Important. Think hard. Now update code/orb.py: It currently generate plots for superduper alerts; change that to sent superduper alerts and plot the green and yellow ones on the same chart using green and yellow.


## Use config.py file instead of .env variables

- [X] /commit /clear /read...
- [X] See if claude -p uses the API or my subscription.
- [X] git switch use-config-py
- [X] /exit
- [X] claude --dangerously-skip-permissions
- [X] /login
- [X] /read...
- [X] Important.  Think hard.  Implement the instructions specs/use_config_py_vs_env_variables.md.
- [X] /exit

- [ ] Oops! It implemented the code immediately.  Work on that. Use markdown!!! I did use markdown!!!!!
- [ ] /commit
- [ ] git push
- [ ] /clear /read...
- [ ] git switch -c buy-market-trailing-sell
- [ ] Important.  Think hard.  Implement the instructions in the PRD specs/buy_market_with_trailing_sell_prd.md
- [ ] /commit
- [ ] git push
- [ ] switch to master and merge current branch



## Cancel all existing orders



## Symbol.toupper()

- [X] do it in arg parse.

## Buy Market

- [ X] /commit /clear /read...
- [X] Important.  Think hard. Create a PRD, specs/buy_market_with_trailing_sell_prd.md: 1. Search internet for how to place a market buy order for Alpaca trading API version two; 2. Create code/alpace.py CLI arg --buy-market it will have the required --symbol and an optional --submit; 3. Test the new code; 4. Create code/alpace.py CLI arg --buy-market-trailing-sell it will have the required --symbol and an optional --submit; 5. buy-market-trailing-sell method: calls the newly created buy market method; polls the alpaca trading api until the order is filled or canceled; if filled, then call the trailing sell method with the number of shares purchased; 6. you will need to research how to determine if the order was placed; 7. Test the code; 8. Update README_alpaca.md.
- [X] Oops! It implemented the code immediately.  Work on that. Use markdown!!!
- [X] /commit
- [X] git push
- [ ] /clear /read...
- [ ] git switch -c buy-market-trailing-sell
- [ ] Important.  Think hard.  Implement the instructions in the PRD specs/buy_market_with_trailing_sell_prd.md
- [ ] /commit
- [ ] git push
- [ ] switch to master and merge current branch


## --buy-trailing

- [X] Think hard. Update code/alpaca.py:
1. Create a new CLI arg --buy-trailing;
2. Mirror --buy;
3. Make --amount a required field for --buy-trailing;
4. Add --trailing-percent that defaults to a value in atoms/api/config.py in a dataclass (file does not currently exist);
5. CLI example: python alpaca.py --buy-trailing --symbol AAPL --amount 1000 --trailing-percent 7.5
6. Python call example:
api.submit_order(
    symbol='AAPL',
    qty=10,
    side='buy',
    type='trailing_stop',
    time_in_force='gtc',
    trail_percent='2'
)
7. Of course --submit is required for a real trade, otherwise it is a dry run
8. Update README_alpaca.md

- [X] Important. Think hard. Change --buy-trailing to --sell-trailing so side='sell'; Update README_alpaca.md of course; Replace the required --amount with --quantity which is required; Give an example of a market buy --amount with a matching --sell-trailing --quantity (you will have to fake the quantity).

- [X] make ./code a python package


## Update Alpaca.py

- [X] /commit /clear /read...
- [X] Think hard. Update code/alpaca.py: 1. Research the alpaca trade api to find out how to liquidate a stock; liquidate all positions; 2. Update the code with CLI args for --liquidate --symbol <symbol> and --liquidate-all. There may be an option to close all other orders for the symbol. 3. for the CLI arg --positions, only display the most vital info, e.g., symbol, quantity and average fill.

## Root Cause Analysis

This stock rocketed.

- [X] Important. Think hard. Do a root cause analysis as to why penetration of this stock was 0.0%.  The stock rocketed briefly.
2025-07-31 13:00:00,320 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:01:00,489 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:02:00,466 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:03:00,512 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:05:00,453 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:06:00,513 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:09:00,517 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:11:00,501 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:14:00,586 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:16:00,498 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:17:00,552 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:18:00,529 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)
2025-07-31 13:19:00,471 ET - __main__ - INFO - 🚫 Filtered superduper alert for PAPL: Penetration too low: 0.0% (need 15.0%)


## Telegram-Alpaca Integration

- [X] /commit /clear /read...
- [X] Important.  Think hard. Create a PRD, specs/telegram_alpaca_integration_prd.md: 1. Review code/alpaca.py to review the CLI options --positions --cash --active-order --buy. 2. Review molecules/telegram_polling.py. 3. Create the PRD to integrate calling alpaca.py from telegram_polling.py: a. the trigger word in the telegram message will be '57chevy' (any character case) and it will be the first word of the message; b. Important. Only use the args in the Telegram message. c. --positions --cash --active-order are to be available. d. --buy and the args that go with it are to be available. e. Arg verification is only to be performed by alpaca.py. f. Return the alpaca.py output only to Telegram user 'Bruce' g. Only 'Bruce' is to receive messages for this integration.  h. Only expect the trigger word and args because the execution of alpaca is implied.
- [X] /commit
- [X] git push
- [X] /clear /read...
- [X] git switch -c telegram_alpaca_integration
- [X] Important.  Think hard.  Implement the instructions in the PRD specs/telegram_alpaca_integration_prd.md
- [X] /commit
- [X] git push
- [X] switch to master and merge current branch

## Superduper Alerts
- [X] Think hard.  Update code/orb_alerts_monitor_superduper.py: Add a new CLI arg --no-telegram; do not send Telegram posts when this arg is used.
- [X] Important, think hard. Update code/orb_alerts_monitor_superduper.py: This code generates superduper alerts; find the section that is marking in the "Trend Analysis" section of the alert and the "Momentum" value as red, yellow, and green; where is that code.  Do not change the code at this time.
- [X] Important, think hard. Update code/orb_alerts_monitor_superduper.py: This code generates superduper alerts; find the section that is creating the "Trend Analysis" section of the alert; find where it is doing the analisys for the last 45 minutes; change it to 30 minutes.




- [X] /read-instruction
- [X] git switch -c superduper_alerts
- [X] Think hard. code/orb_alerts.py generates alerts. code/orb_alerts_monitor.py filters alerts to generate super alerts. create code/orb_alerts_monitor_superduper.py to filter super alerts to generate superduper alerts; Mirror code/orb_alerts_monitor.py for the architecture such as the watchdog and filter; However, update the filtering. Make the filtering an atom; the atom should be passed the path of the latest super alert which the watchdog will make available; the atom should look at all super alerts from the symbol on that day (the date will be in the path from the watchdog); the you are to design the filter logic for the superduper alert based upon the stock is rising in price or consolidating; pass a timeframe to the atom and have it default to 45 minutes; use the timeframe for the filter logic; you are to create a new message format that is to be sent on Telegram.
- [X] Update the superduper filter to only look at files prior to the timestamp in the filepath variable.  This will allow back testing.
- [X] /commit
- [X] Publish the new branch
- [X] git switch master and merge new branch


- [no] Update code/orb_alerts_monitor.py: No alert if red candlestick. Maybe not.  That might be an entry point.
- [X] Review code/orb_alerts_monitor.py: How often is it polling the directory structure for creating super alerts? Answer: It uses a watchdog for file creation events allowing to respond immediately.

- [X] git switch -c opening_range
- [X] Think hard. Update code/orb.py: The code currently conducts PCA for a single date; Add --start and --end optional args;  do the old processing if the new args aren't there; if and only if both new args are there, do PCA analyis but do not generate charts; also add a new arg --opening-range, have it default to 15 which is probably what is in the current code.  Please do PCA across all dates in a single pass and not day-by-day.
- [X] Think hard. Create code/orb_analysis.py: This code is to repeatedly call code/orb.py with the date range in ./data and iterate from 10 to 30 by 5's; it is then to compare and contrast the results to determine the impact of the opening range. Put all the results in a single file.  Then I can run them through you separately.

- [X] Think. Update code/orb_alerts_monitor.py: Add the arg --post-only-urgent; update the call to send_orb_alert by adding a bool argument for the new CLI arg; send_orb_alert is to send telegram messages per this new arg; send_orb_alert sends only urgent methods if --post-only-urgent, otherwise send all messages.




- [X] /commit and git tag -a v0.3.3 -m "Superduper charts fix."

- [X] Think hard. Update code/orb_alerts_summary.py: see historical_data/2025-07-28/alerts/summary/bar_chart_superduper_alerts_20250728.png has 5 symbols on the chart as having superduper alerts; There should only be VWAV see historical_data/2025-07-28/superduper_alerts/bullish/ dir. Once you have found the directory error, you will need two update the code for the other two superduper alert charts in that directory.

- [X] /commit and git tag -a v0.3.4 -m "Superduper summary charts fix."







- [X] Think. Update code/orb.py: It should be putting super alerts on the candlestick charts; change that to superduper alerts.

- [X] Think hard. git switch -c accumulate_symbols.  Create atom/api/build_symbol_list.py: You are in an atoms/molecules architecture. Create an atom to combine all files of the form data/YYYYMMDD.csv; eliminate duplicate symbols; set all other fields to zero; Do not zero the fields of the most recent file.  It is important that all the data/columns of the most recent file are preserved.  You might just want to append it and remember to eliminate duplicate symbols. This is going to be ran every trading day.  You might just establish a file with the accumulated data and append to it daily. Create PyTests; copy real data for the tests. Do not integrate the atom.
- [X] /read-instruction
- [X] cp code/orb_alerts.py code/orb_alerts_2.py; Integrate the atom atom/api/build_symbol_list.py into code/orb_alerts_2.py.
- [X] Update code/orb_alerts_2.py: After calculating the ORB, for each stock: if the "Signal" field is zero, set it to orb high.

- [X] Update code/orb_alerts_summary.py: Create bullish and bearish pie charts as well as bar charts for superduper alerts. All superduper alerts are high impact so it is not necessary to create a high impact chart.

- [X] Create new tag. v0.3.1 "Superduper Charts."
- [be careful that it doesn't ruin a good thing] Update code/orb_alerts_monitor_superduper.py: It is currently using 45 minutes for trend analisis; change it to 30 minutes.

- [X] Think hard. Create tests/backtesting/alerts_backtest.py; Create a backtest for code/orb_alerts.py; It is to have CLI args for symbol and date; use --date and --symbol as the names. This backtest will be kept in the repo.

- [X] Think hard. Create a PRD, specs/telegram_post_prd.md, to ceate Telegram atom to post messages to Telegram API. The atom needs to accept a message as a string and post it to Telegram. Use dotenv to retrieve Telegram keys or whatever.  Have a separate CSV containing users to which to post; the file name is to start with ".", be in the root dir, and added to .gitignore.

- [X] Update code/orb_alerts_summary.py: Create bullish and bearish bar charts for super alerts and bin ("current_price" / "orb_high") by increments of 10 percent.
- [X] Update code/orb_alerts_summary.py: Create additional bullish and bearish pie charts for super alerts that the current price is 20% above signal price for bullish alerts and vice versa.
- [X] Update code/orb_alerts_monitor.py: Only send messages to Telegram users after the filters have been applied.
- [X] Update code/orb_alerts_monitor.py: Send Telegram message as urgent if "original_alert": ("current_price" / "orb_high") >= 1.20; otherwise regular message.


- [X] Update atoms/api/parse_args.py to use "-" instead of "_" in argument names. Update README_alpace.md as appropriate.
- [X] Think hard. Review code/alpaca.py: Review _buy: it is a bracket order; however certain types of orders are not allowed in after hours trading; create a new method for after hours trading.
- [X] Think hard.  Is atoms/utils/parse_args.py only used by code/alpaca.py?  If so, move it to atoms/api with the other alpaca atoms.
- [ ] Think hard. Rename atoms/api to atoms/api_alpaca.


- [ ] Create specs/webull.md.
- [ ] Follow instructions in specs/webull.md to create a PRD specs/webull_prd.md. Think Hard.
- [X] Update code/orb_alerts.py to accept new symbols during the day. Due to other updates, just restart the script.  Have Claude add to exiting table.
- [X] Update code/alpaca.py: for --buy, add optional --amount; use amount to calculate number of shares; round shares to 0 decimal places.
- [X] Update code/alpaca.py: for --buy, add optional --stop_loss and use it when provided; add optional --calc_take_profit.  If --calc_take_profit, calculate it as (latest_quote - stop loss) * 1.5.  --calc_take_profit is only valid when used with --stop_loss so print a warning if used without --stop_loss.  Print warning if --calc_take_profit is used with --take_profit.
- [X] Update code/alpaca.py: for --buy, add optional --stop_loss and use it when provided; add optional --calc_take_profit.  If --calc_take_profit, calculate it as (latest_quote - stop loss) * 1.5.  --calc_take_profit is only valid when used with --stop_loss so print a warning if used without --stop_loss.  Print warning if --calc_take_profit is used with --take_profit.
- [X] Review code/alpaca.py: What CLI args can be combined with --buy
- [X] Update atoms/display/plot_candle_chart.py: Add a second Y axis; it is to be price / orb high.

- [X] Update code/alpaca.py: The return values of orders placed are not being checked.  Process all order return values and handle gracefully.
- [X] Git switch -c premarket_data
- [X] Update code/orb_alerts.py to collect data starting at 9:00 so EMA20 can be calculated at market open.  ORB is still 9:30 to 9:45.
- [X] No alerts in the last 15 minutes of the day.
- [X] You must source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca
- [X] Manually plot EMA20 and see if this would be an indicator to stop trading.
- [X] Update code/orb.py to plot EMA20.
- [X] Review: does code/orb_alerts.py have EMA9 and EMA20?
- [X] Update code/orb_alerts.py to include EMA9, EMA20, and the new technical indicators in the alert.
- [X] Update code/orb_alerts_monitor.py to filter out EMA9 below EMA20 for bullish alerts.
- [X] Update code/orb_alerts_history.py to filter out EMA9 below EMA20 for bullish alerts.
- [X] Update code/orb_alerts_monitor.py to filter out stocks that the current low is below EMA9 for bullish alerts.
- [X] Update code/orb_alerts_history.py to filter out stocks that the current low is below EMA9 for bullish alerts.
- [X] Update code/orb.py to plot super alerts instead of alerts.
- [X] Update the analysis software to handle bearish alerts correctly.
- [X] Turn Claude loose at improving ORB alert.
- [X] Stop printing "INFO - Cleaned up" notifications on deleting historical data.
- [X] Update plot candle sticks to accept optional alerts.
- [ ] Update code/orb.py: Accept a date arg; accept an optional symbol arg; combine alerts; plot
- [ ] Update code/orb_alerts.py to keep only the latest historical data file per symbol.  Implement this when saving more data.
- [ ] Create method to add more stocks to the alerts.
- [ ] Update orb chart to take optional bullish and bearish data.
- [ ] Update orb chart to take optional s/r lines.
- [ ] Accumulate stock data in a single file per stock per day.
- [ ] Fix Phase 3 startup it is currently bypassed
- [ ] Update alpaca.py to take .env as argument.
- [X] atoms.monitoring.performance_tracker - WARNING - High memory usage: 82.3%
- [ ] Update to do trailing stop. Create new method UStrailingStop()
- [ ] Check the return values of orders.
- [X] Focus on creating ORB filter.
- [X] Finish establishing the PCA data.
- [ ] Create some more ORB PyTests.
- [X] Create some more ORB PyTests.
- [X] Create some ORB PyTests.
- [X] ORB Create VWAP atom.
- [X] ORB Create EMA with 9 as the default parameter atom.
- [ ] ORB Create atom to calculate a momentum vector from ORB candlesticks.  Hmm. Is this two values? Yes.  Fit a line and use the angle.
- [X] ORB Method extract as an atom the symbol_data calculation in atoms/display/plot_candle_chart.py
- [X] ORB Create ORB.py to monitor ORB trading strategy.
- [X] ORB Create ORB._get_orb_market_data()
- [X] ORB Calculate the ORB for each stock in the first 15 minutes.
- [ ] Add float rotation calculator.
- [X] Create .envpaper
- [X] Create .envlive
- [X] Create '_future_bracket_order()'
- [X] Create underscore buy.  New cli argument for this.
- [ ] Update print_active_orders to do all the prints.


## Bash

        ```bash
        python code/alpaca.py --bracket_order --symbol AAPL --quantity 10 --market_price 150.00
        python code/alpaca.py --get_latest_quote --symbol NINE
        python code/alpaca.py --buy --symbol NINE --take_profit 1.45
        python code/alpaca.py --buy --symbol AAPL --take_profit 210.00 --submit
        python3 code/alpaca.py --buy --symbol AAPL --stop_price 140.00 --take_profit 200.00 --submit
        python3 code/alpaca.py --buy --symbol STKH --stop_price 2.50 --take_profit 3.10

        # quantity will be automatically calculated.
        python3 code/alpaca.py --future_bracket_order --symbol AAPL --limit_price 145.00 --stop_price 140.00 --take_profit 160.00 --submit

        # Auto-calculate take profit based on risk/reward ratio
        python3 code/alpaca.py --buy --symbol AAPL --stop_loss 145.00 --calc_take_profit
        python3 code/alpaca.py --buy --symbol OPEN --stop_loss 4.75 --calc_take_profit --submit

        # Buy $1000 worth with custom stop-loss and auto take-profit
        python3 code/alpaca.py --buy --symbol AAPL --amount 1000.00 --stop_loss 145.00 --calc_take_profit --submit

        # Short sell with manual take profit
        python3 code/alpaca.py --sell_short --symbol AAPL --stop_loss 105.00 --take_profit 95.00

        # Short sell with auto-calculated take profit
        python3 code/alpaca.py --sell_short --symbol AAPL --stop_loss 105.00 --calc_take_profit

        # Short sell $1000 worth with auto take profit
        python3 code/alpaca.py --sell_short --symbol AAPL --amount 1000.00 --stop_loss 105.00 --calc_take_profit --submit


        python code/orb_alerts.py --test --verbose

        python code/alpaca.py --buy-market-trailing-sell-take-profit-percent --symbol VTAK --take-profit-percent 5 --submit
        57chevy --buy-market-trailing-sell-take-profit-percent --symbol AAPL --take-profit-percent 10

        python alpaca_screener.py --volume-surge 2.0 --surge-days 5 --verbose
        python code/alpaca_screener.py  --account-name Bruce --account live --exchanges NASDAQ --max-symbols 6000 --volume-surge 2.0 --surge-days 5 --export-csv surge_2025-08-28.csv --verbose

        ```

## MCP
        ```bash
        claude mcp add --transport http context7 https://mcp.context7.com/mcp
        ```

        https://support.anthropic.com/en/articles/11176164-pre-built-integrations-using-remote-mcp

