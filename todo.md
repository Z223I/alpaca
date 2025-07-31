# TODO

- [ ] Implement EMA divergence quality prediction system following specs/ema_divergence_prd.md (use only data from 2025-07-28 onwards)

The remaining tasks are lower priority documentation and deployment tools:
  - Production configuration and deployment tools
  - Comprehensive API documentation

Be sure to watch stocks minute-by-minute for buying oportunities Jdun Trades style but with 1 min candles.


- [X] /commit /clear /read...
- [X] Important.  Think hard. Create a PRD, specs/telegram_alpaca_integration_prd.md: 1. Review code/alpaca.py to review the CLI options --positions --cash --active-order --buy. 2. Review molecules/telegram_polling.py. 3. Create the PRD to integrate calling alpaca.py from telegram_polling.py: a. the trigger word in the telegram message will be '57chevy' (any character case) and it will be the first word of the message; b. Important. Only use the args in the Telegram message. c. --positions --cash --active-order are to be available. d. --buy and the args that go with it are to be available. e. Arg verification is only to be performed by alpaca.py. f. Return the alpaca.py output only to Telegram user 'Bruce' g. Only 'Bruce' is to receive messages for this integration.  h. Only expect the trigger word and args because the execution of alpaca is implied.
- [X] /commit
- [X] git push
- [X] /clear /read...
- [X] git switch -c telegram_alpaca_integration
- [ ] Important.  Think hard.  Implement the instructions in the PRD specs/telegram_alpaca_integration_prd.md
- [ ] /commit
- [ ] git push
- [ ] switch to master and merge current branch



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
        ```

## MCP
        ```bash
        claude mcp add --transport http context7 https://mcp.context7.com/mcp
        ```

        https://support.anthropic.com/en/articles/11176164-pre-built-integrations-using-remote-mcp

