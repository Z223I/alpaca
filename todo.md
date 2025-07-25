# TODO

The remaining tasks are lower priority documentation and deployment tools:
  - Production configuration and deployment tools
  - Comprehensive API documentation

Be sure to watch stocks minute-by-minute for buying oportunities Jdun Trades style but with 1 min candles.

- [no] Update code/orb_alerts_monitor.py: No alert if red candlestick. Maybe not.  That might be an entry point.
- [X] Review code/orb_alerts_monitor.py: How often is it polling the directory structure for creating super alerts? Answer: It uses a watchdog for file creation events allowing to respond immediately.


- [X] Think hard. git switch -c accumulate_symbols.  Create atom/api/build_symbol_list.py: You are in an atoms/molecules architecture. Create an atom to combine all files of the form data/YYYYMMDD.csv; eliminate duplicate symbols; set all other fields to zero; Do not zero the fields of the most recent file.  It is important that all the data/columns of the most recent file are preserved.  You might just want to append it and remember to eliminate duplicate symbols. This is going to be ran every trading day.  You might just establish a file with the accumulated data and append to it daily. Create PyTests; copy real data for the tests. Do not integrate the atom.
- [ ] Integrate atom.
- [ ] Update code/orb_alerts.py: After calculating the ORB, for each stock: if the "Signal" field is zero, set it to orb high.

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

