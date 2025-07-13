# TODO

The remaining tasks are lower priority documentation and deployment tools:
  - Production configuration and deployment tools
  - Comprehensive API documentation


- [ ] No alerts in the last 30 minutes of the day.
- [X] You must source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca
- [ ] Manually plot EMA20 and see if this would be an indicator to stop trading.
- [X] Update code/orb.py to plot EMA20.


- [X] Update the analysis software to handle bearish alerts correctly.
- [X] Turn Claude loose at improving ORB alert.

- [ ] Update plot candle sticks to accept optional alerts.
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


        python code/orb_alerts.py --test --verbose
        ```

## MCP
        ```bash
        claude mcp add --transport http context7 https://mcp.context7.com/mcp
        ```

        https://support.anthropic.com/en/articles/11176164-pre-built-integrations-using-remote-mcp

