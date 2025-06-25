# TODO

- [X] Create ORB.py to monitor ORB trading strategy.
- [X] Create ORB._get_orb_market_data()
- [ ] Calculate the ORB for each stock in the first 15 minutes.
- [X] Create .envpaper
- [ ] Create .envlive
- [X] Create '_future_bracket_order()'
- [ ] Update to do trailing stop. Create new method UStrailingStop()
- [ ] Check the return values of orders.
- [X] Create underscore buy.  New cli argument for this.
- [ ] Update print_active_orders to do all the prints.


## Bash

        ```bash
        python code/alpaca.py --bracket_order --symbol AAPL --quantity 10 --market_price 150.00
        python code/alpaca.py --get_latest_quote --symbol NINE
        python code/alpaca.py --buy --symbol NINE --take_profit 1.45
        python code/alpaca.py --buy --symbol AAPL --take_profit 210.00 --submit

        # quantity will be automatically calculated.
        python3 code/alpaca.py --future_bracket_order --symbol AAPL --limit_price 145.00 --stop_price 140.00 --take_profit 160.00 --submit
        ```

## MCP
        ```bash
        claude mcp add --transport http context7 https://mcp.context7.com/mcp
        ```
