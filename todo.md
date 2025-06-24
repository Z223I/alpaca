# TODO

- [ ] Create '_future_bracket_order()'
- [ ] Update to do trailing stop. Create new method UStrailingStop()
- [ ] Check the return values of orders.
- [X] Create underscore buy.  New cli argument for this.
- [ ] Update print_active_orders to do all the prints.
- [ ] Update .env to use both paper and live trading.  Include the URLs.
- [ ] Update code/alpaca.py to take --live arg which defaults to False. and create paper and live versions of
        self.key = os.getenv('ALPACA_API_KEY')
        self.secret = os.getenv('ALPACA_SECRET_KEY')
        self.headers = {'APCA-API-KEY-ID':self.key, 'APCA-API-SECRET-KEY':self.secret}

        self.baseURL = 'https://api.alpaca.markets'

## Bash

        ```bash
        python code/alpaca.py --bracket_order --symbol AAPL --quantity 10 --market_price 150.00
        python code/alpaca.py --get_latest_quote --symbol NINE
        python code/alpaca.py --buy --symbol NINE --take_profit 1.45
        python code/alpaca.py --buy --symbol AAPL --take_profit 210.00 --submit
        ```
