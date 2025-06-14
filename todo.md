# TODO

- [ ] Update .env to use both paper and live trading.  Include the URLs.
- [ ] Update code/alpaca.py to take --live arg which defaults to False. and create paper and live versions of
        self.key = os.getenv('ALPACA_API_KEY')
        self.secret = os.getenv('ALPACA_SECRET_KEY')
        self.headers = {'APCA-API-KEY-ID':self.key, 'APCA-API-SECRET-KEY':self.secret}

        self.baseURL = 'https://api.alpaca.markets'
