# Market Sentinel

## High Level Requirements

You are a highly skilled web designer, Python developer, and software architect.
You will be using Python and Flask to be compatible with GoDaddy websites.  I will have the appropriate hosting plan.

## Mid Level Requirements

Create ./public_html/index.html to be the "Market Sentinel" web interface.

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

The Apache 2 web server is running.  The ./public_html files can be reached using http://localhost/market_sentinel/.

See APACHE_SETUP.md and keep it up to date.

### Standards

Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

I am not receive trade data to my chart.

I ran the starter script:
./services/start_trade_stream.sh
Checking dependencies...
Starting Trade Stream Server...
2025-11-02 09:02:07,409 - __main__ - INFO - Starting Trade Stream Server on 0.0.0.0:8765
2025-11-02 09:02:07,410 - __main__ - INFO - Starting Alpaca data stream...
2025-11-02 09:02:07,410 - websockets.server - INFO - server listening on 0.0.0.0:8765
2025-11-02 09:02:07,410 - __main__ - INFO - ✅ Trade Stream Server running on ws://0.0.0.0:8765
2025-11-02 09:02:26,719 - websockets.server - INFO - connection open
2025-11-02 09:02:26,719 - __main__ - INFO - New client connected: 127.0.0.1:44254
2025-11-02 09:02:31,996 - __main__ - INFO - Client subscribing to AAPL
2025-11-02 09:02:31,996 - __main__ - INFO - Subscribing to Alpaca stream for AAPL
2025-11-02 09:02:31,996 - __main__ - INFO - Client subscribed to AAPL. Total subscribers: 1
2025-11-02 09:02:31,996 - alpaca.data.live.websocket - INFO - started data stream
2025-11-02 09:02:31,996 - alpaca.data.live.websocket - INFO - starting data websocket connection
2025-11-02 09:02:31,996 - alpaca.data.live.websocket - INFO - connecting to wss://stream.data.alpaca.markets/v2/iex
2025-11-02 09:02:32,182 - alpaca.data.live.websocket - INFO - connected to wss://stream.data.alpaca.markets/v2/iex
2025-11-02 09:02:32,215 - alpaca.data.live.websocket - INFO - subscribed to trades: ['AAPL'], corrections: ['AAPL'], cancelErrors: ['AAPL']
