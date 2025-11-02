# Market Sentinel Trade Stream Service

This directory contains the real-time trade streaming WebSocket server for Market Sentinel.

## Quick Start

### Manual Start
```bash
./services/start_trade_stream.sh
```

### Install as System Service (Optional)

1. Copy the service file:
```bash
sudo cp services/market_sentinel_trade_stream.service /etc/systemd/system/
```

2. Reload systemd:
```bash
sudo systemctl daemon-reload
```

3. Enable and start the service:
```bash
sudo systemctl enable market_sentinel_trade_stream
sudo systemctl start market_sentinel_trade_stream
```

4. Check status:
```bash
sudo systemctl status market_sentinel_trade_stream
```

5. View logs:
```bash
sudo journalctl -u market_sentinel_trade_stream -f
```

## How It Works

The trade stream server:
1. Connects to Alpaca's WebSocket API for real-time market data
2. Accepts WebSocket connections from the Market Sentinel web interface
3. Subscribes to trade streams for symbols requested by clients
4. Broadcasts trades in real-time to all connected clients

## Configuration

- **Port:** 8765 (default)
- **Host:** 0.0.0.0 (listens on all interfaces)

Set environment variables in `.env`:
```
TRADE_STREAM_HOST=0.0.0.0
TRADE_STREAM_PORT=8765
```

## Dependencies

- websockets
- alpaca-py
- python-dotenv

All dependencies are installed in the `alpaca` conda environment.
