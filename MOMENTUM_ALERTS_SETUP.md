# Momentum Alerts System - Setup Guide

## Overview

The Momentum Alerts System monitors stocks and generates alerts based on VWAP and EMA9 criteria. It runs continuously, collecting fresh data starting at 9:40 ET and every 20 minutes thereafter, 7 days a week including weekends.

## Current Status

**⚠️ NOT RUNNING AUTOMATICALLY** - You must manually start the service.

## How to Run

### Option 1: Quick Start (Manual - Foreground)

Run in the foreground (logs to console):

```bash
cd ~/dl/github.com/Z223I/alpaca
~/miniconda3/envs/alpaca/bin/python cgi-bin/molecules/alpaca_molecules/momentum_alerts.py
```

- Stops when you close terminal or press `Ctrl+C`
- Good for testing

### Option 2: Background Run (Manual - nohup)

Run in the background with logging:

```bash
cd ~/dl/github.com/Z223I/alpaca
nohup ~/miniconda3/envs/alpaca/bin/python \
  cgi-bin/molecules/alpaca_molecules/momentum_alerts.py \
  > logs/momentum_alerts.log 2>&1 &
```

- Runs in background
- Logs to `logs/momentum_alerts.log`
- Survives terminal close
- **Does NOT restart after reboot**

To stop:
```bash
pkill -f "momentum_alerts.py"
```

### Option 3: Systemd Service (Automatic Startup) ⭐ **RECOMMENDED**

Set up automatic startup on boot with systemd:

#### 1. Install the Service

```bash
cd ~/dl/github.com/Z223I/alpaca
./services/setup_momentum_alerts_service.sh
```

This will:
- Copy the service file to `~/.config/systemd/user/`
- Enable auto-start on boot
- Set up automatic restart on failure

#### 2. Start the Service

```bash
systemctl --user start momentum_alerts
```

#### 3. Check Status

```bash
systemctl --user status momentum_alerts
```

#### 4. View Logs

**Live tail (systemd journal):**
```bash
journalctl --user -u momentum_alerts -f
```

**Log files:**
```bash
# Output log
tail -f ~/dl/github.com/Z223I/alpaca/logs/momentum_alerts.log

# Error log
tail -f ~/dl/github.com/Z223I/alpaca/logs/momentum_alerts_error.log
```

### Systemd Service Commands

```bash
# Start the service
systemctl --user start momentum_alerts

# Stop the service
systemctl --user stop momentum_alerts

# Restart the service
systemctl --user restart momentum_alerts

# Check status
systemctl --user status momentum_alerts

# Enable auto-start on boot
systemctl --user enable momentum_alerts

# Disable auto-start on boot
systemctl --user disable momentum_alerts

# View logs (real-time)
journalctl --user -u momentum_alerts -f

# View logs (last 100 lines)
journalctl --user -u momentum_alerts -n 100
```

## What the System Does

### At Startup

1. **Load Existing Symbols** - Immediately loads symbols from existing CSV files (if present)
2. **Collect Fresh Data** - Runs `market_open_top_gainers.py` to get current top gainers
3. **Volume Surge Scan** - Scans for volume surge patterns
4. **Schedule Next Runs** - Sets up 20-minute intervals starting from 9:40 ET

### During Operation

- **Every 20 minutes** starting at 9:40 ET:
  - Collects top gainers from NASDAQ and AMEX
  - Updates symbol list with up to 40 symbols per source

- **Every minute**:
  - Monitors all symbols for momentum criteria
  - Checks VWAP, EMA9, and urgency filters
  - Sends Telegram alerts for qualifying stocks

- **Continuous monitoring**:
  - Watches CSV files for updates
  - Automatically loads new symbols
  - Tracks stock halts and fundamental data

### Data Sources

Symbols are collected from:

1. **Top Gainers** - `historical_data/{YYYY-MM-DD}/market/gainers_nasdaq_amex.csv`
   - First 40 symbols (excluding those ending in 'W')

2. **Volume Surge** - `historical_data/{YYYY-MM-DD}/volume_surge/relative_volume_nasdaq_amex.csv`
   - First 40 symbols with volume surge detected

3. **Oracle** - `data/{YYYYMMDD}.csv`
   - All symbols from Oracle source (no limit)

### Schedule

```
09:40 ET - First run
10:00 ET - Second run
10:20 ET - Third run
10:40 ET - Fourth run
... continues every 20 minutes ...
23:40 ET - Last run of day
09:40 ET - First run next day (including weekends)
```

## Verification

Check if the service is running:

```bash
# Check process
ps aux | grep momentum_alerts

# Check systemd status
systemctl --user status momentum_alerts

# Check recent logs
journalctl --user -u momentum_alerts -n 50
```

You should see log messages like:
```
📂 Loading existing symbols from CSV files at startup...
✅ Loaded 12 symbols from existing CSV files
📊 Running startup script immediately to collect fresh data...
🚀 Running startup script: market_open_top_gainers.py
📈 Running volume surge scanner at startup
📅 Next startup script run scheduled for: YYYY-MM-DD HH:MM:SS ET
⏰ Runs every 20 minutes starting at 9:40 ET daily (including weekends)
```

## Troubleshooting

### Service won't start

Check logs:
```bash
journalctl --user -u momentum_alerts -n 100
```

Check error log:
```bash
tail -n 50 ~/dl/github.com/Z223I/alpaca/logs/momentum_alerts_error.log
```

### Service keeps restarting

The service is configured with `Restart=always` and `RestartSec=10`, so it will automatically restart if it crashes. Check logs to see why it's failing.

### Logs not appearing

Make sure the logs directory exists:
```bash
mkdir -p ~/dl/github.com/Z223I/alpaca/logs
```

### Check if running

```bash
# Check process
ps aux | grep momentum_alerts | grep -v grep

# Check systemd
systemctl --user is-active momentum_alerts
```

## Files

- **Service file**: `services/momentum_alerts.service`
- **Setup script**: `services/setup_momentum_alerts_service.sh`
- **Main script**: `cgi-bin/molecules/alpaca_molecules/momentum_alerts.py`
- **Config**: `cgi-bin/molecules/alpaca_molecules/momentum_alerts_config.py`
- **Output log**: `logs/momentum_alerts.log`
- **Error log**: `logs/momentum_alerts_error.log`

## Next Steps

1. **Install the service** (if not done):
   ```bash
   ./services/setup_momentum_alerts_service.sh
   ```

2. **Start it**:
   ```bash
   systemctl --user start momentum_alerts
   ```

3. **Verify it's running**:
   ```bash
   systemctl --user status momentum_alerts
   ```

4. **Watch the logs**:
   ```bash
   journalctl --user -u momentum_alerts -f
   ```

---

**Last Updated**: 2025-11-02
**Status**: Ready for deployment
