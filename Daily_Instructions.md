# Daily Instructions

Always running

```bash
~/miniconda3/envs/alpaca/bin/python molecules/telegram_polling_watchdog.py
```

The watchdog automatically starts and monitors the telegram polling service. It will restart the service if errors are detected and creates timestamped logs in the logs/ directory.

Morning

## Collect Symbols

### Windows Machine

### Ubuntu Machine

## Execute Code

### Automated Alerts Watch (Recommended)

Fully automated system that manages all ORB alert processes during market hours (9:30 AM - 4:00 PM ET, Mon-Fri). Automatically starts all processes at market open, monitors them, and runs post-market analysis.

```bash
python code/alerts_watch.py
```

Features:
- Automatic market hours scheduling (Eastern Time)
- Starts/stops all 3 alert processes automatically
- Process monitoring and restart on failures
- Post-market analysis and Telegram summary to Bruce
- Comprehensive logging

This replaces the need to manually run the individual commands below.

### Manual ORB Alert Commands (Alternative)

For manual control or testing, use these individual commands:

#### ORB Alerts Watchdog

Monitors and manages the orb_alerts.py process automatically. Launches orb_alerts on startup, displays output to console, and restarts if the process dies.

```bash
python3 code/orb_watchdog.py
```

Features:
- Automatic startup and restart of orb_alerts.py
- Live output display with [ORB] prefix
- Graceful shutdown with Ctrl+C
- Uses conda environment automatically

#### ORB Super Alerts

Produces ORB super alerts.

```bash
python3 code/orb_alerts_monitor.py --no-telegram --verbose
```

#### ORB Superduper Alerts

Produces superduper alerts.

```bash
python code/orb_alerts_monitor_superduper.py --verbose
```

#### ORB Trades

Generates trades. Configuration required.

```bash
python code/orb_alerts_trade_stocks.py --verbose
```

After market close

ORB Alert Summary

```bash
python code/orb_alerts_summary.py
```

ORB Analysis with Charts

```bash
python3 code/orb.py
```
