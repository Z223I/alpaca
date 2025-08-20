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
- Starts/stops all 4 alert processes automatically (including orb_alerts.py)
- Process monitoring and restart on failures
- Post-market analysis and Telegram summary to Bruce
- Comprehensive logging

This replaces the need to manually run orb_watchdog.py and individual commands below.

### Manual ORB Alert Commands (Alternative)

For manual control or testing, use these individual commands:

#### ORB Basic Alerts (DEPRECATED - Use alerts_watch.py instead)

**⚠️ DEPRECATED**: orb_watchdog.py is no longer recommended. Use `alerts_watch.py` which manages orb_alerts.py along with all other processes.

For manual testing only:

```bash
python3 code/orb_alerts.py --verbose
```

Note: alerts_watch.py now handles orb_alerts.py management automatically.

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
