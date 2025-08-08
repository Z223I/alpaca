# Daily Instructions

Always running

```bash
./start_telegram_polling.sh
```

Morning

## Collect Symbols

### Windows Machine

### Ubuntu Machine

## Execute Code

### Orb Alerts

Produces alerts.

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

### ORB Super Alerts

Produces ORB super alerts.

```bash
python3 code/orb_alerts_monitor.py --no-telegram --verbose
```

### ORB Superduper Alerts

Produces superduper alerts.

```bash
python code/orb_alerts_monitor_superduper.py --verbose
```

### ORB Trades

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
