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

```bash
python3 code/orb_alerts.py --verbose
```

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
