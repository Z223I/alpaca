---
description: "Check ORB alerts system status and recent data"
allowed-tools: ["bash", "ls", "read"]
---

# ORB Alerts System Status

Check the current status of the ORB alerts system including:
- Recent historical data files
- Latest alerts generated
- System logs

## Recent Historical Data
```bash
!ls -la historical_data/$(date +%Y-%m-%d)/market_data/ 2>/dev/null | head -10 || echo "No data for today"
```

## Recent Alerts (Bullish)
```bash
!ls -la alerts/bullish/ 2>/dev/null | tail -5 || echo "No recent bullish alerts"
```

## Recent Alerts (Bearish)  
```bash
!ls -la alerts/bearish/ 2>/dev/null | tail -5 || echo "No recent bearish alerts"
```

## Check if ORB alerts process is running
```bash
!ps aux | grep -v grep | grep orb_alerts || echo "ORB alerts not currently running"
```