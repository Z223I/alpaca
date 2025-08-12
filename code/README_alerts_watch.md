# Alerts Watch - Automated Market Hours Alert System

This system manages ORB alert processes during market hours and provides automated daily summaries.

## Overview

The `alerts_watch.py` script automatically:

1. **Market Open (9:30 AM ET)** - Starts 3 alert processes:
   - `python3 code/orb_alerts_monitor.py --no-telegram --verbose`
   - `python code/orb_alerts_monitor_superduper.py --verbose`
   - `python code/orb_alerts_trade_stocks.py --verbose`

2. **During Market Hours** - Monitors processes and restarts if they fail

3. **Market Close (4:00 PM ET)** - Stops all processes and runs:
   - `python code/orb_alerts_summary.py`
   - `python3 code/orb.py`
   - Sends daily summary to Bruce via Telegram

4. **Schedule**: Only runs Monday-Friday (market days)

## Features

- **Automated Scheduling**: Uses Eastern Time for market hours
- **Process Monitoring**: Restarts failed processes automatically
- **Graceful Shutdown**: Properly stops processes at market close
- **Daily Summaries**: Automated post-market analysis and reporting
- **Telegram Integration**: Sends summary to Bruce after market close
- **Comprehensive Logging**: Timestamped logs in `logs/alerts_watch_YYYYMMDD_HHMMSS.log`

## Usage

### Start the Alerts Watchdog
```bash
# From project root
~/miniconda3/envs/alpaca/bin/python code/alerts_watch.py
```

### Or make it executable and run directly
```bash
chmod +x code/alerts_watch.py
./code/alerts_watch.py
```

### Stop the Watchdog
- Press `Ctrl+C` to stop gracefully
- All managed processes will be stopped automatically

## Configuration

### Market Hours
- **Open**: 9:30 AM ET (Monday-Friday)
- **Close**: 4:00 PM ET (Monday-Friday)
- **Timezone**: US/Eastern (handles daylight savings automatically)

### Managed Processes
1. **ORB Monitor**: Super alert generation system
2. **ORB Superduper**: Enhanced superduper alerts
3. **ORB Trades**: Automated trade execution

### Post-Market Analysis
1. **ORB Alerts Summary**: Daily alert statistics
2. **ORB Analysis**: Chart generation and analysis

## Log Output Example

```
[2025-08-12 09:30:00] INFO: 🔔 MARKET OPEN - Starting alert processes
[2025-08-12 09:30:01] INFO: 🚀 Starting orb_monitor: python3 code/orb_alerts_monitor.py --no-telegram --verbose
[2025-08-12 09:30:02] INFO: ✅ orb_monitor started (PID: 12345)
[2025-08-12 09:30:03] INFO: 🚀 Starting orb_superduper: python code/orb_alerts_monitor_superduper.py --verbose
[2025-08-12 09:30:04] INFO: ✅ orb_superduper started (PID: 12346)
[2025-08-12 09:30:05] INFO: 🚀 Starting orb_trades: python code/orb_alerts_trade_stocks.py --verbose
[2025-08-12 09:30:06] INFO: ✅ orb_trades started (PID: 12347)
[2025-08-12 16:00:00] INFO: 🔔 MARKET CLOSE - Stopping alert processes
[2025-08-12 16:00:01] INFO: 🛑 Stopping orb_monitor (PID: 12345)
[2025-08-12 16:00:02] INFO: ✅ orb_monitor stopped gracefully
[2025-08-12 16:00:05] INFO: 📊 Starting post-market analysis...
[2025-08-12 16:00:10] INFO: ✅ orb_alerts_summary completed successfully
[2025-08-12 16:00:20] INFO: ✅ orb completed successfully
[2025-08-12 16:00:25] INFO: 📤 Sending daily summary to Bruce...
[2025-08-12 16:00:26] INFO: ✅ Summary sent to Bruce (123456789)
```

## Telegram Summary

The daily summary sent to Bruce includes:
- Alert processes status for the day
- Post-market analysis results
- Key highlights and notes
- Error information if any issues occurred
- Reference to full logs for detailed information

## Dependencies

- `schedule`: For task scheduling
- `pytz`: For timezone handling
- `atoms.telegram.telegram_post`: For sending messages
- `atoms.telegram.user_manager`: For finding Bruce's chat ID

## Error Handling

- **Process Failures**: Automatically restarts failed processes during market hours
- **Script Failures**: Logs errors and continues with remaining tasks
- **Telegram Failures**: Continues operation even if summary sending fails
- **Graceful Degradation**: System continues working even if individual components fail

## Integration with Daily Instructions

This replaces the manual execution of the ORB alert commands in `Daily_Instructions.md`. Instead of running each command manually, just start the alerts watchdog and it handles everything automatically.