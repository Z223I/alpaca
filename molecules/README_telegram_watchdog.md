# Telegram Polling Watchdog

This watchdog system ensures that `molecules/telegram_polling.py` is always running.

## Features

- **Process Monitoring**: Continuously monitors if telegram_polling.py is running
- **Auto-Start**: Automatically starts the polling service if it's not running
- **Error Detection**: Monitors logs for specific error patterns and triggers restarts
- **Smart Restart**: Waits 30 seconds after errors before restarting
- **Timestamped Logs**: Creates logs in format `logs/telegram_polling_YYYYMMDD_HHMMSS.log`
- **Graceful Shutdown**: Handles SIGINT/SIGTERM signals properly

## Usage

### Start the Watchdog
```bash
# From project root
~/miniconda3/envs/alpaca/bin/python molecules/telegram_polling_watchdog.py
```

### Or make it executable and run directly
```bash
chmod +x molecules/telegram_polling_watchdog.py
./molecules/telegram_polling_watchdog.py
```

### Stop the Watchdog
- Press `Ctrl+C` to stop gracefully
- The watchdog will also stop the polling service it manages

## Error Detection

The watchdog monitors for these error patterns in the polling service logs:
- `ERROR: Error polling updates:`
- `ERROR:`
- `CRITICAL:`
- `❌ Polling error:`
- `❌ Failed to start polling service:`

When any of these patterns are detected, the watchdog will:
1. Stop the current polling service
2. Wait 30 seconds
3. Start a new instance of the polling service

## Log Files

- **Watchdog Logs**: `logs/telegram_polling_YYYYMMDD_HHMMSS.log`
- **Log Format**: `[YYYY-MM-DD HH:MM:SS] LEVEL: message`
- **Polling Output**: Prefixed with `📤 POLLING:` in watchdog logs

## Configuration

Key settings in `TelegramPollingWatchdog` class:
- `restart_delay = 30`: Seconds to wait before restarting after error
- `check_interval = 5`: Seconds between process health checks
- `error_patterns`: List of error patterns to watch for

## Example Log Output

```
[2025-08-12 15:51:08] INFO: 🚀 Starting Telegram Polling Watchdog
[2025-08-12 15:51:08] INFO: ⏰ Check interval: 5 seconds  
[2025-08-12 15:51:08] INFO: ⏳ Restart delay on error: 30 seconds
[2025-08-12 15:51:08] INFO: 🚀 Starting polling service...
[2025-08-12 15:51:08] INFO: ✅ Polling service started (PID: 12345)
[2025-08-12 15:51:08] INFO: 📤 POLLING: 🚀 Starting Telegram polling service...
[2025-08-12 15:51:15] ERROR: 🚨 Error detected: ERROR: Error polling updates:
[2025-08-12 15:51:15] ERROR: 🚨 Handling polling error: ERROR: Error polling updates: Connection timeout
[2025-08-12 15:51:15] INFO: 🛑 Stopping polling service (PID: 12345)
[2025-08-12 15:51:15] INFO: ⏳ Waiting 30 seconds before restart...
[2025-08-12 15:51:45] INFO: 🔄 Restarting polling service after error...
```

## Status Monitoring

The watchdog provides a `get_status()` method that returns:
- `watchdog_running`: Whether the watchdog is active
- `polling_service_running`: Whether the polling service is running
- `polling_process_pid`: PID of the polling process
- `log_file`: Path to the current log file
- `restart_delay`: Configured restart delay
- `check_interval`: Configured check interval