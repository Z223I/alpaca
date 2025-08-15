# Backtesting Notes

## Directory Creation Behavior

### superduper_alerts_sent Directory
- **Created by**: `atoms/telegram/orb_alerts.py` in the `_store_sent_superduper_alert()` function (line 247)
- **When created**: Only when alerts are successfully sent via Telegram
- **Important**: If `orb_alerts_monitor_superduper.py` is run with `--no-telegram`, the `superduper_alerts_sent` directory is **NOT** created
- **Directory structure**: `historical_data/{date}/superduper_alerts_sent/{bullish,bearish}/{yellow,green}`

### Call Chain for Directory Creation
1. `orb_alerts_monitor_superduper.py` processes superduper alert
2. Calls `send_orb_alert()` (only if `--no-telegram` flag is NOT set)
3. `send_orb_alert()` sends via Telegram
4. If successful, calls `_store_sent_superduper_alert()`
5. `_store_sent_superduper_alert()` creates directory with `os.makedirs(target_dir, exist_ok=True)`

### Other Directory Creation
- **superduper_alerts_dir**: Created by `orb_alerts_monitor_superduper.py` during initialization (line 108)
- **super_alerts_dir**: Created by `orb_alerts_monitor_superduper.py` during initialization (line 107)