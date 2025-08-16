# Backtesting Directory Isolation Problems & Solutions

## Overview
Complex multi-day debugging session to fix directory isolation issues in backtesting system. Root problem: alerts and files were mixing between runs and main directories, causing incorrect analysis results.

## Problems Encountered

### 1. Config Module Caching Issue
**Problem**: Config functions returned cached constants instead of reading environment variables dynamically.
- `get_historical_root_dir()` returned `DEFAULT_HISTORICAL_ROOT_DIR` (hardcoded to ".")
- Environment variables set correctly but config didn't read them

**Solution**: Modified all config getter functions to read `ALPACA_ROOT_PATH` dynamically:
```python
def get_historical_root_dir() -> HistoricalRootDir:
    import os
    current_root_path = os.getenv('ALPACA_ROOT_PATH', '.')
    return HistoricalRootDir(root_path=current_root_path)
```

### 2. Subprocess Environment Inheritance
**Problem**: Child processes (monitors, plot generation) weren't inheriting environment variables.
- `subprocess.Popen()` and `subprocess.run()` missing `env=os.environ.copy()`
- Processes started with clean environment, not parent's modified environment

**Solution**: Added environment inheritance to all subprocess calls:
```python
subprocess.Popen(..., env=os.environ.copy())
subprocess.run(..., env=os.environ.copy())
```

### 3. Config Backup/Restore Corruption
**Problem**: Hitting Ctrl+C during backtesting left config file in modified state.
- Backup/restore mechanism in `finally` block never executed on interruption
- Subsequent runs started with corrupted config

**Solution**: Added signal handlers for graceful cleanup:
```python
signal.signal(signal.SIGINT, self._signal_handler)
signal.signal(signal.SIGTERM, self._signal_handler)
```

### 4. Analysis Script Reading Wrong Data
**Problem**: Analysis script counted old live trading alerts instead of run-specific data.
- Script didn't use run-isolated directories
- Charts showed alerts from main `./historical_data/` not run directories

**Solution**: Fixed analysis to only read from run directories, clear environment variables during analysis.

### 5. Recurring Config Restoration
**Problem**: Config functions kept reverting to cached versions after each run.
- `config_orig.py` backup contained old cached version
- Restore process undid dynamic fixes repeatedly

**Status**: Fixed multiple times, but backup file needs updating with dynamic version.

## Current Status (End of Session)

### ✅ Fixed Components
- Dynamic config system with environment variable reading
- Subprocess environment inheritance for pipeline processes  
- Signal handling for Ctrl+C cleanup with verification
- Analysis script isolation
- Debug logging throughout system

### ❌ Still Broken
- **Plot generation missing again** - plots not appearing in run directories
- Suggests plot subprocess (`code/alpaca.py --plot`) still not getting environment variables
- Environment inheritance fix might not be working for plot generation specifically

## Technical Architecture

### Environment Variables Used
- `ALPACA_ROOT_PATH`: Points to run directory (e.g., `/path/to/runs/run_2025-08-13_xxxxx`)
- `ALPACA_GREEN_THRESHOLD`: Momentum threshold for current run
- `ALPACA_TIMEFRAME`: Analysis timeframe for current run

### Directory Structure
```
runs/run_2025-08-13_xxxxx/
├── data/20250813.csv
├── plots/20250813/SYMBOL_chart.png
├── logs/orb_superduper/...
└── historical_data/2025-08-13/
    ├── super_alerts/bullish/
    ├── superduper_alerts/bullish/
    └── superduper_alerts_sent/bullish/green/
```

### Process Flow
1. Backtesting sets environment variables for run
2. All subprocesses inherit environment via `env=os.environ.copy()`
3. Config functions read environment dynamically
4. Files saved to run-specific directories
5. Analysis reads only from run directories

## Debug Commands for Tomorrow

### Test Environment Variable Reading
```bash
ALPACA_ROOT_PATH="/tmp/test" python3 -c "
from atoms.alerts.config import get_historical_root_dir
print(f'Config test: {get_historical_root_dir().root_path}')
"
```

### Check Plot Generation Specifically
```bash
# Set environment and test plot generation directly
export ALPACA_ROOT_PATH="/path/to/test"
python3 code/alpaca.py --plot --symbol TEST --date 2025-08-13
```

### Verify Subprocess Environment
Add debug logging to `code/alpaca.py` plot generation to show:
- What environment variables the plot subprocess sees
- What directory paths it's using for saves

## Next Steps
1. **Debug plot generation environment inheritance**
   - Add logging to `code/alpaca.py` to show received environment variables
   - Verify plot subprocess is actually getting `ALPACA_ROOT_PATH`
   
2. **Test complete backtesting run**
   - Verify all components (alerts, plots, logs) save to run directories
   - Check analysis shows correct per-run counts
   
3. **Validate Ctrl+C cleanup**
   - Test signal handler restores config properly
   - Verify environment variables are cleared

4. **Update config backup**
   - Ensure `config_orig.py` contains dynamic version to prevent regression

## Files Modified
- `atoms/alerts/config.py` - Dynamic environment variable reading
- `code/backtesting.py` - Environment management, subprocess inheritance, signal handling
- `atoms/telegram/orb_alerts.py` - Debug logging for alert storage
- `code/analyze_backtesting_results.py` - Environment isolation for analysis

## Branch
All fixes committed to `feature/backtesting-problems` branch.
Ready for testing and eventual merge to master.