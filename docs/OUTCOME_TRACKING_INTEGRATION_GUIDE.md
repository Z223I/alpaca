# Outcome Tracking Integration Guide

## Overview

This guide provides step-by-step instructions to integrate outcome tracking into `squeeze_alerts.py`.

**Implementation file:** `docs/OUTCOME_TRACKING_IMPLEMENTATION.py` (contains all code)

**Estimated time:** 30-45 minutes

---

## Pre-Integration Checklist

- [ ] Backup current `squeeze_alerts.py`
- [ ] Review implementation code in `OUTCOME_TRACKING_IMPLEMENTATION.py`
- [ ] Ensure service is running: `systemctl status squeeze_alerts`
- [ ] Note current git commit: `git log -1 --oneline`

---

## Integration Steps

### STEP 1: Add Configuration Constants

**Location:** After existing class constants (around line 70, after `SQUEEZE_PERCENT = 2.0`)

**Action:** Add these constants to the `SqueezeAlertsMonitor` class:

```python
    # ===== OUTCOME TRACKING CONFIGURATION =====
    # Enable/disable outcome tracking globally
    OUTCOME_TRACKING_ENABLED = True

    # Duration to track outcomes after squeeze detection (minutes)
    OUTCOME_TRACKING_DURATION_MINUTES = 10

    # Interval for recording price snapshots (minutes)
    OUTCOME_TRACKING_INTERVAL_MINUTES = 1

    # Time tolerance for interval recording (seconds)
    OUTCOME_INTERVAL_TOLERANCE_SECONDS = 30

    # Maximum concurrent followups to track (memory/performance limit)
    OUTCOME_MAX_CONCURRENT_FOLLOWUPS = 100

    # Stop loss threshold (percentage below entry price)
    OUTCOME_STOP_LOSS_PERCENT = 7.5

    # Target gain thresholds to track achievement (percentages above entry)
    OUTCOME_TARGET_THRESHOLDS = [5.0, 10.0, 15.0]
```

**Verification:**
```bash
grep "OUTCOME_TRACKING_ENABLED" cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py
# Should output the constant definition
```

---

### STEP 2: Add Data Structures to __init__()

**Location:** In `__init__()` method, after line 125 (after `self.latest_spy_timestamp = None`)

**Action:** Add these data structure initializations:

```python
        # ===== OUTCOME TRACKING DATA STRUCTURES =====
        # Active followups: tracks outcomes for squeezes in progress
        # Key format: "AAPL_2025-12-12_152045" (symbol_date_time)
        self.active_followups: Dict[str, Dict[str, Any]] = {}

        # Cumulative volume since squeeze start (for each followup)
        self.followup_volume_tracking: Dict[str, int] = {}

        # Cumulative trades since squeeze start (for each followup)
        self.followup_trades_tracking: Dict[str, int] = {}
```

**Verification:**
```bash
grep "active_followups" cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py
# Should output the dict initialization
```

---

### STEP 3: Add Outcome Tracking Methods

**Location:** After `_calculate_phase1_metrics()` method (around line 1362)

**Action:** Add all 7 methods from `OUTCOME_TRACKING_IMPLEMENTATION.py`:

1. `_start_outcome_tracking()` (~90 lines)
2. `_check_outcome_intervals()` (~80 lines)
3. `_update_followup_statistics()` (~90 lines)
4. `_record_outcome_interval()` (~50 lines)
5. `_build_outcome_summary()` (~80 lines)
6. `_finalize_outcome_tracking()` (~50 lines)
7. `_update_alert_with_outcomes()` (~40 lines)

**Copy-paste all methods from OUTCOME_TRACKING_IMPLEMENTATION.py (Part 3).**

**Verification:**
```bash
grep -c "def _.*outcome" cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py
# Should output: 7
```

---

### STEP 4: Modify _save_squeeze_alert_sent() Return Type

**Location:** Line 1822, method definition

**CHANGE FROM:**
```python
def _save_squeeze_alert_sent(self, symbol: str, first_price: float, last_price: float,
                              # ... more params ...
                              phase1_metrics: Dict[str, Any]) -> None:
```

**CHANGE TO:**
```python
def _save_squeeze_alert_sent(self, symbol: str, first_price: float, last_price: float,
                              # ... more params ...
                              phase1_metrics: Dict[str, Any]) -> str:
```

**AND at the end of the method (around line 1932, before the `except` block):**

**CHANGE FROM:**
```python
            self.logger.debug(f"📝 Saved squeeze alert for {symbol} to {filename}")

        except Exception as e:
```

**CHANGE TO:**
```python
            self.logger.debug(f"📝 Saved squeeze alert for {symbol} to {filename}")

            return filename  # ADD THIS LINE

        except Exception as e:
```

**ALSO update the exception handler to return None:**

**CHANGE FROM:**
```python
        except Exception as e:
            self.logger.error(f"❌ Error saving squeeze alert: {e}")
            import traceback
            traceback.print_exc()
```

**CHANGE TO:**
```python
        except Exception as e:
            self.logger.error(f"❌ Error saving squeeze alert: {e}")
            import traceback
            traceback.print_exc()
            return None  # ADD THIS LINE
```

**Verification:**
```bash
grep "def _save_squeeze_alert_sent" -A 2 cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py
# Should show "-> str:" in the return type
```

---

### STEP 5: Modify _report_squeeze() to Start Tracking

**Location:** Line 1634, after calling `_save_squeeze_alert_sent()`

**CHANGE FROM:**
```python
        # Save squeeze alert to JSON file for scanner display (includes Phase 1 metrics)
        self._save_squeeze_alert_sent(
            symbol, first_price, last_price, percent_change, timestamp, size, sent_users,
            premarket_high, pm_icon, pm_color, pm_percent_off,
            regular_hod, hod_icon, hod_color, hod_percent_off,
            vwap, vwap_icon, vwap_color,
            gain_percent, gain_icon, gain_color, gain_data_error,
            volume_surge_ratio, float_shares, float_rotation, float_rotation_percent,
            spread, spread_percent,
            phase1_metrics
        )
```

**CHANGE TO:**
```python
        # Save squeeze alert to JSON file for scanner display (includes Phase 1 metrics)
        filename = self._save_squeeze_alert_sent(
            symbol, first_price, last_price, percent_change, timestamp, size, sent_users,
            premarket_high, pm_icon, pm_color, pm_percent_off,
            regular_hod, hod_icon, hod_color, hod_percent_off,
            vwap, vwap_icon, vwap_color,
            gain_percent, gain_icon, gain_color, gain_data_error,
            volume_surge_ratio, float_shares, float_rotation, float_rotation_percent,
            spread, spread_percent,
            phase1_metrics
        )

        # Start outcome tracking for this squeeze (if enabled)
        if filename:  # Only start if alert was saved successfully
            self._start_outcome_tracking(
                symbol=symbol,
                squeeze_timestamp=timestamp,
                squeeze_price=last_price,
                alert_filename=filename
            )
```

**Verification:**
```bash
grep "_start_outcome_tracking" cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py | grep -v "def "
# Should show the method call in _report_squeeze
```

---

### STEP 6: Modify _handle_trade() to Check Intervals

**Location:** Line 1043-1051, after tracking volume/day prices, before calling `_detect_squeeze()`

**FIND THIS SECTION:**
```python
        # Track volume history for Phase 1 metrics
        self._track_volume_history(symbol, timestamp, size)

        # Track day prices (open, low) for Phase 1 metrics
        self._track_day_prices(symbol, price, timestamp)

        # Update latest SPY price if this is SPY
        if symbol == 'SPY':
            self.latest_spy_price = price
            self.latest_spy_timestamp = timestamp

        # Detect squeeze
        if (percent_change >= self.squeeze_percent and
```

**ADD AFTER THE SPY TRACKING, BEFORE "# Detect squeeze":**
```python
        # Track volume history for Phase 1 metrics
        self._track_volume_history(symbol, timestamp, size)

        # Track day prices (open, low) for Phase 1 metrics
        self._track_day_prices(symbol, price, timestamp)

        # Update latest SPY price if this is SPY
        if symbol == 'SPY':
            self.latest_spy_price = price
            self.latest_spy_timestamp = timestamp

        # Check outcome tracking intervals for this symbol
        self._check_outcome_intervals(symbol, timestamp, price, size)

        # Detect squeeze
        if (percent_change >= self.squeeze_percent and
```

**Verification:**
```bash
grep -B 3 -A 1 "_check_outcome_intervals" cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py
# Should show it's called in _handle_trade before _detect_squeeze
```

---

### STEP 7: Verify Imports

**Location:** Top of file (lines 1-30)

**Ensure these imports exist:**
```python
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
```

**If missing, add them to the imports section.**

**Verification:**
```bash
head -30 cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py | grep "from typing import"
# Should show: from typing import Dict, List, Optional, Any
```

---

## Post-Integration Checklist

### Syntax Validation

```bash
# Check for syntax errors
python3 -m py_compile cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py

# If successful, no output. If errors, fix them before proceeding.
```

### Service Restart

```bash
# Restart the service
sudo systemctl restart squeeze_alerts

# Check status
sudo systemctl status squeeze_alerts

# Should show: active (running)
```

### Monitor Logs

```bash
# Watch logs in real-time
journalctl -u squeeze_alerts -f

# Look for:
# - No error messages on startup
# - "📊 Started outcome tracking for SYMBOL" when squeeze detected
# - "✅ COMPLETE Outcome tracking for SYMBOL" after 10 minutes
```

---

## Testing Strategy

### Phase 1: Immediate (Service Start)

**Expected:**
- ✅ Service starts without errors
- ✅ No syntax errors in logs
- ✅ Trade processing continues normally
- ✅ Squeeze detection still works

**Check:**
```bash
journalctl -u squeeze_alerts -n 50 --no-pager
```

### Phase 2: First Squeeze (When Market Open)

**Expected:**
- ✅ Squeeze alert sent normally
- ✅ Log message: "📊 Started outcome tracking for SYMBOL (entry: $X.XX, duration: 10min)"
- ✅ Alert JSON file created in `data/squeeze_alerts_sent/`

**Check:**
```bash
# Find latest alert file
ls -lt data/squeeze_alerts_sent/ | head -1

# Verify it has basic structure (before outcomes added)
cat data/squeeze_alerts_sent/alert_SYMBOL_*.json | jq keys
```

### Phase 3: During Tracking (T+1 to T+10 minutes)

**Expected:**
- ✅ Verbose logs (if enabled): "📊 SYMBOL T+Nmin: $X.XX (+Y.YY%)"
- ✅ No errors in logs
- ✅ Service continues processing other trades normally

**Check:**
```bash
journalctl -u squeeze_alerts -f | grep "T+[0-9]min"
```

### Phase 4: Completion (After 10 minutes)

**Expected:**
- ✅ Log message: "✅ COMPLETE Outcome tracking for SYMBOL: max gain +X.XX% @ T+Nmin, final +Y.YY% @ T+10min"
- ✅ Alert JSON file updated with `outcome_tracking` section
- ✅ All 10 intervals recorded

**Check:**
```bash
# Find the alert file for the tracked squeeze
ALERT_FILE=$(ls -t data/squeeze_alerts_sent/ | head -1)

# Verify outcome_tracking section exists
cat "data/squeeze_alerts_sent/$ALERT_FILE" | jq .outcome_tracking

# Verify all intervals recorded
cat "data/squeeze_alerts_sent/$ALERT_FILE" | jq '.outcome_tracking.intervals | keys'
# Should show: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Verify summary statistics
cat "data/squeeze_alerts_sent/$ALERT_FILE" | jq '.outcome_tracking.summary'
```

---

## Validation Script

Save this as `test_outcome_tracking.py`:

```python
#!/usr/bin/env python3
"""Validate outcome tracking integration"""
import json
from pathlib import Path

alerts_dir = Path("data/squeeze_alerts_sent")

# Find most recent alert
alerts = sorted(alerts_dir.glob("alert_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

if not alerts:
    print("❌ No alert files found")
    exit(1)

latest = alerts[0]
print(f"Checking: {latest.name}")

with open(latest) as f:
    data = json.load(f)

# Check for outcome_tracking section
if 'outcome_tracking' not in data:
    print("⏳ Outcome tracking not yet complete (< 10 minutes)")
    exit(0)

outcome = data['outcome_tracking']

# Validate structure
assert outcome['enabled'] == True, "Tracking not enabled"
assert 'intervals' in outcome, "No intervals recorded"
assert 'summary' in outcome, "No summary"

# Check intervals
intervals = outcome['intervals']
interval_count = len(intervals)
print(f"✅ Intervals recorded: {interval_count}/10")

# Check summary
summary = outcome['summary']
print(f"✅ Max gain: {summary['max_gain_percent']:+.2f}% at T+{summary['max_gain_reached_at_minute']}min")
print(f"✅ Final gain: {summary['final_gain_percent']:+.2f}% at T+10min")
print(f"✅ Stop loss hit: {summary['reached_stop_loss']}")
print(f"✅ Tracking complete: {summary['tracking_completed']}")

if summary['tracking_completed']:
    print("\n🎉 Outcome tracking fully validated!")
else:
    print(f"\n⚠️  Partial tracking: only {interval_count}/10 intervals recorded")
```

**Run after first squeeze completes:**
```bash
python3 test_outcome_tracking.py
```

---

## Troubleshooting

### Issue: Service won't start

**Check:**
```bash
journalctl -u squeeze_alerts -n 100 --no-pager
```

**Common causes:**
- Syntax error → Fix Python syntax
- Import error → Verify typing imports
- Indentation error → Check method indentation

### Issue: "📊 Started outcome tracking" not appearing

**Check:**
1. Is `OUTCOME_TRACKING_ENABLED = True`?
2. Was alert saved successfully? (check for "📝 Saved squeeze alert" log)
3. Is `_start_outcome_tracking()` being called in `_report_squeeze()`?

### Issue: No interval logs appearing

**Check:**
1. Is `verbose = True` in configuration?
2. Is `_check_outcome_intervals()` being called in `_handle_trade()`?
3. Are trades coming in for the tracked symbol?

### Issue: Outcome section not added to JSON

**Check:**
1. Did 10 minutes elapse? (check logs for "✅ COMPLETE")
2. Is `_finalize_outcome_tracking()` being called?
3. Check for errors in `_update_alert_with_outcomes()` (file permissions?)

---

## Rollback Plan

If issues occur:

```bash
# Stop service
sudo systemctl stop squeeze_alerts

# Restore backup
cp squeeze_alerts.py.backup cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py

# Restart service
sudo systemctl start squeeze_alerts

# Verify
sudo systemctl status squeeze_alerts
```

---

## Success Criteria

Integration is successful when:

- [x] Service starts without errors
- [x] Syntax validation passes
- [x] First squeeze alert sent normally
- [x] "Started outcome tracking" log appears
- [x] Interval logs appear (if verbose)
- [x] "COMPLETE Outcome tracking" log appears after 10 minutes
- [x] Alert JSON updated with outcome_tracking section
- [x] All 10 intervals recorded
- [x] Summary statistics populated correctly

---

## Next Steps After Successful Integration

1. **Monitor for 1 market day** - Ensure stable operation
2. **Collect 10-20 alerts** - Build initial dataset
3. **Run validation script** - Verify all alerts have outcomes
4. **Begin analysis** - Use `ANALYSIS_CONSIDERATIONS.md` guide
5. **Tune constants if needed** - Adjust thresholds based on results

---

**Last Updated:** 2025-12-12
**Integration Time:** ~30-45 minutes
**Testing Time:** 1 market day
