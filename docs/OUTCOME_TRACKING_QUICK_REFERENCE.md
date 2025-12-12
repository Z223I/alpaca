# Outcome Tracking - Quick Reference

## What It Does

Tracks price/volume for **10 minutes** after each squeeze alert to enable predictive modeling.

---

## Files

| File | Purpose | Size |
|------|---------|------|
| `OUTCOME_TRACKING_SUMMARY.md` | Why it's needed, business value | 7.5 KB |
| `OUTCOME_TRACKING_DESIGN.md` | Complete technical specification | 27 KB |
| `OUTCOME_TRACKING_IMPLEMENTATION.py` | **All code (copy from here)** | 27 KB |
| `OUTCOME_TRACKING_INTEGRATION_GUIDE.md` | Step-by-step integration instructions | 15 KB |

---

## Quick Start

### 1. Review (5 minutes)
```bash
# Read the summary
cat docs/OUTCOME_TRACKING_SUMMARY.md

# Review the implementation code
cat docs/OUTCOME_TRACKING_IMPLEMENTATION.py
```

### 2. Backup (1 minute)
```bash
# Backup current file
cp cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py \
   cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py.backup
```

### 3. Integrate (30-45 minutes)

**Follow:** `OUTCOME_TRACKING_INTEGRATION_GUIDE.md`

**Summary:**
1. Add 9 configuration constants
2. Add 3 data structures to `__init__()`
3. Add 7 new methods (copy from `OUTCOME_TRACKING_IMPLEMENTATION.py`)
4. Modify `_save_squeeze_alert_sent()` to return filename
5. Modify `_report_squeeze()` to start tracking
6. Modify `_handle_trade()` to check intervals
7. Verify imports

### 4. Test (1 market day)
```bash
# Syntax check
python3 -m py_compile cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py

# Restart service
sudo systemctl restart squeeze_alerts

# Monitor logs
journalctl -u squeeze_alerts -f

# Look for:
# - "📊 Started outcome tracking for SYMBOL"
# - "✅ COMPLETE Outcome tracking for SYMBOL"
```

---

## Configuration Constants

```python
OUTCOME_TRACKING_ENABLED = True                    # Enable/disable
OUTCOME_TRACKING_DURATION_MINUTES = 10             # Track for N minutes
OUTCOME_TRACKING_INTERVAL_MINUTES = 1              # Snapshot every N minutes
OUTCOME_INTERVAL_TOLERANCE_SECONDS = 30            # Window for recording
OUTCOME_MAX_CONCURRENT_FOLLOWUPS = 100             # Memory limit
OUTCOME_STOP_LOSS_PERCENT = 7.5                    # Stop loss threshold
OUTCOME_TARGET_THRESHOLDS = [5.0, 10.0, 15.0]      # Target thresholds
```

**Tune these after testing!**

---

## Output Format

Each alert JSON gets a new section:

```json
{
  "symbol": "ATPC",
  "phase1_analysis": { ... },

  "outcome_tracking": {
    "squeeze_entry_price": 0.143,

    "intervals": {
      "1": {"price": 0.145, "gain_percent": 1.40, ...},
      "2": {"price": 0.147, "gain_percent": 2.80, ...},
      ...
      "10": {"price": 0.148, "gain_percent": 3.50, ...}
    },

    "summary": {
      "max_gain_percent": 6.29,
      "max_gain_reached_at_minute": 4,
      "final_gain_percent": 3.50,
      "profitable_at_10min": true,
      "achieved_5pct": true,
      "reached_stop_loss": false
    }
  }
}
```

---

## Key Methods

### 1. `_start_outcome_tracking()`
- Called when squeeze detected
- Initializes tracking for 10 minutes

### 2. `_check_outcome_intervals()`
- Called on every trade
- Records data when interval time reached

### 3. `_update_followup_statistics()`
- Called on every trade
- Updates max gain, min price, stop loss, targets

### 4. `_finalize_outcome_tracking()`
- Called after 10 minutes
- Calculates summary and saves to JSON

---

## How Price Retrieval Works

**Uses existing trade stream (zero additional API calls):**

```
Squeeze at 3:20:00 PM, entry $150.00
    ↓
Every trade comes in via _handle_trade()
    ↓
_check_outcome_intervals() checks: "Is it time for T+1?"
    ↓
If within 30s of 3:21:00, record that trade's price
    ↓
Repeat for T+2, T+3, ..., T+10
    ↓
After T+10, finalize and save outcomes
```

---

## Validation Commands

### Check Service Status
```bash
sudo systemctl status squeeze_alerts
journalctl -u squeeze_alerts -n 50 --no-pager
```

### Check Latest Alert
```bash
# Find latest alert file
ls -lt data/squeeze_alerts_sent/ | head -1

# View outcome section
jq .outcome_tracking data/squeeze_alerts_sent/alert_*.json | tail -1
```

### Check All Intervals Recorded
```bash
# Should show: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
jq '.outcome_tracking.intervals | keys' data/squeeze_alerts_sent/alert_*.json | tail -1
```

### Check Summary Stats
```bash
jq '.outcome_tracking.summary' data/squeeze_alerts_sent/alert_*.json | tail -1
```

---

## Performance Impact

| Metric | Value | Impact |
|--------|-------|--------|
| Memory | ~200 KB (100 followups) | Negligible |
| CPU | ~10 μs per trade | Negligible |
| Disk I/O | 1 JSON update per squeeze | Minimal |

**Safe for production.**

---

## Troubleshooting

### Service won't start
```bash
# Check for syntax errors
python3 -m py_compile cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py
journalctl -u squeeze_alerts -n 100 --no-pager
```

### No "Started tracking" log
- Check: `OUTCOME_TRACKING_ENABLED = True`
- Check: `_start_outcome_tracking()` called in `_report_squeeze()`
- Check: Alert saved successfully (filename returned)

### No intervals recorded
- Check: `_check_outcome_intervals()` called in `_handle_trade()`
- Check: Trades coming in (liquid stock)
- Wait full 10 minutes

### Outcome section missing from JSON
- Check: 10 minutes elapsed?
- Check: "✅ COMPLETE" log appeared?
- Check: File permissions on data/squeeze_alerts_sent/

---

## Success Checklist

After integration:

- [ ] Service starts without errors
- [ ] First squeeze alert sent normally
- [ ] Log: "📊 Started outcome tracking"
- [ ] Log: "✅ COMPLETE Outcome tracking" (after 10 min)
- [ ] Alert JSON has `outcome_tracking` section
- [ ] All 10 intervals recorded: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- [ ] Summary statistics populated

---

## Analysis Examples

### Win Rate by Session
```python
import pandas as pd
import json
from pathlib import Path

alerts = []
for f in Path('data/squeeze_alerts_sent').glob('alert_*.json'):
    with open(f) as file:
        alerts.append(json.load(file))

df = pd.DataFrame(alerts)

# Extract outcome
df['win'] = df['outcome_tracking'].apply(
    lambda x: x['summary']['profitable_at_10min'] if x else None
)

# Group by market session
win_rate = df.groupby('phase1_analysis.market_session')['win'].mean()
print(win_rate)
```

### Expected Value by Volume
```python
df['max_gain'] = df['outcome_tracking'].apply(
    lambda x: x['summary']['max_gain_percent'] if x else None
)

high_vol = df[df['window_volume_vs_1min_avg'] > 3.0]
low_vol = df[df['window_volume_vs_1min_avg'] <= 3.0]

print(f"High volume avg gain: {high_vol['max_gain'].mean():.2f}%")
print(f"Low volume avg gain: {low_vol['max_gain'].mean():.2f}%")
```

---

## Target Variables for ML

After collecting data, you can predict:

```python
# Binary classification
y = df['outcome_tracking'].apply(lambda x: x['summary']['profitable_at_10min'])

# Regression
y = df['outcome_tracking'].apply(lambda x: x['summary']['max_gain_percent'])

# Multi-class
df['outcome_class'] = df['outcome_tracking'].apply(
    lambda x: 'big_win' if x['summary']['max_gain_percent'] >= 10
              else 'small_win' if x['summary']['max_gain_percent'] > 0
              else 'loss'
)
```

---

## Next Steps

1. **Integrate** (use INTEGRATION_GUIDE.md)
2. **Test** for 1 market day
3. **Collect** 50-100 alerts with outcomes
4. **Analyze** using ANALYSIS_CONSIDERATIONS.md
5. **Build filters** based on findings
6. **Iterate** on thresholds and features

---

## Support

- **Design:** `OUTCOME_TRACKING_DESIGN.md`
- **Implementation:** `OUTCOME_TRACKING_IMPLEMENTATION.py`
- **Integration:** `OUTCOME_TRACKING_INTEGRATION_GUIDE.md`
- **Business Case:** `OUTCOME_TRACKING_SUMMARY.md`
- **Statistical Analysis:** `ANALYSIS_CONSIDERATIONS.md`

---

**Version:** 1.0
**Date:** 2025-12-12
**Status:** Ready for Integration
