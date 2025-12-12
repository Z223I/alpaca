# Interval Price Range Tracking - Feature Documentation

**Date:** 2025-12-12
**Feature:** Track low/high prices with timestamps for each outcome tracking interval
**Files Modified:** `cgi-bin/molecules/alpaca_molecules/squeeze_alerts.py`

---

## Overview

The squeeze alerts outcome tracking now captures **interval price ranges** in addition to snapshot prices. For each tracking interval (10s, 20s, 30s, 60s, etc.), the system records:

- **Snapshot price** at the interval time (existing)
- **Interval low**: Lowest price during that interval period (NEW)
- **Interval low timestamp**: When the low occurred (NEW)
- **Interval high**: Highest price during that interval period (NEW)
- **Interval high timestamp**: When the high occurred (NEW)
- **Price range statistics**: Range in dollars and percent (NEW)

---

## Why This Matters

### 1. **Better Trading Simulation**
- **Best case**: Use interval high for optimal exit
- **Worst case**: Use interval low for realistic stop-loss testing
- **Current**: More accurate than single snapshot price

### 2. **Volatility Analysis**
- Identify which intervals have highest volatility
- Understand price action patterns
- Optimize entry/exit timing

### 3. **Opportunity Detection**
- See if price spiked higher before settling
- Identify "wick" patterns (high touched but didn't hold)
- Find optimal scaling-out points

### 4. **Model Enhancement**
- Train models to predict interval volatility
- Use high/low as additional target variables
- Improve risk/reward calculations

---

## JSON Structure (Before vs After)

### Before (Single Snapshot)

```json
"intervals": {
  "30": {
    "timestamp": "2025-12-12T09:29:17.229436-05:00",
    "price": 1.21,
    "volume_since_squeeze": 345488,
    "trades_since_squeeze": 873,
    "gain_percent": 0.83
  }
}
```

### After (With Price Range)

```json
"intervals": {
  "30": {
    "timestamp": "2025-12-12T09:29:17.229436-05:00",
    "price": 1.21,
    "volume_since_squeeze": 345488,
    "trades_since_squeeze": 873,
    "gain_percent": 0.83,

    "interval_low": 1.19,
    "interval_low_timestamp": "2025-12-12T09:29:07.256458-05:00",
    "interval_low_gain_percent": -0.83,

    "interval_high": 1.23,
    "interval_high_timestamp": "2025-12-12T09:29:15.371377-05:00",
    "interval_high_gain_percent": 2.5
  }
}
```

### Interpretation

**For the 30-second interval (20s → 30s):**
- At 30s mark: price = $1.21 (+0.83%)
- During interval: low = $1.19 at 27s mark (-0.83%)
- During interval: high = $1.23 at 28s mark (+2.5%)
- Price range: $0.04 (3.36% volatility)

**Trading insight:**
- Price spiked to +2.5% before settling at +0.83%
- Using snapshot alone would miss the $1.23 opportunity
- Could have scaled out 50% at $1.23, kept 50% at $1.21

---

## Data Fields

### Per-Interval Fields (in `intervals` section)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `interval_low` | float | Lowest price during interval period | 1.19 |
| `interval_low_timestamp` | ISO datetime | When low occurred | "2025-12-12T09:29:07..." |
| `interval_low_gain_percent` | float | Gain % of low vs entry | -0.83 |
| `interval_high` | float | Highest price during interval period | 1.23 |
| `interval_high_timestamp` | ISO datetime | When high occurred | "2025-12-12T09:29:15..." |
| `interval_high_gain_percent` | float | Gain % of high vs entry | 2.5 |

### Additional Section (in `outcome_tracking`)

The system also stores a dedicated `interval_price_ranges` section for easy access:

```json
"outcome_tracking": {
  "interval_price_ranges": {
    "30": {
      "low": 1.19,
      "low_timestamp": "2025-12-12T09:29:07.256458-05:00",
      "low_gain_percent": -0.83,
      "high": 1.23,
      "high_timestamp": "2025-12-12T09:29:15.371377-05:00",
      "high_gain_percent": 2.5,
      "range": 0.04,
      "range_percent": 3.36
    }
  }
}
```

---

## Implementation Details

### Code Changes

**1. Initialize Interval Tracking (`_start_outcome_tracking`):**
```python
# Added to followup data structure:
'interval_price_ranges': {},  # Stores low/high for each interval
'current_interval_low': squeeze_price,
'current_interval_low_time': squeeze_timestamp,
'current_interval_high': squeeze_price,
'current_interval_high_time': squeeze_timestamp,
'current_interval_start_time': squeeze_timestamp,
```

**2. Track Running Low/High (`_update_followup_statistics`):**
```python
# Called on EVERY trade to capture intra-interval extremes
if price < followup['current_interval_low']:
    followup['current_interval_low'] = price
    followup['current_interval_low_time'] = timestamp

if price > followup['current_interval_high']:
    followup['current_interval_high'] = price
    followup['current_interval_high_time'] = timestamp
```

**3. Save Interval Data (`_record_outcome_interval`):**
```python
# When interval snapshot occurs, save accumulated low/high
followup['interval_data'][interval_seconds] = {
    # ... existing fields ...
    'interval_low': interval_low,
    'interval_low_timestamp': low_timestamp,
    'interval_low_gain_percent': low_gain,
    'interval_high': interval_high,
    'interval_high_timestamp': high_timestamp,
    'interval_high_gain_percent': high_gain
}

# Reset for next interval
followup['current_interval_low'] = price
followup['current_interval_high'] = price
```

### Interval Periods

Each interval captures the price range from the **previous interval** to **current interval**:

| Interval | Period Covered | Example |
|----------|----------------|---------|
| 10s | 0s → 10s | From squeeze to 10s |
| 20s | 10s → 20s | From 10s mark to 20s mark |
| 30s | 20s → 30s | From 20s mark to 30s mark |
| 60s | 30s → 60s | From 30s mark to 60s mark |
| 120s | 60s → 120s | From 60s mark to 120s mark |
| etc. | ... | ... |

---

## Usage Examples

### Example 1: Best-Case Trading Simulation

```python
import json

# Load alert with outcome tracking
with open('alert_AAPL_2025-12-12_093045.json') as f:
    alert = json.load(f)

# Simulate best-case scenario (use interval highs)
entry_price = alert['outcome_tracking']['squeeze_entry_price']
total_pnl = 0

for interval_sec, data in alert['outcome_tracking']['intervals'].items():
    interval_high = data['interval_high']
    gain_pct = ((interval_high - entry_price) / entry_price) * 100

    if gain_pct >= 5.0:  # Hit 5% target
        print(f"Best case: Hit 5% at {data['interval_high_timestamp']}")
        print(f"  High: ${interval_high} (+{gain_pct:.2f}%)")
        total_pnl = gain_pct
        break

print(f"Best-case P&L: {total_pnl:.2f}%")
```

### Example 2: Worst-Case Stop Loss Check

```python
# Check if any interval low would have hit stop loss
stop_loss_pct = -7.5
hit_stop_loss = False

for interval_sec, data in alert['outcome_tracking']['intervals'].items():
    if data['interval_low_gain_percent'] <= stop_loss_pct:
        print(f"⚠️  Stop loss hit at interval {interval_sec}s")
        print(f"  Low: ${data['interval_low']} ({data['interval_low_gain_percent']:.2f}%)")
        print(f"  Time: {data['interval_low_timestamp']}")
        hit_stop_loss = True
        break

if not hit_stop_loss:
    print("✓ Stop loss avoided")
```

### Example 3: Volatility Analysis

```python
# Analyze which intervals are most volatile
volatility_by_interval = {}

for interval_sec, data in alert['outcome_tracking']['intervals'].items():
    low = data['interval_low']
    high = data['interval_high']
    range_pct = ((high - low) / low) * 100 if low > 0 else 0

    volatility_by_interval[interval_sec] = {
        'range_pct': range_pct,
        'low': low,
        'high': high
    }

# Sort by volatility
sorted_volatility = sorted(volatility_by_interval.items(),
                          key=lambda x: x[1]['range_pct'],
                          reverse=True)

print("Most volatile intervals:")
for interval_sec, stats in sorted_volatility[:5]:
    print(f"  {interval_sec}s: {stats['range_pct']:.2f}% "
          f"(${stats['low']} - ${stats['high']})")
```

### Example 4: Opportunity Detection

```python
# Find intervals where high was significantly above close
squeeze_price = alert['outcome_tracking']['squeeze_entry_price']

for interval_sec, data in alert['outcome_tracking']['intervals'].items():
    interval_close = data['price']
    interval_high = data['interval_high']

    # Check if high was >1% above close
    spike_pct = ((interval_high - interval_close) / interval_close) * 100

    if spike_pct > 1.0:
        print(f"🎯 Interval {interval_sec}s: Spike detected!")
        print(f"  High: ${interval_high} at {data['interval_high_timestamp']}")
        print(f"  Close: ${interval_close}")
        print(f"  Spike: {spike_pct:.2f}% above interval close")
        print(f"  Opportunity: Could have scaled out at ${interval_high}")
```

---

## Predictive Modeling Applications

### 1. Multi-Target Prediction

Train models to predict interval outcomes:

```python
# Targets for 30s interval
targets = {
    'will_hit_2pct': interval_high_gain >= 2.0,
    'will_drop_below_entry': interval_low_gain < 0,
    'volatility': range_percent,
    'best_exit': interval_high_gain,
    'worst_case': interval_low_gain
}
```

### 2. Risk/Reward Optimization

```python
# Calculate realistic R:R using interval data
entry = squeeze_price
best_case = max(d['interval_high'] for d in intervals.values())
worst_case = min(d['interval_low'] for d in intervals.values())

potential_gain = ((best_case - entry) / entry) * 100
potential_loss = ((worst_case - entry) / entry) * 100

risk_reward_ratio = abs(potential_gain / potential_loss)
```

### 3. Exit Strategy Optimization

```python
# Find optimal exit points using interval highs
exit_strategy = []

for interval_sec, data in sorted(intervals.items()):
    if data['interval_high_gain_percent'] >= 2.0:
        exit_strategy.append({
            'time': interval_sec,
            'price': data['interval_high'],
            'gain': data['interval_high_gain_percent'],
            'action': 'Scale out 50%'
        })

        if data['interval_high_gain_percent'] >= 5.0:
            exit_strategy[-1]['action'] = 'Take full profit'
            break
```

---

## Performance Impact

**Minimal overhead:**
- Tracking adds ~6 fields per interval (negligible memory)
- Calculations done on existing trade stream (no extra API calls)
- No impact on alert latency

**Storage increase:**
- ~100 bytes per interval
- 17 intervals × 100 bytes = 1.7KB per alert
- Acceptable for the additional insight gained

---

## Backward Compatibility

✅ **Fully backward compatible**

- Existing code reading JSON files continues to work
- New fields are additive (don't break existing parsers)
- Prediction scripts can ignore new fields if not needed
- Old JSON files without interval ranges still valid

---

## Testing

### Verification Steps

1. **Check JSON output:**
   ```bash
   cat historical_data/*/squeeze_alerts_sent/alert_*.json | grep interval_low
   ```

2. **Verify timestamp consistency:**
   ```python
   # Low timestamp should be <= High timestamp
   assert data['interval_low_timestamp'] <= data['interval_high_timestamp']
   ```

3. **Validate price bounds:**
   ```python
   # Low <= Close <= High for each interval
   assert data['interval_low'] <= data['price'] <= data['interval_high']
   ```

4. **Check range calculations:**
   ```python
   # Range should match high - low
   expected_range = data['interval_high'] - data['interval_low']
   assert abs(data['range'] - expected_range) < 0.01
   ```

---

## Future Enhancements

### Potential Additions

1. **Volume-weighted price** within interval
2. **Number of trades** that hit low/high
3. **Time spent** near low vs high (distribution)
4. **Bid-ask spread** at low/high moments
5. **Market impact** analysis at extremes

---

## Summary

**What Changed:**
- ✅ Added interval low/high price tracking
- ✅ Added timestamps for when extremes occurred
- ✅ Added range statistics (dollars and percent)
- ✅ Backward compatible with existing code

**Benefits:**
- Better trading simulations (best/worst case)
- Volatility analysis per interval
- Optimal entry/exit point identification
- Enhanced predictive modeling targets

**Files Modified:**
- `squeeze_alerts.py:1666-1670` - Initialize tracking fields
- `squeeze_alerts.py:1815-1822` - Update current interval low/high
- `squeeze_alerts.py:1901-1944` - Save interval data and reset

**Documentation:**
- This file: Complete feature documentation
- Inline comments: Updated in squeeze_alerts.py

---

**Feature Complete:** 2025-12-12
**Status:** ✅ Production Ready
**Next Steps:** Monitor new data in JSON files, update prediction models to use interval ranges
