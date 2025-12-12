# Squeeze Alert Outcome Tracking System - Design Document

## Overview

Tracks what happens to price/volume for 10 minutes after each squeeze alert to enable predictive modeling of squeeze profitability.

---

## Requirements

1. **Track each alerted symbol for 10 minutes** after squeeze detection
2. **Record data at 1-minute intervals** (T+1, T+2, ..., T+10)
3. **Calculate outcome metrics**: max gain, max drawdown, profitability, target achievement
4. **Save outcome data** to original alert JSON file
5. **Handle edge cases**: market close, multiple squeezes per symbol, service restart
6. **Use configuration constants** for all thresholds and intervals

---

## Architecture

### Integration Approach

**Integrate into existing `squeeze_alerts.py`** (not separate service):
- Leverage existing trade stream
- Simpler deployment (no new service)
- Access to all existing data structures

### Data Flow

```
Squeeze Detected
    ↓
_report_squeeze() ← Current function
    ↓
_start_outcome_tracking() ← NEW: Initialize tracking
    ↓
[Store in active_followups dict]
    ↓
_handle_trade() ← Existing function
    ↓
_check_outcome_intervals() ← NEW: Check if interval due
    ↓
_record_outcome_interval() ← NEW: Capture data point
    ↓
[Repeat until 10 minutes elapsed]
    ↓
_finalize_outcome_tracking() ← NEW: Calculate final metrics
    ↓
_update_alert_with_outcomes() ← NEW: Save to JSON
```

---

## Configuration Constants

Add to top of `SqueezeAlertsMonitor` class:

```python
# ===== OUTCOME TRACKING CONFIGURATION =====
# Enable/disable outcome tracking
OUTCOME_TRACKING_ENABLED = True

# Duration to track outcomes after squeeze (minutes)
OUTCOME_TRACKING_DURATION_MINUTES = 10

# Interval for recording snapshots (minutes)
OUTCOME_TRACKING_INTERVAL_MINUTES = 1

# Derived: list of intervals to track [1, 2, 3, ..., 10]
OUTCOME_TRACKING_INTERVALS = list(range(
    OUTCOME_TRACKING_INTERVAL_MINUTES,
    OUTCOME_TRACKING_DURATION_MINUTES + 1,
    OUTCOME_TRACKING_INTERVAL_MINUTES
))

# Stop loss threshold (percentage)
OUTCOME_STOP_LOSS_PERCENT = 7.5

# Target gain thresholds to track achievement (percentages)
OUTCOME_TARGET_THRESHOLDS = [5.0, 10.0, 15.0]

# Time tolerance for interval recording (seconds)
# Record interval if within this many seconds of target time
OUTCOME_INTERVAL_TOLERANCE_SECONDS = 30

# Maximum concurrent followups to track (memory limit)
OUTCOME_MAX_CONCURRENT_FOLLOWUPS = 100
```

---

## Data Structures

### Active Followups Dictionary

Add to `__init__`:

```python
# Outcome tracking: {symbol_timestamp_key: followup_data}
# Key format: "AAPL_2025-12-12_15:30:45" (unique per squeeze)
self.active_followups: Dict[str, Dict] = {}

# Cumulative volume tracking for followups
# Tracks total volume since squeeze start
self.followup_volume_tracking: Dict[str, int] = {}

# Cumulative trades tracking for followups
self.followup_trades_tracking: Dict[str, int] = {}
```

### Followup Data Structure

```python
followup_data = {
    'symbol': str,                    # 'AAPL'
    'squeeze_timestamp': datetime,     # When squeeze occurred
    'squeeze_price': float,            # last_price at squeeze (entry)
    'alert_filename': str,             # JSON file to update

    # Tracking window
    'start_time': datetime,            # squeeze_timestamp
    'end_time': datetime,              # start_time + 10 minutes

    # Next interval to record
    'next_interval': int,              # 1, 2, 3, ..., 10
    'next_interval_time': datetime,    # When to record next interval

    # Recorded data
    'intervals_recorded': List[int],   # [1, 2, 3, ...] completed
    'interval_data': {                 # Data at each interval
        1: {
            'timestamp': datetime,
            'price': float,
            'volume_since_squeeze': int,
            'trades_since_squeeze': int,
            'gain_percent': float
        },
        2: {...},
        # ... up to 10
    },

    # Running statistics (updated on every trade)
    'max_price_seen': float,
    'min_price_seen': float,
    'max_gain_percent': float,
    'max_gain_minute': int,            # Which interval had max gain
    'max_drawdown_percent': float,
    'max_drawdown_minute': int,

    # Threshold achievements (updated on every trade)
    'reached_stop_loss': bool,
    'stop_loss_minute': Optional[int],
    'stop_loss_price': float,          # Actual price when hit

    'reached_targets': {               # Which target thresholds hit
        5.0: {'reached': bool, 'minute': Optional[int], 'price': Optional[float]},
        10.0: {...},
        15.0: {...}
    },

    'profitable_snapshots': {          # Profitability at each interval
        1: Optional[bool],
        2: Optional[bool],
        # ... up to 10
    }
}
```

---

## Method Implementations

### 1. Start Tracking (`_start_outcome_tracking`)

Called from `_report_squeeze()` after saving alert.

```python
def _start_outcome_tracking(self, symbol: str, squeeze_timestamp: datetime,
                            squeeze_price: float, alert_filename: str) -> None:
    """
    Initialize outcome tracking for a squeeze alert.

    Args:
        symbol: Stock symbol
        squeeze_timestamp: When squeeze was detected
        squeeze_price: Price at squeeze detection (entry price)
        alert_filename: Name of alert JSON file to update later
    """
    if not self.OUTCOME_TRACKING_ENABLED:
        return

    # Check concurrent tracking limit
    if len(self.active_followups) >= self.OUTCOME_MAX_CONCURRENT_FOLLOWUPS:
        self.logger.warning(
            f"⚠️  Outcome tracking limit reached ({self.OUTCOME_MAX_CONCURRENT_FOLLOWUPS}). "
            f"Skipping tracking for {symbol}")
        return

    # Create unique key: symbol_timestamp
    key = f"{symbol}_{squeeze_timestamp.strftime('%Y-%m-%d_%H%M%S')}"

    # Calculate tracking window
    from datetime import timedelta
    end_time = squeeze_timestamp + timedelta(
        minutes=self.OUTCOME_TRACKING_DURATION_MINUTES
    )

    # First interval time
    first_interval_time = squeeze_timestamp + timedelta(
        minutes=self.OUTCOME_TRACKING_INTERVAL_MINUTES
    )

    # Initialize target tracking
    reached_targets = {}
    for threshold in self.OUTCOME_TARGET_THRESHOLDS:
        reached_targets[threshold] = {
            'reached': False,
            'minute': None,
            'price': None
        }

    # Initialize profitable snapshots
    profitable_snapshots = {i: None for i in self.OUTCOME_TRACKING_INTERVALS}

    # Create followup data structure
    self.active_followups[key] = {
        'symbol': symbol,
        'squeeze_timestamp': squeeze_timestamp,
        'squeeze_price': squeeze_price,
        'alert_filename': alert_filename,
        'start_time': squeeze_timestamp,
        'end_time': end_time,
        'next_interval': 1,
        'next_interval_time': first_interval_time,
        'intervals_recorded': [],
        'interval_data': {},
        'max_price_seen': squeeze_price,
        'min_price_seen': squeeze_price,
        'max_gain_percent': 0.0,
        'max_gain_minute': 0,
        'max_drawdown_percent': 0.0,
        'max_drawdown_minute': 0,
        'reached_stop_loss': False,
        'stop_loss_minute': None,
        'stop_loss_price': None,
        'reached_targets': reached_targets,
        'profitable_snapshots': profitable_snapshots
    }

    # Initialize volume/trades tracking
    self.followup_volume_tracking[key] = 0
    self.followup_trades_tracking[key] = 0

    self.logger.info(
        f"📊 Started outcome tracking for {symbol} "
        f"(entry: ${squeeze_price:.4f}, duration: {self.OUTCOME_TRACKING_DURATION_MINUTES}min)"
    )
```

### 2. Check Intervals (`_check_outcome_intervals`)

Called from `_handle_trade()` for each trade.

```python
def _check_outcome_intervals(self, symbol: str, timestamp: datetime,
                              price: float, size: int) -> None:
    """
    Check if any outcome intervals are due for recording.
    Called on every trade for symbols with active followups.

    Args:
        symbol: Stock symbol
        timestamp: Current trade timestamp
        price: Current trade price
        size: Current trade size
    """
    if not self.OUTCOME_TRACKING_ENABLED:
        return

    # Find all active followups for this symbol
    keys_to_check = [k for k in self.active_followups.keys() if k.startswith(f"{symbol}_")]

    for key in keys_to_check:
        followup = self.active_followups[key]

        # Update cumulative volume and trades
        self.followup_volume_tracking[key] += size
        self.followup_trades_tracking[key] += 1

        # Update running statistics (max/min price, gains, drawdowns)
        self._update_followup_statistics(key, price, timestamp)

        # Check if tracking period has ended
        if timestamp >= followup['end_time']:
            self._finalize_outcome_tracking(key)
            continue

        # Check if next interval is due
        from datetime import timedelta
        tolerance = timedelta(seconds=self.OUTCOME_INTERVAL_TOLERANCE_SECONDS)

        if timestamp >= (followup['next_interval_time'] - tolerance):
            # Record this interval
            self._record_outcome_interval(
                key,
                followup['next_interval'],
                timestamp,
                price,
                self.followup_volume_tracking[key],
                self.followup_trades_tracking[key]
            )

            # Advance to next interval
            next_interval_num = followup['next_interval'] + 1

            if next_interval_num <= self.OUTCOME_TRACKING_DURATION_MINUTES:
                # More intervals to track
                followup['next_interval'] = next_interval_num
                followup['next_interval_time'] = followup['start_time'] + timedelta(
                    minutes=next_interval_num * self.OUTCOME_TRACKING_INTERVAL_MINUTES
                )
            else:
                # All intervals recorded, finalize
                self._finalize_outcome_tracking(key)
```

### 3. Update Statistics (`_update_followup_statistics`)

Called on every trade during tracking period.

```python
def _update_followup_statistics(self, key: str, price: float, timestamp: datetime) -> None:
    """
    Update running statistics for an active followup.

    Args:
        key: Followup key
        price: Current price
        timestamp: Current timestamp
    """
    followup = self.active_followups[key]
    squeeze_price = followup['squeeze_price']

    # Calculate gain from squeeze entry
    gain_percent = ((price - squeeze_price) / squeeze_price) * 100

    # Update max price and gain
    if price > followup['max_price_seen']:
        followup['max_price_seen'] = price
        followup['max_gain_percent'] = gain_percent

        # Calculate which minute we're in
        elapsed = (timestamp - followup['start_time']).total_seconds() / 60
        followup['max_gain_minute'] = int(elapsed) + 1

    # Update min price and drawdown
    if price < followup['min_price_seen']:
        followup['min_price_seen'] = price
        followup['max_drawdown_percent'] = gain_percent

        elapsed = (timestamp - followup['start_time']).total_seconds() / 60
        followup['max_drawdown_minute'] = int(elapsed) + 1

    # Check stop loss
    stop_loss_threshold = -self.OUTCOME_STOP_LOSS_PERCENT
    if not followup['reached_stop_loss'] and gain_percent <= stop_loss_threshold:
        followup['reached_stop_loss'] = True
        followup['stop_loss_price'] = price

        elapsed = (timestamp - followup['start_time']).total_seconds() / 60
        followup['stop_loss_minute'] = int(elapsed) + 1

        self.logger.warning(
            f"🛑 {followup['symbol']} hit stop loss at ${price:.4f} "
            f"({gain_percent:.2f}%) after {followup['stop_loss_minute']} min"
        )

    # Check target thresholds
    for threshold in self.OUTCOME_TARGET_THRESHOLDS:
        target_info = followup['reached_targets'][threshold]

        if not target_info['reached'] and gain_percent >= threshold:
            target_info['reached'] = True
            target_info['price'] = price

            elapsed = (timestamp - followup['start_time']).total_seconds() / 60
            target_info['minute'] = int(elapsed) + 1

            self.logger.info(
                f"🎯 {followup['symbol']} hit +{threshold}% target at ${price:.4f} "
                f"after {target_info['minute']} min"
            )
```

### 4. Record Interval (`_record_outcome_interval`)

Called when an interval time is reached.

```python
def _record_outcome_interval(self, key: str, interval_num: int,
                              timestamp: datetime, price: float,
                              volume: int, trades: int) -> None:
    """
    Record data for a specific outcome interval.

    Args:
        key: Followup key
        interval_num: Interval number (1-10)
        timestamp: Current timestamp
        price: Current price
        volume: Cumulative volume since squeeze
        trades: Cumulative trades since squeeze
    """
    followup = self.active_followups[key]
    squeeze_price = followup['squeeze_price']

    # Calculate gain from entry
    gain_percent = ((price - squeeze_price) / squeeze_price) * 100

    # Record interval data
    followup['interval_data'][interval_num] = {
        'timestamp': timestamp.isoformat(),
        'price': float(price),
        'volume_since_squeeze': int(volume),
        'trades_since_squeeze': int(trades),
        'gain_percent': round(gain_percent, 2)
    }

    followup['intervals_recorded'].append(interval_num)

    # Record profitability snapshot
    followup['profitable_snapshots'][interval_num] = (price > squeeze_price)

    if self.verbose:
        self.logger.debug(
            f"📊 {followup['symbol']} T+{interval_num}min: "
            f"${price:.4f} ({gain_percent:+.2f}%)"
        )
```

### 5. Finalize Tracking (`_finalize_outcome_tracking`)

Called when 10 minutes elapsed or all intervals recorded.

```python
def _finalize_outcome_tracking(self, key: str) -> None:
    """
    Finalize outcome tracking and save results to alert JSON.

    Args:
        key: Followup key
    """
    followup = self.active_followups[key]

    # Build outcome summary
    summary = self._build_outcome_summary(followup)

    # Update alert JSON file with outcomes
    self._update_alert_with_outcomes(followup['alert_filename'], followup, summary)

    # Clean up
    del self.active_followups[key]
    del self.followup_volume_tracking[key]
    del self.followup_trades_tracking[key]

    self.logger.info(
        f"✅ Completed outcome tracking for {followup['symbol']} "
        f"(max gain: {summary['max_gain_percent']:+.2f}%, "
        f"final: {summary['final_gain_percent']:+.2f}%)"
    )
```

### 6. Build Summary (`_build_outcome_summary`)

```python
def _build_outcome_summary(self, followup: Dict) -> Dict:
    """
    Build summary statistics from followup data.

    Args:
        followup: Followup data dictionary

    Returns:
        Dictionary of summary statistics
    """
    # Get final interval data (T+10)
    final_interval = self.OUTCOME_TRACKING_DURATION_MINUTES
    final_data = followup['interval_data'].get(final_interval)

    if final_data:
        final_price = final_data['price']
        final_gain = final_data['gain_percent']
    else:
        # Use last recorded interval if T+10 not reached
        last_interval = max(followup['intervals_recorded']) if followup['intervals_recorded'] else 0
        if last_interval > 0:
            final_data = followup['interval_data'][last_interval]
            final_price = final_data['price']
            final_gain = final_data['gain_percent']
        else:
            final_price = followup['squeeze_price']
            final_gain = 0.0

    # Build target achievement summary
    targets_achieved = {}
    for threshold in self.OUTCOME_TARGET_THRESHOLDS:
        target_info = followup['reached_targets'][threshold]
        targets_achieved[f'achieved_{int(threshold)}pct'] = target_info['reached']
        targets_achieved[f'time_to_{int(threshold)}pct_minutes'] = target_info['minute']
        targets_achieved[f'price_at_{int(threshold)}pct'] = target_info['price']

    # Build profitability summary
    profitable_at = {}
    for interval in [1, 2, 5, 10]:  # Key intervals
        if interval in followup['profitable_snapshots']:
            profitable_at[f'profitable_at_{interval}min'] = followup['profitable_snapshots'][interval]

    summary = {
        # Max/min statistics
        'max_price': float(followup['max_price_seen']),
        'max_gain_percent': round(followup['max_gain_percent'], 2),
        'max_gain_reached_at_minute': followup['max_gain_minute'],

        'min_price': float(followup['min_price_seen']),
        'max_drawdown_percent': round(followup['max_drawdown_percent'], 2),
        'max_drawdown_reached_at_minute': followup['max_drawdown_minute'],

        # Final statistics
        'price_at_10min': float(final_price),
        'final_gain_percent': round(final_gain, 2),

        # Stop loss
        'reached_stop_loss': followup['reached_stop_loss'],
        'time_to_stop_loss_minutes': followup['stop_loss_minute'],
        'price_at_stop_loss': float(followup['stop_loss_price']) if followup['stop_loss_price'] else None,

        # Profitability snapshots
        **profitable_at,

        # Target achievements
        **targets_achieved,

        # Tracking metadata
        'intervals_recorded': followup['intervals_recorded'],
        'tracking_completed': len(followup['intervals_recorded']) == self.OUTCOME_TRACKING_DURATION_MINUTES
    }

    return summary
```

### 7. Update Alert JSON (`_update_alert_with_outcomes`)

```python
def _update_alert_with_outcomes(self, alert_filename: str,
                                  followup: Dict, summary: Dict) -> None:
    """
    Update the original alert JSON file with outcome tracking data.

    Args:
        alert_filename: Name of alert JSON file
        followup: Followup data dictionary
        summary: Summary statistics dictionary
    """
    try:
        filepath = self.squeeze_alerts_sent_dir / alert_filename

        # Read existing alert data
        with open(filepath, 'r') as f:
            alert_data = json.load(f)

        # Add outcome tracking section
        alert_data['outcome_tracking'] = {
            'enabled': True,
            'tracking_start': followup['start_time'].isoformat(),
            'tracking_end': followup['end_time'].isoformat(),
            'squeeze_entry_price': float(followup['squeeze_price']),
            'duration_minutes': self.OUTCOME_TRACKING_DURATION_MINUTES,
            'interval_minutes': self.OUTCOME_TRACKING_INTERVAL_MINUTES,

            # Interval snapshots
            'intervals': followup['interval_data'],

            # Summary statistics
            'summary': summary
        }

        # Write updated data back to file
        with open(filepath, 'w') as f:
            json.dump(alert_data, f, indent=2)

        self.logger.debug(f"📝 Updated {alert_filename} with outcome data")

    except Exception as e:
        self.logger.error(f"❌ Error updating alert with outcomes: {e}")
        import traceback
        traceback.print_exc()
```

---

## Integration Points

### Modify `_report_squeeze()`

Add after line 1643 (after `_save_squeeze_alert_sent`):

```python
# Start outcome tracking for this squeeze
self._start_outcome_tracking(
    symbol=symbol,
    squeeze_timestamp=timestamp,
    squeeze_price=last_price,
    alert_filename=filename  # Need to return this from _save_squeeze_alert_sent
)
```

**Note:** Modify `_save_squeeze_alert_sent()` to return the filename:

```python
def _save_squeeze_alert_sent(...) -> str:
    # ... existing code ...

    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(alert_json, f, indent=2)

    self.logger.debug(f"📝 Saved squeeze alert for {symbol} to {filename}")

    return filename  # ADD THIS LINE
```

### Modify `_handle_trade()`

Add after line 1051 (after SPY tracking):

```python
# Check outcome tracking intervals for this symbol
self._check_outcome_intervals(symbol, timestamp, price, size)
```

---

## Edge Cases

### 1. Market Close During Tracking

If market closes before 10 minutes elapsed:

```python
# In _check_outcome_intervals, check market hours
if timestamp.time() >= datetime.time(16, 0, 0):  # 4:00 PM ET
    # Market closed, finalize early
    self._finalize_outcome_tracking(key)
    continue
```

### 2. No Trades During Interval

If no trades occur exactly at interval time:
- Use `OUTCOME_INTERVAL_TOLERANCE_SECONDS = 30`
- Record interval when first trade arrives within tolerance
- If no trade within tolerance, use last known price (could enhance to check every second)

### 3. Multiple Squeezes Same Symbol

Handled by unique key format: `"AAPL_2025-12-12_15:30:45"`
- Each squeeze gets independent tracking
- Can have multiple concurrent followups for same symbol

### 4. Service Restart

Current design: lose active followups on restart
- Acceptable for Phase 1 (10 minutes is short)
- Phase 2 enhancement: persist active_followups to disk

### 5. Memory Limits

- `OUTCOME_MAX_CONCURRENT_FOLLOWUPS = 100` limit
- With 100 followups × ~2KB each = ~200KB memory usage
- Negligible overhead

---

## Testing Strategy

### Unit Tests

```python
# tests/test_outcome_tracking.py

def test_start_outcome_tracking():
    """Test initialization of outcome tracking"""
    pass

def test_record_interval():
    """Test recording interval data"""
    pass

def test_max_gain_detection():
    """Test max gain tracking on price spike"""
    pass

def test_stop_loss_detection():
    """Test stop loss hit detection"""
    pass

def test_target_achievement():
    """Test target threshold achievement"""
    pass

def test_finalize_summary():
    """Test summary calculation"""
    pass

def test_json_update():
    """Test alert JSON file update"""
    pass
```

### Integration Tests

1. **Mock scenario:** Squeeze at T=0, price increases linearly for 10 min
   - Verify all 10 intervals recorded
   - Verify max gain at minute 10
   - Verify no stop loss hit

2. **Mock scenario:** Squeeze at T=0, price spikes at T+3, drops at T+7
   - Verify max gain at minute 3
   - Verify max drawdown at minute 7

3. **Mock scenario:** Squeeze at T=0, hits stop loss at T+4
   - Verify stop loss detection
   - Verify tracking continues to T+10

---

## Performance Impact

### Memory Usage

Per active followup:
- Followup dict: ~1-2 KB
- 100 concurrent followups: ~200 KB
- **Negligible impact**

### CPU Usage

Per trade for tracked symbols:
- Statistics update: O(1) operations
- Interval check: O(1) comparison
- ~10 microseconds per trade
- **Negligible impact**

### Disk I/O

Per completed followup:
- One JSON file update (read + write)
- File size: ~5-10 KB
- Once per 10 minutes per squeeze
- **Minimal impact**

---

## Phase 2 Enhancements

Once basic outcome tracking is validated:

1. **Higher resolution tracking:**
   - Record every 30 seconds instead of 1 minute
   - Better capture of rapid moves

2. **Volume profile tracking:**
   - Track volume distribution across price levels
   - Identify support/resistance

3. **Tape reading features:**
   - Aggressor side (buy vs sell pressure)
   - Large print detection during outcome period
   - Order flow imbalance

4. **Comparative metrics:**
   - SPY performance during same 10 minutes
   - Sector ETF performance
   - Relative strength calculation

5. **Persistence for restart:**
   - Save active_followups to disk
   - Resume tracking after service restart

---

## Expected Output Format

After implementation, each alert JSON will include:

```json
{
  "symbol": "ATPC",
  "timestamp": "2025-12-11T15:20:38.869008-05:00",
  "first_price": 0.1401,
  "last_price": 0.143,
  "percent_change": 2.07,

  "phase1_analysis": { ... },

  "outcome_tracking": {
    "enabled": true,
    "tracking_start": "2025-12-11T15:20:38.869008-05:00",
    "tracking_end": "2025-12-11T15:30:38.869008-05:00",
    "squeeze_entry_price": 0.143,
    "duration_minutes": 10,
    "interval_minutes": 1,

    "intervals": {
      "1": {
        "timestamp": "2025-12-11T15:21:38-05:00",
        "price": 0.145,
        "volume_since_squeeze": 120000,
        "trades_since_squeeze": 45,
        "gain_percent": 1.40
      },
      "2": { ... },
      ...
      "10": {
        "timestamp": "2025-12-11T15:30:38-05:00",
        "price": 0.148,
        "volume_since_squeeze": 850000,
        "trades_since_squeeze": 312,
        "gain_percent": 3.50
      }
    },

    "summary": {
      "max_price": 0.152,
      "max_gain_percent": 6.29,
      "max_gain_reached_at_minute": 4,

      "min_price": 0.142,
      "max_drawdown_percent": -0.70,
      "max_drawdown_reached_at_minute": 8,

      "price_at_10min": 0.148,
      "final_gain_percent": 3.50,

      "reached_stop_loss": false,
      "time_to_stop_loss_minutes": null,
      "price_at_stop_loss": null,

      "profitable_at_1min": true,
      "profitable_at_2min": true,
      "profitable_at_5min": true,
      "profitable_at_10min": true,

      "achieved_5pct": true,
      "time_to_5pct_minutes": 3,
      "price_at_5pct": 0.150,

      "achieved_10pct": false,
      "time_to_10pct_minutes": null,
      "price_at_10pct": null,

      "achieved_15pct": false,
      "time_to_15pct_minutes": null,
      "price_at_15pct": null,

      "intervals_recorded": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "tracking_completed": true
    }
  }
}
```

---

## Implementation Checklist

- [ ] Add configuration constants to class
- [ ] Add data structures to `__init__`
- [ ] Implement `_start_outcome_tracking()`
- [ ] Implement `_check_outcome_intervals()`
- [ ] Implement `_update_followup_statistics()`
- [ ] Implement `_record_outcome_interval()`
- [ ] Implement `_build_outcome_summary()`
- [ ] Implement `_finalize_outcome_tracking()`
- [ ] Implement `_update_alert_with_outcomes()`
- [ ] Modify `_save_squeeze_alert_sent()` to return filename
- [ ] Modify `_report_squeeze()` to call `_start_outcome_tracking()`
- [ ] Modify `_handle_trade()` to call `_check_outcome_intervals()`
- [ ] Add market close handling
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Test with live market data
- [ ] Validate JSON output format
- [ ] Update documentation

---

**Status:** Design Complete - Ready for Implementation

**Estimated Lines of Code:** ~400-500 lines

**Estimated Implementation Time:** 3-4 hours

**Testing Time:** 1-2 hours (plus 1 market day for live validation)
