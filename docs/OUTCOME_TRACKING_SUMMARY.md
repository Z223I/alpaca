# Outcome Tracking - Executive Summary

## The Critical Gap

**Current State:**
- ✅ Capture 44 fields at squeeze detection time
- ✅ Rich context: timing, volume, price levels, market conditions

**Missing:**
- ❌ What happens AFTER the squeeze
- ❌ Whether trades would be profitable
- ❌ Which setups actually work

**Impact:**
- Can only do descriptive statistics (what squeezes look like)
- **CANNOT** build predictive models (which squeezes to trade)

---

## The Solution

**Track each symbol for 10 minutes after squeeze detection:**

1. **Record snapshots at T+1, T+2, ..., T+10 minutes**
   - Price, volume, gain % at each minute

2. **Calculate outcome metrics:**
   - Max gain achieved (and when)
   - Max drawdown (and when)
   - Stop loss hit? (and when)
   - Target thresholds hit? (5%, 10%, 15%)
   - Final profitability at 10 minutes

3. **Save to alert JSON file**
   - Add `outcome_tracking` section
   - Keep all data together for analysis

---

## What This Enables

### Before Outcome Tracking
```python
# Can only describe squeezes
"ATPC had a 2.07% squeeze at 3:20 PM"
"Average squeeze is +2.5%"
"Most squeezes occur during power hour"
```

### After Outcome Tracking
```python
# Can predict and optimize
"Squeezes during power hour with volume_trend='increasing' have 73% win rate"
"Average max gain is +8.4% within 4 minutes"
"Only trade squeezes with window_volume_vs_1min_avg > 3.0x (68% profitable)"
"Expected value: +$142 per trade after filtering"
```

---

## Key Design Decisions

### 1. Configuration Constants

All thresholds stored as constants (easy to tune):

```python
OUTCOME_TRACKING_DURATION_MINUTES = 10
OUTCOME_TRACKING_INTERVAL_MINUTES = 1
OUTCOME_STOP_LOSS_PERCENT = 7.5
OUTCOME_TARGET_THRESHOLDS = [5.0, 10.0, 15.0]
OUTCOME_MAX_CONCURRENT_FOLLOWUPS = 100
```

### 2. Unique Tracking Keys

Each squeeze tracked independently:
- Key format: `"AAPL_2025-12-12_15:30:45"`
- Handles multiple squeezes per symbol
- No conflicts

### 3. Running Statistics

Updated on **every trade** during tracking period:
- Max price seen (not just at interval snapshots)
- Min price seen
- Stop loss detection (immediate, not just at intervals)
- Target achievement (immediate)

This captures rapid moves between 1-minute intervals.

### 4. Minimal Performance Impact

- Memory: ~200KB for 100 concurrent followups
- CPU: ~10 microseconds per trade
- Disk: One JSON update per squeeze (10 min)

---

## Target Variables for Machine Learning

After collecting outcome data, you can predict:

### Binary Classification
```python
# Will squeeze be profitable at T+10?
y = (final_gain_percent > 0)

# Will squeeze reach +5% within 10 minutes?
y = achieved_5pct

# Will squeeze hit stop loss?
y = reached_stop_loss
```

### Regression
```python
# Predict maximum gain
y = max_gain_percent

# Predict final gain at T+10
y = final_gain_percent

# Predict time to reach +5%
y = time_to_5pct_minutes
```

### Multi-class Classification
```python
# Outcome categories
y = {
    'big_win': max_gain_percent >= 10,
    'small_win': 0 < max_gain_percent < 10,
    'loss': max_gain_percent <= 0
}
```

---

## Analysis Examples

### Example 1: Win Rate by Market Session

```python
import pandas as pd

df = pd.read_json('squeeze_alerts/')

# Define "win" as profitable at 10 minutes
df['win'] = df['outcome_tracking'].apply(
    lambda x: x['summary']['profitable_at_10min']
)

# Group by market session
win_rates = df.groupby('phase1_analysis.market_session')['win'].mean()

print(win_rates)
# early:       0.58  (58% win rate)
# mid_day:     0.52
# power_hour:  0.73  ← BEST
# close:       0.45
```

**Actionable:** Only trade power hour squeezes.

### Example 2: Expected Value by Volume Surge

```python
# Calculate expected value
df['ev'] = df.apply(lambda row:
    row['outcome_tracking']['summary']['final_gain_percent'] *
    row['last_price'] * 100,  # Assume 100 shares
    axis=1
)

# Group by volume surge
high_volume = df[df['window_volume_vs_1min_avg'] > 3.0]
low_volume = df[df['window_volume_vs_1min_avg'] <= 3.0]

print(f"High volume EV: ${high_volume['ev'].mean():.2f}")
print(f"Low volume EV: ${low_volume['ev'].mean():.2f}")

# Output:
# High volume EV: $142.30  ← Trade these
# Low volume EV: -$23.50   ← Skip these
```

**Actionable:** Only trade squeezes with volume surge > 3.0x.

### Example 3: Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare features (Phase 1 fields)
features = [
    'percent_change',
    'window_volume_vs_1min_avg',
    'distance_from_vwap_percent',
    'distance_from_day_low_percent',
    'squeeze_number_today',
    'spy_percent_change_concurrent',
    # ... more
]

X = df[features]
y = df['outcome_tracking'].apply(
    lambda x: x['summary']['profitable_at_10min']
)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))

# Top predictive features:
# 1. window_volume_vs_1min_avg    0.18
# 2. distance_from_vwap_percent   0.14
# 3. squeeze_number_today         0.12
# 4. market_session               0.11
# 5. volume_trend                 0.09
```

**Actionable:** Focus on these 5 features for filtering.

---

## Implementation Priority

### Phase 1: Basic Outcome Tracking (This Design)
- Track 10 minutes at 1-minute intervals
- Calculate all outcome metrics
- Save to JSON

**Timeline:** 3-4 hours implementation + 1 market day validation

### Phase 2: Collect Data
- Run for 5-10 market days
- Collect 50-200 squeeze alerts with outcomes
- Validate data quality

**Timeline:** 1-2 weeks

### Phase 3: Initial Analysis
- Compute correlation between features and outcomes
- Identify high win-rate conditions
- Calculate expected values

**Timeline:** 2-3 days

### Phase 4: Filtering/Alerts
- Implement filters based on analysis
- Only send alerts for high-probability setups
- Track performance of filtered alerts

**Timeline:** 1-2 days

### Phase 5: ML Models (Optional)
- Build predictive models
- Ensemble methods
- Real-time probability scores

**Timeline:** 1 week

---

## Success Metrics

After implementation, you should be able to answer:

1. **What is the overall win rate?**
   - % of squeezes profitable at T+10

2. **What is the average max gain?**
   - Mean of max_gain_percent across all squeezes

3. **What is the stop loss hit rate?**
   - % of squeezes that hit -7.5%

4. **Which conditions produce best results?**
   - Win rate by market_session, volume_trend, etc.

5. **What is the expected value per trade?**
   - Average profit/loss accounting for win rate and gain magnitude

6. **Which features predict profitability?**
   - Feature importance from models

---

## Next Steps

1. **Review design document:** `OUTCOME_TRACKING_DESIGN.md`

2. **Implement outcome tracking:**
   - Add constants to class
   - Implement 7 new methods
   - Integrate into existing flow

3. **Test thoroughly:**
   - Unit tests for each method
   - Integration test with mock data
   - Live market validation

4. **Collect data:**
   - Run for 5-10 days
   - Monitor for errors
   - Validate JSON output

5. **Begin analysis:**
   - Export to DataFrame
   - Compute win rates
   - Calculate expected values
   - Build initial filters

---

**Bottom Line:** Without outcome tracking, you're flying blind. With it, you can build a systematically profitable trading strategy based on data, not hunches.
