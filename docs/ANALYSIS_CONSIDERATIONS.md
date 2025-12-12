# Statistical Independence Analysis of Squeeze Alert Fields

**SHORT ANSWER: NO - Many fields are NOT statistically independent. You have significant multicollinearity issues.**

## Complete Field Inventory

Based on `squeeze_alerts.py:1822-1927`, each alert contains:

**Main Fields (24):**
- symbol, timestamp, first_price, last_price, percent_change, size, window_trades
- day_gain, vwap, premarket_high, regular_hours_hod
- volume_surge_ratio, float_shares, float_rotation, float_rotation_percent
- spread, spread_percent
- Status fields (icons/colors for gain, vwap, pm_high, hod)

**Phase 1 Analysis Fields (20):**
- Timing: time_since_market_open_minutes, hour_of_day, market_session, squeeze_number_today, minutes_since_last_squeeze
- Volume: window_volume, window_volume_vs_1min_avg, window_volume_vs_5min_avg, volume_trend
- Price Context: distance_from_prev_close_percent, distance_from_vwap_percent, distance_from_day_low_percent, distance_from_open_percent
- Risk: estimated_stop_loss_price, stop_loss_distance_percent, potential_target_price, risk_reward_ratio
- Market: spy_percent_change_day, spy_percent_change_concurrent

---

## Dependency Groups (NOT Independent)

### 🔴 **GROUP 1: Perfect Mathematical Dependencies**

These fields are **deterministic functions** of each other:

```python
# Timing (lines 1183-1204)
hour_of_day = timestamp.hour
time_since_market_open_minutes = f(timestamp)
market_session = f(hour_of_day)  # Deterministic mapping
```
**Correlation: ρ ≈ 1.0**
- Knowing `hour_of_day` tells you `market_session` exactly
- Use ONLY ONE: Recommend `market_session` (most meaningful)

```python
# Float rotation (lines 1508-1511)
float_rotation = total_volume / float_shares
float_rotation_percent = float_rotation * 100  # Perfect correlation
```
**Correlation: ρ = 1.0**
- These are identical information, just scaled
- Use ONLY `float_rotation` (drop float_rotation_percent)

```python
# Spread (lines 1449-1454)
spread = ask_price - bid_price
spread_percent = (spread / last_price) * 100
```
**Correlation: ρ ≈ 0.95+**
- Nearly perfect correlation adjusted for price level
- Use ONLY `spread_percent` (normalized, more meaningful)

```python
# Stop loss (lines 1317-1321)
stop_loss_distance_percent = 7.5  # CONSTANT
estimated_stop_loss_price = last_price * 0.925  # Function of last_price
```
**Correlation: stop_loss_distance_percent has ZERO variance** (constant)
- Drop `stop_loss_distance_percent` entirely (no information)
- `estimated_stop_loss_price` is just 0.925 × last_price (redundant with last_price)

---

### 🟡 **GROUP 2: Strong Dependencies (ρ > 0.7)**

```python
# Day gain is duplicated (lines 1281-1282, 1890)
day_gain = ((last_price - prev_close) / prev_close) * 100
distance_from_prev_close_percent = same calculation
```
**These are THE SAME FIELD with different names!**
- Drop one entirely (keep `day_gain`)

```python
# Volume metrics (lines 1221-1272)
window_volume_vs_1min_avg = window_volume / avg_1min
window_volume_vs_5min_avg = window_volume / avg_5min
volume_surge_ratio = similar concept, different window
```
**Correlation: ρ ≈ 0.7-0.85**
- All measure volume spikes, just different time windows
- Consider using ONLY ONE (recommend `window_volume_vs_1min_avg` for recency)

```python
# Price distances (lines 1274-1310)
distance_from_vwap_percent = ((last_price - vwap) / vwap) * 100
distance_from_day_low_percent = ((last_price - day_low) / day_low) * 100
distance_from_open_percent = ((last_price - open) / open) * 100
distance_from_prev_close_percent = ((last_price - prev_close) / prev_close) * 100
```
**Correlation: ρ ≈ 0.5-0.8 (depends on price action)**
- All measure "how far has price moved from X"
- If stock is trending, these will be highly correlated
- On gap days, prev_close vs open will differ significantly

---

### 🟠 **GROUP 3: Moderate Dependencies (ρ = 0.3-0.7)**

```python
# Risk/reward (lines 1323-1338)
risk_reward_ratio = (potential_target_price - last_price) / (last_price - estimated_stop_loss_price)
# This is a composite of: potential_target_price, last_price, estimated_stop_loss_price
```
**Dependency:** Derived from other fields
- Contains information about target but also depends on last_price
- Consider separately: potential_target_price is more independent

```python
# SPY correlation (lines 1342-1360)
spy_percent_change_day = total SPY move for day
spy_percent_change_concurrent = SPY move during squeeze window
```
**Correlation: ρ ≈ 0.4-0.6**
- Moderately correlated (trending days)
- Both provide somewhat independent information
- Keep both (one shows overall market regime, other shows concurrent correlation)

---

## Recommendations for Statistical Analysis

### ✅ **Phase 1: Correlation Matrix**

Before ANY analysis, compute correlation matrix:

```python
import pandas as pd
import numpy as np

# Load all alerts
alerts = []  # Load from JSON files
df = pd.DataFrame(alerts)

# Compute correlation matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

# Flag high correlations
high_corr = corr_matrix[(abs(corr_matrix) > 0.7) & (corr_matrix != 1.0)]
```

### ✅ **Phase 2: Feature Selection - Recommended Subset**

For modeling, use ONE field from each dependency group:

**INDEPENDENT/PRIMARY FEATURES (Use These):**
1. **Timing:** `market_session` (drop hour_of_day, time_since_market_open_minutes)
2. **Squeeze sequence:** `squeeze_number_today`, `minutes_since_last_squeeze`
3. **Price action:** `percent_change` (the squeeze magnitude)
4. **Day performance:** `day_gain` (drop distance_from_prev_close_percent - it's identical)
5. **Volume spike:** `window_volume_vs_1min_avg` (drop vs_5min_avg, volume_surge_ratio)
6. **Volume trend:** `volume_trend` (categorical: increasing/decreasing/stable)
7. **Price level context:**
   - `distance_from_vwap_percent` (current positioning)
   - `distance_from_day_low_percent` (momentum measure)
   - Choose ONE MORE: `distance_from_open_percent` OR drop if correlated
8. **Target potential:** `potential_target_price` (drop risk_reward_ratio initially)
9. **Fundamental:** `float_shares` (drop float_rotation_percent, keep float_rotation if needed)
10. **Float liquidity:** `float_rotation` (if independent of volume_surge_ratio)
11. **Microstructure:** `spread_percent` (drop absolute spread)
12. **Market regime:** `spy_percent_change_day`, `spy_percent_change_concurrent` (keep both)
13. **Historical levels:** `premarket_high`, `regular_hours_hod` (as ratios to last_price if needed)

**DROP ENTIRELY (No Information):**
- `stop_loss_distance_percent` (constant 7.5)
- All status icons/colors (derived from numeric values)
- `float_rotation_percent` (redundant with float_rotation)
- `estimated_stop_loss_price` (redundant with last_price)

---

## Critical Issues for Your Analysis

### ❌ **Problem 1: Multicollinearity**
If you use correlated fields together in regression/ML models:
- Coefficients become unstable
- Standard errors inflate
- P-values become unreliable
- Model interpretation breaks down

### ❌ **Problem 2: Information Leakage**
Some fields are calculated FROM the outcome you want to predict:
- If predicting "will price reach HOD?", don't use `distance_from_hod_percent` (circular)

### ❌ **Problem 3: Overfitting Risk**
44+ fields with many highly correlated = recipe for overfitting
- Use regularization (Lasso/Ridge)
- Use PCA/dimensionality reduction
- Use feature selection (eliminate correlated pairs)

---

## Action Plan

1. **Compute correlation matrix FIRST** (spend 1 hour on this)
2. **Identify correlation groups** (automate with clustering)
3. **Select ONE representative from each group** (domain knowledge + stats)
4. **Validate with VIF** (Variance Inflation Factor < 5)
5. **Then proceed with analysis**

**Bottom line:** You have ~15-20 truly independent pieces of information, not 44. Plan accordingly.

---

## Suggested Next Steps

### Immediate Actions

1. **Export alert data to CSV/DataFrame**
   ```python
   import json
   import pandas as pd
   from pathlib import Path

   alerts_dir = Path("data/squeeze_alerts_sent")
   alerts = []

   for alert_file in alerts_dir.glob("alert_*.json"):
       with open(alert_file) as f:
           data = json.load(f)
           # Flatten phase1_analysis into main dict
           if 'phase1_analysis' in data:
               phase1 = data.pop('phase1_analysis')
               data.update(phase1)
           alerts.append(data)

   df = pd.DataFrame(alerts)
   df.to_csv('squeeze_alerts_analysis.csv', index=False)
   ```

2. **Compute correlation matrix and heatmap**
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   # Select only numeric columns
   numeric_df = df.select_dtypes(include=['number'])

   # Compute correlations
   corr = numeric_df.corr()

   # Visualize
   plt.figure(figsize=(20, 18))
   sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
               vmin=-1, vmax=1, square=True)
   plt.title('Squeeze Alert Field Correlations')
   plt.tight_layout()
   plt.savefig('correlation_matrix.png', dpi=300)
   ```

3. **Identify high correlation pairs**
   ```python
   # Find pairs with |correlation| > 0.7
   high_corr_pairs = []
   for i in range(len(corr.columns)):
       for j in range(i+1, len(corr.columns)):
           if abs(corr.iloc[i, j]) > 0.7:
               high_corr_pairs.append({
                   'field1': corr.columns[i],
                   'field2': corr.columns[j],
                   'correlation': corr.iloc[i, j]
               })

   high_corr_df = pd.DataFrame(high_corr_pairs)
   high_corr_df = high_corr_df.sort_values('correlation',
                                             key=abs,
                                             ascending=False)
   print(high_corr_df)
   ```

4. **Calculate VIF for multicollinearity detection**
   ```python
   from statsmodels.stats.outliers_influence import variance_inflation_factor

   # Select features for VIF calculation (exclude target variable)
   feature_cols = [col for col in numeric_df.columns
                   if col not in ['percent_change', 'symbol']]
   X = numeric_df[feature_cols].dropna()

   # Calculate VIF for each feature
   vif_data = pd.DataFrame()
   vif_data["feature"] = X.columns
   vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                      for i in range(len(X.columns))]

   # Flag high VIF (>5 indicates multicollinearity)
   print(vif_data.sort_values('VIF', ascending=False))
   ```

### Analysis Workflow

**Week 1: Data Exploration**
- Export all alerts to DataFrame
- Generate summary statistics
- Create correlation matrix and heatmap
- Identify and document all duplicate/redundant fields

**Week 2: Feature Engineering**
- Remove constant fields (stop_loss_distance_percent)
- Remove perfect duplicates (day_gain vs distance_from_prev_close_percent)
- Create derived features if needed (e.g., price_momentum = distance_from_day_low / time_since_open)
- Normalize features where appropriate

**Week 3: Feature Selection**
- Apply correlation threshold (keep one from each |ρ| > 0.7 pair)
- Calculate VIF, remove features with VIF > 5
- Use domain knowledge to choose between correlated pairs
- Document final feature set

**Week 4: Initial Modeling**
- Define target variable (e.g., "squeeze success" = price reached HOD within N minutes)
- Split train/test sets (time-based split, not random)
- Try simple models first (logistic regression, decision tree)
- Evaluate feature importance

---

## Important Notes

### Time-Series Considerations

Your data is time-series, which adds complexity:

1. **Don't use random train/test split** - Use temporal split
   ```python
   # Split by date, not randomly
   train = df[df['timestamp'] < '2025-12-01']
   test = df[df['timestamp'] >= '2025-12-01']
   ```

2. **Watch for look-ahead bias** - Don't use future information
   - Example: `minutes_since_last_squeeze` uses only PAST squeezes (good)
   - Be careful with `regular_hours_hod` if squeeze occurs mid-day

3. **Consider autocorrelation** - Same symbol squeezes may not be independent
   - Group by symbol for validation
   - Use clustered standard errors

### Sample Size Requirements

With ~20 independent features, you need:
- **Minimum 200 observations** for basic analysis (10 per feature)
- **Better: 400+ observations** for robust results (20 per feature)
- **Ideal: 1000+ observations** for complex modeling

Monitor your data collection and plan analysis timeline accordingly.

---

## Related Documentation

- `PHASE1_COMPLETE.md` - Complete list of implemented fields
- `SQUEEZE_ALERT_ENHANCEMENT_RECOMMENDATIONS.md` - Original enhancement proposals
- `squeeze_alerts.py:1176-1362` - Phase 1 metrics calculation code
- `squeeze_alerts.py:1822-1927` - Alert JSON structure

---

**Last Updated:** 2025-12-12
**Purpose:** Guide statistical analysis of squeeze alert data with proper consideration of field dependencies and multicollinearity issues
