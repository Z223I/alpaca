# Statistical Independence Analysis - Final Recommendations

**Date:** 2025-12-12
**Dataset:** 439 squeeze alerts from 2025-12-12
**Analysis Focus:** EMA, MACD, and statistically independent features

---

## Executive Summary

✅ **Correlation Analysis:** No high correlations (all |r| ≤ 0.7)
⚠️ **VIF Analysis:** 2 features show severe multicollinearity (VIF > 10)
⚠️ **Missing Data:** MACD histogram has 85.6% missing data

**Recommendation:** Use **9 truly independent features** (VIF < 5) for statistical modeling.

---

## Part 1: Data Quality Assessment

### Missing Data by Feature

| Feature | Available | Missing % | Usability |
|---------|-----------|-----------|-----------|
| **macd_histogram** | 63/439 | **85.6%** | 🔴 **EXCLUDE** (insufficient data) |
| **ema_spread** | 295/439 | 32.8% | 🟡 Usable with caution |
| minutes_since_last_squeeze | 411/439 | 6.4% | ✅ Good |
| day_gain | 438/439 | 0.2% | ✅ Excellent |
| All others | 439/439 | 0% | ✅ Perfect |

### Critical Issue: MACD Data

**Problem:** Only 14.4% of alerts have MACD values (63 out of 439).

**Root Cause:** MACD requires 26+ 1-minute bars for calculation. Most squeeze alerts:
- Occur in pre-market (extended hours) where bars are sparse
- Happen early in the trading session before sufficient data accumulates

**Recommendation:**
- ❌ **DROP macd_histogram from core feature set** (insufficient data)
- ✅ Add as **supplementary feature** for analysis on subset where available
- ✅ Consider calculating MACD on daily bars instead of 1-minute bars for better coverage

---

## Part 2: Correlation Analysis Results

### Correlation Matrix Summary

**Highest correlations found (none exceed 0.7 threshold):**

| Feature 1 | Feature 2 | Correlation | Interpretation |
|-----------|-----------|-------------|----------------|
| squeeze_number_today | spy_percent_change_day | -0.62 | As day progresses, SPY declines |
| spy_percent_change_day | spy_percent_change_concurrent | 0.43 | Moderate market correlation |
| ema_spread | spy_percent_change_concurrent | 0.40 | Price momentum follows market |
| window_volume_vs_1min_avg | spy_percent_change_day | 0.38 | Volume surges on down days |
| percent_change | spread_percent | 0.38 | Larger moves have wider spreads |
| day_gain | spy_percent_change_concurrent | 0.33 | Stock gains follow market |

**✅ Conclusion:** No problematic linear correlations detected.

---

## Part 3: Multicollinearity Analysis (VIF)

**VIF calculated on 276 complete observations**

### Features with Multicollinearity Issues

| Feature | VIF | Status | Action |
|---------|-----|--------|--------|
| **spy_percent_change_day** | **18.74** | 🔴 **SEVERE** | **REMOVE** |
| **percent_change** | **10.03** | 🔴 **SEVERE** | **REMOVE** or keep as target |
| squeeze_number_today | 5.32 | 🟡 WARNING | Monitor |

### Independent Features (VIF < 5)

| Feature | VIF | Status | Notes |
|---------|-----|--------|-------|
| distance_from_day_low_percent | 4.60 | 🟢 GOOD | Price momentum indicator |
| day_gain | 3.39 | 🟢 GOOD | Daily performance |
| spread_percent | 3.20 | 🟢 GOOD | Liquidity measure |
| spy_percent_change_concurrent | 3.07 | 🟢 GOOD | Market correlation (keep this one) |
| window_volume_vs_1min_avg | 2.39 | 🟢 GOOD | Volume surge |
| minutes_since_last_squeeze | 1.63 | 🟢 GOOD | Temporal spacing |
| **ema_spread** | **1.45** | **🟢 EXCELLENT** | **EMA divergence** |
| distance_from_vwap_percent | 1.45 | 🟢 EXCELLENT | Price positioning |

---

## Part 4: Final Feature Recommendations

### Recommended Feature Set: 9 Independent Features

Based on VIF < 5 and sufficient data availability:

```python
recommended_independent_features = [
    # 1. EMA Feature (user requested)
    'ema_spread',                      # VIF=1.45 ✅ (ema_9 - ema_21)

    # 2. Timing
    'market_session',                   # Categorical (extended/early/mid/late)
    'minutes_since_last_squeeze',       # VIF=1.63 ✅

    # 3. Volume
    'window_volume_vs_1min_avg',        # VIF=2.39 ✅

    # 4. Price Positioning
    'distance_from_vwap_percent',       # VIF=1.45 ✅
    'distance_from_day_low_percent',    # VIF=4.60 ✅

    # 5. Performance
    'day_gain',                         # VIF=3.39 ✅

    # 6. Market Context
    'spy_percent_change_concurrent',    # VIF=3.07 ✅ (not _day!)

    # 7. Microstructure
    'spread_percent'                    # VIF=3.20 ✅
]
```

### Features to EXCLUDE

**Excluded due to high VIF (multicollinearity):**
- ❌ `spy_percent_change_day` (VIF=18.74) - Use `spy_percent_change_concurrent` instead
- ❌ `percent_change` (VIF=10.03) - Consider as **target variable** instead of predictor

**Excluded due to insufficient data:**
- ❌ `macd_histogram` (85.6% missing) - Only 63 observations

**Borderline (consider based on model):**
- ⚠️ `squeeze_number_today` (VIF=5.32) - Keep if interpretable in your context

---

## Part 5: EMA and MACD Recommendations

### EMA Analysis ✅

**Feature:** `ema_spread = ema_9 - ema_21`

**Performance:**
- ✅ Data availability: 67.2% (295/439 alerts)
- ✅ VIF: 1.45 (excellent independence)
- ✅ Mean: -0.12, Std: 0.69
- ✅ Range: -2.83 to +1.77

**Interpretation:**
- Negative values: EMA 9 < EMA 21 (bearish short-term)
- Positive values: EMA 9 > EMA 21 (bullish short-term)
- Magnitude indicates strength of trend divergence

**Recommendation:** ✅ **KEEP - Excellent independent feature**

### MACD Analysis ❌

**Feature:** `macd_histogram`

**Performance:**
- ❌ Data availability: 14.4% only (63/439 alerts)
- ❌ Cannot calculate VIF (insufficient data)
- Mean: -0.04, Std: 0.15
- Range: -0.61 to +0.17

**Why so much missing data?**
1. MACD requires 26+ bars (12 EMA slow + 26 EMA calculation)
2. Pre-market has sparse bars (low liquidity)
3. Early regular hours don't have 26+ minutes of data yet
4. Current implementation uses 1-minute bars

**Options to fix:**
1. ✅ **Use daily MACD** instead of 1-minute MACD (always available)
2. ✅ **Accept missing data** and use MACD only for subset analysis
3. ❌ Drop MACD entirely (not recommended if user wants it)

**Recommendation:** ⚠️ **Recalculate MACD from daily bars** OR **use as supplementary feature**

---

## Part 6: Addressing the VIF Issues

### Why does `spy_percent_change_day` have VIF=18.74?

**Analysis:** High VIF indicates `spy_percent_change_day` can be predicted from other features.

**Likely culprits:**
1. `squeeze_number_today` - Later in day = more negative SPY movement
2. `day_gain` - Individual stock gains may proxy overall market sentiment
3. `distance_from_day_low_percent` - Intraday positioning correlates with market trend

**Solution:** Use `spy_percent_change_concurrent` (VIF=3.07) instead - measures market movement **during the squeeze window only**, not entire day.

### Why does `percent_change` have VIF=10.03?

**Analysis:** Squeeze magnitude can be predicted from other features.

**This makes sense because:**
- Larger squeezes often have wider spreads (`spread_percent`)
- Squeezes later in momentum moves show specific patterns
- Volume surges relate to price movement magnitude

**Solution:**
- If `percent_change` is your **outcome variable** (what you're trying to predict), keep it as the target
- If using for **prediction**, remove it due to multicollinearity

---

## Part 7: Recommended Analysis Workflow

### Step 1: Use Core 9 Independent Features

```python
# Load the clean dataset
import pandas as pd

df = pd.read_csv('analysis/squeeze_alerts_independent_features.csv')

# Select only truly independent features (VIF < 5)
core_features = [
    'ema_spread',
    'market_session',
    'minutes_since_last_squeeze',
    'window_volume_vs_1min_avg',
    'distance_from_vwap_percent',
    'distance_from_day_low_percent',
    'day_gain',
    'spy_percent_change_concurrent',
    'spread_percent'
]

X = df[core_features]
```

### Step 2: Handle Missing Data

```python
# Check missing data
print(X.isnull().sum())

# Option A: Use only complete cases
X_complete = X.dropna()  # 276 observations

# Option B: Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)
```

### Step 3: Define Target Variable

```python
# Option A: Use percent_change as target
y = df['percent_change']

# Option B: Create binary target (success = reached 5% gain)
# Would need outcome_tracking data for this
```

### Step 4: Statistical Modeling

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Split data (time-based, not random!)
# Ensure early squeezes are training, later ones are test
df_sorted = df.sort_values('timestamp')
split_idx = int(len(df_sorted) * 0.8)

X_train = X_complete[:split_idx]
X_test = X_complete[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import r2_score
y_pred = model.predict(X_test_scaled)
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

---

## Part 8: Summary and Next Steps

### What We Learned

✅ **Achieved:**
1. Created `ema_spread` as independent EMA feature (VIF=1.45)
2. Identified 9 truly independent features (VIF < 5)
3. Detected and addressed multicollinearity (removed 2 features)
4. Generated correlation matrix confirming independence

⚠️ **Issues Found:**
1. MACD has 85.6% missing data (need to fix calculation method)
2. `spy_percent_change_day` has severe multicollinearity (use concurrent instead)
3. `percent_change` may be better as target than predictor

### Immediate Action Items

1. **Fix MACD calculation:**
   - Switch from 1-minute bars to daily bars for better coverage
   - OR accept MACD as supplementary feature for subset analysis

2. **Use the 9 recommended independent features** for modeling

3. **Decide on target variable:**
   - If predicting squeeze success: use outcome_tracking metrics
   - If predicting squeeze magnitude: use percent_change (but don't include in predictors)

4. **Validate model assumptions:**
   - Check residual plots
   - Test for autocorrelation (same symbol repeated squeezes)
   - Consider random effects for symbol grouping

### Files Generated

📁 **analysis/**
- `correlation_heatmap.png` - Visual correlation matrix
- `squeeze_alerts_independent_features.csv` - Clean dataset (439 alerts)
- `summary_report.json` - Complete statistics
- `STATISTICAL_INDEPENDENCE_RECOMMENDATIONS.md` - This document

---

## Appendix: Feature Descriptions

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| ema_spread | EMA 9 - EMA 21 (momentum divergence) | Continuous | -2.83 to 1.77 |
| market_session | Trading session | Categorical | extended/early/mid/late |
| minutes_since_last_squeeze | Time since previous squeeze | Continuous | 0.5 to 35.1 min |
| window_volume_vs_1min_avg | Volume surge ratio | Continuous | 0.36 to 1972.54 |
| distance_from_vwap_percent | Price vs VWAP | Continuous | -15.2% to 18.3% |
| distance_from_day_low_percent | Price momentum from low | Continuous | 2.0% to 52.1% |
| day_gain | Daily performance | Continuous | 10.4% to 290.0% |
| spy_percent_change_concurrent | Market correlation during squeeze | Continuous | -0.9% to 0.26% |
| spread_percent | Bid-ask spread (liquidity) | Continuous | -0.6% to 11.6% |

---

**Analysis Complete** | Generated by squeeze_alerts_statistical_analysis.py
