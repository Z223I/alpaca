# Final Independent Feature Set for Squeeze Alerts Analysis

**Date:** 2025-12-12
**Dataset:** 465 squeeze alerts
**Status:** ✅ **VALIDATED - All features statistically independent**

---

## ✅ Final Feature Set: 10 Independent Features

All features have **VIF < 5** (no multicollinearity) and **correlations < 0.7** (no high linear dependence).

### Complete List

| # | Feature | VIF | Mean | Std | Availability | Notes |
|---|---------|-----|------|-----|--------------|-------|
| 1 | **ema_spread** | **1.31** | -0.13 | 0.67 | 69.0% | **EMA 9 - EMA 21** (user requested) |
| 2 | distance_from_vwap_percent | 1.32 | 1.53 | 3.58 | 100% | Price positioning vs fair value |
| 3 | minutes_since_last_squeeze | 1.38 | 2.12 | 3.29 | 94.0% | Temporal spacing between squeezes |
| 4 | window_volume_vs_1min_avg | 2.11 | 253.06 | 282.10 | 100% | Volume surge magnitude |
| 5 | spy_percent_change_concurrent | 2.31 | -0.21% | 0.25% | 100% | Market correlation during squeeze |
| 6 | spread_percent | 2.37 | 1.59% | 1.51% | 100% | Liquidity measure (bid-ask) |
| 7 | day_gain | 3.01 | 84.04% | 62.84% | 99.8% | Daily stock performance |
| 8 | squeeze_number_today | 3.75 | 28.72 | 23.11 | 100% | Squeeze sequence counter |
| 9 | distance_from_day_low_percent | 4.02 | 11.86% | 8.37% | 100% | Price momentum from low |
| 10 | market_session | N/A | - | - | 100% | Categorical (extended/early/mid/late) |

### Supplementary Feature (High Missing Data)

| Feature | VIF | Availability | Status |
|---------|-----|--------------|--------|
| macd_histogram | N/A | **19.1%** | ⚠️ **80.9% missing** - use only for subset analysis |

---

## ❌ Removed Features (High Multicollinearity)

| Feature | VIF (before removal) | Reason |
|---------|---------------------|--------|
| **spy_percent_change_day** | **18.74** | 🔴 Severe multicollinearity - can be predicted from other features |
| **percent_change** | **10.03** | 🔴 Severe multicollinearity - better used as target variable |

**Impact of removal:**
- `squeeze_number_today` VIF dropped from 5.32 → 3.75 ✅
- All remaining features now have VIF < 5 ✅

---

## 📊 Validation Results

### Correlation Analysis
✅ **PASSED** - No correlations above 0.7 threshold

**Highest correlations found:**
- `macd_histogram` ↔ `squeeze_number_today`: -0.47 (moderate negative)
- `macd_histogram` ↔ `spread_percent`: -0.43 (moderate negative)
- `ema_spread` ↔ `macd_histogram`: 0.40 (moderate positive)
- `ema_spread` ↔ `spy_percent_change_concurrent`: 0.36 (low-moderate)
- `day_gain` ↔ `spread_percent`: 0.32 (low)

All acceptable for statistical modeling.

### VIF Analysis
✅ **PASSED** - All VIF scores < 5

**VIF Distribution:**
- 🟢 Excellent (VIF < 2): 3 features
- 🟢 Good (VIF 2-3): 3 features
- 🟢 Acceptable (VIF 3-5): 3 features
- 🔴 Problematic (VIF > 5): 0 features ✅

**Tested on:** 302 complete observations (65% of dataset)

---

## 📈 Data Quality Summary

### Data Availability

| Status | Feature Count | Details |
|--------|--------------|---------|
| ✅ Perfect (100%) | 7 features | No missing data |
| ✅ Excellent (>90%) | 2 features | <10% missing |
| 🟡 Good (60-90%) | 1 feature | ema_spread (69.0%) |
| 🔴 Poor (<60%) | 1 feature | macd_histogram (19.1%) |

### Sample Size
- **Total alerts:** 465
- **Complete cases (all 10 features):** 302 (65%)
- **Sufficient for modeling:** ✅ Yes (30+ samples per feature required, have 302/10 = 30.2)

---

## 🎯 EMA Feature Engineering (User Requested)

### EMA Spread = EMA 9 - EMA 21

**Purpose:** Capture short-term vs medium-term momentum divergence

**Statistical Properties:**
- ✅ **VIF: 1.31** (excellent independence)
- ✅ **Availability: 69.0%** (321/465 alerts)
- ✅ **Correlation with other features:** All < 0.4
- Mean: -0.13 (slightly bearish bias)
- Range: -2.83 to +1.77

**Interpretation:**
- **Negative values** (EMA 9 < EMA 21): Short-term bearish momentum
- **Positive values** (EMA 9 > EMA 21): Short-term bullish momentum
- **Magnitude**: Strength of divergence/convergence

**Why not include EMA 9 and EMA 21 separately?**
- They would have high correlation (both track price)
- The *difference* captures the independent signal (momentum divergence)
- VIF would increase if both included

---

## ⚠️ MACD Issue and Recommendations

### Current Status
- **MACD Histogram availability:** 19.1% (89/465 alerts)
- **Missing:** 80.9% of alerts have no MACD data

### Why So Much Missing Data?

1. **MACD calculation requires:**
   - 12-period EMA (fast)
   - 26-period EMA (slow)
   - 9-period EMA of MACD (signal)
   - **Minimum:** 26+ bars (1-minute bars in current implementation)

2. **Squeeze alerts occur:**
   - Pre-market (extended hours): Very sparse 1-minute bars
   - Early regular hours: < 26 minutes into session
   - Fast-moving stocks: Gaps in bar data

### Solutions (Choose One)

**Option 1: Use Daily MACD (Recommended)**
```python
# Calculate MACD on daily bars instead of 1-minute bars
# This will have 100% coverage (always available)
```
- ✅ 100% data availability
- ✅ More stable signal
- ❌ Less responsive to intraday momentum

**Option 2: Accept Limited Coverage**
- Use MACD only when available (subset analysis)
- Perform separate analysis on 89 alerts with MACD
- ❌ Loses 80% of data

**Option 3: Different Intraday Period**
- Use 5-minute bars (need 130 minutes of data)
- Use 15-minute bars (need 390 minutes of data)
- ❌ Still won't work for pre-market squeezes

**Recommendation:** Switch to **daily MACD** for universal coverage.

---

## 💻 Usage Example

### Load Clean Dataset

```python
import pandas as pd
import numpy as np

# Load the clean feature set
df = pd.read_csv('analysis/squeeze_alerts_independent_features.csv')

# Define independent features (excluding MACD due to missing data)
features = [
    'ema_spread',
    'distance_from_vwap_percent',
    'minutes_since_last_squeeze',
    'window_volume_vs_1min_avg',
    'spy_percent_change_concurrent',
    'spread_percent',
    'day_gain',
    'squeeze_number_today',
    'distance_from_day_low_percent',
    'market_session'
]

# Select feature matrix
X = df[features].copy()

# Handle categorical (market_session)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['market_session_encoded'] = le.fit_transform(X['market_session'])
X = X.drop('market_session', axis=1)

# Handle missing data (mainly ema_spread at 31% missing)
print(X.isnull().sum())

# Option A: Drop rows with missing data
X_complete = X.dropna()  # 302 observations

# Option B: Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

print(f"Complete cases: {len(X_complete)}")
print(f"After imputation: {len(X_imputed)}")
```

### Define Target Variable

```python
# Example target: squeeze success (reached 5% gain within 10 minutes)
# Note: This requires outcome_tracking data from JSON files

# Option A: Load from outcome_tracking in JSON
# y = df['outcome_tracking.summary.achieved_5pct']

# Option B: Use percent_change as continuous target
# (Note: percent_change was REMOVED from features due to high VIF)
y = df['percent_change']  # Available but not in feature set

# Option C: Create binary classification target
y_binary = (df['percent_change'] > 3.0).astype(int)  # 1 if >3%, 0 otherwise
```

### Statistical Modeling

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, classification_report

# Time-based split (CRITICAL for time-series data)
df_sorted = df.sort_values('timestamp').reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)

# Using complete cases
X_complete = X_complete[:len(df_sorted)]  # Align with sorted df
y = y[:len(X_complete)]

X_train = X_complete[:split_idx]
X_test = X_complete[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

# Standardize features (important for regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

print(f"Linear Regression R²: {r2_score(y_test, y_pred):.3f}")

# Feature importance (use tree-based model)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)
```

---

## 📁 Generated Files

```
analysis/
├── squeeze_alerts_statistical_analysis.py    # Analysis script
├── correlation_heatmap.png                    # Correlation matrix visualization
├── squeeze_alerts_independent_features.csv   # Clean dataset (465 alerts)
├── summary_report.json                        # Statistical summary
├── STATISTICAL_INDEPENDENCE_RECOMMENDATIONS.md  # Detailed analysis
└── FINAL_FEATURE_SET.md                       # This document
```

---

## 🎯 Key Takeaways

### ✅ What We Achieved

1. **Created statistically independent feature set**
   - All VIF scores < 5 ✅
   - No high correlations (all |r| < 0.7) ✅

2. **Engineered EMA spread feature**
   - ema_spread = ema_9 - ema_21
   - VIF = 1.31 (excellent) ✅
   - Captures momentum divergence ✅

3. **Removed multicollinear features**
   - spy_percent_change_day (VIF was 18.74)
   - percent_change (VIF was 10.03)

4. **Validated on real data**
   - 465 alerts from 2025-12-12
   - 302 complete observations for VIF
   - Sufficient sample size ✅

### ⚠️ Outstanding Issues

1. **MACD data coverage: 19.1%**
   - Consider switching to daily MACD
   - Or accept limited subset analysis

2. **EMA spread: 31% missing**
   - Use imputation or complete-case analysis
   - 321 observations still sufficient

### 🚀 Ready for Next Steps

The feature set is now **validated and ready** for:
- ✅ Regression modeling
- ✅ Classification (squeeze success prediction)
- ✅ Feature importance analysis
- ✅ Machine learning (RF, XGBoost, etc.)
- ✅ Statistical hypothesis testing

---

**Analysis validated:** 2025-12-12
**Total features:** 10 independent + 1 supplementary (MACD)
**VIF status:** ✅ All < 5
**Correlation status:** ✅ All < 0.7
**Sample size:** ✅ 302 complete cases (sufficient)
