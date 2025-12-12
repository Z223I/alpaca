# Predictive Modeling Approach for Squeeze Alert Price Action

**Goal:** Predict price movement after squeeze alerts using the 10 independent features

**Data Available:**
- **Features:** 10 independent variables (analysis/squeeze_alerts_independent_features.csv)
- **Outcomes:** Price tracking data from JSON files (outcome_tracking section)

---

## Part 1: Target Variable Selection

### Available Outcome Data (from JSON outcome_tracking.summary)

Each squeeze alert tracks price movement at multiple intervals:

**Time-based price changes:**
- 10s, 20s, 30s, 40s, 50s intervals (sub-minute)
- 60s (1min), 90s, 120s (2min), 150s, 180s (3min), 240s (4min), 300s (5min)
- Up to 600s (10min) total tracking

**Summary metrics:**
- `max_gain_percent` - Highest gain reached during tracking period
- `final_gain_percent` - Gain at end of 10-minute window
- `max_drawdown_percent` - Worst drawdown encountered
- `reached_stop_loss` - Binary: hit 7.5% stop loss
- `achieved_5pct`, `achieved_10pct`, `achieved_15pct` - Binary milestones
- `time_to_5pct_minutes` - How quickly 5% was reached

### Recommended Target Variables (Choose Based on Strategy)

#### Option A: Binary Classification (Recommended for Beginners)

**Target 1: Quick Success (30-second prediction)**
```python
y = (gain_at_30s > 1.0)  # Did price gain >1% in 30 seconds?
```
- **Use case:** Scalping, ultra-short holds
- **Advantage:** Quick feedback, clear signal
- **Evaluation:** Accuracy, Precision, Recall, F1-score

**Target 2: 5% Milestone Achievement**
```python
y = achieved_5pct  # Did squeeze reach 5% gain within 10 minutes?
```
- **Use case:** Swing entries, position building
- **Advantage:** Aligns with typical profit targets
- **Evaluation:** Precision (avoid false positives), ROI simulation

**Target 3: Stop Loss Avoidance**
```python
y = (reached_stop_loss == False)  # Did squeeze avoid 7.5% stop loss?
```
- **Use case:** Risk management, trade filtering
- **Advantage:** Protects capital
- **Evaluation:** Recall (catch all risky trades)

#### Option B: Multi-Class Classification

**Target: Outcome Category**
```python
def categorize_outcome(row):
    if row['reached_stop_loss']:
        return 'LOSS'  # Hit stop loss
    elif row['achieved_10pct']:
        return 'BIG_WIN'  # >10% gain
    elif row['achieved_5pct']:
        return 'WIN'  # 5-10% gain
    elif row['final_gain_percent'] > 0:
        return 'SMALL_WIN'  # Small positive
    else:
        return 'FAIL'  # Negative but didn't hit stop

y = df.apply(categorize_outcome, axis=1)
```
- **Classes:** LOSS, FAIL, SMALL_WIN, WIN, BIG_WIN
- **Evaluation:** Confusion matrix, per-class precision/recall

#### Option C: Regression (Continuous Prediction)

**Target 1: Maximum Gain Prediction**
```python
y = max_gain_percent  # Predict highest gain during 10-min window
```
- **Use case:** Position sizing, profit target setting
- **Evaluation:** RMSE, MAE, R²

**Target 2: 1-Minute Price Change**
```python
y = gain_at_60s  # Predict exact gain at 1 minute
```
- **Use case:** High-frequency entry/exit timing
- **Evaluation:** RMSE, directional accuracy

**Target 3: Risk-Adjusted Return**
```python
y = max_gain_percent / abs(max_drawdown_percent)  # Sharpe-like ratio
```
- **Use case:** Quality scoring for trade selection
- **Evaluation:** RMSE, correlation with actual performance

---

## Part 2: Data Preparation Strategy

### Step 1: Extract Outcomes from JSON Files

```python
import json
from pathlib import Path
import pandas as pd

def extract_outcomes(json_dir):
    """Extract outcome_tracking data from JSON files."""
    outcomes = []

    for json_file in Path(json_dir).glob('alert_*.json'):
        with open(json_file) as f:
            data = json.load(f)

            # Create unique key for joining
            key = {
                'symbol': data['symbol'],
                'timestamp': data['timestamp']
            }

            # Extract outcome summary
            if 'outcome_tracking' in data and 'summary' in data['outcome_tracking']:
                summary = data['outcome_tracking']['summary']

                # Add specific interval gains
                intervals = data['outcome_tracking'].get('intervals', {})

                outcome_row = {
                    **key,
                    'max_gain_percent': summary.get('max_gain_percent'),
                    'final_gain_percent': summary.get('final_gain_percent'),
                    'max_drawdown_percent': summary.get('max_drawdown_percent'),
                    'reached_stop_loss': summary.get('reached_stop_loss'),
                    'achieved_5pct': summary.get('achieved_5pct'),
                    'achieved_10pct': summary.get('achieved_10pct'),
                    'time_to_5pct_minutes': summary.get('time_to_5pct_minutes'),
                    # Add interval-specific gains
                    'gain_at_10s': intervals.get('10', {}).get('gain_percent'),
                    'gain_at_30s': intervals.get('30', {}).get('gain_percent'),
                    'gain_at_60s': intervals.get('60', {}).get('gain_percent'),
                    'gain_at_120s': intervals.get('120', {}).get('gain_percent'),
                    'gain_at_300s': intervals.get('300', {}).get('gain_percent'),
                }

                outcomes.append(outcome_row)

    return pd.DataFrame(outcomes)
```

### Step 2: Merge with Independent Features

```python
# Load independent features
features_df = pd.read_csv('analysis/squeeze_alerts_independent_features.csv')

# Load outcomes
outcomes_df = extract_outcomes('historical_data/2025-12-12/squeeze_alerts_sent/')

# Merge on symbol + timestamp
df = features_df.merge(
    outcomes_df,
    on=['symbol', 'timestamp'],
    how='inner'  # Only keep rows with both features and outcomes
)

print(f"Merged dataset: {len(df)} observations")
```

### Step 3: Handle Missing Data

**Missing Data Strategy:**

```python
# Check missing data
print(df.isnull().sum())

# Strategy A: Drop rows with missing features (recommended)
# Keep ema_spread null handling since it's 31% missing
df_clean = df.dropna(subset=[
    'minutes_since_last_squeeze',
    'day_gain',
    # Don't drop ema_spread yet - handle separately
])

# Strategy B: Impute ema_spread (31% missing)
from sklearn.impute import SimpleImputer
ema_imputer = SimpleImputer(strategy='median')
df_clean['ema_spread_imputed'] = ema_imputer.fit_transform(
    df_clean[['ema_spread']]
)

# Strategy C: Create indicator variable for missing EMA
df_clean['ema_missing'] = df_clean['ema_spread'].isnull().astype(int)
df_clean['ema_spread'].fillna(0, inplace=True)  # Fill with 0 + use indicator
```

---

## Part 3: Train/Test Split Strategy

### ⚠️ CRITICAL: Time-Based Split (NOT Random)

**Why time-based?**
- Squeeze alerts are time-series data
- Random split causes **data leakage** (training on future, testing on past)
- Real trading = predict future from past

**Recommended Approach:**

```python
# Sort by timestamp
df_sorted = df.sort_values('timestamp').reset_index(drop=True)

# Strategy A: Simple 80/20 time split
split_idx = int(len(df_sorted) * 0.80)
train_df = df_sorted[:split_idx]
test_df = df_sorted[split_idx:]

print(f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
```

**Alternative: Cross-Validation (Time-Series Aware)**

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(df_sorted)):
    train_fold = df_sorted.iloc[train_idx]
    test_fold = df_sorted.iloc[test_idx]
    # Train and evaluate model
```

---

## Part 4: Model Selection Strategy

### Recommended Models (in order of complexity)

#### Level 1: Baseline Models (Start Here)

**1. Logistic Regression (Binary Classification)**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Pros: Fast, interpretable, shows feature coefficients
# Cons: Assumes linear relationships
# Use for: Quick baseline, understanding feature importance
```

**2. Linear Regression (Continuous Targets)**
```python
from sklearn.linear_model import LinearRegression

# Pros: Simple, interpretable
# Cons: Assumes linear relationships
# Use for: Predicting max_gain_percent, gain_at_60s
```

#### Level 2: Tree-Based Models (Recommended)

**3. Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Pros: Handles non-linear relationships, feature importance, robust
# Cons: Can overfit, black box
# Use for: Best general-purpose model, feature selection
# Hyperparameters: n_estimators=100-500, max_depth=5-15
```

**4. XGBoost (Best Performance)**
```python
import xgboost as xgb

# Pros: Best performance, handles missing data, regularization
# Cons: Requires tuning, slower training
# Use for: Production models, competitions
# Hyperparameters: learning_rate=0.01-0.1, max_depth=3-7, n_estimators=100-1000
```

#### Level 3: Advanced Models

**5. LightGBM (Fast Alternative to XGBoost)**
```python
import lightgbm as lgb

# Pros: Faster than XGBoost, similar performance
# Cons: Requires tuning
# Use for: Large datasets, production
```

**6. Neural Networks (If Lots of Data)**
```python
from sklearn.neural_network import MLPClassifier

# Pros: Can capture complex patterns
# Cons: Requires lots of data (>1000 samples), hard to tune
# Use for: Only if tree-based models don't work
```

### Model Comparison Strategy

```python
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {'accuracy': accuracy, 'f1': f1}
    print(f"{name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
```

---

## Part 5: Evaluation Strategy

### Binary Classification Metrics

**Primary Metrics:**
- **Precision:** Of squeezes predicted as "successful", what % actually succeeded?
  - High precision = avoid false positives (bad trades)
- **Recall:** Of actually successful squeezes, what % did we predict?
  - High recall = catch all good opportunities
- **F1-Score:** Harmonic mean of precision and recall

**Trading-Specific Metrics:**
```python
from sklearn.metrics import classification_report, confusion_matrix

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
#                 Predicted
#                 0    1
# Actual    0   TN   FP   <- False positives = bad trades taken
#           1   FN   TP   <- False negatives = missed opportunities

# Classification Report
print(classification_report(y_test, y_pred))

# Trading Metrics
predicted_trades = y_pred == 1  # Trades we would take
actual_winners = y_test == 1

win_rate = precision_score(y_test, y_pred)  # % of our trades that win
capture_rate = recall_score(y_test, y_pred)  # % of winners we caught
```

**ROI Simulation:**
```python
def simulate_trading(y_true, y_pred, gains, losses):
    """Simulate P&L if we trade based on predictions."""
    total_pnl = 0
    trades_taken = 0

    for i in range(len(y_pred)):
        if y_pred[i] == 1:  # Model says take trade
            trades_taken += 1
            if y_true[i] == 1:  # Actually won
                total_pnl += gains[i]
            else:  # Actually lost
                total_pnl += losses[i]

    avg_pnl = total_pnl / trades_taken if trades_taken > 0 else 0
    return total_pnl, trades_taken, avg_pnl
```

### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Directional Accuracy (more important for trading)
directional_accuracy = np.mean((y_pred > 0) == (y_test > 0))
```

---

## Part 6: Feature Importance Analysis

### Why Feature Importance Matters

1. **Understand what drives price action**
2. **Simplify model** (drop unimportant features)
3. **Generate trading insights**

### Extract Feature Importance

```python
# Random Forest / XGBoost
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Logistic Regression (coefficients)
coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
```

### SHAP Values (Advanced)

```python
import shap

# Calculate SHAP values (explains each prediction)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

---

## Part 7: Recommended Starting Approach

### Phase 1: Quick Binary Classification (Day 1)

**Target:** `achieved_5pct` (reached 5% gain in 10 minutes)

**Steps:**
1. Extract outcomes from JSON
2. Merge with independent features
3. Create binary target: `y = achieved_5pct`
4. 80/20 time-based split
5. Train Random Forest (simple, good default)
6. Evaluate precision, recall, F1
7. Check feature importance

**Success Criteria:**
- Precision > 60% (avoid too many false positives)
- F1 > 0.55 (balanced performance)

### Phase 2: Optimize Model (Day 2-3)

1. Try XGBoost
2. Hyperparameter tuning (grid search)
3. Add feature engineering:
   - Interaction terms (e.g., `ema_spread * day_gain`)
   - Polynomial features
4. Handle class imbalance (if needed)

### Phase 3: Multiple Targets (Day 4-5)

Compare predictions for:
- 30-second gains (ultra-short)
- 1-minute gains (short-term)
- 5% achievement (medium-term)

Select best-performing target for your strategy.

### Phase 4: Production (Week 2)

1. Backtest on additional dates
2. Validate on out-of-sample data
3. Build prediction pipeline
4. Monitor performance over time

---

## Part 8: Key Considerations

### Class Imbalance

If 70% of squeezes succeed and 30% fail:
- Model may predict "success" for everything (70% accuracy but useless)
- **Solutions:**
  - Use `class_weight='balanced'` in model
  - Focus on precision/recall instead of accuracy
  - Use stratified sampling

### Overfitting Prevention

**Signs of overfitting:**
- Train accuracy 95%, test accuracy 60%
- Model too complex for data size

**Solutions:**
- Reduce model complexity (lower max_depth, fewer trees)
- Cross-validation
- Regularization (L1/L2 for logistic regression)
- More data (collect more days)

### Sample Size Requirements

Current dataset: ~465 alerts

**Minimum requirements:**
- Binary classification: 200+ samples (have 465 ✅)
- Multi-class (5 classes): 500+ samples (borderline)
- Regression: 200+ samples (have 465 ✅)

**With 10 features:**
- Need 20+ samples per feature = 200 minimum ✅
- Ideal: 50+ samples per feature = 500 ideal (close)

---

## Summary: Recommended First Approach

```python
# 1. Target Variable
y = achieved_5pct  # Binary: reached 5% gain

# 2. Features
X = [ema_spread, distance_from_vwap_percent, minutes_since_last_squeeze,
     window_volume_vs_1min_avg, spy_percent_change_concurrent,
     spread_percent, day_gain, squeeze_number_today,
     distance_from_day_low_percent, market_session]

# 3. Model
RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')

# 4. Evaluation
- Precision (avoid bad trades)
- Recall (catch good opportunities)
- F1-score (balance)
- Feature importance

# 5. Success Metric
If precision > 60% and F1 > 0.55, model is useful for trade filtering
```

---

**Next Step:** Implement this approach in a Python script?
