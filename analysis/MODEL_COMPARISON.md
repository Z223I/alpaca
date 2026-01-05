# XGBoost vs DNN: Implementation Comparison

## Quick Reference

| Aspect | XGBoost | DNN |
|--------|---------|-----|
| **File** | `analysis/tune_models.py` | `analysis/dnn.py` |
| **Config** | `analysis/tuning_config.yaml` | `analysis/dnn_config.yaml` |
| **Framework** | XGBoost 3.1+ | TensorFlow/Keras |
| **Optimization** | Optuna (TPE sampler) | W&B Sweeps (Bayesian) |
| **GPU Support** | `device='cuda:0'` | TensorFlow auto-detect |
| **Model Type** | Gradient Boosted Trees | Deep Neural Network |
| **Best For** | Tabular data, interpretability | Complex patterns, scalability |

## Usage Comparison

### XGBoost

```bash
# Single threshold training
python analysis/tune_models.py train --threshold 6.0 --trials 100

# All thresholds
python analysis/tune_models.py train --all-thresholds --trials 50

# Resume study
python analysis/tune_models.py train --threshold 6.0 --resume --study-name my-study

# Dry run
python analysis/tune_models.py train --threshold 6.0 --dry-run
```

### DNN

```bash
# Single model training
python analysis/dnn.py train --threshold 6.0 --epochs 100

# Initialize sweep
python analysis/dnn.py init-sweep --threshold 6.0

# Run sweep agent
export DNN_CONFIG_PATH='analysis/dnn_config.yaml'
export DNN_THRESHOLD='6.0'
export DNN_TRAILING_STOP='2.0'
python analysis/dnn.py run-sweep --sweep-id <id>

# Dry run
python analysis/dnn.py train --threshold 6.0 --dry-run
```

## Hyperparameter Mapping

### XGBoost → DNN Equivalents

| XGBoost Parameter | Range | DNN Equivalent | Range | Reasoning |
|-------------------|-------|----------------|-------|-----------|
| `max_depth` | 3-10 | `n_layers` | 2-4 | Tree depth → Network depth |
| `min_child_weight` | 1-10 | `dropout_rate` | 0.1-0.5 | Both control model complexity |
| `n_estimators` | 100-500 | `epochs` | 50-200 | Training iterations |
| `learning_rate` | 0.001-0.3 | `learning_rate` | 0.0001-0.01 | Step size |
| `gamma` | 0.0001-1.0 | `l2_regularization` | 0.00001-0.01 | Regularization |
| `reg_alpha` (L1) | 0.0001-10.0 | N/A | - | DNNs typically use L2 only |
| `reg_lambda` (L2) | 0.0001-10.0 | `l2_regularization` | 0.00001-0.01 | L2 penalty |
| `subsample` | 0.5-1.0 | `batch_size` | 16-128 | Sample fraction vs batch training |
| `colsample_bytree` | 0.5-1.0 | N/A | - | Dropout provides similar effect |
| `colsample_bylevel` | 0.5-1.0 | N/A | - | Dropout provides similar effect |
| `colsample_bynode` | 0.5-1.0 | N/A | - | Dropout provides similar effect |

### DNN-Specific Hyperparameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `units_layer1` | 64-512 | First layer units |
| `units_layer2` | 32-256 | Second layer units |
| `units_layer3` | 16-128 | Third layer units |
| `units_layer4` | 8-64 | Fourth layer units |
| `activation` | relu/elu/selu | Activation function |
| `use_batch_norm` | true/false | Batch normalization |
| `early_stopping_patience` | 10-30 | Early stopping patience |

## Architecture Comparison

### XGBoost Architecture

```
Input Features
    ↓
[Tree 1] [Tree 2] [Tree 3] ... [Tree n_estimators]
    ↓      ↓      ↓              ↓
    Sum all tree predictions
    ↓
  Sigmoid
    ↓
 Prediction
```

**Key Properties:**
- Sequential tree building
- Each tree corrects previous errors
- Feature importance via gain/split metrics
- Naturally handles missing values
- No feature scaling needed

### DNN Architecture

```
Input Features (scaled)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(32)  → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(1, sigmoid)
    ↓
Prediction
```

**Key Properties:**
- Parallel processing of all layers
- Learns hierarchical representations
- Feature importance via gradient analysis
- Requires feature scaling
- More prone to overfitting

## Optimization Methods

### XGBoost (Optuna)

```python
# Optuna TPE sampler
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=10)
)

# Optimize
study.optimize(objective, n_trials=100)

# Storage
storage = "sqlite:///analysis/optuna_studies.db"
```

**Features:**
- Trial-based optimization
- Median pruning for early stopping
- Persistent SQLite storage
- Resume capabilities
- Single or parallel optimization

### DNN (W&B Sweeps)

```python
# W&B Bayesian sweep
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'f1_score', 'goal': 'maximize'},
    'parameters': {...}
}

# Initialize
sweep_id = wandb.sweep(sweep_config, project='alpaca-squeeze-prediction')

# Run agent
wandb.agent(sweep_id, function=sweep_train)
```

**Features:**
- Run-based optimization
- Hyperband early termination
- Cloud storage (W&B)
- Web dashboard monitoring
- Multi-agent parallel optimization

## Output Comparison

### XGBoost Output Files

```
analysis/tuned_models/
├── xgboost_tuned_6pct_stop2.0.json          # XGBoost model
├── xgboost_tuned_6pct_stop2.0_scaler.pkl    # StandardScaler
└── xgboost_tuned_6pct_stop2.0_info.json     # Metadata

analysis/plots/
├── 6pct_confusion_matrix.png                # Confusion matrix
└── aligned_cumulative_profit_6pct.png       # Profit chart

analysis/optuna_studies.db                    # Optuna study storage
```

### DNN Output Files

```
analysis/tuned_models/
├── dnn_tuned_6pct_stop2.0.h5                # Keras model
├── dnn_tuned_6pct_stop2.0_scaler.pkl        # StandardScaler
└── dnn_tuned_6pct_stop2.0_info.json         # Metadata

analysis/plots/
├── 6pct_confusion_matrix_dnn.png            # Confusion matrix
└── aligned_cumulative_profit_6pct_dnn.png   # Profit chart

W&B Cloud Storage                             # W&B sweep runs
```

## Performance Characteristics

### Training Speed

| Configuration | XGBoost | DNN | Winner |
|---------------|---------|-----|--------|
| CPU (100 iterations) | 5-10 min | 20-30 min | XGBoost |
| GPU (100 iterations) | 3-5 min | 5-10 min | XGBoost |
| Single trial/run | ~3s | ~5-10s | XGBoost |

**Note**: XGBoost is generally faster for tabular data.

### Memory Usage

| Aspect | XGBoost | DNN |
|--------|---------|-----|
| Model size | 1-5 MB | 5-20 MB |
| Training memory | Low | Medium-High |
| GPU memory | Moderate | High |

### Prediction Speed

| Batch Size | XGBoost | DNN | Winner |
|------------|---------|-----|--------|
| Single sample | <1ms | ~1ms | XGBoost |
| 100 samples | ~5ms | ~10ms | XGBoost |
| 1000 samples | ~30ms | ~50ms | XGBoost |

## Interpretability

### XGBoost

**Highly Interpretable:**

```python
# Feature importance
importance = model.get_booster().get_score(importance_type='gain')

# SHAP values
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Tree visualization
xgb.plot_tree(model, num_trees=0)
```

**Pros:**
- Feature importance built-in
- SHAP values work natively
- Can visualize decision trees
- Monotonic constraints possible

### DNN

**Less Interpretable:**

```python
# Approximate feature importance via gradients
from tensorflow import GradientTape

# Layer activations
intermediate_model = keras.Model(inputs=model.input, outputs=model.layers[0].output)

# Attention mechanisms (would require architecture changes)
```

**Pros:**
- Can add attention layers
- Gradient-based importance
- Layer activation analysis

**Cons:**
- Black box by nature
- SHAP values computationally expensive
- Harder to explain to non-technical users

## When to Use Each

### Use XGBoost When:

✅ You need interpretability
✅ You have tabular data
✅ You want fast training
✅ You need feature importance
✅ Data has missing values
✅ You want battle-tested performance
✅ Stakeholders require explanations

### Use DNN When:

✅ You have large amounts of data (>100k samples)
✅ You suspect complex non-linear patterns
✅ You plan to ensemble multiple models
✅ You want to experiment with architecture
✅ You need to scale to very high dimensions
✅ You're comfortable with black-box models
✅ You want to leverage pre-trained representations (future)

### Use Both When:

🔥 You want to ensemble for better performance
🔥 You want to compare and validate results
🔥 You're exploring what works best
🔥 You have computational resources
🔥 You need confidence through multiple approaches

## Ensemble Strategy

### Simple Averaging

```python
# Load both models
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('analysis/tuned_models/xgboost_tuned_6pct_stop2.0.json')

dnn_model = keras.models.load_model('analysis/tuned_models/dnn_tuned_6pct_stop2.0.h5')

# Predict
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
dnn_proba = dnn_model.predict(X_test).flatten()

# Average
ensemble_proba = (xgb_proba + dnn_proba) / 2
ensemble_pred = (ensemble_proba >= 0.5).astype(int)
```

### Weighted Averaging

```python
# Weight by validation F1-score
xgb_weight = 0.6  # F1=0.65
dnn_weight = 0.4  # F1=0.62

ensemble_proba = (xgb_weight * xgb_proba + dnn_weight * dnn_proba)
```

### Stacking

```python
from sklearn.linear_model import LogisticRegression

# Use predictions as meta-features
meta_features = np.column_stack([xgb_proba, dnn_proba])

# Train meta-model
meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y_train)

# Predict
final_pred = meta_model.predict(meta_features_test)
```

## Shared Components

Both implementations share:

1. **Data Loading**: Same `load_and_prepare_data()` function
2. **Filtering**: Same time/price/volume filters
3. **Feature Engineering**: Same `SqueezeOutcomePredictor` pipeline
4. **Realistic Labels**: Same OCO trailing stop logic
5. **Evaluation Metrics**: Same `calculate_metrics()` function
6. **Visualization**: Same confusion matrix and profit plots
7. **W&B Logging**: Both log to W&B (XGBoost via manual logging, DNN via WandbCallback)

## Migration Path

### From XGBoost to DNN

1. Use same config structure (just different parameters)
2. Reuse same data loading code
3. Keep same evaluation pipeline
4. Same output directory structure

### From tune_models.py to dnn.py

```bash
# XGBoost
python analysis/tune_models.py train --threshold 6.0 --trials 100

# DNN equivalent
python analysis/dnn.py init-sweep --threshold 6.0
python analysis/dnn.py run-sweep --sweep-id <id> --count 100
```

## Recommendations

### For Production

1. **Start with XGBoost**: Faster, more interpretable, proven performance
2. **Experiment with DNN**: See if it finds patterns XGBoost misses
3. **Compare Results**: Use same test set for fair comparison
4. **Ensemble if Different**: If they make different mistakes, ensemble them
5. **Monitor Performance**: Track both models over time

### For Research

1. **Train Both**: Understand strengths/weaknesses
2. **Analyze Disagreements**: Where do they differ? Why?
3. **Feature Importance**: Compare XGBoost feature importance vs DNN gradients
4. **Architecture Search**: Try different DNN architectures
5. **Document Findings**: What worked? What didn't?

## Summary

| Criteria | XGBoost | DNN | Notes |
|----------|---------|-----|-------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | XGBoost simpler |
| **Training Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | XGBoost faster |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐ | XGBoost clearer |
| **Scalability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | DNN scales better |
| **Pattern Learning** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | DNN more flexible |
| **Hyperparameter Tuning** | ⭐⭐⭐⭐ | ⭐⭐⭐ | XGBoost more stable |

**Overall Winner for Tabular Data**: **XGBoost** (in most cases)

**Best Strategy**: **Train both, compare, and ensemble if beneficial**
