# Deep Neural Network (DNN) for Squeeze Alert Prediction

## Overview

This implementation provides a deep neural network alternative to the XGBoost model for predicting squeeze alert outcomes. It uses TensorFlow/Keras with W&B Sweeps for hyperparameter optimization.

## Key Features

- **Deep Feedforward Neural Network**: Multi-layer perceptron for tabular data
- **W&B Sweeps**: Bayesian hyperparameter optimization (similar to Optuna for XGBoost)
- **GPU Acceleration**: Automatic GPU detection and usage
- **Same Data Pipeline**: Reuses the exact same data loading, filtering, and evaluation pipeline as XGBoost
- **Realistic OCO Labels**: Uses trailing stop-loss logic for realistic trading outcomes
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC

## Architecture

### Model Structure

The DNN uses a feedforward architecture with:
- **Input Layer**: Number of features from the dataset
- **Hidden Layers**: 2-4 configurable dense layers with:
  - Batch Normalization (optional)
  - Activation function (ReLU, ELU, or SELU)
  - Dropout for regularization
  - L2 regularization
- **Output Layer**: Single sigmoid unit for binary classification

### Default Architecture
```
Input (n_features)
  → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
  → Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
  → Dense(32)  → BatchNorm → ReLU → Dropout(0.3)
  → Dense(1, sigmoid)
```

## Hyperparameters

### Architecture Hyperparameters
- `n_layers`: Number of hidden layers (2-4)
- `units_layer1`: Units in layer 1 (64, 128, 256, 512)
- `units_layer2`: Units in layer 2 (32, 64, 128, 256)
- `units_layer3`: Units in layer 3 (16, 32, 64, 128)
- `units_layer4`: Units in layer 4 (8, 16, 32, 64)

### Regularization Hyperparameters
- `dropout_rate`: Dropout rate (0.1 - 0.5)
- `l2_regularization`: L2 penalty (0.00001 - 0.01)

### Training Hyperparameters
- `learning_rate`: Initial learning rate (0.0001 - 0.01)
- `batch_size`: Training batch size (16, 32, 64, 128)
- `epochs`: Maximum epochs (50 - 200)
- `early_stopping_patience`: Early stopping patience (10 - 30)
- `activation`: Activation function (relu, elu, selu)
- `use_batch_norm`: Whether to use batch normalization (true/false)

## Comparison: DNN vs XGBoost

### XGBoost Parameters → DNN Equivalents

| XGBoost Parameter | DNN Equivalent | Notes |
|-------------------|----------------|-------|
| `max_depth` | `n_layers` | Tree depth → Network depth |
| `n_estimators` | `epochs` | Number of trees → Training iterations |
| `learning_rate` | `learning_rate` | Same concept, different scale |
| `min_child_weight` | `dropout_rate` | Both control overfitting |
| `gamma`, `reg_alpha`, `reg_lambda` | `l2_regularization` | Regularization strength |
| `subsample` | `batch_size` | Sample fraction → Batch training |
| `colsample_bytree` | N/A | DNNs use all features (dropout provides similar effect) |

### Optimization Methods

| XGBoost | DNN |
|---------|-----|
| Optuna with TPE sampler | W&B Sweeps with Bayesian optimization |
| MedianPruner for early stopping | Hyperband for early termination |
| SQLite storage for study persistence | W&B cloud storage |
| Trial-based optimization | Run-based optimization |

## Usage

### 1. Train a Single Model (No Sweep)

```bash
# Train with default hyperparameters
python analysis/dnn.py train --threshold 6.0 --epochs 100

# Train with custom parameters
python analysis/dnn.py train --threshold 6.0 --epochs 150 --batch-size 64

# Disable filters
python analysis/dnn.py train --threshold 6.0 --epochs 100 --no-filter-volume

# Dry run (10 epochs, no W&B)
python analysis/dnn.py train --threshold 6.0 --dry-run
```

### 2. Hyperparameter Tuning with W&B Sweeps

#### Step 1: Initialize Sweep

```bash
# Initialize sweep for 6% threshold
python analysis/dnn.py init-sweep --threshold 6.0 --config analysis/dnn_config.yaml

# With custom trailing stop
python analysis/dnn.py init-sweep --threshold 6.0 --trailing-stop 3.0

# Disable specific filters
python analysis/dnn.py init-sweep --threshold 6.0 --no-filter-volume
```

This will output:
```
✓ Sweep initialized!
  Sweep ID: abc123xyz

To run the sweep agent:
  export DNN_CONFIG_PATH='analysis/dnn_config.yaml'
  export DNN_THRESHOLD='6.0'
  export DNN_TRAILING_STOP='2.0'
  export DNN_FILTER_TIME='True'
  export DNN_FILTER_PRICE='True'
  export DNN_FILTER_VOLUME='True'
  python analysis/dnn.py run-sweep --sweep-id abc123xyz
```

#### Step 2: Run Sweep Agent

```bash
# Set environment variables (from init-sweep output)
export DNN_CONFIG_PATH='analysis/dnn_config.yaml'
export DNN_THRESHOLD='6.0'
export DNN_TRAILING_STOP='2.0'
export DNN_FILTER_TIME='True'
export DNN_FILTER_PRICE='True'
export DNN_FILTER_VOLUME='True'

# Run sweep agent (unlimited runs)
python analysis/dnn.py run-sweep --sweep-id abc123xyz

# Run sweep agent with count limit
python analysis/dnn.py run-sweep --sweep-id abc123xyz --count 50
```

#### Step 3: Monitor on W&B Dashboard

Open the W&B dashboard at: https://wandb.ai/your-username/alpaca-squeeze-prediction

### 3. Parallel Sweep Agents

You can run multiple sweep agents in parallel:

```bash
# Terminal 1
python analysis/dnn.py run-sweep --sweep-id abc123xyz --count 25

# Terminal 2
python analysis/dnn.py run-sweep --sweep-id abc123xyz --count 25

# Terminal 3
python analysis/dnn.py run-sweep --sweep-id abc123xyz --count 25
```

## Data Filters

Same as XGBoost - filters are **ENABLED BY DEFAULT**:

- **Time Filter**: 09:45 - 16:00 ET
- **Price Filter**: $2.00 - $10.00
- **Volume Filter**: Avg volume > 80,000 in 10 min before alert

Disable with:
- `--no-filter-time`
- `--no-filter-price`
- `--no-filter-volume`

## Output Files

### Model Files (saved to `analysis/tuned_models/`)

```
dnn_tuned_6pct_stop2.0.h5              # Keras model
dnn_tuned_6pct_stop2.0_scaler.pkl      # StandardScaler
dnn_tuned_6pct_stop2.0_info.json       # Metadata and metrics
```

### Plots (saved to `analysis/plots/`)

```
6pct_confusion_matrix_dnn.png          # Confusion matrix
aligned_cumulative_profit_6pct_dnn.png # Cumulative profit chart
```

### Metadata JSON Structure

```json
{
  "threshold": 6.0,
  "gain_threshold": 6.0,
  "trailing_stop_pct": 2.0,
  "model_type": "dnn",
  "test_metrics": {
    "accuracy": 0.65,
    "precision": 0.63,
    "recall": 0.68,
    "f1_score": 0.65,
    "roc_auc": 0.70
  },
  "feature_names": [...],
  "training_info": {
    "timestamp": "2025-01-03T...",
    "gpu_used": true,
    "tensorflow_version": "2.x.x"
  },
  "outcome_categories": {...}
}
```

## W&B Metrics Logged

### Per-Run Metrics
- `accuracy`: Test set accuracy
- `precision`: Test set precision
- `recall`: Test set recall
- `f1_score`: Test set F1-score (optimization metric)
- `roc_auc`: Test set ROC-AUC
- `best_epoch`: Best epoch (lowest val_loss)
- `best_val_loss`: Best validation loss

### Training Curves (via WandbCallback)
- `loss`: Training loss per epoch
- `val_loss`: Validation loss per epoch
- `accuracy`: Training accuracy per epoch
- `val_accuracy`: Validation accuracy per epoch
- `precision`: Training precision per epoch
- `val_precision`: Validation precision per epoch
- `recall`: Training recall per epoch
- `val_recall`: Validation recall per epoch
- `auc`: Training AUC per epoch
- `val_auc`: Validation AUC per epoch

## Configuration File

Edit `analysis/dnn_config.yaml` to customize:

### Data Configuration
```yaml
data:
  json_dir: "/path/to/historical_data"
  features_csv: "analysis/squeeze_alerts_independent_features.csv"
  train_end_date: "2025-12-22"
  test_size: 0.20
  random_seed: 42
```

### Search Space
```yaml
dnn_search_space:
  n_layers:
    type: "int"
    min: 2
    max: 4
  learning_rate:
    type: "log_uniform"
    min: 0.0001
    max: 0.01
  # ... more parameters
```

### W&B Sweep Settings
```yaml
wandb_sweep:
  method: "bayes"  # bayes, grid, random
  metric: "f1_score"
  goal: "maximize"
  early_terminate: true
```

## GPU Usage

### Automatic GPU Detection

The script automatically detects and uses GPUs:

```
✓ GPU detected: 1 device(s)
  Device name: /device:GPU:0
  TensorFlow version: 2.x.x
```

### GPU Memory Growth

TensorFlow is configured to allocate GPU memory as needed (not all at once):

```python
tf.config.experimental.set_memory_growth(gpu, True)
```

### Force CPU Usage

```bash
# Disable GPU
export CUDA_VISIBLE_DEVICES=""
python analysis/dnn.py train --threshold 6.0 --epochs 100
```

## Advanced Usage

### Custom Hyperparameters (Single Training)

Modify `analysis/dnn_config.yaml`:

```yaml
dnn_hyperparameters:
  n_layers: 4
  units_layer1: 256
  units_layer2: 128
  units_layer3: 64
  units_layer4: 32
  dropout_rate: 0.4
  l2_regularization: 0.005
  learning_rate: 0.0005
  activation: "elu"
  use_batch_norm: true
```

Then train:

```bash
python analysis/dnn.py train --threshold 6.0 --epochs 100
```

### Multiple Thresholds

Train models for different thresholds:

```bash
for threshold in 1.5 2.0 2.5 3.0 4.0 5.0 6.0 7.0; do
  python analysis/dnn.py train --threshold $threshold --epochs 100
done
```

### Custom Time and Price Ranges

Currently, time/price ranges are hardcoded in the filter function. To customize, edit the `filter_dataframe()` call in `load_and_prepare_data()`.

## Troubleshooting

### 1. Out of Memory (GPU)

Reduce batch size:

```bash
python analysis/dnn.py train --threshold 6.0 --batch-size 16
```

Or use CPU:

```bash
export CUDA_VISIBLE_DEVICES=""
python analysis/dnn.py train --threshold 6.0
```

### 2. Sweep Not Finding DNN_CONFIG_PATH

Make sure environment variables are set before running sweep agent:

```bash
export DNN_CONFIG_PATH='analysis/dnn_config.yaml'
export DNN_THRESHOLD='6.0'
export DNN_TRAILING_STOP='2.0'
# ... other variables
python analysis/dnn.py run-sweep --sweep-id <id>
```

### 3. W&B Not Logging

Check W&B mode in config:

```yaml
wandb:
  mode: "online"  # Change from "disabled" or "offline"
```

Login to W&B:

```bash
wandb login
```

### 4. TensorFlow Warnings

Suppress TensorFlow warnings:

```bash
export TF_CPP_MIN_LOG_LEVEL=2
python analysis/dnn.py train --threshold 6.0
```

## Performance Comparison

### Expected Training Times (on GPU)

| Configuration | Time per Epoch | Total Time (100 epochs) |
|---------------|----------------|-------------------------|
| Batch=16, 3 layers | ~5s | ~8 min |
| Batch=32, 3 layers | ~3s | ~5 min |
| Batch=64, 4 layers | ~4s | ~7 min |
| Batch=128, 2 layers | ~2s | ~3 min |

### Expected Performance (F1-Score)

Based on similar tabular classification tasks:

| Model | Expected F1-Score Range |
|-------|------------------------|
| XGBoost (tuned) | 0.60 - 0.70 |
| DNN (tuned) | 0.58 - 0.68 |
| Baseline (random) | 0.50 |

**Note**: DNNs typically perform similarly to XGBoost on tabular data, sometimes slightly worse. The advantage is in the ability to:
- Learn complex non-linear relationships
- Handle high-dimensional sparse features
- Transfer learn from pre-trained representations
- Ensemble with other models

## Best Practices

### 1. Start with Single Training

Before running sweeps, train a single model to verify data loading works:

```bash
python analysis/dnn.py train --threshold 6.0 --epochs 10 --dry-run
```

### 2. Use Dry Run for Testing

Always test with `--dry-run` first:

```bash
python analysis/dnn.py train --threshold 6.0 --dry-run
```

### 3. Monitor Early Stopping

Watch for early stopping in logs:

```
Epoch 45: early stopping
Restoring model weights from the end of the best epoch: 25
```

This indicates the model is converging well.

### 4. Check for Overfitting

Compare training vs validation metrics:

```
Epoch 100/100
loss: 0.45 - val_loss: 0.62  # Gap indicates overfitting
```

If overfitting, increase dropout or L2 regularization.

### 5. Balance Sweep Count vs Time

- **Quick exploration**: 20-30 runs
- **Thorough search**: 50-100 runs
- **Production**: 100-200 runs

## References

- **XGBoost Implementation**: `analysis/tune_models.py`
- **Data Pipeline**: `analysis/predict_squeeze_outcomes.py`
- **W&B Sweeps Documentation**: https://docs.wandb.ai/guides/sweeps
- **TensorFlow Documentation**: https://www.tensorflow.org/api_docs
- **Keras Documentation**: https://keras.io/api/
