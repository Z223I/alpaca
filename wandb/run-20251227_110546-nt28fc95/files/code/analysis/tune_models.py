#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning with Optuna and Weights & Biases

This script provides a comprehensive hyperparameter tuning system for XGBoost
using Bayesian optimization (Optuna) with experiment tracking (W&B) and GPU acceleration.

Usage:
    # Single threshold tuning
    python analysis/tune_models.py train --threshold 6.0 --trials 100

    # Single threshold with custom end date
    python analysis/tune_models.py train --threshold 6.0 --trials 100 --end-date 2025-12-20

    # All thresholds
    python analysis/tune_models.py train --all-thresholds --trials 50

    # Resume previous study
    python analysis/tune_models.py train --threshold 6.0 --resume

    # Dry run
    python analysis/tune_models.py train --threshold 6.0 --trials 5 --dry-run
"""

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yaml

# ML imports
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Plotting imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Optuna imports
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial, TrialState
try:
    from optuna.integration import XGBoostPruningCallback
except ImportError:
    print("Warning: XGBoostPruningCallback not available in this version of Optuna")
    XGBoostPruningCallback = None

# W&B import
import wandb

# Import from existing prediction script
sys.path.append(str(Path(__file__).parent))
from predict_squeeze_outcomes import SqueezeOutcomePredictor
from atoms_analysis.plotting import generate_aligned_cumulative_profit_plot


# =============================================================================
# GPU Detection
# =============================================================================

def detect_gpu() -> Tuple[bool, str]:
    """
    Detect if NVIDIA GPU is available for XGBoost training.

    Returns:
        Tuple of (has_gpu: bool, tree_method: str)
        - has_gpu: True if GPU detected
        - tree_method: "gpu_hist" if GPU available, "hist" otherwise
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()
            print(f"✓ GPU detected: {gpu_name}")
            print(f"  Using tree_method='gpu_hist' for GPU acceleration")
            return True, "gpu_hist"
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"⚠️  GPU detection failed: {e}")

    print("⚠️  No GPU detected. Using CPU with tree_method='hist'")
    return False, "hist"


# =============================================================================
# Data Loading
# =============================================================================

def load_and_prepare_data(
    threshold: float,
    config: Dict,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, List[str], SqueezeOutcomePredictor]:
    """
    Load and prepare data using existing SqueezeOutcomePredictor pipeline.

    Args:
        threshold: Gain threshold (e.g., 6.0 for 6%)
        config: Configuration dictionary
        end_date: Optional end date filter (format: "YYYY-MM-DD")

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, feature_names, predictor)
    """
    print("\n" + "="*80)
    print(f"LOADING DATA FOR {threshold}% THRESHOLD")
    print("="*80)

    # Create predictor instance
    predictor = SqueezeOutcomePredictor(
        json_dir=config['data']['json_dir'],
        features_csv=config['data']['features_csv']
    )

    # Extract outcomes
    outcomes_df = predictor.extract_outcomes(
        gain_threshold=threshold,
        start_date="2025-12-12",
        end_date=end_date
    )
    print(f"✓ Extracted outcomes: {len(outcomes_df)} alerts")

    # Merge with features
    merged_df = predictor.merge_with_features(outcomes_df)
    print(f"✓ Merged with features: {len(merged_df)} samples")

    # Create target variable based on REALISTIC outcomes (OCO trailing stop logic)
    target_col = f"achieved_{threshold}pct"
    if target_col not in merged_df.columns:
        print(f"\n⚙️  Calculating realistic outcomes with 2% trailing stop...")

        import json
        from pathlib import Path

        TRAILING_STOP_PCT = 2.0
        realistic_outcomes = []
        json_loaded = 0
        json_missing = 0

        for idx in merged_df.index:
            row = merged_df.loc[idx]
            symbol = row.get('symbol', 'N/A')
            timestamp = row.get('timestamp', '')

            try:
                # Load JSON with interval data
                date_obj = pd.to_datetime(timestamp)
                date_str = date_obj.strftime('%Y-%m-%d')
                time_str = date_obj.strftime('%H%M%S')
                json_file = Path(config['data']['json_dir']) / date_str / 'squeeze_alerts_sent' / f'alert_{symbol}_{date_str}_{time_str}.json'

                if json_file.exists():
                    json_loaded += 1
                    with open(json_file, 'r') as f:
                        full_data = json.load(f)

                    # Create enhanced row with outcome_tracking
                    enhanced_row = row.copy()
                    if 'outcome_tracking' in full_data:
                        enhanced_row['outcome_tracking'] = full_data['outcome_tracking']

                    # Calculate realistic outcome using OCO trailing stop logic
                    gain, reason, holding_time = predictor._calculate_trade_outcome(
                        enhanced_row, threshold, TRAILING_STOP_PCT
                    )

                    # Label as "achieved" if we actually made profit (or hit target)
                    # Using >= 0 means any profit counts as success
                    # Using >= threshold means we need to actually hit the target
                    realistic_outcomes.append(1 if gain >= threshold else 0)
                else:
                    json_missing += 1
                    # Fallback to max_gain if no interval data
                    realistic_outcomes.append(1 if row.get('max_gain_percent', 0) >= threshold else 0)
            except Exception as e:
                # Fallback on error
                realistic_outcomes.append(1 if row.get('max_gain_percent', 0) >= threshold else 0)

        merged_df[target_col] = realistic_outcomes

        print(f"✓ Realistic outcomes calculated:")
        print(f"  - JSON files loaded: {json_loaded}")
        print(f"  - JSON files missing: {json_missing}")
        print(f"  - Achieved (realistic): {sum(realistic_outcomes)} ({sum(realistic_outcomes)/len(realistic_outcomes)*100:.1f}%)")

        # Compare to old max_gain labels
        old_labels = (merged_df['max_gain_percent'] >= threshold).astype(int)
        print(f"  - Achieved (max_gain): {sum(old_labels)} ({sum(old_labels)/len(old_labels)*100:.1f}%)")
        print(f"  - Label difference: {abs(sum(old_labels) - sum(realistic_outcomes))} samples ({abs(sum(old_labels) - sum(realistic_outcomes))/len(old_labels)*100:.1f}%)")

    # Prepare features (handle missing data, encode categoricals)
    X, y = predictor.prepare_features(merged_df, target_variable=target_col)
    feature_names = predictor.feature_names
    print(f"✓ Prepared {len(feature_names)} features")

    # Time-based split
    X_train, X_test, y_train, y_test = predictor.time_based_split(
        merged_df, X, y, test_size=config['data']['test_size']
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, predictor


# =============================================================================
# Hyperparameter Suggestion
# =============================================================================

def suggest_hyperparameters(trial: Trial, config: Dict, tree_method: str) -> Dict:
    """
    Suggest hyperparameters for XGBoost based on search space in config.

    Args:
        trial: Optuna trial object
        config: Configuration dictionary with search space
        tree_method: "gpu_hist" or "hist"

    Returns:
        Dictionary of XGBoost parameters
    """
    search_space = config['xgboost_search_space']
    params = {}

    # Suggest tunable parameters
    for param_name, param_config in search_space.items():
        if param_name == 'fixed':
            continue

        param_type = param_config['type']

        if param_type == 'int':
            params[param_name] = trial.suggest_int(
                param_name,
                param_config['low'],
                param_config['high'],
                step=param_config.get('step', 1)
            )
        elif param_type == 'uniform':
            params[param_name] = trial.suggest_float(
                param_name,
                param_config['low'],
                param_config['high']
            )
        elif param_type == 'loguniform':
            params[param_name] = trial.suggest_float(
                param_name,
                param_config['low'],
                param_config['high'],
                log=True
            )

    # Add fixed parameters
    params.update(search_space['fixed'])

    # Add tree method (GPU or CPU)
    params['tree_method'] = tree_method

    # If GPU, set GPU-specific params
    if tree_method == 'gpu_hist':
        params['gpu_id'] = config['gpu']['gpu_id']
        params['predictor'] = 'gpu_predictor'
        params['n_jobs'] = 1  # GPU doesn't need multi-threading

    return params


# =============================================================================
# Cross-Validation
# =============================================================================

def time_series_cross_validation(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    scale_pos_weight: float = 1.0
) -> Dict[str, float]:
    """
    Perform time-series cross-validation that respects temporal order.

    Args:
        model: XGBoost model (will be cloned for each fold)
        X: Feature matrix
        y: Target variable
        n_folds: Number of CV folds
        scale_pos_weight: Class weight for imbalance

    Returns:
        Dictionary with mean and std for each metric
    """
    tscv = TimeSeriesSplit(n_splits=n_folds)

    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # Split data
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Clone model with same params
        fold_model = xgb.XGBClassifier(**model.get_params())
        fold_model.set_params(scale_pos_weight=scale_pos_weight)

        # Train
        fold_model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )

        # Evaluate
        y_pred = fold_model.predict(X_val_fold)
        y_proba = fold_model.predict_proba(X_val_fold)[:, 1]

        scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
        scores['precision'].append(
            precision_score(y_val_fold, y_pred, zero_division=0)
        )
        scores['recall'].append(
            recall_score(y_val_fold, y_pred, zero_division=0)
        )
        scores['f1_score'].append(
            f1_score(y_val_fold, y_pred, zero_division=0)
        )
        scores['roc_auc'].append(roc_auc_score(y_val_fold, y_proba))

    # Aggregate scores
    cv_results = {}
    for metric, values in scores.items():
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)

    return cv_results


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities

    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    output_dir: Path = Path("analysis/plots")
) -> Path:
    """
    Generate and save confusion matrix plot.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        threshold: Gain threshold percentage
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot using seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Count'},
        xticklabels=['Did Not Achieve', 'Achieved'],
        yticklabels=['Did Not Achieve', 'Achieved']
    )

    # Add labels and title
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix - {threshold}% Gain Threshold\nBest Model Performance',
              fontsize=14, fontweight='bold')

    # Add metrics annotation
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Add text box with metrics
    metrics_text = (
        f'Accuracy: {accuracy:.3f}\n'
        f'Precision: {precision:.3f}\n'
        f'Recall: {recall:.3f}\n'
        f'F1-Score: {f1:.3f}\n'
        f'\n'
        f'Total: {total:,} samples'
    )
    plt.text(
        1.5, 0.5, metrics_text,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='center',
        transform=plt.gca().transAxes
    )

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"{threshold}pct_confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def calculate_realistic_profits_and_plot(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    threshold: float,
    predictor: SqueezeOutcomePredictor,
    config: Dict
) -> Path:
    """
    Calculate realistic trading profits using trailing stop logic and generate cumulative profit chart.

    This mirrors the logic from predict_squeeze_outcomes.py:
    - Uses 2% trailing stop loss
    - Processes interval data chronologically
    - Compares model predictions vs take-all strategy

    Args:
        X_test: Test feature matrix
        y_test: Test labels
        y_pred: Model predictions
        threshold: Gain threshold percentage
        predictor: SqueezeOutcomePredictor instance
        config: Configuration dictionary

    Returns:
        Path to saved cumulative profit plot
    """
    print(f"\nCalculating realistic trading profits with trailing stop...")

    TRAILING_STOP_PCT = 2.0

    # Load the merged data to get outcome_tracking information
    # We need to reload because X_test doesn't have the raw outcome data
    json_dir = config['data']['json_dir']
    features_csv = config['data']['features_csv']

    # Re-extract outcomes to get the full dataframe
    outcomes_df = predictor.extract_outcomes(
        gain_threshold=threshold,
        start_date="2025-12-12",
        end_date=config.get('end_date')
    )

    merged_df = predictor.merge_with_features(outcomes_df)

    # Get the test set indices (they should match X_test)
    test_df = merged_df.loc[X_test.index].copy()

    # Add predictions
    test_df['predicted_outcome'] = y_pred
    test_df['actual_outcome'] = y_test.values

    # Calculate realistic profit for each test sample
    realistic_profits = []

    for idx in test_df.index:
        outcome_row = test_df.loc[idx]

        try:
            # Try to load full JSON with interval data
            symbol = outcome_row.get('symbol', 'N/A')
            timestamp = outcome_row.get('timestamp', '')

            if timestamp:
                date_obj = pd.to_datetime(timestamp)
                date_str = date_obj.strftime('%Y-%m-%d')
                time_str = date_obj.strftime('%H%M%S')
                json_file = Path(json_dir) / date_str / 'squeeze_alerts_sent' / f'alert_{symbol}_{date_str}_{time_str}.json'

                if json_file.exists():
                    with open(json_file, 'r') as f:
                        full_data = json.load(f)

                    enhanced_row = outcome_row.copy()
                    if 'outcome_tracking' in full_data:
                        enhanced_row['outcome_tracking'] = full_data['outcome_tracking']

                    gain, _, _ = predictor._calculate_trade_outcome(
                        enhanced_row, threshold, TRAILING_STOP_PCT
                    )
                    realistic_profits.append(gain)
                else:
                    # Fallback without interval data
                    gain, _, _ = predictor._calculate_trade_outcome(
                        outcome_row, threshold, TRAILING_STOP_PCT
                    )
                    realistic_profits.append(gain)
            else:
                gain, _, _ = predictor._calculate_trade_outcome(
                    outcome_row, threshold, TRAILING_STOP_PCT
                )
                realistic_profits.append(gain)

        except Exception as e:
            # Fallback on error
            gain, _, _ = predictor._calculate_trade_outcome(
                outcome_row, threshold, TRAILING_STOP_PCT
            )
            realistic_profits.append(gain)

    test_df['realistic_profit'] = realistic_profits

    # Separate into model trades (predictions == 1) and all trades
    model_trades = test_df[test_df['predicted_outcome'] == 1].copy()

    print(f"✓ Calculated realistic profits for {len(test_df)} test samples")
    print(f"  Model selected {len(model_trades)} trades ({len(model_trades)/len(test_df)*100:.1f}%)")
    print(f"  Average profit (take-all): {test_df['realistic_profit'].mean():.2f}%")
    if len(model_trades) > 0:
        print(f"  Average profit (model): {model_trades['realistic_profit'].mean():.2f}%")

    # Generate cumulative profit plot
    print(f"\nGenerating cumulative profit chart...")
    threshold_suffix = f"_{threshold}pct"
    generate_aligned_cumulative_profit_plot(
        predictions_df=test_df,
        model_trades=model_trades,
        threshold_suffix=threshold_suffix,
        gain_threshold=threshold
    )

    plot_path = Path("analysis/plots") / f"aligned_cumulative_profit{threshold_suffix}.png"
    return plot_path


# =============================================================================
# W&B Logging
# =============================================================================

def initialize_wandb(config: Dict, threshold: float, args) -> wandb.sdk.wandb_run.Run:
    """
    Initialize Weights & Biases run.

    Args:
        config: Configuration dictionary
        threshold: Gain threshold
        args: Command-line arguments

    Returns:
        W&B run object
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = f"tune-xgb-{threshold}pct-{timestamp}"

    tags = config['wandb']['tags'].copy()
    tags.append(f"{threshold}pct")

    run = wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=run_name,
        config={
            'threshold': threshold,
            'n_trials': args.trials,
            'cv_folds': args.cv_folds if hasattr(args, 'cv_folds') else config['cross_validation']['n_folds'],
            'optimization_metric': config['optimization']['primary_metric'],
            'search_space': config['xgboost_search_space'],
        },
        tags=tags,
        mode=config['wandb']['mode'],
        save_code=config['wandb']['save_code']
    )

    return run


def log_trial_to_wandb(
    trial: Trial,
    params: Dict,
    metrics: Dict,
    cv_results: Optional[Dict],
    threshold: float,
    training_time: float
):
    """
    Log trial results to Weights & Biases.

    Args:
        trial: Optuna trial object
        params: Hyperparameters used
        metrics: Test set metrics
        cv_results: Cross-validation results (optional)
        threshold: Gain threshold
        training_time: Training time in seconds
    """
    log_dict = {
        'trial_number': trial.number,
        'training_time_seconds': training_time,
    }

    # Log hyperparameters
    for param_name, param_value in params.items():
        if param_name not in ['objective', 'eval_metric', 'random_state', 'verbosity']:
            log_dict[f'params/{param_name}'] = param_value

    # Log test metrics
    for metric_name, metric_value in metrics.items():
        log_dict[f'test/{metric_name}'] = metric_value

    # Log CV metrics if available
    if cv_results:
        for metric_name, metric_value in cv_results.items():
            log_dict[f'cv/{metric_name}'] = metric_value

    wandb.log(log_dict)


def log_study_results(study: optuna.Study, threshold: float):
    """
    Log Optuna study results and visualizations to W&B.

    Args:
        study: Completed Optuna study
        threshold: Gain threshold
    """
    import plotly

    print("\nLogging study results to W&B...")

    # Summary statistics
    wandb.run.summary.update({
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'total_trials': len(study.trials),
        'pruned_trials': len([t for t in study.trials if t.state == TrialState.PRUNED]),
        'failed_trials': len([t for t in study.trials if t.state == TrialState.FAIL]),
        'complete_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
    })

    # Visualizations
    try:
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        wandb.log({"optuna/optimization_history": wandb.Plotly(fig)})

        # Parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        wandb.log({"optuna/param_importances": wandb.Plotly(fig)})

        # Parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study)
        wandb.log({"optuna/parallel_coordinate": wandb.Plotly(fig)})

        # Slice plot
        fig = optuna.visualization.plot_slice(study)
        wandb.log({"optuna/slice": wandb.Plotly(fig)})

        # EDF plot
        fig = optuna.visualization.plot_edf(study)
        wandb.log({"optuna/edf": wandb.Plotly(fig)})

        print("✓ Optuna visualizations logged to W&B")

    except Exception as e:
        print(f"⚠️  Warning: Could not create some visualizations: {e}")


# =============================================================================
# Objective Function
# =============================================================================

def objective(
    trial: Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict,
    threshold: float,
    tree_method: str
) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_test, y_test: Test/validation data
        config: Configuration dictionary
        threshold: Gain threshold
        tree_method: "gpu_hist" or "hist"

    Returns:
        Primary metric value (F1-score by default)
    """
    import time
    start_time = time.time()

    # Suggest hyperparameters
    params = suggest_hyperparameters(trial, config, tree_method)

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params['scale_pos_weight'] = scale_pos_weight

    # Create model
    model = xgb.XGBClassifier(**params)

    # Cross-validation or single train/val split
    if config['cross_validation']['enabled']:
        # Time-series CV
        cv_results = time_series_cross_validation(
            model, X_train, y_train,
            n_folds=config['cross_validation']['n_folds'],
            scale_pos_weight=scale_pos_weight
        )

        # Use CV score as primary metric
        primary_metric_value = cv_results[f"{config['optimization']['primary_metric']}_mean"]

        # Also evaluate on test set for logging
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_pred, y_proba)

    else:
        # Single train/validation split with pruning
        eval_set = [(X_test, y_test)]
        callbacks = []

        if XGBoostPruningCallback is not None:
            callbacks.append(XGBoostPruningCallback(trial, "validation_0-logloss"))

        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            callbacks=callbacks if callbacks else None
        )

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_pred, y_proba)

        primary_metric_value = test_metrics[config['optimization']['primary_metric']]
        cv_results = None

    # Log to W&B
    training_time = time.time() - start_time
    log_trial_to_wandb(
        trial, params, test_metrics, cv_results, threshold, training_time
    )

    return primary_metric_value


# =============================================================================
# Model Training and Saving
# =============================================================================

def train_final_model(
    best_params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict
) -> xgb.XGBClassifier:
    """
    Train final model with best parameters on full training set.

    Args:
        best_params: Best hyperparameters from Optuna
        X_train: Full training set
        y_train: Full training labels
        config: Configuration dictionary

    Returns:
        Trained XGBoost model
    """
    print("\nTraining final model with best parameters...")

    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    best_params['scale_pos_weight'] = scale_pos_weight

    # Create and train model
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=True)

    print(f"✓ Final model trained ({model.n_estimators} trees)")

    return model


def save_and_log_best_model(
    model: xgb.XGBClassifier,
    study: optuna.Study,
    test_metrics: Dict,
    cv_results: Optional[Dict],
    threshold: float,
    scaler: StandardScaler,
    feature_names: List[str],
    baseline_metrics: Dict,
    config: Dict,
    cm_plot_path: Optional[Path] = None
):
    """
    Save best model and log as W&B artifact.

    Args:
        model: Trained XGBoost model
        study: Completed Optuna study
        test_metrics: Test set metrics
        cv_results: Cross-validation results
        threshold: Gain threshold
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        baseline_metrics: Baseline model metrics for comparison
        config: Configuration dictionary
        cm_plot_path: Path to confusion matrix plot (optional)
    """
    print(f"\nSaving best model for {threshold}% threshold...")

    # Create output directory
    output_dir = Path("analysis/tuned_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / f"xgboost_tuned_{threshold}pct.json"
    model.save_model(model_path)
    print(f"✓ Saved model: {model_path}")

    # Save scaler
    scaler_path = output_dir / f"xgboost_tuned_{threshold}pct_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: {scaler_path}")

    # Save metadata
    info = {
        'threshold': threshold,
        'study_name': study.study_name,
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'test_metrics': test_metrics,
        'cv_results': cv_results,
        'feature_names': feature_names,
        'baseline_comparison': {
            'baseline_f1': baseline_metrics.get('f1_score', 0),
            'tuned_f1': test_metrics['f1_score'],
            'improvement_percent': (
                (test_metrics['f1_score'] - baseline_metrics.get('f1_score', 0)) /
                baseline_metrics.get('f1_score', 1) * 100
            ) if baseline_metrics.get('f1_score') else 0,
        },
        'training_info': {
            'n_trials': len(study.trials),
            'pruned_trials': len([t for t in study.trials if t.state == TrialState.PRUNED]),
            'timestamp': datetime.now().isoformat(),
            'gpu_used': 'gpu_hist' in str(study.best_params.get('tree_method', '')),
        }
    }

    info_path = Path(f"analysis/best_params_{threshold}pct.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✓ Saved metadata: {info_path}")

    # Log as W&B artifact
    try:
        artifact = wandb.Artifact(
            name=f"xgboost-tuned-{threshold}pct",
            type="model",
            description=f"Best XGBoost model for {threshold}% gain threshold",
            metadata={
                'threshold': threshold,
                'best_f1': test_metrics['f1_score'],
                'best_params': study.best_params,
            }
        )
        artifact.add_file(str(model_path))
        artifact.add_file(str(scaler_path))
        artifact.add_file(str(info_path))

        # Add confusion matrix plot if available
        if cm_plot_path and cm_plot_path.exists():
            artifact.add_file(str(cm_plot_path))
            # Also log as image to W&B
            wandb.log({f"confusion_matrix_{threshold}pct": wandb.Image(str(cm_plot_path))})

        wandb.log_artifact(artifact)
        print(f"✓ Logged artifact to W&B")
    except Exception as e:
        print(f"⚠️  Warning: Could not log artifact to W&B: {e}")


# =============================================================================
# Main Tuning Orchestrator
# =============================================================================

def tune_for_threshold(
    threshold: float,
    config: Dict,
    args
) -> Dict:
    """
    Main tuning orchestrator for a single threshold.

    Args:
        threshold: Gain threshold percentage
        config: Configuration dictionary
        args: Command-line arguments

    Returns:
        Dictionary with tuning results
    """
    print("\n" + "="*80)
    print(f"HYPERPARAMETER TUNING FOR {threshold}% GAIN THRESHOLD")
    print("="*80)

    # Detect GPU
    has_gpu, tree_method = detect_gpu()
    if args.use_gpu and not has_gpu:
        print("⚠️  --use-gpu specified but no GPU detected. Using CPU.")
        tree_method = "hist"
    elif not args.use_gpu:
        tree_method = "hist"
        print("GPU disabled by user (--use-gpu not specified)")

    # Initialize W&B
    wandb_run = initialize_wandb(config, threshold, args)

    try:
        # Load data
        X_train, X_test, y_train, y_test, scaler, feature_names, predictor = load_and_prepare_data(
            threshold, config, end_date=args.end_date if hasattr(args, 'end_date') else None
        )

        # Create or resume Optuna study
        study_name = args.study_name if hasattr(args, 'study_name') and args.study_name else f"xgb-{threshold}pct-study"
        storage = config['optuna']['study_storage']

        if args.resume and hasattr(args, 'study_name'):
            print(f"\nResuming study: {study_name}")
            study = optuna.load_study(study_name=study_name, storage=storage)
        else:
            print(f"\nCreating new study: {study_name}")
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=config['optimization']['direction'],
                sampler=TPESampler(seed=config['data']['random_seed']),
                pruner=MedianPruner(
                    n_startup_trials=config['optuna']['pruner_config']['n_startup_trials'],
                    n_warmup_steps=config['optuna']['pruner_config']['n_warmup_steps'],
                    interval_steps=config['optuna']['pruner_config']['interval_steps']
                ),
                load_if_exists=True
            )

        # Run optimization
        print(f"\nStarting optimization with {args.trials} trials...")
        print(f"Optimizing metric: {config['optimization']['primary_metric']}")

        study.optimize(
            lambda trial: objective(
                trial, X_train, y_train, X_test, y_test,
                config, threshold, tree_method
            ),
            n_trials=args.trials,
            timeout=config['optuna']['timeout'],
            n_jobs=config['optuna']['n_jobs'],
            show_progress_bar=True
        )

        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best {config['optimization']['primary_metric']}: {study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")

        # Log study results to W&B
        log_study_results(study, threshold)

        # Train final model with best params
        final_model = train_final_model(study.best_params, X_train, y_train, config)

        # Evaluate on test set
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]
        final_metrics = calculate_metrics(y_test, y_pred, y_proba)

        print(f"\nFinal test set performance:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Generate and save confusion matrix
        print(f"\nGenerating confusion matrix...")
        cm_plot_path = plot_confusion_matrix(y_test, y_pred, threshold)
        print(f"✓ Saved confusion matrix: {cm_plot_path}")

        # Generate cumulative profit chart with trailing stop logic
        cumulative_profit_path = calculate_realistic_profits_and_plot(
            X_test, y_test, y_pred, threshold, predictor, config
        )
        print(f"✓ Saved cumulative profit chart: {cumulative_profit_path}")

        # Get baseline metrics for comparison
        baseline_metrics = config.get('baseline', {})

        # Cross-validation results
        cv_results = None
        if config['cross_validation']['enabled']:
            cv_results = time_series_cross_validation(
                final_model, X_train, y_train,
                n_folds=config['cross_validation']['n_folds'],
                scale_pos_weight=study.best_params.get('scale_pos_weight', 1.0)
            )

        # Save model and artifacts
        save_and_log_best_model(
            final_model, study, final_metrics, cv_results,
            threshold, scaler, feature_names, baseline_metrics, config,
            cm_plot_path=cm_plot_path
        )

        # Finish W&B run
        wandb.finish()

        return {
            'study': study,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'final_metrics': final_metrics,
            'cv_results': cv_results,
            'model': final_model
        }

    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish(exit_code=1)
        raise


# =============================================================================
# CLI and Main Entry Point
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="XGBoost Hyperparameter Tuning with Optuna and W&B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from project root):
  # Train model for single threshold
  python analysis/tune_models.py train --threshold 6.0 --trials 100

  # Train with custom end date
  python analysis/tune_models.py train --threshold 6.0 --trials 100 --end-date 2025-12-20

  # Train all thresholds
  python analysis/tune_models.py train --all-thresholds --trials 50
        """
    )

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train models with hyperparameter tuning')
    train_parser.add_argument(
        '--threshold', type=float, default=6.0,
        help='Gain threshold percentage (e.g., 6.0 for 6%%)'
    )
    train_parser.add_argument(
        '--all-thresholds', action='store_true',
        help='Tune for all thresholds (1.5%%, 2%%, 2.5%%, 3%%, 4%%, 5%%, 6%%, 7%%)'
    )
    train_parser.add_argument(
        '--trials', type=int, default=100,
        help='Number of Optuna trials'
    )
    train_parser.add_argument(
        '--cv-folds', type=int, default=None,
        help='Number of cross-validation folds (overrides config)'
    )
    train_parser.add_argument(
        '--metric', type=str, default=None,
        help='Optimization metric (overrides config)'
    )
    train_parser.add_argument(
        '--use-gpu', action='store_true',
        help='Use GPU acceleration (auto-detected by default)'
    )
    train_parser.add_argument(
        '--resume', action='store_true',
        help='Resume previous study'
    )
    train_parser.add_argument(
        '--study-name', type=str, default=None,
        help='Study name for resume (default: auto-generated)'
    )
    train_parser.add_argument(
        '--dry-run', action='store_true',
        help='Dry run mode (limited trials for testing)'
    )
    train_parser.add_argument(
        '--config', type=str, default='analysis/tuning_config.yaml',
        help='Path to configuration file'
    )
    train_parser.add_argument(
        '--end-date', type=str, default='2025-12-22',
        help='Training data cutoff date (YYYY-MM-DD). Data after this date is reserved for prediction/testing. Default: 2025-12-22'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Check if mode is specified
    if not args.mode:
        print("❌ Error: No mode specified. Use 'train' subcommand.")
        print("\nUsage: python analysis/tune_models.py train --threshold 6.0 --trials 100")
        print("Run with --help for more information.")
        sys.exit(1)

    if args.mode == 'train':
        # Training mode
        # Load configuration
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Override config with CLI arguments
        if args.trials:
            config['optuna']['n_trials'] = args.trials
        if args.cv_folds:
            config['cross_validation']['n_folds'] = args.cv_folds
        if args.metric:
            config['optimization']['primary_metric'] = args.metric

        # Dry run mode
        if args.dry_run:
            print("\n🧪 DRY RUN MODE - Using 5 trials for testing")
            args.trials = 5
            config['optuna']['n_trials'] = 5
            config['wandb']['mode'] = 'disabled'

        # Determine thresholds to tune
        if args.all_thresholds:
            thresholds = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]
            print(f"\nTuning for all {len(thresholds)} thresholds")
        else:
            thresholds = [args.threshold]

        # Tune for each threshold
        all_results = {}
        for threshold in thresholds:
            try:
                results = tune_for_threshold(threshold, config, args)
                all_results[threshold] = results
                print(f"\n✓ Completed tuning for {threshold}% threshold")
            except Exception as e:
                print(f"\n❌ Failed tuning for {threshold}% threshold: {e}")
                continue

        # Summary
        print("\n" + "="*80)
        print("TUNING SUMMARY")
        print("="*80)
        for threshold, results in all_results.items():
            print(f"\n{threshold}% threshold:")
            print(f"  Best F1-score: {results['final_metrics']['f1_score']:.4f}")
            print(f"  Best trial: {results['best_params']}")

        print("\n✓ All tuning complete!")

    else:
        print(f"❌ Error: Unknown mode '{args.mode}'")
        sys.exit(1)


if __name__ == '__main__':
    main()
