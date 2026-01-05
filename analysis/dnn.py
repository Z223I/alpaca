#!/usr/bin/env python3
"""
Deep Neural Network Training with W&B Sweeps (PyTorch Implementation)

This script provides a comprehensive hyperparameter tuning system for DNNs
using Bayesian optimization (W&B Sweeps) with experiment tracking and GPU acceleration.

Filters are ENABLED BY DEFAULT:
    - Time: 09:45 to 16:00 (Eastern Time)
    - Price: $2.00 to $10.00
    - Volume: Average > 80,000 in 10 minutes before alert (uses SIP feed)

Usage:
    # Initialize W&B sweep
    python analysis/dnn.py init-sweep --threshold 6.0 --config analysis/dnn_config.yaml

    # Run sweep agent
    python analysis/dnn.py run-sweep --sweep-id <sweep_id>

    # Train single model (no sweep)
    python analysis/dnn.py train --threshold 6.0 --epochs 100

    # Dry run
    python analysis/dnn.py train --threshold 6.0 --epochs 10 --dry-run
"""

import argparse
import json
import pickle
import subprocess
import sys
import os
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yaml
import pytz

# Alpaca API imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import alpaca_trade_api as tradeapi
from atoms.api.init_alpaca_client import init_alpaca_client

# ML imports - PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# W&B import
import wandb

# Import from existing prediction script
sys.path.append(str(Path(__file__).parent))
from predict_squeeze_outcomes import SqueezeOutcomePredictor
from atoms_analysis.plotting import generate_aligned_cumulative_profit_plot


# =============================================================================
# Constants
# =============================================================================

# Training data start date
TRAINING_START_DATE = "2025-12-15"


# =============================================================================
# Data Filtering Functions (Reused from tune_models.py)
# =============================================================================

def check_volume_criteria(api, symbol: str, alert_time: datetime, min_avg_volume: float = 80000) -> Tuple[bool, Optional[float]]:
    """
    Check if the average volume in the 10 minutes before the alert meets the minimum.

    Args:
        api: Alpaca API client
        symbol: Stock symbol
        alert_time: Time of the alert
        min_avg_volume: Minimum average volume required (default: 80000)

    Returns:
        Tuple of (meets_criteria: bool, avg_volume: float or None)
    """
    try:
        # Get data for 10 minutes before the alert
        end_time = alert_time
        start_time = alert_time - pd.Timedelta(minutes=10)

        # Format as RFC3339 with proper timezone format
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%S') + start_time.strftime('%z')[:3] + ':' + start_time.strftime('%z')[3:]
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%S') + end_time.strftime('%z')[:3] + ':' + end_time.strftime('%z')[3:]

        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Minute,
            start=start_str,
            end=end_str,
            limit=10,
            feed='sip'
        )

        if not bars or len(bars) == 0:
            return (False, None)

        # Calculate average volume
        volumes = [int(bar.v) for bar in bars]
        avg_volume = sum(volumes) / len(volumes)

        return (avg_volume >= min_avg_volume, avg_volume)

    except Exception as e:
        # If we can't get the data, exclude this alert
        return (False, None)


def filter_dataframe(
    df: pd.DataFrame,
    start_hour: int = 9,
    start_minute: int = 45,
    end_hour: int = 16,
    end_minute: int = 0,
    min_price: float = 2.0,
    max_price: float = 10.0,
    min_avg_volume: Optional[float] = None,
    api = None
) -> pd.DataFrame:
    """
    Filter DataFrame based on time, price, and optionally volume criteria.

    Args:
        df: DataFrame with squeeze alert data
        start_hour: Start hour filter (default: 9)
        start_minute: Start minute filter (default: 45)
        end_hour: End hour filter (default: 16)
        end_minute: End minute filter (default: 0)
        min_price: Minimum stock price (default: 2.0)
        max_price: Maximum stock price (default: 10.0)
        min_avg_volume: Minimum average volume in 10 min before alert (default: None, skip volume filter)
        api: Alpaca API client (required if min_avg_volume is set)

    Returns:
        Filtered DataFrame
    """
    print(f"\n{'='*80}")
    print(f"APPLYING DATA FILTERS")
    print(f"{'='*80}")
    print(f"  Time range: {start_hour:02d}:{start_minute:02d} to {end_hour:02d}:{end_minute:02d}")
    print(f"  Price range: ${min_price:.2f} to ${max_price:.2f}")
    if min_avg_volume:
        print(f"  Min avg volume: {min_avg_volume:,.0f} (10 min before alert)")

    initial_count = len(df)
    print(f"  Initial samples: {initial_count}")

    # Filter by time range
    et_tz = pytz.timezone('America/New_York')
    start_time = dt_time(start_hour, start_minute)
    end_time = dt_time(end_hour, end_minute)

    time_mask = df['timestamp'].apply(lambda ts: start_time <= pd.to_datetime(ts).time() <= end_time)
    df = df[time_mask].copy()
    print(f"  After time filter: {len(df)} samples ({len(df)/initial_count*100:.1f}%)")

    # Filter by price range
    if 'last_price' in df.columns:
        price_col = 'last_price'
    elif 'current_price' in df.columns:
        price_col = 'current_price'
    else:
        print("  ⚠ Warning: No price column found, skipping price filter")
        price_col = None

    if price_col:
        price_mask = (df[price_col] >= min_price) & (df[price_col] <= max_price)
        df = df[price_mask].copy()
        print(f"  After price filter: {len(df)} samples ({len(df)/initial_count*100:.1f}%)")

    # Filter by volume (optional)
    if min_avg_volume and api:
        print(f"  Applying volume filter (this may take a while)...")
        volume_filtered = []
        checked = 0
        passed = 0

        for idx, row in df.iterrows():
            checked += 1
            if checked % 100 == 0:
                print(f"    Checked {checked}/{len(df)} ({passed} passed so far)...")

            try:
                timestamp = pd.to_datetime(row['timestamp'])
                symbol = row['symbol']

                # Ensure timestamp is timezone-aware
                if timestamp.tzinfo is None:
                    timestamp = et_tz.localize(timestamp)

                meets_criteria, avg_vol = check_volume_criteria(api, symbol, timestamp, min_avg_volume)

                if meets_criteria:
                    passed += 1
                    volume_filtered.append(idx)
            except Exception as e:
                # Skip on error
                continue

        df = df.loc[volume_filtered].copy()
        print(f"  After volume filter: {len(df)} samples ({len(df)/initial_count*100:.1f}%)")

    print(f"{'='*80}")
    print(f"FILTERING COMPLETE: {len(df)}/{initial_count} samples retained ({len(df)/initial_count*100:.1f}%)")
    print(f"{'='*80}\n")

    return df


# =============================================================================
# GPU Detection
# =============================================================================

def detect_gpu() -> Tuple[bool, str, int]:
    """
    Detect if GPU is available for PyTorch training.

    Returns:
        Tuple of (has_gpu: bool, device_name: str, gpu_count: int)
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)

        print(f"✓ GPU detected: {gpu_count} device(s)")
        print(f"  Device name: {device_name}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")

        return True, device_name, gpu_count
    else:
        print("⚠️  No GPU detected. Using CPU")
        print(f"  PyTorch version: {torch.__version__}")
        return False, "CPU", 0


# =============================================================================
# Data Loading (Reused from tune_models.py with modifications)
# =============================================================================

def load_and_prepare_data(
    threshold: float,
    config: Dict,
    end_date: Optional[str] = None,
    trailing_stop_pct: float = 2.0,
    filter_time: bool = True,
    filter_price: bool = True,
    filter_volume: bool = True,
    start_hour: int = 9,
    start_minute: int = 45,
    end_hour: int = 16,
    end_minute: int = 0,
    min_price: float = 2.0,
    max_price: float = 10.0,
    min_avg_volume: float = 80000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, List[str], SqueezeOutcomePredictor, Optional[Dict]]:
    """
    Load and prepare data using existing SqueezeOutcomePredictor pipeline.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, feature_names, predictor, category_stats)
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
        start_date=TRAINING_START_DATE,
        end_date=end_date
    )
    print(f"✓ Extracted outcomes: {len(outcomes_df)} alerts")

    # Merge with features
    merged_df = predictor.merge_with_features(outcomes_df)
    print(f"✓ Merged with features: {len(merged_df)} samples")

    # Apply filters if requested
    if filter_time or filter_price or filter_volume:
        # Initialize API client if volume filtering is requested
        api = None
        if filter_volume:
            print("Initializing Alpaca API for volume filtering...")
            api = init_alpaca_client("alpaca")

        # Apply filters
        merged_df = filter_dataframe(
            merged_df,
            start_hour=start_hour if filter_time else 0,
            start_minute=start_minute if filter_time else 0,
            end_hour=end_hour if filter_time else 23,
            end_minute=end_minute if filter_time else 59,
            min_price=min_price if filter_price else 0,
            max_price=max_price if filter_price else float('inf'),
            min_avg_volume=min_avg_volume if filter_volume else None,
            api=api
        )

        # Reset index to ensure sequential indexing after filtering
        merged_df = merged_df.reset_index(drop=True)

    # Create target variable based on REALISTIC outcomes
    target_col = f"achieved_{threshold}pct"
    category_stats = None

    if target_col not in merged_df.columns:
        print(f"\n⚙️  Calculating realistic outcomes with {trailing_stop_pct}% trailing stop...")

        TRAILING_STOP_PCT = trailing_stop_pct
        realistic_outcomes = []
        outcome_reasons = []
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

                    enhanced_row = row.copy()
                    if 'outcome_tracking' in full_data:
                        enhanced_row['outcome_tracking'] = full_data['outcome_tracking']

                    gain, reason, holding_time = predictor._calculate_trade_outcome(
                        enhanced_row, threshold, TRAILING_STOP_PCT
                    )

                    realistic_outcomes.append(1 if gain >= threshold else 0)
                    outcome_reasons.append(reason)
                else:
                    json_missing += 1
                    realistic_outcomes.append(1 if row.get('max_gain_percent', 0) >= threshold else 0)
                    outcome_reasons.append('fallback_max_gain')
            except Exception as e:
                realistic_outcomes.append(1 if row.get('max_gain_percent', 0) >= threshold else 0)
                outcome_reasons.append('fallback_error')

        merged_df[target_col] = realistic_outcomes

        print(f"✓ Realistic outcomes calculated:")
        print(f"  - JSON files loaded: {json_loaded}")
        print(f"  - JSON files missing: {json_missing}")
        print(f"  - Achieved (realistic): {sum(realistic_outcomes)} ({sum(realistic_outcomes)/len(realistic_outcomes)*100:.1f}%)")

        # Compare to old max_gain labels
        old_labels = (merged_df['max_gain_percent'] >= threshold).astype(int)
        print(f"  - Achieved (max_gain): {sum(old_labels)} ({sum(old_labels)/len(old_labels)*100:.1f}%)")
        print(f"  - Label difference: {abs(sum(old_labels) - sum(realistic_outcomes))} samples ({abs(sum(old_labels) - sum(realistic_outcomes))/len(old_labels)*100:.1f}%)")

        # Categorize outcome reasons
        print(f"\n📊 Trade Outcome Categories (OCO Bracket Order Logic):")
        total_trades = len(outcome_reasons)

        target_hit_first = sum(1 for r in outcome_reasons if 'target_hit_first' in r or 'target_hit_at' in r or 'snapshot_target' in r or 'final_target' in r or 'legacy_target' in r)
        stop_hit_first = sum(1 for r in outcome_reasons if 'stop_hit_first' in r or ('stop' in r and 'target' not in r and 'fallback' not in r))
        both_hit = sum(1 for r in outcome_reasons if 'target_hit_first' in r or 'stop_hit_first' in r)
        neither_hit = sum(1 for r in outcome_reasons if 'final_no_target' in r or 'legacy_no_target' in r)
        fallback_cases = sum(1 for r in outcome_reasons if 'fallback' in r)

        print(f"  1️⃣  Target Hit First (Label=1):      {target_hit_first:4d} ({target_hit_first/total_trades*100:5.1f}%)")
        print(f"  2️⃣  Stop Hit First (Label=0):        {stop_hit_first:4d} ({stop_hit_first/total_trades*100:5.1f}%)")
        print(f"  3️⃣  Both Hit (OCO timestamp logic):  {both_hit:4d} ({both_hit/total_trades*100:5.1f}%)")
        print(f"  4️⃣  Neither Hit (final gain check):  {neither_hit:4d} ({neither_hit/total_trades*100:5.1f}%)")
        print(f"  📋 Fallback (no interval data):     {fallback_cases:4d} ({fallback_cases/total_trades*100:5.1f}%)")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  📈 Total:                            {total_trades:4d} (100.0%)")

        category_stats = {
            'total_trades': total_trades,
            'target_hit_first': {'count': target_hit_first, 'percentage': target_hit_first/total_trades*100},
            'stop_hit_first': {'count': stop_hit_first, 'percentage': stop_hit_first/total_trades*100},
            'both_hit_oco_decision': {'count': both_hit, 'percentage': both_hit/total_trades*100},
            'neither_hit': {'count': neither_hit, 'percentage': neither_hit/total_trades*100},
            'fallback_no_interval_data': {'count': fallback_cases, 'percentage': fallback_cases/total_trades*100},
        }

    # Prepare features
    X, y = predictor.prepare_features(merged_df, target_variable=target_col)
    feature_names = predictor.feature_names
    print(f"✓ Prepared {len(feature_names)} features")

    # Time-based split
    X_train, X_test, y_train, y_test = predictor.time_based_split(
        merged_df, X, y, test_size=config['data']['test_size']
    )

    # Balanced sampling
    print(f"\n⚙️  Applying balanced sampling to training set...")
    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()
    print(f"  - Before balancing: {n_positive} positive, {n_negative} negative ({n_positive/(n_positive+n_negative)*100:.1f}% positive)")

    n_samples_per_class = min(n_positive, n_negative)

    positive_indices = y_train[y_train == 1].index
    negative_indices = y_train[y_train == 0].index

    np.random.seed(42)
    sampled_positive_indices = np.random.choice(positive_indices, size=n_samples_per_class, replace=False)
    sampled_negative_indices = np.random.choice(negative_indices, size=n_samples_per_class, replace=False)

    balanced_indices = np.concatenate([sampled_positive_indices, sampled_negative_indices])
    np.random.shuffle(balanced_indices)

    X_train = X_train.loc[balanced_indices]
    y_train = y_train.loc[balanced_indices]

    print(f"  - After balancing: {n_samples_per_class} positive, {n_samples_per_class} negative (50.0% positive)")
    print(f"  - Training set size: {len(y_train)} samples (reduced from {n_positive + n_negative})")

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

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, predictor, category_stats


# =============================================================================
# PyTorch DNN Model
# =============================================================================

class DNNClassifier(nn.Module):
    """
    Feedforward Deep Neural Network for binary classification.
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int = 3,
        units_layer1: int = 128,
        units_layer2: int = 64,
        units_layer3: int = 32,
        units_layer4: int = 16,
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        super(DNNClassifier, self).__init__()

        self.n_layers = n_layers
        self.use_batch_norm = use_batch_norm

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()

        # Layer 1
        self.layers.append(nn.Linear(input_dim, units_layer1))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(units_layer1))
        self.dropouts.append(nn.Dropout(dropout_rate))

        # Layer 2
        if n_layers >= 2:
            self.layers.append(nn.Linear(units_layer1, units_layer2))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(units_layer2))
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Layer 3
        if n_layers >= 3:
            self.layers.append(nn.Linear(units_layer2, units_layer3))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(units_layer3))
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Layer 4
        if n_layers >= 4:
            self.layers.append(nn.Linear(units_layer3, units_layer4))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(units_layer4))
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Output layer
        if n_layers == 1:
            self.output = nn.Linear(units_layer1, 1)
        elif n_layers == 2:
            self.output = nn.Linear(units_layer2, 1)
        elif n_layers == 3:
            self.output = nn.Linear(units_layer3, 1)
        else:  # n_layers == 4
            self.output = nn.Linear(units_layer4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through the network."""
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)

        x = self.output(x)
        x = self.sigmoid(x)
        return x


def build_dnn_model(
    input_dim: int,
    config: Dict,
    hp_override: Optional[Dict] = None,
    device: str = 'cpu'
) -> DNNClassifier:
    """
    Build a PyTorch DNN for binary classification.

    Args:
        input_dim: Number of input features
        config: Configuration dictionary with hyperparameters
        hp_override: Optional hyperparameter overrides from W&B sweep
        device: Device to use ('cuda' or 'cpu')

    Returns:
        DNNClassifier model
    """
    # Get hyperparameters (use overrides if provided, else config)
    if hp_override:
        hp = hp_override
    else:
        hp = config['dnn_hyperparameters']

    # Extract hyperparameters
    n_layers = hp.get('n_layers', 3)
    units_layer1 = hp.get('units_layer1', 128)
    units_layer2 = hp.get('units_layer2', 64)
    units_layer3 = hp.get('units_layer3', 32)
    units_layer4 = hp.get('units_layer4', 16)
    dropout_rate = hp.get('dropout_rate', 0.3)
    activation = hp.get('activation', 'relu')
    use_batch_norm = hp.get('use_batch_norm', True)

    # Build model
    model = DNNClassifier(
        input_dim=input_dim,
        n_layers=n_layers,
        units_layer1=units_layer1,
        units_layer2=units_layer2,
        units_layer3=units_layer3,
        units_layer4=units_layer4,
        dropout_rate=dropout_rate,
        activation=activation,
        use_batch_norm=use_batch_norm
    )

    # Move to device
    model = model.to(device)

    return model


# =============================================================================
# Training Function
# =============================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict,
    threshold: float,
    hp_override: Optional[Dict] = None,
    use_wandb: bool = True
) -> Tuple[DNNClassifier, Dict[str, float], Dict]:
    """
    Train DNN model with given hyperparameters.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: Configuration dictionary
        threshold: Gain threshold
        hp_override: Optional hyperparameter overrides from W&B sweep
        use_wandb: Whether to use W&B logging

    Returns:
        Tuple of (model, test_metrics, history)
    """
    print("\n" + "="*80)
    print(f"TRAINING DNN MODEL FOR {threshold}% THRESHOLD")
    print("="*80)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Get hyperparameters
    if hp_override:
        hp = hp_override
        epochs = hp.get('epochs', 100)
        batch_size = hp.get('batch_size', 32)
        learning_rate = hp.get('learning_rate', 0.001)
        l2_reg = hp.get('l2_regularization', 0.001)
        early_stopping_patience = hp.get('early_stopping_patience', 20)
    else:
        hp = config['dnn_hyperparameters']
        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        learning_rate = hp.get('learning_rate', 0.001)
        l2_reg = hp.get('l2_regularization', 0.001)
        early_stopping_patience = config['training']['early_stopping_patience']

    # Build model
    model = build_dnn_model(X_train.shape[1], config, hp_override, device)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train.values),
        torch.FloatTensor(y_train.values).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test.values),
        torch.FloatTensor(y_test.values).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )

    # W&B watch model
    if use_wandb and wandb.run is not None:
        wandb.watch(model, criterion, log='all', log_freq=100)

    # Training loop
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                # Collect predictions
                probs = outputs.cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                labels = batch_y.cpu().numpy()

                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.flatten())

        val_loss /= len(test_loader)
        history['val_loss'].append(val_loss)

        # Calculate metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        val_auc = roc_auc_score(all_labels, all_probs)

        history['val_accuracy'].append(val_accuracy)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
                  f"val_f1: {val_f1:.4f} - val_auc: {val_auc:.4f}")

        # Log to W&B
        if use_wandb and wandb.run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model from epoch with val_loss: {best_val_loss:.4f}")

    # Final evaluation
    print(f"\nEvaluating on test set...")
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
        y_proba = model(X_test_tensor).cpu().numpy().flatten()
        y_pred = (y_proba >= 0.5).astype(int)

    test_metrics = calculate_metrics(y_test.values, y_pred, y_proba)

    print(f"\nTest set performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return model, test_metrics, history


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
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

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

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix - {threshold}% Gain Threshold\nDNN (PyTorch) Best Model Performance',
              fontsize=14, fontweight='bold')

    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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

    output_file = output_dir / f"{threshold}pct_confusion_matrix_dnn.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def calculate_realistic_profits_and_plot(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    threshold: float,
    predictor: SqueezeOutcomePredictor,
    config: Dict,
    trailing_stop_pct: float = 2.0
) -> Path:
    """
    Calculate realistic trading profits and generate cumulative profit chart.

    Args:
        X_test: Test feature matrix
        y_test: Test labels
        y_pred: Model predictions
        threshold: Gain threshold percentage
        predictor: SqueezeOutcomePredictor instance
        config: Configuration dictionary
        trailing_stop_pct: Trailing stop loss percentage

    Returns:
        Path to saved cumulative profit plot
    """
    print(f"\nCalculating realistic trading profits with {trailing_stop_pct}% trailing stop...")

    TRAILING_STOP_PCT = trailing_stop_pct

    json_dir = config['data']['json_dir']
    features_csv = config['data']['features_csv']

    # Re-extract outcomes to get the full dataframe
    outcomes_df = predictor.extract_outcomes(
        gain_threshold=threshold,
        start_date=TRAINING_START_DATE,
        end_date=config.get('end_date')
    )

    merged_df = predictor.merge_with_features(outcomes_df)

    # Get the test set indices
    test_df = merged_df.loc[X_test.index].copy()

    # Add predictions
    test_df['predicted_outcome'] = y_pred
    test_df['actual_outcome'] = y_test.values

    # Calculate realistic profit for each test sample
    realistic_profits = []

    for idx in test_df.index:
        outcome_row = test_df.loc[idx]

        try:
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
            gain, _, _ = predictor._calculate_trade_outcome(
                outcome_row, threshold, TRAILING_STOP_PCT
            )
            realistic_profits.append(gain)

    test_df['realistic_profit'] = realistic_profits

    # Separate into model trades and all trades
    model_trades = test_df[test_df['predicted_outcome'] == 1].copy()

    print(f"✓ Calculated realistic profits for {len(test_df)} test samples")
    print(f"  Model selected {len(model_trades)} trades ({len(model_trades)/len(test_df)*100:.1f}%)")
    print(f"  Average profit (take-all): {test_df['realistic_profit'].mean():.2f}%")
    if len(model_trades) > 0:
        print(f"  Average profit (model): {model_trades['realistic_profit'].mean():.2f}%")

    # Generate cumulative profit plot
    print(f"\nGenerating cumulative profit chart...")
    threshold_suffix = f"_{threshold}pct_dnn"
    generate_aligned_cumulative_profit_plot(
        predictions_df=test_df,
        model_trades=model_trades,
        threshold_suffix=threshold_suffix,
        gain_threshold=threshold
    )

    plot_path = Path("analysis/plots") / f"aligned_cumulative_profit{threshold_suffix}.png"
    return plot_path


# =============================================================================
# Model Saving
# =============================================================================

def save_best_model(
    model: DNNClassifier,
    test_metrics: Dict,
    threshold: float,
    scaler: StandardScaler,
    feature_names: List[str],
    config: Dict,
    trailing_stop_pct: float = 2.0,
    category_stats: Optional[Dict] = None,
    cm_plot_path: Optional[Path] = None
):
    """
    Save best model and metadata.

    Args:
        model: Trained PyTorch model
        test_metrics: Test set metrics
        threshold: Gain threshold
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        config: Configuration dictionary
        trailing_stop_pct: Trailing stop loss percentage
        category_stats: Trade outcome category statistics
        cm_plot_path: Path to confusion matrix plot
    """
    print(f"\nSaving best model for {threshold}% threshold with {trailing_stop_pct}% trailing stop...")

    # Create output directory
    output_dir = Path("analysis/tuned_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # File naming
    model_name = f"dnn_tuned_{threshold}pct_stop{trailing_stop_pct}"

    # Save model (PyTorch format)
    model_path = output_dir / f"{model_name}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_dim': model.layers[0].in_features,
            'n_layers': model.n_layers,
            'units_layer1': model.layers[0].out_features,
            'units_layer2': model.layers[1].out_features if model.n_layers >= 2 else None,
            'units_layer3': model.layers[2].out_features if model.n_layers >= 3 else None,
            'units_layer4': model.layers[3].out_features if model.n_layers >= 4 else None,
            'dropout_rate': model.dropouts[0].p,
            'use_batch_norm': model.use_batch_norm
        }
    }, model_path)
    print(f"✓ Saved model: {model_path}")

    # Save scaler
    scaler_path = output_dir / f"{model_name}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: {scaler_path}")

    # Save metadata
    info = {
        'threshold': threshold,
        'gain_threshold': threshold,
        'trailing_stop_pct': trailing_stop_pct,
        'model_type': 'dnn_pytorch',
        'test_metrics': test_metrics,
        'feature_names': feature_names,
        'training_info': {
            'timestamp': datetime.now().isoformat(),
            'gpu_used': torch.cuda.is_available(),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        },
        'outcome_categories': category_stats if category_stats else None
    }

    info_path = output_dir / f"{model_name}_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✓ Saved metadata: {info_path}")

    # Log as W&B artifact
    if wandb.run is not None:
        try:
            artifact = wandb.Artifact(
                name=f"dnn-tuned-{threshold}pct-stop{trailing_stop_pct}",
                type="model",
                description=f"Best DNN (PyTorch) model for {threshold}% gain threshold with {trailing_stop_pct}% trailing stop",
                metadata={
                    'threshold': threshold,
                    'trailing_stop_pct': trailing_stop_pct,
                    'best_f1': test_metrics['f1_score'],
                }
            )
            artifact.add_file(str(model_path))
            artifact.add_file(str(scaler_path))
            artifact.add_file(str(info_path))

            if cm_plot_path and cm_plot_path.exists():
                artifact.add_file(str(cm_plot_path))
                wandb.log({f"confusion_matrix_{threshold}pct": wandb.Image(str(cm_plot_path))})

            wandb.log_artifact(artifact)
            print(f"✓ Logged artifact to W&B")
        except Exception as e:
            print(f"⚠️  Warning: Could not log artifact to W&B: {e}")


# =============================================================================
# W&B Sweep Functions
# =============================================================================

def create_sweep_config(config: Dict, threshold: float) -> Dict:
    """
    Create W&B sweep configuration from config file.

    Args:
        config: Configuration dictionary
        threshold: Gain threshold

    Returns:
        W&B sweep configuration
    """
    sweep_config = {
        'name': f'dnn-pytorch-sweep-{threshold}pct',
        'method': config['wandb_sweep']['method'],
        'metric': {
            'name': config['wandb_sweep']['metric'],
            'goal': config['wandb_sweep']['goal']
        },
        'parameters': {}
    }

    # Add hyperparameters from search space
    search_space = config['dnn_search_space']

    for param_name, param_config in search_space.items():
        param_type = param_config['type']

        if param_type == 'int':
            sweep_config['parameters'][param_name] = {
                'distribution': 'int_uniform',
                'min': param_config['min'],
                'max': param_config['max']
            }
        elif param_type == 'float':
            sweep_config['parameters'][param_name] = {
                'distribution': 'uniform',
                'min': param_config['min'],
                'max': param_config['max']
            }
        elif param_type == 'log_uniform':
            sweep_config['parameters'][param_name] = {
                'distribution': 'log_uniform_values',
                'min': param_config['min'],
                'max': param_config['max']
            }
        elif param_type == 'categorical':
            sweep_config['parameters'][param_name] = {
                'values': param_config['values']
            }

    # Add early stopping
    if config['wandb_sweep'].get('early_terminate'):
        sweep_config['early_terminate'] = {
            'type': 'hyperband',
            'min_iter': config['wandb_sweep']['early_terminate_config']['min_iter']
        }

    return sweep_config


def sweep_train():
    """
    Train function for W&B sweep agent.
    This will be called by wandb.agent() for each sweep run.
    """
    # Initialize W&B run
    run = wandb.init()

    # Get hyperparameters from sweep
    hp = wandb.config

    # Get global config from environment
    config_path = os.environ.get('DNN_CONFIG_PATH', 'analysis/dnn_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    threshold = float(os.environ.get('DNN_THRESHOLD', '6.0'))
    trailing_stop = float(os.environ.get('DNN_TRAILING_STOP', '2.0'))

    # Parse filter flags
    filter_time = os.environ.get('DNN_FILTER_TIME', 'True') == 'True'
    filter_price = os.environ.get('DNN_FILTER_PRICE', 'True') == 'True'
    filter_volume = os.environ.get('DNN_FILTER_VOLUME', 'True') == 'True'

    try:
        # Load data
        X_train, X_test, y_train, y_test, scaler, feature_names, predictor, category_stats = load_and_prepare_data(
            threshold, config,
            end_date=config['data'].get('train_end_date'),
            trailing_stop_pct=trailing_stop,
            filter_time=filter_time,
            filter_price=filter_price,
            filter_volume=filter_volume
        )

        # Train model with sweep hyperparameters
        hp_dict = dict(hp)
        model, test_metrics, history = train_model(
            X_train, y_train, X_test, y_test,
            config, threshold, hp_override=hp_dict, use_wandb=True
        )

        # Log metrics to W&B
        wandb.log(test_metrics)

        # Log best epoch info
        best_epoch = np.argmin(history['val_loss'])
        wandb.log({
            'best_epoch': best_epoch,
            'best_val_loss': history['val_loss'][best_epoch]
        })

    except Exception as e:
        print(f"❌ Error during sweep training: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish(exit_code=1)
        raise

    wandb.finish()


# =============================================================================
# CLI and Main Entry Point
# =============================================================================

def parse_time_string(time_str: str) -> Tuple[int, int]:
    """Parse time string in HHMM format to hour and minute."""
    if len(time_str) != 4 or not time_str.isdigit():
        raise ValueError(f"Invalid time format: {time_str}. Expected HHMM format (e.g., 0945)")

    hour = int(time_str[:2])
    minute = int(time_str[2:])

    if hour < 0 or hour > 23:
        raise ValueError(f"Invalid hour: {hour}. Must be 0-23")
    if minute < 0 or minute > 59:
        raise ValueError(f"Invalid minute: {minute}. Must be 0-59")

    return hour, minute


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Neural Network Training with W&B Sweeps (PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from project root):
  # Initialize sweep
  python analysis/dnn.py init-sweep --threshold 6.0 --config analysis/dnn_config.yaml

  # Run sweep agent
  python analysis/dnn.py run-sweep --sweep-id <sweep_id>

  # Train single model
  python analysis/dnn.py train --threshold 6.0 --epochs 100

  # Dry run
  python analysis/dnn.py train --threshold 6.0 --epochs 10 --dry-run
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Init-sweep subcommand
    init_parser = subparsers.add_parser('init-sweep', help='Initialize W&B sweep')
    init_parser.add_argument(
        '--threshold', type=float, default=6.0,
        help='Gain threshold percentage (e.g., 6.0 for 6%%)'
    )
    init_parser.add_argument(
        '--config', type=str, default='analysis/dnn_config.yaml',
        help='Path to configuration file'
    )
    init_parser.add_argument(
        '--trailing-stop', type=float, default=2.0,
        help='Trailing stop loss percentage (default: 2.0)'
    )
    init_parser.add_argument(
        '--no-filter-time', action='store_true',
        help='Disable time range filter'
    )
    init_parser.add_argument(
        '--no-filter-price', action='store_true',
        help='Disable price range filter'
    )
    init_parser.add_argument(
        '--no-filter-volume', action='store_true',
        help='Disable volume filter'
    )

    # Run-sweep subcommand
    run_parser = subparsers.add_parser('run-sweep', help='Run W&B sweep agent')
    run_parser.add_argument(
        '--sweep-id', type=str, required=True,
        help='W&B sweep ID'
    )
    run_parser.add_argument(
        '--count', type=int, default=None,
        help='Number of runs (default: unlimited)'
    )

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train single model (no sweep)')
    train_parser.add_argument(
        '--threshold', type=float, default=6.0,
        help='Gain threshold percentage'
    )
    train_parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size'
    )
    train_parser.add_argument(
        '--config', type=str, default='analysis/dnn_config.yaml',
        help='Path to configuration file'
    )
    train_parser.add_argument(
        '--trailing-stop', type=float, default=2.0,
        help='Trailing stop loss percentage'
    )
    train_parser.add_argument(
        '--no-filter-time', action='store_true',
        help='Disable time range filter'
    )
    train_parser.add_argument(
        '--no-filter-price', action='store_true',
        help='Disable price range filter'
    )
    train_parser.add_argument(
        '--no-filter-volume', action='store_true',
        help='Disable volume filter'
    )
    train_parser.add_argument(
        '--dry-run', action='store_true',
        help='Dry run mode (limited epochs for testing)'
    )
    train_parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable W&B logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Print CLI arguments
    print("\n" + "="*80)
    print("CLI ARGUMENTS")
    print("="*80)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*80 + "\n")

    # Check if mode is specified
    if not args.mode:
        print("❌ Error: No mode specified. Use 'init-sweep', 'run-sweep', or 'train' subcommand.")
        print("\nUsage examples:")
        print("  python analysis/dnn.py init-sweep --threshold 6.0")
        print("  python analysis/dnn.py run-sweep --sweep-id <id>")
        print("  python analysis/dnn.py train --threshold 6.0 --epochs 100")
        print("\nRun with --help for more information.")
        sys.exit(1)

    if args.mode == 'init-sweep':
        # Initialize W&B sweep
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Detect GPU
        has_gpu, device_name, gpu_count = detect_gpu()

        # Create sweep configuration
        sweep_config = create_sweep_config(config, args.threshold)

        # Initialize sweep
        print(f"\nInitializing W&B sweep for {args.threshold}% threshold...")
        sweep_id = wandb.sweep(
            sweep_config,
            project=config['wandb']['project'],
            entity=config['wandb'].get('entity')
        )

        print(f"\n✓ Sweep initialized!")
        print(f"  Sweep ID: {sweep_id}")
        print(f"\nTo run the sweep agent:")
        print(f"  export DNN_CONFIG_PATH='{args.config}'")
        print(f"  export DNN_THRESHOLD='{args.threshold}'")
        print(f"  export DNN_TRAILING_STOP='{args.trailing_stop}'")
        print(f"  export DNN_FILTER_TIME='{not args.no_filter_time}'")
        print(f"  export DNN_FILTER_PRICE='{not args.no_filter_price}'")
        print(f"  export DNN_FILTER_VOLUME='{not args.no_filter_volume}'")
        print(f"  python analysis/dnn.py run-sweep --sweep-id {sweep_id}")

    elif args.mode == 'run-sweep':
        # Run sweep agent
        print(f"\nStarting W&B sweep agent...")
        print(f"  Sweep ID: {args.sweep_id}")

        # Check environment variables
        if 'DNN_CONFIG_PATH' not in os.environ:
            print("❌ Error: DNN_CONFIG_PATH environment variable not set")
            print("Please set it before running the sweep agent (see init-sweep output)")
            sys.exit(1)

        config_path = os.environ['DNN_CONFIG_PATH']
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Detect GPU
        has_gpu, device_name, gpu_count = detect_gpu()

        # Run agent
        wandb.agent(
            args.sweep_id,
            function=sweep_train,
            count=args.count,
            project=config['wandb']['project'],
            entity=config['wandb'].get('entity')
        )

        print("\n✓ Sweep agent completed!")

    elif args.mode == 'train':
        # Single model training (no sweep)
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Dry run mode
        if args.dry_run:
            print("\n🧪 DRY RUN MODE - Using 10 epochs for testing")
            args.epochs = 10
            config['wandb']['mode'] = 'disabled'

        # Detect GPU
        has_gpu, device_name, gpu_count = detect_gpu()

        # Initialize W&B
        use_wandb = not args.no_wandb and config['wandb']['mode'] != 'disabled'
        if use_wandb:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            run_name = f"dnn-pytorch-{args.threshold}pct-stop{args.trailing_stop}-{timestamp}"

            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb'].get('entity'),
                name=run_name,
                config={
                    'threshold': args.threshold,
                    'trailing_stop_pct': args.trailing_stop,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'framework': 'pytorch',
                    **config['dnn_hyperparameters']
                },
                tags=config['wandb']['tags'] + [f"{args.threshold}pct", f"stop{args.trailing_stop}", "pytorch"],
                mode=config['wandb']['mode']
            )

        try:
            # Load data
            X_train, X_test, y_train, y_test, scaler, feature_names, predictor, category_stats = load_and_prepare_data(
                args.threshold, config,
                end_date=config['data'].get('train_end_date'),
                trailing_stop_pct=args.trailing_stop,
                filter_time=not args.no_filter_time,
                filter_price=not args.no_filter_price,
                filter_volume=not args.no_filter_volume
            )

            # Override epochs and batch size
            config['training']['epochs'] = args.epochs
            config['training']['batch_size'] = args.batch_size

            # Train model
            model, test_metrics, history = train_model(
                X_train, y_train, X_test, y_test,
                config, args.threshold, use_wandb=use_wandb
            )

            # Generate confusion matrix
            print(f"\nGenerating confusion matrix...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test.values).to(device)
                y_proba = model(X_test_tensor).cpu().numpy().flatten()
                y_pred = (y_proba >= 0.5).astype(int)

            cm_plot_path = plot_confusion_matrix(y_test.values, y_pred, args.threshold)
            print(f"✓ Saved confusion matrix: {cm_plot_path}")

            # Generate cumulative profit chart
            cumulative_profit_path = calculate_realistic_profits_and_plot(
                X_test, y_test, y_pred, args.threshold, predictor, config,
                trailing_stop_pct=args.trailing_stop
            )
            print(f"✓ Saved cumulative profit chart: {cumulative_profit_path}")

            # Save model
            save_best_model(
                model, test_metrics, args.threshold, scaler, feature_names, config,
                trailing_stop_pct=args.trailing_stop, category_stats=category_stats,
                cm_plot_path=cm_plot_path
            )

            # Finish W&B
            if use_wandb:
                wandb.finish()

            print("\n✓ Training complete!")

        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            if use_wandb:
                wandb.finish(exit_code=1)
            sys.exit(1)

    else:
        print(f"❌ Error: Unknown mode '{args.mode}'")
        sys.exit(1)


if __name__ == '__main__':
    main()
