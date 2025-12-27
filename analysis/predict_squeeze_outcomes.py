#!/usr/bin/env python3
"""
Squeeze Alert Outcome Prediction

Predicts price action following squeeze alerts using statistically independent features.

This script supports two modes:
1. **TRAIN MODE** - Train ML models and evaluate performance
2. **PREDICT MODE** - Use trained models to make predictions on new data

TRAINING WORKFLOW:
1. Extracts outcome_tracking data from JSON files
2. Merges with independent features
3. Trains multiple ML models (Logistic Regression, Random Forest, XGBoost)
4. Evaluates trading-specific metrics (precision, recall, ROI simulation)
5. Generates feature importance analysis
6. Saves trained models for later use

PREDICTION WORKFLOW:
1. Loads a trained XGBoost model
2. Extracts outcome data from specified date range
3. Applies same preprocessing as training
4. Makes predictions and evaluates performance
5. Saves predictions with probabilities to CSV

USAGE EXAMPLES (run from project root):

  # Train models for all thresholds (1.5%, 2%, 2.5%, 3%, 4%, 5%, 6%, 7%)
  python analysis/predict_squeeze_outcomes.py train

  # Train model for specific threshold
  python analysis/predict_squeeze_outcomes.py train --threshold 1.5

  # Make predictions on a specific date's data
  python analysis/predict_squeeze_outcomes.py predict \
    --model analysis/xgboost_model_1.5pct.json \
    --start-date 2025-12-17 \
    --end-date 2025-12-17

  # Make predictions on a date range
  python analysis/predict_squeeze_outcomes.py predict \
    --model analysis/xgboost_model_2pct.json \
    --start-date 2025-12-17 \
    --end-date 2025-12-18

  # Make predictions with custom threshold (overrides model's default)
  python analysis/predict_squeeze_outcomes.py predict \
    --model analysis/xgboost_model_1.5pct.json \
    --start-date 2025-12-17 \
    --threshold 2.0

OUTPUT FILES (Training):
  - analysis/plots/feature_importance_{threshold}pct.png
  - analysis/plots/roc_curves_{threshold}pct.png
  - analysis/prediction_summary_{threshold}pct.txt
  - analysis/plots/class_distribution_{threshold}pct.png
  - analysis/plots/price_category_analysis_{threshold}pct.png
  - analysis/plots/time_of_day_analysis_{threshold}pct.png
  - analysis/xgboost_model_{threshold}pct.json
  - analysis/xgboost_model_{threshold}pct_info.json

OUTPUT FILES (Prediction):
  - analysis/predictions_{threshold}pct.csv
    (Columns: symbol, timestamp, actual_outcome, predicted_outcome,
     prediction_probability, squeeze_entry_price, price_at_10min,
     max_gain_percent, final_gain_percent, realistic_profit)

NOTE: Some features have missing data in current dataset:
- ema_spread_pct: ~10% missing (price-normalized EMA spread)
- macd_histogram: ~15% missing (will be 100% available when switched to daily MACD)

For now, we handle missing data via imputation and indicator variables.

Author: Predictive Analytics Module
Date: 2025-12-12
Updated: 2025-12-16 (Added predict mode)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Import plotting functions from atoms_analysis
from atoms_analysis.plotting import (
    plot_feature_importance,
    plot_roc_curves,
    plot_class_distribution,
    plot_price_category_analysis,
    plot_time_of_day_analysis,
    generate_prediction_plots,
    generate_aligned_cumulative_profit_plot,
    generate_time_binned_outcomes_chart,
    generate_price_binned_outcomes_chart
)


class SqueezeOutcomePredictor:
    """Predict squeeze alert outcomes using machine learning."""

    def __init__(self, json_dir: str, features_csv: str):
        """
        Initialize predictor with data sources.

        Args:
            json_dir: Base historical_data directory containing date subdirectories (e.g., 2025-12-12/)
            features_csv: Path to independent features CSV
        """
        self.json_dir = Path(json_dir)
        self.features_csv = features_csv
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = None

        # Create analysis/plots directory if it doesn't exist
        plots_dir = Path('analysis/plots')
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        np.random.seed(42)

    def extract_outcomes(self, gain_threshold: float = 5.0, start_date: str = "2025-12-12", end_date: str = None) -> pd.DataFrame:
        """
        Extract outcome_tracking data from JSON files across multiple date directories.

        Args:
            gain_threshold: Gain percentage threshold for success classification
            start_date: Starting date for directory scan (YYYY-MM-DD format)
            end_date: Ending date for directory scan (YYYY-MM-DD format). If None, includes all dates >= start_date.

        Returns:
            DataFrame with outcome metrics for each alert
        """
        print("="*80)
        print("STEP 1: EXTRACTING OUTCOME DATA FROM JSON FILES")
        print("="*80)
        print(f"Gain threshold: {gain_threshold}%")
        if end_date:
            print(f"Scanning directories from {start_date} to {end_date} (inclusive)")
        else:
            print(f"Scanning directories from {start_date} onwards")

        outcomes = []

        # Find all date directories within range
        date_dirs = []
        for date_dir in self.json_dir.glob('????-??-??'):
            date_name = date_dir.name
            # Check if within range
            if date_dir.is_dir() and date_name >= start_date:
                if end_date is None or date_name <= end_date:
                    squeeze_dir = date_dir / 'squeeze_alerts_sent'
                    if squeeze_dir.exists():
                        date_dirs.append(squeeze_dir)

        date_dirs.sort()  # Sort chronologically
        print(f"Found {len(date_dirs)} date directories: {[d.parent.name for d in date_dirs]}")

        # Collect all JSON files from all directories
        json_files = []
        for squeeze_dir in date_dirs:
            json_files.extend(list(squeeze_dir.glob('alert_*.json')))

        print(f"Found {len(json_files)} total JSON files across all directories")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Check if outcome tracking exists
                if 'outcome_tracking' not in data:
                    continue

                outcome_tracking = data['outcome_tracking']
                if 'summary' not in outcome_tracking:
                    continue

                summary = outcome_tracking['summary']
                intervals = outcome_tracking.get('intervals', {})

                # Extract key identifying information
                outcome_row = {
                    'symbol': data['symbol'],
                    'timestamp': data['timestamp'],
                    'squeeze_entry_price': outcome_tracking.get('squeeze_entry_price'),

                    # Summary metrics
                    'max_gain_percent': summary.get('max_gain_percent'),
                    'final_gain_percent': summary.get('final_gain_percent'),
                    'max_drawdown_percent': summary.get('max_drawdown_percent'),
                    'reached_stop_loss': summary.get('reached_stop_loss'),
                    'achieved_5pct': summary.get('achieved_5pct'),
                    'achieved_10pct': summary.get('achieved_10pct'),
                    'achieved_15pct': summary.get('achieved_15pct'),
                    'time_to_5pct_minutes': summary.get('time_to_5pct_minutes'),
                    'price_at_10min': summary.get('price_at_10min'),

                    # Interval-specific gains (for alternative targets)
                    'gain_at_10s': intervals.get('10', {}).get('gain_percent'),
                    'gain_at_20s': intervals.get('20', {}).get('gain_percent'),
                    'gain_at_30s': intervals.get('30', {}).get('gain_percent'),
                    'gain_at_60s': intervals.get('60', {}).get('gain_percent'),
                    'gain_at_120s': intervals.get('120', {}).get('gain_percent'),
                    'gain_at_300s': intervals.get('300', {}).get('gain_percent'),
                }

                outcomes.append(outcome_row)

            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
                continue

        outcomes_df = pd.DataFrame(outcomes)
        print(f"✓ Extracted {len(outcomes_df)} alerts with outcome data")

        # Create custom achieved threshold based on max_gain_percent
        threshold_col = f'achieved_{int(gain_threshold)}pct'
        outcomes_df[threshold_col] = outcomes_df['max_gain_percent'] >= gain_threshold

        # Show summary statistics
        print(f"\nOutcome Summary:")
        print(f"  Achieved {gain_threshold}%: {outcomes_df[threshold_col].sum()} / {len(outcomes_df)} ({outcomes_df[threshold_col].mean()*100:.1f}%)")
        print(f"  Achieved 5%:  {outcomes_df['achieved_5pct'].sum()} / {len(outcomes_df)} ({outcomes_df['achieved_5pct'].mean()*100:.1f}%)")
        print(f"  Achieved 10%: {outcomes_df['achieved_10pct'].sum()} / {len(outcomes_df)} ({outcomes_df['achieved_10pct'].mean()*100:.1f}%)")
        print(f"  Hit Stop Loss: {outcomes_df['reached_stop_loss'].sum()} / {len(outcomes_df)} ({outcomes_df['reached_stop_loss'].mean()*100:.1f}%)")
        print(f"  Avg Max Gain: {outcomes_df['max_gain_percent'].mean():.2f}%")
        print(f"  Avg Final Gain: {outcomes_df['final_gain_percent'].mean():.2f}%")

        return outcomes_df

    def merge_with_features(self, outcomes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge outcome data with independent features.

        Args:
            outcomes_df: DataFrame with outcome metrics

        Returns:
            Merged DataFrame ready for modeling
        """
        print("\n" + "="*80)
        print("STEP 2: MERGING OUTCOMES WITH INDEPENDENT FEATURES")
        print("="*80)

        # Load independent features
        features_df = pd.read_csv(self.features_csv)
        print(f"Loaded {len(features_df)} alerts with features")

        # Merge on symbol + timestamp
        df = features_df.merge(
            outcomes_df,
            on=['symbol', 'timestamp'],
            how='inner'
        )

        print(f"✓ Merged dataset: {len(df)} observations")
        print(f"  Features: {features_df.shape[1]} columns")
        print(f"  Outcomes: {outcomes_df.shape[1]} columns")
        print(f"  Combined: {df.shape[1]} columns")

        return df

    def prepare_features(self, df: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable.

        Handles:
        - Missing data in ema_spread_pct (~10% missing - price-normalized)
        - Missing data in macd_histogram (~15% missing - will be 100% with daily MACD)
        - Categorical encoding for market_session and price_category
        - Feature scaling

        Args:
            df: Merged DataFrame
            target_variable: Name of target column

        Returns:
            Tuple of (X, y) - features and target
        """
        print("\n" + "="*80)
        print("STEP 3: PREPARING FEATURES AND TARGET")
        print("="*80)

        # Define feature columns (11 independent features)
        # UPDATED: ema_spread_pct (price-normalized), removed distance_from_day_low_percent
        feature_cols = [
            'ema_spread_pct',              # Price-normalized EMA momentum
            'price_category',              # Stock price bin (categorical)
            'macd_histogram',              # MACD momentum indicator
            'market_session',              # Time of day (categorical)
            'squeeze_number_today',        # Squeeze sequence number
            'minutes_since_last_squeeze',  # Time since last squeeze
            'window_volume_vs_1min_avg',   # Volume surge ratio
            'distance_from_vwap_percent',  # Distance from VWAP
            'day_gain',                    # Day gain percentage
            'spy_percent_change_concurrent',  # SPY correlation
            'spread_percent'               # Bid-ask spread
        ]

        # Extract features
        X = df[feature_cols].copy()

        # Handle categorical variables
        if 'market_session' in X.columns:
            le_session = LabelEncoder()
            X['market_session_encoded'] = le_session.fit_transform(X['market_session'].fillna('unknown'))
            X = X.drop('market_session', axis=1)
            print(f"✓ Encoded market_session: {list(le_session.classes_)}")

        if 'price_category' in X.columns:
            le_price = LabelEncoder()
            X['price_category_encoded'] = le_price.fit_transform(X['price_category'].fillna('unknown'))
            X = X.drop('price_category', axis=1)
            print(f"✓ Encoded price_category: {list(le_price.classes_)}")

        # Handle missing data
        print(f"\nMissing Data Analysis:")
        missing_pct = (X.isnull().sum() / len(X) * 100).sort_values(ascending=False)
        for col, pct in missing_pct[missing_pct > 0].items():
            print(f"  {col}: {pct:.1f}% missing")

        # NOTE: In future, ema_spread_pct and macd_histogram will be 100% available
        # For now, we impute missing values

        # Strategy: Impute with median + create missing indicators
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # REMOVED: ema_spread_pct_missing indicator
        # Previously created indicator for missing values, but this caused distribution
        # mismatch between train/test (9% missing in train, 0% in some test sets).
        # Now we just impute missing values without the indicator feature.

        # Exclude macd_histogram if too much missing data (>50%)
        if 'macd_histogram' in X_imputed.columns:
            macd_missing_pct = (X['macd_histogram'].isnull().sum() / len(X)) * 100
            if macd_missing_pct > 50:
                print(f"⚠️  Excluding macd_histogram ({macd_missing_pct:.1f}% missing)")
                X_imputed = X_imputed.drop('macd_histogram', axis=1, errors='ignore')

        # Extract target variable
        y = df[target_variable].copy()

        # Remove rows where target is null
        valid_mask = y.notna()
        X_imputed = X_imputed[valid_mask]
        y = y[valid_mask]

        print(f"\n✓ Feature matrix: {X_imputed.shape}")
        print(f"✓ Target variable '{target_variable}': {y.shape}")
        print(f"  - Class 0 (False): {(y == False).sum()} ({(y == False).mean()*100:.1f}%)")
        print(f"  - Class 1 (True):  {(y == True).sum()} ({(y == True).mean()*100:.1f}%)")

        self.feature_names = X_imputed.columns.tolist()

        return X_imputed, y

    def time_based_split(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                         test_size: float = 0.20) -> Tuple:
        """
        Perform time-based train/test split (NOT random).

        Critical for time-series data to avoid data leakage.

        Args:
            df: Original DataFrame (for timestamp sorting)
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*80)
        print("STEP 4: TIME-BASED TRAIN/TEST SPLIT")
        print("="*80)

        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)

        # Align X and y with sorted df
        X_sorted = X.loc[df_sorted.index]
        y_sorted = y.loc[df_sorted.index]

        # Calculate split index
        split_idx = int(len(df_sorted) * (1 - test_size))

        # Split
        X_train = X_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_train = y_sorted.iloc[:split_idx]
        y_test = y_sorted.iloc[split_idx:]

        # Get timestamp ranges
        train_timestamps = df_sorted.iloc[:split_idx]['timestamp']
        test_timestamps = df_sorted.iloc[split_idx:]['timestamp']

        print(f"✓ Train set: {len(X_train)} samples")
        print(f"  Time range: {train_timestamps.min()} to {train_timestamps.max()}")
        print(f"  Class balance: {y_train.mean()*100:.1f}% positive")

        print(f"\n✓ Test set: {len(X_test)} samples")
        print(f"  Time range: {test_timestamps.min()} to {test_timestamps.max()}")
        print(f"  Class balance: {y_test.mean()*100:.1f}% positive")

        # Verify no time overlap (test should be after train)
        assert train_timestamps.max() < test_timestamps.min(), "Time leakage detected!"
        print(f"\n✓ Verified: Test data is strictly after training data (no time leakage)")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
        """
        Standardize features (mean=0, std=1).

        Important for models like Logistic Regression.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        print("\n" + "="*80)
        print("STEP 5: FEATURE SCALING")
        print("="*80)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Store scaler for later use in predictions
        self.scaler = scaler

        print(f"✓ Scaled features to mean=0, std=1")
        print(f"  Training mean: {X_train_scaled.mean():.4f}")
        print(f"  Training std: {X_train_scaled.std():.4f}")

        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        return X_train_scaled, X_test_scaled

    def balance_classes(self, X_train: pd.DataFrame, y_train: pd.Series,
                       method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance class distribution in training data using resampling techniques.

        This addresses class imbalance by either oversampling the minority class,
        undersampling the majority class, or a combination of both.

        Methods:
        - 'smote': SMOTE (Synthetic Minority Over-sampling Technique) - creates synthetic samples
        - 'oversample': Random oversampling - duplicates minority class samples
        - 'undersample': Random undersampling - reduces majority class samples
        - 'smoteenn': SMOTE + Edited Nearest Neighbors - combines oversampling and cleaning

        Args:
            X_train: Training feature matrix
            y_train: Training target variable
            method: Resampling method ('smote', 'oversample', 'undersample', 'smoteenn')

        Returns:
            Tuple of (X_train_balanced, y_train_balanced)
        """
        print("\n" + "="*80)
        print("STEP 5B: CLASS BALANCING")
        print("="*80)

        # Show original class distribution
        class_counts = y_train.value_counts()
        total = len(y_train)
        print(f"\nOriginal class distribution:")
        print(f"  Class 0 (unsuccessful): {class_counts.get(0, 0):4d} samples ({class_counts.get(0, 0)/total*100:5.1f}%)")
        print(f"  Class 1 (successful):   {class_counts.get(1, 0):4d} samples ({class_counts.get(1, 0)/total*100:5.1f}%)")
        print(f"  Imbalance ratio: {class_counts.get(0, 0) / class_counts.get(1, 1):.2f}:1")

        # Apply resampling
        try:
            if method == 'smote':
                print(f"\nApplying SMOTE (Synthetic Minority Over-sampling)...")
                # SMOTE requires at least k_neighbors+1 samples in minority class
                min_samples = min(class_counts)
                k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
                sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            elif method == 'oversample':
                print(f"\nApplying Random Oversampling...")
                sampler = RandomOverSampler(random_state=42)
            elif method == 'undersample':
                print(f"\nApplying Random Undersampling...")
                sampler = RandomUnderSampler(random_state=42)
            elif method == 'smoteenn':
                print(f"\nApplying SMOTE-ENN (SMOTE + Edited Nearest Neighbors)...")
                min_samples = min(class_counts)
                k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
                sampler = SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=k_neighbors))
            else:
                raise ValueError(f"Unknown resampling method: {method}")

            # Perform resampling
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)

            # Convert back to DataFrame/Series with proper columns
            X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
            y_train_balanced = pd.Series(y_train_balanced, name=y_train.name)

            # Show new class distribution
            class_counts_new = y_train_balanced.value_counts()
            total_new = len(y_train_balanced)
            print(f"\n✓ Balanced class distribution:")
            print(f"  Class 0 (unsuccessful): {class_counts_new.get(0, 0):4d} samples ({class_counts_new.get(0, 0)/total_new*100:5.1f}%)")
            print(f"  Class 1 (successful):   {class_counts_new.get(1, 0):4d} samples ({class_counts_new.get(1, 0)/total_new*100:5.1f}%)")
            print(f"  New ratio: {class_counts_new.get(0, 0) / class_counts_new.get(1, 1):.2f}:1")
            print(f"  Total samples: {total} → {total_new} ({total_new - total:+d})")

            return X_train_balanced, y_train_balanced

        except ImportError:
            print("\n⚠️  imbalanced-learn library not available.")
            print("    Install with: pip install imbalanced-learn")
            print("    Continuing without resampling...")
            return X_train, y_train
        except Exception as e:
            print(f"\n⚠️  Error during resampling: {e}")
            print("    Continuing without resampling...")
            return X_train, y_train

    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series) -> Dict:
        """
        Train multiple classification models.

        Models:
        1. Logistic Regression (baseline, interpretable)
        2. Random Forest (recommended, good default)
        3. XGBoost (best performance, requires tuning)

        Args:
            X_train, X_test: Scaled feature matrices
            y_train, y_test: Target variables

        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*80)
        print("STEP 6: TRAINING MODELS")
        print("="*80)

        models = {}

        # 1. Logistic Regression (Baseline)
        print("\n[1/3] Training Logistic Regression...")
        lr = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        )
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr
        print(f"✓ Logistic Regression trained")

        # 2. Random Forest (Recommended)
        print("\n[2/3] Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,        # More trees = better, but slower
            max_depth=10,            # Limit depth to prevent overfitting
            min_samples_split=10,    # Require at least 10 samples to split
            min_samples_leaf=5,      # Require at least 5 samples per leaf
            class_weight='balanced', # Handle class imbalance
            random_state=42,
            n_jobs=-1                # Use all CPU cores
        )
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        print(f"✓ Random Forest trained")

        # 3. XGBoost (Best Performance)
        try:
            import xgboost as xgb

            print("\n[3/3] Training XGBoost...")

            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model
            print(f"✓ XGBoost trained")

        except ImportError:
            print("⚠️  XGBoost not available. Install with: pip install xgboost")

        self.models = models
        return models

    def evaluate_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series) -> Dict:
        """
        Evaluate all models with comprehensive metrics.

        Metrics:
        - Accuracy
        - Precision (avoid false positives = bad trades)
        - Recall (catch opportunities)
        - F1-score (balance)
        - ROC-AUC (overall discrimination)

        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Target variables

        Returns:
            Dictionary of results
        """
        print("\n" + "="*80)
        print("STEP 7: MODEL EVALUATION")
        print("="*80)

        results = {}

        for name, model in self.models.items():
            print(f"\n{name}:")
            print("="*60)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Probability predictions (for ROC-AUC)
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = None

            # Calculate metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, zero_division=0)
            recall = recall_score(y_test, y_test_pred, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, zero_division=0)

            if y_test_proba is not None:
                roc_auc = roc_auc_score(y_test, y_test_proba)
            else:
                roc_auc = None

            # Store results
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba
            }

            # Print results
            print(f"Train Accuracy:    {train_acc:.4f}")
            print(f"Test Accuracy:     {test_acc:.4f}")
            print(f"Precision:         {precision:.4f}  (of predicted wins, % actually won)")
            print(f"Recall:            {recall:.4f}  (of actual wins, % we predicted)")
            print(f"F1-Score:          {f1:.4f}  (balanced metric)")
            if roc_auc:
                print(f"ROC-AUC:           {roc_auc:.4f}  (discrimination ability)")

            # Check for overfitting
            overfit_gap = train_acc - test_acc
            if overfit_gap > 0.10:
                print(f"⚠️  Warning: Possible overfitting (train-test gap: {overfit_gap:.4f})")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_test_pred)
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"                0       1")
            print(f"Actual    0   {cm[0,0]:4d}   {cm[0,1]:4d}  <- False Positives (bad trades)")
            print(f"          1   {cm[1,0]:4d}   {cm[1,1]:4d}  <- False Negatives (missed opps)")

        self.results = results
        return results

    def plot_feature_importance(self, model_name: str = 'Random Forest',
                                output_path: str = 'analysis/plots/feature_importance.png'):
        """
        Plot feature importance for tree-based models.

        Args:
            model_name: Name of model to analyze
            output_path: Where to save plot
        """
        model = self.models.get(model_name)
        if model is None:
            print(f"Model '{model_name}' not found")
            return

        if not hasattr(model, 'feature_importances_'):
            print(f"Model '{model_name}' does not support feature_importances_")
            return

        return plot_feature_importance(
            self.feature_names,
            model.feature_importances_,
            model_name,
            output_path
        )

    def plot_roc_curves(self, output_path: str = 'analysis/plots/roc_curves.png'):
        """
        Plot ROC curves for all models.

        Args:
            output_path: Where to save plot
        """
        plot_roc_curves(self.results, self.y_test, output_path)

    def _calculate_trade_outcome(self, outcome: pd.Series, gain_threshold: float,
                                  stop_loss_pct: float) -> Tuple[float, str, int]:
        """
        Calculate trade outcome using TRAILING STOP logic.

        A trailing stop adjusts upward as the stock price rises:
        - Initial stop loss: -stop_loss_pct from entry
        - As price rises, stop loss rises with it
        - Exit when price drops stop_loss_pct from the highest point reached

        Example with 2% trailing stop and 5% target:
        - Entry: $10.00, initial stop at $9.80 (-2%)
        - Price reaches $10.50 (+5%), hits target, exit at +5.0%
        - OR price reaches $10.40 (+4%), stop now at $10.19 (+1.9%)
        - Price drops to $10.19, trailing stop triggered, exit at +1.9%

        Args:
            outcome: Row from test DataFrame with outcome_tracking data
            gain_threshold: Target gain percentage
            stop_loss_pct: Trailing stop percentage (positive number, e.g., 2.0 for -2%)

        Returns:
            Tuple of (gain_percent, reason_string, holding_time_seconds)
        """
        from datetime import datetime

        # Try to load interval data from JSON if available
        intervals = None

        if hasattr(outcome, 'get'):
            if 'outcome_tracking' in outcome and outcome['outcome_tracking'] is not None:
                if isinstance(outcome['outcome_tracking'], dict):
                    intervals = outcome['outcome_tracking'].get('intervals', {})

        # Use interval-based trailing stop logic
        if intervals and len(intervals) > 0:
            # Process intervals in chronological order
            sorted_intervals = sorted(intervals.items(), key=lambda x: int(x[0]))

            peak_gain = 0.0  # Track highest gain reached during the trade

            for interval_sec_str, interval_data in sorted_intervals:
                interval_sec = int(interval_sec_str)

                # Check if we have interval range data
                has_range_data = ('interval_low' in interval_data and
                                 'interval_high' in interval_data and
                                 'interval_low_timestamp' in interval_data and
                                 'interval_high_timestamp' in interval_data)

                if has_range_data:
                    low_gain = interval_data['interval_low_gain_percent']
                    high_gain = interval_data['interval_high_gain_percent']
                    low_timestamp = interval_data['interval_low_timestamp']
                    high_timestamp = interval_data['interval_high_timestamp']

                    # OCO (One Cancels Other) BRACKET ORDER LOGIC:
                    # Check if BOTH target (limit order) and stop hit within this interval
                    # Use timestamps to determine which happened FIRST

                    # Update peak for trailing stop calculation
                    potential_peak = max(peak_gain, high_gain)

                    # Determine if target would be hit
                    target_hit = high_gain >= gain_threshold

                    # Determine if stop would be hit
                    stop_hit = False
                    stop_price = None
                    stop_reason = None

                    if peak_gain == 0:
                        # No previous gains - check initial stop loss
                        if low_gain <= -stop_loss_pct:
                            stop_hit = True
                            stop_price = -stop_loss_pct
                            stop_reason = f'initial_stop_at_{interval_sec_str}s'
                    else:
                        # Previous gains exist - check trailing stop
                        trailing_stop_level = peak_gain - stop_loss_pct
                        if low_gain <= trailing_stop_level:
                            stop_hit = True
                            stop_price = low_gain  # Exit at actual low, not stop level
                            stop_reason = f'trailing_stop_at_{interval_sec_str}s'

                    # OCO DECISION: If both hit, use timestamp to determine which came first
                    if target_hit and stop_hit:
                        if high_timestamp < low_timestamp:
                            # Target hit FIRST - exit with profit (limit order fills)
                            return (gain_threshold, f'target_hit_first_at_{interval_sec_str}s', interval_sec)
                        else:
                            # Stop hit FIRST - exit with loss (stop order fills)
                            return (stop_price, f'stop_hit_first_at_{interval_sec_str}s', interval_sec)

                    elif target_hit:
                        # Only target hit - exit with profit
                        peak_gain = potential_peak  # Update peak before exiting
                        return (gain_threshold, f'target_hit_at_{interval_sec_str}s', interval_sec)

                    elif stop_hit:
                        # Only stop hit - exit with loss
                        return (stop_price, stop_reason, interval_sec)

                    else:
                        # Neither hit - update peak and continue
                        peak_gain = potential_peak

                else:
                    # FALLBACK: Use snapshot price if no range data
                    if 'gain_percent' in interval_data:
                        gain = interval_data['gain_percent']

                        # Update peak
                        potential_peak = max(peak_gain, gain)

                        # Check if target hit
                        target_hit = gain >= gain_threshold

                        # Check if stop hit
                        stop_hit = False
                        if peak_gain > 0:
                            trailing_stop_level = peak_gain - stop_loss_pct
                            stop_hit = gain <= trailing_stop_level

                        # OCO logic for snapshot data
                        if target_hit:
                            # Target hit - lock in profit with limit order
                            return (gain_threshold, f'snapshot_target_at_{interval_sec_str}s', interval_sec)
                        elif stop_hit:
                            # Trailing stop hit - exit at actual price, not stop level
                            return (gain, f'snapshot_trailing_stop_at_{interval_sec_str}s', interval_sec)
                        else:
                            # Update peak and continue
                            peak_gain = potential_peak

            # If we went through all intervals without hitting target or trailing stop
            final_gain = outcome.get('final_gain_percent', 0)

            if final_gain >= gain_threshold:
                return (gain_threshold, 'final_target', 600)
            else:
                # Check if final price triggered trailing stop
                trailing_stop_level = peak_gain - stop_loss_pct
                if final_gain <= trailing_stop_level:
                    return (trailing_stop_level, 'final_trailing_stop', 600)
                else:
                    # Trade ended without hitting target or trailing stop
                    return (final_gain, 'final_no_target', 600)

        # FALLBACK: Use max_gain_percent if no interval data
        max_gain = outcome.get('max_gain_percent', 0)
        final_gain = outcome.get('final_gain_percent', max_gain)

        if max_gain >= gain_threshold:
            return (gain_threshold, 'legacy_target_hit', 600)
        else:
            # Apply trailing stop to legacy data
            trailing_stop_level = max_gain - stop_loss_pct
            if final_gain <= trailing_stop_level:
                return (trailing_stop_level, 'legacy_trailing_stop', 600)
            else:
                return (final_gain, 'legacy_no_target', 600)

    def simulate_trading(self, df: pd.DataFrame, model_name: str = 'Random Forest',
                         gain_threshold: float = 5.0, stop_loss_pct: float = 2.0) -> Dict:
        """
        Simulate trading P&L based on model predictions.

        NEW: Uses interval low/high price range data with chronological logic
        - If high timestamp < low timestamp: price went up first, check if hit target
        - If low timestamp < high timestamp: price went down first, check if hit stop loss
        - This provides more realistic simulation of actual trade outcomes

        Strategy:
        - Take trade only if model predicts success
        - Use interval price ranges to determine chronological order of events
        - If actual success: gain = interval_high or max_gain_percent (capped at 10%)
        - If actual failure: loss = -stop_loss_pct (default -1%)

        Args:
            df: Original DataFrame with outcome data
            model_name: Which model's predictions to use
            gain_threshold: Percentage gain threshold for success (default 5.0%)
            stop_loss_pct: Stop loss percentage (default 1.0%)

        Returns:
            Dictionary with trading simulation results
        """
        print("\n" + "="*80)
        print(f"STEP 9: TRADING SIMULATION - {model_name}")
        print(f"Target: {gain_threshold}% gain | Stop Loss: -{stop_loss_pct}%")
        print("="*80)

        result = self.results.get(model_name)
        if result is None:
            print(f"Model '{model_name}' not found")
            return {}

        y_test_pred = result['y_test_pred']

        # Get corresponding outcome data
        test_outcomes = df.loc[self.y_test.index]

        # Simulate trading
        trades_taken = y_test_pred == 1
        num_trades = trades_taken.sum()

        if num_trades == 0:
            print("⚠️  Model predicted no trades")
            return {}

        # Calculate P&L for each trade using chronological interval logic
        import json
        from datetime import datetime

        pnl_list = []
        detailed_results = []  # Track detailed outcome for each trade

        for idx, take_trade in zip(test_outcomes.index, trades_taken):
            if not take_trade:
                continue  # Skip this alert

            outcome = test_outcomes.loc[idx]

            # Load full JSON data to get interval information
            try:
                symbol = outcome.get('symbol', 'N/A')
                timestamp = outcome.get('timestamp', '')

                # Find the JSON file for this alert
                # Format: historical_data/YYYY-MM-DD/squeeze_alerts_sent/alert_SYMBOL_YYYY-MM-DD_HHMMSS.json
                date_obj = pd.to_datetime(timestamp)
                date_str = date_obj.strftime('%Y-%m-%d')
                time_str = date_obj.strftime('%H%M%S')

                json_file = Path(self.json_dir) / date_str / 'squeeze_alerts_sent' / f'alert_{symbol}_{date_str}_{time_str}.json'

                if json_file.exists():
                    with open(json_file, 'r') as f:
                        full_data = json.load(f)

                    # Create enhanced row with full outcome_tracking structure
                    enhanced_outcome = outcome.copy()
                    if 'outcome_tracking' in full_data:
                        enhanced_outcome['outcome_tracking'] = full_data['outcome_tracking']

                    gain, outcome_reason, holding_time = self._calculate_trade_outcome(
                        enhanced_outcome, gain_threshold, stop_loss_pct
                    )
                else:
                    # Fallback if JSON not found
                    gain, outcome_reason, holding_time = self._calculate_trade_outcome(
                        outcome, gain_threshold, stop_loss_pct
                    )
            except Exception as e:
                # Fallback on error
                gain, outcome_reason, holding_time = self._calculate_trade_outcome(
                    outcome, gain_threshold, stop_loss_pct
                )

            pnl_list.append(gain)
            detailed_results.append({
                'symbol': outcome.get('symbol', 'N/A'),
                'gain': gain,
                'reason': outcome_reason,
                'holding_time_sec': holding_time
            })

        # Calculate statistics
        total_pnl = sum(pnl_list)
        avg_pnl = np.mean(pnl_list)
        win_rate = sum(1 for x in pnl_list if x > 0) / len(pnl_list)
        avg_win = np.mean([x for x in pnl_list if x > 0]) if any(x > 0 for x in pnl_list) else 0
        avg_loss = np.mean([x for x in pnl_list if x < 0]) if any(x < 0 for x in pnl_list) else 0

        # Print results
        print(f"\nTrading Simulation Results:")
        print(f"="*60)
        print(f"Total Trades Taken:    {num_trades}")
        print(f"Win Rate:              {win_rate*100:.1f}%")
        print(f"Average Trade P&L:     {avg_pnl:.2f}%")
        print(f"Total P&L:             {total_pnl:.2f}%")
        print(f"Average Win:           {avg_win:.2f}%")
        print(f"Average Loss:          {avg_loss:.2f}%")
        print(f"Profit Factor:         {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")

        # Compare to taking all trades (using same chronological logic)
        all_trades_pnl = []
        for idx in test_outcomes.index:
            outcome = test_outcomes.loc[idx]

            # Load full JSON data for accurate calculation
            try:
                symbol = outcome.get('symbol', 'N/A')
                timestamp = outcome.get('timestamp', '')
                date_obj = pd.to_datetime(timestamp)
                date_str = date_obj.strftime('%Y-%m-%d')
                time_str = date_obj.strftime('%H%M%S')
                json_file = Path(self.json_dir) / date_str / 'squeeze_alerts_sent' / f'alert_{symbol}_{date_str}_{time_str}.json'

                if json_file.exists():
                    with open(json_file, 'r') as f:
                        full_data = json.load(f)
                    enhanced_outcome = outcome.copy()
                    if 'outcome_tracking' in full_data:
                        enhanced_outcome['outcome_tracking'] = full_data['outcome_tracking']
                    gain, _, _ = self._calculate_trade_outcome(
                        enhanced_outcome, gain_threshold, stop_loss_pct
                    )
                else:
                    gain, _, _ = self._calculate_trade_outcome(
                        outcome, gain_threshold, stop_loss_pct
                    )
            except Exception:
                gain, _, _ = self._calculate_trade_outcome(
                    outcome, gain_threshold, stop_loss_pct
                )

            all_trades_pnl.append(gain)

        all_trades_avg = np.mean(all_trades_pnl)
        print(f"\n📊 Comparison:")
        print(f"  Model-filtered:      {avg_pnl:.2f}% per trade")
        print(f"  Take all alerts:     {all_trades_avg:.2f}% per trade")
        print(f"  Improvement:         {avg_pnl - all_trades_avg:.2f}%")

        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'all_trades_avg': all_trades_avg,
            'improvement': avg_pnl - all_trades_avg,
            'detailed_results': detailed_results
        }

    def analyze_holding_period_distribution(self, trading_results: Dict,
                                           gain_threshold: float = 5.0,
                                           output_path: str = 'analysis/plots/holding_period_distribution.png'):
        """
        Analyze and visualize the distribution of holding periods for wins vs losses.

        Args:
            trading_results: Results from simulate_trading containing detailed_results
            gain_threshold: Gain threshold used for classification
            output_path: Where to save the distribution plot
        """
        print("\n" + "="*80)
        print("ANALYZING HOLDING PERIOD DISTRIBUTION")
        print("="*80)

        detailed_results = trading_results.get('detailed_results', [])
        if not detailed_results:
            print("⚠️  No detailed results available")
            return

        # Separate wins and losses based on profitability
        # Win = positive gain, Loss = negative or zero gain
        wins = [r for r in detailed_results if r['gain'] > 0]
        losses = [r for r in detailed_results if r['gain'] <= 0]

        win_times = [r['holding_time_sec'] for r in wins]
        loss_times = [r['holding_time_sec'] for r in losses]

        # Calculate statistics
        print(f"\nHolding Period Statistics:")
        print(f"="*60)

        if win_times:
            print(f"\nWINS ({len(wins)} trades):")
            print(f"  Average holding time: {np.mean(win_times):.1f} seconds ({np.mean(win_times)/60:.1f} minutes)")
            print(f"  Median holding time:  {np.median(win_times):.1f} seconds ({np.median(win_times)/60:.1f} minutes)")
            print(f"  Min holding time:     {min(win_times)} seconds ({min(win_times)/60:.1f} minutes)")
            print(f"  Max holding time:     {max(win_times)} seconds ({max(win_times)/60:.1f} minutes)")
            print(f"  Std deviation:        {np.std(win_times):.1f} seconds")

        if loss_times:
            print(f"\nLOSSES ({len(losses)} trades):")
            print(f"  Average holding time: {np.mean(loss_times):.1f} seconds ({np.mean(loss_times)/60:.1f} minutes)")
            print(f"  Median holding time:  {np.median(loss_times):.1f} seconds ({np.median(loss_times)/60:.1f} minutes)")
            print(f"  Min holding time:     {min(loss_times)} seconds ({min(loss_times)/60:.1f} minutes)")
            print(f"  Max holding time:     {max(loss_times)} seconds ({max(loss_times)/60:.1f} minutes)")
            print(f"  Std deviation:        {np.std(loss_times):.1f} seconds")

        if win_times and loss_times:
            print(f"\nCOMPARISON:")
            diff = np.mean(win_times) - np.mean(loss_times)
            win_rate = len(wins) / (len(wins) + len(losses)) * 100
            print(f"  Win rate: {win_rate:.1f}% ({len(wins)}/{len(wins)+len(losses)} profitable)")
            if diff > 0:
                print(f"  Wins held {diff:.1f} seconds ({diff/60:.1f} minutes) LONGER on average")
            else:
                print(f"  Losses held {abs(diff):.1f} seconds ({abs(diff)/60:.1f} minutes) LONGER on average")

        # Create distribution plots
        import matplotlib.pyplot as plt

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Holding Period Distribution - Wins vs Losses (Target: {gain_threshold}%)',
                    fontsize=14, fontweight='bold')

        # 1. Histogram - wins vs losses (seconds)
        ax1 = axes[0, 0]
        bins = [10, 20, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
        if win_times and loss_times:
            ax1.hist([win_times, loss_times], bins=bins, alpha=0.7,
                    label=[f'Wins (n={len(wins)})', f'Losses (n={len(losses)})'],
                    color=['green', 'red'], edgecolor='black')
        elif win_times:
            ax1.hist(win_times, bins=bins, alpha=0.7,
                    label=f'Wins (n={len(wins)})', color='green', edgecolor='black')
        elif loss_times:
            ax1.hist(loss_times, bins=bins, alpha=0.7,
                    label=f'Losses (n={len(losses)})', color='red', edgecolor='black')
        ax1.set_xlabel('Holding Time (seconds)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Distribution by Time (Seconds)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Histogram - wins vs losses (minutes)
        ax2 = axes[0, 1]
        bins_minutes = [0.17, 0.33, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        win_times_min = [t/60 for t in win_times]
        loss_times_min = [t/60 for t in loss_times]
        if win_times_min and loss_times_min:
            ax2.hist([win_times_min, loss_times_min], bins=bins_minutes, alpha=0.7,
                    label=[f'Wins (n={len(wins)})', f'Losses (n={len(losses)})'],
                    color=['green', 'red'], edgecolor='black')
        elif win_times_min:
            ax2.hist(win_times_min, bins=bins_minutes, alpha=0.7,
                    label=f'Wins (n={len(wins)})', color='green', edgecolor='black')
        elif loss_times_min:
            ax2.hist(loss_times_min, bins=bins_minutes, alpha=0.7,
                    label=f'Losses (n={len(losses)})', color='red', edgecolor='black')
        ax2.set_xlabel('Holding Time (minutes)', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Distribution by Time (Minutes)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Box plot comparison
        ax3 = axes[1, 0]
        data_to_plot = []
        labels = []
        if win_times:
            data_to_plot.append([t/60 for t in win_times])
            labels.append(f'Wins\n(n={len(wins)})')
        if loss_times:
            data_to_plot.append([t/60 for t in loss_times])
            labels.append(f'Losses\n(n={len(losses)})')

        if data_to_plot:
            bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightgreen' if 'Win' in label else 'lightcoral' for label in labels]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        ax3.set_ylabel('Holding Time (minutes)', fontweight='bold')
        ax3.set_title('Box Plot Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Cumulative distribution
        ax4 = axes[1, 1]
        if win_times:
            win_times_sorted = np.sort(win_times_min)
            win_cumulative = np.arange(1, len(win_times_sorted)+1) / len(win_times_sorted) * 100
            ax4.plot(win_times_sorted, win_cumulative, label=f'Wins (n={len(wins)})',
                    color='green', linewidth=2, marker='o', markersize=3)
        if loss_times:
            loss_times_sorted = np.sort(loss_times_min)
            loss_cumulative = np.arange(1, len(loss_times_sorted)+1) / len(loss_times_sorted) * 100
            ax4.plot(loss_times_sorted, loss_cumulative, label=f'Losses (n={len(losses)})',
                    color='red', linewidth=2, marker='s', markersize=3)
        ax4.set_xlabel('Holding Time (minutes)', fontweight='bold')
        ax4.set_ylabel('Cumulative Percentage (%)', fontweight='bold')
        ax4.set_title('Cumulative Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(left=0)
        ax4.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved holding period distribution plot to: {output_file}")
        plt.close()

        # Print time bucket analysis
        print(f"\nTime Bucket Analysis:")
        print(f"="*60)
        time_buckets = [
            (0, 30, "0-30s"),
            (30, 60, "30-60s"),
            (60, 120, "1-2 min"),
            (120, 300, "2-5 min"),
            (300, 600, "5-10 min")
        ]

        for min_time, max_time, label in time_buckets:
            wins_in_bucket = sum(1 for t in win_times if min_time <= t < max_time)
            losses_in_bucket = sum(1 for t in loss_times if min_time <= t < max_time)
            total_in_bucket = wins_in_bucket + losses_in_bucket

            if total_in_bucket > 0:
                win_pct = (wins_in_bucket / total_in_bucket * 100) if total_in_bucket else 0
                print(f"  {label:12s}: {wins_in_bucket:3d} wins, {losses_in_bucket:3d} losses "
                      f"({win_pct:.1f}% win rate)")

    def generate_summary_report(self, output_path: str = 'analysis/prediction_summary.txt'):
        """
        Generate comprehensive text summary of results.

        Args:
            output_path: Where to save report
        """
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)

        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SQUEEZE ALERT OUTCOME PREDICTION - SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATASET\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Observations: {len(self.df)}\n")
            f.write(f"Training Set: {len(self.X_train)} samples\n")
            f.write(f"Test Set: {len(self.X_test)} samples\n")
            f.write(f"Features: {len(self.feature_names)}\n")
            f.write(f"Target Variable: achieved_5pct (reached 5% gain in 10 minutes)\n\n")

            # Add class distribution
            target_col = 'achieved_5pct'
            if target_col in self.df.columns:
                total = len(self.df)
                successes = int(self.df[target_col].sum())
                failures = total - successes
                success_pct = (successes / total * 100) if total > 0 else 0
                failure_pct = (failures / total * 100) if total > 0 else 0
                imbalance_ratio = failures / successes if successes > 0 else 0

                f.write("CLASS DISTRIBUTION\n")
                f.write("-"*80 + "\n")
                f.write(f"Successes: {successes} ({success_pct:.1f}%)\n")
                f.write(f"Failures:  {failures} ({failure_pct:.1f}%)\n")
                f.write(f"Imbalance Ratio: 1 : {imbalance_ratio:.2f}\n")

                if 45 <= success_pct <= 55:
                    f.write("Balance Assessment: ✓ Excellent (45-55%)\n\n")
                elif 40 <= success_pct <= 60:
                    f.write("Balance Assessment: ~ Good (40-60%)\n\n")
                elif 35 <= success_pct <= 65:
                    f.write("Balance Assessment: ⚠ Moderate Imbalance (35-65%)\n\n")
                else:
                    f.write("Balance Assessment: ❌ Severe Imbalance - May cause overfitting\n\n")
            else:
                f.write("\n")

            f.write("MODEL PERFORMANCE\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}\n")
            f.write("-"*80 + "\n")

            for name, result in self.results.items():
                f.write(f"{name:<20} {result['test_accuracy']:>10.4f} {result['precision']:>10.4f} "
                       f"{result['recall']:>10.4f} {result['f1_score']:>10.4f}")
                if result['roc_auc']:
                    f.write(f" {result['roc_auc']:>10.4f}\n")
                else:
                    f.write(f" {'N/A':>10}\n")

            f.write("\n")
            f.write("INTERPRETATION\n")
            f.write("-"*80 + "\n")
            f.write("Precision: Of squeezes predicted as successful, % that actually succeeded\n")
            f.write("Recall:    Of actually successful squeezes, % that we predicted\n")
            f.write("F1-Score:  Balanced metric (harmonic mean of precision and recall)\n")
            f.write("ROC-AUC:   Overall discrimination ability (0.5=random, 1.0=perfect)\n\n")

            f.write("RECOMMENDATION\n")
            f.write("-"*80 + "\n")

            # Find best model based on F1-score
            best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
            f.write(f"Best Model: {best_model[0]}\n")
            f.write(f"  F1-Score: {best_model[1]['f1_score']:.4f}\n")
            f.write(f"  Precision: {best_model[1]['precision']:.4f}\n")
            f.write(f"  Recall: {best_model[1]['recall']:.4f}\n\n")

            if best_model[1]['f1_score'] > 0.60:
                f.write("✓ Model shows good predictive performance (F1 > 0.60)\n")
                f.write("  Recommended for trade filtering in production\n")
            elif best_model[1]['f1_score'] > 0.50:
                f.write("~ Model shows moderate predictive performance (F1 > 0.50)\n")
                f.write("  Consider additional feature engineering or more data\n")
            else:
                f.write("✗ Model shows weak predictive performance (F1 < 0.50)\n")
                f.write("  Not recommended for production use. Investigate:\n")
                f.write("  - Need more training data\n")
                f.write("  - Try different features\n")
                f.write("  - Check for data quality issues\n")

        print(f"✓ Saved summary report to: {output_file}")

    def analyze_class_distribution(self, gain_threshold: float = 5.0) -> Dict[str, Any]:
        """
        Analyze class distribution (success vs failure) for the dataset.

        Args:
            gain_threshold: Gain percentage threshold for success

        Returns:
            Dictionary with class distribution statistics
        """
        target_col = f'achieved_{int(gain_threshold)}pct'

        if target_col not in self.df.columns:
            print(f"⚠️  Target column {target_col} not found in dataset")
            return {}

        # Count successes and failures
        total = len(self.df)
        successes = self.df[target_col].sum()
        failures = total - successes

        success_pct = (successes / total * 100) if total > 0 else 0
        failure_pct = (failures / total * 100) if total > 0 else 0

        imbalance_ratio = failures / successes if successes > 0 else 0

        stats = {
            'total': total,
            'successes': int(successes),
            'failures': int(failures),
            'success_pct': success_pct,
            'failure_pct': failure_pct,
            'imbalance_ratio': imbalance_ratio
        }

        print(f"\n{'='*80}")
        print(f"CLASS DISTRIBUTION - {int(gain_threshold)}% TARGET")
        print(f"{'='*80}")
        print(f"Total Alerts:    {total}")
        print(f"Successes:       {successes} ({success_pct:.1f}%)")
        print(f"Failures:        {failures} ({failure_pct:.1f}%)")
        print(f"Imbalance Ratio: 1 : {imbalance_ratio:.2f}")
        print(f"{'='*80}")

        # Assess balance quality
        if 45 <= success_pct <= 55:
            print("✓ EXCELLENT BALANCE (45-55% success rate)")
        elif 40 <= success_pct <= 60:
            print("~ GOOD BALANCE (40-60% success rate)")
        elif 35 <= success_pct <= 65:
            print("⚠️  MODERATE IMBALANCE (35-65% success rate)")
        else:
            print("❌ SEVERE IMBALANCE - May cause model overfitting")
            print("   Recommendation: Use downsampling, upsampling, or different threshold")

        return stats

    def plot_class_distribution(self, gain_threshold: float = 5.0,
                                output_path: str = 'analysis/plots/class_distribution_5pct.png',
                                y_train_balanced: Optional[pd.Series] = None):
        """
        Create pie charts showing both original and SMOTE-balanced class distributions.

        Args:
            gain_threshold: Gain percentage threshold for success
            output_path: Where to save the plot
            y_train_balanced: Balanced training labels after SMOTE (optional)
        """
        stats = self.analyze_class_distribution(gain_threshold)
        plot_class_distribution(stats, gain_threshold, output_path, y_train_balanced)

    def analyze_by_price_category(self, gain_threshold: float = 5.0,
                                   output_path: str = 'analysis/plots/price_category_analysis_5pct.png') -> pd.DataFrame:
        """
        Analyze squeeze outcomes by price category to identify best performing bins.

        Args:
            gain_threshold: Gain percentage threshold for success
            output_path: Where to save visualization

        Returns:
            DataFrame with performance metrics by price category
        """
        print("\n" + "="*80)
        print(f"PRICE CATEGORY ANALYSIS - {gain_threshold}% TARGET")
        print("="*80)

        # Check if price_category exists in dataframe
        if 'price_category' not in self.df.columns:
            print("⚠️  price_category not found in dataset")
            return None

        # Create target column name
        target_col = f'achieved_{int(gain_threshold)}pct'

        # Group by price_category and calculate metrics
        grouped = self.df.groupby('price_category').agg({
            'symbol': 'count',
            target_col: 'mean',
            'achieved_10pct': 'mean',
            'reached_stop_loss': 'mean',
            'max_gain_percent': 'mean',
            'final_gain_percent': 'mean'
        }).round(4)

        grouped.columns = ['Count', f'Win_Rate_{int(gain_threshold)}pct', 'Win_Rate_10pct',
                          'Stop_Loss_Rate', 'Avg_Max_Gain', 'Avg_Final_Gain']

        # Calculate profitability score (win rate - stop loss rate)
        grouped['Profitability_Score'] = grouped[f'Win_Rate_{int(gain_threshold)}pct'] - grouped['Stop_Loss_Rate']

        # Sort by win rate descending
        grouped = grouped.sort_values(f'Win_Rate_{int(gain_threshold)}pct', ascending=False)

        # Print table
        print(f"\nPerformance by Price Category (sorted by {gain_threshold}% win rate):")
        print("="*95)
        print(f"{'Price':<10} {'Count':>7} {f'{int(gain_threshold)}% Win':>10} {'10% Win':>10} "
              f"{'Stop Loss':>11} {'Avg Max':>10} {'Avg Final':>11} {'Profit Score':>13}")
        print("-"*95)

        for price_cat, row in grouped.iterrows():
            print(f"{price_cat:<10} {int(row['Count']):7d} "
                  f"{row[f'Win_Rate_{int(gain_threshold)}pct']*100:9.1f}% "
                  f"{row['Win_Rate_10pct']*100:9.1f}% "
                  f"{row['Stop_Loss_Rate']*100:10.1f}% "
                  f"{row['Avg_Max_Gain']:9.2f}% "
                  f"{row['Avg_Final_Gain']:10.2f}% "
                  f"{row['Profitability_Score']*100:12.1f}%")

        print("="*95)

        # Find best bin
        best_bin = grouped['Profitability_Score'].idxmax()
        print(f"\n🏆 Best Performing Bin: {best_bin}")
        print(f"   {int(gain_threshold)}% Win Rate: {grouped.loc[best_bin, f'Win_Rate_{int(gain_threshold)}pct']*100:.1f}%")
        print(f"   Stop Loss Rate: {grouped.loc[best_bin, 'Stop_Loss_Rate']*100:.1f}%")
        print(f"   Profitability Score: {grouped.loc[best_bin, 'Profitability_Score']*100:.1f}%")

        # Create visualization
        plot_price_category_analysis(grouped, gain_threshold, output_path)

        return grouped

    def analyze_by_time_of_day(self, gain_threshold: float = 5.0,
                                output_path: str = 'analysis/plots/time_of_day_analysis_5pct.png') -> pd.DataFrame:
        """
        Analyze squeeze outcomes by time of day (30-minute bins).

        Args:
            gain_threshold: Gain percentage threshold for success
            output_path: Where to save visualization

        Returns:
            DataFrame with performance metrics by time bin
        """
        print("\n" + "="*80)
        print(f"TIME OF DAY ANALYSIS - {gain_threshold}% TARGET")
        print("="*80)

        # Create time bins from timestamp
        df_copy = self.df.copy()

        # Parse timestamp and extract time
        df_copy['timestamp_parsed'] = pd.to_datetime(df_copy['timestamp'])
        df_copy['hour'] = df_copy['timestamp_parsed'].dt.hour
        df_copy['minute'] = df_copy['timestamp_parsed'].dt.minute

        # Create 30-minute time bins (format: "HH:MM-HH:MM")
        def create_time_bin(row):
            hour = row['hour']
            minute = row['minute']

            # Round down to nearest 30-minute interval
            if minute < 30:
                start_min = 0
                end_min = 30
            else:
                start_min = 30
                end_min = 0
                if end_min == 0:
                    end_hour = hour + 1
                else:
                    end_hour = hour

            if minute < 30:
                end_hour = hour
            else:
                end_hour = hour + 1

            start_time = f"{hour:02d}:{start_min:02d}"
            end_time = f"{end_hour:02d}:{end_min:02d}"

            return f"{start_time}-{end_time}"

        df_copy['time_bin'] = df_copy.apply(create_time_bin, axis=1)

        # Create target column name
        target_col = f'achieved_{int(gain_threshold)}pct'

        # Group by time_bin and calculate metrics
        grouped = df_copy.groupby('time_bin').agg({
            'symbol': 'count',
            target_col: 'mean',
            'achieved_10pct': 'mean',
            'reached_stop_loss': 'mean',
            'max_gain_percent': 'mean',
            'final_gain_percent': 'mean'
        }).round(4)

        grouped.columns = ['Count', f'Win_Rate_{int(gain_threshold)}pct', 'Win_Rate_10pct',
                          'Stop_Loss_Rate', 'Avg_Max_Gain', 'Avg_Final_Gain']

        # Calculate profitability score
        grouped['Profitability_Score'] = grouped[f'Win_Rate_{int(gain_threshold)}pct'] - grouped['Stop_Loss_Rate']

        # Sort by time bin (chronologically)
        grouped = grouped.sort_index()

        # Print table
        print(f"\nPerformance by Time of Day (sorted chronologically, 30-min bins):")
        print("="*95)
        print(f"{'Time Bin':<13} {'Count':>7} {f'{int(gain_threshold)}% Win':>10} {'10% Win':>10} "
              f"{'Stop Loss':>11} {'Avg Max':>10} {'Avg Final':>11} {'Profit Score':>13}")
        print("-"*95)

        for time_bin, row in grouped.iterrows():
            print(f"{time_bin:<13} {int(row['Count']):7d} "
                  f"{row[f'Win_Rate_{int(gain_threshold)}pct']*100:9.1f}% "
                  f"{row['Win_Rate_10pct']*100:9.1f}% "
                  f"{row['Stop_Loss_Rate']*100:10.1f}% "
                  f"{row['Avg_Max_Gain']:9.2f}% "
                  f"{row['Avg_Final_Gain']:10.2f}% "
                  f"{row['Profitability_Score']*100:12.1f}%")

        print("="*95)

        # Find best time bin
        best_bin = grouped['Profitability_Score'].idxmax()
        print(f"\n🏆 Best Performing Time: {best_bin}")
        print(f"   {int(gain_threshold)}% Win Rate: {grouped.loc[best_bin, f'Win_Rate_{int(gain_threshold)}pct']*100:.1f}%")
        print(f"   Stop Loss Rate: {grouped.loc[best_bin, 'Stop_Loss_Rate']*100:.1f}%")
        print(f"   Profitability Score: {grouped.loc[best_bin, 'Profitability_Score']*100:.1f}%")

        # Highlight first hour after open (9:30-10:30)
        morning_bins = ['09:30-10:00', '10:00-10:30']
        morning_data = grouped.loc[grouped.index.intersection(morning_bins)]

        if len(morning_data) > 0:
            print(f"\n📊 First Hour After Open (9:30-10:30 AM):")
            for time_bin in morning_bins:
                if time_bin in grouped.index:
                    row = grouped.loc[time_bin]
                    print(f"   {time_bin}: {row[f'Win_Rate_{int(gain_threshold)}pct']*100:.1f}% win rate, "
                          f"{row['Profitability_Score']*100:.1f}% profit score ({int(row['Count'])} alerts)")

        # Create visualization
        plot_time_of_day_analysis(grouped, gain_threshold, output_path)

        return grouped

def ensure_features_updated(features_csv: str = "analysis/squeeze_alerts_independent_features.csv",
                           data_dir: str = "/home/wilsonb/dl/github.com/Z223I/alpaca/historical_data") -> bool:
    """
    Intelligently check if features CSV needs updating and regenerate if necessary.

    Checks:
    1. Does the CSV exist?
    2. Are there date directories newer than the latest data in the CSV?

    Args:
        features_csv: Path to the features CSV file
        data_dir: Path to historical_data directory

    Returns:
        True if CSV was regenerated, False if it was already up to date
    """
    from datetime import datetime

    csv_path = Path(features_csv)
    data_path = Path(data_dir)

    # Get all date directories (YYYY-MM-DD format)
    date_dirs = []
    for date_dir in data_path.glob('????-??-??'):
        if date_dir.is_dir() and (date_dir / 'squeeze_alerts_sent').exists():
            date_dirs.append(date_dir.name)

    if not date_dirs:
        print("⚠️  No date directories found - skipping features update")
        return False

    date_dirs.sort()
    latest_data_date = date_dirs[-1]

    # Check if CSV exists and get its latest date
    needs_update = False
    if not csv_path.exists():
        print(f"✓ Features CSV not found - will generate: {features_csv}")
        needs_update = True
    else:
        # Read CSV and find latest timestamp
        try:
            df = pd.read_csv(csv_path)
            if 'timestamp' in df.columns and len(df) > 0:
                # Extract dates from timestamps
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                latest_csv_date = df['date'].max().strftime('%Y-%m-%d')

                if latest_data_date > latest_csv_date:
                    print(f"✓ New data detected: CSV has data through {latest_csv_date}, "
                          f"but {latest_data_date} directory exists")
                    needs_update = True
                else:
                    print(f"✓ Features CSV is up to date (latest: {latest_csv_date})")
                    return False
            else:
                print("⚠️  Features CSV exists but appears invalid - will regenerate")
                needs_update = True
        except Exception as e:
            print(f"⚠️  Error reading features CSV: {e} - will regenerate")
            needs_update = True

    if needs_update:
        print("\n" + "="*80)
        print("REGENERATING FEATURES CSV")
        print("="*80)
        print(f"Running statistical analysis on all data in: {data_dir}")
        print("="*80 + "\n")

        # Import and run the statistical analysis
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from squeeze_alerts_statistical_analysis import SqueezeAlertsAnalyzer

            # Initialize analyzer
            analyzer = SqueezeAlertsAnalyzer(data_dir)

            # Run the analysis pipeline
            analyzer.load_alerts()
            analyzer.engineer_features()
            analyzer.select_independent_features()
            analyzer.export_clean_dataset(output_path=features_csv)

            print("\n" + "="*80)
            print("✓ Features CSV regenerated successfully")
            print("="*80 + "\n")
            return True

        except Exception as e:
            print(f"\n❌ Error regenerating features CSV: {e}")
            print("Please run manually: python analysis/squeeze_alerts_statistical_analysis.py")
            raise

    return False


def train(gain_threshold: float = 5.0, end_date: str = "2025-12-16",
          min_price: Optional[float] = None, max_price: Optional[float] = None):
    """
    Train models and generate analysis for a given gain threshold.

    Args:
        gain_threshold: Percentage gain threshold for success classification (default: 5.0)
        end_date: Training data cutoff date (YYYY-MM-DD). Data after this is reserved for prediction.
        min_price: Minimum squeeze entry price to include in training (optional)
        max_price: Maximum squeeze entry price to include in training (optional)
    """

    # Configuration
    json_dir = "/home/wilsonb/dl/github.com/Z223I/alpaca/historical_data"
    features_csv = "analysis/squeeze_alerts_independent_features.csv"

    # Auto-update features CSV if needed
    ensure_features_updated(features_csv=features_csv, data_dir=json_dir)

    print("="*80)
    print("SQUEEZE ALERT OUTCOME PREDICTION")
    print("="*80)
    print("Predicting price action using statistically independent features")
    print(f"Target: Predict if squeeze reaches {gain_threshold}% gain within 10 minutes")
    print()
    print("NOTE: Some features have missing data in current dataset:")
    print("  - ema_spread_pct: ~10% missing (price-normalized EMA spread)")
    print("  - macd_histogram: ~15% missing (will be 100% with daily MACD)")
    print("  We handle this via imputation for now.")
    print("="*80)

    # Initialize predictor
    predictor = SqueezeOutcomePredictor(json_dir, features_csv)

    # Step 1: Extract outcomes from all directories starting 2025-12-12
    # CRITICAL: Use end_date to prevent temporal contamination
    # Training should ONLY use data BEFORE prediction dates
    print(f"\n⚠️  Training data cutoff: {end_date}")
    print(f"   Data from {end_date} onwards is reserved for prediction/testing")
    outcomes_df = predictor.extract_outcomes(
        gain_threshold=gain_threshold,
        start_date="2025-12-12",
        end_date=end_date  # Configurable via --end-date argument
    )

    # Step 2: Merge with features
    df = predictor.merge_with_features(outcomes_df)
    predictor.df = df

    # Step 2.5: Filter by price if specified
    if min_price is not None or max_price is not None:
        print("\n" + "="*80)
        print("PRICE FILTERING")
        print("="*80)
        original_count = len(df)

        if min_price is not None:
            df = df[df['squeeze_entry_price'] >= min_price]
            print(f"Min price: ${min_price:.2f}")

        if max_price is not None:
            df = df[df['squeeze_entry_price'] <= max_price]
            print(f"Max price: ${max_price:.2f}")

        filtered_count = len(df)
        removed_count = original_count - filtered_count
        print(f"Filtered: {original_count} → {filtered_count} alerts ({removed_count} removed)")
        print("="*80)

        # Reset index to avoid issues with non-contiguous indices
        df = df.reset_index(drop=True)

        # Update predictor's dataframe
        predictor.df = df

    # Step 3: Prepare features and target
    # Target: achieved_{threshold}pct (binary - did squeeze reach threshold% gain?)
    target_col = f'achieved_{int(gain_threshold)}pct'
    X, y = predictor.prepare_features(df, target_variable=target_col)

    # Step 4: Time-based train/test split
    X_train, X_test, y_train, y_test = predictor.time_based_split(df, X, y, test_size=0.20)

    # Store for later use
    predictor.X_train = X_train
    predictor.X_test = X_test
    predictor.y_train = y_train
    predictor.y_test = y_test

    # Step 5: Scale features
    X_train_scaled, X_test_scaled = predictor.scale_features(X_train, X_test)

    # Store scaler for later use in predictions
    predictor.scaler = predictor.scaler  # scaler is created in scale_features method

    # Step 5B: REMOVED SMOTE - Now using class weights in models instead
    # SMOTE creates synthetic samples that can cause overfitting
    # Class weights provide similar benefit without synthetic data
    print("\n" + "="*80)
    print("CLASS IMBALANCE HANDLING")
    print("="*80)
    print("Using class_weight='balanced' in models instead of SMOTE")
    print("This avoids overfitting to synthetic samples")
    print("="*80)

    # Step 6: Train models (using original training data with class weights)
    models = predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Step 7: Evaluate models (using original training data)
    results = predictor.evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Step 8: Feature importance
    # Format threshold for filename (1.5 -> "1.5", 2.0 -> "2")
    threshold_str = f"{gain_threshold:.1f}".rstrip('0').rstrip('.') if gain_threshold % 1 else str(int(gain_threshold))
    threshold_suffix = f"_{threshold_str}pct"
    importance_df = predictor.plot_feature_importance(
        model_name='Random Forest',
        output_path=f'analysis/plots/feature_importance{threshold_suffix}.png'
    )

    # Step 9: ROC curves
    predictor.plot_roc_curves(output_path=f'analysis/plots/roc_curves{threshold_suffix}.png')

    # Step 10: Trading simulation
    trading_results = predictor.simulate_trading(df, model_name='Random Forest',
                                                  gain_threshold=gain_threshold,
                                                  stop_loss_pct=1.0)

    # Step 10a: Analyze holding period distribution
    predictor.analyze_holding_period_distribution(
        trading_results,
        gain_threshold=gain_threshold,
        output_path=f'analysis/plots/holding_period_distribution{threshold_suffix}.png'
    )

    # Step 11: Generate summary report
    predictor.generate_summary_report(output_path=f'analysis/prediction_summary{threshold_suffix}.txt')

    # Step 11.5: Class distribution analysis
    predictor.plot_class_distribution(
        gain_threshold=gain_threshold,
        output_path=f'analysis/plots/class_distribution{threshold_suffix}.png',
        y_train_balanced=None  # No longer using SMOTE balancing
    )

    # Step 12: Price category analysis
    price_analysis = predictor.analyze_by_price_category(
        gain_threshold=gain_threshold,
        output_path=f'analysis/plots/price_category_analysis{threshold_suffix}.png'
    )

    # Step 13: Time of day analysis
    time_analysis = predictor.analyze_by_time_of_day(
        gain_threshold=gain_threshold,
        output_path=f'analysis/plots/time_of_day_analysis{threshold_suffix}.png'
    )

    # Step 14: Save XGBoost model for all thresholds
    if 'XGBoost' in predictor.models:
        print("\n" + "="*80)
        print(f"SAVING XGBOOST MODEL FOR {threshold_str}% TARGET")
        print("="*80)

        model_path = Path(f'analysis/xgboost_model{threshold_suffix}.json')
        predictor.models['XGBoost'].save_model(model_path)
        print(f"✓ Saved XGBoost model to: {model_path}")

        # Also save feature names and scaler for later use
        import json
        import pickle

        # Save the scaler
        scaler_path = Path(f'analysis/xgboost_model{threshold_suffix}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(predictor.scaler, f)
        print(f"✓ Saved feature scaler to: {scaler_path}")

        feature_info = {
            'feature_names': predictor.feature_names,
            'gain_threshold': gain_threshold,
            'train_samples': len(predictor.X_train),
            'test_samples': len(predictor.X_test),
            'model_performance': {
                'test_accuracy': predictor.results['XGBoost']['test_accuracy'],
                'precision': predictor.results['XGBoost']['precision'],
                'recall': predictor.results['XGBoost']['recall'],
                'f1_score': predictor.results['XGBoost']['f1_score'],
                'roc_auc': predictor.results['XGBoost']['roc_auc']
            }
        }

        feature_info_path = Path(f'analysis/xgboost_model{threshold_suffix}_info.json')
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"✓ Saved model metadata to: {feature_info_path}")

    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE - {gain_threshold}% GAIN TARGET")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - analysis/plots/feature_importance{threshold_suffix}.png")
    print(f"  - analysis/plots/roc_curves{threshold_suffix}.png")
    print(f"  - analysis/prediction_summary{threshold_suffix}.txt")
    print(f"  - analysis/plots/class_distribution{threshold_suffix}.png")
    print(f"  - analysis/plots/price_category_analysis{threshold_suffix}.png")
    print(f"  - analysis/plots/time_of_day_analysis{threshold_suffix}.png")
    print(f"  - analysis/xgboost_model{threshold_suffix}.json")
    print(f"  - analysis/xgboost_model{threshold_suffix}_info.json")
    print("\nNext steps:")
    print("  1. Review model performance (aim for F1 > 0.60)")
    print("  2. Check feature importance (which features drive success?)")
    print("  3. Review price category analysis (which price bins perform best?)")
    print("  4. Review time of day analysis (when are squeezes most profitable?)")
    print("  5. Validate on additional dates (out-of-sample testing)")
    print("  6. Consider alternative targets (30s gains, stop loss avoidance)")
    print("="*80)

    return predictor, results


def _generate_prediction_report(predictions_df: pd.DataFrame, model_trades: pd.DataFrame,
                                 threshold_suffix: str, gain_threshold: float,
                                 accuracy: float, precision: float, recall: float,
                                 f1: float, roc_auc: float, model_info: dict, date_range: str):
    """Generate markdown report for predictions."""
    from datetime import datetime

    report_file = Path(f'analysis/prediction_report{threshold_suffix}.md')

    with open(report_file, 'w') as f:
        f.write(f"# Prediction Report - {gain_threshold}% Target\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Date Range:** `{date_range}`\n\n")
        f.write(f"**Model:** `{model_info.get('gain_threshold', gain_threshold)}%` target\n\n")

        f.write("---\n\n")
        f.write("## Model Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Accuracy | {accuracy:.4f} |\n")
        f.write(f"| Precision | {precision:.4f} |\n")
        f.write(f"| Recall | {recall:.4f} |\n")
        f.write(f"| F1-Score | {f1:.4f} |\n")
        f.write(f"| ROC-AUC | {roc_auc:.4f} |\n\n")

        f.write("---\n\n")
        f.write("## Trading Performance\n\n")
        f.write("### Strategy: 1.5% Take-Profit + 1% Trailing Stop\n\n")

        if len(model_trades) > 0:
            # Use compounding for total profit
            model_total = ((1 + model_trades['realistic_profit']/100).prod() - 1) * 100
            model_avg = model_trades['realistic_profit'].mean()
            model_wins = model_trades[model_trades['realistic_profit'] > 0]
            model_losses = model_trades[model_trades['realistic_profit'] <= 0]
            win_rate = len(model_wins) / len(model_trades) * 100

            f.write("#### Model-Selected Trades\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Trades Taken | {len(model_trades)} / {len(predictions_df)} ({len(model_trades)/len(predictions_df)*100:.1f}%) |\n")
            f.write(f"| Total Profit (Compounded) | {model_total:.2f}% |\n")
            f.write(f"| Average Profit | {model_avg:.2f}% per trade |\n")
            f.write(f"| Win Rate | {win_rate:.1f}% ({len(model_wins)} wins, {len(model_losses)} losses) |\n")
            if len(model_wins) > 0:
                f.write(f"| Average Win | {model_wins['realistic_profit'].mean():.2f}% |\n")
            if len(model_losses) > 0:
                avg_loss = model_losses['realistic_profit'].mean()
                f.write(f"| Average Loss | {avg_loss:.2f}% |\n")
                if len(model_wins) > 0:
                    profit_factor = abs(model_wins['realistic_profit'].sum() / model_losses['realistic_profit'].sum())
                    f.write(f"| Profit Factor | {profit_factor:.2f} |\n")

            # Use compounding for total profit
            all_total = ((1 + predictions_df['realistic_profit']/100).prod() - 1) * 100
            all_avg = predictions_df['realistic_profit'].mean()

            f.write("\n#### Comparison: Model vs Take-All\n\n")
            f.write("| Strategy | Trades | Total Profit (Compounded) | Avg/Trade |\n")
            f.write("|----------|--------|---------------------------|----------|\n")
            f.write(f"| Model | {len(model_trades)} | {model_total:.2f}% | {model_avg:.2f}% |\n")
            f.write(f"| Take-All | {len(predictions_df)} | {all_total:.2f}% | {all_avg:.2f}% |\n")
            f.write(f"| **Difference** | {len(model_trades) - len(predictions_df)} | **{model_total - all_total:+.2f}%** | **{model_avg - all_avg:+.2f}%** |\n\n")

            f.write("---\n\n")
            f.write("## Analysis\n\n")

            if model_avg > all_avg:
                f.write(f"✅ **Model Edge:** The model achieves {model_avg - all_avg:+.2f}% better average profit per trade.\n\n")
            else:
                f.write(f"❌ **No Edge:** The model underperforms by {model_avg - all_avg:.2f}% per trade.\n\n")

            if model_total < all_total:
                missed_profit = all_total - model_total
                f.write(f"⚠️ **Opportunity Cost:** By being selective ({len(model_trades)}/{len(predictions_df)} trades), ")
                f.write(f"the model left **{missed_profit:.2f}%** profit on the table.\n\n")

            f.write("---\n\n")
            f.write("## Visualizations\n\n")
            f.write(f"![Prediction Analysis](plots/prediction_analysis{threshold_suffix}.png)\n\n")
            f.write(f"![Win/Loss Analysis](plots/win_loss_analysis{threshold_suffix}.png)\n\n")

    print(f"✓ Saved markdown report to: {report_file}")


def predict(model_path: str, start_date: str, end_date: str | None = None, gain_threshold: float | None = None,
           prediction_threshold: float = 0.5, start_time: str | None = None, end_time: str | None = None,
           min_price: float | None = None, max_price: float | None = None) -> pd.DataFrame:
    """
    Load a trained model and make predictions on new data.

    Args:
        model_path: Path to saved XGBoost model (e.g., 'analysis/xgboost_model_1.5pct.json')
        start_date: Start date for prediction (YYYY-MM-DD, e.g., '2025-12-17')
        end_date: End date for prediction (YYYY-MM-DD). If None, uses only start_date
        gain_threshold: Percentage gain threshold (if None, reads from model metadata)
        prediction_threshold: Probability threshold for binary classification (default: 0.5)
        start_time: Filter by start time of day (HH:MM format). Only alerts at or after this time will be included.
        end_time: Filter by end time of day (HH:MM format). Only alerts before this time will be included.
        min_price: Minimum squeeze entry price to include in predictions (optional)
        max_price: Maximum squeeze entry price to include in predictions (optional)

    Returns:
        DataFrame with predictions and actual outcomes
    """
    import json
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )

    # Auto-update features CSV if needed (before any other processing)
    features_csv = "analysis/squeeze_alerts_independent_features.csv"
    base_data_dir = "/home/wilsonb/dl/github.com/Z223I/alpaca/historical_data"
    ensure_features_updated(features_csv=features_csv, data_dir=base_data_dir)

    # Set end_date to start_date if not provided (single date mode)
    if end_date is None:
        end_date = start_date

    print("="*80)
    print("SQUEEZE ALERT OUTCOME PREDICTION - PREDICTION MODE")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Date Range: {start_date} to {end_date}")
    print("="*80)

    # Step 1: Load model metadata
    model_file = Path(model_path)
    info_path = Path(str(model_file).replace('.json', '_info.json'))

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    if not info_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {info_path}")

    with open(info_path, 'r') as f:
        model_info = json.load(f)

    feature_names = model_info['feature_names']
    if gain_threshold is None:
        gain_threshold = float(model_info['gain_threshold'])

    # Now gain_threshold is guaranteed to be a float
    threshold_str = f"{gain_threshold:.1f}".rstrip('0').rstrip('.') if gain_threshold % 1 else str(int(gain_threshold))
    threshold_suffix = f"_{threshold_str}pct"

    print(f"\nModel Info:")
    print(f"  Gain Threshold: {gain_threshold}%")
    print(f"  Features: {len(feature_names)}")
    print(f"  Training Samples: {model_info['train_samples']}")
    print(f"  Training Performance:")
    print(f"    - Accuracy: {model_info['model_performance']['test_accuracy']:.4f}")
    print(f"    - Precision: {model_info['model_performance']['precision']:.4f}")
    print(f"    - Recall: {model_info['model_performance']['recall']:.4f}")
    print(f"    - F1-Score: {model_info['model_performance']['f1_score']:.4f}")

    # Step 2: Load model and scaler
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model = xgb.XGBClassifier()
    model.load_model(model_file)
    print(f"✓ Loaded XGBoost model from: {model_file}")

    # Load scaler
    import pickle
    scaler_path = Path(str(model_file).replace('.json', '_scaler.pkl'))
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}. Please retrain the model.")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Loaded feature scaler from: {scaler_path}")

    # Step 3: Load test data
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)

    # Use base historical_data directory
    json_dir = base_data_dir
    predictor = SqueezeOutcomePredictor(json_dir, features_csv)

    # Extract outcomes for the specified date range
    print(f"Extracting data from {start_date} to {end_date}")
    outcomes_df = predictor.extract_outcomes(
        gain_threshold=gain_threshold,
        start_date=start_date,
        end_date=end_date
    )
    print(f"✓ Extracted {len(outcomes_df)} alerts from date range")

    # Merge with features
    df = predictor.merge_with_features(outcomes_df)
    print(f"✓ Loaded {len(df)} test samples")

    # Step 3.5: Filter by time of day if specified
    if start_time or end_time:
        print("\n" + "="*80)
        print("FILTERING BY TIME OF DAY")
        print("="*80)

        # Convert timestamp to datetime if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract time of day
        df['time_of_day'] = df['timestamp'].dt.time

        initial_count = len(df)

        # Apply start time filter
        if start_time:
            from datetime import datetime
            start_time_obj = datetime.strptime(start_time, '%H:%M').time()
            df = df[df['time_of_day'] >= start_time_obj]
            print(f"  After start time filter (>= {start_time}): {len(df)} samples")

        # Apply end time filter
        if end_time:
            from datetime import datetime
            end_time_obj = datetime.strptime(end_time, '%H:%M').time()
            df = df[df['time_of_day'] < end_time_obj]
            print(f"  After end time filter (< {end_time}): {len(df)} samples")

        # Clean up temporary column
        df = df.drop(columns=['time_of_day'])

        filtered_count = initial_count - len(df)
        print(f"✓ Filtered out {filtered_count} samples, {len(df)} remaining")

        if len(df) == 0:
            raise ValueError("No samples remaining after time filtering. Adjust your time range.")

    # Step 3.6: Filter by price if specified
    if min_price is not None or max_price is not None:
        print("\n" + "="*80)
        print("FILTERING BY PRICE")
        print("="*80)

        initial_count = len(df)

        # Apply minimum price filter
        if min_price is not None:
            df = df[df['squeeze_entry_price'] >= min_price]
            print(f"  After min price filter (>= ${min_price:.2f}): {len(df)} samples")

        # Apply maximum price filter
        if max_price is not None:
            df = df[df['squeeze_entry_price'] <= max_price]
            print(f"  After max price filter (<= ${max_price:.2f}): {len(df)} samples")

        filtered_count = initial_count - len(df)
        print(f"✓ Filtered out {filtered_count} samples, {len(df)} remaining")

        if len(df) == 0:
            raise ValueError("No samples remaining after price filtering. Adjust your price range.")

        # Reset index to avoid issues with non-contiguous indices
        df = df.reset_index(drop=True)

    # Step 4: Prepare features (same preprocessing as training)
    print("\n" + "="*80)
    print("PREPARING FEATURES")
    print("="*80)

    target_col = f'achieved_{int(gain_threshold)}pct'
    X, y = predictor.prepare_features(df, target_variable=target_col)

    # Verify feature alignment and create missing features
    if list(X.columns) != feature_names:
        print("⚠️  Warning: Feature names don't match exactly. Attempting to fix...")

        # Create any missing features (typically missing indicators)
        missing_features = set(feature_names) - set(X.columns)
        if missing_features:
            print(f"  Creating missing features: {missing_features}")
            for feat in missing_features:
                # Missing indicators should be 0 (no missing data)
                if feat.endswith('_missing'):
                    X[feat] = 0
                else:
                    raise ValueError(f"Unexpected missing feature: {feat}")

        # Reorder to match training
        try:
            X = X[feature_names]
            print("✓ Features aligned to match training")
        except KeyError as e:
            raise ValueError(f"Feature mismatch after alignment: {e}")

    print(f"✓ Prepared {len(X)} samples with {len(X.columns)} features")

    # Step 5: Scale features and make predictions
    print("\n" + "="*80)
    print("MAKING PREDICTIONS")
    print("="*80)

    # Apply scaler to features (critical - model was trained on scaled features!)
    X_scaled = scaler.transform(X)
    print(f"✓ Scaled features using training scaler")

    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    # Apply custom threshold for binary classification
    y_pred = (y_pred_proba >= prediction_threshold).astype(int)

    print(f"✓ Generated predictions for {len(y_pred)} samples")
    print(f"  Prediction threshold: {prediction_threshold:.2f} {'(default)' if prediction_threshold == 0.5 else '(custom)'}")
    print(f"  Predicted positives: {y_pred.sum()} ({y_pred.mean()*100:.1f}%)")
    print(f"  Actual positives: {y.sum()} ({y.mean()*100:.1f}%)")

    # Step 6: Evaluate predictions
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_pred_proba)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}  (of predicted wins, % actually won)")
    print(f"Recall:      {recall:.4f}  (of actual wins, % we predicted)")
    print(f"F1-Score:    {f1:.4f}  (balanced metric)")
    print(f"ROC-AUC:     {roc_auc:.4f}  (discrimination ability)")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                0       1")
    print(f"Actual    0   {cm[0,0]:4d}   {cm[0,1]:4d}  <- False Positives (bad trades)")
    print(f"          1   {cm[1,0]:4d}   {cm[1,1]:4d}  <- False Negatives (missed opps)")

    # Step 7: Create predictions DataFrame
    predictions_df = df[['symbol', 'timestamp']].copy()
    predictions_df['actual_outcome'] = y
    predictions_df['predicted_outcome'] = y_pred
    predictions_df['prediction_probability'] = y_pred_proba
    predictions_df['squeeze_entry_price'] = df['squeeze_entry_price']
    predictions_df['price_at_10min'] = df['price_at_10min']
    predictions_df['max_gain_percent'] = df['max_gain_percent']
    predictions_df['final_gain_percent'] = df['final_gain_percent']

    # Step 7a: Calculate realistic trading outcomes with interval-based stop loss
    print("\n" + "="*80)
    print(f"CALCULATING TRADING PROFITS ({gain_threshold}% Target + 1% Stop Loss)")
    print("Using interval-based chronological logic for realistic simulation")
    print("="*80)

    TRAILING_STOP_PCT = 2.0

    # Calculate realistic profit using _calculate_trade_outcome with interval data
    import json
    realistic_profits = []
    exit_reasons = []
    holding_times = []
    json_loaded_count = 0
    json_missing_count = 0

    print(f"Processing {len(df)} entries...")

    for idx in df.index:
        outcome_row = df.loc[idx]
        symbol = outcome_row.get('symbol', 'N/A')
        timestamp = outcome_row.get('timestamp', '')

        # Load full JSON to get interval data
        try:
            date_obj = pd.to_datetime(timestamp)
            date_str = date_obj.strftime('%Y-%m-%d')
            time_str = date_obj.strftime('%H%M%S')
            json_file = Path(json_dir) / date_str / 'squeeze_alerts_sent' / f'alert_{symbol}_{date_str}_{time_str}.json'

            if json_file.exists():
                json_loaded_count += 1
                with open(json_file, 'r') as f:
                    full_data = json.load(f)

                enhanced_row = outcome_row.copy()
                if 'outcome_tracking' in full_data:
                    enhanced_row['outcome_tracking'] = full_data['outcome_tracking']

                gain, reason, holding_time = predictor._calculate_trade_outcome(
                    enhanced_row, gain_threshold, TRAILING_STOP_PCT
                )

                realistic_profits.append(gain)
                exit_reasons.append(reason)
                holding_times.append(holding_time)
            else:
                json_missing_count += 1
                # Fallback - use simple calculation
                if json_missing_count <= 3:
                    print(f"WARNING: JSON not found for {symbol} {timestamp}")
                    print(f"  Looking for: {json_file}")
                gain, reason, holding_time = predictor._calculate_trade_outcome(
                    outcome_row, gain_threshold, TRAILING_STOP_PCT
                )
                realistic_profits.append(gain)
                exit_reasons.append(reason)
                holding_times.append(holding_time)
        except Exception as e:
            # Fallback on error
            print(f"ERROR processing {symbol} {timestamp}: {e}")
            gain, reason, holding_time = predictor._calculate_trade_outcome(
                outcome_row, gain_threshold, TRAILING_STOP_PCT
            )
            realistic_profits.append(gain)
            exit_reasons.append(reason)
            holding_times.append(holding_time)

    predictions_df['realistic_profit'] = realistic_profits
    predictions_df['exit_reason'] = exit_reasons
    predictions_df['holding_time_sec'] = holding_times

    print(f"\nJSON Loading Summary:")
    print(f"  JSON files loaded: {json_loaded_count}")
    print(f"  JSON files missing: {json_missing_count}")
    print(f"  Total processed: {len(realistic_profits)}")

    # Save predictions
    output_path = Path(f'analysis/predictions{threshold_suffix}.csv')
    predictions_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to: {output_path}")

    # Step 8: Summary report
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"\nModel trained on {model_info['train_samples']} samples")
    print(f"Tested on {len(y)} samples")
    print(f"\nPerformance Comparison:")
    print(f"  Training F1-Score: {model_info['model_performance']['f1_score']:.4f}")
    print(f"  Test F1-Score:     {f1:.4f}")
    print(f"  Difference:        {f1 - model_info['model_performance']['f1_score']:.4f}")

    if f1 >= model_info['model_performance']['f1_score'] - 0.05:
        print("\n✓ Model generalizes well to new data")
    else:
        print("\n⚠️  Warning: Performance degradation detected")

    print("="*80)

    # Step 9: Profit Analysis
    print("\n" + "="*80)
    print("TRADING PROFIT ANALYSIS")
    print("="*80)

    # Model-selected trades
    model_trades = predictions_df[predictions_df['predicted_outcome'] == 1].copy()
    num_model_trades = len(model_trades)

    if num_model_trades > 0:
        # Use compounding for total profit
        model_total_profit = ((1 + model_trades['realistic_profit']/100).prod() - 1) * 100
        model_avg_profit = model_trades['realistic_profit'].mean()
        model_wins = model_trades[model_trades['realistic_profit'] > 0]
        model_losses = model_trades[model_trades['realistic_profit'] <= 0]
        model_win_rate = len(model_wins) / num_model_trades if num_model_trades > 0 else 0

        print(f"\nModel-Selected Trades ({num_model_trades} trades):")
        print(f"  Total Profit:   {model_total_profit:.2f}% (compounded)")
        print(f"  Average Profit: {model_avg_profit:.2f}% per trade")
        print(f"  Win Rate:       {model_win_rate*100:.1f}%")
        if len(model_wins) > 0:
            print(f"  Average Win:    {model_wins['realistic_profit'].mean():.2f}%")
        if len(model_losses) > 0:
            print(f"  Average Loss:   {model_losses['realistic_profit'].mean():.2f}%")
            profit_factor = abs(model_wins['realistic_profit'].sum() / model_losses['realistic_profit'].sum()) if len(model_losses) > 0 else 0
            print(f"  Profit Factor:  {profit_factor:.2f}")

        # Compare to take-all
        # Use compounding for total profit
        all_total_profit = ((1 + predictions_df['realistic_profit']/100).prod() - 1) * 100
        all_avg_profit = predictions_df['realistic_profit'].mean()

        print(f"\nTake-All Strategy ({len(predictions_df)} trades):")
        print(f"  Total Profit:   {all_total_profit:.2f}% (compounded)")
        print(f"  Average Profit: {all_avg_profit:.2f}% per trade")

        print(f"\nModel Edge:")
        print(f"  Per Trade: {model_avg_profit - all_avg_profit:+.2f}%")
        print(f"  Total:     {model_total_profit - all_total_profit:+.2f}% ({(model_total_profit/all_total_profit - 1)*100:+.1f}%)")

        # Step 9a: Analyze holding period distribution
        print("\n" + "="*80)
        print("CALCULATING HOLDING PERIOD DISTRIBUTION")
        print("Using interval-based chronological logic (same as realistic_profit)")
        print("="*80)

        # Use _calculate_trade_outcome for holding period analysis (realistic stop loss logic)
        detailed_results = []
        json_loaded_count = 0
        json_missing_count = 0

        for idx in model_trades.index:
            trade_row = model_trades.loc[idx]
            symbol = trade_row.get('symbol', 'N/A')
            timestamp = trade_row.get('timestamp', '')

            # Load full JSON to get interval data and calculate realistic outcome
            try:
                # Find the JSON file for this alert
                # Format: historical_data/YYYY-MM-DD/squeeze_alerts_sent/alert_SYMBOL_YYYY-MM-DD_HHMMSS.json
                date_obj = pd.to_datetime(timestamp)
                date_str = date_obj.strftime('%Y-%m-%d')
                time_str = date_obj.strftime('%H%M%S')

                json_file = Path(json_dir) / date_str / 'squeeze_alerts_sent' / f'alert_{symbol}_{date_str}_{time_str}.json'

                if json_file.exists():
                    with open(json_file, 'r') as f:
                        full_data = json.load(f)

                    # Create enhanced row with outcome_tracking structure
                    outcome_row = df.loc[idx]
                    enhanced_row = outcome_row.copy()
                    if 'outcome_tracking' in full_data:
                        enhanced_row['outcome_tracking'] = full_data['outcome_tracking']

                    gain, reason, holding_time = predictor._calculate_trade_outcome(
                        enhanced_row, gain_threshold, TRAILING_STOP_PCT
                    )
                    json_loaded_count += 1
                else:
                    # Fallback if JSON not found
                    outcome_row = df.loc[idx]
                    gain, reason, holding_time = predictor._calculate_trade_outcome(
                        outcome_row, gain_threshold, TRAILING_STOP_PCT
                    )
                    json_missing_count += 1

            except Exception as e:
                # Fallback on error
                outcome_row = df.loc[idx]
                gain, reason, holding_time = predictor._calculate_trade_outcome(
                    outcome_row, gain_threshold, TRAILING_STOP_PCT
                )
                json_missing_count += 1

            detailed_results.append({
                'symbol': symbol,
                'gain': gain,
                'reason': reason,
                'holding_time_sec': holding_time
            })

        print(f"✓ Processed {json_loaded_count} JSON files with interval data")
        if json_missing_count > 0:
            print(f"⚠️  {json_missing_count} files not found or had errors (using fallback calculation)")

        # Create trading_results structure for holding period analysis
        trading_results_for_analysis = {
            'detailed_results': detailed_results,
            'num_trades': num_model_trades,
            'win_rate': model_win_rate,
            'avg_pnl': model_avg_profit
        }

        # Run holding period analysis
        predictor.analyze_holding_period_distribution(
            trading_results_for_analysis,
            gain_threshold=gain_threshold,
            output_path=f'analysis/plots/holding_period_distribution_predict{threshold_suffix}.png'
        )

        # Step 10: Generate plots
        generate_prediction_plots(predictions_df, model_trades, threshold_suffix, gain_threshold)

        # Step 10b: Generate aligned cumulative profit plot
        generate_aligned_cumulative_profit_plot(predictions_df, model_trades, threshold_suffix, gain_threshold)

        # Step 10c: Generate time-binned outcomes chart
        generate_time_binned_outcomes_chart(predictions_df, threshold_suffix, gain_threshold)

        # Step 10d: Generate price-binned outcomes chart
        generate_price_binned_outcomes_chart(predictions_df, threshold_suffix, gain_threshold)

        # Step 11: Generate markdown report
        date_range = f"{start_date} to {end_date}" if start_date != end_date else start_date
        _generate_prediction_report(
            predictions_df, model_trades, threshold_suffix, gain_threshold,
            accuracy, precision, recall, f1, roc_auc,
            model_info, date_range
        )
    else:
        print("\n⚠️  No trades selected by model - skipping profit analysis")

    return predictions_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Squeeze Alert Outcome Prediction - Train or Predict',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from project root):
  # Train model with specific threshold
  python analysis/predict_squeeze_outcomes.py train --threshold 1.5

  # Train models for all thresholds (default)
  python analysis/predict_squeeze_outcomes.py train

  # Make predictions using trained model (single date)
  python analysis/predict_squeeze_outcomes.py predict --model analysis/xgboost_model_1.5pct.json --start-date 2025-12-17 --end-date 2025-12-17

  # Predict on date range
  python analysis/predict_squeeze_outcomes.py predict --model analysis/xgboost_model_2pct.json --start-date 2025-12-17 --end-date 2025-12-18

  # Predict with time-of-day filtering (e.g., only alerts between 9:30 AM and 4:00 PM)
  python analysis/predict_squeeze_outcomes.py predict --model analysis/xgboost_model_2pct.json --start-date 2025-12-17 --start-time 09:30 --end-time 16:00
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or predict')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--threshold', type=float, default=None,
                             help='Gain threshold percentage (e.g., 1.5, 2.0). If not specified, trains all thresholds [1.5, 2.0, 2.5, 3.0]')
    train_parser.add_argument('--end-date', type=str, default='2025-12-16',
                             help='Training data cutoff date (YYYY-MM-DD). Data after this date is reserved for prediction. Default: 2025-12-16')
    train_parser.add_argument('--min-price', type=float, default=None,
                             help='Minimum squeeze entry price to include in training (optional)')
    train_parser.add_argument('--max-price', type=float, default=None,
                             help='Maximum squeeze entry price to include in training (optional)')

    # Predict mode
    predict_parser = subparsers.add_parser('predict', help='Make predictions using trained model')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='Path to trained model (e.g., analysis/xgboost_model_1.5pct.json)')
    predict_parser.add_argument('--start-date', type=str, required=True,
                               help='Start date for prediction (YYYY-MM-DD, e.g., 2025-12-17)')
    predict_parser.add_argument('--end-date', type=str, default=None,
                               help='End date for prediction (YYYY-MM-DD). If not specified, uses only start-date')
    predict_parser.add_argument('--threshold', type=float, default=None,
                               help='Gain threshold (optional, will use model\'s threshold if not specified)')
    predict_parser.add_argument('--prediction-threshold', type=float, default=0.5,
                               help='Prediction probability threshold (default: 0.5). Higher = more conservative (fewer trades, higher precision). Lower = more aggressive (more trades, higher recall)')
    predict_parser.add_argument('--start-time', type=str, default=None,
                               help='Filter predictions by start time of day (HH:MM format, e.g., 09:30). Only alerts at or after this time will be included.')
    predict_parser.add_argument('--end-time', type=str, default=None,
                               help='Filter predictions by end time of day (HH:MM format, e.g., 16:00). Only alerts before this time will be included.')
    predict_parser.add_argument('--min-price', type=float, default=None,
                               help='Minimum squeeze entry price to include in predictions (optional)')
    predict_parser.add_argument('--max-price', type=float, default=None,
                               help='Maximum squeeze entry price to include in predictions (optional)')

    args = parser.parse_args()

    # Default to train mode if no mode specified
    if args.mode is None:
        args.mode = 'train'
        args.threshold = None
        args.min_price = None
        args.max_price = None

    if args.mode == 'train':
        if args.threshold is not None:
            # Train single threshold
            print(f"\nTraining model with {args.threshold}% gain threshold\n")
            train(gain_threshold=args.threshold, end_date=args.end_date,
                  min_price=args.min_price, max_price=args.max_price)
        else:
            # Run multiple thresholds for comparison
            thresholds = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]
            predictors = {}
            results_all = {}

            for threshold in thresholds:
                print(f"\n{'='*80}")
                # Format threshold properly (1.5 -> "1.5%", 2.0 -> "2%")
                threshold_str = f"{threshold:.1f}".rstrip('0').rstrip('.') if threshold % 1 else str(int(threshold))
                print(f"RUNNING ANALYSIS: {threshold_str}% GAIN TARGET")
                print(f"{'='*80}\n")
                predictor, results = train(gain_threshold=threshold, end_date=args.end_date,
                                          min_price=args.min_price, max_price=args.max_price)
                predictors[threshold] = predictor
                results_all[threshold] = results

            # Print comprehensive comparison
            print(f"\n\n{'='*80}")
            print("FINAL COMPARISON: ALL GAIN TARGETS")
            print(f"{'='*80}\n")

            # Header - format thresholds properly
            header_cols = []
            for t in thresholds:
                t_str = f"{t:.1f}".rstrip('0').rstrip('.') if t % 1 else str(int(t))
                header_cols.append(f"{t_str}% Target")
            print(f"{'Metric':<20} " + "".join([f"{col:>15}" for col in header_cols]))
            print("-"*80)

            # Metrics comparison (Random Forest)
            metrics = ['test_accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            metric_names = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

            for metric, metric_name in zip(metrics, metric_names):
                print(f"{metric_name:<20} ", end="")
                for threshold in thresholds:
                    rf_result = results_all[threshold]['Random Forest']
                    value = rf_result[metric]
                    if value is not None:
                        print(f"{value:>14.4f} ", end="")
                    else:
                        print(f"{'N/A':>14} ", end="")
                print()

            # Find best model based on F1-score
            print("\n" + "="*80)
            print("RECOMMENDATION")
            print("="*80)

            best_threshold = thresholds[0]  # Initialize with first threshold
            best_f1 = results_all[thresholds[0]]['Random Forest']['f1_score']
            for threshold in thresholds[1:]:
                rf_result = results_all[threshold]['Random Forest']
                if rf_result['f1_score'] > best_f1:
                    best_f1 = rf_result['f1_score']
                    best_threshold = threshold

            # Format best threshold
            best_t_str = f"{best_threshold:.1f}".rstrip('0').rstrip('.') if best_threshold % 1 else str(int(best_threshold))
            print(f"✓ Best performing target: {best_t_str}% (F1-Score: {best_f1:.4f})")
            print(f"\nF1-Score ranking:")
            sorted_by_f1 = sorted(thresholds,
                                  key=lambda t: results_all[t]['Random Forest']['f1_score'],
                                  reverse=True)
            for i, threshold in enumerate(sorted_by_f1, 1):
                f1 = results_all[threshold]['Random Forest']['f1_score']
                t_str = f"{threshold:.1f}".rstrip('0').rstrip('.') if threshold % 1 else str(int(threshold))
                print(f"  {i}. {t_str}% target: F1 = {f1:.4f}")

            print("\n" + "="*80)

    elif args.mode == 'predict':
        # Prediction mode
        # If end_date not specified, use start_date (single date mode)
        end_date = args.end_date if args.end_date else args.start_date
        predictions_df = predict(
            model_path=args.model,
            start_date=args.start_date,
            end_date=end_date,
            gain_threshold=args.threshold,
            prediction_threshold=args.prediction_threshold,
            start_time=args.start_time,
            end_time=args.end_time,
            min_price=args.min_price,
            max_price=args.max_price
        )
        print(f"\n✓ Prediction complete. Results saved to analysis/predictions_*.csv")
