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
2. Extracts outcome data from specified test directory
3. Applies same preprocessing as training
4. Makes predictions and evaluates performance
5. Saves predictions with probabilities to CSV

USAGE EXAMPLES (run from project root):

  # Train models for all thresholds (1.5%, 2%, 2.5%, 3%)
  python analysis/predict_squeeze_outcomes.py train

  # Train model for specific threshold
  python analysis/predict_squeeze_outcomes.py train --threshold 1.5

  # Make predictions on a specific date's data
  python analysis/predict_squeeze_outcomes.py predict \
    --model analysis/xgboost_model_1.5pct.json \
    --test-dir historical_data/2025-12-15

  python analysis/predict_squeeze_outcomes.py predict --model analysis/xgboost_model_1.5pct.json --test-dir historical_data/2025-12-16
  python analysis/predict_squeeze_outcomes.py predict --model analysis/xgboost_model_3pct.json --test-dir historical_data/2025-12-16

  # Make predictions on all dates in directory
  python analysis/predict_squeeze_outcomes.py predict \\
    --model analysis/xgboost_model_2pct.json \\
    --test-dir historical_data

  # Make predictions with custom threshold (overrides model's default)
  python analysis/predict_squeeze_outcomes.py predict \\
    --model analysis/xgboost_model_1.5pct.json \\
    --test-dir historical_data/2025-12-15 \\
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
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
import warnings
warnings.filterwarnings('ignore')


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

    def extract_outcomes(self, gain_threshold: float = 5.0, start_date: str = "2025-12-12") -> pd.DataFrame:
        """
        Extract outcome_tracking data from JSON files across multiple date directories.

        Args:
            gain_threshold: Gain percentage threshold for success classification
            start_date: Starting date for directory scan (YYYY-MM-DD format)

        Returns:
            DataFrame with outcome metrics for each alert
        """
        print("="*80)
        print("STEP 1: EXTRACTING OUTCOME DATA FROM JSON FILES")
        print("="*80)
        print(f"Gain threshold: {gain_threshold}%")
        print(f"Scanning directories from {start_date} onwards")

        outcomes = []

        # Find all date directories >= start_date
        date_dirs = []
        for date_dir in self.json_dir.glob('????-??-??'):
            if date_dir.is_dir() and date_dir.name >= start_date:
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
        print("\n" + "="*80)
        print(f"STEP 8: FEATURE IMPORTANCE ANALYSIS - {model_name}")
        print("="*80)

        model = self.models.get(model_name)
        if model is None:
            print(f"Model '{model_name}' not found")
            return

        if not hasattr(model, 'feature_importances_'):
            print(f"Model '{model_name}' does not support feature_importances_")
            return

        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Print top features
        print("\nTop 10 Most Important Features:")
        print("="*60)
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:35s} {row['importance']:.4f}")

        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        output_file = Path(output_path)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved feature importance plot to: {output_file}")
        plt.close()

        return importance_df

    def plot_roc_curves(self, output_path: str = 'analysis/plots/roc_curves.png'):
        """
        Plot ROC curves for all models.

        Args:
            output_path: Where to save plot
        """
        print("\n" + "="*80)
        print("PLOTTING ROC CURVES")
        print("="*80)

        plt.figure(figsize=(10, 8))

        for name, result in self.results.items():
            if result['y_test_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_test_proba'])
                auc = result['roc_auc']
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_file = Path(output_path)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves to: {output_file}")
        plt.close()

    def _calculate_trade_outcome(self, outcome: pd.Series, gain_threshold: float,
                                  stop_loss_pct: float) -> Tuple[float, str]:
        """
        Calculate trade outcome using chronological interval price range logic.

        NEW: Uses interval low/high with timestamps to determine which event occurred first:
        - If high timestamp < low timestamp: price went up first
          → Check if high reached gain threshold (potential win)
          → If not, check if low hit stop loss (loss)
        - If low timestamp < high timestamp: price went down first
          → Check if low hit stop loss (loss)
          → If not, check if high reached gain threshold (win)

        Falls back to simple max_gain_percent if interval data not available.

        Args:
            outcome: Row from test DataFrame with outcome_tracking data
            gain_threshold: Target gain percentage
            stop_loss_pct: Stop loss percentage (positive number, e.g., 2.0 for -2%)

        Returns:
            Tuple of (gain_percent, reason_string)
        """
        from datetime import datetime

        # Try to load interval data from JSON if available
        # The outcome row may have nested JSON structure
        intervals = None

        # Check if we have direct access to interval data
        # This would be the case if we loaded the full JSON structure
        if hasattr(outcome, 'get'):
            # Try different possible locations for interval data
            if 'outcome_tracking' in outcome and outcome['outcome_tracking'] is not None:
                if isinstance(outcome['outcome_tracking'], dict):
                    intervals = outcome['outcome_tracking'].get('intervals', {})

        # Strategy: Look through intervals chronologically and apply trading logic
        if intervals and len(intervals) > 0:
            # Process intervals in chronological order
            sorted_intervals = sorted(intervals.items(), key=lambda x: int(x[0]))

            for interval_sec_str, interval_data in sorted_intervals:
                # Check if we have the new interval_low and interval_high fields
                has_range_data = ('interval_low' in interval_data and
                                 'interval_high' in interval_data and
                                 'interval_low_timestamp' in interval_data and
                                 'interval_high_timestamp' in interval_data)

                if has_range_data:
                    # NEW LOGIC: Use chronological high/low testing
                    low_gain = interval_data['interval_low_gain_percent']
                    high_gain = interval_data['interval_high_gain_percent']
                    low_timestamp = interval_data['interval_low_timestamp']
                    high_timestamp = interval_data['interval_high_timestamp']

                    # Determine which event happened first
                    if high_timestamp < low_timestamp:
                        # Price went UP first, then DOWN
                        # Check if we hit target before stop loss
                        if high_gain >= gain_threshold:
                            # WIN: Hit target before stop loss (exact limit order)
                            return (gain_threshold, f'target_hit_at_{interval_sec_str}s')

                        # Check if stop loss was hit
                        if low_gain <= -stop_loss_pct:
                            # LOSS: Hit stop loss
                            return (-stop_loss_pct, f'stop_loss_at_{interval_sec_str}s')

                    else:
                        # Price went DOWN first, then UP (or at same time)
                        # Check if we hit stop loss before target
                        if low_gain <= -stop_loss_pct:
                            # LOSS: Hit stop loss before target
                            return (-stop_loss_pct, f'stop_loss_at_{interval_sec_str}s')

                        # Check if target was hit
                        if high_gain >= gain_threshold:
                            # WIN: Hit target (exact limit order)
                            return (gain_threshold, f'target_hit_at_{interval_sec_str}s')

                else:
                    # FALLBACK: Use snapshot price if no range data
                    if 'gain_percent' in interval_data:
                        gain = interval_data['gain_percent']

                        # Check if hit target (exact limit order)
                        if gain >= gain_threshold:
                            return (gain_threshold, f'snapshot_target_at_{interval_sec_str}s')

                        # Check if hit stop loss
                        if gain <= -stop_loss_pct:
                            return (-stop_loss_pct, f'snapshot_stop_at_{interval_sec_str}s')

            # If we went through all intervals without hitting target or stop loss
            # Use the final outcome
            final_gain = outcome.get('final_gain_percent', 0)
            if final_gain >= gain_threshold:
                return (gain_threshold, 'final_target')
            elif final_gain <= -stop_loss_pct:
                return (-stop_loss_pct, 'final_stop_loss')
            else:
                return (final_gain, 'final_no_target')

        # FALLBACK: Use max_gain_percent if no interval data
        # This maintains backward compatibility
        max_gain = outcome.get('max_gain_percent', 0)

        if max_gain >= gain_threshold:
            return (gain_threshold, 'legacy_target_hit')
        elif max_gain <= -stop_loss_pct:
            return (-stop_loss_pct, 'legacy_stop_loss')
        else:
            # Didn't hit target or stop loss
            final_gain = outcome.get('final_gain_percent', max_gain)
            return (max(final_gain, -stop_loss_pct), 'legacy_no_target')

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
        - If actual failure: loss = -stop_loss_pct (default -2%)

        Args:
            df: Original DataFrame with outcome data
            model_name: Which model's predictions to use
            gain_threshold: Percentage gain threshold for success (default 5.0%)
            stop_loss_pct: Stop loss percentage (default 2.0%)

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
        pnl_list = []
        detailed_results = []  # Track detailed outcome for each trade

        for idx, take_trade in zip(test_outcomes.index, trades_taken):
            if not take_trade:
                continue  # Skip this alert

            outcome = test_outcomes.loc[idx]

            # Try to use interval price range data (NEW feature)
            # This gives us chronological high/low information
            gain, outcome_reason = self._calculate_trade_outcome(
                outcome, gain_threshold, stop_loss_pct
            )

            pnl_list.append(gain)
            detailed_results.append({
                'symbol': outcome.get('symbol', 'N/A'),
                'gain': gain,
                'reason': outcome_reason
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

            # Use same calculation logic
            gain, _ = self._calculate_trade_outcome(
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
            'improvement': avg_pnl - all_trades_avg
        }

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
                                output_path: str = 'analysis/plots/class_distribution_5pct.png'):
        """
        Create pie chart showing class distribution.

        Args:
            gain_threshold: Gain percentage threshold for success
            output_path: Where to save the plot
        """
        stats = self.analyze_class_distribution(gain_threshold)

        if not stats:
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Data for pie chart
        labels = ['Success', 'Failure']
        sizes = [stats['successes'], stats['failures']]
        colors = ['#2ecc71', '#e74c3c']  # Green for success, red for failure
        explode = (0.05, 0)  # Slightly separate the success slice

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90,
            textprops={'fontsize': 14, 'weight': 'bold'}
        )

        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(16)

        # Add title with statistics
        ax.set_title(
            f'Class Distribution - {int(gain_threshold)}% Target\n\n'
            f'Total Alerts: {stats["total"]:,}\n'
            f'Success: {stats["successes"]:,} ({stats["success_pct"]:.1f}%) | '
            f'Failure: {stats["failures"]:,} ({stats["failure_pct"]:.1f}%)\n'
            f'Imbalance Ratio: 1 : {stats["imbalance_ratio"]:.2f}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')

        # Add assessment text
        if 45 <= stats['success_pct'] <= 55:
            assessment = "✓ Excellent Balance"
            assessment_color = 'green'
        elif 40 <= stats['success_pct'] <= 60:
            assessment = "~ Good Balance"
            assessment_color = 'orange'
        elif 35 <= stats['success_pct'] <= 65:
            assessment = "⚠ Moderate Imbalance"
            assessment_color = 'darkorange'
        else:
            assessment = "❌ Severe Imbalance"
            assessment_color = 'red'

        plt.figtext(
            0.5, 0.02,
            assessment,
            ha='center',
            fontsize=12,
            weight='bold',
            color=assessment_color
        )

        plt.tight_layout()

        # Save plot
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved class distribution plot to: {output_file}")

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
        self._plot_price_category_analysis(grouped, gain_threshold, output_path)

        return grouped

    def _plot_price_category_analysis(self, grouped: pd.DataFrame, gain_threshold: float,
                                       output_path: str):
        """
        Create visualization of price category performance.

        Args:
            grouped: DataFrame with metrics by price category
            gain_threshold: Gain percentage threshold
            output_path: Where to save plot
        """
        # Sort by price category order for better visualization
        price_order = ['<$2', '$2-5', '$5-10', '$10-20', '$20-40', '>$40']
        grouped_sorted = grouped.reindex([p for p in price_order if p in grouped.index])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Squeeze Performance by Price Category - {gain_threshold}% Target',
                     fontsize=16, fontweight='bold')

        # 1. Win Rate comparison
        ax1 = axes[0, 0]
        x_pos = range(len(grouped_sorted))
        win_col = f'Win_Rate_{int(gain_threshold)}pct'
        bars = ax1.bar(x_pos, grouped_sorted[win_col] * 100, color='green', alpha=0.7)
        ax1.axhline(y=grouped_sorted[win_col].mean() * 100, color='red', linestyle='--',
                   label=f'Average ({grouped_sorted[win_col].mean()*100:.1f}%)')
        ax1.set_xlabel('Price Category', fontweight='bold')
        ax1.set_ylabel(f'{int(gain_threshold)}% Win Rate (%)', fontweight='bold')
        ax1.set_title(f'{int(gain_threshold)}% Win Rate by Price Category', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(grouped_sorted.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, grouped_sorted[win_col] * 100)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        # 2. Stop Loss Rate
        ax2 = axes[0, 1]
        bars = ax2.bar(x_pos, grouped_sorted['Stop_Loss_Rate'] * 100, color='red', alpha=0.7)
        ax2.axhline(y=grouped_sorted['Stop_Loss_Rate'].mean() * 100, color='blue', linestyle='--',
                   label=f'Average ({grouped_sorted["Stop_Loss_Rate"].mean()*100:.1f}%)')
        ax2.set_xlabel('Price Category', fontweight='bold')
        ax2.set_ylabel('Stop Loss Rate (%)', fontweight='bold')
        ax2.set_title('Stop Loss Rate by Price Category', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(grouped_sorted.index, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, grouped_sorted['Stop_Loss_Rate'] * 100)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        # 3. Profitability Score (Win Rate - Stop Loss Rate)
        ax3 = axes[1, 0]
        colors = ['green' if x > 0 else 'red' for x in grouped_sorted['Profitability_Score']]
        bars = ax3.bar(x_pos, grouped_sorted['Profitability_Score'] * 100, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_xlabel('Price Category', fontweight='bold')
        ax3.set_ylabel('Profitability Score (%)', fontweight='bold')
        ax3.set_title('Profitability Score (Win Rate - Stop Loss Rate)', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(grouped_sorted.index, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, grouped_sorted['Profitability_Score'] * 100)):
            y_pos = bar.get_height() + 1 if val > 0 else bar.get_height() - 3
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

        # 4. Average Final Gain
        ax4 = axes[1, 1]
        colors = ['green' if x > 0 else 'red' for x in grouped_sorted['Avg_Final_Gain']]
        bars = ax4.bar(x_pos, grouped_sorted['Avg_Final_Gain'], color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_xlabel('Price Category', fontweight='bold')
        ax4.set_ylabel('Average Final Gain (%)', fontweight='bold')
        ax4.set_title('Average Final Gain by Price Category', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(grouped_sorted.index, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, grouped_sorted['Avg_Final_Gain'])):
            y_pos = bar.get_height() + 0.3 if val > 0 else bar.get_height() - 0.5
            ax4.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

        plt.tight_layout()

        output_file = Path(output_path)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved price category analysis plot to: {output_file}")
        plt.close()

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
        self._plot_time_of_day_analysis(grouped, gain_threshold, output_path)

        return grouped

    def _plot_time_of_day_analysis(self, grouped: pd.DataFrame, gain_threshold: float,
                                    output_path: str):
        """
        Create visualization of time of day performance.

        Args:
            grouped: DataFrame with metrics by time bin
            gain_threshold: Gain percentage threshold
            output_path: Where to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Squeeze Performance by Time of Day - {gain_threshold}% Target',
                     fontsize=16, fontweight='bold')

        x_pos = range(len(grouped))
        time_labels = grouped.index.tolist()

        # 1. Win Rate by time
        ax1 = axes[0, 0]
        win_col = f'Win_Rate_{int(gain_threshold)}pct'
        bars = ax1.bar(x_pos, grouped[win_col] * 100, color='green', alpha=0.7)
        ax1.axhline(y=grouped[win_col].mean() * 100, color='red', linestyle='--',
                   label=f'Average ({grouped[win_col].mean()*100:.1f}%)')

        # Highlight first hour (9:30-10:30)
        morning_bins = ['09:30-10:00', '10:00-10:30']
        for i, time_bin in enumerate(time_labels):
            if time_bin in morning_bins:
                bars[i].set_color('darkgreen')
                bars[i].set_alpha(0.9)

        ax1.set_xlabel('Time of Day', fontweight='bold')
        ax1.set_ylabel(f'{int(gain_threshold)}% Win Rate (%)', fontweight='bold')
        ax1.set_title(f'{int(gain_threshold)}% Win Rate by Time of Day (First hour highlighted)', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. Stop Loss Rate
        ax2 = axes[0, 1]
        bars = ax2.bar(x_pos, grouped['Stop_Loss_Rate'] * 100, color='red', alpha=0.7)
        ax2.axhline(y=grouped['Stop_Loss_Rate'].mean() * 100, color='blue', linestyle='--',
                   label=f'Average ({grouped["Stop_Loss_Rate"].mean()*100:.1f}%)')
        ax2.set_xlabel('Time of Day', fontweight='bold')
        ax2.set_ylabel('Stop Loss Rate (%)', fontweight='bold')
        ax2.set_title('Stop Loss Rate by Time of Day', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3. Profitability Score
        ax3 = axes[1, 0]
        colors = ['green' if x > 0 else 'red' for x in grouped['Profitability_Score']]
        bars = ax3.bar(x_pos, grouped['Profitability_Score'] * 100, color=colors, alpha=0.7)

        # Highlight first hour
        for i, time_bin in enumerate(time_labels):
            if time_bin in morning_bins:
                if grouped.iloc[i]['Profitability_Score'] > 0:
                    bars[i].set_color('darkgreen')
                    bars[i].set_alpha(0.9)

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_xlabel('Time of Day', fontweight='bold')
        ax3.set_ylabel('Profitability Score (%)', fontweight='bold')
        ax3.set_title('Profitability Score by Time of Day (First hour highlighted)', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
        ax3.grid(axis='y', alpha=0.3)

        # 4. Alert Count (volume throughout day)
        ax4 = axes[1, 1]
        bars = ax4.bar(x_pos, grouped['Count'], color='blue', alpha=0.7)
        ax4.set_xlabel('Time of Day', fontweight='bold')
        ax4.set_ylabel('Number of Alerts', fontweight='bold')
        ax4.set_title('Alert Volume by Time of Day', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=8)
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels on volume bars
        for i, (bar, val) in enumerate(zip(bars, grouped['Count'])):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(grouped['Count'])*0.01,
                    f'{int(val)}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        output_file = Path(output_path)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved time of day analysis plot to: {output_file}")
        plt.close()


def train(gain_threshold: float = 5.0):
    """
    Train models and generate analysis for a given gain threshold.

    Args:
        gain_threshold: Percentage gain threshold for success classification (default: 5.0)
    """

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

    # Configuration
    # Use base historical_data directory to scan all date directories from 2025-12-12 onwards
    json_dir = "/home/wilsonb/dl/github.com/Z223I/alpaca/historical_data"
    features_csv = "analysis/squeeze_alerts_independent_features.csv"

    # Initialize predictor
    predictor = SqueezeOutcomePredictor(json_dir, features_csv)

    # Step 1: Extract outcomes from all directories starting 2025-12-12
    outcomes_df = predictor.extract_outcomes(gain_threshold=gain_threshold, start_date="2025-12-12")

    # Step 2: Merge with features
    df = predictor.merge_with_features(outcomes_df)
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

    # Step 6: Train models
    models = predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Step 7: Evaluate models
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
                                                  stop_loss_pct=2.0)

    # Step 11: Generate summary report
    predictor.generate_summary_report(output_path=f'analysis/prediction_summary{threshold_suffix}.txt')

    # Step 11.5: Class distribution analysis
    predictor.plot_class_distribution(
        gain_threshold=gain_threshold,
        output_path=f'analysis/plots/class_distribution{threshold_suffix}.png'
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


def _generate_prediction_plots(predictions_df: pd.DataFrame, model_trades: pd.DataFrame,
                                threshold_suffix: str, gain_threshold: float):
    """Generate plots for prediction analysis."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = Path('analysis/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Plot 1: Profit Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Prediction Analysis - {gain_threshold}% Target', fontsize=16, fontweight='bold')

    # 1a: Model trades profit distribution
    ax1 = axes[0, 0]
    if len(model_trades) > 0:
        ax1.hist(model_trades['realistic_profit'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=model_trades['realistic_profit'].mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Avg: {model_trades["realistic_profit"].mean():.2f}%')
        ax1.set_xlabel('Profit %', fontweight='bold')
        ax1.set_ylabel('Number of Trades', fontweight='bold')
        ax1.set_title(f'Model Trades Profit Distribution ({len(model_trades)} trades)', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

    # 1b: All trades profit distribution
    ax2 = axes[0, 1]
    ax2.hist(predictions_df['realistic_profit'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax2.axvline(x=predictions_df['realistic_profit'].mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Avg: {predictions_df["realistic_profit"].mean():.2f}%')
    ax2.set_xlabel('Profit %', fontweight='bold')
    ax2.set_ylabel('Number of Trades', fontweight='bold')
    ax2.set_title(f'All Opportunities Profit Distribution ({len(predictions_df)} trades)', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 1c: Cumulative profit comparison
    ax3 = axes[1, 0]
    if len(model_trades) > 0:
        model_cumulative = model_trades['realistic_profit'].cumsum()
        all_cumulative = predictions_df['realistic_profit'].cumsum()
        ax3.plot(model_cumulative.values, label=f'Model ({len(model_trades)} trades)', linewidth=2, color='green')
        ax3.plot(all_cumulative.values, label=f'Take-All ({len(predictions_df)} trades)', linewidth=2, color='orange')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax3.set_xlabel('Trade Number', fontweight='bold')
        ax3.set_ylabel('Cumulative Profit %', fontweight='bold')
        ax3.set_title('Cumulative Profit Comparison', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

    # 1d: Probability distribution of selected vs missed trades
    ax4 = axes[1, 1]
    selected = predictions_df[predictions_df['predicted_outcome'] == 1]['prediction_probability']
    missed = predictions_df[predictions_df['predicted_outcome'] == 0]['prediction_probability']
    ax4.hist([selected, missed], bins=30, alpha=0.7, label=['Selected', 'Missed'], color=['green', 'red'])
    ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax4.set_xlabel('Prediction Probability', fontweight='bold')
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('Probability Distribution: Selected vs Missed', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plot_file = plots_dir / f'prediction_analysis{threshold_suffix}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved analysis plots to: {plot_file}")
    plt.close()

    # Plot 2: Win/Loss Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Win/Loss Analysis - {gain_threshold}% Target', fontsize=16, fontweight='bold')

    # 2a: Model trades pie chart
    ax1 = axes[0]
    if len(model_trades) > 0:
        model_wins = len(model_trades[model_trades['realistic_profit'] > 0])
        model_losses = len(model_trades[model_trades['realistic_profit'] <= 0])
        ax1.pie([model_wins, model_losses], labels=['Wins', 'Losses'],
               autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
        ax1.set_title(f'Model Trades\n({model_wins} wins, {model_losses} losses)', fontweight='bold')

    # 2b: Comparison bar chart
    ax2 = axes[1]
    if len(model_trades) > 0:
        model_total = model_trades['realistic_profit'].sum()
        all_total = predictions_df['realistic_profit'].sum()
        model_avg = model_trades['realistic_profit'].mean()
        all_avg = predictions_df['realistic_profit'].mean()

        x = np.arange(2)
        width = 0.35
        ax2.bar(x - width/2, [model_total, model_avg], width, label='Model', color='green', alpha=0.7)
        ax2.bar(x + width/2, [all_total, all_avg], width, label='Take-All', color='orange', alpha=0.7)
        ax2.set_ylabel('Profit %', fontweight='bold')
        ax2.set_title('Profit Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Total Profit', 'Avg Per Trade'])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bars1, bars2) in enumerate(zip(ax2.containers[0], ax2.containers[1])):
            ax2.text(bars1.get_x() + bars1.get_width()/2, bars1.get_height(),
                    f'{bars1.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
            ax2.text(bars2.get_x() + bars2.get_width()/2, bars2.get_height(),
                    f'{bars2.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_file = plots_dir / f'win_loss_analysis{threshold_suffix}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved win/loss plots to: {plot_file}")
    plt.close()


def _generate_prediction_report(predictions_df: pd.DataFrame, model_trades: pd.DataFrame,
                                 threshold_suffix: str, gain_threshold: float,
                                 accuracy: float, precision: float, recall: float,
                                 f1: float, roc_auc: float, model_info: dict, test_dir: str):
    """Generate markdown report for predictions."""
    from datetime import datetime

    report_file = Path(f'analysis/prediction_report{threshold_suffix}.md')

    with open(report_file, 'w') as f:
        f.write(f"# Prediction Report - {gain_threshold}% Target\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Test Directory:** `{test_dir}`\n\n")
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
        f.write("### Strategy: 1.5% Take-Profit + 2% Trailing Stop\n\n")

        if len(model_trades) > 0:
            model_total = model_trades['realistic_profit'].sum()
            model_avg = model_trades['realistic_profit'].mean()
            model_wins = model_trades[model_trades['realistic_profit'] > 0]
            model_losses = model_trades[model_trades['realistic_profit'] <= 0]
            win_rate = len(model_wins) / len(model_trades) * 100

            f.write("#### Model-Selected Trades\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Trades Taken | {len(model_trades)} / {len(predictions_df)} ({len(model_trades)/len(predictions_df)*100:.1f}%) |\n")
            f.write(f"| Total Profit | {model_total:.2f}% |\n")
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

            all_total = predictions_df['realistic_profit'].sum()
            all_avg = predictions_df['realistic_profit'].mean()

            f.write("\n#### Comparison: Model vs Take-All\n\n")
            f.write("| Strategy | Trades | Total Profit | Avg/Trade |\n")
            f.write("|----------|--------|--------------|----------|\n")
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


def predict(model_path: str, test_dir: str, gain_threshold: float | None = None) -> pd.DataFrame:
    """
    Load a trained model and make predictions on new data.

    Args:
        model_path: Path to saved XGBoost model (e.g., 'analysis/xgboost_model_1.5pct.json')
        test_dir: Directory containing test data (e.g., 'historical_data/2025-12-15')
        gain_threshold: Percentage gain threshold (if None, reads from model metadata)

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

    print("="*80)
    print("SQUEEZE ALERT OUTCOME PREDICTION - PREDICTION MODE")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Test directory: {test_dir}")
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

    # Create predictor instance
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Determine if test_dir is a date directory or base directory
    single_date_mode = False
    if (test_path / 'squeeze_alerts_sent').exists():
        # Single date directory
        json_dir = test_path.parent
        start_date = test_path.name
        single_date_mode = True
        print(f"Single date mode: extracting only from {start_date}")
    else:
        # Base directory, use all dates
        json_dir = test_path
        start_date = "2025-01-01"  # Default to all dates
        print(f"Multi-date mode: extracting from {start_date} onwards")

    features_csv = "analysis/squeeze_alerts_independent_features.csv"
    predictor = SqueezeOutcomePredictor(str(json_dir), features_csv)

    # Extract outcomes
    outcomes_df = predictor.extract_outcomes(gain_threshold=gain_threshold, start_date=start_date)

    # If single date mode, filter to only that specific date
    if single_date_mode:
        # Filter outcomes to only include the specific date
        outcomes_df['date'] = pd.to_datetime(outcomes_df['timestamp']).dt.date
        target_date = pd.to_datetime(start_date).date()
        outcomes_df = outcomes_df[outcomes_df['date'] == target_date].copy()
        outcomes_df = outcomes_df.drop('date', axis=1)
        print(f"✓ Filtered to {len(outcomes_df)} alerts from {start_date} only")

    # Merge with features
    df = predictor.merge_with_features(outcomes_df)
    print(f"✓ Loaded {len(df)} test samples")

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

    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    print(f"✓ Generated predictions for {len(y_pred)} samples")
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
    predictions_df['max_gain_percent'] = df['max_gain_percent']
    predictions_df['final_gain_percent'] = df['final_gain_percent']

    # Step 7a: Calculate realistic trading outcomes with trailing stop
    print("\n" + "="*80)
    print("CALCULATING TRADING PROFITS (1.5% Target + 2% Trailing Stop)")
    print("="*80)

    TRAILING_STOP_PCT = 2.0

    def calculate_trailing_stop_outcome(row):
        """Calculate trade outcome with trailing stop logic."""
        max_gain = row['max_gain_percent']
        final_gain = row['final_gain_percent']

        # Hit target?
        if max_gain >= gain_threshold:
            return gain_threshold

        # Trailing stop: max_gain - 2%, but never below -2%
        trailing_stop_level = max_gain - TRAILING_STOP_PCT
        effective_stop = max(trailing_stop_level, -TRAILING_STOP_PCT)

        # Got stopped out?
        if final_gain < effective_stop:
            return effective_stop
        else:
            return final_gain

    predictions_df['realistic_profit'] = predictions_df.apply(calculate_trailing_stop_outcome, axis=1)

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
        model_total_profit = model_trades['realistic_profit'].sum()
        model_avg_profit = model_trades['realistic_profit'].mean()
        model_wins = model_trades[model_trades['realistic_profit'] > 0]
        model_losses = model_trades[model_trades['realistic_profit'] <= 0]
        model_win_rate = len(model_wins) / num_model_trades if num_model_trades > 0 else 0

        print(f"\nModel-Selected Trades ({num_model_trades} trades):")
        print(f"  Total Profit:   {model_total_profit:.2f}%")
        print(f"  Average Profit: {model_avg_profit:.2f}% per trade")
        print(f"  Win Rate:       {model_win_rate*100:.1f}%")
        if len(model_wins) > 0:
            print(f"  Average Win:    {model_wins['realistic_profit'].mean():.2f}%")
        if len(model_losses) > 0:
            print(f"  Average Loss:   {model_losses['realistic_profit'].mean():.2f}%")
            profit_factor = abs(model_wins['realistic_profit'].sum() / model_losses['realistic_profit'].sum()) if len(model_losses) > 0 else 0
            print(f"  Profit Factor:  {profit_factor:.2f}")

        # Compare to take-all
        all_total_profit = predictions_df['realistic_profit'].sum()
        all_avg_profit = predictions_df['realistic_profit'].mean()

        print(f"\nTake-All Strategy ({len(predictions_df)} trades):")
        print(f"  Total Profit:   {all_total_profit:.2f}%")
        print(f"  Average Profit: {all_avg_profit:.2f}% per trade")

        print(f"\nModel Edge:")
        print(f"  Per Trade: {model_avg_profit - all_avg_profit:+.2f}%")
        print(f"  Total:     {model_total_profit - all_total_profit:+.2f}% ({(model_total_profit/all_total_profit - 1)*100:+.1f}%)")

        # Step 10: Generate plots
        _generate_prediction_plots(predictions_df, model_trades, threshold_suffix, gain_threshold)

        # Step 11: Generate markdown report
        _generate_prediction_report(
            predictions_df, model_trades, threshold_suffix, gain_threshold,
            accuracy, precision, recall, f1, roc_auc,
            model_info, test_dir
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

  # Make predictions using trained model
  python analysis/predict_squeeze_outcomes.py predict --model analysis/xgboost_model_1.5pct.json --test-dir historical_data/2025-12-15

  # Predict on all dates in directory
  python analysis/predict_squeeze_outcomes.py predict --model analysis/xgboost_model_2pct.json --test-dir historical_data
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or predict')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--threshold', type=float, default=None,
                             help='Gain threshold percentage (e.g., 1.5, 2.0). If not specified, trains all thresholds [1.5, 2.0, 2.5, 3.0]')

    # Predict mode
    predict_parser = subparsers.add_parser('predict', help='Make predictions using trained model')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='Path to trained model (e.g., analysis/xgboost_model_1.5pct.json)')
    predict_parser.add_argument('--test-dir', type=str, required=True,
                               help='Directory containing test data (e.g., historical_data/2025-12-15)')
    predict_parser.add_argument('--threshold', type=float, default=None,
                               help='Gain threshold (optional, will use model\'s threshold if not specified)')

    args = parser.parse_args()

    # Default to train mode if no mode specified
    if args.mode is None:
        args.mode = 'train'
        args.threshold = None

    if args.mode == 'train':
        if args.threshold is not None:
            # Train single threshold
            print(f"\nTraining model with {args.threshold}% gain threshold\n")
            train(gain_threshold=args.threshold)
        else:
            # Run multiple thresholds for comparison
            thresholds = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
            predictors = {}
            results_all = {}

            for threshold in thresholds:
                print(f"\n{'='*80}")
                # Format threshold properly (1.5 -> "1.5%", 2.0 -> "2%")
                threshold_str = f"{threshold:.1f}".rstrip('0').rstrip('.') if threshold % 1 else str(int(threshold))
                print(f"RUNNING ANALYSIS: {threshold_str}% GAIN TARGET")
                print(f"{'='*80}\n")
                predictor, results = train(gain_threshold=threshold)
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
        predictions_df = predict(
            model_path=args.model,
            test_dir=args.test_dir,
            gain_threshold=args.threshold
        )
        print(f"\n✓ Prediction complete. Results saved to analysis/predictions_*.csv")
