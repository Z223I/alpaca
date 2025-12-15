#!/usr/bin/env python3
"""
Squeeze Alert Outcome Prediction

Predicts price action following squeeze alerts using statistically independent features.

This script:
1. Extracts outcome_tracking data from JSON files
2. Merges with independent features
3. Trains multiple ML models (Logistic Regression, Random Forest, XGBoost)
4. Evaluates trading-specific metrics (precision, recall, ROI simulation)
5. Generates feature importance analysis

NOTE: Some features have missing data in current dataset:
- ema_spread_pct: ~10% missing (price-normalized EMA spread)
- macd_histogram: ~15% missing (will be 100% available when switched to daily MACD)

For now, we handle missing data via imputation and indicator variables.

Author: Predictive Analytics Module
Date: 2025-12-12
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

        # Create indicator variables for features with significant missing data
        if 'ema_spread_pct' in X.columns and X['ema_spread_pct'].isnull().sum() > 0:
            X_imputed['ema_spread_pct_missing'] = X['ema_spread_pct'].isnull().astype(int)
            print(f"✓ Created ema_spread_pct_missing indicator")

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
                                output_path: str = 'analysis/feature_importance.png'):
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

    def plot_roc_curves(self, output_path: str = 'analysis/roc_curves.png'):
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

    def analyze_by_price_category(self, gain_threshold: float = 5.0,
                                   output_path: str = 'analysis/price_category_analysis_5pct.png') -> pd.DataFrame:
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
                                output_path: str = 'analysis/time_of_day_analysis_5pct.png') -> pd.DataFrame:
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


def main(gain_threshold: float = 5.0):
    """
    Main execution function.

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
        output_path=f'analysis/feature_importance{threshold_suffix}.png'
    )

    # Step 9: ROC curves
    predictor.plot_roc_curves(output_path=f'analysis/roc_curves{threshold_suffix}.png')

    # Step 10: Trading simulation
    trading_results = predictor.simulate_trading(df, model_name='Random Forest',
                                                  gain_threshold=gain_threshold,
                                                  stop_loss_pct=2.0)

    # Step 11: Generate summary report
    predictor.generate_summary_report(output_path=f'analysis/prediction_summary{threshold_suffix}.txt')

    # Step 12: Price category analysis
    price_analysis = predictor.analyze_by_price_category(
        gain_threshold=gain_threshold,
        output_path=f'analysis/price_category_analysis{threshold_suffix}.png'
    )

    # Step 13: Time of day analysis
    time_analysis = predictor.analyze_by_time_of_day(
        gain_threshold=gain_threshold,
        output_path=f'analysis/time_of_day_analysis{threshold_suffix}.png'
    )

    # Step 14: Save XGBoost model for 1.5% target
    if gain_threshold == 1.5 and 'XGBoost' in predictor.models:
        print("\n" + "="*80)
        print("SAVING XGBOOST MODEL FOR 1.5% TARGET")
        print("="*80)

        model_path = Path('analysis/xgboost_model_1.5pct.json')
        predictor.models['XGBoost'].save_model(model_path)
        print(f"✓ Saved XGBoost model to: {model_path}")

        # Also save feature names for later use
        import json
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

        feature_info_path = Path('analysis/xgboost_model_1.5pct_info.json')
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"✓ Saved model metadata to: {feature_info_path}")

    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE - {gain_threshold}% GAIN TARGET")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - analysis/feature_importance{threshold_suffix}.png")
    print(f"  - analysis/roc_curves{threshold_suffix}.png")
    print(f"  - analysis/prediction_summary{threshold_suffix}.txt")
    print(f"  - analysis/price_category_analysis{threshold_suffix}.png")
    print(f"  - analysis/time_of_day_analysis{threshold_suffix}.png")
    if gain_threshold == 1.5:
        print(f"  - analysis/xgboost_model_1.5pct.json")
        print(f"  - analysis/xgboost_model_1.5pct_info.json")
    print("\nNext steps:")
    print("  1. Review model performance (aim for F1 > 0.60)")
    print("  2. Check feature importance (which features drive success?)")
    print("  3. Review price category analysis (which price bins perform best?)")
    print("  4. Review time of day analysis (when are squeezes most profitable?)")
    print("  5. Validate on additional dates (out-of-sample testing)")
    print("  6. Consider alternative targets (30s gains, stop loss avoidance)")
    print("="*80)

    return predictor, results


if __name__ == "__main__":
    import sys

    # Check for command-line argument
    if len(sys.argv) > 1:
        threshold = float(sys.argv[1])
        print(f"\nRunning analysis with {threshold}% gain threshold\n")
        main(gain_threshold=threshold)
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
            predictor, results = main(gain_threshold=threshold)
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
