#!/usr/bin/env python3
"""
Statistical Analysis of Squeeze Alerts Data

This script performs limited statistical analysis on squeeze alert data with focus on:
1. Feature engineering for statistical independence:
   - ema_spread_pct: Price-normalized EMA momentum (%)
   - price_category: Stock price bins (<$2, $2-5, $5-10, $10-20, $20-40, >$40)
2. Correlation matrix analysis
3. Variance Inflation Factor (VIF) calculation
4. Export of clean, independent feature set

Author: Statistical Analysis Module
Date: 2025-12-15
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class SqueezeAlertsAnalyzer:
    """Analyze squeeze alerts data with focus on statistical independence."""

    def __init__(self, data_directory: str):
        """
        Initialize analyzer with data directory.

        Args:
            data_directory: Path to directory containing squeeze alert JSON files
        """
        self.data_dir = Path(data_directory)
        self.alerts_df = None
        self.independent_features = None
        self.correlation_matrix = None
        self.vif_scores = None

        # Define the independent features (excluding high VIF features)
        # REMOVED: spy_percent_change_day (VIF=18.74), percent_change (VIF=10.03)
        # REMOVED: distance_from_day_low_percent (VIF=18.04, r=0.96 with day_gain)
        self.feature_names = [
            'ema_spread_pct',          # Derived: (ema_9 - ema_21) / price * 100 (price-independent)
            'price_category',          # Categorical: <$2, $2-5, $5-10, $10-20, $20-40, >$40
            'macd_histogram',          # MACD momentum indicator (86% missing)
            'market_session',          # Categorical timing
            'squeeze_number_today',    # Squeeze sequence (VIF=5.32)
            'minutes_since_last_squeeze',  # VIF=1.63
            'window_volume_vs_1min_avg',   # VIF=2.39
            'distance_from_vwap_percent',  # VIF=1.45
            'day_gain',                    # VIF=3.39
            'spy_percent_change_concurrent',  # VIF=3.07 (kept over _day)
            'spread_percent'               # VIF=3.20
        ]

    def load_alerts(self) -> pd.DataFrame:
        """
        Load all squeeze alert JSON files and flatten into DataFrame.

        Returns:
            DataFrame with all alerts and flattened phase1_analysis fields
        """
        print(f"Loading alerts from: {self.data_dir}")
        alerts = []

        json_files = list(self.data_dir.glob("alert_*.json"))
        print(f"Found {len(json_files)} alert files")

        for alert_file in json_files:
            try:
                with open(alert_file, 'r') as f:
                    data = json.load(f)

                    # Flatten phase1_analysis into main dict
                    if 'phase1_analysis' in data:
                        phase1 = data.pop('phase1_analysis')
                        data.update(phase1)

                    # Remove nested objects that aren't needed for analysis
                    data.pop('day_gain_status', None)
                    data.pop('vwap_status', None)
                    data.pop('premarket_high_status', None)
                    data.pop('regular_hours_hod_status', None)
                    data.pop('outcome_tracking', None)
                    data.pop('sent_to_users', None)

                    alerts.append(data)

            except Exception as e:
                print(f"Error loading {alert_file}: {e}")
                continue

        self.alerts_df = pd.DataFrame(alerts)
        print(f"Loaded {len(self.alerts_df)} alerts")
        print(f"Columns: {self.alerts_df.columns.tolist()}")

        return self.alerts_df

    def engineer_features(self) -> pd.DataFrame:
        """
        Create derived features for statistical independence.

        Returns:
            DataFrame with engineered features added
        """
        print("\nEngineering derived features...")

        # 1. Create ema_spread_pct (price-independent EMA momentum)
        # Formula: (ema_9 - ema_21) / price * 100
        # This captures EMA divergence/convergence as a percentage of price
        self.alerts_df['ema_spread_pct'] = None

        for idx, row in self.alerts_df.iterrows():
            ema_9 = row.get('ema_9')
            ema_21 = row.get('ema_21')
            last_price = row.get('last_price')

            if ema_9 is not None and ema_21 is not None and last_price is not None and last_price > 0:
                self.alerts_df.at[idx, 'ema_spread_pct'] = ((ema_9 - ema_21) / last_price) * 100

        # Convert to numeric
        self.alerts_df['ema_spread_pct'] = pd.to_numeric(self.alerts_df['ema_spread_pct'], errors='coerce')

        print(f"✓ Created ema_spread_pct ((ema_9 - ema_21) / price * 100)")
        print(f"  - Non-null values: {self.alerts_df['ema_spread_pct'].notna().sum()}")
        print(f"  - Mean: {self.alerts_df['ema_spread_pct'].mean():.4f}%")
        print(f"  - Std: {self.alerts_df['ema_spread_pct'].std():.4f}%")

        # 2. Create price_category (captures price-level effects on squeeze behavior)
        # Bins: <$2, $2-5, $5-10, $10-20, $20-40, >$40
        def categorize_price(price):
            if pd.isna(price) or price is None:
                return None
            elif price < 2:
                return '<$2'
            elif price < 5:
                return '$2-5'
            elif price < 10:
                return '$5-10'
            elif price < 20:
                return '$10-20'
            elif price < 40:
                return '$20-40'
            else:
                return '>$40'

        self.alerts_df['price_category'] = self.alerts_df['last_price'].apply(categorize_price)

        print(f"✓ Created price_category (bins: <$2, $2-5, $5-10, $10-20, $20-40, >$40)")
        print(f"  - Non-null values: {self.alerts_df['price_category'].notna().sum()}")
        if self.alerts_df['price_category'].notna().any():
            print(f"  - Distribution:")
            for category, count in self.alerts_df['price_category'].value_counts().sort_index().items():
                pct = (count / self.alerts_df['price_category'].notna().sum()) * 100
                print(f"    {category:>8}: {count:4d} ({pct:5.1f}%)")

        return self.alerts_df

    def select_independent_features(self) -> pd.DataFrame:
        """
        Select only the 12 statistically independent features.

        Returns:
            DataFrame with only independent features
        """
        print("\nSelecting independent features...")

        # Check which features exist in the data
        available_features = []
        missing_features = []

        for feature in self.feature_names:
            if feature in self.alerts_df.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)

        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")

        print(f"✓ Available features: {len(available_features)}/{len(self.feature_names)}")

        # Create subset with only independent features
        self.independent_features = self.alerts_df[available_features].copy()

        # Convert categorical variables to numeric codes for numerical analysis
        if 'market_session' in self.independent_features.columns:
            session_mapping = {'extended': 0, 'early': 1, 'mid': 2, 'late': 3}
            self.independent_features['market_session_code'] = self.independent_features['market_session'].map(session_mapping)
            # Keep both for reference, but use _code for correlations

        if 'price_category' in self.independent_features.columns:
            price_mapping = {'<$2': 0, '$2-5': 1, '$5-10': 2, '$10-20': 3, '$20-40': 4, '>$40': 5}
            self.independent_features['price_category_code'] = self.independent_features['price_category'].map(price_mapping)
            # Keep both for reference, but use _code for correlations

        return self.independent_features

    def calculate_correlations(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for independent features.

        Returns:
            Correlation matrix DataFrame
        """
        print("\nCalculating correlation matrix...")

        # Select only numeric columns
        numeric_cols = self.independent_features.select_dtypes(include=[np.number]).columns
        numeric_data = self.independent_features[numeric_cols]

        # Remove columns with zero variance (all NaN or constant)
        variance = numeric_data.var()
        valid_cols = variance[variance > 0].index.tolist()

        print(f"Numeric columns: {len(numeric_cols)}")
        print(f"Valid columns (non-zero variance): {len(valid_cols)}")

        # Calculate correlation matrix
        self.correlation_matrix = numeric_data[valid_cols].corr()

        # Find high correlations (|r| > 0.7, excluding diagonal)
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_corr_pairs.append({
                        'feature_1': self.correlation_matrix.columns[i],
                        'feature_2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })

        if high_corr_pairs:
            print(f"\n⚠️  Found {len(high_corr_pairs)} high correlation pairs (|r| > 0.7):")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
                print(f"  {pair['feature_1']:30s} ↔ {pair['feature_2']:30s}: {pair['correlation']:7.4f}")
        else:
            print("\n✓ No high correlations found (all |r| ≤ 0.7)")

        return self.correlation_matrix

    def plot_correlation_heatmap(self, output_path: str = 'analysis/correlation_heatmap.png'):
        """
        Generate and save correlation heatmap visualization.

        Args:
            output_path: Path to save the heatmap image
        """
        print(f"\nGenerating correlation heatmap...")

        if self.correlation_matrix is None:
            print("Error: Must calculate correlations first")
            return

        # Create figure
        plt.figure(figsize=(14, 12))

        # Create heatmap
        sns.heatmap(
            self.correlation_matrix,
            annot=True,           # Show correlation values
            fmt='.2f',            # 2 decimal places
            cmap='coolwarm',      # Red-blue colormap
            center=0,             # Center at 0
            vmin=-1,              # Min correlation
            vmax=1,               # Max correlation
            square=True,          # Square cells
            linewidths=0.5,       # Cell borders
            cbar_kws={'label': 'Correlation Coefficient'}
        )

        plt.title('Correlation Matrix: Independent Features for Squeeze Alerts',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save figure
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved correlation heatmap to: {output_file}")

        plt.close()

    def calculate_vif(self) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor (VIF) for multicollinearity detection.

        VIF > 5 indicates problematic multicollinearity
        VIF > 10 indicates severe multicollinearity

        Returns:
            DataFrame with VIF scores for each feature
        """
        print("\nCalculating Variance Inflation Factor (VIF)...")

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            print("⚠️  statsmodels not available. Install with: pip install statsmodels")
            return None

        # Select numeric columns only (excluding categorical)
        numeric_cols = self.independent_features.select_dtypes(include=[np.number]).columns.tolist()

        # Remove market_session_code if present (it's derived from categorical)
        if 'market_session_code' in numeric_cols:
            numeric_cols.remove('market_session_code')

        # Strategy: Calculate VIF on features with sufficient data
        # First, check data availability for each feature
        print("\nData Availability Check:")
        print("="*60)
        print(f"{'Feature':<35} {'Available':>10} {'%':>8}")
        print("="*60)

        feature_availability = {}
        for col in numeric_cols:
            count = self.independent_features[col].notna().sum()
            pct = (count / len(self.independent_features)) * 100
            feature_availability[col] = {'count': count, 'pct': pct}
            print(f"{col:<35} {count:10d} {pct:7.1f}%")

        print("="*60)

        # Separate features into two groups:
        # 1. High availability (>50% data) - calculate VIF
        # 2. Low availability (<=50% data) - skip VIF
        high_avail_features = [col for col, stats in feature_availability.items() if stats['pct'] > 50]
        low_avail_features = [col for col, stats in feature_availability.items() if stats['pct'] <= 50]

        if low_avail_features:
            print(f"\n⚠️  Features with >50% missing data (excluded from VIF):")
            for col in low_avail_features:
                pct_missing = 100 - feature_availability[col]['pct']
                print(f"  - {col}: {pct_missing:.1f}% missing")

        if not high_avail_features:
            print("\n⚠️  No features with sufficient data for VIF calculation")
            return None

        # Create clean dataset with only high-availability features
        X = self.independent_features[high_avail_features].dropna()

        if len(X) == 0:
            print("\n⚠️  No complete observations after removing NaN values")
            print("   Attempting VIF calculation on pairwise-complete cases...")

            # Fallback: use all available data (some NaN allowed)
            X = self.independent_features[high_avail_features].copy()

            # For VIF, we need complete cases, so let's just use the features that work
            valid_features = []
            for col in high_avail_features:
                if X[col].notna().sum() >= 30:  # At least 30 observations
                    valid_features.append(col)

            if not valid_features:
                print("\n⚠️  Insufficient data for VIF calculation")
                return None

            X = X[valid_features].dropna()

        print(f"\n✓ Calculating VIF for {len(X.columns)} features on {len(X)} complete observations")

        # Calculate VIF for each feature
        vif_data = []
        for i, col in enumerate(X.columns):
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_data.append({
                    'feature': col,
                    'VIF': vif,
                    'data_points': len(X)
                })
            except Exception as e:
                print(f"  Error calculating VIF for {col}: {e}")
                vif_data.append({
                    'feature': col,
                    'VIF': np.nan,
                    'data_points': len(X)
                })

        self.vif_scores = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

        # Display results
        print("\nVIF Scores (sorted by severity):")
        print("="*70)
        print(f"{'Feature':<35} {'VIF':>10} {'N':>8} {'Status':>12}")
        print("="*70)

        for _, row in self.vif_scores.iterrows():
            feature = row['feature']
            vif = row['VIF']
            n = row['data_points']

            if pd.isna(vif):
                status = "ERROR"
            elif vif > 10:
                status = "🔴 SEVERE"
            elif vif > 5:
                status = "🟡 WARNING"
            else:
                status = "🟢 GOOD"

            print(f"{feature:<35} {vif:10.2f} {n:8d} {status:>12}")

        print("="*70)
        print("\nVIF Interpretation:")
        print("  VIF < 5:   🟢 Low multicollinearity (good)")
        print("  VIF 5-10:  🟡 Moderate multicollinearity (concerning)")
        print("  VIF > 10:  🔴 High multicollinearity (problematic)")
        print(f"\nNote: VIF calculated on {len(X)} complete observations")

        return self.vif_scores

    def export_clean_dataset(self, output_path: str = 'analysis/squeeze_alerts_independent_features.csv'):
        """
        Export clean dataset with only independent features.

        Args:
            output_path: Path to save CSV file
        """
        print(f"\nExporting clean dataset...")

        # Add metadata columns for reference
        metadata_cols = ['symbol', 'timestamp', 'last_price']
        export_cols = []

        # Add metadata if available
        for col in metadata_cols:
            if col in self.alerts_df.columns:
                export_cols.append(col)

        # Add independent features
        export_cols.extend(self.feature_names)

        # Create export dataframe
        export_df = self.alerts_df[export_cols].copy()

        # Save to CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(output_file, index=False)

        print(f"✓ Exported {len(export_df)} alerts with {len(export_cols)} columns to: {output_file}")
        print(f"  Columns: {export_cols}")

        return export_df

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics report.

        Returns:
            Dictionary containing summary statistics
        """
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS SUMMARY REPORT")
        print("="*80)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_directory': str(self.data_dir),
            'total_alerts': len(self.alerts_df),
            'features': {}
        }

        print(f"\nDataset: {self.data_dir}")
        print(f"Total Alerts: {len(self.alerts_df)}")
        print(f"Date Range: {self.alerts_df['timestamp'].min()} to {self.alerts_df['timestamp'].max()}")

        print(f"\n{'Feature Statistics':-^80}")
        print(f"{'Feature':<35} {'Count':>8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-"*80)

        for feature in self.feature_names:
            if feature in self.independent_features.columns:
                series = self.independent_features[feature]

                # Skip non-numeric
                if not pd.api.types.is_numeric_dtype(series):
                    continue

                count = series.notna().sum()
                mean = series.mean()
                std = series.std()
                min_val = series.min()
                max_val = series.max()

                print(f"{feature:<35} {count:8d} {mean:10.4f} {std:10.4f} {min_val:10.4f} {max_val:10.4f}")

                summary['features'][feature] = {
                    'count': int(count),
                    'mean': float(mean) if not pd.isna(mean) else None,
                    'std': float(std) if not pd.isna(std) else None,
                    'min': float(min_val) if not pd.isna(min_val) else None,
                    'max': float(max_val) if not pd.isna(max_val) else None
                }

        print("-"*80)

        # Missing data analysis
        print(f"\n{'Missing Data Analysis':-^80}")
        missing_pct = (self.independent_features[self.feature_names].isna().sum() / len(self.independent_features) * 100)
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

        if len(missing_pct) > 0:
            print(f"{'Feature':<35} {'Missing %':>12}")
            print("-"*50)
            for feature, pct in missing_pct.items():
                print(f"{feature:<35} {pct:11.2f}%")
        else:
            print("✓ No missing data")

        print("\n" + "="*80)

        return summary


def main():
    """Main execution function."""

    # Configuration
    data_dir = "/home/wilsonb/dl/github.com/Z223I/alpaca/historical_data/2025-12-15/squeeze_alerts_sent"

    print("="*80)
    print("SQUEEZE ALERTS STATISTICAL ANALYSIS")
    print("="*80)
    print(f"Analyzing data with focus on statistical independence")
    print(f"Target: 11 independent features (VIF < 5)")
    print(f"NEW: ema_spread_pct (price-normalized), price_category (6 bins)")
    print(f"EXCLUDED: spy_percent_change_day, percent_change, distance_from_day_low_percent")
    print("="*80)

    # Initialize analyzer
    analyzer = SqueezeAlertsAnalyzer(data_dir)

    # Step 1: Load data
    analyzer.load_alerts()

    # Step 2: Engineer derived features
    analyzer.engineer_features()

    # Step 3: Select independent features
    analyzer.select_independent_features()

    # Step 4: Calculate correlations
    analyzer.calculate_correlations()

    # Step 5: Generate correlation heatmap
    analyzer.plot_correlation_heatmap()

    # Step 6: Calculate VIF scores
    analyzer.calculate_vif()

    # Step 7: Export clean dataset
    analyzer.export_clean_dataset()

    # Step 8: Generate summary report
    summary = analyzer.generate_summary_report()

    # Save summary to JSON
    summary_path = Path('analysis/summary_report.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary report to: {summary_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - analysis/correlation_heatmap.png")
    print("  - analysis/squeeze_alerts_independent_features.csv")
    print("  - analysis/summary_report.json")
    print("\nNext steps:")
    print("  1. Review correlation heatmap for any unexpected correlations")
    print("  2. Check VIF scores - all should be < 5 for true independence")
    print("  3. Use the clean CSV for further statistical modeling")
    print("="*80)


if __name__ == "__main__":
    main()
