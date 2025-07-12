#!/usr/bin/env python3
"""
ORB PCA Analysis for Profitability Improvement

This script performs Principal Component Analysis on historical market data
to identify the most important features for profitable ORB breakouts.
Uses data from 2025-07-10 and 2025-07-11 to optimize the ORB alert system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, time
import pytz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ORBPCAAnalyzer:
    """PCA analyzer for ORB trading patterns."""
    
    def __init__(self, data_dir: str = "historical_data"):
        self.data_dir = Path(data_dir)
        self.market_open = time(9, 30)  # 9:30 AM ET
        self.orb_period_minutes = 15
        self.features = []
        self.target_returns = []
        self.symbol_data = {}
        
    def load_historical_data(self, dates: List[str]) -> Dict[str, pd.DataFrame]:
        """Load historical market data for specified dates."""
        all_data = {}
        
        for date in dates:
            date_dir = self.data_dir / date / "market_data"
            if not date_dir.exists():
                print(f"Warning: No data directory found for {date}")
                continue
                
            # Load all CSV files for this date
            csv_files = list(date_dir.glob("*.csv"))
            print(f"Found {len(csv_files)} data files for {date}")
            
            for csv_file in csv_files:
                # Extract symbol from filename
                symbol = csv_file.stem.split('_')[0]
                
                try:
                    df = pd.read_csv(csv_file)
                    if df.empty:
                        continue
                        
                    # Add date and symbol info
                    df['date'] = date
                    df['symbol'] = symbol
                    
                    # Convert timestamp to datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Combine data for this symbol
                    if symbol not in all_data:
                        all_data[symbol] = []
                    all_data[symbol].append(df)
                    
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
        
        # Concatenate data for each symbol
        combined_data = {}
        for symbol, dfs in all_data.items():
            if dfs:
                combined_data[symbol] = pd.concat(dfs, ignore_index=True)
                print(f"Loaded {len(combined_data[symbol])} records for {symbol}")
        
        return combined_data
    
    def load_alert_data(self, dates: List[str]) -> List[Dict]:
        """Load historical alert data for specified dates."""
        all_alerts = []
        
        for date in dates:
            alert_dir = self.data_dir / date / "alerts"
            if not alert_dir.exists():
                continue
                
            # Load alerts from both bullish and bearish subdirectories
            for subdir in ["bullish", "bearish"]:
                subdir_path = alert_dir / subdir
                if not subdir_path.exists():
                    continue
                    
                for alert_file in subdir_path.glob("*.json"):
                    try:
                        with open(alert_file, 'r') as f:
                            alert_data = json.load(f)
                            alert_data['alert_type'] = subdir
                            alert_data['date'] = date
                            all_alerts.append(alert_data)
                    except Exception as e:
                        print(f"Error loading alert {alert_file}: {e}")
        
        print(f"Loaded {len(all_alerts)} historical alerts")
        return all_alerts
    
    def calculate_orb_features(self, symbol_data: pd.DataFrame) -> Optional[Dict]:
        """Calculate ORB-specific features for a symbol's data."""
        if symbol_data.empty:
            return None
            
        # Filter for opening range period
        orb_data = self._filter_opening_range(symbol_data)
        if orb_data.empty or len(orb_data) < 5:
            return None
            
        # Basic ORB levels
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()
        orb_range = orb_high - orb_low
        orb_midpoint = (orb_high + orb_low) / 2
        
        # Volume features during ORB period
        orb_volume = orb_data['volume'].sum()
        orb_avg_volume = orb_data['volume'].mean()
        orb_volume_ratio = orb_data['volume'].max() / orb_data['volume'].mean() if orb_data['volume'].mean() > 0 else 1
        
        # Price movement features
        orb_open = orb_data['close'].iloc[0] if len(orb_data) > 0 else orb_low
        orb_close = orb_data['close'].iloc[-1] if len(orb_data) > 0 else orb_high
        orb_price_change = orb_close - orb_open
        orb_price_change_pct = (orb_price_change / orb_open * 100) if orb_open > 0 else 0
        
        # Volatility features
        orb_volatility = orb_data['close'].std() if len(orb_data) > 1 else 0
        orb_range_pct = (orb_range / orb_midpoint * 100) if orb_midpoint > 0 else 0
        
        # Time-based features
        orb_duration_minutes = len(orb_data)  # Number of minute bars
        
        # Technical indicators within ORB
        if len(orb_data) >= 3:
            orb_trend = self._calculate_trend(orb_data['close'])
            orb_momentum = self._calculate_momentum(orb_data['close'])
        else:
            orb_trend = 0
            orb_momentum = 0
            
        # Volume-weighted average price (VWAP) during ORB
        if orb_data['volume'].sum() > 0:
            orb_vwap = (orb_data['close'] * orb_data['volume']).sum() / orb_data['volume'].sum()
        else:
            orb_vwap = orb_midpoint
            
        return {
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_range': orb_range,
            'orb_range_pct': orb_range_pct,
            'orb_midpoint': orb_midpoint,
            'orb_volume': orb_volume,
            'orb_avg_volume': orb_avg_volume,
            'orb_volume_ratio': orb_volume_ratio,
            'orb_price_change': orb_price_change,
            'orb_price_change_pct': orb_price_change_pct,
            'orb_volatility': orb_volatility,
            'orb_duration_minutes': orb_duration_minutes,
            'orb_trend': orb_trend,
            'orb_momentum': orb_momentum,
            'orb_vwap': orb_vwap,
            'orb_open': orb_open,
            'orb_close': orb_close
        }
    
    def calculate_post_orb_returns(self, symbol_data: pd.DataFrame, orb_features: Dict) -> Dict:
        """Calculate returns after ORB period using different strategies."""
        # Filter data after ORB period
        post_orb_data = self._filter_post_orb(symbol_data)
        if post_orb_data.empty:
            return {'return_1h': 0, 'return_2h': 0, 'return_eod': 0, 'max_return': 0, 'min_return': 0}
            
        orb_close = orb_features['orb_close']
        
        # Calculate returns at different time horizons
        returns = {}
        
        # 1-hour return
        hour_1_data = post_orb_data[post_orb_data['timestamp'] <= post_orb_data['timestamp'].iloc[0] + pd.Timedelta(hours=1)]
        if not hour_1_data.empty:
            returns['return_1h'] = ((hour_1_data['close'].iloc[-1] - orb_close) / orb_close * 100)
        else:
            returns['return_1h'] = 0
            
        # 2-hour return
        hour_2_data = post_orb_data[post_orb_data['timestamp'] <= post_orb_data['timestamp'].iloc[0] + pd.Timedelta(hours=2)]
        if not hour_2_data.empty:
            returns['return_2h'] = ((hour_2_data['close'].iloc[-1] - orb_close) / orb_close * 100)
        else:
            returns['return_2h'] = 0
            
        # End-of-day return
        returns['return_eod'] = ((post_orb_data['close'].iloc[-1] - orb_close) / orb_close * 100)
        
        # Max and min returns during the day
        max_price = post_orb_data['high'].max()
        min_price = post_orb_data['low'].min()
        returns['max_return'] = ((max_price - orb_close) / orb_close * 100)
        returns['min_return'] = ((min_price - orb_close) / orb_close * 100)
        
        # Breakout success metrics
        orb_high = orb_features['orb_high']
        orb_low = orb_features['orb_low']
        
        # Check if price broke above ORB high
        breakout_above = post_orb_data['high'].max() > orb_high * 1.002  # 0.2% threshold
        # Check if price broke below ORB low
        breakdown_below = post_orb_data['low'].min() < orb_low * 0.998  # 0.2% threshold
        
        returns['breakout_above'] = 1 if breakout_above else 0
        returns['breakdown_below'] = 1 if breakdown_below else 0
        
        return returns
    
    def _filter_opening_range(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Filter data for opening range period."""
        if price_data.empty or 'timestamp' not in price_data.columns:
            return pd.DataFrame()
            
        price_data = price_data.copy()
        
        # Convert to time for filtering
        if not pd.api.types.is_datetime64_any_dtype(price_data['timestamp']):
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        price_data['time'] = price_data['timestamp'].dt.time
        
        # Calculate ORB end time
        market_open_minutes = self.market_open.hour * 60 + self.market_open.minute
        orb_end_minutes = market_open_minutes + self.orb_period_minutes
        orb_end_time = time(orb_end_minutes // 60, orb_end_minutes % 60)
        
        # Filter for ORB period
        orb_mask = (price_data['time'] >= self.market_open) & (price_data['time'] <= orb_end_time)
        return price_data[orb_mask]
    
    def _filter_post_orb(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Filter data for post-ORB period."""
        if price_data.empty or 'timestamp' not in price_data.columns:
            return pd.DataFrame()
            
        price_data = price_data.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(price_data['timestamp']):
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        price_data['time'] = price_data['timestamp'].dt.time
        
        # Calculate ORB end time
        market_open_minutes = self.market_open.hour * 60 + self.market_open.minute
        orb_end_minutes = market_open_minutes + self.orb_period_minutes
        orb_end_time = time(orb_end_minutes // 60, orb_end_minutes % 60)
        
        # Filter for post-ORB period
        post_orb_mask = price_data['time'] > orb_end_time
        return price_data[post_orb_mask]
    
    def _calculate_trend(self, prices: pd.Series) -> float:
        """Calculate simple trend indicator."""
        if len(prices) < 2:
            return 0
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
    
    def _calculate_momentum(self, prices: pd.Series) -> float:
        """Calculate momentum indicator."""
        if len(prices) < 3:
            return 0
        return prices.diff().mean()
    
    def build_feature_dataset(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build comprehensive feature dataset for PCA analysis."""
        feature_rows = []
        
        for symbol, data in symbol_data.items():
            if data.empty:
                continue
                
            # Group by date to analyze each day separately
            for date in data['date'].unique():
                date_data = data[data['date'] == date].copy()
                date_data = date_data.sort_values('timestamp')
                
                # Calculate ORB features
                orb_features = self.calculate_orb_features(date_data)
                if orb_features is None:
                    continue
                    
                # Calculate post-ORB returns
                returns = self.calculate_post_orb_returns(date_data, orb_features)
                
                # Combine features and returns
                feature_row = {
                    'symbol': symbol,
                    'date': date,
                    **orb_features,
                    **returns
                }
                
                feature_rows.append(feature_row)
        
        return pd.DataFrame(feature_rows)
    
    def perform_pca_analysis(self, feature_df: pd.DataFrame) -> Tuple[PCA, pd.DataFrame, List[str]]:
        """Perform PCA analysis on ORB features."""
        # Select features for PCA (exclude non-numeric columns and target variables)
        feature_columns = [
            'orb_range', 'orb_range_pct', 'orb_volume', 'orb_avg_volume', 'orb_volume_ratio',
            'orb_price_change', 'orb_price_change_pct', 'orb_volatility', 'orb_duration_minutes',
            'orb_trend', 'orb_momentum', 'orb_high', 'orb_low', 'orb_midpoint'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in feature_df.columns]
        
        if not available_features:
            raise ValueError("No valid features found for PCA analysis")
        
        # Prepare feature matrix
        X = feature_df[available_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA DataFrame
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=feature_df.index)
        
        return pca, pca_df, available_features
    
    def analyze_profitability_factors(self, feature_df: pd.DataFrame, pca_df: pd.DataFrame, 
                                    pca: PCA, feature_names: List[str]) -> Dict:
        """Analyze which factors contribute most to profitability."""
        results = {}
        
        # Target variables to analyze
        target_vars = ['return_1h', 'return_2h', 'return_eod', 'max_return']
        
        for target in target_vars:
            if target not in feature_df.columns:
                continue
                
            print(f"\nAnalyzing {target}...")
            
            # Remove rows with missing target values
            valid_mask = ~feature_df[target].isna()
            X_valid = pca_df[valid_mask]
            y_valid = feature_df.loc[valid_mask, target]
            
            if len(y_valid) < 10:  # Need minimum samples
                continue
            
            # Train Random Forest to find important components
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_valid, y_valid)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Analyze which original features contribute to top components
            component_analysis = {}
            for i, importance in enumerate(importances[:5]):  # Top 5 components
                pc_name = f'PC{i+1}'
                component_loadings = pca.components_[i]
                
                # Find most important original features for this component
                feature_contributions = {}
                for j, loading in enumerate(component_loadings):
                    if j < len(feature_names):
                        feature_contributions[feature_names[j]] = abs(loading)
                
                # Sort by contribution
                sorted_features = sorted(feature_contributions.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                component_analysis[pc_name] = {
                    'importance': importance,
                    'explained_variance': pca.explained_variance_ratio_[i],
                    'top_features': sorted_features[:5]
                }
            
            results[target] = {
                'component_analysis': component_analysis,
                'model_score': rf.score(X_valid, y_valid),
                'total_samples': len(y_valid)
            }
        
        return results
    
    def generate_recommendations(self, analysis_results: Dict, feature_df: pd.DataFrame) -> Dict:
        """Generate actionable recommendations for ORB strategy improvement."""
        recommendations = {
            'key_features': {},
            'thresholds': {},
            'strategy_improvements': []
        }
        
        # Aggregate feature importance across all targets
        feature_importance_sum = {}
        
        for target, results in analysis_results.items():
            if 'component_analysis' not in results:
                continue
                
            for pc_name, pc_data in results['component_analysis'].items():
                for feature_name, contribution in pc_data['top_features']:
                    if feature_name not in feature_importance_sum:
                        feature_importance_sum[feature_name] = 0
                    feature_importance_sum[feature_name] += contribution * pc_data['importance']
        
        # Sort features by total importance
        sorted_features = sorted(feature_importance_sum.items(), 
                               key=lambda x: x[1], reverse=True)
        
        recommendations['key_features'] = dict(sorted_features[:10])
        
        # Calculate optimal thresholds for top features
        top_features = [name for name, _ in sorted_features[:5]]
        
        for feature in top_features:
            if feature in feature_df.columns:
                # Find profitable vs unprofitable thresholds
                feature_values = feature_df[feature].dropna()
                returns = feature_df.loc[feature_values.index, 'return_eod'].fillna(0)
                
                # Find median split that maximizes return difference
                median_val = feature_values.median()
                above_median_returns = returns[feature_values > median_val].mean()
                below_median_returns = returns[feature_values <= median_val].mean()
                
                recommendations['thresholds'][feature] = {
                    'optimal_threshold': median_val,
                    'above_median_return': above_median_returns,
                    'below_median_return': below_median_returns,
                    'return_difference': above_median_returns - below_median_returns
                }
        
        # Generate strategy improvements
        if 'orb_range_pct' in recommendations['key_features']:
            recommendations['strategy_improvements'].append(
                "Add ORB range percentage filter - focus on symbols with optimal range size"
            )
        
        if 'orb_volume_ratio' in recommendations['key_features']:
            recommendations['strategy_improvements'].append(
                "Enhance volume filtering - prioritize symbols with high volume concentration"
            )
        
        if 'orb_trend' in recommendations['key_features']:
            recommendations['strategy_improvements'].append(
                "Incorporate trend direction - bias toward symbols showing directional movement in ORB"
            )
        
        return recommendations
    
    def run_complete_analysis(self, dates: List[str] = ["2025-07-10", "2025-07-11"]) -> Dict:
        """Run complete PCA analysis and generate recommendations."""
        print("Starting ORB PCA Analysis...")
        
        # Load historical data
        print("\n1. Loading historical market data...")
        symbol_data = self.load_historical_data(dates)
        
        if not symbol_data:
            raise ValueError("No historical data found")
        
        # Build feature dataset
        print("\n2. Building feature dataset...")
        feature_df = self.build_feature_dataset(symbol_data)
        print(f"Built feature dataset with {len(feature_df)} samples")
        
        if feature_df.empty:
            raise ValueError("Failed to build feature dataset")
        
        # Perform PCA analysis
        print("\n3. Performing PCA analysis...")
        pca, pca_df, feature_names = self.perform_pca_analysis(feature_df)
        
        # Analyze profitability factors
        print("\n4. Analyzing profitability factors...")
        analysis_results = self.analyze_profitability_factors(feature_df, pca_df, pca, feature_names)
        
        # Generate recommendations
        print("\n5. Generating recommendations...")
        recommendations = self.generate_recommendations(analysis_results, feature_df)
        
        # Compile complete results
        complete_results = {
            'analysis_date': datetime.now().isoformat(),
            'data_dates': dates,
            'total_samples': len(feature_df),
            'total_symbols': feature_df['symbol'].nunique(),
            'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'feature_names': feature_names,
            'profitability_analysis': analysis_results,
            'recommendations': recommendations,
            'feature_statistics': feature_df.describe().to_dict()
        }
        
        return complete_results
    
    def save_results(self, results: Dict, output_file: str = "orb_pca_analysis_results.json"):
        """Save analysis results to file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")


def main():
    """Main execution function."""
    print("ORB PCA Analysis for Profitability Improvement")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ORBPCAAnalyzer()
    
    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Save results
        analyzer.save_results(results)
        
        # Print key findings
        print("\n" + "=" * 50)
        print("KEY FINDINGS")
        print("=" * 50)
        
        # PCA Summary
        cumulative_var = results['cumulative_variance']
        print(f"\nPCA Explained Variance:")
        for i, var in enumerate(results['pca_explained_variance'][:5]):
            print(f"  PC{i+1}: {var:.1%} (Cumulative: {cumulative_var[i]:.1%})")
        
        # Top Features
        print(f"\nMost Important Features for Profitability:")
        for feature, importance in list(results['recommendations']['key_features'].items())[:5]:
            print(f"  {feature}: {importance:.3f}")
        
        # Recommended Thresholds
        print(f"\nRecommended Thresholds:")
        for feature, threshold_data in results['recommendations']['thresholds'].items():
            print(f"  {feature}: {threshold_data['optimal_threshold']:.3f}")
            print(f"    Return difference: {threshold_data['return_difference']:.2f}%")
        
        # Strategy Improvements
        print(f"\nRecommended Strategy Improvements:")
        for i, improvement in enumerate(results['recommendations']['strategy_improvements'], 1):
            print(f"  {i}. {improvement}")
        
        print("\n" + "=" * 50)
        print("Analysis completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()