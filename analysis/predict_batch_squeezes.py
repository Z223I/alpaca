#!/usr/bin/env python3
"""
Batch process squeeze alerts using trained XGBoost model.

Filter alerts by price range, time range, and symbols, then predict outcomes.

Usage:
    # Interactive mode (prompts for all inputs)
    python3 analysis/predict_batch_squeezes.py

    # Command-line mode (no prompts)
    python3 analysis/predict_batch_squeezes.py \
        --model analysis/xgboost_model_1.5pct.json \
        --directory historical_data/2025-12-15/squeeze_alerts_sent \
        --min-price 5.0 \
        --max-price 10.0 \
        --start-time "09:30" \
        --end-time "10:30" \
        --symbols AAPL TSLA NVDA

    # Filter by price only
    python3 analysis/predict_batch_squeezes.py \
        --model analysis/xgboost_model_1.5pct.json \
        --directory historical_data/2025-12-15/squeeze_alerts_sent \
        --min-price 5.0 \
        --max-price 10.0

Author: Predictive Analytics Module
Date: 2025-12-15
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, time
from typing import List, Optional, Dict
import pandas as pd
import xgboost as xgb
import numpy as np


class BatchSqueezePredictor:
    def __init__(self, model_path: str, info_path: str):
        """Load trained model and metadata."""

        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        # Load metadata
        with open(info_path, 'r') as f:
            info = json.load(f)
            self.feature_names = info['feature_names']
            self.gain_threshold = info['gain_threshold']
            self.performance = info['model_performance']

        print(f"\n✓ Loaded model for {self.gain_threshold}% gain target")
        print(f"  Test Accuracy: {self.performance['test_accuracy']:.3f}")
        print(f"  Precision: {self.performance['precision']:.3f}")
        print(f"  F1-Score: {self.performance['f1_score']:.3f}\n")

    def predict_single_alert(self, alert_data: dict) -> dict:
        """Predict outcome for a single squeeze alert."""

        # Encode categorical variables
        market_session_map = {'extended': 0, 'early': 1, 'mid_day': 2, 'unknown': 3}
        price_category_map = {'<$2': 0, '$2-5': 1, '$5-10': 2, '$10-20': 3, '$20-40': 4, '>$40': 5, 'unknown': 6}

        # Build feature vector
        features = []

        # Extract and calculate features
        last_price = alert_data.get('last_price', 0)
        ema_9 = alert_data.get('ema_9')
        ema_21 = alert_data.get('ema_21')

        # 1. ema_spread_pct
        if ema_9 is not None and ema_21 is not None and last_price > 0:
            ema_spread_pct = ((ema_9 - ema_21) / last_price) * 100
            ema_missing = 0
        else:
            ema_spread_pct = 0.0
            ema_missing = 1
        features.append(ema_spread_pct)

        # 2. price_category_encoded
        def get_price_category(price):
            if price < 2:
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

        price_cat = get_price_category(last_price) if last_price > 0 else 'unknown'
        features.append(price_category_map.get(price_cat, 6))

        # 3. macd_histogram
        features.append(alert_data.get('macd_histogram', 0.0))

        # 4. market_session_encoded
        market_session = alert_data.get('market_session', 'unknown')
        features.append(market_session_map.get(market_session, 3))

        # 5-11. Other numeric features
        features.append(alert_data.get('squeeze_number_today', 1))
        features.append(alert_data.get('minutes_since_last_squeeze', 5.0))
        features.append(alert_data.get('window_volume_vs_1min_avg', 1.0))
        features.append(alert_data.get('distance_from_vwap_percent', 0.0))
        features.append(alert_data.get('day_gain', 0.0))
        features.append(alert_data.get('spy_percent_change_concurrent', 0.0))
        features.append(alert_data.get('spread_percent', 0.5))

        # 12. ema_spread_pct_missing
        features.append(ema_missing)

        # Make prediction
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]

        return {
            'prediction': bool(prediction),
            'probability': float(probability),
            'price_category': price_cat
        }


def load_alerts_from_directory(directory: Path,
                               min_price: Optional[float] = None,
                               max_price: Optional[float] = None,
                               start_time: Optional[time] = None,
                               end_time: Optional[time] = None,
                               symbols: Optional[List[str]] = None) -> List[Dict]:
    """
    Load and filter squeeze alerts from directory.

    Args:
        directory: Path to directory containing alert JSON files
        min_price: Minimum stock price (inclusive)
        max_price: Maximum stock price (inclusive)
        start_time: Start time filter (e.g., 09:30)
        end_time: End time filter (e.g., 10:30)
        symbols: List of symbols to include (None = all symbols)

    Returns:
        List of filtered alert dictionaries
    """
    alerts = []

    # Find all JSON files
    json_files = list(directory.glob('alert_*.json'))

    print(f"Scanning directory: {directory}")
    print(f"Found {len(json_files)} alert files")

    # Apply filters
    filters_applied = []
    if min_price is not None:
        filters_applied.append(f"min_price >= ${min_price:.2f}")
    if max_price is not None:
        filters_applied.append(f"max_price <= ${max_price:.2f}")
    if start_time is not None:
        filters_applied.append(f"time >= {start_time.strftime('%H:%M')}")
    if end_time is not None:
        filters_applied.append(f"time <= {end_time.strftime('%H:%M')}")
    if symbols:
        filters_applied.append(f"symbols in {symbols}")

    if filters_applied:
        print(f"Filters: {', '.join(filters_applied)}")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                alert = json.load(f)

            # Apply price filter
            last_price = alert.get('last_price', 0)
            if min_price is not None and last_price < min_price:
                continue
            if max_price is not None and last_price > max_price:
                continue

            # Apply time filter
            if start_time is not None or end_time is not None:
                timestamp_str = alert.get('timestamp')
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    alert_time = timestamp.time()

                    if start_time is not None and alert_time < start_time:
                        continue
                    if end_time is not None and alert_time > end_time:
                        continue

            # Apply symbol filter
            if symbols is not None:
                symbol = alert.get('symbol', '')
                if symbol not in symbols:
                    continue

            alerts.append(alert)

        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            continue

    print(f"✓ Loaded {len(alerts)} alerts matching filters\n")
    return alerts


def parse_time(time_str: str) -> time:
    """Parse time string in HH:MM format."""
    try:
        hour, minute = map(int, time_str.split(':'))
        return time(hour, minute)
    except:
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM (e.g., 09:30)")


def prompt_for_input(prompt_text: str, default: Optional[str] = None) -> str:
    """Prompt user for input with optional default."""
    if default:
        prompt_text = f"{prompt_text} [{default}]"

    user_input = input(f"{prompt_text}: ").strip()

    if not user_input and default:
        return default

    return user_input


def main():
    parser = argparse.ArgumentParser(
        description='Batch process squeeze alerts using trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 analysis/predict_batch_squeezes.py

  # Filter by price range ($5-10)
  python3 analysis/predict_batch_squeezes.py \\
      --model analysis/xgboost_model_1.5pct.json \\
      --directory historical_data/2025-12-15/squeeze_alerts_sent \\
      --min-price 5.0 --max-price 10.0

  # First hour after open, specific symbols
  python3 analysis/predict_batch_squeezes.py \\
      --model analysis/xgboost_model_1.5pct.json \\
      --directory historical_data/2025-12-15/squeeze_alerts_sent \\
      --start-time 09:30 --end-time 10:30 \\
      --symbols AAPL TSLA NVDA
        """
    )

    parser.add_argument('--model', type=str,
                       help='Path to trained XGBoost model (e.g., analysis/xgboost_model_1.5pct.json)')
    parser.add_argument('--directory', type=str,
                       help='Directory containing alert JSON files')
    parser.add_argument('--min-price', type=float,
                       help='Minimum stock price (inclusive)')
    parser.add_argument('--max-price', type=float,
                       help='Maximum stock price (inclusive)')
    parser.add_argument('--start-time', type=str,
                       help='Start time in HH:MM format (e.g., 09:30)')
    parser.add_argument('--end-time', type=str,
                       help='End time in HH:MM format (e.g., 10:30)')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='List of symbols to include (space-separated)')
    parser.add_argument('--output', type=str, default='analysis/batch_predictions.csv',
                       help='Output CSV file path (default: analysis/batch_predictions.csv)')

    args = parser.parse_args()

    print("="*80)
    print("BATCH SQUEEZE ALERT PREDICTION")
    print("="*80)

    # Get model path
    if args.model:
        model_path = args.model
    else:
        model_path = prompt_for_input("Model path", "analysis/xgboost_model_1.5pct.json")

    # Derive info path
    info_path = model_path.replace('.json', '_info.json')

    # Verify files exist
    if not Path(model_path).exists():
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    if not Path(info_path).exists():
        print(f"❌ Error: Info file not found: {info_path}")
        sys.exit(1)

    # Get directory path
    if args.directory:
        directory = Path(args.directory)
    else:
        dir_path = prompt_for_input("Alert directory", "historical_data/2025-12-15/squeeze_alerts_sent")
        directory = Path(dir_path)

    if not directory.exists():
        print(f"❌ Error: Directory not found: {directory}")
        sys.exit(1)

    # Get optional filters
    min_price = args.min_price
    if min_price is None and not args.model:  # Only prompt in interactive mode
        min_input = prompt_for_input("Minimum price (press Enter to skip)", "")
        min_price = float(min_input) if min_input else None

    max_price = args.max_price
    if max_price is None and not args.model:
        max_input = prompt_for_input("Maximum price (press Enter to skip)", "")
        max_price = float(max_input) if max_input else None

    start_time = None
    if args.start_time:
        start_time = parse_time(args.start_time)
    elif not args.model:
        start_input = prompt_for_input("Start time HH:MM (press Enter to skip)", "")
        start_time = parse_time(start_input) if start_input else None

    end_time = None
    if args.end_time:
        end_time = parse_time(args.end_time)
    elif not args.model:
        end_input = prompt_for_input("End time HH:MM (press Enter to skip)", "")
        end_time = parse_time(end_input) if end_input else None

    symbols = args.symbols
    if symbols is None and not args.model:
        symbols_input = prompt_for_input("Symbols (space-separated, press Enter to skip)", "")
        symbols = symbols_input.split() if symbols_input else None

    # Load model
    predictor = BatchSqueezePredictor(model_path, info_path)

    # Load and filter alerts
    alerts = load_alerts_from_directory(
        directory,
        min_price=min_price,
        max_price=max_price,
        start_time=start_time,
        end_time=end_time,
        symbols=symbols
    )

    if len(alerts) == 0:
        print("⚠️  No alerts match the filters. Exiting.")
        sys.exit(0)

    # Process alerts
    print("Processing alerts...")
    results = []

    for alert in alerts:
        prediction = predictor.predict_single_alert(alert)

        # Parse timestamp
        timestamp_str = alert.get('timestamp', '')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            time_str = timestamp.strftime('%H:%M:%S')
        else:
            time_str = 'N/A'

        results.append({
            'symbol': alert.get('symbol', 'N/A'),
            'timestamp': timestamp_str,
            'time': time_str,
            'price': alert.get('last_price', 0),
            'price_category': prediction['price_category'],
            'market_session': alert.get('market_session', 'N/A'),
            'prediction': 'TAKE' if prediction['prediction'] else 'SKIP',
            'probability': prediction['probability'],
            'day_gain': alert.get('day_gain', 0),
            'volume_surge': alert.get('window_volume_vs_1min_avg', 0),
            'distance_vwap': alert.get('distance_from_vwap_percent', 0)
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Display summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total Alerts Processed: {len(df)}")
    print(f"TAKE recommendations: {(df['prediction'] == 'TAKE').sum()} ({(df['prediction'] == 'TAKE').mean()*100:.1f}%)")
    print(f"SKIP recommendations: {(df['prediction'] == 'SKIP').sum()} ({(df['prediction'] == 'SKIP').mean()*100:.1f}%)")
    print(f"\nAverage Success Probability: {df['probability'].mean():.1%}")
    print(f"High Confidence (>70%): {(df['probability'] > 0.7).sum()} alerts")
    print(f"Low Confidence (<30%): {(df['probability'] < 0.3).sum()} alerts")

    # Display top predictions
    print("\n" + "="*80)
    print("TOP 10 PREDICTIONS (Highest Probability)")
    print("="*80)
    top_10 = df.nlargest(10, 'probability')
    print(top_10[['symbol', 'time', 'price', 'prediction', 'probability', 'day_gain']].to_string(index=False))

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df)} predictions to: {output_path}")

    # Breakdown by price category
    print("\n" + "="*80)
    print("BREAKDOWN BY PRICE CATEGORY")
    print("="*80)
    category_summary = df.groupby('price_category').agg({
        'symbol': 'count',
        'prediction': lambda x: (x == 'TAKE').sum(),
        'probability': 'mean'
    }).round(3)
    category_summary.columns = ['Count', 'TAKE_Count', 'Avg_Probability']
    category_summary['TAKE_Pct'] = (category_summary['TAKE_Count'] / category_summary['Count'] * 100).round(1)
    print(category_summary.to_string())

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
