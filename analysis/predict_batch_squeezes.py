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
from typing import List, Optional, Dict, Tuple
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


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

            # Flatten phase1_analysis into main alert dict (CRITICAL FIX)
            if 'phase1_analysis' in alert:
                phase1 = alert.pop('phase1_analysis')
                alert.update(phase1)

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


def calculate_actual_profit(alert: Dict, take_profit_pct: float = 1.5,
                           stop_loss_pct: float = 2.0) -> Tuple[float, str]:
    """
    Calculate actual profit from outcome_tracking data.

    Uses chronological interval logic with take profit and trailing stop loss.

    Args:
        alert: Alert dictionary with outcome_tracking
        take_profit_pct: Take profit percentage (default 1.5%)
        stop_loss_pct: Stop loss percentage (default 2.0%)

    Returns:
        Tuple of (profit_pct, outcome_reason)
    """
    outcome_tracking = alert.get('outcome_tracking')
    if not outcome_tracking:
        return (None, 'no_outcome_data')

    intervals = outcome_tracking.get('intervals', {})
    if not intervals:
        # Fallback to summary data
        summary = outcome_tracking.get('summary', {})
        max_gain = summary.get('max_gain_percent', 0)
        if max_gain >= take_profit_pct:
            return (take_profit_pct, 'summary_target_hit')
        elif max_gain <= -stop_loss_pct:
            return (-stop_loss_pct, 'summary_stop_loss')
        else:
            final_gain = summary.get('final_gain_percent', 0)
            return (max(final_gain, -stop_loss_pct), 'summary_no_target')

    # Process intervals chronologically
    sorted_intervals = sorted(intervals.items(), key=lambda x: int(x[0]))

    for interval_sec_str, interval_data in sorted_intervals:
        # Check for interval_low and interval_high (new format)
        has_range_data = ('interval_low_gain_percent' in interval_data and
                         'interval_high_gain_percent' in interval_data and
                         'interval_low_timestamp' in interval_data and
                         'interval_high_timestamp' in interval_data)

        if has_range_data:
            low_gain = interval_data['interval_low_gain_percent']
            high_gain = interval_data['interval_high_gain_percent']
            low_ts = interval_data['interval_low_timestamp']
            high_ts = interval_data['interval_high_timestamp']

            # Determine which happened first
            if high_ts < low_ts:
                # Price went UP first
                if high_gain >= take_profit_pct:
                    return (take_profit_pct, f'target_hit_{interval_sec_str}s')
                if low_gain <= -stop_loss_pct:
                    return (-stop_loss_pct, f'stop_loss_{interval_sec_str}s')
            else:
                # Price went DOWN first
                if low_gain <= -stop_loss_pct:
                    return (-stop_loss_pct, f'stop_loss_{interval_sec_str}s')
                if high_gain >= take_profit_pct:
                    return (take_profit_pct, f'target_hit_{interval_sec_str}s')
        else:
            # Fallback: snapshot data
            gain = interval_data.get('gain_percent', 0)
            if gain >= take_profit_pct:
                return (take_profit_pct, f'snapshot_target_{interval_sec_str}s')
            if gain <= -stop_loss_pct:
                return (-stop_loss_pct, f'snapshot_stop_{interval_sec_str}s')

    # No target or stop hit - use final gain
    summary = outcome_tracking.get('summary', {})
    final_gain = summary.get('final_gain_percent', 0)
    return (max(final_gain, -stop_loss_pct), 'final_no_target')


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
    has_outcomes = False

    for alert in alerts:
        prediction = predictor.predict_single_alert(alert)

        # Parse timestamp
        timestamp_str = alert.get('timestamp', '')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            time_str = timestamp.strftime('%H:%M:%S')
        else:
            time_str = 'N/A'

        # Calculate actual profit if outcome_tracking exists
        actual_profit, outcome_reason = calculate_actual_profit(alert, take_profit_pct=1.5, stop_loss_pct=2.0)
        if actual_profit is not None:
            has_outcomes = True

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
            'distance_vwap': alert.get('distance_from_vwap_percent', 0),
            'actual_profit': actual_profit,
            'outcome_reason': outcome_reason
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

    # BACKTEST ANALYSIS (if outcome data available)
    if has_outcomes:
        print("\n" + "="*80)
        print("BACKTEST ANALYSIS - ACTUAL OUTCOMES AVAILABLE")
        print("="*80)

        # Filter to alerts with outcome data
        df_backtest = df[df['actual_profit'].notna()].copy()

        print(f"\nBacktest Sample: {len(df_backtest)} alerts with outcome data")
        print(f"Take Profit Target: 1.5%")
        print(f"Stop Loss: -2.0%")

        # Calculate actual results
        df_backtest['actual_success'] = df_backtest['actual_profit'] > 0

        # Calculate metrics for MODEL predictions
        model_take = df_backtest[df_backtest['prediction'] == 'TAKE']
        model_skip = df_backtest[df_backtest['prediction'] == 'SKIP']

        if len(model_take) > 0:
            model_win_rate = (model_take['actual_success']).mean()
            model_avg_profit = model_take['actual_profit'].mean()
            model_total_profit = model_take['actual_profit'].sum()
            model_wins = (model_take['actual_profit'] > 0).sum()
            model_losses = (model_take['actual_profit'] < 0).sum()
        else:
            model_win_rate = model_avg_profit = model_total_profit = 0
            model_wins = model_losses = 0

        # Calculate metrics for ALL trades (baseline)
        all_win_rate = df_backtest['actual_success'].mean()
        all_avg_profit = df_backtest['actual_profit'].mean()
        all_total_profit = df_backtest['actual_profit'].sum()

        # Display results
        print("\n" + "="*80)
        print("MODEL PERFORMANCE vs BASELINE")
        print("="*80)
        print(f"{'Metric':<30} {'Model (TAKE only)':<20} {'Baseline (All)':<20} {'Improvement':<15}")
        print("-"*85)
        print(f"{'Trades Taken':<30} {len(model_take):<20} {len(df_backtest):<20} {'':<15}")
        print(f"{'Win Rate':<30} {model_win_rate*100:>18.1f}% {all_win_rate*100:>18.1f}% {(model_win_rate-all_win_rate)*100:>13.1f}%")
        print(f"{'Avg Profit per Trade':<30} {model_avg_profit:>18.2f}% {all_avg_profit:>18.2f}% {model_avg_profit-all_avg_profit:>13.2f}%")
        print(f"{'Total Profit':<30} {model_total_profit:>18.2f}% {all_total_profit:>18.2f}% {model_total_profit-all_total_profit:>13.2f}%")

        # Calculate baseline wins/losses separately to avoid f-string nesting
        baseline_wins = (df_backtest['actual_profit'] > 0).sum()
        baseline_losses = (df_backtest['actual_profit'] < 0).sum()
        print(f"{'Wins / Losses':<30} {f'{model_wins} / {model_losses}':<20} {f'{baseline_wins} / {baseline_losses}':<20} {'':<15}")

        # Confusion Matrix
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)

        # Create binary columns for matrix
        df_backtest['predicted_success'] = df_backtest['prediction'] == 'TAKE'

        # Calculate confusion matrix values
        tp = ((df_backtest['predicted_success'] == True) & (df_backtest['actual_success'] == True)).sum()
        fp = ((df_backtest['predicted_success'] == True) & (df_backtest['actual_success'] == False)).sum()
        tn = ((df_backtest['predicted_success'] == False) & (df_backtest['actual_success'] == False)).sum()
        fn = ((df_backtest['predicted_success'] == False) & (df_backtest['actual_success'] == True)).sum()

        # Calculate metrics
        accuracy = (tp + tn) / len(df_backtest) if len(df_backtest) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"                Predicted")
        print(f"                SKIP    TAKE")
        print(f"Actual  LOSS   {tn:5d}   {fp:5d}  <- False Positives (bad trades)")
        print(f"        WIN    {fn:5d}   {tp:5d}  <- True Positives (good trades)")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  Precision: {precision:.3f} (of TAKE predictions, {precision*100:.1f}% were winners)")
        print(f"  Recall:    {recall:.3f} (of actual winners, we predicted {recall*100:.1f}%)")
        print(f"  F1-Score:  {f1:.3f}")

        # Generate charts
        print("\n" + "="*80)
        print("GENERATING ANALYSIS CHARTS")
        print("="*80)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Backtest Analysis - 1.5% Target / 2.0% Stop', fontsize=16, fontweight='bold')

        # Chart 1: Profit Distribution by Prediction
        ax1 = axes[0, 0]
        take_profits = df_backtest[df_backtest['prediction'] == 'TAKE']['actual_profit']
        skip_profits = df_backtest[df_backtest['prediction'] == 'SKIP']['actual_profit']

        ax1.hist([take_profits, skip_profits], bins=30, label=['TAKE', 'SKIP'],
                alpha=0.7, color=['green', 'red'])
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.axvline(x=1.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Take Profit')
        ax1.axvline(x=-2.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Stop Loss')
        ax1.set_xlabel('Actual Profit (%)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Profit Distribution by Prediction', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Chart 2: Cumulative Profit
        ax2 = axes[0, 1]
        model_trades = df_backtest[df_backtest['prediction'] == 'TAKE'].sort_values('timestamp')
        all_trades = df_backtest.sort_values('timestamp')

        if len(model_trades) > 0:
            model_cumulative = model_trades['actual_profit'].cumsum()
            ax2.plot(range(len(model_cumulative)), model_cumulative.values,
                    label='Model (TAKE only)', color='green', linewidth=2)

        all_cumulative = all_trades['actual_profit'].cumsum()
        ax2.plot(range(len(all_cumulative)), all_cumulative.values,
                label='Baseline (All trades)', color='blue', linewidth=2, alpha=0.7)

        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Trade Number', fontweight='bold')
        ax2.set_ylabel('Cumulative Profit (%)', fontweight='bold')
        ax2.set_title('Cumulative Profit Over Time', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Chart 3: Confusion Matrix Heatmap
        ax3 = axes[1, 0]
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax3,
                   xticklabels=['SKIP', 'TAKE'],
                   yticklabels=['LOSS', 'WIN'],
                   cbar_kws={'label': 'Count'})
        ax3.set_xlabel('Predicted', fontweight='bold')
        ax3.set_ylabel('Actual', fontweight='bold')
        ax3.set_title(f'Confusion Matrix (Accuracy: {accuracy:.1%})', fontweight='bold')

        # Chart 4: Win Rate by Probability Bucket
        ax4 = axes[1, 1]
        df_backtest['prob_bucket'] = pd.cut(df_backtest['probability'],
                                            bins=[0, 0.3, 0.5, 0.7, 1.0],
                                            labels=['<30%', '30-50%', '50-70%', '>70%'])
        bucket_stats = df_backtest.groupby('prob_bucket').agg({
            'actual_success': 'mean',
            'symbol': 'count'
        })

        if len(bucket_stats) > 0:
            x_pos = range(len(bucket_stats))
            bars = ax4.bar(x_pos, bucket_stats['actual_success'] * 100, color='green', alpha=0.7)
            ax4.set_xlabel('Probability Bucket', fontweight='bold')
            ax4.set_ylabel('Actual Win Rate (%)', fontweight='bold')
            ax4.set_title('Calibration: Predicted vs Actual Win Rate', fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(bucket_stats.index)
            ax4.grid(axis='y', alpha=0.3)

            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, bucket_stats['symbol'])):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'n={int(count)}', ha='center', fontsize=9)

        plt.tight_layout()

        # Save chart
        chart_path = Path(args.output).parent / 'batch_backtest_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved backtest analysis chart to: {chart_path}")
        plt.close()

        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)
        print(f"\nKey Finding: Model filtering improves avg profit by {model_avg_profit - all_avg_profit:+.2f}%")
        print(f"Win Rate: {model_win_rate*100:.1f}% (model) vs {all_win_rate*100:.1f}% (baseline)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
