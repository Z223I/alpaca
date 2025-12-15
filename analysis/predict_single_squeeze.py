#!/usr/bin/env python3
"""
Predict outcome for a single squeeze alert using trained XGBoost model.

Usage:
    python3 analysis/predict_single_squeeze.py

Author: Predictive Analytics Module
Date: 2025-12-15
"""
import xgboost as xgb
import json
import numpy as np
from pathlib import Path


class SqueezePredictor:
    def __init__(self, model_path='analysis/xgboost_model_1.5pct.json',
                 info_path='analysis/xgboost_model_1.5pct_info.json'):
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

        print(f"✓ Loaded model for {self.gain_threshold}% gain target")
        print(f"  Test Accuracy: {self.performance['test_accuracy']:.3f}")
        print(f"  Precision: {self.performance['precision']:.3f}")
        print(f"  F1-Score: {self.performance['f1_score']:.3f}")
        print(f"\n  Required features: {len(self.feature_names)}")

    def predict_single_alert(self, alert_data: dict) -> dict:
        """
        Predict outcome for a single squeeze alert.

        Args:
            alert_data: Dictionary with squeeze alert features

        Required features:
            - ema_spread_pct: (ema_9 - ema_21) / price * 100
            - price_category: '<$2', '$2-5', '$5-10', '$10-20', '$20-40', '>$40'
            - macd_histogram: MACD histogram value
            - market_session: 'extended', 'early', 'mid_day'
            - squeeze_number_today: Count of squeezes today
            - minutes_since_last_squeeze: Time since last squeeze
            - window_volume_vs_1min_avg: Volume surge ratio
            - distance_from_vwap_percent: Distance from VWAP (%)
            - day_gain: Stock gain today (%)
            - spy_percent_change_concurrent: SPY correlation (%)
            - spread_percent: Bid-ask spread (%)

        Returns:
            Dictionary with prediction and probability
        """

        # Encode categorical variables
        market_session_map = {'extended': 0, 'early': 1, 'mid_day': 2, 'unknown': 3}
        price_category_map = {'<$2': 0, '$2-5': 1, '$5-10': 2, '$10-20': 3, '$20-40': 4, '>$40': 5, 'unknown': 6}

        # Build feature vector in correct order
        features = []

        # 1. ema_spread_pct
        ema_spread_pct = alert_data.get('ema_spread_pct')
        if ema_spread_pct is None:
            # Handle missing: impute with 0 (median from training data)
            ema_spread_pct = 0.0
            ema_missing = 1
        else:
            ema_missing = 0
        features.append(ema_spread_pct)

        # 2. price_category_encoded
        price_cat = alert_data.get('price_category', 'unknown')
        features.append(price_category_map.get(price_cat, 6))

        # 3. macd_histogram (impute with 0 if missing)
        features.append(alert_data.get('macd_histogram', 0.0))

        # 4. market_session_encoded
        market_session = alert_data.get('market_session', 'unknown')
        features.append(market_session_map.get(market_session, 3))

        # 5. squeeze_number_today
        features.append(alert_data.get('squeeze_number_today', 1))

        # 6. minutes_since_last_squeeze
        features.append(alert_data.get('minutes_since_last_squeeze', 5.0))

        # 7. window_volume_vs_1min_avg
        features.append(alert_data.get('window_volume_vs_1min_avg', 1.0))

        # 8. distance_from_vwap_percent
        features.append(alert_data.get('distance_from_vwap_percent', 0.0))

        # 9. day_gain
        features.append(alert_data.get('day_gain', 0.0))

        # 10. spy_percent_change_concurrent
        features.append(alert_data.get('spy_percent_change_concurrent', 0.0))

        # 11. spread_percent
        features.append(alert_data.get('spread_percent', 0.5))

        # 12. ema_spread_pct_missing (indicator variable)
        features.append(ema_missing)

        # Convert to numpy array and reshape for single prediction
        X = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = self.model.predict(X)[0]  # 0 or 1
        probability = self.model.predict_proba(X)[0, 1]  # Probability of success

        return {
            'will_succeed': bool(prediction),
            'success_probability': float(probability),
            'confidence': 'HIGH' if probability > 0.7 or probability < 0.3 else 'MEDIUM',
            'recommendation': 'TAKE TRADE' if prediction == 1 else 'SKIP',
            'gain_target': f'{self.gain_threshold}%'
        }


# Example usage
if __name__ == "__main__":

    # Load predictor
    predictor = SqueezePredictor()

    # Example squeeze alert data
    alert = {
        'symbol': 'AAPL',
        'ema_spread_pct': 0.25,          # EMA spread as % of price
        'price_category': '$5-10',        # Price bin (best performing!)
        'macd_histogram': 0.05,          # MACD momentum
        'market_session': 'early',       # Time of day (9:30-10:30 is best)
        'squeeze_number_today': 5,       # 5th squeeze today
        'minutes_since_last_squeeze': 3.5,
        'window_volume_vs_1min_avg': 250.0,  # 250x normal volume
        'distance_from_vwap_percent': 1.5,   # 1.5% above VWAP
        'day_gain': 15.0,                # Stock up 15% today
        'spy_percent_change_concurrent': -0.2,  # SPY down 0.2%
        'spread_percent': 0.8            # 0.8% bid-ask spread
    }

    # Get prediction
    result = predictor.predict_single_alert(alert)

    # Display results
    print("\n" + "="*60)
    print("SQUEEZE PREDICTION")
    print("="*60)
    print(f"Symbol: {alert['symbol']}")
    print(f"Target: {result['gain_target']} gain")
    print(f"\nPrediction: {result['recommendation']}")
    print(f"Will Succeed: {result['will_succeed']}")
    print(f"Success Probability: {result['success_probability']:.1%}")
    print(f"Confidence: {result['confidence']}")
    print("="*60)

    # Trade decision logic
    if result['success_probability'] >= 0.6:
        print("\n✅ TRADE SIGNAL: High probability of success")
        print(f"   Expected to reach {result['gain_target']} gain")
        print("\n   Best practices:")
        print("   - Use $5-10 price range stocks (58.7% win rate)")
        print("   - Trade during first hour (9:30-10:30 AM)")
        print("   - Set stop loss at -2%")
    elif result['success_probability'] >= 0.4:
        print("\n⚠️  CAUTION: Moderate probability")
        print("   Consider other factors before trading:")
        print("   - Is stock in $5-10 range?")
        print("   - Is it during first hour after open?")
        print("   - Is volume surge >250x?")
    else:
        print("\n❌ SKIP: Low probability of success")
        print("   Not recommended for trading")
        print("\n   Negative factors may include:")
        print("   - Stock price >$20 (poor performance)")
        print("   - Late in trading day")
        print("   - Low volume surge")
