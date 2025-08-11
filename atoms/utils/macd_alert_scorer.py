"""
MACD-Based Alert Scoring System for Superduper Alerts

This module provides a scoring system that classifies superduper alerts based solely on MACD conditions
at the time of the alert. The scoring uses a red/yellow/green system to indicate MACD strength.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from datetime import datetime
import pytz
from .calculate_macd import calculate_macd


class MACDAlertScorer:
    """
    MACD-based scoring system for superduper alerts.
    
    Methodology:
    
    GREEN (Excellent MACD Conditions):
    - MACD Line > Signal Line (bullish crossover active)
    - MACD Line > 0 (strong upward momentum)
    - Histogram > 0 (momentum increasing)  
    - MACD Rising (5-period lookback shows upward trend)
    - Score: 4/4 or 3/4 conditions met
    
    YELLOW (Moderate MACD Conditions):
    - MACD Line > Signal Line OR MACD > 0 (at least one key condition)
    - Mixed histogram signals acceptable
    - Score: 2/4 conditions met
    
    RED (Poor MACD Conditions):
    - MACD Line < Signal Line (bearish)
    - MACD Line < 0 (negative momentum)
    - Histogram < 0 (momentum decreasing)
    - MACD Falling (declining trend)
    - Score: 0/4 or 1/4 conditions met
    
    The scoring prioritizes:
    1. MACD vs Signal relationship (most important for entry timing)
    2. MACD vs zero line (overall momentum direction)
    3. Histogram direction (momentum acceleration/deceleration)
    4. MACD trend (sustainability of momentum)
    """
    
    def __init__(self):
        """Initialize the MACD Alert Scorer."""
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.lookback_periods = 5  # For trend analysis
        
    def calculate_macd_conditions(self, df: pd.DataFrame, alert_timestamp: datetime) -> Dict[str, Any]:
        """
        Calculate MACD conditions at the time of an alert.
        
        Args:
            df: DataFrame with market data including timestamp and OHLCV
            alert_timestamp: Timestamp of the alert to analyze
            
        Returns:
            Dictionary with MACD analysis and conditions
        """
        try:
            # Calculate MACD for the entire dataset
            macd_success, macd_values = calculate_macd(
                df, 
                fast_length=self.macd_fast,
                slow_length=self.macd_slow, 
                signal_length=self.macd_signal,
                source='close'
            )
            
            if not macd_success:
                return {
                    'error': 'MACD calculation failed',
                    'is_valid': False
                }
            
            # Convert alert timestamp to match DataFrame timezone
            if alert_timestamp.tzinfo is None:
                et_tz = pytz.timezone('America/New_York')
                alert_timestamp = et_tz.localize(alert_timestamp)
            
            # Find the closest timestamp in the DataFrame
            df_timestamps = pd.to_datetime(df['timestamp'])
            if df_timestamps.dt.tz is None:
                et_tz = pytz.timezone('America/New_York')
                df_timestamps = df_timestamps.dt.tz_localize(et_tz)
                
            time_diff = abs(df_timestamps - alert_timestamp)
            closest_idx = time_diff.argmin()
            
            if time_diff.iloc[closest_idx].total_seconds() > 300:  # More than 5 minutes away
                return {
                    'error': f'No market data within 5 minutes of alert time {alert_timestamp}',
                    'is_valid': False
                }
            
            # Get MACD values at the alert timestamp
            macd_line = float(macd_values['macd'].iloc[closest_idx])
            signal_line = float(macd_values['signal'].iloc[closest_idx])
            histogram = float(macd_values['histogram'].iloc[closest_idx])
            
            # Calculate MACD trend using lookback periods
            macd_trend = self._calculate_macd_trend(macd_values['macd'], closest_idx)
            
            # Evaluate individual conditions
            conditions = {
                'macd_above_signal': macd_line > signal_line,
                'macd_above_zero': macd_line > 0,
                'histogram_positive': histogram > 0,
                'macd_rising': macd_trend == 'rising'
            }
            
            # Calculate bullish score (0-4)
            bullish_score = sum(conditions.values())
            
            return {
                'is_valid': True,
                'timestamp': alert_timestamp,
                'closest_data_time': df_timestamps.iloc[closest_idx],
                'time_diff_seconds': time_diff.iloc[closest_idx].total_seconds(),
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram,
                'macd_trend': macd_trend,
                'conditions': conditions,
                'bullish_score': bullish_score,
                'price': float(df.iloc[closest_idx]['close']) if 'close' in df.columns else None
            }
            
        except Exception as e:
            return {
                'error': f'Error analyzing MACD: {str(e)}',
                'is_valid': False
            }
    
    def _calculate_macd_trend(self, macd_series: pd.Series, current_idx: int) -> str:
        """
        Calculate MACD trend direction using lookback periods.
        
        Args:
            macd_series: MACD line values
            current_idx: Current index position
            
        Returns:
            Trend direction: 'rising', 'falling', or 'neutral'
        """
        if current_idx < self.lookback_periods:
            return 'neutral'  # Not enough data for trend
        
        current_macd = macd_series.iloc[current_idx]
        previous_macd = macd_series.iloc[current_idx - self.lookback_periods]
        
        # Use 5% threshold to determine significant trend
        change_threshold = 0.05
        
        if current_macd > previous_macd * (1 + change_threshold):
            return 'rising'
        elif current_macd < previous_macd * (1 - change_threshold):
            return 'falling'
        else:
            return 'neutral'
    
    def score_alert(self, macd_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score an alert based on MACD analysis using red/yellow/green system.
        
        Args:
            macd_analysis: Result from calculate_macd_conditions()
            
        Returns:
            Dictionary with score, color, and reasoning
        """
        if not macd_analysis.get('is_valid', False):
            return {
                'color': 'red',
                'score': 0,
                'confidence': 'low',
                'reasoning': f"Invalid MACD analysis: {macd_analysis.get('error', 'Unknown error')}",
                'conditions_met': 0,
                'total_conditions': 4
            }
        
        bullish_score = macd_analysis['bullish_score']
        conditions = macd_analysis['conditions']
        
        # Determine color based on bullish score
        if bullish_score >= 3:
            color = 'green'
            confidence = 'high' if bullish_score == 4 else 'medium-high'
        elif bullish_score == 2:
            color = 'yellow'  
            confidence = 'medium'
        else:
            color = 'red'
            confidence = 'low'
        
        # Generate reasoning based on conditions
        reasoning_parts = []
        
        if conditions['macd_above_signal']:
            reasoning_parts.append("MACD > Signal (bullish crossover active)")
        else:
            reasoning_parts.append("MACD < Signal (bearish crossover)")
            
        if conditions['macd_above_zero']:
            reasoning_parts.append("MACD > 0 (positive momentum)")
        else:
            reasoning_parts.append("MACD < 0 (negative momentum)")
            
        if conditions['histogram_positive']:
            reasoning_parts.append("Histogram > 0 (momentum increasing)")
        else:
            reasoning_parts.append("Histogram < 0 (momentum decreasing)")
            
        if conditions['macd_rising']:
            reasoning_parts.append("MACD trending up")
        elif macd_analysis['macd_trend'] == 'falling':
            reasoning_parts.append("MACD trending down")
        else:
            reasoning_parts.append("MACD trend neutral")
        
        reasoning = f"Score {bullish_score}/4: " + " | ".join(reasoning_parts)
        
        return {
            'color': color,
            'score': bullish_score,
            'confidence': confidence,
            'reasoning': reasoning,
            'conditions_met': bullish_score,
            'total_conditions': 4,
            'macd_values': {
                'macd_line': macd_analysis['macd_line'],
                'signal_line': macd_analysis['signal_line'],
                'histogram': macd_analysis['histogram'],
                'trend': macd_analysis['macd_trend']
            }
        }
    
    def score_alerts_batch(self, df: pd.DataFrame, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score multiple alerts in batch for efficiency.
        
        Args:
            df: DataFrame with market data
            alerts: List of alert dictionaries with 'timestamp_dt' field
            
        Returns:
            List of alerts with added 'macd_score' field
        """
        scored_alerts = []
        
        print(f"Scoring {len(alerts)} alerts using MACD analysis...")
        
        for i, alert in enumerate(alerts):
            alert_copy = alert.copy()
            
            if 'timestamp_dt' not in alert:
                alert_copy['macd_score'] = {
                    'color': 'red',
                    'score': 0,
                    'reasoning': 'No timestamp found in alert data',
                    'error': True
                }
                scored_alerts.append(alert_copy)
                continue
            
            # Analyze MACD conditions at alert time
            macd_analysis = self.calculate_macd_conditions(df, alert['timestamp_dt'])
            
            # Score the alert
            score_result = self.score_alert(macd_analysis)
            
            # Add detailed analysis for debugging
            score_result['macd_analysis'] = macd_analysis
            
            alert_copy['macd_score'] = score_result
            scored_alerts.append(alert_copy)
            
            # Print progress for large batches
            if (i + 1) % 5 == 0 or (i + 1) == len(alerts):
                print(f"  Processed {i + 1}/{len(alerts)} alerts")
        
        # Print summary statistics
        colors = [alert['macd_score']['color'] for alert in scored_alerts if 'macd_score' in alert]
        green_count = colors.count('green')
        yellow_count = colors.count('yellow') 
        red_count = colors.count('red')
        
        print(f"MACD Scoring Summary:")
        print(f"  🟢 Green: {green_count} alerts ({green_count/len(colors)*100:.1f}%)")
        print(f"  🟡 Yellow: {yellow_count} alerts ({yellow_count/len(colors)*100:.1f}%)")
        print(f"  🔴 Red: {red_count} alerts ({red_count/len(colors)*100:.1f}%)")
        
        return scored_alerts


# Convenience function for quick scoring
def score_alerts_with_macd(df: pd.DataFrame, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to score alerts using MACD analysis.
    
    Args:
        df: DataFrame with market data
        alerts: List of alert dictionaries
        
    Returns:
        List of alerts with MACD scoring added
    """
    scorer = MACDAlertScorer()
    return scorer.score_alerts_batch(df, alerts)


# Example usage and testing
if __name__ == "__main__":
    print("MACD Alert Scoring System")
    print("=" * 50)
    print("\nScoring Methodology:")
    print("🟢 GREEN (3-4/4): Excellent MACD conditions")
    print("   - MACD > Signal, MACD > 0, Histogram > 0, MACD Rising")
    print("🟡 YELLOW (2/4): Moderate MACD conditions")  
    print("   - Mixed signals, at least MACD > Signal OR MACD > 0")
    print("🔴 RED (0-1/4): Poor MACD conditions")
    print("   - Bearish MACD, negative momentum")
    print("\nConditions Evaluated:")
    print("1. MACD Line > Signal Line (bullish crossover)")
    print("2. MACD Line > 0 (positive momentum)")
    print("3. Histogram > 0 (momentum increasing)")
    print("4. MACD Rising (5-period upward trend)")
    print("\nUse with: scorer = MACDAlertScorer()")
    print("          scored_alerts = scorer.score_alerts_batch(df, alerts)")