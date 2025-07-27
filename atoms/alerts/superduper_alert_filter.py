"""
Superduper Alert Filter - Advanced Filtering Logic for Superduper Alert Generation

This atom analyzes super alerts to determine which should be promoted to superduper alerts
based on price movement patterns, consolidation analysis, and momentum indicators.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pytz


class SuperduperAlertData:
    """Data structure for superduper alert analysis."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.super_alerts = []  # List of super alerts for this symbol
        self.price_progression = []  # Price changes over time
        self.timeframe_minutes = 45  # Default timeframe
        
    def add_super_alert(self, super_alert: Dict[str, Any]) -> None:
        """Add a super alert to the analysis."""
        self.super_alerts.append(super_alert)
        
        # Extract price and timestamp for progression analysis
        price = super_alert.get('signal_analysis', {}).get('current_price')
        timestamp_str = super_alert.get('timestamp')
        
        if price and timestamp_str:
            try:
                # Parse timestamp
                if timestamp_str.endswith(('+0000', '-0400', '-0500')):
                    clean_timestamp = timestamp_str[:-5]
                    timestamp = datetime.fromisoformat(clean_timestamp)
                else:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', ''))
                
                # Add to price progression
                self.price_progression.append({
                    'timestamp': timestamp,
                    'price': price,
                    'penetration': super_alert.get('signal_analysis', {}).get('penetration_percent', 0)
                })
                
                # Sort by timestamp
                self.price_progression.sort(key=lambda x: x['timestamp'])
                
            except Exception as e:
                logging.warning(f"Error parsing timestamp {timestamp_str}: {e}")
    
    def analyze_trend(self, timeframe_minutes: int = 45) -> Tuple[str, float, Dict[str, Any]]:
        """
        Analyze price trend over the specified timeframe.
        
        Args:
            timeframe_minutes: Minutes to look back for trend analysis
            
        Returns:
            Tuple of (trend_type, strength, analysis_data)
            - trend_type: 'rising', 'consolidating', 'declining', 'insufficient_data'
            - strength: Float from 0.0 to 1.0 indicating trend strength
            - analysis_data: Dictionary with detailed analysis metrics
        """
        if len(self.price_progression) < 2:
            return 'insufficient_data', 0.0, {'reason': 'Less than 2 data points'}
        
        # Filter data points within timeframe
        latest_time = self.price_progression[-1]['timestamp']
        cutoff_time = latest_time - timedelta(minutes=timeframe_minutes)
        
        relevant_data = [
            point for point in self.price_progression 
            if point['timestamp'] >= cutoff_time
        ]
        
        if len(relevant_data) < 2:
            return 'insufficient_data', 0.0, {'reason': 'Insufficient data within timeframe'}
        
        # Calculate trend metrics
        prices = [point['price'] for point in relevant_data]
        timestamps = [point['timestamp'] for point in relevant_data]
        penetrations = [point['penetration'] for point in relevant_data]
        
        # Price change analysis
        price_start = prices[0]
        price_end = prices[-1]
        price_change_percent = ((price_end - price_start) / price_start) * 100
        
        # Calculate moving averages and volatility
        price_mean = sum(prices) / len(prices)
        price_volatility = self._calculate_volatility(prices)
        
        # Analyze price momentum (slope)
        time_span_minutes = (timestamps[-1] - timestamps[0]).total_seconds() / 60
        price_momentum = price_change_percent / max(time_span_minutes, 1)  # % per minute
        
        # Penetration trend analysis
        penetration_change = penetrations[-1] - penetrations[0] if penetrations else 0
        avg_penetration = sum(penetrations) / len(penetrations) if penetrations else 0
        
        # Determine trend type and strength
        trend_type, strength = self._classify_trend(
            price_change_percent, price_momentum, price_volatility, 
            penetration_change, avg_penetration
        )
        
        analysis_data = {
            'timeframe_minutes': timeframe_minutes,
            'data_points': len(relevant_data),
            'price_change_percent': round(price_change_percent, 3),
            'price_momentum': round(price_momentum, 6),
            'price_volatility': round(price_volatility, 4),
            'penetration_change': round(penetration_change, 2),
            'avg_penetration': round(avg_penetration, 2),
            'price_start': price_start,
            'price_end': price_end,
            'time_span_minutes': round(time_span_minutes, 1)
        }
        
        return trend_type, strength, analysis_data
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility as coefficient of variation."""
        if len(prices) < 2:
            return 0.0
        
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        
        return (std_dev / mean_price) if mean_price > 0 else 0.0
    
    def _classify_trend(self, price_change_percent: float, momentum: float, 
                       volatility: float, penetration_change: float, 
                       avg_penetration: float) -> Tuple[str, float]:
        """
        Classify trend type and calculate strength.
        
        Returns:
            Tuple of (trend_type, strength)
        """
        # Thresholds for classification
        RISING_THRESHOLD = 1.0  # 1% price increase
        CONSOLIDATING_VOLATILITY_THRESHOLD = 0.05  # 5% volatility
        MOMENTUM_THRESHOLD = 0.02  # 0.02% per minute
        
        strength = 0.0
        
        # Rising trend criteria
        if (price_change_percent > RISING_THRESHOLD and 
            momentum > MOMENTUM_THRESHOLD and 
            penetration_change > 5):  # Increasing penetration
            
            # Calculate strength based on multiple factors
            price_strength = min(price_change_percent / 5.0, 1.0)  # Max at 5%
            momentum_strength = min(momentum / 0.1, 1.0)  # Max at 0.1% per minute
            penetration_strength = min(penetration_change / 20.0, 1.0)  # Max at 20%
            
            strength = (price_strength + momentum_strength + penetration_strength) / 3.0
            return 'rising', strength
        
        # Consolidating trend criteria
        elif (abs(price_change_percent) < RISING_THRESHOLD and 
              volatility < CONSOLIDATING_VOLATILITY_THRESHOLD and
              avg_penetration > 10):  # Sustained penetration
            
            # Consolidation strength based on stability and penetration
            stability_strength = max(0, 1.0 - (volatility / CONSOLIDATING_VOLATILITY_THRESHOLD))
            penetration_strength = min(avg_penetration / 50.0, 1.0)  # Max at 50%
            
            strength = (stability_strength + penetration_strength) / 2.0
            return 'consolidating', strength
        
        # Declining trend
        elif price_change_percent < -RISING_THRESHOLD:
            strength = min(abs(price_change_percent) / 5.0, 1.0)
            return 'declining', strength
        
        # Default to insufficient data if no clear pattern
        return 'insufficient_data', 0.0


class SuperduperAlertFilter:
    """Filters super alerts for superduper alert generation."""
    
    def __init__(self, timeframe_minutes: int = 45):
        """
        Initialize the superduper alert filter.
        
        Args:
            timeframe_minutes: Time window for trend analysis (default 45)
        """
        self.timeframe_minutes = timeframe_minutes
        self.logger = logging.getLogger(__name__)
        self.symbol_data = {}  # Dict of symbol -> SuperduperAlertData
    
    def should_create_superduper_alert(self, super_alert_path: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Determine if a super alert should be promoted to a superduper alert.
        
        Args:
            super_alert_path: Path to the latest super alert file
            
        Returns:
            Tuple of (should_create, filter_reason, analysis_data)
            - should_create: True if superduper alert should be created
            - filter_reason: Reason for filtering if should_create is False
            - analysis_data: Detailed analysis data for logging/debugging
        """
        try:
            # Load the latest super alert
            with open(super_alert_path, 'r') as f:
                latest_super_alert = json.load(f)
            
            symbol = latest_super_alert.get('symbol')
            if not symbol:
                return False, "Invalid super alert data - no symbol", None
            
            # Extract date from path to find all super alerts for this symbol on this day
            date_str = self._extract_date_from_path(super_alert_path)
            if not date_str:
                return False, "Could not extract date from path", None
            
            # Load all super alerts for this symbol on this day
            all_super_alerts = self._load_symbol_super_alerts(symbol, date_str, super_alert_path)
            if len(all_super_alerts) < 2:
                return False, f"Insufficient super alerts for {symbol} (need at least 2, found {len(all_super_alerts)})", None
            
            # Create or update symbol data
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = SuperduperAlertData(symbol)
            
            symbol_data = self.symbol_data[symbol]
            
            # Clear previous data and reload with current day's alerts
            symbol_data.super_alerts = []
            symbol_data.price_progression = []
            
            for super_alert in all_super_alerts:
                symbol_data.add_super_alert(super_alert)
            
            # Analyze trend
            trend_type, strength, analysis_data = symbol_data.analyze_trend(self.timeframe_minutes)
            
            # Decision logic for superduper alerts
            should_create, reason = self._evaluate_superduper_criteria(
                trend_type, strength, analysis_data, latest_super_alert
            )
            
            return should_create, reason, analysis_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing super alert {super_alert_path}: {e}")
            return False, f"Analysis error: {str(e)}", None
    
    def _extract_date_from_path(self, file_path: str) -> Optional[str]:
        """Extract date string from super alert file path."""
        path = Path(file_path)
        
        # Look for date in path components (e.g., historical_data/2025-07-23/...)
        for part in path.parts:
            if len(part) == 10 and part.count('-') == 2:
                try:
                    datetime.strptime(part, '%Y-%m-%d')
                    return part
                except ValueError:
                    continue
        
        return None
    
    def _extract_timestamp_from_path(self, file_path: str) -> Optional[datetime]:
        """Extract timestamp from super alert file path for backtesting chronological filtering."""
        try:
            # Super alert filenames have format: super_alert_{symbol}_{YYYYMMDD_HHMMSS}.json
            # Example: super_alert_CLSD_20250723_145400.json
            filename = Path(file_path).name
            
            # Extract the timestamp part after the last underscore and before .json
            parts = filename.replace('.json', '').split('_')
            if len(parts) >= 4:
                # Get date and time parts (last two parts)
                date_part = parts[-2]  # YYYYMMDD
                time_part = parts[-1]  # HHMMSS
                
                if len(date_part) == 8 and len(time_part) == 6:
                    # Parse the timestamp
                    timestamp_str = f"{date_part}_{time_part}"
                    return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
        except Exception as e:
            self.logger.warning(f"Error extracting timestamp from {file_path}: {e}")
        
        return None
    
    def _load_symbol_super_alerts(self, symbol: str, date_str: str, current_path: str) -> List[Dict[str, Any]]:
        """Load all super alerts for a symbol on a specific date, only including files prior to current timestamp."""
        super_alerts = []
        
        # Extract timestamp from current file for backtesting filter
        current_timestamp = self._extract_timestamp_from_path(current_path)
        if not current_timestamp:
            self.logger.warning(f"Could not extract timestamp from {current_path}")
            return super_alerts
        
        # Construct directory path
        current_path_obj = Path(current_path)
        super_alerts_dir = current_path_obj.parent
        
        try:
            # Find all super alert files for this symbol
            pattern = f"super_alert_{symbol}_*.json"
            for file_path in super_alerts_dir.glob(pattern):
                try:
                    # Extract timestamp from this file
                    file_timestamp = self._extract_timestamp_from_path(str(file_path))
                    
                    # Only include files with timestamps before or equal to current file
                    # This ensures backtesting only uses historically available data
                    if file_timestamp and file_timestamp <= current_timestamp:
                        with open(file_path, 'r') as f:
                            super_alert = json.load(f)
                            super_alerts.append(super_alert)
                    else:
                        self.logger.debug(f"Skipping future file for backtesting: {file_path}")
                        
                except Exception as e:
                    self.logger.warning(f"Error loading {file_path}: {e}")
            
            # Sort by timestamp
            super_alerts.sort(key=lambda x: x.get('timestamp', ''))
            
        except Exception as e:
            self.logger.error(f"Error loading super alerts for {symbol}: {e}")
        
        return super_alerts
    
    def _evaluate_superduper_criteria(self, trend_type: str, strength: float, 
                                    analysis_data: Dict[str, Any], 
                                    latest_super_alert: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate if the trend analysis meets superduper alert criteria.
        
        Args:
            trend_type: Type of trend ('rising', 'consolidating', 'declining', 'insufficient_data')
            strength: Trend strength (0.0 to 1.0)
            analysis_data: Detailed analysis metrics
            latest_super_alert: The latest super alert data
            
        Returns:
            Tuple of (should_create, reason)
        """
        symbol = latest_super_alert.get('symbol', 'UNKNOWN')
        current_price = latest_super_alert.get('signal_analysis', {}).get('current_price', 0)
        penetration = latest_super_alert.get('signal_analysis', {}).get('penetration_percent', 0)
        
        # Minimum strength threshold
        MIN_STRENGTH = 0.3
        MIN_PENETRATION = 15.0  # 15% penetration required
        MIN_DATA_POINTS = 2
        
        # Check basic requirements
        if analysis_data.get('data_points', 0) < MIN_DATA_POINTS:
            return False, f"Insufficient data points: {analysis_data.get('data_points', 0)}"
        
        if penetration < MIN_PENETRATION:
            return False, f"Penetration too low: {penetration:.1f}% (need {MIN_PENETRATION:.1f}%)"
        
        if strength < MIN_STRENGTH:
            return False, f"Trend strength too weak: {strength:.2f} (need {MIN_STRENGTH:.2f})"
        
        # Criteria for different trend types
        if trend_type == 'rising':
            # Rising trend: strong upward momentum with increasing penetration
            price_change = analysis_data.get('price_change_percent', 0)
            momentum = analysis_data.get('price_momentum', 0)
            penetration_change = analysis_data.get('penetration_change', 0)
            
            if price_change > 2.0 and momentum > 0.03 and penetration_change > 10:
                return True, f"Strong rising trend: {price_change:.2f}% price change, {penetration_change:.1f}% penetration increase"
            else:
                return False, f"Rising trend too weak: price {price_change:.2f}%, momentum {momentum:.4f}, penetration change {penetration_change:.1f}%"
        
        elif trend_type == 'consolidating':
            # Consolidating: stable price with sustained high penetration
            avg_penetration = analysis_data.get('avg_penetration', 0)
            volatility = analysis_data.get('price_volatility', 0)
            
            if avg_penetration > 25.0 and volatility < 0.03:
                return True, f"Strong consolidation: {avg_penetration:.1f}% avg penetration, {volatility:.3f} volatility"
            else:
                return False, f"Consolidation too weak: {avg_penetration:.1f}% penetration, {volatility:.3f} volatility"
        
        elif trend_type == 'declining':
            return False, f"Declining trend not suitable for superduper alerts"
        
        else:
            return False, f"Trend type '{trend_type}' not suitable for superduper alerts"