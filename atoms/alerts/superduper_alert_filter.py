"""
Superduper Alert Filter - Advanced Filtering Logic for Superduper Alert Generation

This atom analyzes super alerts to determine which should be promoted to superduper alerts
based on price movement patterns, consolidation analysis, and momentum indicators.
Enhanced with full market data integration for more accurate trend analysis.
"""

import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pytz


class SuperduperAlertData:
    """Data structure for superduper alert analysis with full market data integration."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.super_alerts = []  # List of super alerts for this symbol
        self.price_progression = []  # Price changes over time (from super alerts)
        self.market_data = []  # Full market data for comprehensive trend analysis
        self.timeframe_minutes = 45  # Default timeframe
        self.logger = logging.getLogger(__name__)

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

    def load_market_data(self, date_str: str, current_timestamp: datetime) -> bool:
        """
        Load full market data for comprehensive trend analysis.

        Args:
            date_str: Date string in YYYY-MM-DD format
            current_timestamp: Current alert timestamp for backtesting chronological filtering

        Returns:
            True if market data was successfully loaded
        """
        try:
            # Construct path to market data
            market_data_dir = Path(f"historical_data/{date_str}/market_data")
            if not market_data_dir.exists():
                self.logger.warning(f"Market data directory not found: {market_data_dir}")
                return False

            # Find market data files for this symbol
            pattern = f"{self.symbol}_*.csv"
            data_files = list(market_data_dir.glob(pattern))

            if not data_files:
                self.logger.warning(f"No market data files found for {self.symbol} on {date_str}")
                return False

            # Load and combine all data files
            all_data = []
            for file_path in sorted(data_files):
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        all_data.append(df)
                        self.logger.debug(f"Loaded {len(df)} market data records from {file_path.name}")
                except Exception as e:
                    self.logger.warning(f"Error loading market data from {file_path}: {e}")

            if not all_data:
                self.logger.warning(f"No valid market data loaded for {self.symbol}")
                return False

            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)

            # Ensure timestamp column exists and convert to datetime
            if 'timestamp' not in combined_data.columns:
                self.logger.warning(f"Market data missing timestamp column for {self.symbol}")
                return False

            combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])

            # Sort by timestamp and remove duplicates
            combined_data = combined_data.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)

            # Filter data to only include timestamps before current_timestamp (for backtesting accuracy)
            # Convert current_timestamp to timezone-naive if needed
            if current_timestamp.tzinfo is not None:
                et_tz = pytz.timezone('US/Eastern')
                current_timestamp_naive = current_timestamp.astimezone(et_tz).replace(tzinfo=None)
            else:
                current_timestamp_naive = current_timestamp

            # Filter to data before current timestamp
            mask = combined_data['timestamp'] <= current_timestamp_naive
            filtered_data = combined_data[mask].copy()

            # Convert to list of dictionaries for easier processing
            self.market_data = []
            for _, row in filtered_data.iterrows():
                self.market_data.append({
                    'timestamp': row['timestamp'].to_pydatetime(),
                    'price': float(row.get('close', row.get('price', 0))),
                    'open': float(row.get('open', row.get('price', 0))),
                    'high': float(row.get('high', row.get('price', 0))),
                    'low': float(row.get('low', row.get('price', 0))),
                    'volume': int(row.get('volume', 0)),
                    'vwap': float(row.get('vwap', row.get('price', 0)))
                })

            self.logger.info(f"Loaded {len(self.market_data)} market data points for {self.symbol} (filtered to before {current_timestamp_naive})")
            return True

        except Exception as e:
            self.logger.error(f"Error loading market data for {self.symbol}: {e}")
            return False

    def analyze_trend(self, timeframe_minutes: int = 45) -> Tuple[str, float, Dict[str, Any]]:
        """
        Analyze price trend over the specified timeframe using full market data.

        Args:
            timeframe_minutes: Minutes to look back for trend analysis

        Returns:
            Tuple of (trend_type, strength, analysis_data)
            - trend_type: 'rising', 'consolidating', 'declining', 'insufficient_data'
            - strength: Float from 0.0 to 1.0 indicating trend strength
            - analysis_data: Dictionary with detailed analysis metrics
        """
        # Use market data if available, fallback to super alerts
        data_source = self.market_data if self.market_data else self.price_progression

        if len(data_source) < 2:
            return 'insufficient_data', 0.0, {'reason': f'Less than 2 data points (source: {"market_data" if self.market_data else "super_alerts"})'}

        # Get latest timestamp (from super alerts to maintain superduper alert timing)
        if self.price_progression:
            latest_time = self.price_progression[-1]['timestamp']
        elif data_source:
            latest_time = data_source[-1]['timestamp']
        else:
            return 'insufficient_data', 0.0, {'reason': 'No timestamp reference available'}

        cutoff_time = latest_time - timedelta(minutes=timeframe_minutes)

        # Filter data points within timeframe
        relevant_data = [
            point for point in data_source 
            if point['timestamp'] >= cutoff_time
        ]

        if len(relevant_data) < 2:
            return 'insufficient_data', 0.0, {'reason': f'Insufficient data within timeframe (found {len(relevant_data)} points)'}

        # Calculate trend metrics using enhanced market data
        prices = [point['price'] for point in relevant_data]
        timestamps = [point['timestamp'] for point in relevant_data]

        # Get penetration data from super alerts (only available there)
        penetrations = []
        if self.market_data:
            # For market data, we need to map penetration from super alerts
            super_alert_penetrations = {point['timestamp']: point['penetration'] for point in self.price_progression}
            # Use interpolation or default to 0 for market data points without penetration info
            penetrations = [super_alert_penetrations.get(point['timestamp'], 0) for point in relevant_data]
        else:
            # Using super alert data directly
            penetrations = [point.get('penetration', 0) for point in relevant_data]

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
        using_market_data = bool(self.market_data)
        trend_type, strength = self._classify_trend(
            price_change_percent, price_momentum, price_volatility, 
            penetration_change, avg_penetration, using_market_data
        )

        analysis_data = {
            'timeframe_minutes': timeframe_minutes,
            'data_points': len(relevant_data),
            'data_source': 'market_data' if self.market_data else 'super_alerts',
            'market_data_points': len(self.market_data),
            'super_alert_points': len(self.price_progression),
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
                       avg_penetration: float, using_market_data: bool = False) -> Tuple[str, float]:
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

        # Rising trend criteria - adjusted for market data analysis        
        if using_market_data:
            # For market data: focus on price and momentum (penetration less reliable)
            rising_condition = (price_change_percent > RISING_THRESHOLD and 
                               momentum > MOMENTUM_THRESHOLD)
        else:
            # For super alert data: require penetration change
            rising_condition = (price_change_percent > RISING_THRESHOLD and 
                               momentum > MOMENTUM_THRESHOLD and 
                               penetration_change > 5)

        if rising_condition:
            # Calculate strength based on multiple factors
            price_strength = min(price_change_percent / 5.0, 1.0)  # Max at 5%
            momentum_strength = min(momentum / 0.1, 1.0)  # Max at 0.1% per minute

            if using_market_data:
                # For market data: weight price and momentum more heavily
                volatility_strength = max(0, 1.0 - (volatility / 0.10))  # Lower volatility is better
                strength = (price_strength * 0.4 + momentum_strength * 0.4 + volatility_strength * 0.2)
            else:
                # For super alert data: include penetration
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
            symbol_data.market_data = []  # Clear market data

            for super_alert in all_super_alerts:
                symbol_data.add_super_alert(super_alert)

            # Load full market data for enhanced trend analysis
            latest_timestamp = latest_super_alert.get('timestamp')
            if latest_timestamp:
                try:
                    # Parse the timestamp for market data filtering
                    if latest_timestamp.endswith(('+0000', '-0400', '-0500')):
                        clean_timestamp = latest_timestamp[:-5]
                        parsed_timestamp = datetime.fromisoformat(clean_timestamp)
                    else:
                        parsed_timestamp = datetime.fromisoformat(latest_timestamp.replace('Z', ''))

                    # Load market data up to this timestamp
                    market_data_loaded = symbol_data.load_market_data(date_str, parsed_timestamp)
                    if market_data_loaded:
                        self.logger.debug(f"Enhanced trend analysis for {symbol} using {len(symbol_data.market_data)} market data points")
                    else:
                        self.logger.debug(f"Fallback trend analysis for {symbol} using {len(symbol_data.price_progression)} super alert points")
                except Exception as e:
                    self.logger.warning(f"Error parsing timestamp for market data loading: {e}")

            # Analyze trend (now with enhanced market data if available)
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
            # The trend strength calculation already handles the logic properly
            # For market data vs super alert data differences
            data_source = analysis_data.get('data_source', 'super_alerts')
            price_change = analysis_data.get('price_change_percent', 0)
            momentum = analysis_data.get('price_momentum', 0)

            if data_source == 'market_data':
                # For market data: trust the strength calculation (already accounts for price & momentum)
                return True, f"Strong rising trend (market data): {price_change:.2f}% price change, {momentum:.4f} momentum, strength {strength:.2f}"
            else:
                # For super alert data: also check penetration change
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