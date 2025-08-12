"""
Super Alert Filter - Filtering Logic for Super Alert Generation

This atom encapsulates the filtering logic used to determine which ORB alerts
should be promoted to super alerts based on technical criteria.
"""

from typing import Dict, Optional, Any
import logging


class SuperAlertData:
    """Data structure for super alert information."""

    def __init__(self, symbol: str, signal_price: float, resistance_price: float):
        self.symbol = symbol
        self.signal_price = signal_price
        self.resistance_price = resistance_price
        self.range_percent = (resistance_price / signal_price) if signal_price > 0 else 0
        self.alerts_triggered = []

    def calculate_penetration(self, current_price: float) -> float:
        """Calculate penetration percentage into Signal-to-Resistance range."""
        if current_price < self.signal_price:
            return 0.0

        range_size = self.resistance_price - self.signal_price
        if range_size <= 0:
            return 0.0

        penetration = (current_price - self.signal_price) / range_size
        return min(penetration * 100, 100.0)  # Cap at 100%


class SuperAlertFilter:
    """Filters ORB alerts for super alert generation."""

    def __init__(self, symbol_data: Dict[str, SuperAlertData]):
        """
        Initialize the super alert filter.

        Args:
            symbol_data: Dictionary mapping symbols to SuperAlertData objects
        """
        self.symbol_data = symbol_data
        self.logger = logging.getLogger(__name__)
        self.filtered_alerts = set()  # Track filtered alerts

    def should_create_super_alert(self, alert_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Determine if an alert should be promoted to a super alert.

        Args:
            alert_data: Alert data dictionary

        Returns:
            Tuple of (should_create, filter_reason)
            - should_create: True if super alert should be created
            - filter_reason: Reason for filtering if should_create is False
        """
        symbol = alert_data.get('symbol')
        current_price = alert_data.get('current_price')

        # Basic validation
        if not symbol or current_price is None:
            return False, "Invalid alert data"

        # Filter 1: Symbol must have Signal/Resistance data
        if symbol not in self.symbol_data:
            return False, f"No signal data for {symbol}"

        # Filter 2: Red candle filter
        passes_filter, filter_reason = self._passes_red_candle_filter(alert_data)
        if not passes_filter:
            return False, filter_reason

        # Filter 3: Candlestick low vs EMA9 filter (for bullish alerts only)
        breakout_type = alert_data.get('breakout_type', '').lower()
        if breakout_type == 'bullish_breakout':
            if not self._passes_ema9_low_filter(alert_data):
                return False, f"{symbol}: Current candlestick low below EMA9"

        # Filter 4: Signal price threshold
        symbol_info = self.symbol_data[symbol]
        if current_price < symbol_info.signal_price:
            return False, f"Price ${current_price:.2f} below Signal ${symbol_info.signal_price:.2f}"

        return True, None

    def _passes_ema9_low_filter(self, alert_data: Dict[str, Any]) -> bool:
        """
        Check if bullish alert passes the EMA9 low filter.

        Args:
            alert_data: Alert data dictionary

        Returns:
            True if alert passes the filter
        """
        symbol = alert_data.get('symbol')
        ema_9 = alert_data.get('ema_9')

        # Get the current candlestick low from the original alert data
        original_alert = alert_data.get('original_alert', alert_data)
        current_low = original_alert.get('low_price')  # Current candlestick low
        if current_low is None:
            # Fallback to looking for 'low' field (legacy) or 'current_low' field directly
            current_low = original_alert.get('low')
            if current_low is None:
                current_low = alert_data.get('current_low')

        if ema_9 is not None and current_low is not None:
            if current_low < ema_9:
                self.filtered_alerts.add(f"{symbol}_{alert_data.get('timestamp', 'unknown')}")
                self.logger.info(f"🚫 Filtered bullish alert for {symbol}: Current candlestick low ${current_low:.2f} below EMA9 ${ema_9:.2f}")
                return False
            else:
                self.logger.debug(f"✅ Allowing bullish alert for {symbol}: Current candlestick low ${current_low:.2f} above EMA9 ${ema_9:.2f}")
                return True
        elif ema_9 is None:
            self.logger.warning(f"No EMA9 data available for {symbol} low filter, allowing alert")
            return True
        elif current_low is None:
            self.logger.warning(f"No current candlestick low data available for {symbol} EMA9 filter, allowing alert")
            return True

        return True

    def _passes_red_candle_filter(self, alert_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Check if alert passes the red candle filter (reject red candles where open > close).

        Args:
            alert_data: Alert data dictionary

        Returns:
            Tuple of (passes_filter, filter_reason)
            - passes_filter: True if alert passes the filter (not a red candle)
            - filter_reason: Specific reason for filtering if passes_filter is False
        """
        symbol = alert_data.get('symbol')
        
        # Get the current candlestick open and close prices from the original alert data
        original_alert = alert_data.get('original_alert', alert_data)
        
        # Try multiple field names for open price
        open_price = original_alert.get('open_price')
        if open_price is None:
            open_price = original_alert.get('open')
        if open_price is None:
            open_price = alert_data.get('open_price')
        if open_price is None:
            open_price = alert_data.get('open')
            
        # Try multiple field names for close price  
        close_price = original_alert.get('close_price')
        if close_price is None:
            close_price = original_alert.get('close')
        if close_price is None:
            close_price = alert_data.get('close_price')
        if close_price is None:
            close_price = alert_data.get('close')

        if open_price is not None and close_price is not None:
            # Filter out red candles (open > close)
            if open_price > close_price:
                self.filtered_alerts.add(f"{symbol}_{alert_data.get('timestamp', 'unknown')}")
                reason = f"{symbol}: Red candlestick filtered (Open ${open_price:.4f} > Close ${close_price:.4f})"
                self.logger.info(f"🚫 Filtered red candle alert for {symbol}: Open ${open_price:.4f} > Close ${close_price:.4f}")
                return False, reason
            else:
                self.logger.debug(f"✅ Allowing green/doji candle for {symbol}: Open ${open_price:.4f} <= Close ${close_price:.4f}")
                return True, None
        elif open_price is None:
            self.filtered_alerts.add(f"{symbol}_{alert_data.get('timestamp', 'unknown')}")
            reason = f"{symbol}: Missing open price data"
            self.logger.info(f"🚫 Filtered alert for {symbol}: No open price data available for red candle filter")
            return False, reason
        elif close_price is None:
            self.filtered_alerts.add(f"{symbol}_{alert_data.get('timestamp', 'unknown')}")
            reason = f"{symbol}: Missing close price data"
            self.logger.info(f"🚫 Filtered alert for {symbol}: No close price data available for red candle filter")
            return False, reason

        return False, f"{symbol}: Unknown OHLC data issue"

    def get_symbol_info(self, symbol: str) -> Optional[SuperAlertData]:
        """Get SuperAlertData for a symbol."""
        return self.symbol_data.get(symbol)

    def get_filtered_count(self) -> int:
        """Get count of filtered alerts."""
        return len(self.filtered_alerts)