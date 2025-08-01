"""
Super Alert Generator - Creates Enhanced Super Alerts

This atom handles the creation of super alerts with enhanced analysis
and risk assessment metrics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pytz

from .super_alert_filter import SuperAlertData


class SuperAlertGenerator:
    """Generates enhanced super alerts from filtered ORB alerts."""

    def __init__(self, super_alerts_dir: Path, test_mode: bool = False):
        """
        Initialize the super alert generator.

        Args:
            super_alerts_dir: Directory to save super alerts
            test_mode: Whether running in test mode
        """
        self.super_alerts_dir = super_alerts_dir
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)

        # Ensure directory exists
        self.super_alerts_dir.mkdir(parents=True, exist_ok=True)

    def create_super_alert(self, alert_data: Dict[str, Any], symbol_info: SuperAlertData, use_original_timestamp: bool = False) -> Optional[Dict[str, Any]]:
        """
        Create a super alert with enhanced analysis.

        Args:
            alert_data: Original alert data
            symbol_info: Symbol information with Signal/Resistance prices
            use_original_timestamp: If True, use original alert timestamp instead of current time

        Returns:
            Super alert dictionary or None if creation failed
        """
        try:
            symbol = alert_data['symbol']
            current_price = alert_data['current_price']
            original_timestamp = alert_data['timestamp']

            # Create ET timestamp for when super alert is generated
            et_tz = pytz.timezone('US/Eastern')
            if use_original_timestamp:
                # For historical processing, use the original alert timestamp
                et_timestamp = original_timestamp
            else:
                # For live monitoring, use current timestamp
                super_alert_time = datetime.now(et_tz)
                et_timestamp = super_alert_time.strftime('%Y-%m-%dT%H:%M:%S%z')

            # Calculate metrics
            penetration = symbol_info.calculate_penetration(current_price)
            range_percent = symbol_info.range_percent

            # Create super alert data
            super_alert = {
                "symbol": symbol,
                "timestamp": et_timestamp,
                "original_alert_timestamp": original_timestamp,
                "alert_type": "super_alert",
                "trigger_condition": "signal_price_reached",
                "original_alert": alert_data,
                "signal_analysis": {
                    "signal_price": symbol_info.signal_price,
                    "resistance_price": symbol_info.resistance_price,
                    "current_price": current_price,
                    "penetration_percent": round(penetration, 2),
                    "range_percent": round(range_percent, 2),
                    "signal_reached": True,
                    "resistance_reached": current_price >= symbol_info.resistance_price
                },
                "metrics": {
                    "signal_to_resistance_range": symbol_info.resistance_price - symbol_info.signal_price,
                    "price_above_signal": current_price - symbol_info.signal_price,
                    "distance_to_resistance": symbol_info.resistance_price - current_price
                },
                "risk_assessment": {
                    "entry_price": symbol_info.signal_price,
                    "target_price": symbol_info.resistance_price,
                    "current_risk_reward": (symbol_info.resistance_price - current_price) / (current_price - symbol_info.signal_price) if current_price > symbol_info.signal_price else 0
                }
            }

            return super_alert

        except Exception as e:
            self.logger.error(f"Error creating super alert for {alert_data.get('symbol', 'unknown')}: {e}")
            return None

    def save_super_alert(self, super_alert: Dict[str, Any]) -> Optional[str]:
        """
        Save super alert to file.

        Args:
            super_alert: Super alert data

        Returns:
            Filename if saved successfully, None otherwise
        """
        try:
            symbol = super_alert['symbol']
            timestamp_str = super_alert['timestamp']

            # Parse timestamp to get filename format
            et_tz = pytz.timezone('US/Eastern')
            try:
                # Handle timezone format in timestamp
                if timestamp_str.endswith(('+0000', '-0400', '-0500')):
                    # Remove timezone suffix and parse
                    clean_timestamp = timestamp_str[:-5]
                    super_alert_time = datetime.fromisoformat(clean_timestamp)
                else:
                    super_alert_time = datetime.fromisoformat(timestamp_str.replace('Z', ''))

                # If no timezone info, assume ET
                if super_alert_time.tzinfo is None:
                    super_alert_time = et_tz.localize(super_alert_time)
                else:
                    super_alert_time = super_alert_time.astimezone(et_tz)

            except Exception:
                # Fallback to current time
                super_alert_time = datetime.now(et_tz)

            # Generate filename
            filename = f"super_alert_{symbol}_{super_alert_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.super_alerts_dir / filename

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(super_alert, f, indent=2)

            self.logger.debug(f"Super alert saved: {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Error saving super alert: {e}")
            return None

    def create_and_save_super_alert(self, alert_data: Dict[str, Any], symbol_info: SuperAlertData, use_original_timestamp: bool = False) -> Optional[str]:
        """
        Create and save a super alert in one operation.

        Args:
            alert_data: Original alert data
            symbol_info: Symbol information
            use_original_timestamp: If True, use original alert timestamp instead of current time

        Returns:
            Filename if successful, None otherwise
        """
        super_alert = self.create_super_alert(alert_data, symbol_info, use_original_timestamp)
        if super_alert is None:
            return None

        filename = self.save_super_alert(super_alert)
        if filename is None:
            return None

        # Display super alert
        self._display_super_alert(super_alert, filename)

        # Track this alert
        symbol_info.alerts_triggered.append(super_alert)

        return filename

    def _display_super_alert(self, super_alert: Dict[str, Any], filename: str) -> None:
        """Display super alert information."""
        try:
            symbol = super_alert['symbol']
            signal_analysis = super_alert['signal_analysis']
            current_price = signal_analysis['current_price']
            signal_price = signal_analysis['signal_price']
            resistance_price = signal_analysis['resistance_price']
            penetration = signal_analysis['penetration_percent']
            range_percent = signal_analysis['range_percent']

            message = (f"🚀 SUPER ALERT: {symbol} @ ${current_price:.2f}\n"
                      f"   Signal: ${signal_price:.2f} ✅ | "
                      f"Resistance: ${resistance_price:.2f}\n"
                      f"   Penetration: {penetration:.1f}% | "
                      f"Range %: {range_percent:.1f}%\n"
                      f"   Saved: {filename}")

            if self.test_mode:
                print(f"[TEST MODE] {message}")
            else:
                print(message)

            self.logger.info(f"Super alert created for {symbol} at ${current_price:.2f}")

        except Exception as e:
            self.logger.error(f"Error displaying super alert: {e}")