"""
Symbol Data Loader - Loads Symbol Data with Signal/Resistance Prices

This atom handles loading symbol data from CSV files for super alert generation.
"""

import csv
import logging
import sys
import select
import threading
import time
from typing import Dict, Optional
from pathlib import Path

from ..alerts.super_alert_filter import SuperAlertData


class SymbolDataLoader:
    """Loads symbol data with Signal and Resistance prices from CSV files."""

    def __init__(self, symbols_file: Optional[str] = None):
        """
        Initialize the symbol data loader.

        Args:
            symbols_file: Path to symbols CSV file
        """
        self.symbols_file = symbols_file
        self.logger = logging.getLogger(__name__)

        # Determine symbols file if not provided
        if self.symbols_file is None:
            from datetime import datetime
            import pytz
            et_tz = pytz.timezone('US/Eastern')
            current_date = datetime.now(et_tz).strftime('%Y%m%d')
            self.symbols_file = f"data/{current_date}.csv"

    def _get_user_input_with_timeout(self, prompt: str, timeout: int = 10) -> Optional[str]:
        """
        Get user input with a timeout.

        Args:
            prompt: Prompt to display to user
            timeout: Timeout in seconds

        Returns:
            User input string or None if timeout
        """
        print(prompt, end='', flush=True)

        # For non-Unix systems or if stdin is not a TTY, return None immediately
        if not hasattr(select, 'select') or not sys.stdin.isatty():
            time.sleep(timeout)
            print(f"\nNo response after {timeout} seconds, using automatic fix.")
            return None

        # Use select to wait for input with timeout
        ready, _, _ = select.select([sys.stdin], [], [], timeout)

        if ready:
            return sys.stdin.readline().strip()
        else:
            print(f"\nNo response after {timeout} seconds, using automatic fix.")
            return None

    def _validate_and_fix_inverted_prices(self, symbol: str, signal_price: float,
                                          resistance_price: float) -> tuple[float, float]:
        """
        Validate signal/resistance prices and fix if inverted.

        Args:
            symbol: Symbol name
            signal_price: Signal price
            resistance_price: Resistance price

        Returns:
            Tuple of (validated_signal_price, validated_resistance_price)
        """
        # Check if prices are inverted (resistance should be > signal for bullish breakouts)
        if resistance_price <= signal_price:
            warning_msg = (f"⚠️  INVERTED PRICES for {symbol}: Signal=${signal_price:.4f} >= "
                           f"Resistance=${resistance_price:.4f}")
            self.logger.warning(warning_msg)
            print("\n🚨 INVERTED SIGNAL/RESISTANCE PRICES DETECTED 🚨")
            print(f"Symbol: {symbol}")
            print(f"Signal Price: ${signal_price:.4f}")
            print(f"Resistance Price: ${resistance_price:.4f}")
            print("For bullish breakouts, Resistance should be HIGHER than Signal price.")
            print()

            # Prompt user for new resistance value
            prompt = (f"Enter new Resistance price for {symbol} (or press Enter to auto-fix "
                      f"to ${signal_price * 1.10:.4f}): ")
            user_response = self._get_user_input_with_timeout(prompt)

            if user_response is not None and user_response.strip():
                try:
                    new_resistance = float(user_response.strip())
                    if new_resistance > signal_price:
                        self.logger.info(f"✅ User corrected {symbol} Resistance to ${new_resistance:.4f}")
                        return signal_price, new_resistance
                    else:
                        warning_msg = (f"User-provided resistance ${new_resistance:.4f} still <= "
                                       f"signal ${signal_price:.4f}, using auto-fix")
                        self.logger.warning(warning_msg)
                except ValueError:
                    self.logger.warning(f"Invalid user input '{user_response}', using auto-fix")

            # Auto-fix: set resistance = signal * 1.10
            auto_resistance = signal_price * 1.10
            info_msg = (f"🔧 Auto-fixed {symbol}: Resistance ${resistance_price:.4f} → "
                        f"${auto_resistance:.4f} (Signal * 1.10)")
            self.logger.info(info_msg)
            print(f"✅ Auto-fixed {symbol}: Resistance = ${auto_resistance:.4f} (Signal * 1.10)")
            return signal_price, auto_resistance

        return signal_price, resistance_price

    def load_symbol_data(self) -> Dict[str, SuperAlertData]:
        """
        Load symbol data from CSV file with Signal and Resistance prices.

        Returns:
            Dictionary mapping symbols to SuperAlertData objects
        """
        symbol_data = {}

        try:
            if not Path(self.symbols_file).exists():
                self.logger.error(f"Symbols file not found: {self.symbols_file}")
                return symbol_data

            with open(self.symbols_file, 'r') as file:
                reader = csv.DictReader(file)

                for row in reader:
                    symbol = row.get('Symbol', '').strip().upper()
                    if not symbol or symbol in ['SYMBOL', 'TICKER', 'STOCK']:
                        continue

                    # Parse Signal and Resistance prices
                    try:
                        signal_str = row.get('Signal', '0').strip()
                        resistance_str = row.get('Resistance', '0').strip()

                        # Handle empty values
                        if not signal_str or not resistance_str:
                            continue

                        signal_price = float(signal_str)
                        resistance_price = float(resistance_str)

                        if signal_price > 0 and resistance_price > 0:
                            # Validate and fix inverted prices
                            validated_signal, validated_resistance = self._validate_and_fix_inverted_prices(
                                symbol, signal_price, resistance_price
                            )

                            symbol_data[symbol] = SuperAlertData(symbol, validated_signal, validated_resistance)
                            debug_msg = (f"Loaded {symbol}: Signal=${validated_signal:.2f}, "
                                         f"Resistance=${validated_resistance:.2f}")
                            self.logger.debug(debug_msg)

                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Could not parse prices for {symbol}: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"Error loading symbol data from {self.symbols_file}: {e}")

        info_msg = (f"Loaded {len(symbol_data)} symbols with Signal/Resistance data "
                    f"from {self.symbols_file}")
        self.logger.info(info_msg)
        return symbol_data

    def get_symbols_file(self) -> str:
        """Get the symbols file path."""
        return self.symbols_file
