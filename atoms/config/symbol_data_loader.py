"""
Symbol Data Loader - Loads Symbol Data with Signal/Resistance Prices

This atom handles loading symbol data from CSV files for super alert generation.
"""

import csv
import logging
from typing import Dict, Optional
from pathlib import Path

from ..alerts.super_alert_filter import SuperAlertData
from ..telegram.telegram_post import TelegramPoster


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

    def _validate_and_fix_inverted_prices(self, symbol: str, signal_price: float,
                                          resistance_price: float) -> tuple[float, float]:
        """
        Validate signal/resistance prices and automatically fix if inverted.

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

            # Auto-fix: set resistance = signal + 0.02
            auto_resistance = signal_price + 0.02
            info_msg = (f"🔧 Auto-fixed {symbol}: Resistance ${resistance_price:.4f} → "
                        f"${auto_resistance:.4f} (Signal + $0.02)")
            self.logger.info(info_msg)
            print(f"✅ Auto-fixed {symbol}: Resistance = ${auto_resistance:.4f} (Signal + $0.02)")
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
            self.logger.info(f"Loading symbol data from: {self.symbols_file}")
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

        # Send symbols list to Bruce via Telegram
        if symbol_data:
            self._send_symbols_to_bruce(list(symbol_data.keys()))

        return symbol_data

    def _send_symbols_to_bruce(self, symbols_list: list) -> None:
        """
        Send the loaded symbols list to Bruce via Telegram.

        Args:
            symbols_list: List of symbol strings
        """
        try:
            # Skip Telegram notification if this is a backtesting run
            # (detected by symbols file path containing 'runs/')
            if 'runs/' in str(self.symbols_file):
                self.logger.info(f"Backtesting detected - skipping Telegram notification "
                                 f"for {len(symbols_list)} symbols")
                return

            telegram_poster = TelegramPoster()

            # Format the message
            symbols_str = ", ".join(symbols_list)
            message = (f"📊 **Symbol Data Loaded**\n\n"
                       f"**Count:** {len(symbols_list)} symbols\n"
                       f"**Symbols:** {symbols_str}\n\n"
                       f"All prices have been validated and auto-corrected if needed.")

            # Send to Bruce specifically
            result = telegram_poster.send_message_to_user(message, "bruce", urgent=False)

            if result['success']:
                self.logger.info(f"✅ Sent symbols list to Bruce via Telegram "
                                 f"({len(symbols_list)} symbols)")
            else:
                self.logger.warning(f"❌ Failed to send symbols to Bruce: "
                                    f"{result.get('errors', [])}")

        except Exception as e:
            self.logger.error(f"Error sending symbols to Bruce via Telegram: {e}")

    def get_symbols_file(self) -> str:
        """Get the symbols file path."""
        return self.symbols_file
