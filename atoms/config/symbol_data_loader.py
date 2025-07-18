"""
Symbol Data Loader - Loads Symbol Data with Signal/Resistance Prices

This atom handles loading symbol data from CSV files for super alert generation.
"""

import csv
import logging
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
                            symbol_data[symbol] = SuperAlertData(symbol, signal_price, resistance_price)
                            self.logger.debug(f"Loaded {symbol}: Signal=${signal_price:.2f}, Resistance=${resistance_price:.2f}")
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Could not parse prices for {symbol}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error loading symbol data from {self.symbols_file}: {e}")
            
        self.logger.info(f"Loaded {len(symbol_data)} symbols with Signal/Resistance data from {self.symbols_file}")
        return symbol_data
    
    def get_symbols_file(self) -> str:
        """Get the symbols file path."""
        return self.symbols_file