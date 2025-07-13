"""
Symbol Management for ORB Alerts

This module handles loading and managing the watchlist of symbols for ORB alerts.
Reads from date-specific CSV files (YYYYMMDD.csv) and provides filtering capabilities.
"""

import csv
import os
from typing import List, Set, Optional
from pathlib import Path
from datetime import datetime
import pytz


class SymbolManager:
    """Manages the watchlist of symbols for ORB alerts."""
    
    def __init__(self, symbols_file: Optional[str] = None):
        """
        Initialize symbol manager.
        
        Args:
            symbols_file: Path to CSV file containing symbols. If None, uses current date file (data/YYYYMMDD.csv)
        """
        self.symbols_file = symbols_file or self._get_current_date_file()
        self._symbols: Set[str] = set()
        self._load_symbols()
    
    def _get_current_date_file(self) -> str:
        """
        Get the current date-specific symbols file path.
        
        Returns:
            Path to current date CSV file (data/YYYYMMDD.csv)
        """
        # Use Eastern Time for determining the current trading date
        et_tz = pytz.timezone('US/Eastern')
        current_date = datetime.now(et_tz).strftime('%Y%m%d')
        return f"data/{current_date}.csv"
    
    def _load_symbols(self) -> None:
        """Load symbols from CSV file, handling both date-specific and legacy formats."""
        if not os.path.exists(self.symbols_file):
            raise FileNotFoundError(f"Symbols file not found: {self.symbols_file}")
        
        with open(self.symbols_file, 'r') as file:
            reader = csv.reader(file)
            first_row = True
            for row in reader:
                if row and row[0].strip():  # Skip empty rows
                    symbol = row[0].strip().upper()
                    
                    # Skip header row if it contains "Symbol" (for date-specific files)
                    if first_row and symbol == "SYMBOL":
                        first_row = False
                        continue
                        
                    # Skip any other common header patterns
                    if symbol in ["SYMBOL", "TICKER", "STOCK"]:
                        continue
                        
                    if symbol:
                        self._symbols.add(symbol)
                
                first_row = False
        
        if not self._symbols:
            raise ValueError(f"No symbols found in {self.symbols_file}")
    
    def get_symbols(self) -> List[str]:
        """
        Get list of all symbols.
        
        Returns:
            List of symbol strings
        """
        return sorted(list(self._symbols))
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to the watchlist.
        
        Args:
            symbol: Symbol to add
        """
        self._symbols.add(symbol.upper())
    
    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from the watchlist.
        
        Args:
            symbol: Symbol to remove
        """
        self._symbols.discard(symbol.upper())
    
    def is_symbol_tracked(self, symbol: str) -> bool:
        """
        Check if a symbol is in the watchlist.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if symbol is tracked
        """
        return symbol.upper() in self._symbols
    
    def get_symbol_count(self) -> int:
        """
        Get number of symbols in watchlist.
        
        Returns:
            Number of symbols
        """
        return len(self._symbols)
    
    def save_symbols(self, filename: Optional[str] = None) -> None:
        """
        Save symbols to CSV file.
        
        Args:
            filename: Optional filename to save to (defaults to current file)
        """
        target_file = filename or self.symbols_file
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        with open(target_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for symbol in sorted(self._symbols):
                writer.writerow([symbol])
    
    def reload_symbols(self) -> None:
        """Reload symbols from file."""
        self._symbols.clear()
        self._load_symbols()
    
    def __len__(self) -> int:
        """Return number of symbols."""
        return len(self._symbols)
    
    def __iter__(self):
        """Iterate over symbols."""
        return iter(sorted(self._symbols))
    
    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in watchlist."""
        return symbol.upper() in self._symbols