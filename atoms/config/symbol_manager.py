"""
Symbol Management for ORB Alerts

This module handles loading and managing the watchlist of symbols for ORB alerts.
Reads from date-specific CSV files (YYYYMMDD.csv) and provides filtering capabilities.
Includes file monitoring to automatically reload symbols when the file changes.
"""

import csv
import os
from typing import List, Set, Optional, Callable
from pathlib import Path
from datetime import datetime
import pytz
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class SymbolFileHandler(FileSystemEventHandler):
    """File system event handler for monitoring symbol file changes."""

    def __init__(self, symbol_manager):
        self.symbol_manager = symbol_manager
        super().__init__()

    def on_modified(self, event):
        if not event.is_directory and event.src_path == str(self.symbol_manager.symbols_file_path):
            self.symbol_manager._handle_file_change()


class SymbolManager:
    """Manages the watchlist of symbols for ORB alerts."""

    def __init__(self, symbols_file: Optional[str] = None):
        """
        Initialize symbol manager.

        Args:
            symbols_file: Path to CSV file containing symbols. If None, uses current date file (data/YYYYMMDD.csv)
        """
        self.symbols_file = symbols_file or self._get_current_date_file()
        self.symbols_file_path = Path(self.symbols_file).resolve()
        self._symbols: Set[str] = set()
        self._change_callbacks: List[Callable[[List[str]], None]] = []
        self.logger = logging.getLogger(__name__)

        # File monitoring setup
        self._observer: Optional[Observer] = None
        self._file_handler: Optional[SymbolFileHandler] = None

        self._load_symbols()
        self._start_file_monitoring()

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

        self.logger.info(f"Loaded {len(self._symbols)} symbols from {self.symbols_file}")

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

    def add_change_callback(self, callback: Callable[[List[str]], None]) -> None:
        """
        Add a callback function to be called when symbols change.

        Args:
            callback: Function to call with new symbols list when file changes
        """
        self._change_callbacks.append(callback)

    def _start_file_monitoring(self) -> None:
        """Start monitoring the symbols file for changes."""
        try:
            if not self.symbols_file_path.exists():
                self.logger.warning(f"Symbols file does not exist, skipping file monitoring: {self.symbols_file_path}")
                return

            self._file_handler = SymbolFileHandler(self)
            self._observer = Observer()

            # Monitor the directory containing the file
            watch_dir = self.symbols_file_path.parent
            self._observer.schedule(self._file_handler, str(watch_dir), recursive=False)
            self._observer.start()

            self.logger.info(f"Started monitoring {self.symbols_file_path} for changes")

        except Exception as e:
            self.logger.error(f"Failed to start file monitoring: {e}")

    def _handle_file_change(self) -> None:
        """Handle file change event by reloading symbols."""
        try:
            self.logger.info(f"Detected change in {self.symbols_file_path}, reloading symbols...")

            # Store old symbols for comparison
            old_symbols = set(self._symbols)

            # Reload symbols
            self._load_symbols()

            # Check if symbols actually changed
            new_symbols = set(self._symbols)
            if old_symbols != new_symbols:
                added = new_symbols - old_symbols
                removed = old_symbols - new_symbols

                if added:
                    self.logger.info(f"Added symbols: {', '.join(sorted(added))}")
                if removed:
                    self.logger.info(f"Removed symbols: {', '.join(sorted(removed))}")

                self.logger.info(f"Symbol list updated: {len(old_symbols)} -> {len(new_symbols)} symbols")

                # Notify callbacks
                for callback in self._change_callbacks:
                    try:
                        callback(self.get_symbols())
                    except Exception as e:
                        self.logger.error(f"Error in symbol change callback: {e}")
            else:
                self.logger.debug("File changed but symbol list is identical")

        except Exception as e:
            self.logger.error(f"Error handling file change: {e}")

    def stop_monitoring(self) -> None:
        """Stop file monitoring."""
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self.logger.info("Stopped file monitoring")
            except Exception as e:
                self.logger.error(f"Error stopping file monitoring: {e}")

    def __del__(self):
        """Cleanup file monitoring on object destruction."""
        self.stop_monitoring()