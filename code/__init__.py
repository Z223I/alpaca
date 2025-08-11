"""
Alpaca Trading System - Core Package

This package contains the main trading modules for the Alpaca trading system.
"""

__version__ = "1.0.0"
__author__ = "Alpaca Trading System"

# Import main classes for easy access
from .alpaca import AlpacaPrivate, execMain

# Create alias for backwards compatibility
alpaca_private = AlpacaPrivate

__all__ = [
    'AlpacaPrivate',
    'alpaca_private',
    'execMain'
]