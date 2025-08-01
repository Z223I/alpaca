"""
Alpaca Trading System - Core Package

This package contains the main trading modules for the Alpaca trading system.
"""

__version__ = "1.0.0"
__author__ = "Alpaca Trading System"

# Import main classes for easy access
from .alpaca import alpaca_private, execMain

__all__ = [
    'alpaca_private',
    'execMain'
]