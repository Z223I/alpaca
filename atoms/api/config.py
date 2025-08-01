from dataclasses import dataclass
from typing import Optional


@dataclass
class TradingConfig:
    """Configuration settings for trading operations."""

    # Default trailing stop percentage for trailing buy orders
    DEFAULT_TRAILING_PERCENT: float = 7.5

    # Other configuration constants can be added here as needed
    # Example: DEFAULT_STOP_LOSS_PERCENT: float = 5.0