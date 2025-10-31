"""
Alpaca Trading Configuration

This file contains the configuration dataclasses for the Alpaca trading system.
It replaces environment variable-based configuration with a structured approach.

Example configuration structure:
{
  "Provider": {
    "alpaca": {
      "Bruce": {
        "paper": {
          "APP Key": "PKC4FO2UPXGD9NFHLG3O",
          "APP Secret": "O35hUrAthJa0JhAlNMe3mm45rAprdZBNnbR7vHZm",
          "URL": "https://paper-api.alpaca.markets"
        },
        "live": {
          "APP Key": "your_live_key",
          "APP Secret": "your_live_secret",
          "URL": "https://api.alpaca.markets"
        },
        "cash": {
          "APP Key": "your_cash_key",
          "APP Secret": "your_cash_secret",
          "URL": "https://api.alpaca.markets"
        }
      },
      "Dale Wilson": {
        "paper": {
          "APP Key": "PKC4FO2UPXGD9NFHLG3O",
          "APP Secret": "O35hUrAthJa0JhAlNMe3mm45rAprdZBNnbR7vHZm",
          "URL": "https://paper-api.alpaca.markets"
        },
        "live": {
          "APP Key": "your_live_key",
          "APP Secret": "your_live_secret",
          "URL": "https://api.alpaca.markets"
        },
        "cash": {
          "APP Key": "your_cash_key",
          "APP Secret": "your_cash_secret",
          "URL": "https://api.alpaca.markets"
        }
      }
    }
  }
}
"""

from dataclasses import dataclass
from typing import Dict, Optional
import os


@dataclass
class EnvironmentConfig:
    """Configuration for a specific trading environment (paper/live/cash)."""
    app_key: str
    app_secret: str
    url: str
    auto_trade: str = "no"
    auto_amount: int = 10
    trailing_percent: float = 12.5
    take_profit_percent: float = 10.0
    max_trades_per_day: int = 1

    def __post_init__(self):
        """Validate that all required fields are provided."""
        if not self.app_key:
            raise ValueError("APP Key is required")
        if not self.app_secret:
            raise ValueError("APP Secret is required")
        if not self.url:
            raise ValueError("URL is required")


@dataclass
class AccountConfig:
    """Configuration for a specific account with multiple environments."""
    paper: EnvironmentConfig
    live: EnvironmentConfig
    cash: EnvironmentConfig


@dataclass
class ProviderConfig:
    """Configuration for a specific provider with multiple accounts."""
    accounts: Dict[str, AccountConfig]


@dataclass
class AlpacaConfig:
    """Main configuration class for the Alpaca trading system."""
    providers: Dict[str, ProviderConfig]
    portfolio_risk: float = 0.10

    def get_environment_config(self, provider: str = "alpaca", account: str = "Bruce", environment: str = "paper") -> EnvironmentConfig:
        """
        Get configuration for a specific provider/account/environment combination.

        Args:
            provider: Provider name (default: "alpaca")
            account: Account name (default: "Bruce")
            environment: Environment name (default: "paper")

        Returns:
            EnvironmentConfig for the specified combination

        Raises:
            KeyError: If the specified configuration is not found
        """
        try:
            return getattr(self.providers[provider].accounts[account], environment)
        except KeyError as e:
            raise KeyError(f"Configuration not found for {provider}/{account}/{environment}: {e}")


# Example configuration instance
# Users should modify this with their actual credentials
CONFIG = AlpacaConfig(
    providers={
        "alpaca": ProviderConfig(
            accounts={
                "Bruce": AccountConfig(
                    paper=EnvironmentConfig(
                        app_key="PKC4FO2UPXGD9NFHLG3O",
                        app_secret="O35hUrAthJa0JhAlNMe3mm45rAprdZBNnbR7vHZm",
                        url="https://paper-api.alpaca.markets",
                        auto_trade="no",
                        auto_amount=4000,
                        trailing_percent=12.5,
                        take_profit_percent=10.0,
                        max_trades_per_day=2
                    ),
                    live=EnvironmentConfig(
                        app_key=os.getenv("ALPACA_LIVE_API_KEY", "AKE3BC8FAL1SH0V2C1CE"),
                        app_secret=os.getenv("ALPACA_LIVE_SECRET_KEY", "QT8h5l8GJ6EakAwkSM9VsAM2XgIEtiq6xd6NM2Tb"),
                        url="https://api.alpaca.markets",
                        auto_trade="no",
                        auto_amount=100,
                        trailing_percent=10.0,
                        take_profit_percent=12.0,
                        max_trades_per_day=1
                    ),
                    cash=EnvironmentConfig(
                        app_key=os.getenv("ALPACA_CASH_API_KEY", "your_cash_api_key_here"),
                        app_secret=os.getenv("ALPACA_CASH_SECRET_KEY", "your_cash_secret_key_here"),
                        url="https://api.alpaca.markets",
                        auto_trade="no",
                        auto_amount=10,
                        trailing_percent=12.5,
                        take_profit_percent=10.0,
                        max_trades_per_day=5
                    )
                ),
                "Dale Wilson": AccountConfig(
                    paper=EnvironmentConfig(
                        app_key="PKC4FO2UPXGD9NFHLG3O",
                        app_secret="O35hUrAthJa0JhAlNMe3mm45rAprdZBNnbR7vHZm",
                        url="https://paper-api.alpaca.markets",
                        auto_trade="no",
                        auto_amount=10,
                        trailing_percent=12.5,
                        take_profit_percent=10.0,
                        max_trades_per_day=5
                    ),
                    live=EnvironmentConfig(
                        app_key=os.getenv("ALPACA_LIVE_API_KEY", "your_live_api_key_here"),
                        app_secret=os.getenv("ALPACA_LIVE_SECRET_KEY", "your_live_secret_key_here"),
                        url="https://api.alpaca.markets",
                        auto_trade="no",
                        auto_amount=10,
                        trailing_percent=12.5,
                        take_profit_percent=10.0,
                        max_trades_per_day=5
                    ),
                    cash=EnvironmentConfig(
                        app_key=os.getenv("ALPACA_CASH_API_KEY", "your_cash_api_key_here"),
                        app_secret=os.getenv("ALPACA_CASH_SECRET_KEY", "your_cash_secret_key_here"),
                        url="https://api.alpaca.markets",
                        auto_trade="no",
                        auto_amount=10,
                        trailing_percent=12.5,
                        take_profit_percent=10.0,
                        max_trades_per_day=5
                    )
                ),
                "Janice": AccountConfig(
                    paper=EnvironmentConfig(
                        app_key="PKCCH3M73MPLXI3FSMSM",
                        app_secret="SHUp8w7HFTUz2UwsZwNF26rdtcZjE5gzwaEgkhAN",
                        url="https://paper-api.alpaca.markets",
                        auto_trade="no",
                        auto_amount=2000,
                        trailing_percent=12.5,
                        take_profit_percent=10.0,
                        max_trades_per_day=1
                    ),
                    live=EnvironmentConfig(
                        app_key=os.getenv("ALPACA_LIVE_API_KEY", "your_live_api_key_here"),
                        app_secret=os.getenv("ALPACA_LIVE_SECRET_KEY", "your_live_secret_key_here"),
                        url="https://api.alpaca.markets",
                        auto_trade="no",
                        auto_amount=10,
                        trailing_percent=12.5,
                        take_profit_percent=10.0,
                        max_trades_per_day=5
                    ),
                    cash=EnvironmentConfig(
                        app_key=os.getenv("ALPACA_CASH_API_KEY", "your_cash_api_key_here"),
                        app_secret=os.getenv("ALPACA_CASH_SECRET_KEY", "your_cash_secret_key_here"),
                        url="https://api.alpaca.markets",
                        auto_trade="no",
                        auto_amount=10,
                        trailing_percent=12.5,
                        take_profit_percent=10.0,
                        max_trades_per_day=5
                    )
                )
            }
        )
    },
    portfolio_risk=float(os.getenv("PORTFOLIO_RISK", "0.10"))
)


def get_current_config() -> AlpacaConfig:
    """
    Get the current configuration instance.

    Returns:
        The current AlpacaConfig instance
    """
    return CONFIG


def get_api_credentials(provider: str = "alpaca", account: str = "Bruce", environment: str = "paper") -> tuple[str, str, str]:
    """
    Get API credentials for a specific configuration.

    Args:
        provider: Provider name (default: "alpaca")
        account: Account name (default: "Bruce")
        environment: Environment name (default: "paper")

    Returns:
        Tuple of (api_key, secret_key, base_url)
    """
    env_config = CONFIG.get_environment_config(provider, account, environment)
    return env_config.app_key, env_config.app_secret, env_config.url