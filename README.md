# Alpaca Trading API Wrapper

A Python wrapper for automated trading operations using the Alpaca trading API. This script provides functionality for position management, order execution, and bracket order trading with built-in risk management.

## Features

- **Automated Trading**: Execute buy orders with automatic stop-loss protection
- **Bracket Orders**: Create bracket orders with configurable stop-loss percentages
- **Portfolio Management**: View positions, cash balance, and active orders
- **Risk Management**: Built-in 7.5% stop-loss protection and configurable portfolio risk
- **Quote Retrieval**: Get latest market quotes for symbols
- **Environment Configuration**: Secure API key management via environment variables

## Prerequisites

- Python 3.6+
- Alpaca trading account (paper or live)
- Required Python packages (see Installation)

## Installation

1. Install required dependencies:
```bash
pip3 install alpaca-trade-api python-dotenv
```

2. Set up environment variables by creating a `.env` file in the project root:
```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
PORTFOLIO_RISK=0.10  # Optional: default is 0.10 (10%)
```

## Usage

### Basic Commands

View current portfolio status:
```bash
python3 code/alpaca.py
```

Get latest quote for a symbol:
```bash
python3 code/alpaca.py --get-latest-quote --symbol AAPL
```

Execute a buy order (dry run):
```bash
python3 code/alpaca.py --buy --symbol AAPL
```

Execute a buy order (actual submission):
```bash
python3 code/alpaca.py --buy --symbol AAPL --submit
```

Execute a buy order with take profit:
```bash
python3 code/alpaca.py --buy --symbol AAPL --submit --take_profit 200.00
```

Create a bracket order:
```bash
python3 code/alpaca.py --bracket-order --symbol AAPL --quantity 10 --market-price 150.00 --submit
```

### Command Line Arguments

- `--buy`: Execute a buy order with automatic position sizing
- `--symbol SYMBOL`: Specify the stock symbol to trade
- `--submit`: Actually submit orders (without this flag, orders are displayed but not executed)
- `--get-latest-quote`: Retrieve the latest quote for a symbol
- `--bracket-order`: Create a bracket order
- `--quantity N`: Number of shares for bracket orders
- `--market-price PRICE`: Market price for bracket order calculations
- `--take_profit PRICE`: Optional take profit price for buy orders

## Risk Management

### Stop Loss Protection
- All buy orders automatically include a 7.5% stop-loss order
- Stop-loss percentage is configurable via the `STOP_LOSS_PERCENT` class constant

### Position Sizing
- **First Position**: Uses `PORTFOLIO_RISK` percentage of available cash (default: 10%)
- **Subsequent Positions**: Uses all remaining cash for diversification
- Position sizes are automatically calculated based on current market price

### Portfolio Risk
- Configurable via `PORTFOLIO_RISK` environment variable
- Default: 10% of available cash for initial positions
- Note: Logic currently optimized for 50% risk - see code comments for updates needed

## Code Structure

### Main Classes

- `alpaca_private`: Main trading class that handles API interactions and order execution

### Key Methods

- `_buy()`: Execute buy orders with automatic bracket order protection
- `_bracketOrder()`: Create bracket orders with stop-loss protection
- `Exec()`: Main execution logic that handles command-line operations

### Dependencies

The script uses modular components from the `atoms` package:
- `atoms.api.*`: API interaction functions
- `atoms.display.*`: Display and printing functions
- `atoms.utils.*`: Utility functions for argument parsing and delays

## Safety Features

- **Dry Run Mode**: Default behavior shows order details without execution
- **Environment Variables**: Secure API key storage
- **Bracket Orders**: Automatic stop-loss protection on all positions
- **Position Validation**: Checks existing positions before sizing new orders

## Development

### Debugging
```bash
python3 -m pdb code/alpaca.py
```

### IDE Integration
The script is configured for VS Code debugging with F5/Ctrl+F5 shortcuts when using the `alpaca` conda environment.

## Important Notes

- **Paper Trading**: Use `https://paper-api.alpaca.markets` for testing
- **Live Trading**: Use `https://api.alpaca.markets` for live trading (exercise caution)
- **Order Execution**: Always test with paper trading first
- **Risk Management**: Review and adjust `PORTFOLIO_RISK` and `STOP_LOSS_PERCENT` based on your risk tolerance

## License

This project is for educational and personal use. Please ensure compliance with your local financial regulations and Alpaca's terms of service.
