# Alpaca Trading API Wrapper

A Python wrapper for automated trading operations using the Alpaca trading API. This script provides functionality for position management, order execution, and bracket order trading with built-in risk management.

## Features

- **Automated Trading**: Execute buy orders with automatic stop-loss protection
- **Bracket Orders**: Create bracket orders with configurable stop-loss percentages
- **Future Bracket Orders**: Create limit entry bracket orders with automatic quantity calculation
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
pip3 install alpaca-trade-api python-dotenv matplotlib pytest pytest-cov
```

2. Set up environment variables by creating a `.env` file in the project root:
```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
PORTFOLIO_RISK=0.10  # Optional: default is 0.10 (10%)
```

## Testing

This project includes comprehensive test suites for all components. Two convenient test runners are provided for easy testing.

### Quick Testing

**Using the shell script (recommended for quick testing):**
```bash
./test.sh                    # Run all tests
./test.sh verbose            # Run tests with verbose output
./test.sh orb               # Run only ORB tests
./test.sh coverage          # Run tests with coverage report
./test.sh lint              # Run linting then tests
```

**Using the Python test runner (more options):**
```bash
python run_tests.py                    # Run all tests
python run_tests.py -v                 # Run tests with verbose output
python run_tests.py -f test_orb.py     # Run specific test file
python run_tests.py -c                 # Run tests with coverage
python run_tests.py -l                 # Run linting before tests
python run_tests.py -k "test_filter"   # Run tests matching pattern
```

### Direct Testing Commands

**Pytest (recommended):**
```bash
python -m pytest tests/                                                          # Run all tests
python -m pytest tests/test_orb.py -v                                            # Run ORB tests with verbose output
python -m pytest tests/ --cov code --cov atoms --cov-report html --cov-report term  # Run with coverage (HTML + terminal)
python -m pytest tests/ --cov code --cov-report html                             # Run with HTML coverage only
python -m pytest tests/ --cov code --cov-report term                             # Run with terminal coverage only
```

**Legacy unittest:**
```bash
python -m unittest discover -s test        # Run old unittest tests
python test/test_Alpaca.py TestAlpaca.test_order  # Run specific test
```

### Test Structure

- **`tests/test_orb.py`**: Comprehensive tests for ORB (Opening Range Breakout) functionality
  - Data filtering and time range validation
  - PCA data preparation and validation
  - Integration tests with real market data
  - Mock testing for external API dependencies

- **`test/test_Alpaca.py`**: Legacy tests for main trading functionality
  - Order execution and validation
  - Portfolio management
  - API interaction tests

### Testing Features

- **Mocked Dependencies**: All external API calls are mocked for consistent testing
- **Timezone Handling**: Proper EST/EDT timezone testing for market hours
- **Real Data Integration**: Tests use actual market data from `stock_data/` directory
- **Coverage Reports**: HTML and terminal coverage reports available
- **Linting Integration**: Optional code quality checks before testing

### Debugging Tests

```bash
python -m pdb test/test_Alpaca.py TestAlpaca.test_order  # Debug specific test
python -m pytest tests/test_orb.py::TestORB::test_filter_stock_data_by_time_success --pdb  # Debug with pytest
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
python3 code/alpaca.py --bracket_order --symbol AAPL --quantity 10 --market_price 150.00 --submit
```

Create a future bracket order with limit entry:
```bash
python3 code/alpaca.py --future_bracket_order --symbol AAPL --quantity 10 --limit_price 145.00 --stop_price 140.00 --take_profit 160.00 --submit
```

Create a future bracket order with automatic quantity calculation:
```bash
python3 code/alpaca.py --future_bracket_order --symbol AAPL --limit_price 145.00 --stop_price 140.00 --take_profit 160.00 --submit
```

### Command Line Arguments

- `--buy`: Execute a buy order with automatic position sizing
- `--symbol SYMBOL`: Specify the stock symbol to trade
- `--submit`: Actually submit orders (without this flag, orders are displayed but not executed)
- `--get_latest_quote`: Retrieve the latest quote for a symbol
- `--bracket_order`: Create a bracket order with market entry
- `--future_bracket_order`: Create a future bracket order with limit entry
- `--quantity N`: Number of shares for bracket orders (optional for future bracket orders - will auto-calculate if omitted)
- `--market_price PRICE`: Market price for bracket order calculations
- `--limit_price PRICE`: Limit price for future bracket order entry
- `--stop_price PRICE`: Stop loss price for future bracket orders
- `--take_profit PRICE`: Take profit price for bracket orders

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
- `_futureBracketOrder()`: Create future bracket orders with limit entry and stop-loss protection
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
