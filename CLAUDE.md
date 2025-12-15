# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based Alpaca trading API wrapper for automated stock trading operations. The application provides functionality for position management, order execution, and bracket order trading with built-in risk management features.

## Environment Setup

1. Install dependencies:
   ```bash
   pip3 install alpaca-trade-api python-dotenv matplotlib pytest pytest-cov
   ```

2. Create `.env` file with required API credentials:
   ```env
   ALPACA_API_KEY=your_api_key_here
   ALPACA_SECRET_KEY=your_secret_key_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
   PORTFOLIO_RISK=0.10  # Optional: default is 0.10 (10%)
   ```

## Common Commands

### Running the Application
```bash
python3 code/alpaca.py                                              # View portfolio status
python3 code/alpaca.py --get-latest-quote --symbol AAPL            # Get latest quote
python3 code/alpaca.py --buy --symbol AAPL                         # Buy order (dry run)
python3 code/alpaca.py --buy --symbol AAPL --submit                 # Execute buy order
python3 code/alpaca.py --bracket-order --symbol AAPL --quantity 10 --market-price 150.00 --submit
```

### Testing
```bash
# Using the test scripts (recommended)
./test.sh                                                          # Run all tests
./test.sh verbose                                                  # Run tests with verbose output
./test.sh orb                                                      # Run only ORB tests
./test.sh coverage                                                 # Run tests with coverage report
./test.sh lint                                                     # Run linting then tests

# Using Python test runner (more options)
python run_tests.py                                                # Run all tests
python run_tests.py -v                                             # Run tests with verbose output
python run_tests.py -f test_orb.py                                 # Run specific test file
python run_tests.py -c                                             # Run tests with coverage
python run_tests.py -l                                             # Run linting before tests
python run_tests.py -k "test_filter"                               # Run tests matching pattern

# Direct pytest commands
python -m pytest tests/                                            # Run all tests
python -m pytest tests/test_orb.py -v                              # Run ORB tests with verbose output
python -m unittest discover -s test                                # Run old unittest tests
```

### Debugging
```bash
python3 -m pdb code/alpaca.py                                      # Debug main script
python -m pdb test/test_Alpaca.py TestAlpaca.test_order           # Debug specific test
```

### Linting
```bash
flake8 code/ atoms/                                                # Lint codebase (based on setup.cfg)
```

## Code Architecture

### Main Components

- **`code/alpaca.py`**: Main entry point containing the `alpaca_private` class
- **`atoms/`**: Modular components organized by functionality:
  - `atoms/api/`: API interaction functions (get_cash, get_positions, get_latest_quote, etc.)
  - `atoms/display/`: Display and printing functions (print_cash, print_orders, etc.)
  - `atoms/utils/`: Utility functions (parse_args, delay)

### Key Classes and Methods

- **`alpaca_private`**: Main trading class in `code/alpaca.py:35`
  - `_buy()`: Execute buy orders with automatic bracket order protection (`code/alpaca.py:72`)
  - `_bracketOrder()`: Create bracket orders with stop-loss protection (`code/alpaca.py:138`)
  - `Exec()`: Main execution logic handling command-line operations (`code/alpaca.py:180`)

### Risk Management Features

- **Stop Loss**: Default 7.5% stop-loss on all orders (configurable via `STOP_LOSS_PERCENT`)
- **Position Sizing**: 
  - First position uses `PORTFOLIO_RISK` percentage of cash (default 10%)
  - Subsequent positions use remaining cash
- **Bracket Orders**: Automatic stop-loss and take-profit protection

### Configuration

- **Portfolio Risk**: Controlled by `PORTFOLIO_RISK` environment variable
- **Stop Loss**: Configurable via `STOP_LOSS_PERCENT` class constant
- **API Endpoints**: Paper trading vs live trading via `ALPACA_BASE_URL`

## Development Notes

- The codebase uses conda environment `alpaca` for VS Code debugging
- All orders default to dry-run mode unless `--submit` flag is used
- The application supports both paper trading and live trading environments
- Position sizing logic is currently optimized for 50% risk (see TODO in `code/alpaca.py:97`)
- Use conda environment alpaca
- you are not to make commits unless told