# Alpaca Trading System

A comprehensive Python-based trading system for automated stock trading using the Alpaca API. This system provides functionality for position management, order execution, bracket orders, after-hours trading, and ORB (Opening Range Breakout) strategies with built-in risk management features.

## Features

### Core Trading Capabilities
- **Market & Limit Orders**: Execute buy and sell orders with various order types
- **Bracket Orders**: Automated stop-loss and take-profit protection
- **After-Hours Trading**: Extended hours trading with limit orders only
- **Short Selling**: Bearish trading strategies with proper risk management
- **Position Management**: Automated position sizing based on portfolio risk
- **Position Liquidation**: Close individual positions or liquidate entire portfolio
- **Quote Retrieval**: Real-time market data and price quotes

### Advanced Features
- **ORB (Opening Range Breakout)**: Automated breakout detection and alerts
- **Risk Management**: Configurable stop-loss and take-profit levels
- **Portfolio Risk Control**: Percentage-based position sizing
- **Telegram Integration**: Real-time trading alerts via Telegram
- **Performance Monitoring**: Trade tracking and performance analysis
- **Chart Generation**: Candlestick charts with technical indicators

## Quick Start

### 1. Environment Setup

Install dependencies:
```bash
pip3 install alpaca-trade-api python-dotenv matplotlib pytest pytest-cov
```

### 2. Configuration

Create a `.env` file in the project root:
```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
PORTFOLIO_RISK=0.10  # Optional: default is 0.10 (10%)
```

### 3. Basic Usage

```bash
# View portfolio status
python3 code/alpaca.py

# Get latest quote
python3 code/alpaca.py --get-latest-quote --symbol AAPL

# Execute a buy order (dry run)
python3 code/alpaca.py --buy --symbol AAPL --take-profit 160.00

# Execute a buy order (live)
python3 code/alpaca.py --buy --symbol AAPL --take-profit 160.00 --submit

# After-hours trading
python3 code/alpaca.py --buy --symbol AAPL --after-hours --take-profit 160.00 --submit
```

## Complete Command Line Reference

### Command Categories

#### Trading Operations
- `-b, --bracket-order` - Execute bracket order
- `-f, --future-bracket-order` - Execute future bracket order with limit entry  
- `--buy` - Execute a buy order for the specified symbol
- `--sell-short` - Execute a short sell order for bearish predictions
- `-q, --get-latest-quote` - Get latest quote for a symbol

#### Liquidation Operations
- `--liquidate` - Liquidate position for a specific symbol (requires --symbol)
- `--liquidate-all` - Liquidate all open positions and optionally cancel all orders
- `--cancel-orders` - Cancel all open orders (used with --liquidate-all)

#### Order Parameters
- `--symbol SYMBOL` - Stock symbol
- `--quantity QUANTITY` - Number of shares
- `--market-price MARKET_PRICE` - Current market price for bracket order
- `--limit-price LIMIT_PRICE` - Limit price for future bracket order entry
- `--stop-price STOP_PRICE` - Stop loss price for future bracket order
- `--take-profit TAKE_PROFIT` - Take profit price for bracket orders
- `--stop-loss STOP_LOSS` - Custom stop loss price for buy/short orders
- `--amount AMOUNT` - Dollar amount to invest (calculates quantity automatically)

#### Order Modifiers
- `--submit` - Actually submit the order (default: False for dry run)
- `--after-hours` - Execute order for after-hours/extended hours trading
- `--custom-limit-price CUSTOM_LIMIT_PRICE` - Custom limit price for after-hours orders
- `--calc-take-profit` - Calculate take profit as (latest_quote - stop_loss) * 1.5

#### Display-Only Commands
- `--positions` - Display current positions only (improved concise format)
- `--cash` - Display cash balance only  
- `--active-order` - Display active orders only

### Usage Examples

#### Basic Orders
```bash
# Buy with automatic bracket order protection
python3 code/alpaca.py --buy --symbol AAPL --take-profit 160.00 --submit

# Short sell with protection
python3 code/alpaca.py --sell-short --symbol AAPL --take-profit 140.00 --submit

# Buy with custom amount and calculated take profit
python3 code/alpaca.py --buy --symbol AAPL --amount 1000 --stop-loss 145.00 --calc-take-profit --submit
```

#### Position Liquidation
```bash
# Liquidate a specific position (dry run)
python3 code/alpaca.py --liquidate --symbol AAPL

# Actually liquidate a specific position
python3 code/alpaca.py --liquidate --symbol AAPL --submit

# Liquidate all positions (dry run)
python3 code/alpaca.py --liquidate-all

# Liquidate all positions and cancel all orders
python3 code/alpaca.py --liquidate-all --cancel-orders --submit
```

#### Display Commands
```bash
# Show only positions (improved concise format)
python3 code/alpaca.py --positions

# Show only cash balance
python3 code/alpaca.py --cash

# Show only active orders
python3 code/alpaca.py --active-order
```

#### Bracket Orders
```bash
# Standard bracket order
python3 code/alpaca.py --bracket-order --symbol AAPL --quantity 10 --market-price 150.00 --take-profit 160.00 --submit

# Future bracket order (limit entry)
python3 code/alpaca.py --future-bracket-order --symbol AAPL --limit-price 148.00 --stop-price 140.00 --take-profit 160.00 --submit
```

#### After-Hours Trading
```bash
# After-hours buy with protection
python3 code/alpaca.py --buy --after-hours --symbol AAPL --take-profit 160.00 --stop-loss 145.00 --submit

# After-hours with custom limit price
python3 code/alpaca.py --buy --after-hours --symbol AAPL --custom-limit-price 151.00 --take-profit 160.00 --submit

# After-hours with specific dollar amount
python3 code/alpaca.py --buy --after-hours --symbol AAPL --amount 1000 --take-profit 160.00 --stop-loss 145.00 --submit

# After-hours amount without protection
python3 code/alpaca.py --buy --after-hours --symbol AAPL --amount 500 --submit
```

## Project Structure

```
├── code/                           # Main application modules
│   ├── alpaca.py                  # Main trading application
│   ├── orb_alerts.py              # ORB breakout alert system
│   ├── orb_alerts_monitor.py      # Real-time ORB monitoring
│   └── market_open.py             # Market hours utilities
├── atoms/                         # Modular components
│   ├── api/                       # Alpaca API interactions
│   │   ├── get_cash.py           # Account cash balance
│   │   ├── get_positions.py      # Current positions
│   │   ├── get_latest_quote.py   # Market data quotes
│   │   └── parse_args.py         # Command line parsing
│   ├── alerts/                    # Alert system components
│   │   ├── breakout_detector.py  # ORB breakout detection
│   │   ├── confidence_scorer.py  # Signal confidence scoring
│   │   └── alert_formatter.py    # Alert message formatting
│   ├── display/                   # Output formatting
│   │   ├── print_cash.py         # Portfolio display
│   │   ├── print_positions.py    # Position summaries
│   │   └── plot_candle_chart.py  # Chart generation
│   ├── telegram/                  # Telegram integration
│   │   ├── telegram_post.py      # Message posting
│   │   └── user_manager.py       # User management
│   └── utils/                     # Utility functions
│       ├── calculate_ema.py       # Technical indicators
│       ├── calculate_vwap.py      # Volume weighted average price
│       └── delay.py               # Rate limiting
├── tests/                         # Test suite
│   ├── test_alpaca_after_hours.py # After-hours trading tests
│   ├── test_orb.py               # ORB functionality tests
│   └── test_utils_parse_args.py  # CLI argument tests
└── historical_data/               # ORB historical data storage
```

## Key Classes and Methods

### `alpaca_private` Class (Main Trading Engine)

Located in `code/alpaca.py`, this is the core trading class:

#### Core Trading Methods
- `_buy()`: Execute buy orders with automatic bracket protection
- `_sell_short()`: Execute short sell orders with protection
- `_bracketOrder()`: Create standard bracket orders
- `_futureBracketOrder()`: Create limit entry bracket orders

#### After-Hours Trading Methods
- `_buy_after_hours()`: Simple after-hours buy orders
- `_buy_after_hours_protected()`: After-hours buy with stop-loss/take-profit
- `_sell_short_after_hours()`: Simple after-hours short orders
- `_sell_short_after_hours_protected()`: After-hours short with protection

#### Position Liquidation Methods  
- `_liquidate_position()`: Close specific position and cancel related orders
- `_liquidate_all()`: Close all positions and optionally cancel all orders

#### Utility Methods
- `_calculateQuantity()`: Automatic position sizing
- `Exec()`: Main execution logic and command line handling

## Risk Management

### Automatic Risk Controls
- **Default Stop Loss**: 5% stop-loss on all orders (configurable via `STOP_LOSS_PERCENT`)
- **Portfolio Risk**: Configurable percentage of cash for position sizing (default 10%)
- **Take Profit Calculation**: Automatic take profit calculation based on risk-reward ratios
- **Bracket Order Protection**: All regular trading includes automatic stop-loss and take-profit

### Position Sizing Logic
- **First Position**: Uses `PORTFOLIO_RISK` percentage of available cash
- **Subsequent Positions**: Uses remaining available cash
- **Amount-Based Orders**: Custom dollar amount with automatic share calculation

### After-Hours Restrictions
- **Limit Orders Only**: No market orders allowed in extended hours
- **Extended Hours Time-in-Force**: Proper order routing for after-hours execution
- **No Bracket Orders**: After-hours trading uses separate protection methods

## ORB (Opening Range Breakout) System

### Components
- **`orb_alerts.py`**: Real-time breakout monitoring and alert generation
- **`orb_alerts_monitor.py`**: Continuous monitoring with database storage
- **Breakout Detection**: Automated identification of price breakouts above/below opening range
- **Confidence Scoring**: Multi-factor confidence analysis for trading signals
- **Alert Filtering**: Priority-based alert system (HIGH/MEDIUM/LOW)

### ORB Features
- **15-minute Opening Range**: Standard ORB calculation period
- **Volume Analysis**: Volume ratio considerations for signal validation
- **Technical Indicators**: EMA, VWAP, and momentum indicators
- **Historical Data**: Automatic data collection and storage
- **Chart Generation**: Visual analysis with breakout annotations

## Testing

### Test Categories
- **After-Hours Trading**: 26 comprehensive tests for extended hours functionality
- **Risk Management**: Portfolio risk and position sizing validation
- **ORB Functionality**: Breakout detection and signal generation tests
- **CLI Arguments**: Command line parsing and validation tests
- **Integration Tests**: End-to-end trading workflow validation

### Running Tests
```bash
# Run all tests
./test.sh

# Run specific test categories
./test.sh orb                    # ORB-specific tests
./test.sh verbose               # Verbose output
./test.sh coverage              # Coverage report

# Run individual test files
python -m pytest tests/test_alpaca_after_hours.py -v
python -m pytest tests/test_orb.py -v
```

## Telegram Integration

### Setup
1. Create a Telegram bot and get API token
2. Add Telegram configuration to `.env`:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Features
- **Real-time Alerts**: Instant ORB breakout notifications
- **Trade Confirmations**: Order execution confirmations
- **Portfolio Updates**: Balance and position changes
- **Error Notifications**: System error alerts

## Development and Debugging

### Environment Setup
The project uses a conda environment for development:
```bash
conda create -n alpaca python=3.10
conda activate alpaca
pip install -r requirements.txt
```

### Debugging
```bash
# Debug main script
python3 -m pdb code/alpaca.py --buy --symbol AAPL

# Debug specific functionality
python3 -m pdb code/orb_alerts.py
```

### Linting
```bash
flake8 code/ atoms/                # Lint codebase
```

## Configuration Options

### Environment Variables
- `ALPACA_API_KEY`: Alpaca API key (required)
- `ALPACA_SECRET_KEY`: Alpaca secret key (required)
- `ALPACA_BASE_URL`: API endpoint (paper vs live trading)
- `PORTFOLIO_RISK`: Position sizing percentage (default: 0.10)
- `ORB_DEBUG`: Enable ORB debugging output
- `TELEGRAM_BOT_TOKEN`: Telegram bot token (optional)
- `TELEGRAM_CHAT_ID`: Telegram chat ID (optional)

### Trading Configuration
- **Stop Loss Percentage**: Configurable in `alpaca_private.STOP_LOSS_PERCENT`
- **Take Profit Calculation**: Risk-reward ratio based calculation
- **ORB Period**: 15-minute opening range (configurable)
- **Alert Thresholds**: Breakout percentage and volume multipliers

## Safety Features

### Dry Run Mode
- All orders default to dry-run mode unless `--submit` flag is explicitly used
- Comprehensive order preview before execution
- Risk calculation validation before order submission

### Paper Trading
- Supports both paper trading and live trading environments
- Paper trading recommended for testing and development
- Full feature parity between paper and live trading modes

### Error Handling
- Comprehensive error handling and retry logic
- Order validation before submission
- Market hours validation for order types
- Connection failure recovery

## Support and Documentation

### Getting Help
- Review the complete command line help: `python3 code/alpaca.py --help`
- Check test files for usage examples
- Examine the `atoms/` modules for detailed functionality

### Position Display Format
The `--positions` command now shows a clean, concise table format:
```
positions:
  Symbol    Qty      Avg Fill
  ────────  ──────── ────────
  AAPL           100  $150.25
  TSLA            50  $245.80
```

### Common Issues
- **API Credentials**: Ensure `.env` file is properly configured
- **Market Hours**: After-hours trading has different restrictions
- **Order Types**: Some order types not available in extended hours
- **Position Sizing**: Ensure sufficient cash balance for orders

### Contributing
- Follow existing code patterns and structure
- Add tests for new functionality
- Update documentation for new features
- Use the atomic component structure in `atoms/`

## License

This project is for educational and development purposes. Users are responsible for compliance with all applicable trading regulations and Alpaca API terms of service.