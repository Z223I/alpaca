# Product Requirements Document: Market Buy with Trailing Sell

**Version:** 1.0  
**Date:** 2025-08-01  
**Status:** Draft

## Overview

This PRD defines the implementation of two new CLI commands for the Alpaca trading system:
1. `--buy-market`: Simple market buy order execution
2. `--buy-market-trailing-sell`: Market buy followed by automatic trailing sell

These features will enhance the existing trading automation by providing immediate market execution with built-in profit-taking mechanisms.

## Problem Statement

Current system limitations:
- No pure market buy functionality (only bracket orders with take-profit/stop-loss)
- No automated order status polling and conditional execution
- Manual intervention required to execute trailing sell after buy orders fill

## Objectives

### Primary Goals
- Implement market buy order functionality via CLI
- Create automated buy-to-trailing-sell workflow
- Add order status polling mechanism
- Maintain consistency with existing codebase patterns

### Success Criteria
- Market buy orders execute immediately at current market price
- Order polling accurately detects fill status
- Trailing sell orders are automatically placed after buy order fills
- All functionality works in both dry-run and live modes
- Integration with existing risk management (portfolio risk percentage)

## Technical Requirements

### API Compatibility
- **Primary:** Continue using existing `alpaca-trade-api` for consistency with current codebase
- **Fallback:** Research migration path to `alpaca-py` SDK if needed for advanced features

### Implementation Details

#### 1. --buy-market Command
```python
python3 code/alpaca.py --buy-market --symbol AAPL [--submit]
```

**Features:**
- Required `--symbol` parameter
- Optional `--submit` flag (dry-run by default)
- Uses existing `_calculateQuantity()` method for position sizing
- Market order type for immediate execution
- No bracket order protection (pure market buy)

**Order Parameters:**
- `side='buy'`
- `type='market'`
- `time_in_force='day'`
- `qty=calculated_quantity`

#### 2. --buy-market-trailing-sell Command
```python
python3 code/alpaca.py --buy-market-trailing-sell --symbol AAPL [--submit]
```

**Workflow:**
1. Execute market buy order using `_buy_market()` method
2. Poll order status every 2-5 seconds until filled or canceled
3. If filled: extract filled quantity and call `_sell_trailing()`
4. If canceled/rejected: log error and exit

**Order Status Polling:**
- Use `api.get_order(order_id)` method
- Check `order.status` field for: `'filled'`, `'canceled'`, `'rejected'`
- Implement timeout mechanism (max 60 seconds)
- Log status changes for debugging

#### 3. Order Status Detection
**Research Required:** Determine all possible order status values in alpaca-trade-api:
- Submitted, pending, filled, canceled, rejected, etc.
- Handle partial fills appropriately

### Integration Points

#### Existing Methods to Leverage
- `_calculateQuantity()`: Position sizing logic
- `_sell_trailing()`: Existing trailing sell implementation  
- `get_latest_quote_avg()`: Market price discovery
- Error handling patterns from existing order methods

#### New Methods to Implement
- `_buy_market()`: Pure market buy execution
- `_buy_market_trailing_sell()`: Combined workflow method
- `_poll_order_status()`: Order status monitoring utility

### Risk Management

#### Position Sizing
- Use existing `PORTFOLIO_RISK` environment variable
- Leverage `_calculateQuantity()` method for consistency
- First position: `cash * PORTFOLIO_RISK / price`
- Subsequent positions: `cash / price`

#### Error Handling
- Network/API failures during polling
- Order rejection scenarios
- Partial fill handling
- Market hours validation

### CLI Integration

#### Argument Parser Updates
```python
# Add to atoms/api/parse_args.py
parser.add_argument('--buy-market', action='store_true')
parser.add_argument('--buy-market-trailing-sell', action='store_true')
```

#### Execution Logic Updates
```python
# Add to alpaca_private.Exec() method
if self.args.buy_market:
    # Execute _buy_market()
if self.args.buy_market_trailing_sell:
    # Execute _buy_market_trailing_sell()
```

## Testing Strategy

### Unit Tests
- Test market buy order generation (dry-run)
- Test order status polling with mock responses
- Validate quantity calculations
- Error handling scenarios

### Integration Tests  
- End-to-end workflow with paper trading account
- Network timeout scenarios
- Market hours edge cases
- Order rejection handling

### Test Files
- `tests/test_market_buy.py`: New test module
- Extend existing test suites as needed

## Documentation Updates

### README_alpaca.md Updates
```markdown
### Market Orders
```bash
# Market buy order
python3 code/alpaca.py --buy-market --symbol AAPL --submit

# Market buy with automatic trailing sell
python3 code/alpaca.py --buy-market-trailing-sell --symbol AAPL --submit
```

### Common Commands Section
Add examples with different symbols and scenarios

## Implementation Timeline

### Phase 1: Research & Design (Completed)
- [x] Research Alpaca API v2 market order methods
- [x] Research order status polling techniques
- [x] Create PRD document

### Phase 2: Core Implementation
- [ ] Implement `_buy_market()` method
- [ ] Add CLI arguments and parsing
- [ ] Unit tests for basic functionality
- [ ] Integration with existing `Exec()` method

### Phase 3: Order Polling & Workflow
- [ ] Implement `_poll_order_status()` utility
- [ ] Implement `_buy_market_trailing_sell()` workflow
- [ ] Handle edge cases and error scenarios
- [ ] Integration tests

### Phase 4: Documentation & Validation
- [ ] Update README_alpaca.md
- [ ] Comprehensive testing with paper account
- [ ] Performance validation and optimization

## Risk Assessment

### Technical Risks
- **Order polling reliability**: API rate limits or network issues
- **Partial fills**: Handling scenarios where only part of order executes
- **Market hours**: Orders placed outside trading hours

### Mitigation Strategies
- Implement exponential backoff for polling
- Handle partial fills by using actual filled quantity
- Add market hours validation before order submission

## Dependencies

### External Dependencies
- `alpaca-trade-api`: Primary trading API (existing)
- `time`: For polling delays (built-in)

### Internal Dependencies  
- Existing order execution patterns
- Risk management configuration
- Error handling utilities

## Success Metrics

### Functional Metrics
- Market buy orders execute within 5 seconds
- Order polling detects fills within 10 seconds
- Trailing sell orders placed within 15 seconds of buy fill
- 99% success rate for dry-run validations

### Performance Metrics
- CLI response time < 2 seconds for dry-run
- Memory usage within existing application bounds
- No degradation of existing functionality

## Future Enhancements

### Potential Extensions
- Support for dollar-amount market buys (notional orders)
- Configurable polling intervals
- Webhook-based order status updates
- Multiple symbol support in single command

### Migration Considerations
- Path to newer `alpaca-py` SDK if needed
- WebSocket streaming for real-time order updates
- Enhanced error recovery mechanisms

---

**Document Owner:** Development Team  
**Stakeholders:** Trading Operations, System Architecture  
**Review Status:** Pending Technical Review