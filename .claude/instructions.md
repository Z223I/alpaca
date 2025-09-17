# Alpaca ORB Trading System Instructions

## Environment Setup

When running any Python code, tests, or commands from this repository, always use the conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca
```

## Project Context

This is an ORB (Opening Range Breakout) trading system built for Alpaca API. Key components:

- **Main ORB alerts system**: `code/orb_alerts.py`
- **Historical data storage**: `historical_data/` directory with date-based organization
- **Test suite**: Run with `./test.sh` or `./test.sh orb` for ORB-specific tests
- **Dependencies**: alpaca-trade-api, pandas, pytest, pytz (all in conda environment)

## Important Notes

- All Python commands should be prefixed with the conda activation
- Tests require the alpaca conda environment to access proper dependencies
- The system uses Eastern Time (ET) for all trading operations
- Historical data files are automatically cleaned up (only latest per symbol kept)
- Environment variables are loaded from `.env` file for API credentials

## Git Repository Management

- **NEVER create git tags** without explicit user permission
- Git tags should only be created by the user to mark important milestones
- When asked to "list tags" or review tags, only display existing tags, do not create new ones
- If tag creation is needed, ask the user to create them manually

## Common Patterns

- Testing: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && ./test.sh orb`
- Running alerts: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && python3 code/orb_alerts.py`
- Linting: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && flake8 code/`

## Temporary Files

- When you create files such as scripts that should be deleted, stick them in the ./tmp directory.

## Test Execution Notes (Claude Code)

**IMPORTANT**: Standard conda activation commands often fail in the Claude Code environment. Use these workarounds:

### Working Test Execution Methods:
1. **Direct Python Path** (RECOMMENDED):
   ```bash
   ~/miniconda3/envs/alpaca/bin/python -m pytest tests/ -v
   ```

2. **Specific Test Files**:
   ```bash
   ~/miniconda3/envs/alpaca/bin/python -m pytest tests/test_alpaca_after_hours.py -v
   ```

3. **Quick Test Runs** (no verbose):
   ```bash
   ~/miniconda3/envs/alpaca/bin/python -m pytest tests/ -q
   ```

### Common Issues Encountered:
- `source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca` often fails with "CondaError: Run 'conda init'"
- Standard `./test.sh` script may not work due to environment activation issues
- Environment variables from conda may not be properly loaded

### Debugging Test Issues:
- Check if pytest is available: `~/miniconda3/envs/alpaca/bin/python -c "import pytest; print('pytest available')"`
- List installed packages: `~/miniconda3/envs/alpaca/bin/pip list`
- For import issues, check sys.path in test files or use direct module paths

### Test Categories:
- **After-hours trading**: `tests/test_alpaca_after_hours.py` (26 tests)
- **Risk management**: Look for tests with "risk_calculation" in name
- **ORB functionality**: `tests/test_orb.py`
- **Alert formatting**: `tests/test_alert_formatter.py`
- **Integration tests**: May be skipped or removed if too complex

## Telegram Integration

### Sending Messages to Bruce

Use the `TelegramPoster` class from `atoms.telegram.telegram_post` to send messages:

```python
from atoms.telegram.telegram_post import TelegramPoster

# Send message to Bruce specifically
telegram_poster = TelegramPoster()
result = telegram_poster.send_message_to_user(message, "bruce", urgent=False)

# Check result
if result['success']:
    print("✅ Message sent to Bruce")
else:
    print(f"❌ Failed: {result.get('errors', [])}")
```

### Quick Telegram Patterns:

1. **Send to Bruce only**:
   ```python
   from atoms.telegram.telegram_post import TelegramPoster
   poster = TelegramPoster()
   poster.send_message_to_user("Your message here", "bruce")
   ```

2. **Send alert with formatting**:
   ```python
   from atoms.telegram.telegram_post import send_alert
   send_alert("Alert message", level='info')  # levels: info, warning, error, success
   ```

3. **Send to all users**:
   ```python
   from atoms.telegram.telegram_post import send_message
   send_message("Message for everyone")
   ```

### Telegram Message Formatting:
- Messages support markdown formatting
- Use `**bold**` for bold text
- Use `*italic*` for italic text
- Use emojis for visual appeal (📊 🚨 ✅ ❌ ⚠️ 🔧)
- Keep messages concise and informative

## CRITICAL API NOTES

### Alpaca Bar Object Attributes

**⚠️ CRITICAL**: Alpaca Bar objects use single-letter attributes, NOT full words:

```python
# ❌ WRONG - These will cause AttributeError
bar.open   # Does not exist
bar.high   # Does not exist 
bar.low    # Does not exist
bar.close  # Does not exist
bar.volume # Does not exist

# ✅ CORRECT - Use single letters
bar.o  # Open price
bar.h  # High price
bar.l  # Low price  
bar.c  # Close price
bar.v  # Volume
bar.t  # Timestamp
```

**Common mistake locations:**
- `atoms/alerts/superduper_alert_filter.py` (MACD data collection)
- Any code processing Alpaca market data bars
- MACD analysis failures often trace back to this attribute naming issue

**Impact:** Using wrong attributes causes "BLIND FLIGHT" MACD conditions instead of proper 🟢/🟡/🔴 analysis.

## Code Development

When developing new code or modifying existing code, always follow this checklist:

- **Conform to repo standards**: Follow existing code patterns, naming conventions, and architectural decisions
- **Check linting compliance**: Run `~/miniconda3/envs/alpaca/bin/python -m flake8` on modified files
- **Check for VS Code integration errors**: Use diagnostic tools to ensure no integration issues
- **Test**: Write and run appropriate tests to verify functionality