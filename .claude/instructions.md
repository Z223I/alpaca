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