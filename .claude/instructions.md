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

## Common Patterns

- Testing: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && ./test.sh orb`
- Running alerts: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && python3 code/orb_alerts.py`
- Linting: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && flake8 code/`