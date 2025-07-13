---
description: "Setup conda environment and verify ORB alerts system dependencies"
allowed-tools: ["bash"]
---

# Setup ORB Alerts Environment

Activate the conda environment and verify all dependencies for the ORB alerts system.

## Activate Environment
```bash
!source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && echo "✅ Conda environment 'alpaca' activated"
```

## Check Key Dependencies
```bash
!source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && python -c "
try:
    import alpaca_trade_api
    print('✅ alpaca-trade-api available')
except ImportError:
    print('❌ alpaca-trade-api missing')

try:
    import pandas as pd
    print('✅ pandas available')
except ImportError:
    print('❌ pandas missing')

try:
    import pytest
    print('✅ pytest available')
except ImportError:
    print('❌ pytest missing')

try:
    import pytz
    print('✅ pytz available')  
except ImportError:
    print('❌ pytz missing')
"
```

## Environment Variables Check
```bash
!echo "Checking environment variables..."
![ -f .env ] && echo "✅ .env file exists" || echo "❌ .env file missing"
![ -n "$ALPACA_API_KEY" ] && echo "✅ ALPACA_API_KEY set" || echo "⚠️  ALPACA_API_KEY not set in environment"
```