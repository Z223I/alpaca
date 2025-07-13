---
description: "Start the ORB alerts system with proper environment setup"
allowed-tools: ["bash"]
---

# Start ORB Alerts System

Start the ORB alerts monitoring system with proper conda environment activation.

$ARGUMENTS

## Quick Start (Test Mode)
```bash
!source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && python3 code/orb_alerts.py --test
```

## Full Production Start
```bash
!source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && python3 code/orb_alerts.py
```

## Start with Verbose Logging
```bash
!source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && python3 code/orb_alerts.py --verbose
```

## Custom Symbols File
```bash
!source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && python3 code/orb_alerts.py --symbols-file data/symbols.csv
```

## Show Daily Summary
```bash
!source ~/miniconda3/etc/profile.d/conda.sh && conda activate alpaca && python3 code/orb_alerts.py --summary
```

> **Note**: Use Ctrl+C to stop the alerts system when running in production mode.