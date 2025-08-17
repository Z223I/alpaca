# Fix Analyzer - Critical Issue Documentation

## Problem Summary

The backtesting analyzer (`code/analyze_backtesting_results.py`) is generating **SYNTHETIC/FAKE DATA** for heatmaps instead of reading actual simulation results. This causes a critical discrepancy where:

- **Real logs show**: 0 alerts generated (correct)
- **Heatmap shows**: 8-23 alerts (synthetic data from math formulas)

## Root Cause

The `create_parameter_combinations()` method in `analyze_backtesting_results.py:93-136` generates synthetic data using mathematical formulas instead of reading actual run results:

```python
def create_parameter_combinations(self):
    """Create synthetic data based on parameters.json for demonstration."""
    # Creates FAKE data using formulas like:
    base_alerts = 20
    threshold_factor = max(0.1, 1.5 - threshold)  
    timeframe_factor = 1.2 if timeframe in [20, 25, 30] else 0.8
    alerts_sent = max(0, int(base_alerts * threshold_factor * timeframe_factor + noise))
```

## Issues to Fix

### 1. Directory Structure Mismatch
- **Current code looks for**: `runs/run_*` (old flat structure)
- **Actual structure is**: `runs/{date}/{symbol}/run_{date}_tf{timeframe}_th{threshold}_{uuid}/`

### 2. Missing Parameter Extraction
- Code has no way to extract timeframe/threshold from run directory names
- Currently uses hardcoded defaults: `timeframe=30, threshold=0.65`

### 3. Results Reading Logic
- `collect_run_data()` exists but is ignored for heatmap generation
- `_analyze_run_results()` method doesn't exist or is broken
- No logic to read `simulation_results_*.json` files

## Fix Strategy

### Step 1: Update Directory Scanning
Replace the `collect_run_data()` method to scan the new nested structure:

```python
def collect_run_data(self):
    """Collect data from all completed backtesting runs."""
    runs_dir = Path("runs")
    
    for date_dir in runs_dir.glob("2025-*"):  # Date directories
        for symbol_dir in date_dir.glob("*"):  # Symbol directories  
            for run_dir in symbol_dir.glob("run_*"):  # Individual runs
                # Extract parameters from run_dir.name
                # Read simulation_results_*.json
                # Add to self.results_data
```

### Step 2: Extract Parameters from Directory Names
Parse run directory names to extract actual parameters:

```python
def extract_parameters_from_run_name(self, run_name):
    """Extract timeframe and threshold from run directory name.
    
    Example: run_2025-08-04_tf10_th0.65_e101f2f1
    Returns: {'timeframe': 10, 'threshold': 0.65, 'date': '2025-08-04'}
    """
    import re
    pattern = r'run_(\d{4}-\d{2}-\d{2})_tf(\d+)_th([\d.]+)_([a-f0-9]+)'
    match = re.match(pattern, run_name)
    if match:
        return {
            'date': match.group(1),
            'timeframe': int(match.group(2)),
            'threshold': float(match.group(3)),
            'uuid': match.group(4)
        }
    return None
```

### Step 3: Read Actual Simulation Results
Read the `simulation_results_*.json` files for real alert counts:

```python
def read_simulation_results(self, run_dir):
    """Read simulation results from logs directory."""
    logs_dir = run_dir / "logs"
    
    for date_dir in logs_dir.glob("2025-*"):
        for results_file in date_dir.glob("simulation_results_*.json"):
            with open(results_file, 'r') as f:
                data = json.load(f)
                return {
                    'total_alerts': data.get('total_alerts', 0),
                    'alerts_generated': len(data.get('generated_alerts', [])),
                    'symbol_results': data.get('symbol_results', [])
                }
    return {'total_alerts': 0, 'alerts_generated': 0, 'symbol_results': []}
```

### Step 4: Replace Synthetic Data Generation
Remove `create_parameter_combinations()` and use real data from `collect_run_data()`:

```python
def create_heatmap(self):
    """Create heatmap showing alerts by timeframe vs threshold."""
    if not self.results_data:
        print("❌ No real results data found - cannot create heatmap")
        return
        
    # Convert results_data to DataFrame
    df = pd.DataFrame(self.results_data)
    
    # Pivot data for heatmap using REAL results
    heatmap_data = df.pivot(index='threshold', columns='timeframe', values='total_alerts')
    
    # Rest of heatmap code unchanged...
```

## Implementation Priority

1. **High Priority**: Fix `collect_run_data()` to read new directory structure
2. **High Priority**: Add parameter extraction from run directory names  
3. **High Priority**: Read actual `simulation_results_*.json` files
4. **Medium Priority**: Remove synthetic data generation completely
5. **Low Priority**: Add validation to ensure real data is being used

## Verification Steps

After fixing:

1. Run analyzer: `python3 code/analyze_backtesting_results.py`
2. Verify heatmap shows 0 alerts (matching logs)
3. Check that parameters are correctly extracted from run names
4. Confirm no synthetic data warnings in output

## Files to Modify

- `code/analyze_backtesting_results.py` (main fixes)
- Add validation that heatmap data matches log data
- Consider adding a `--verify-real-data` flag for debugging

## Expected Outcome

After fixing, the heatmap should show all zeros (or actual alert counts) matching the simulation logs, eliminating the discrepancy between real results and visualization.