# Momentum Alerts

## High Level Requirements

You are a highly skilled Python developer and software architect.

## Mid Level Requirements

Update code/alpaca.py -plot based on low level requirements.

### Background Information

#### Review

##### Alpaca Plot

code/alpaca.py -plot

##### Volume Surge

```bash
python code/volume_profile.py --symbol COOT --days 1 --timeframe 5Min --time-per-profile DAY
Loading 1 days of 5Min data for COOT...
Calculating volume profiles...
Analysis complete!
============================================================
VOLUME PROFILE ANALYSIS SUMMARY
============================================================
Configuration:
  Time Per Profile: DAY
  Price Mode: AUTOMATIC
  Value Area %: 70.0%

Results:
  Total Profiles: 2
  Time Range: 2025-10-14 09:25:00-04:00 to 2025-10-15 09:20:00-04:00
  Average POC: $2.68
  POC Range: $2.14 - 3.21
  Avg Value Area Width: $1.44

Latest Profile:
  📅 Period: 2025-10-15 04:00:00-04:00 to 2025-10-15 09:20:00-04:00
  🎯 Point of Control: $3.21
  📊 Value Area: $3.03 - $4.64
  📈 Profile Range: $2.83 - $5.09
  📦 Total Volume: 96,037,466
============================================================

Generating volume profile chart...
📊 Volume profile chart saved: ./historical_data/2025-10-15/volume_profile_output/COOT_volume_profile_20251014.png
💾 Volume profile data saved: ./historical_data/2025-10-15/volume_profile_output/COOT_volume_profile_20251014.json
```

### Standards
Conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Test.

## Low Level Requirements

### Calculate Major Resistance

1) Execute Volume Profile

python code/volume_profile.py --symbol {symbol} --days 1 --timeframe 5Min --time-per-profile DAY

2) Collect Major Resistance

Read the .JSON ouput from volume_profile.py.  Format: ./historical_data/YYYY-MM-DD/volume_profile_output/COOT_volume_profile_YYYYMMDD.json

There will be a series of "profiles".  Collect major_resistance as a list of the "profile_high" fields.

3) Plot Major Resistance

Pass the major_resistance list to the plotting routine.  Plot the lines horizontally because they are prices and make them a bright green.

4) Test

Test with the symbol IRBT.



