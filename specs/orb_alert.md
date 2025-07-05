# ORB Alerts

## Background

Here are the results of the ORB trading strategy performed in this repo:

PCA Analysis Results - July 3, 2025 Data

  Dataset Overview:
  - 37 symbols loaded from data/20250703.csv
  - 22 symbols had sufficient data (≥90 samples) for PCA analysis
  - 1,980 total feature rows generated (90 samples × 22 symbols)

  PCA Component Analysis:

  PC1 (82.31% variance explained):
  Top loadings indicate ORB breakout momentum patterns:
  - close_vs_orb_high: 0.3852 (price relative to opening range high)
  - high_vs_orb_high: 0.3763 (highs relative to ORB)
  - close_vs_orb_low: 0.3735 (price relative to opening range low)
  - session_progress: 0.3515 (time-based momentum)

  PC2 (8.54% variance explained):
  Captures intraday volatility and volume dynamics:
  - volume_ratio: 0.6894 (volume surge patterns)
  - intraday_range: 0.4325 (volatility spikes)
  - upper_wick: 0.3426 (rejection at highs)

  PC3 (3.78% variance explained):
  Technical indicator divergences:
  - close_vs_ema: -0.5689 (EMA deviation)
  - close_vs_vwap: -0.5234 (VWAP deviation)
  - return_prev_close: 0.4289 (gap behavior)

  Key Trading Insights:

  1. ORB Dominance: 82.31% of price behavior is explained by opening range breakout patterns, validating the ORB strategy focus.
  2. Volume Confirmation: PC2 shows volume surges are the secondary driver, critical for confirming breakout validity.
  3. Technical Divergence: PC3 captures when price moves against key indicators (EMA/VWAP), often signaling reversal points.

  Symbols Successfully Analyzed:
  BMNR, KLTO, VOR, ONDS, VERO, BBAI, JOBY, WBUY, ADIL, ASTI, BEEM, CYN, GCTK, GNS, HIMS, IBO, NAAS, NVTS, PSTV, QS, QXO, RZLV

  Strategic Application:
  The 82.31% PC1 variance suggests most intraday price action follows predictable ORB patterns, making this an ideal dataset for developing systematic breakout trading strategies
   with high statistical confidence.

## High Level Requirements

You are a highly skilled Python developer and architect.
You are a highly skilled statistician.
You are working in an atom - molecules architecture.

## Mid Level Requirements

Use the ORB class in orb.py.
Create a private method for PCA data standardization.
It is to conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Create PyTest.
Test.

## Low Level Requirements

We will create another method later to call this new method.

### Step 1

Use self.pca_data.
Remove the symbol and timestamp fields.
Extract the volume
Extract the vector_angle

### Step 2

Perform the same statistical standardization across the remaining columns.  They are all prices.
Standardize the volume.
Standardize the vector_angle.
Put all the columns back together side by side.

### Step 3

Create local is_debugging variable and set it to True.
If debugging, print the resulting dataframe.



