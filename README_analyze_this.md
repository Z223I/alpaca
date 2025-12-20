# Stock Price Movement Prediction - Feature Analysis

## Current Variables Analysis

Based on `analysis/predict_squeeze_outcomes.py`, the model uses **11 independent features** to predict whether a squeeze alert will reach gain thresholds (1.5%-7%) within 10 minutes.

### Current Features Assessment

**Strong Predictive Variables:**
1. **ema_spread_pct** - Price-normalized EMA momentum `(ema_9 - ema_21) / price * 100`
   - ✓ Good: Captures short-term momentum independent of price level
   - ⚠️ Concern: ~10% missing data

2. **window_volume_vs_1min_avg** - Volume surge ratio
   - ✓ Good: Volume is crucial for price movement prediction
   - ✓ Strong signal: High volume = more likely to sustain movement

3. **distance_from_vwap_percent** - Distance from VWAP
   - ✓ Good: Mean reversion/momentum indicator
   - ✓ Reasonable: Overextended stocks may reverse or continue

4. **spy_percent_change_concurrent** - SPY correlation at alert time
   - ✓ Good: Market regime affects individual stock behavior
   - ⚠️ Limitation: Only captures correlation at ONE moment, not direction

5. **day_gain** - Intraday gain percentage
   - ✓ Good: Momentum continuation vs. mean reversion
   - ⚠️ Concern: May have high correlation with distance_from_vwap

**Moderate Predictive Variables:**
6. **macd_histogram** - MACD momentum
   - ⚠️ Problem: ~15% missing (will be 86% when switched to daily)
   - ✓ Reasonable: Standard momentum indicator
   - ⚠️ Concern: May be redundant with ema_spread_pct

7. **market_session** - Time of day (extended/early/mid_day/power_hour/close)
   - ✓ Good: Volatility/liquidity varies by time
   - ⚠️ Limitation: Categorical, loses intraday granularity

8. **spread_percent** - Bid-ask spread
   - ✓ Good: Liquidity indicator (tight spreads = easier to profit)
   - ⚠️ Concern: May correlate with price_category

**Weak/Questionable Variables:**
9. **price_category** - Stock price bins (<$2, $2-5, $5-10, etc.)
   - ⚠️ Questionable: Penny stocks behave differently, but loses information through binning
   - ✓ Helps: Captures volatility differences by price level

10. **squeeze_number_today** - Nth squeeze of the day
   - ⚠️ Weak: High squeeze number may indicate "played out" momentum
   - ⚠️ Limitation: Doesn't account for time spacing or success rate of prior squeezes

11. **minutes_since_last_squeeze** - Time since previous squeeze
   - ⚠️ Weak: More relevant might be whether LAST squeeze succeeded
   - ⚠️ Missing context: Doesn't know if stock is in consolidation or exhaustion

---

## Critical Missing Variables

### HIGH PRIORITY - Momentum & Trend

1. **RSI (Relative Strength Index)**
   - Why: Identifies overbought/oversold conditions
   - Implementation: 14-period RSI at squeeze time
   - Value: RSI 30-70 may predict continuation; <30 or >70 may predict reversal

2. **Price vs. Moving Averages (SMA_50, SMA_200 position)**
   - Why: Establishes trend context
   - Implementation: `(current_price - SMA_50) / SMA_50 * 100`
   - Value: Squeezes above long-term MA have stronger trend support

3. **ATR (Average True Range) - Normalized**
   - Why: Volatility context (ATR/price * 100)
   - Implementation: 14-period ATR percentage
   - Value: High ATR = more likely to hit thresholds (and stop losses)

4. **Bollinger Band Position**
   - Why: Identifies squeeze vs. expansion regimes
   - Implementation: `(price - BB_lower) / (BB_upper - BB_lower)` (0-1 scale)
   - Value: Squeezes near lower band = potential reversal; middle band = continuation

5. **Recent Price Velocity**
   - Why: Rate of price change momentum
   - Implementation: `(current_price - price_5min_ago) / price_5min_ago * 100`
   - Value: Accelerating momentum vs. decelerating

### HIGH PRIORITY - Volume & Liquidity

6. **Volume Trend (Volume MA Ratio)**
   - Why: Sustained volume vs. one-time spike
   - Implementation: `current_volume / SMA_volume_20periods`
   - Value: You have surge ratio, but not sustained trend

7. **Order Book Imbalance (if available)**
   - Why: Bid/ask size ratio predicts direction
   - Implementation: `(bid_volume - ask_volume) / (bid_volume + ask_volume)`
   - Value: Strong buy pressure = more likely to continue up

8. **Recent Volume Profile**
   - Why: Volume pattern in last 5-10 minutes
   - Implementation: Slope of volume over last 5 bars
   - Value: Increasing volume = stronger signal

### MEDIUM PRIORITY - Market Context

9. **Sector/Industry Momentum**
   - Why: Stock moves with peers
   - Implementation: Sector ETF % change (XLF, XLE, XLK, etc.)
   - Value: Sector tailwind increases success probability

10. **VIX Level**
    - Why: Market volatility regime
    - Implementation: Current VIX value (or % change)
    - Value: High VIX = higher individual stock volatility

11. **Time Since Market Open (continuous)**
    - Why: More granular than market_session
    - Implementation: Minutes since 9:30 AM
    - Value: First 30 minutes very different from 10:00-10:30

12. **News/Catalyst Indicator**
    - Why: Fundamental catalysts drive sustained moves
    - Implementation: Unusual volume + price spike in last hour (boolean)
    - Value: Distinguishes technical squeeze from news-driven

### MEDIUM PRIORITY - Historical Context

13. **Success Rate of Prior Squeezes Today**
    - Why: Better than just counting squeezes
    - Implementation: `successful_squeezes_today / total_squeezes_today`
    - Value: If prior squeezes worked, stock may still have momentum

14. **Average Gain of Prior Squeezes Today**
    - Why: Magnitude of prior success matters
    - Implementation: Mean of max_gain_percent for today's earlier squeezes
    - Value: Stock showing consistent follow-through vs. fading

15. **Distance from Day High/Low** (already had distance_from_day_low but removed)
    - Why: Proximity to breakout levels
    - Implementation: `(day_high - current_price) / current_price * 100`
    - Value: Near day high = potential breakout; far = pullback opportunity

16. **Candle Pattern Features**
    - Why: Price action patterns (hammer, doji, engulfing)
    - Implementation: Boolean flags or numeric scoring
    - Value: Certain patterns precede continuation/reversal

### LOWER PRIORITY - Advanced

17. **Options Flow (if available)**
    - Why: Smart money positioning
    - Implementation: Call/put volume ratio
    - Value: Institutions may front-run moves

18. **Short Interest**
    - Why: Short squeezes = explosive moves
    - Implementation: Days-to-cover or % float short
    - Value: High short interest + squeeze = potential gamma squeeze

19. **Institutional Ownership**
    - Why: Liquidity and volatility characteristics
    - Implementation: % shares held by institutions
    - Value: High ownership = less float for squeezes

20. **Correlation with Crypto (BTC/ETH)**
    - Why: Some growth stocks correlate with risk-on assets
    - Implementation: BTC % change concurrent
    - Value: Alternative to just SPY correlation

---

## Recommended Implementation Order

### Phase 1 (Immediate - Easy to Calculate)
1. RSI (14-period)
2. ATR percentage
3. Price vs SMA_50 and SMA_200
4. Time since market open (continuous)
5. Bollinger Band position
6. Recent price velocity (5-min change rate)

### Phase 2 (Medium Effort)
7. Volume MA ratio (sustained volume trend)
8. Success rate of prior squeezes today
9. Average gain of prior squeezes today
10. Sector ETF correlation
11. VIX level
12. Distance from day high

### Phase 3 (Advanced - If Data Available)
13. Order book imbalance
14. Options flow indicators
15. News/catalyst detection
16. Short interest metrics

---

## Model Performance Context

Current XGBoost performance (5% threshold):
- **F1-Score: 0.57** (moderate)
- **Precision: 0.55** (55% of predictions correct)
- **Recall: 0.60** (catches 60% of actual winners)
- **ROC-AUC: 0.68** (better than random 0.50)

**Expected improvement with better features: F1 → 0.65-0.75**

The moderate performance suggests your features capture some signal but are missing key drivers. The variables suggested above—especially RSI, ATR, Bollinger Bands, and improved volume metrics—should significantly improve predictive power.

---

## Data Sources

- **Current Features**: `analysis/squeeze_alerts_independent_features.csv`
- **Model Training**: `analysis/predict_squeeze_outcomes.py`
- **Feature Engineering**: `analysis/squeeze_alerts_statistical_analysis.py`
- **Model Performance**: `analysis/prediction_summary_*.txt`

## Key Findings

1. **Current features are statistically independent** (VIF < 5) - good foundation
2. **Missing critical momentum indicators** (RSI, ATR, Bollinger Bands)
3. **Volume analysis is incomplete** - have surge ratio but not sustained trend
4. **Historical context underutilized** - not tracking success patterns of prior squeezes
5. **Market context limited** - only SPY, missing sector/VIX/broader regime
