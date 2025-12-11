# Squeeze Alert Data Enhancement Recommendations

## Overview

This document provides recommendations for additional data fields to capture in squeeze alerts to enable statistical analysis and improve stock selection. The goal is to identify which metrics are most predictive of successful trading opportunities.

## Current Data Structure

### Existing Fields (as of 2025-12-11)

**Basic Information:**
- `symbol`: Stock ticker
- `timestamp`: Alert generation time
- `sent_to_users`: List of recipients
- `sent_count`: Number of recipients

**Price Action:**
- `first_price`: Price at start of squeeze window
- `last_price`: Price at end of squeeze window
- `percent_change`: Percentage change during squeeze window
- `window_trades`: Number of trades in the window
- `size`: Window duration in seconds
- `squeeze_threshold`: Threshold that triggered the alert

**Daily Performance:**
- `day_gain`: Percentage gain from market open
- `vwap`: Volume-weighted average price
- `premarket_high`: Highest price in pre-market
- `regular_hours_hod`: Regular hours high of day

**Volume & Float Metrics:**
- `volume_surge_ratio`: Current volume vs average volume
- `float_shares`: Number of shares in the float
- `float_rotation`: How many times the float has traded
- `float_rotation_percent`: Float rotation as percentage

**Liquidity:**
- `spread`: Bid-ask spread in dollars
- `spread_percent`: Bid-ask spread as percentage

**Display Metadata:**
- `day_gain_status`: Visual indicators (icon, color)
- `vwap_status`: Visual indicators
- `premarket_high_status`: Visual indicators with percent_off
- `regular_hours_hod_status`: Visual indicators with percent_off

---

## Recommended Additional Fields

### 1. Momentum & Timing Metrics

**Purpose:** Understand when and how fast the price is moving

```json
"time_since_market_open_minutes": 350,
"hour_of_day": 15,
"market_session": "power_hour",
"squeeze_number_today": 3,
"minutes_since_last_squeeze": 45,
"price_velocity_1min": 5.2,
"price_velocity_5min": 3.1,
"acceleration": 2.1
```

**Field Definitions:**
- `time_since_market_open_minutes`: Minutes elapsed since 9:30 AM ET
- `hour_of_day`: Hour in 24-hour format (9-16 for market hours)
- `market_session`: Categorical time period
  - `"early"` (9:30-11:00)
  - `"mid_day"` (11:00-14:00)
  - `"power_hour"` (14:00-15:00)
  - `"close"` (15:00-16:00)
- `squeeze_number_today`: Count of squeezes for this symbol today (1st, 2nd, 3rd, etc.)
- `minutes_since_last_squeeze`: Time since previous squeeze (null if first)
- `price_velocity_1min`: Percent change per minute over last 1 minute
- `price_velocity_5min`: Percent change per minute over last 5 minutes
- `acceleration`: Change in velocity (velocity_1min - velocity_5min)

**Why This Matters:**
- First squeezes often perform better than subsequent ones
- Time of day affects continuation probability (morning momentum vs afternoon fade)
- Acceleration helps identify if momentum is building or slowing

---

### 2. Volume Intelligence

**Purpose:** Deeper understanding of volume dynamics

```json
"window_volume": 6621844,
"window_volume_vs_1min_avg": 8.5,
"window_volume_vs_5min_avg": 6.2,
"relative_volume_for_time_of_day": 12.3,
"volume_trend": "increasing",
"buy_sell_imbalance": 0.65
```

**Field Definitions:**
- `window_volume`: Actual shares traded during squeeze window
- `window_volume_vs_1min_avg`: Ratio vs 1-minute rolling average
- `window_volume_vs_5min_avg`: Ratio vs 5-minute rolling average
- `relative_volume_for_time_of_day`: Volume vs typical volume for this hour
- `volume_trend`: Direction of volume change
  - `"increasing"`: Volume growing
  - `"decreasing"`: Volume declining
  - `"stable"`: Relatively flat
- `buy_sell_imbalance`: Ratio of buy-side vs sell-side volume (0-1 scale)
  - `> 0.5`: More buying pressure
  - `< 0.5`: More selling pressure
  - Requires tick-by-tick tape analysis

**Why This Matters:**
- Increasing volume during squeeze indicates strong interest
- Time-of-day relative volume filters unusual activity
- Buy/sell imbalance shows institutional interest direction

---

### 3. Price Level Context

**Purpose:** Understand where price is relative to key levels

```json
"distance_from_open_percent": 42.3,
"distance_from_prev_close_percent": 99.9,
"distance_from_vwap_percent": 2.6,
"distance_from_day_low_percent": 104.5,
"intraday_range_atr": 0.025,
"price_at_whole_dollar": false,
"nearest_whole_dollar": 1.00,
"distance_to_whole_dollar_percent": 14.3
```

**Field Definitions:**
- `distance_from_open_percent`: (current_price - open_price) / open_price * 100
- `distance_from_prev_close_percent`: (current_price - prev_close) / prev_close * 100
- `distance_from_vwap_percent`: (current_price - vwap) / vwap * 100
- `distance_from_day_low_percent`: (current_price - day_low) / day_low * 100
- `intraday_range_atr`: Average True Range as volatility measure
- `price_at_whole_dollar`: Boolean, true if within 2% of whole dollar
- `nearest_whole_dollar`: Closest whole dollar amount (0.00, 1.00, 2.00, etc.)
- `distance_to_whole_dollar_percent`: Distance to nearest whole dollar

**Why This Matters:**
- Stocks far from VWAP may revert
- Distance from HOD/LOD shows room to run
- Whole dollar levels act as psychological resistance/support

---

### 4. Market Context

**Purpose:** Understand broader market conditions

```json
"spy_percent_change_day": 0.45,
"spy_percent_change_concurrent": 0.02,
"market_correlation": 0.15,
"sector": "Technology",
"average_true_range_percent": 12.5
```

**Field Definitions:**
- `spy_percent_change_day`: SPY performance from open to current time
- `spy_percent_change_concurrent`: SPY change during same squeeze window
- `market_correlation`: Correlation coefficient (-1 to 1)
  - `> 0.5`: Stock follows market
  - `< -0.5`: Inverse correlation
  - `~0`: Independent movement
- `sector`: Stock sector/industry (if available)
- `average_true_range_percent`: ATR as percentage of price (volatility)

**Why This Matters:**
- Stocks moving independently of market are stronger signals
- Market direction affects continuation probability
- Sector rotation patterns

---

### 5. Continuation Signals

**Purpose:** Technical indicators predicting further movement

```json
"consecutive_green_candles": 5,
"higher_lows_count": 4,
"trend_strength_score": 0.82,
"pullback_depth_from_hod_percent": -16.23,
"consolidation_before_squeeze": true,
"consolidation_duration_minutes": 15
```

**Field Definitions:**
- `consecutive_green_candles`: Number of consecutive 1-min green candles
- `higher_lows_count`: Sequential higher lows on 1-min chart
- `trend_strength_score`: Composite score 0-1 (ADX-like indicator)
- `pullback_depth_from_hod_percent`: How far from HOD (negative = below HOD)
- `consolidation_before_squeeze`: Boolean, was there tight range before squeeze
- `consolidation_duration_minutes`: How long the consolidation lasted

**Why This Matters:**
- Consecutive candles show sustained pressure
- Higher lows indicate uptrend structure
- Consolidation before breakout = stronger move

---

### 6. Risk/Reward Metrics

**Purpose:** Quantify trade opportunity quality

```json
"estimated_stop_loss_price": 0.135,
"stop_loss_distance_percent": 5.6,
"potential_target_price": 0.180,
"risk_reward_ratio": 3.2,
"liquidity_score": 0.75,
"slippage_risk": "low"
```

**Field Definitions:**
- `estimated_stop_loss_price`: Suggested stop price (below recent support)
- `stop_loss_distance_percent`: Distance from current price to stop
- `potential_target_price`: Next resistance or HOD extension target
- `risk_reward_ratio`: (target - entry) / (entry - stop)
- `liquidity_score`: 0-1 composite score based on:
  - Volume
  - Spread
  - Float
  - Tape velocity
- `slippage_risk`: Categorical assessment
  - `"low"`: Tight spreads, high volume
  - `"medium"`: Moderate spreads/volume
  - `"high"`: Wide spreads, low volume

**Why This Matters:**
- Only trade setups with favorable risk/reward (>2:1)
- Liquidity affects execution quality
- Quantifies opportunity quality objectively

---

### 7. Historical Performance

**Purpose:** Learn from past behavior (requires database)

```json
"symbol_squeeze_history_30d": {
  "total_squeezes": 12,
  "successful_continuations": 8,
  "win_rate": 0.67,
  "average_gain_after_alert": 5.4,
  "average_duration_minutes": 12,
  "best_entry_time_pattern": "first_squeeze"
},
"day_of_week": "Wednesday",
"similar_pattern_win_rate": 0.71
```

**Field Definitions:**
- `total_squeezes`: Number of squeezes for this symbol in last 30 days
- `successful_continuations`: How many continued higher after alert
- `win_rate`: Percentage of successful continuations
- `average_gain_after_alert`: Mean percentage gain in following 5-30 minutes
- `average_duration_minutes`: Mean time gains were held
- `best_entry_time_pattern`: When entries typically worked best
  - `"first_squeeze"`: First squeeze of day
  - `"continuation_squeeze"`: Squeezes after consolidation
  - `"any"`: No pattern
- `day_of_week`: Monday-Friday
- `similar_pattern_win_rate`: Success rate for similar setups (same time, similar day_gain, etc.)

**Why This Matters:**
- Some symbols are "serial squeezers" with predictable behavior
- Historical win rates inform position sizing
- Pattern recognition improves over time

---

### 8. Order Flow & Tape

**Purpose:** Institutional activity and buying pressure (requires L1/L2 data)

```json
"large_prints_count": 3,
"largest_print_size": 25000,
"largest_print_side": "buy",
"tape_velocity_trades_per_second": 15.7,
"uptick_ratio": 0.68,
"bid_size_total": 50000,
"ask_size_total": 35000,
"bid_ask_size_ratio": 1.43
```

**Field Definitions:**
- `large_prints_count`: Number of prints > 10,000 shares in window
- `largest_print_size`: Size of largest single trade
- `largest_print_side`: Direction of largest trade
  - `"buy"`: At ask or above
  - `"sell"`: At bid or below
  - `"unknown"`: Mid-point
- `tape_velocity_trades_per_second`: Trade frequency (window_trades / size)
- `uptick_ratio`: Percentage of trades on uptick vs downtick (0-1)
- `bid_size_total`: Total shares on bid (from Level 1)
- `ask_size_total`: Total shares on ask (from Level 1)
- `bid_ask_size_ratio`: Bid size / ask size (>1 = more bid pressure)

**Why This Matters:**
- Large prints indicate institutional activity
- Uptick ratio shows buying vs selling pressure
- Bid/ask imbalance predicts short-term direction

---

### 9. Squeeze Quality Metrics

**Purpose:** Characterize the squeeze itself

```json
"squeeze_magnitude": 2.07,
"squeeze_duration_seconds": 479,
"trades_per_second": 0.33,
"price_efficiency": 0.85,
"volatility_during_squeeze": 0.15,
"volume_concentration": 0.72
```

**Field Definitions:**
- `squeeze_magnitude`: Percentage change (already captured as percent_change)
- `squeeze_duration_seconds`: Length of squeeze window (already captured as size)
- `trades_per_second`: window_trades / size
- `price_efficiency`: How direct was the move (0-1 scale)
  - `1.0`: Straight line up
  - `0.5`: Choppy, back-and-forth
  - Formula: (last_price - first_price) / sum(abs(tick_changes))
- `volatility_during_squeeze`: Standard deviation of price during window
- `volume_concentration`: How concentrated was volume (0-1)
  - `1.0`: All volume in few trades
  - `0.0`: Evenly distributed

**Why This Matters:**
- Efficient, smooth moves indicate conviction
- High volatility squeezes may reverse quickly
- Volume concentration shows institutional vs retail

---

## Implementation Priority

### Phase 1: High Priority (Implement First)

These fields provide immediate value with minimal complexity:

1. **Timing Metrics**
   - `time_since_market_open_minutes`
   - `hour_of_day`
   - `market_session`
   - `squeeze_number_today`
   - `minutes_since_last_squeeze`

2. **Volume Expansion**
   - `window_volume`
   - `window_volume_vs_1min_avg`
   - `window_volume_vs_5min_avg`
   - `volume_trend`

3. **Price Level Context**
   - `distance_from_open_percent`
   - `distance_from_prev_close_percent`
   - `distance_from_vwap_percent`
   - `distance_from_day_low_percent`

4. **Risk/Reward**
   - `estimated_stop_loss_price`
   - `stop_loss_distance_percent`
   - `potential_target_price`
   - `risk_reward_ratio`

5. **Market Context**
   - `spy_percent_change_day`
   - `spy_percent_change_concurrent`

**Estimated Implementation Time:** 2-4 hours
**Data Requirements:** Current real-time data + SPY quote

---

### Phase 2: Medium Priority

These fields require additional computation or short-term data tracking:

6. **Momentum Indicators**
   - `price_velocity_1min`
   - `price_velocity_5min`
   - `acceleration`
   - `consecutive_green_candles`
   - `higher_lows_count`

7. **Continuation Signals**
   - `consolidation_before_squeeze`
   - `consolidation_duration_minutes`
   - `trend_strength_score`

8. **Order Flow Basics**
   - `uptick_ratio`
   - `large_prints_count`
   - `largest_print_size`
   - `tape_velocity_trades_per_second`

**Estimated Implementation Time:** 4-8 hours
**Data Requirements:** Need to track recent candles and tick data

---

### Phase 3: Low Priority (Nice to Have)

These require significant infrastructure or external data:

9. **Historical Analytics**
   - `symbol_squeeze_history_30d`
   - `similar_pattern_win_rate`
   - (Requires database of past alerts and outcomes)

10. **Advanced Order Flow**
    - `bid_size_total`
    - `ask_size_total`
    - `bid_ask_size_ratio`
    - (Requires Level 2 data subscription)

11. **Advanced Market Context**
    - `market_correlation`
    - `sector`
    - (Requires historical correlation analysis and sector mapping)

**Estimated Implementation Time:** 16+ hours
**Data Requirements:** Database, Level 2 data feed, sector classification

---

## Statistical Analysis Strategy

Once enhanced data is collected, perform these analyses:

### 1. Univariate Analysis

For each field, analyze:
- **Correlation with success**: Does higher/lower value predict continuation?
- **Distribution analysis**: What's the typical range? Outliers?
- **Threshold identification**: Is there a cutoff value that matters?

**Example:**
```
Field: squeeze_number_today
Finding: First squeeze of day has 72% continuation rate
         Second+ squeezes have 48% continuation rate
Action: Prioritize first squeezes
```

### 2. Multivariate Analysis

Identify field combinations that predict success:

**Example Decision Tree:**
```
If squeeze_number_today == 1
  AND time_since_market_open_minutes < 120
  AND risk_reward_ratio > 2.5
  AND volume_surge_ratio > 50
  → 83% success rate (high-confidence setup)
```

### 3. Cluster Analysis

Group similar alerts and identify which clusters perform best:

**Example Clusters:**
- **Morning Gappers**: High day_gain, early session, first squeeze
- **Mid-Day Breakouts**: Moderate day_gain, consolidation before squeeze
- **Power Hour Runners**: Late session, high float_rotation

### 4. Time-Based Patterns

Analyze performance by:
- Hour of day
- Day of week
- Time since market open
- Market conditions (SPY up/down day)

### 5. Feature Importance Ranking

Use machine learning (Random Forest, XGBoost) to rank fields by predictive power:

**Example Output:**
```
1. squeeze_number_today         (importance: 0.18)
2. risk_reward_ratio            (importance: 0.15)
3. volume_surge_ratio           (importance: 0.13)
4. time_since_market_open       (importance: 0.11)
5. distance_from_vwap_percent   (importance: 0.09)
...
```

---

## Success Metrics to Track

To validate these enhancements, track:

1. **Continuation Rate**: % of alerts that moved higher in next 5/15/30 minutes
2. **Average Gain**: Mean % gain after alert (for continuations)
3. **Average Loss**: Mean % loss after alert (for reversals)
4. **Win Rate by Setup**: Success rate for different field combinations
5. **False Positive Rate**: % of alerts that immediately reversed
6. **Opportunity Cost**: % of alerts that continued but with poor R:R

---

## Data Collection Recommendations

### Storage Format

Store enhanced alerts in both:
1. **JSON files** (current format) - for easy inspection
2. **SQLite/PostgreSQL database** - for efficient querying and analysis

### Database Schema Suggestion

```sql
CREATE TABLE squeeze_alerts (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,

    -- Price action
    first_price DECIMAL(10, 4),
    last_price DECIMAL(10, 4),
    percent_change DECIMAL(10, 4),

    -- Add all recommended fields...

    -- Outcome tracking (filled in later)
    price_5min_later DECIMAL(10, 4),
    price_15min_later DECIMAL(10, 4),
    price_30min_later DECIMAL(10, 4),
    max_gain_5min DECIMAL(10, 4),
    max_gain_15min DECIMAL(10, 4),
    continuation_success BOOLEAN,

    INDEX idx_symbol_timestamp (symbol, timestamp),
    INDEX idx_timestamp (timestamp)
);
```

### Outcome Tracking

Critical: Track what happens AFTER each alert:
- Price at +5, +15, +30 minutes after alert
- Maximum gain achieved in following 30 minutes
- Whether price continued higher (success) or reversed (failure)

This outcome data enables supervised learning and pattern validation.

---

## Example Enhanced Alert

```json
{
  "symbol": "ATPC",
  "timestamp": "2025-12-11T15:20:38.869008-05:00",

  // Existing fields...
  "first_price": 0.1401,
  "last_price": 0.143,
  "percent_change": 2.07,
  "day_gain": 99.86,
  "volume_surge_ratio": 134.7,
  "float_rotation": 30.52,

  // Phase 1: High Priority additions
  "time_since_market_open_minutes": 350,
  "hour_of_day": 15,
  "market_session": "power_hour",
  "squeeze_number_today": 3,
  "minutes_since_last_squeeze": 45,
  "window_volume": 6621844,
  "distance_from_open_percent": 42.3,
  "distance_from_vwap_percent": 2.6,
  "estimated_stop_loss_price": 0.135,
  "stop_loss_distance_percent": 5.6,
  "risk_reward_ratio": 3.2,
  "spy_percent_change_day": 0.45,

  // Phase 2: Medium Priority
  "price_velocity_1min": 5.2,
  "acceleration": 2.1,
  "consecutive_green_candles": 5,
  "uptick_ratio": 0.68,

  // Phase 3: Low Priority (future)
  "symbol_squeeze_history_30d": {
    "win_rate": 0.67,
    "average_gain_after_alert": 5.4
  }
}
```

---

## Next Steps

1. **Review & Prioritize**: Review these recommendations and select Phase 1 fields to implement
2. **Code Implementation**: Modify squeeze alert generation code to capture new fields
3. **Data Collection**: Begin collecting enhanced alerts for 2-4 weeks
4. **Statistical Analysis**: Run correlation studies and feature importance analysis
5. **Model Development**: Build predictive model for alert quality
6. **Refinement**: Iterate based on findings

---

## Questions for Consideration

Before implementation, decide:

1. **Data Retention**: How long to keep historical alerts? (Recommend: 1 year minimum)
2. **Database Choice**: SQLite for simplicity or PostgreSQL for scale?
3. **Real-time vs Batch**: Calculate fields in real-time or backfill?
4. **Privacy**: Any PII concerns with tracking sent_to_users?
5. **Performance**: Can system handle additional computation without lag?
6. **External Data**: Budget for Level 2 data feeds if needed?

---

## References & Resources

- **Technical Indicators**: [TA-Lib](https://ta-lib.org/) for ATR, RSI, etc.
- **Statistical Analysis**: [pandas](https://pandas.pydata.org/), [scipy](https://scipy.org/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/), [xgboost](https://xgboost.readthedocs.io/)
- **Database**: [SQLite](https://www.sqlite.org/), [PostgreSQL](https://www.postgresql.org/)
- **Market Data**: Current Alpaca API already provides most needed data

---

**Document Version:** 1.0
**Last Updated:** 2025-12-11
**Author:** Claude Code Analysis
