# Momentum Alert Popup Implementation

## Overview
Implemented a real-time popup notification system for the Market Sentinel web interface (`public_html/index.html`) that displays momentum alerts sent by the `momentum_alerts.py` service.

## Components Created

### 1. API Endpoint: `cgi-bin/api/momentum_alerts_api.py`
- **Purpose**: Provides REST API access to momentum alerts
- **Location**: Reads from `historical_data/{YYYY-MM-DD}/momentum_alerts_sent/bullish/`
- **Key Features**:
  - Only reads alerts that were actually sent to Telegram users (from `momentum_alerts_sent`)
  - Only shows bullish momentum alerts
  - Supports two actions:
    - `recent`: Get alerts from last N minutes (default 60)
    - `poll`: Get new alerts since a specific timestamp
  - Returns JSON with alert data including price, momentum, volume, etc.

### 2. Web Interface Updates: `public_html/index.html`

#### CSS Styling (lines 708-816)
- Professional dark-themed popup overlay
- Animated fade-in and slide-in effects
- Responsive design with max-width 600px
- Styled buttons for "OK" and "View Chart" actions

#### JavaScript Functionality (lines 2767-2899)
- **Alert Polling**: Checks for new alerts every 10 seconds
- **Alert Queue**: Manages multiple alerts, showing them one at a time
- **Alert Display**: Formats and displays alert data in a popup
- **Chart Integration**: "View Chart" button opens the stock chart

#### HTML Structure (lines 2902-2916)
- Fixed overlay covering entire viewport
- Centered popup with alert content
- Two action buttons: OK and View Chart

## How It Works

1. **Alert Generation**: 
   - `momentum_alerts.py` detects momentum criteria and sends Telegram alerts
   - Saves sent alerts to `historical_data/{date}/momentum_alerts_sent/bullish/`

2. **Web Polling**:
   - Page polls API every 10 seconds for new alerts
   - Tracks last check time to only fetch new alerts

3. **Alert Display**:
   - New alerts appear as popup overlays
   - Shows: Date, Time, Price, VWAP, EMA9, Momentum indicators, Volume, etc.
   - Multiple alerts are queued and shown sequentially

4. **User Actions**:
   - **OK Button**: Closes popup, shows next alert if queued
   - **View Chart Button**: Closes popup and loads the stock chart for that symbol

## Alert Format

The popup displays:
- 📅 Date and ⏰ Time at the top
- 💰 Current Price
- 🌅 Market Open Price and gain percentage (if available)
- 📊 VWAP and 📈 EMA9 indicators
- ⚡ Momentum metrics (standard, short, squeeze)
- 📈 Volume with emoji indicator
- 🚦 Halt status
- 🎯 Urgency level

## Testing

To test the system:
1. Ensure `momentum_alerts.py` service is running
2. Open `public_html/index.html` in a browser
3. Check browser console for "Starting momentum alert polling..." message
4. When momentum criteria are met and alerts are sent, popup will appear automatically

## File Locations

- API: `/home/wilsonb/dl/github.com/Z223I/alpaca/cgi-bin/api/momentum_alerts_api.py`
- Web Page: `/home/wilsonb/dl/github.com/Z223I/alpaca/public_html/index.html`
- Alert Data: `/home/wilsonb/dl/github.com/Z223I/alpaca/historical_data/{YYYY-MM-DD}/momentum_alerts_sent/bullish/`
- Service: `/home/wilsonb/dl/github.com/Z223I/alpaca/cgi-bin/molecules/alpaca_molecules/momentum_alerts.py`

## Debug Mode

The momentum_alerts.py script now includes a debug mode for testing the alert popup system.

### Configuration

Located in `momentum_alerts.py` at line 138:
```python
self.debug_mode = True  # Set to False to disable debug alerts
```

### Behavior

When `debug_mode = True`:
- Sends a random test alert every 60 seconds
- Uses a predefined list of test symbols: AAPL, TSLA, NVDA, AMD, GOOGL, MSFT, AMZN, META
- Generates realistic random data for all alert fields:
  - Random prices ($50-$500)
  - Random momentum values
  - Random volume data
  - Random emoji indicators
  - Random source flags (gainers, volume surge, oracle)

### Purpose

- Test the web popup functionality without waiting for real momentum alerts
- Verify the coordination between backend service and frontend display
- Debug the alert queue and display system
- Test the "View Chart" button functionality

### Usage

1. **Enable Debug Mode**: Set `self.debug_mode = True` in momentum_alerts.py (currently enabled)
2. **Restart Service**: 
   ```bash
   pkill -f momentum_alerts.py
   nohup ~/miniconda3/envs/alpaca/bin/python cgi-bin/molecules/alpaca_molecules/momentum_alerts.py &
   ```
3. **Open Web Interface**: Navigate to `public_html/index.html`
4. **Watch for Alerts**: A new random alert will appear every minute

### Disabling Debug Mode

To return to production mode:
1. Set `self.debug_mode = False` in momentum_alerts.py (line 138)
2. Restart the service

**Note**: Debug alerts are sent to Telegram (if configured) and saved to the same directories as real alerts, making them indistinguishable from production alerts in the web interface.
