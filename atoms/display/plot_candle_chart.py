"""
Atom for plotting candlestick charts with volume for stock market data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from typing import Optional, Tuple, List
from datetime import time, datetime
import pytz

from ..indicators.orb_calculator import ORBCalculator
from ..utils.extract_symbol_data import extract_symbol_data
from ..utils.calculate_ema import calculate_ema
from ..utils.calculate_vwap import calculate_vwap_typical
from ..utils.calculate_macd import calculate_macd


def plot_candle_chart(df: pd.DataFrame, symbol: str, output_dir: str = 'plots', alerts: Optional[List] = None, major_resistance: Optional[List[float]] = None) -> bool:
    """
    Create a candlestick chart with volume and ORB rectangle for a single stock symbol.

    Args:
        df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        symbol: Stock symbol name
        output_dir: Directory to save the chart
        alerts: Optional list of alert dictionaries with timestamp_dt, alert_type, and alert_level
        major_resistance: Optional list of major resistance price levels from volume profile

    Returns:
        True if successful, False otherwise
    """

    def check_data_completeness(symbol_data, et_tz):
        """Check if we have complete trading session data, especially opening range."""

        if len(symbol_data) == 0:
            return {'complete': False, 'missing_opening': True, 'message': 'No data available'}

        # Get the trading date
        first_timestamp = symbol_data['timestamp'].iloc[0]
        if hasattr(first_timestamp, 'date'):
            trading_date = first_timestamp.date()
        else:
            trading_date = first_timestamp.to_pydatetime().date()

        # Define expected opening range: 9:30-9:45 AM ET
        expected_open = et_tz.localize(datetime.combine(trading_date, time(9, 30)))
        opening_range_end = et_tz.localize(datetime.combine(trading_date, time(9, 45)))

        # Check if we have any data in the opening range
        opening_mask = (symbol_data['timestamp'] >= expected_open) & (symbol_data['timestamp'] <= opening_range_end)
        opening_data = symbol_data[opening_mask]

        # Check actual data start time
        actual_start = symbol_data['timestamp'].min()
        if actual_start.tzinfo is None:
            actual_start = et_tz.localize(actual_start)

        missing_opening = len(opening_data) == 0
        minutes_late = (actual_start - expected_open).total_seconds() / 60 if actual_start > expected_open else 0

        completeness = {
            'complete': not missing_opening and minutes_late < 5,  # Allow 5 minute grace period
            'missing_opening': missing_opening,
            'minutes_late': minutes_late,
            'actual_start': actual_start,
            'expected_start': expected_open,
            'opening_data_points': len(opening_data),
            'message': f'Data starts {minutes_late:.0f} minutes late' if minutes_late > 0 else 'Complete data'
        }

        return completeness
    try:
        # Extract symbol data using the dedicated atom
        symbol_data = extract_symbol_data(df, symbol)

        if symbol_data is None:
            print(f"No data found for symbol: {symbol}")
            return False

        # Convert timestamps to Eastern Time for display
        et_tz = pytz.timezone('America/New_York')
        symbol_data = symbol_data.copy()

        # Check data completeness before processing
        completeness = check_data_completeness(symbol_data, et_tz)

        # Print data completeness warnings
        if completeness['missing_opening']:
            print(f"⚠️  WARNING: Missing opening range data for {symbol}")
            print(f"    Data starts at {completeness['actual_start'].strftime('%H:%M')} instead of 9:30 AM")
            print(f"    ORB levels may be inaccurate without opening range data")
        elif completeness['minutes_late'] > 5:
            print(f"⚠️  WARNING: Data for {symbol} starts {completeness['minutes_late']:.0f} minutes late")

        # Convert timestamps to ET (handles both EDT and EST automatically)
        if pd.api.types.is_datetime64_any_dtype(symbol_data['timestamp']):
            # If timezone-aware, convert to ET; if timezone-naive, localize to UTC first then convert
            if symbol_data['timestamp'].dt.tz is None:
                # Assume UTC if no timezone info
                symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'], utc=True)
            symbol_data['timestamp'] = symbol_data['timestamp'].dt.tz_convert(et_tz)

        # Calculate ORB levels using the proper ORBCalculator atom
        orb_calculator = ORBCalculator()
        orb_result = orb_calculator.calculate_orb_levels(symbol, symbol_data)

        if orb_result is not None:
            orb_high = orb_result.orb_high
            orb_low = orb_result.orb_low
        else:
            orb_high = None
            orb_low = None

        # Calculate EMA (9-period) using typical price (HLC/3)
        ema_success, ema_values = calculate_ema(symbol_data, period=9)

        # Calculate EMA (21-period) using typical price (HLC/3)
        ema21_success, ema21_values = calculate_ema(symbol_data, period=21)

        # Calculate EMA (34-period) using typical price (HLC/3)
        ema34_success, ema34_values = calculate_ema(symbol_data, period=34)

        # Calculate EMA (50-period) using typical price (HLC/3)
        ema50_success, ema50_values = calculate_ema(symbol_data, period=50)

        # Calculate VWAP using typical price (HLC/3)
        vwap_success, vwap_values = calculate_vwap_typical(symbol_data)

        # Calculate MACD (FastLength=12, SlowLength=26, Source=close, SignalLength=9)
        macd_success, macd_values = calculate_macd(symbol_data, fast_length=12, slow_length=26, 
                                                  signal_length=9, source='close')

        # Create figure with subplots (price, MACD, and volume)
        # Height ratios: Price=3, MACD=1.2 (80% of 1.5), Volume=0.9 (90% of 1.0)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), 
                                           gridspec_kw={'height_ratios': [3, 1.2, 0.9]},
                                           sharex=True)

        # Create secondary Y axis for price/ORB high ratio
        ax1_secondary = ax1.twinx()

        # Plot candlesticks on top subplot
        for idx, row in symbol_data.iterrows():
            timestamp = row['timestamp']
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']

            # Determine color (green for up, red for down)
            color = 'green' if close_price >= open_price else 'red'

            # Draw the high-low line
            ax1.plot([timestamp, timestamp], [low_price, high_price], 
                    color='black', linewidth=1, zorder=1)

            # Draw the open-close rectangle
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)

            rect = Rectangle((float(mdates.date2num(timestamp)) - 0.0003, bottom), 
                           0.0006, height, 
                           facecolor=color, edgecolor='none', alpha=0.8,
                           zorder=2)
            ax1.add_patch(rect)

        # Add ORB rectangle if ORB levels were calculated
        if orb_high is not None and orb_low is not None:
            # Filter data to only show ORB rectangle between 9:30 AM and 9:45 AM
            # Get the trading date from the data
            if len(symbol_data) > 0:
                first_timestamp = symbol_data['timestamp'].iloc[0]
                trading_date = first_timestamp.date()

                # Define ORB time window: 9:30 AM to 9:45 AM ET
                orb_start = et_tz.localize(datetime.combine(trading_date, time(9, 30)))
                orb_end = et_tz.localize(datetime.combine(trading_date, time(9, 45)))

                # Filter symbol data to only include ORB period
                orb_mask = (symbol_data['timestamp'] >= orb_start) & (symbol_data['timestamp'] <= orb_end)
                orb_data = symbol_data[orb_mask]

                # Only draw ORB rectangle if we have data in the ORB period
                if len(orb_data) > 0:
                    # Get time range for ORB rectangle (9:30 AM to 9:45 AM or last available data in that period)
                    x_min = float(mdates.date2num(orb_data['timestamp'].min()))
                    x_max = float(mdates.date2num(orb_data['timestamp'].max()))

                    # Set rectangle width to cover the actual ORB period
                    orb_width = x_max - x_min + 0.0012  # Add small padding to cover the last candlestick

                    # Draw ORB rectangle (no fill with thin edge)
                    orb_rect = Rectangle((x_min - 0.0006, orb_low), orb_width, orb_high - orb_low,
                                       facecolor='none', edgecolor='black', alpha=1.0, linewidth=1.5)
                    ax1.add_patch(orb_rect)

                    # Add ORB level lines
                    ax1.axhline(y=orb_high, color='black', linestyle='--', linewidth=1, alpha=0.8, label=f'ORB High: ${orb_high:.2f}')
                    ax1.axhline(y=orb_low, color='black', linestyle='--', linewidth=1, alpha=0.8, label=f'ORB Low: ${orb_low:.2f}')
                else:
                    # No data in ORB period, but still show the ORB level lines for reference
                    ax1.axhline(y=orb_high, color='black', linestyle='--', linewidth=1, alpha=0.8, label=f'ORB High: ${orb_high:.2f}')
                    ax1.axhline(y=orb_low, color='black', linestyle='--', linewidth=1, alpha=0.8, label=f'ORB Low: ${orb_low:.2f}')

            isDebugging = False  # Set to True for debugging output
            if isDebugging:
                # Add legend for ORB levels
                ax1.legend(loc='upper left', fontsize=10)

                print(f"ORB levels for {symbol}: High=${orb_high:.2f}, Low=${orb_low:.2f}")

        # Add EMA line if calculation was successful
        if ema_success:
            ax1.plot(symbol_data['timestamp'], ema_values,
                     color='blue', linewidth=2, alpha=0.8, label='EMA(9)')

        # Add EMA21 line if calculation was successful
        if ema21_success:
            ax1.plot(symbol_data['timestamp'], ema21_values,
                     color='orange', linewidth=2, alpha=0.8, label='EMA(21)')

        # Add EMA34 line if calculation was successful
        if ema34_success:
            ax1.plot(symbol_data['timestamp'], ema34_values,
                    color='cyan', linewidth=2, alpha=0.8, label='EMA(34)')

        # Add EMA50 line if calculation was successful
        if ema50_success:
            ax1.plot(symbol_data['timestamp'], ema50_values,
                    color='brown', linewidth=2, alpha=0.8, label='EMA(50)')

        # Add VWAP line if calculation was successful
        if vwap_success:
            ax1.plot(symbol_data['timestamp'], vwap_values,
                    color='purple', linewidth=2, alpha=0.8, label='VWAP')

        # Add major resistance line if provided (highest level)
        if major_resistance and len(major_resistance) > 0:
            for resistance_level in major_resistance:
                ax1.axhline(y=resistance_level, color='lime', linestyle='-',
                           linewidth=2, alpha=0.7,
                           label=f'Major Resistance: ${resistance_level:.2f}')
            print(f"Plotted major resistance level: "
                  f"${major_resistance[0]:.2f}")

        # Setup secondary Y axis for price/ORB high ratio if ORB high is available
        if orb_high is not None and orb_high > 0:
            # Calculate price/ORB high ratio range for axis scaling
            price_orb_ratio = symbol_data['close'] / orb_high

            # Set secondary Y axis properties
            ax1_secondary.set_ylabel('Price / ORB High Ratio', fontsize=12, color='gray')
            ax1_secondary.tick_params(axis='y', labelcolor='gray')

            # Set the Y axis limits to match the ratio range
            ratio_min = price_orb_ratio.min()
            ratio_max = price_orb_ratio.max()
            padding = (ratio_max - ratio_min) * 0.05  # 5% padding
            ax1_secondary.set_ylim(ratio_min - padding, ratio_max + padding)

            # Add horizontal line at ratio = 1.0 (breakout level)
            ax1_secondary.axhline(y=1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

            # Format secondary Y axis to show ratio with 3 decimal places
            ax1_secondary.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))

        # Plot alerts as vertical bars if provided
        if alerts:
            chart_start_time = symbol_data['timestamp'].min()
            chart_end_time = symbol_data['timestamp'].max()

            # Convert chart times to timezone-aware for proper comparison
            # Market data should be timezone-aware, but if not, assume ET
            if chart_start_time.tzinfo is None:
                chart_start_time = et_tz.localize(chart_start_time)
                chart_end_time = et_tz.localize(chart_end_time)

            alert_count = 0
            print(f"DEBUG: Chart timeframe for {symbol}: {chart_start_time} to {chart_end_time}")

            for alert in alerts:
                alert_time = alert.get('timestamp_dt')
                alert_type = alert.get('alert_type', 'unknown')
                alert_level = alert.get('alert_level', 'unknown')

                if not alert_time:
                    continue

                # Ensure alert time is timezone-aware for proper comparison
                if alert_time.tzinfo is None:
                    alert_time = et_tz.localize(alert_time)

                # Debug: Show each alert time comparison
                within_range = chart_start_time <= alert_time <= chart_end_time
                print(f"DEBUG: Alert at {alert_time} ({'in' if within_range else 'out of'} range) - {alert_type} {alert_level}")

                # Show ALL alerts regardless of chart timeframe (removed market hours filtering)
                # Set color based on alert type first, then MACD score for superduper alerts

                # Check for momentum alerts first (highest priority)
                if alert_type == 'momentum_bullish' or alert_level == 'momentum':
                    color = 'blue'  # Blue color for momentum alerts (distinct from superduper alerts)
                elif alert_type == 'vwap_bounce':
                    color = 'cyan'  # Cyan color for VWAP bounce alerts (different from momentum)
                else:
                    # For superduper alerts, use MACD score if available, otherwise fall back to alert level
                    macd_score = alert.get('macd_score', None)
                    if macd_score and isinstance(macd_score, dict):
                        score_color = macd_score.get('color', '').upper()
                        if score_color == 'GREEN':
                            color = 'green'
                        elif score_color == 'YELLOW':
                            color = 'gold'  # Use gold for better visibility than pure yellow
                        elif score_color == 'RED':
                            color = 'red'
                        else:
                            color = 'gray'  # Unknown MACD score
                    elif macd_score and isinstance(macd_score, str):
                        # Handle legacy string format
                        if macd_score == 'GREEN':
                            color = 'green'
                        elif macd_score == 'YELLOW':
                            color = 'gold'
                        elif macd_score == 'RED':
                            color = 'red'
                        else:
                            color = 'gray'
                    elif alert_level == 'green':
                        color = 'green'
                    elif alert_level == 'yellow':
                        color = 'gold'  # Use gold for better visibility than pure yellow
                    else:
                        # Fallback to old logic if neither macd_score nor alert_level is available
                        color = 'green' if alert_type == 'bullish' else 'red'
                alpha = 0.3

                # For alerts outside the chart timeframe, draw them at the chart edges
                if alert_time < chart_start_time:
                    # Draw at left edge of chart
                    plot_time = chart_start_time
                    alpha = 0.6  # Make edge alerts more visible
                elif alert_time > chart_end_time:
                    # Draw at right edge of chart
                    plot_time = chart_end_time
                    alpha = 0.6  # Make edge alerts more visible
                else:
                    # Draw at actual time within chart
                    plot_time = alert_time

                # Draw vertical line spanning the price chart
                y_min, y_max = ax1.get_ylim()
                ax1.axvline(x=plot_time, color=color, alpha=alpha, linewidth=2, linestyle='-')

                # Add small label at top of chart
                label_y = y_max - (y_max - y_min) * 0.05  # 5% from top
                if alert_type == 'vwap_bounce':
                    alert_symbol = 'V'  # V for VWAP bounce
                elif alert_type == 'momentum_bullish':
                    alert_symbol = 'M'  # M for momentum alerts
                else:
                    alert_symbol = '↑' if alert_type == 'bullish' else '↓'

                # Add edge indicator for out-of-range alerts
                if not within_range:
                    edge_indicator = '◀' if alert_time < chart_start_time else '▶'
                    alert_symbol = edge_indicator + alert_symbol

                ax1.text(plot_time, label_y, alert_symbol, 
                        color=color, fontsize=12, fontweight='bold', 
                        ha='center', va='top')

                alert_count += 1

            if alert_count > 0:
                print(f"Plotted {alert_count} alerts for {symbol} (superduper and momentum alerts, including out-of-market-hours)")

        # Add legend if any indicators were calculated
        if (ema_success or ema21_success or ema34_success or ema50_success or
                vwap_success):
            # Determine legend position based on price trend
            # If closing price is higher than opening price, put legend at bottom right
            # Otherwise, put it at top right to avoid overlapping with rising prices
            first_close = symbol_data['close'].iloc[0]
            last_close = symbol_data['close'].iloc[-1]

            if last_close > first_close:
                legend_location = 'lower right'
            else:
                legend_location = 'upper right'

            ax1.legend(loc=legend_location, fontsize=10)

        # Format price chart
        title = symbol

        # Get date from first timestamp and format for US (MM/DD/YYYY)
        # Also determine if we're in EDT or EST for the label
        first_timestamp = symbol_data['timestamp'].iloc[0]
        chart_date = first_timestamp.strftime('%m/%d/%Y')
        timezone_name = first_timestamp.strftime('%Z')  # Will be EDT or EST

        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.text(0.5, 0.95, f'{chart_date} ({timezone_name})', transform=ax1.transAxes, fontsize=12, 
                ha='center', va='top', style='italic')
        ax1.text(0.5, 0.90, 'Sent Superduper Alerts', transform=ax1.transAxes, fontsize=11, 
                ha='center', va='top', style='italic', color='darkblue')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Set x-axis formatter to show time in ET timezone with more frequent labels
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=et_tz))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Show every hour
        ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # Minor ticks every 30 minutes

        # Set explicit x-axis limits to show full trading session (9:30 AM to 4:00 PM ET)
        if len(symbol_data) > 0:
            # Get the date from the first timestamp
            first_timestamp = symbol_data['timestamp'].iloc[0]
            trading_date = first_timestamp.date()

            # Create full trading session boundaries in ET timezone
            session_start = et_tz.localize(datetime.combine(trading_date, time(9, 30)))
            session_end = et_tz.localize(datetime.combine(trading_date, time(16, 0)))

            ax1.set_xlim(session_start, session_end)

        # Format Y-axis to show prices with 2 decimal places
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.2f}'))

        # Plot MACD on middle subplot
        if macd_success:
            # Plot MACD line
            ax2.plot(symbol_data['timestamp'], macd_values['macd'], 
                    color='blue', linewidth=1.5, alpha=0.8, label='MACD')
            
            # Plot Signal line
            ax2.plot(symbol_data['timestamp'], macd_values['signal'], 
                    color='red', linewidth=1.5, alpha=0.8, label='Signal')
            
            # Plot Histogram as bars
            colors = ['green' if h >= 0 else 'red' for h in macd_values['histogram']]
            ax2.bar(symbol_data['timestamp'], macd_values['histogram'], 
                   color=colors, alpha=0.6, width=0.0008, label='Histogram')
            
            ax2.set_ylabel('MACD', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left', fontsize=10)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.7)

        # Set x-axis limits for MACD chart to match price chart
        if len(symbol_data) > 0:
            # Get the date from the first timestamp
            first_timestamp = symbol_data['timestamp'].iloc[0]
            trading_date = first_timestamp.date()

            # Create full trading session boundaries in ET timezone
            session_start = et_tz.localize(datetime.combine(trading_date, time(9, 30)))
            session_end = et_tz.localize(datetime.combine(trading_date, time(16, 0)))

            ax2.set_xlim(session_start, session_end)

        # Plot volume on bottom subplot
        ax3.bar(symbol_data['timestamp'], symbol_data['volume'], 
               color='blue', alpha=0.6, width=0.0008)
        ax3.set_ylabel('Volume', fontsize=12)
        ax3.set_xlabel(f'Time ({timezone_name})', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # Set x-axis formatter for MACD chart
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=et_tz))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Show every hour
        ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # Minor ticks every 30 minutes

        # Set x-axis formatter to show time in ET timezone for volume chart too
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=et_tz))
        ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Show every hour
        ax3.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # Minor ticks every 30 minutes

        # Set same x-axis limits for volume chart to match price chart
        if len(symbol_data) > 0:
            # Get the date from the first timestamp
            first_timestamp = symbol_data['timestamp'].iloc[0]
            trading_date = first_timestamp.date()

            # Create full trading session boundaries in ET timezone
            session_start = et_tz.localize(datetime.combine(trading_date, time(9, 30)))
            session_end = et_tz.localize(datetime.combine(trading_date, time(16, 0)))

            ax3.set_xlim(session_start, session_end)

        # Rotate x-axis labels
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Extract date from chart data and create date-specific subdirectory
        chart_date_obj = symbol_data['timestamp'].iloc[0].date()
        date_subdir = chart_date_obj.strftime('%Y%m%d')
        date_specific_output_dir = os.path.join(output_dir, date_subdir)

        # Create output directory if it doesn't exist
        os.makedirs(date_specific_output_dir, exist_ok=True)

        # Save the chart
        filename = f"{symbol}_chart.png"
        filepath = os.path.join(date_specific_output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        # Generate alert log file if alerts were provided
        if alerts:
            _generate_alerts_log(alerts, date_specific_output_dir, symbol, chart_date_obj)

        # Close the plot to free memory
        plt.close(fig)

        print(f"Chart saved: {filepath}")
        return True

    except Exception as e:
        print(f"Error creating chart for {symbol}: {e}")
        return False


def _generate_alerts_log(alerts: List, output_dir: str, symbol: str, chart_date) -> bool:
    """
    Generate a log file with detailed information about sent superduper alerts.
    
    Args:
        alerts: List of alert dictionaries
        output_dir: Directory to save the log file
        symbol: Stock symbol for this chart
        chart_date: Date object for the chart
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create log filename
        log_filename = "sent_superduper_alerts.log"
        log_filepath = os.path.join(output_dir, log_filename)
        
        # Filter alerts for this symbol only
        symbol_alerts = [alert for alert in alerts if alert.get('symbol', '').upper() == symbol.upper()]
        
        if not symbol_alerts:
            return True  # No alerts for this symbol, nothing to log
            
        # Check if log file exists to determine if we need to write header
        file_exists = os.path.exists(log_filepath)
        
        # Open log file in append mode
        with open(log_filepath, 'a') as log_file:
            # Write header if this is a new file
            if not file_exists:
                log_file.write(f"# Sent Superduper Alerts Log - {chart_date.strftime('%Y-%m-%d')}\n")
                log_file.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write("# Format: TIMESTAMP | SYMBOL | TYPE | LEVEL | MACD_SCORE | PRICE | DETAILS\n")
                log_file.write("# " + "="*80 + "\n\n")
            
            # Write section header for this symbol
            log_file.write(f"## {symbol} Alerts ({len(symbol_alerts)} total)\n")
            log_file.write("-" * 40 + "\n")
            
            # Sort alerts by timestamp
            sorted_alerts = sorted(symbol_alerts, 
                                   key=lambda x: x.get('timestamp_dt', datetime.min.replace(tzinfo=pytz.UTC)))
            
            # Write each alert
            for alert in sorted_alerts:
                timestamp = alert.get('timestamp_dt')
                alert_type = alert.get('alert_type', 'unknown')
                alert_level = alert.get('alert_level', 'unknown')
                
                # Format timestamp
                if timestamp:
                    timestamp_str = timestamp.strftime('%H:%M:%S %Z')
                else:
                    timestamp_str = 'unknown'
                
                # Get MACD score information
                macd_score = alert.get('macd_score', {})
                if isinstance(macd_score, dict):
                    macd_color = macd_score.get('color', 'unknown')
                    macd_details = f"{macd_color}"
                elif isinstance(macd_score, str):
                    macd_details = macd_score
                else:
                    macd_details = 'unknown'
                
                # Get price information if available
                price_info = ""
                if 'current_price' in alert:
                    price_info = f"${alert['current_price']:.2f}"
                elif 'price' in alert:
                    price_info = f"${alert['price']:.2f}"
                
                # Get ORB levels if available
                orb_info = ""
                if 'orb_high' in alert and 'orb_low' in alert:
                    orb_info = f" | ORB: ${alert['orb_high']:.2f}-${alert['orb_low']:.2f}"
                
                # Write alert line
                log_line = f"{timestamp_str} | {symbol} | {alert_type} | {alert_level} | {macd_details} | {price_info}{orb_info}\n"
                log_file.write(log_line)
                
                # Add additional details if available
                if 'alert_message' in alert:
                    log_file.write(f"    Message: {alert['alert_message']}\n")
                if 'volume' in alert:
                    log_file.write(f"    Volume: {alert['volume']:,}\n")
                if 'reason' in alert:
                    log_file.write(f"    Reason: {alert['reason']}\n")
                
                log_file.write("\n")
            
            log_file.write("\n")
        
        print(f"Alert log updated: {log_filepath} ({len(symbol_alerts)} alerts for {symbol})")
        return True
        
    except Exception as e:
        print(f"Error generating alerts log for {symbol}: {e}")
        return False
