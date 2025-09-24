#!/usr/bin/env python3
"""
Volume Profile Implementation - Python conversion of ThinkOrSwim volumeprofile
script

This module implements volume profile analysis similar to Charles Schwab's
ThinkOrSwim volumeprofile script. It calculates volume distribution at
different price levels and provides Point of Control (POC) and Value Area
analysis.

Key Features:
- Multiple time period aggregations (CHART, MINUTE, HOUR, DAY, WEEK, MONTH,
  BAR)
- Point of Control calculation (price with highest volume)
- Value Area calculation (default 70% of volume)
- Profile high/low tracking
- Configurable price granularity
"""

"""
Execution:
python code/volume_profile.py --symbol BBAI --days 1 --timeframe 5Min --time-per-profile DAY --chart
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import pytz
import alpaca_trade_api as tradeapi
import json

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from atoms.api.init_alpaca_client import init_alpaca_client  # noqa: E402


class PricePerRowHeightMode(Enum):
    """Price per row height calculation modes"""
    AUTOMATIC = "AUTOMATIC"
    TICKSIZE = "TICKSIZE"
    CUSTOM = "CUSTOM"


class TimePerProfile(Enum):
    """Time aggregation periods for volume profiles"""
    CHART = "CHART"
    MINUTE = "MINUTE"
    HOUR = "HOUR"
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    OPT_EXP = "OPT_EXP"
    BAR = "BAR"


class VolumeProfile:
    """
    Volume Profile calculator implementing ThinkOrSwim volumeprofile logic
    """

    def __init__(self,
                 price_per_row_height_mode: PricePerRowHeightMode = (
                     PricePerRowHeightMode.AUTOMATIC),
                 custom_row_height: float = 1.0,
                 time_per_profile: TimePerProfile = TimePerProfile.CHART,
                 multiplier: int = 1,
                 on_expansion: bool = True,
                 profiles: int = 1000,
                 show_point_of_control: bool = True,
                 show_value_area: bool = True,
                 value_area_percent: float = 70.0,
                 opacity: int = 50):
        """
        Initialize Volume Profile calculator with ToS-compatible parameters

        Args:
            price_per_row_height_mode: How to calculate price levels
                (AUTOMATIC, TICKSIZE, CUSTOM)
            custom_row_height: Custom price increment when using CUSTOM mode
            time_per_profile: Time aggregation period
            multiplier: Profile period multiplier
            on_expansion: Whether to show profiles on expansion bars
            profiles: Maximum number of profiles to maintain
            show_point_of_control: Whether to display POC
            show_value_area: Whether to display value area
            value_area_percent: Percentage of volume for value area
                calculation
            opacity: Display opacity (0-100)
        """
        self.price_per_row_height_mode = price_per_row_height_mode
        self.custom_row_height = custom_row_height
        self.time_per_profile = time_per_profile
        self.multiplier = multiplier
        self.on_expansion = on_expansion
        self.profiles = profiles
        self.show_point_of_control = show_point_of_control
        self.show_value_area = show_value_area
        self.value_area_percent = value_area_percent
        self.opacity = opacity

        # Internal state
        self.data = None
        self.volume_profiles = []
        self.poc_values = []
        self.value_area_high = []
        self.value_area_low = []
        self.profile_high = []
        self.profile_low = []

    def load_data_from_alpaca(self, symbol: str,
                              timeframe: tradeapi.TimeFrame,
                              start_date: datetime, end_date: datetime,
                              api_key: Optional[str] = None,
                              secret_key: Optional[str] = None,
                              base_url: Optional[str] = None):
        """
        Load stock data from Alpaca API

        Args:
            symbol: Stock symbol
            timeframe: Data timeframe (tradeapi.TimeFrame)
            start_date: Start date
            end_date: End date
            api_key: Alpaca API key (optional, will use init_alpaca_client
                if not provided)
            secret_key: Alpaca secret key (optional)
            base_url: Alpaca base URL (optional)
        """
        # Use init_alpaca_client if credentials not provided
        if api_key and secret_key:
            api = tradeapi.REST(api_key, secret_key, base_url)
        else:
            api = init_alpaca_client()

        # Format dates for API call
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        # Get bars from Alpaca API
        bars = api.get_bars(
            symbol,
            timeframe,
            start=start_str,
            end=end_str,
            limit=10000
        )

        # Convert to DataFrame format
        data_list = []
        for bar in bars:
            data_list.append({
                'timestamp': bar.t,
                'open': float(bar.o),
                'high': float(bar.h),
                'low': float(bar.l),
                'close': float(bar.c),
                'volume': int(bar.v)
            })

        self.data = pd.DataFrame(data_list)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)

    def load_data_from_dataframe(self, df: pd.DataFrame):
        """
        Load data from pandas DataFrame

        Args:
            df: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.data = df.copy()
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)

    def _calculate_period_groups(self) -> pd.Series:
        """
        Calculate period groupings based on time_per_profile setting

        Returns:
            Series with period group assignments for each row
        """
        if self.data is None or len(self.data) == 0:
            return pd.Series([], dtype=int)

        timestamps = self.data['timestamp']
        et_tz = pytz.timezone('America/New_York')

        # Ensure timestamps are timezone-aware
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(et_tz)
        else:
            timestamps = timestamps.dt.tz_convert(et_tz)

        if self.time_per_profile == TimePerProfile.CHART:
            # Single profile for entire chart
            return pd.Series([0] * len(self.data), index=self.data.index)

        elif self.time_per_profile == TimePerProfile.MINUTE:
            # Group by minute
            return ((timestamps - timestamps.iloc[0]).dt.total_seconds() // 60).astype(int)

        elif self.time_per_profile == TimePerProfile.HOUR:
            # Group by hour
            return ((timestamps - timestamps.iloc[0]).dt.total_seconds() // 3600).astype(int)

        elif self.time_per_profile == TimePerProfile.DAY:
            # Group by trading day
            dates = timestamps.dt.date
            unique_dates = sorted(dates.unique())
            date_to_period = {date: i for i, date in enumerate(unique_dates)}
            return dates.map(date_to_period)

        elif self.time_per_profile == TimePerProfile.WEEK:
            # Group by week
            return ((timestamps - timestamps.iloc[0]).dt.days // 7).astype(int)

        elif self.time_per_profile == TimePerProfile.MONTH:
            # Group by month
            return (timestamps.dt.year - timestamps.iloc[0].year) * 12 + (timestamps.dt.month - timestamps.iloc[0].month)

        elif self.time_per_profile == TimePerProfile.BAR:
            # Group by bar number
            return pd.Series(range(len(self.data)), index=self.data.index)

        else:
            # Default to CHART mode
            return pd.Series([0] * len(self.data), index=self.data.index)

    def _calculate_price_levels(self, group_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate price levels for volume aggregation

        Args:
            group_data: DataFrame for a single time period group

        Returns:
            Array of price levels
        """
        min_price = group_data['low'].min()
        max_price = group_data['high'].max()

        if self.price_per_row_height_mode == PricePerRowHeightMode.CUSTOM:
            price_increment = self.custom_row_height
        elif self.price_per_row_height_mode == PricePerRowHeightMode.TICKSIZE:
            # Simplified tick size calculation - could be enhanced with actual instrument tick sizes
            price_increment = 0.01 if max_price < 100 else 0.05
        else:  # AUTOMATIC
            # Automatic calculation based on price range
            price_range = max_price - min_price
            num_levels = min(100, max(10, int(price_range * 100)))  # Aim for reasonable number of levels
            price_increment = price_range / num_levels

        return np.arange(min_price, max_price + price_increment, price_increment)

    def _calculate_volume_at_price(self, group_data: pd.DataFrame, price_levels: np.ndarray) -> Dict[float, float]:
        """
        Calculate volume distribution at each price level for a time period

        Args:
            group_data: DataFrame for a single time period group
            price_levels: Array of price levels

        Returns:
            Dictionary mapping price levels to volume
        """
        volume_at_price = {price: 0.0 for price in price_levels}

        for _, row in group_data.iterrows():
            bar_high = row['high']
            bar_low = row['low']
            bar_volume = row['volume']

            # Find price levels within this bar's range
            relevant_levels = price_levels[(price_levels >= bar_low) & (price_levels <= bar_high)]

            if len(relevant_levels) > 0:
                # Distribute volume evenly across price levels within the bar
                volume_per_level = bar_volume / len(relevant_levels)
                for price in relevant_levels:
                    volume_at_price[price] += volume_per_level

        return volume_at_price

    def _calculate_point_of_control(self, volume_at_price: Dict[float, float]) -> float:
        """
        Calculate Point of Control (price level with highest volume)

        Args:
            volume_at_price: Dictionary mapping price levels to volume

        Returns:
            Price level with highest volume
        """
        if not volume_at_price:
            return 0.0

        return max(volume_at_price.keys(), key=lambda price: volume_at_price[price])

    def _calculate_value_area(self, volume_at_price: Dict[float, float]) -> Tuple[float, float]:
        """
        Calculate Value Area (price range containing specified percentage of volume)

        Args:
            volume_at_price: Dictionary mapping price levels to volume

        Returns:
            Tuple of (value_area_low, value_area_high)
        """
        if not volume_at_price:
            return 0.0, 0.0

        # Sort by volume descending
        sorted_by_volume = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)

        total_volume = sum(volume_at_price.values())
        target_volume = total_volume * (self.value_area_percent / 100.0)

        accumulated_volume = 0.0
        value_area_prices = []

        for price, volume in sorted_by_volume:
            accumulated_volume += volume
            value_area_prices.append(price)

            if accumulated_volume >= target_volume:
                break

        if not value_area_prices:
            return 0.0, 0.0

        return min(value_area_prices), max(value_area_prices)

    def calculate_volume_profiles(self):
        """
        Calculate volume profiles for all time periods
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data_from_alpaca() or load_data_from_dataframe() first.")

        # Calculate period groups
        period_groups = self._calculate_period_groups()

        # Reset internal state
        self.volume_profiles = []
        self.poc_values = []
        self.value_area_high = []
        self.value_area_low = []
        self.profile_high = []
        self.profile_low = []

        # Process each period group
        for period in sorted(period_groups.unique()):
            period_mask = period_groups == period
            group_data = self.data[period_mask].copy()

            if len(group_data) == 0:
                continue

            # Calculate price levels for this period
            price_levels = self._calculate_price_levels(group_data)

            # Calculate volume at each price level
            volume_at_price = self._calculate_volume_at_price(group_data, price_levels)

            # Calculate key metrics
            poc = self._calculate_point_of_control(volume_at_price)
            va_low, va_high = self._calculate_value_area(volume_at_price)
            prof_high = group_data['high'].max()
            prof_low = group_data['low'].min()

            # Store results
            period_result = {
                'period': period,
                'start_time': group_data['timestamp'].min(),
                'end_time': group_data['timestamp'].max(),
                'volume_at_price': volume_at_price,
                'point_of_control': poc,
                'value_area_high': va_high,
                'value_area_low': va_low,
                'profile_high': prof_high,
                'profile_low': prof_low,
                'total_volume': sum(volume_at_price.values())
            }

            self.volume_profiles.append(period_result)
            self.poc_values.append(poc)
            self.value_area_high.append(va_high)
            self.value_area_low.append(va_low)
            self.profile_high.append(prof_high)
            self.profile_low.append(prof_low)

    def get_latest_profile(self) -> Optional[Dict]:
        """
        Get the most recent volume profile

        Returns:
            Latest volume profile dictionary or None
        """
        if not self.volume_profiles:
            return None
        return self.volume_profiles[-1]

    def get_profile_summary(self) -> Dict:
        """
        Get summary statistics for all profiles

        Returns:
            Dictionary with summary statistics
        """
        if not self.volume_profiles:
            return {}

        return {
            'total_profiles': len(self.volume_profiles),
            'time_range': f"{self.volume_profiles[0]['start_time']} to {self.volume_profiles[-1]['end_time']}",
            'avg_poc': np.mean(self.poc_values) if self.poc_values else 0,
            'poc_range': f"{min(self.poc_values):.2f} - {max(self.poc_values):.2f}" if self.poc_values else "N/A",
            'avg_value_area_width': np.mean([va_h - va_l for va_h, va_l in zip(self.value_area_high, self.value_area_low)]) if self.value_area_high else 0
        }

    def print_profile_summary(self):
        """Print volume profile summary to console"""
        if not self.volume_profiles:
            print("No volume profiles calculated.")
            return

        summary = self.get_profile_summary()
        latest = self.get_latest_profile()

        print("=" * 60)
        print("VOLUME PROFILE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Time Per Profile: {self.time_per_profile.value}")
        print(f"  Price Mode: {self.price_per_row_height_mode.value}")
        print(f"  Value Area %: {self.value_area_percent}%")
        print()
        print(f"Results:")
        print(f"  Total Profiles: {summary['total_profiles']}")
        print(f"  Time Range: {summary['time_range']}")
        print(f"  Average POC: ${summary['avg_poc']:.2f}")
        print(f"  POC Range: ${summary['poc_range']}")
        print(f"  Avg Value Area Width: ${summary['avg_value_area_width']:.2f}")

        if latest:
            print()
            print("Latest Profile:")
            print(f"  📅 Period: {latest['start_time']} to {latest['end_time']}")
            print(f"  🎯 Point of Control: ${latest['point_of_control']:.2f}")
            print(f"  📊 Value Area: ${latest['value_area_low']:.2f} - ${latest['value_area_high']:.2f}")
            print(f"  📈 Profile Range: ${latest['profile_low']:.2f} - ${latest['profile_high']:.2f}")
            print(f"  📦 Total Volume: {latest['total_volume']:,.0f}")
        print("=" * 60)

    def generate_chart(self, symbol: str, output_dir: str = 'volume_profile_output') -> bool:
        """
        Generate volume profile chart with price and volume data

        Args:
            symbol: Stock symbol for chart title
            output_dir: Directory to save chart

        Returns:
            True if successful, False otherwise
        """
        if not self.volume_profiles or self.data is None:
            print("❌ No volume profiles or data available for charting")
            return False

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Create the chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                         gridspec_kw={'height_ratios': [3, 1]})

            # Prepare data for candlestick chart - ensure ET timezone
            timestamps = pd.to_datetime(self.data['timestamp'])

            # Alpaca data is already in Eastern Time, just need to localize it properly
            et_tz = pytz.timezone('America/New_York')
            if timestamps.dt.tz is None:
                # Data is already in ET, just add the timezone info
                timestamps = timestamps.dt.tz_localize(et_tz)
            else:
                # Convert to ET if it has a different timezone
                timestamps = timestamps.dt.tz_convert(et_tz)

            opens = self.data['open']
            highs = self.data['high']
            lows = self.data['low']
            closes = self.data['close']
            volumes = self.data['volume']

            # Plot candlesticks on main axis
            for i in range(len(self.data)):
                color = 'green' if closes.iloc[i] >= opens.iloc[i] else 'red'

                # Draw the wick (high-low line)
                ax1.plot([timestamps.iloc[i], timestamps.iloc[i]],
                        [lows.iloc[i], highs.iloc[i]],
                        color='black', linewidth=0.5)

                # Draw the body (open-close rectangle)
                body_height = abs(closes.iloc[i] - opens.iloc[i])
                body_bottom = min(opens.iloc[i], closes.iloc[i])

                rect = Rectangle((mdates.date2num(timestamps.iloc[i]) - 0.0003, body_bottom),
                               0.0006, body_height,
                               facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax1.add_patch(rect)

            # Plot volume profile data
            latest_profile = self.get_latest_profile()
            if latest_profile and latest_profile['volume_at_price']:
                volume_data = latest_profile['volume_at_price']
                prices = list(volume_data.keys())
                volumes_at_price = list(volume_data.values())

                # Create volume profile bars on the right side of the chart
                max_volume = max(volumes_at_price) if volumes_at_price else 1
                price_range = max(prices) - min(prices) if len(prices) > 1 else 1
                bar_width_factor = price_range * 0.60  # Scale bars to 60% of price range (4x wider)

                # Get chart x-axis limits for positioning volume bars
                xlims = ax1.get_xlim()
                x_position = xlims[1] - (xlims[1] - xlims[0]) * 0.07  # 7% from right edge

                for price, volume in volume_data.items():
                    if volume > 0:
                        bar_width = (volume / max_volume) * bar_width_factor

                        # Color coding for volume bars
                        if price == latest_profile['point_of_control']:
                            color = 'blue'  # POC in blue
                            alpha = 0.8
                        elif (latest_profile['value_area_low'] <= price <=
                              latest_profile['value_area_high']):
                            color = 'orange'  # Value area in orange
                            alpha = 0.6
                        else:
                            color = 'gray'  # Other areas in gray
                            alpha = 0.4

                        # Draw horizontal volume bar
                        bar_rect = Rectangle((x_position, price - 0.01),
                                           bar_width, 0.02,
                                           facecolor=color, alpha=alpha,
                                           edgecolor='black', linewidth=0.3)
                        ax1.add_patch(bar_rect)

                # Plot POC line
                poc_price = latest_profile['point_of_control']
                ax1.axhline(y=poc_price, color='blue', linestyle='-',
                           linewidth=2, alpha=0.8, label=f'POC: ${poc_price:.2f}')

                # Plot Value Area lines
                va_high = latest_profile['value_area_high']
                va_low = latest_profile['value_area_low']
                ax1.axhline(y=va_high, color='orange', linestyle='--',
                           linewidth=1.5, alpha=0.7,
                           label=f'VA High: ${va_high:.2f}')
                ax1.axhline(y=va_low, color='orange', linestyle='--',
                           linewidth=1.5, alpha=0.7,
                           label=f'VA Low: ${va_low:.2f}')

            # Configure main chart
            ax1.set_title(f'{symbol} - Volume Profile Analysis\n'
                         f'Time Period: {self.time_per_profile.value} | '
                         f'Value Area: {self.value_area_percent}%',
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')

            # Format x-axis for timestamps with dates in ET timezone
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M', tz=et_tz))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

            # Plot volume bars on bottom subplot
            bar_colors = ['green' if c >= o else 'red'
                         for c, o in zip(closes, opens)]
            ax2.bar(timestamps, volumes, color=bar_colors, alpha=0.7, width=0.0006)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Time (ET)', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Format volume axis
            ax2.yaxis.set_major_formatter(FuncFormatter(
                lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

            # Sync x-axes
            ax2.set_xlim(ax1.get_xlim())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M', tz=et_tz))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            # Adjust layout and save
            plt.tight_layout()

            # Generate filename with timestamp
            chart_date = timestamps.iloc[0].date()
            filename = f"{symbol}_volume_profile_{chart_date.strftime('%Y%m%d')}.png"
            filepath = os.path.join(output_dir, filename)

            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 Volume profile chart saved: {filepath}")

            # Save JSON data with same naming convention
            json_filename = f"{symbol}_volume_profile_{chart_date.strftime('%Y%m%d')}.json"
            json_filepath = os.path.join(output_dir, json_filename)

            if self._save_json_output(json_filepath, symbol):
                print(f"💾 Volume profile data saved: {json_filepath}")

            return True

        except Exception as e:
            print(f"❌ Error generating chart: {e}")
            return False

    def _save_json_output(self, filepath: str, symbol: str) -> bool:
        """
        Save volume profile analysis results to JSON file

        Args:
            filepath: Path to save JSON file
            symbol: Stock symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare JSON output data
            output_data = {
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "configuration": {
                    "time_per_profile": self.time_per_profile.value,
                    "price_per_row_height_mode": self.price_per_row_height_mode.value,
                    "value_area_percent": self.value_area_percent,
                    "custom_row_height": self.custom_row_height,
                    "multiplier": self.multiplier,
                    "on_expansion": self.on_expansion,
                    "profiles": self.profiles
                },
                "summary": self.get_profile_summary(),
                "profiles": []
            }

            # Add detailed profile data
            for profile in self.volume_profiles:
                profile_data = {
                    "period": profile["period"],
                    "start_time": profile["start_time"].isoformat(),
                    "end_time": profile["end_time"].isoformat(),
                    "point_of_control": profile["point_of_control"],
                    "value_area_high": profile["value_area_high"],
                    "value_area_low": profile["value_area_low"],
                    "profile_high": profile["profile_high"],
                    "profile_low": profile["profile_low"],
                    "total_volume": profile["total_volume"],
                    "volume_at_price": {str(price): volume for price, volume in profile["volume_at_price"].items()}
                }
                output_data["profiles"].append(profile_data)

            # Add raw OHLCV data
            if self.data is not None:
                ohlcv_data = []
                for _, row in self.data.iterrows():
                    ohlcv_data.append({
                        "timestamp": row["timestamp"].isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"])
                    })
                output_data["ohlcv_data"] = ohlcv_data

            # Write JSON file
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)

            return True

        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
            return False


def main():
    """Main entry point for volume profile analysis"""
    parser = argparse.ArgumentParser(description='Volume Profile Analysis - ToS Script Conversion')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to analyze')
    parser.add_argument('--days', type=int, default=5, help='Number of days of data to analyze')
    parser.add_argument('--timeframe', type=str, default='1Min', choices=['1Min', '5Min', '15Min', '1Hour', '1Day'],
                       help='Data timeframe')
    parser.add_argument('--time-per-profile', type=str, default='DAY',
                       choices=['CHART', 'MINUTE', 'HOUR', 'DAY', 'WEEK', 'MONTH', 'BAR'],
                       help='Time aggregation for profiles')
    parser.add_argument('--price-mode', type=str, default='AUTOMATIC',
                       choices=['AUTOMATIC', 'TICKSIZE', 'CUSTOM'],
                       help='Price level calculation mode')
    parser.add_argument('--custom-height', type=float, default=1.0,
                       help='Custom price increment (when using CUSTOM mode)')
    parser.add_argument('--value-area-percent', type=float, default=70.0,
                       help='Value area percentage (default: 70)')
    parser.add_argument('--output-dir', type=str, default='volume_profile_output',
                       help='Output directory for results')
    parser.add_argument('--chart', action='store_true',
                       help='Generate volume profile chart')

    args = parser.parse_args()

    # Calculate date range
    end_date = datetime.now(pytz.timezone('America/New_York'))
    start_date = end_date - timedelta(days=args.days)

    # Map timeframe for alpaca_trade_api
    timeframe_map = {
        '1Min': tradeapi.TimeFrame.Minute,
        '5Min': tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
        '15Min': tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
        '1Hour': tradeapi.TimeFrame.Hour,
        '1Day': tradeapi.TimeFrame.Day
    }
    timeframe = timeframe_map[args.timeframe]

    # Create volume profile analyzer
    vp = VolumeProfile(
        price_per_row_height_mode=PricePerRowHeightMode(args.price_mode),
        custom_row_height=args.custom_height,
        time_per_profile=TimePerProfile(args.time_per_profile),
        value_area_percent=args.value_area_percent
    )

    try:
        print(f"Loading {args.days} days of {args.timeframe} data for {args.symbol}...")
        vp.load_data_from_alpaca(
            symbol=args.symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        print("Calculating volume profiles...")
        vp.calculate_volume_profiles()

        print("Analysis complete!")
        vp.print_profile_summary()

        # Generate chart if requested
        if args.chart:
            print("\nGenerating volume profile chart...")
            chart_success = vp.generate_chart(args.symbol, args.output_dir)
            if not chart_success:
                print("⚠️  Chart generation failed, but analysis completed successfully")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
