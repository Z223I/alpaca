"""
ORB Alerts Summary Generator

This script combines all alerts for a specific date and saves a comprehensive summary
in the historical_data/YYYYMMDD directory. It allows selection from the 10 most recent
dates with the most recent as default.

Usage:
    python3 code/alerts_summary.py                    # Use most recent date
    python3 code/alerts_summary.py --date 2025-07-14  # Use specific date
    python3 code/alerts_summary.py --list-dates       # List available dates
    python3 code/alerts_summary.py --verbose          # Enable verbose output
"""

import argparse
import json
import os
import sys
import csv
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytz
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/z223i/alpaca')


class AlertsSummaryGenerator:
    """Generates comprehensive summaries of ORB alerts for specific dates."""
    
    def __init__(self, date: Optional[str] = None, verbose: bool = False):
        """
        Initialize the alerts summary generator.
        
        Args:
            date: Date in YYYY-MM-DD format (None for most recent)
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.historical_data_dir = Path("historical_data")
        
        # Get available dates
        self.available_dates = self._get_available_dates()
        
        if not self.available_dates:
            raise ValueError("No historical data directories found")
        
        # Set target date
        if date:
            if date not in self.available_dates:
                raise ValueError(f"Date {date} not found in available dates: {self.available_dates[:10]}")
            self.target_date = date
        else:
            self.target_date = self.available_dates[0]  # Most recent
        
        self.target_dir = self.historical_data_dir / self.target_date
        self.alerts_dir = self.target_dir / "alerts"
        
        if self.verbose:
            print(f"Target date: {self.target_date}")
            print(f"Target directory: {self.target_dir}")
    
    def _get_available_dates(self) -> List[str]:
        """Get list of available dates, sorted by most recent first."""
        if not self.historical_data_dir.exists():
            return []
        
        dates = []
        for item in self.historical_data_dir.iterdir():
            if item.is_dir() and item.name.startswith("2025-"):
                dates.append(item.name)
        
        # Sort by date (most recent first) and limit to 10
        dates.sort(reverse=True)
        return dates[:10]
    
    def _load_alert_files(self, alert_type: str) -> List[Dict]:
        """Load all alert files of a specific type."""
        alerts = []
        alert_type_dir = self.alerts_dir / alert_type
        
        if not alert_type_dir.exists():
            if self.verbose:
                print(f"No {alert_type} alerts directory found")
            return alerts
        
        # Load all JSON files in the directory
        for json_file in alert_type_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    alert_data = json.load(f)
                    alert_data['_source_file'] = json_file.name
                    alerts.append(alert_data)
            except Exception as e:
                if self.verbose:
                    print(f"Error loading {json_file}: {e}")
        
        # Sort by timestamp
        alerts.sort(key=lambda x: x.get('timestamp', ''))
        
        if self.verbose:
            print(f"Loaded {len(alerts)} {alert_type} alerts")
        
        return alerts
    
    def _calculate_symbol_statistics(self, alerts: List[Dict]) -> Dict:
        """Calculate statistics per symbol."""
        symbol_stats = defaultdict(lambda: {
            'total_alerts': 0,
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 1.0,
            'avg_breakout_percentage': 0.0,
            'max_breakout_percentage': 0.0,
            'avg_volume_ratio': 0.0,
            'max_volume_ratio': 0.0,
            'priority_counts': Counter(),
            'confidence_levels': Counter(),
            'price_range': {'min': float('inf'), 'max': 0.0},
            'first_alert_time': None,
            'last_alert_time': None,
            'alerts': []
        })
        
        for alert in alerts:
            symbol = alert.get('symbol', 'UNKNOWN')
            stats = symbol_stats[symbol]
            
            # Basic counts
            stats['total_alerts'] += 1
            stats['alerts'].append(alert)
            
            # Confidence statistics
            confidence = alert.get('confidence_score', 0.0)
            stats['avg_confidence'] += confidence
            stats['max_confidence'] = max(stats['max_confidence'], confidence)
            stats['min_confidence'] = min(stats['min_confidence'], confidence)
            
            # Breakout percentage
            breakout_pct = alert.get('breakout_percentage', 0.0)
            stats['avg_breakout_percentage'] += breakout_pct
            stats['max_breakout_percentage'] = max(stats['max_breakout_percentage'], breakout_pct)
            
            # Volume ratio
            volume_ratio = alert.get('volume_ratio', 0.0)
            stats['avg_volume_ratio'] += volume_ratio
            stats['max_volume_ratio'] = max(stats['max_volume_ratio'], volume_ratio)
            
            # Price range
            current_price = alert.get('current_price', 0.0)
            if current_price > 0:
                stats['price_range']['min'] = min(stats['price_range']['min'], current_price)
                stats['price_range']['max'] = max(stats['price_range']['max'], current_price)
            
            # Categorical data
            stats['priority_counts'][alert.get('priority', 'UNKNOWN')] += 1
            stats['confidence_levels'][alert.get('confidence_level', 'UNKNOWN')] += 1
            
            # Time tracking
            timestamp = alert.get('timestamp')
            if timestamp:
                if not stats['first_alert_time'] or timestamp < stats['first_alert_time']:
                    stats['first_alert_time'] = timestamp
                if not stats['last_alert_time'] or timestamp > stats['last_alert_time']:
                    stats['last_alert_time'] = timestamp
        
        # Calculate averages
        for symbol, stats in symbol_stats.items():
            if stats['total_alerts'] > 0:
                stats['avg_confidence'] /= stats['total_alerts']
                stats['avg_breakout_percentage'] /= stats['total_alerts']
                stats['avg_volume_ratio'] /= stats['total_alerts']
                
                if stats['price_range']['min'] == float('inf'):
                    stats['price_range']['min'] = 0.0
        
        return dict(symbol_stats)
    
    def _generate_time_analysis(self, alerts: List[Dict]) -> Dict:
        """Analyze alert timing patterns."""
        hourly_counts = Counter()
        minute_counts = Counter()
        
        for alert in alerts:
            timestamp = alert.get('timestamp')
            if timestamp:
                try:
                    # Parse timestamp
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    
                    # Convert to ET for analysis
                    et_tz = pytz.timezone('US/Eastern')
                    if dt.tzinfo is None:
                        dt = et_tz.localize(dt)
                    else:
                        dt = dt.astimezone(et_tz)
                    
                    hour = dt.hour
                    minute = dt.minute
                    
                    hourly_counts[hour] += 1
                    minute_counts[minute] += 1
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error parsing timestamp {timestamp}: {e}")
        
        return {
            'hourly_distribution': dict(hourly_counts),
            'minute_distribution': dict(minute_counts),
            'peak_hour': hourly_counts.most_common(1)[0] if hourly_counts else None,
            'peak_minute': minute_counts.most_common(1)[0] if minute_counts else None
        }
    
    def generate_summary(self) -> Dict:
        """Generate comprehensive alerts summary for the target date."""
        print(f"\n🔍 Generating alerts summary for {self.target_date}...")
        
        # Load all alert types
        bullish_alerts = self._load_alert_files("bullish")
        bearish_alerts = self._load_alert_files("bearish")
        
        # Check for super alerts (bullish and bearish)
        bullish_super_alerts = []
        bearish_super_alerts = []
        
        # Load bullish superduper alerts
        bullish_super_dir = self.target_dir / "super_alerts" / "bullish"
        if bullish_super_dir.exists():
            for json_file in bullish_super_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        super_alert = json.load(f)
                        super_alert['_source_file'] = json_file.name
                        super_alert['alert_direction'] = 'bullish'
                        bullish_super_alerts.append(super_alert)
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading bullish superduper alert {json_file}: {e}")
        
        # Load bearish superduper alerts
        bearish_super_dir = self.target_dir / "super_alerts" / "bearish"
        if bearish_super_dir.exists():
            for json_file in bearish_super_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        super_alert = json.load(f)
                        super_alert['_source_file'] = json_file.name
                        super_alert['alert_direction'] = 'bearish'
                        bearish_super_alerts.append(super_alert)
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading bearish superduper alert {json_file}: {e}")
        
        super_alerts = bullish_super_alerts + bearish_super_alerts
        
        # Generate statistics
        all_alerts = bullish_alerts + bearish_alerts
        
        summary = {
            'metadata': {
                'date': self.target_date,
                'generated_at': datetime.now().isoformat(),
                'total_alerts': len(all_alerts),
                'bullish_alerts': len(bullish_alerts),
                'bearish_alerts': len(bearish_alerts),
                'superduper_alerts': len(super_alerts),
                'unique_symbols': len(set(alert.get('symbol') for alert in all_alerts))
            },
            'bullish_analysis': {
                'symbol_statistics': self._calculate_symbol_statistics(bullish_alerts),
                'time_analysis': self._generate_time_analysis(bullish_alerts),
                'alerts': bullish_alerts
            },
            'bearish_analysis': {
                'symbol_statistics': self._calculate_symbol_statistics(bearish_alerts),
                'time_analysis': self._generate_time_analysis(bearish_alerts),
                'alerts': bearish_alerts
            },
            'super_alerts': {
                'total_count': len(super_alerts),
                'bullish_count': len(bullish_super_alerts),
                'bearish_count': len(bearish_super_alerts),
                'alerts': super_alerts,
                'bullish_alerts': bullish_super_alerts,
                'bearish_alerts': bearish_super_alerts
            },
            'overall_statistics': {
                'most_active_symbols': self._get_most_active_symbols(all_alerts),
                'highest_confidence_alerts': self._get_highest_confidence_alerts(all_alerts),
                'largest_breakouts': self._get_largest_breakouts(all_alerts),
                'time_distribution': self._generate_time_analysis(all_alerts)
            }
        }
        
        return summary
    
    def _get_most_active_symbols(self, alerts: List[Dict], limit: int = 10) -> List[Tuple[str, int]]:
        """Get symbols with most alerts."""
        symbol_counts = Counter(alert.get('symbol') for alert in alerts)
        return symbol_counts.most_common(limit)
    
    def _get_highest_confidence_alerts(self, alerts: List[Dict], limit: int = 10) -> List[Dict]:
        """Get alerts with highest confidence scores."""
        sorted_alerts = sorted(alerts, key=lambda x: x.get('confidence_score', 0), reverse=True)
        return sorted_alerts[:limit]
    
    def _get_largest_breakouts(self, alerts: List[Dict], limit: int = 10) -> List[Dict]:
        """Get alerts with largest breakout percentages."""
        sorted_alerts = sorted(alerts, key=lambda x: x.get('breakout_percentage', 0), reverse=True)
        return sorted_alerts[:limit]
    
    def save_summary(self, summary: Dict) -> Tuple[str, str, List[str]]:
        """Save summary to JSON, CSV files and generate pie charts in alerts directory."""
        # Save one level down in alerts directory
        summary_dir = self.alerts_dir / "summary"
        summary_dir.mkdir(exist_ok=True)
        
        # Create filenames
        date_str = self.target_date.replace('-', '')
        json_filename = f"alerts_summary_{date_str}.json"
        csv_filename = f"alerts_summary_{date_str}.csv"
        
        json_filepath = summary_dir / json_filename
        csv_filepath = summary_dir / csv_filename
        
        # Save to JSON with pretty formatting
        with open(json_filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save to CSV format
        self._save_summary_csv(summary, csv_filepath)
        
        # Generate regular alerts charts
        regular_chart_files = self._generate_regular_alerts_charts(summary, summary_dir, date_str)
        
        # Generate superduper alerts charts (bullish and bearish pie charts + bar charts)
        super_chart_files = self._generate_superduper_alerts_charts(summary, summary_dir, date_str)
        
        # Note: High-impact charts are not generated since all superduper alerts are already high impact
        high_impact_chart_files = []
        
        # Generate superduper alerts ratio bar charts (current_price / orb_high binned by 10%)
        ratio_bar_chart_files = self._generate_superduper_alerts_ratio_bar_charts(summary, summary_dir, date_str)
        
        # Combine all chart files
        chart_files = regular_chart_files + super_chart_files + high_impact_chart_files + ratio_bar_chart_files
        
        print(f"💾 JSON summary saved to: {json_filepath}")
        print(f"📊 CSV summary saved to: {csv_filepath}")
        for chart_file in chart_files:
            print(f"📈 Chart saved to: {chart_file}")
        
        return str(json_filepath), str(csv_filepath), chart_files
    
    def _save_summary_csv(self, summary: Dict, csv_filepath: Path) -> None:
        """Save summary data to CSV format."""
        # Combine all alerts for CSV export
        all_alerts = []
        
        # Add bullish alerts
        for alert in summary['bullish_analysis']['alerts']:
            alert_row = self._alert_to_csv_row(alert, 'bullish')
            all_alerts.append(alert_row)
        
        # Add bearish alerts
        for alert in summary['bearish_analysis']['alerts']:
            alert_row = self._alert_to_csv_row(alert, 'bearish')
            all_alerts.append(alert_row)
        
        # Add super alerts
        for alert in summary['super_alerts']['alerts']:
            alert_row = self._super_alert_to_csv_row(alert)
            all_alerts.append(alert_row)
        
        # Sort by timestamp
        all_alerts.sort(key=lambda x: x.get('timestamp', ''))
        
        if not all_alerts:
            # Create empty CSV with headers
            with open(csv_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self._get_csv_headers())
            return
        
        # Write to CSV
        with open(csv_filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_csv_headers())
            writer.writeheader()
            writer.writerows(all_alerts)
    
    def _get_csv_headers(self) -> List[str]:
        """Get CSV column headers."""
        return [
            'timestamp', 'symbol', 'alert_type', 'current_price', 'orb_high', 'orb_low',
            'orb_range', 'orb_midpoint', 'breakout_type', 'breakout_percentage',
            'volume_ratio', 'confidence_score', 'priority', 'confidence_level',
            'recommended_stop_loss', 'recommended_take_profit', 'source_file',
            # Super alert specific fields
            'signal_price', 'resistance_price', 'penetration_percent', 'range_percent'
        ]
    
    def _alert_to_csv_row(self, alert: Dict, alert_type: str) -> Dict:
        """Convert alert to CSV row format."""
        return {
            'timestamp': alert.get('timestamp', ''),
            'symbol': alert.get('symbol', ''),
            'alert_type': alert_type,
            'current_price': alert.get('current_price', ''),
            'orb_high': alert.get('orb_high', ''),
            'orb_low': alert.get('orb_low', ''),
            'orb_range': alert.get('orb_range', ''),
            'orb_midpoint': alert.get('orb_midpoint', ''),
            'breakout_type': alert.get('breakout_type', ''),
            'breakout_percentage': alert.get('breakout_percentage', ''),
            'volume_ratio': alert.get('volume_ratio', ''),
            'confidence_score': alert.get('confidence_score', ''),
            'priority': alert.get('priority', ''),
            'confidence_level': alert.get('confidence_level', ''),
            'recommended_stop_loss': alert.get('recommended_stop_loss', ''),
            'recommended_take_profit': alert.get('recommended_take_profit', ''),
            'source_file': alert.get('_source_file', ''),
            'signal_price': '',
            'resistance_price': '',
            'penetration_percent': '',
            'range_percent': ''
        }
    
    def _super_alert_to_csv_row(self, super_alert: Dict) -> Dict:
        """Convert super alert to CSV row format."""
        signal_analysis = super_alert.get('signal_analysis', {})
        
        return {
            'timestamp': super_alert.get('timestamp', ''),
            'symbol': super_alert.get('symbol', ''),
            'alert_type': 'super',
            'current_price': signal_analysis.get('current_price', ''),
            'orb_high': '',
            'orb_low': '',
            'orb_range': '',
            'orb_midpoint': '',
            'breakout_type': '',
            'breakout_percentage': '',
            'volume_ratio': '',
            'confidence_score': '',
            'priority': '',
            'confidence_level': '',
            'recommended_stop_loss': '',
            'recommended_take_profit': '',
            'source_file': super_alert.get('_source_file', ''),
            'signal_price': signal_analysis.get('signal_price', ''),
            'resistance_price': signal_analysis.get('resistance_price', ''),
            'penetration_percent': signal_analysis.get('penetration_percent', ''),
            'range_percent': signal_analysis.get('range_percent', '')
        }
    
    def _generate_regular_alerts_charts(self, summary: Dict, summary_dir: Path, date_str: str) -> List[str]:
        """Generate pie charts and bar chart for regular alerts by symbol count."""
        chart_files = []
        
        # Set matplotlib style for better looking charts
        plt.style.use('default')
        
        # Generate bullish alerts pie chart
        bullish_chart = self._create_alert_pie_chart(
            summary['bullish_analysis']['symbol_statistics'],
            'Bullish Alerts by Symbol',
            'bullish_alerts',
            summary_dir,
            date_str
        )
        if bullish_chart:
            chart_files.append(bullish_chart)
        
        # Generate bearish alerts pie chart  
        bearish_chart = self._create_alert_pie_chart(
            summary['bearish_analysis']['symbol_statistics'],
            'Bearish Alerts by Symbol',
            'bearish_alerts',
            summary_dir,
            date_str
        )
        if bearish_chart:
            chart_files.append(bearish_chart)
        
        # Generate bar chart for regular alerts
        bar_chart = self._generate_bar_chart(summary, summary_dir, date_str)
        if bar_chart:
            chart_files.append(bar_chart)
        
        return chart_files
    
    def _generate_superduper_alerts_charts(self, summary: Dict, summary_dir: Path, date_str: str) -> List[str]:
        """Generate bullish and bearish pie charts as well as bar charts for superduper alerts by symbol count."""
        chart_files = []
        
        # Set matplotlib style for better looking charts
        plt.style.use('default')
        
        # Calculate superduper alert symbol statistics
        bullish_superduper_stats = self._calculate_super_alert_symbol_statistics(
            summary['super_alerts']['bullish_alerts']
        )
        bearish_superduper_stats = self._calculate_super_alert_symbol_statistics(
            summary['super_alerts']['bearish_alerts']
        )
        
        if self.verbose:
            bullish_count = sum(stats['total_alerts'] for stats in bullish_superduper_stats.values()) if bullish_superduper_stats else 0
            bearish_count = sum(stats['total_alerts'] for stats in bearish_superduper_stats.values()) if bearish_superduper_stats else 0
            print(f"Generating superduper alerts charts: {bullish_count} bullish, {bearish_count} bearish")
        
        # Generate bullish superduper alerts pie chart
        if bullish_superduper_stats:
            bullish_superduper_chart = self._create_alert_pie_chart(
                bullish_superduper_stats,
                'Bullish Superduper Alerts by Symbol',
                'bullish_superduper_alerts',
                summary_dir,
                date_str
            )
            if bullish_superduper_chart:
                chart_files.append(bullish_superduper_chart)
                if self.verbose:
                    print(f"Generated bullish superduper alerts pie chart")
        elif self.verbose:
            print("No bullish superduper alerts found for pie chart")
        
        # Generate bearish superduper alerts pie chart
        if bearish_superduper_stats:
            bearish_superduper_chart = self._create_alert_pie_chart(
                bearish_superduper_stats,
                'Bearish Superduper Alerts by Symbol',
                'bearish_superduper_alerts',
                summary_dir,
                date_str
            )
            if bearish_superduper_chart:
                chart_files.append(bearish_superduper_chart)
                if self.verbose:
                    print(f"Generated bearish superduper alerts pie chart")
        elif self.verbose:
            print("No bearish superduper alerts found for pie chart")
        
        # Generate combined superduper alerts bar chart (bullish positive, bearish negative)
        superduper_bar_chart = self._generate_superduper_alerts_bar_chart(summary, summary_dir, date_str)
        if superduper_bar_chart:
            chart_files.append(superduper_bar_chart)
            if self.verbose:
                print(f"Generated superduper alerts bar chart")
        
        return chart_files
    
    def _generate_high_impact_super_alerts_charts(self, summary: Dict, summary_dir: Path, date_str: str) -> List[str]:
        """Generate pie charts for high-impact super alerts (20% price movement from signal)."""
        chart_files = []
        
        # Set matplotlib style for better looking charts
        plt.style.use('default')
        
        # Filter bullish super alerts where current_price >= signal_price * 1.20
        high_impact_bullish_super_stats = self._calculate_high_impact_super_alert_symbol_statistics(
            summary['super_alerts']['bullish_alerts'], 'bullish'
        )
        
        # Filter bearish super alerts where current_price <= signal_price * 0.80
        high_impact_bearish_super_stats = self._calculate_high_impact_super_alert_symbol_statistics(
            summary['super_alerts']['bearish_alerts'], 'bearish'
        )
        
        # Generate bullish high-impact super alerts pie chart
        if high_impact_bullish_super_stats:
            bullish_high_impact_chart = self._create_alert_pie_chart(
                high_impact_bullish_super_stats,
                'High-Impact Bullish Super Alerts by Symbol (≥20% Above Signal)',
                'high_impact_bullish_super_alerts',
                summary_dir,
                date_str
            )
            if bullish_high_impact_chart:
                chart_files.append(bullish_high_impact_chart)
        
        # Generate bearish high-impact super alerts pie chart
        if high_impact_bearish_super_stats:
            bearish_high_impact_chart = self._create_alert_pie_chart(
                high_impact_bearish_super_stats,
                'High-Impact Bearish Super Alerts by Symbol (≥20% Below Signal)',
                'high_impact_bearish_super_alerts',
                summary_dir,
                date_str
            )
            if bearish_high_impact_chart:
                chart_files.append(bearish_high_impact_chart)
        
        if self.verbose:
            bullish_count = sum(stats['total_alerts'] for stats in high_impact_bullish_super_stats.values()) if high_impact_bullish_super_stats else 0
            bearish_count = sum(stats['total_alerts'] for stats in high_impact_bearish_super_stats.values()) if high_impact_bearish_super_stats else 0
            print(f"Generated high-impact super alerts charts: {bullish_count} bullish, {bearish_count} bearish")
        
        return chart_files
    
    def _calculate_high_impact_super_alert_symbol_statistics(self, super_alerts: List[Dict], alert_direction: str) -> Dict:
        """Calculate symbol statistics for high-impact super alerts based on price movement."""
        symbol_stats = defaultdict(lambda: {
            'total_alerts': 0,
            'alerts': []
        })
        
        for alert in super_alerts:
            signal_analysis = alert.get('signal_analysis', {})
            current_price = signal_analysis.get('current_price')
            signal_price = signal_analysis.get('signal_price')
            
            # Skip alerts without required price data
            if not current_price or not signal_price:
                if self.verbose:
                    print(f"Skipping alert due to missing price data: current={current_price}, signal={signal_price}")
                continue
            
            # Determine if this is a high-impact alert based on direction
            is_high_impact = False
            
            if alert_direction == 'bullish':
                # Bullish: current_price >= signal_price * 1.20 (20% above signal)
                is_high_impact = current_price >= signal_price * 1.20
                if self.verbose and is_high_impact:
                    ratio = (current_price / signal_price - 1) * 100
                    print(f"High-impact bullish alert for {alert.get('symbol')}: {ratio:.1f}% above signal")
            elif alert_direction == 'bearish':
                # Bearish: current_price <= signal_price * 0.80 (20% below signal)
                is_high_impact = current_price <= signal_price * 0.80
                if self.verbose and is_high_impact:
                    ratio = (1 - current_price / signal_price) * 100
                    print(f"High-impact bearish alert for {alert.get('symbol')}: {ratio:.1f}% below signal")
            
            # Add to statistics if it's a high-impact alert
            if is_high_impact:
                symbol = alert.get('symbol', 'UNKNOWN')
                stats = symbol_stats[symbol]
                stats['total_alerts'] += 1
                stats['alerts'].append(alert)
        
        return dict(symbol_stats)
    
    def _generate_superduper_alerts_ratio_bar_charts(self, summary: Dict, summary_dir: Path, date_str: str) -> List[str]:
        """Generate bar charts for superduper alerts binned by (current_price / orb_high) ratio in 10% increments."""
        chart_files = []
        
        # Set matplotlib style for better looking charts
        plt.style.use('default')
        
        # Generate bullish superduper alerts ratio bar chart
        bullish_ratio_chart = self._create_superduper_alerts_ratio_bar_chart(
            summary['super_alerts']['bullish_alerts'], 
            'bullish', 
            summary_dir, 
            date_str
        )
        if bullish_ratio_chart:
            chart_files.append(bullish_ratio_chart)
        
        # Generate bearish superduper alerts ratio bar chart
        bearish_ratio_chart = self._create_superduper_alerts_ratio_bar_chart(
            summary['super_alerts']['bearish_alerts'], 
            'bearish', 
            summary_dir, 
            date_str
        )
        if bearish_ratio_chart:
            chart_files.append(bearish_ratio_chart)
        
        if self.verbose and chart_files:
            print(f"Generated {len(chart_files)} superduper alerts ratio bar charts")
        
        return chart_files
    
    def _create_superduper_alerts_ratio_bar_chart(self, superduper_alerts: List[Dict], alert_direction: str, 
                                           summary_dir: Path, date_str: str) -> Optional[str]:
        """Create a bar chart for superduper alerts binned by (current_price / orb_high) ratio."""
        if not superduper_alerts:
            if self.verbose:
                print(f"No {alert_direction} superduper alerts for ratio bar chart")
            return None
        
        # Calculate ratios and bin them
        ratios = []
        valid_alerts = 0
        
        for alert in superduper_alerts:
            original_alert = alert.get('original_alert', {})
            current_price = original_alert.get('current_price')
            orb_high = original_alert.get('orb_high')
            
            if current_price and orb_high and orb_high > 0:
                ratio = current_price / orb_high
                ratios.append(ratio)
                valid_alerts += 1
            elif self.verbose:
                print(f"Skipping alert due to missing data: current_price={current_price}, orb_high={orb_high}")
        
        if not ratios:
            if self.verbose:
                print(f"No valid ratio data for {alert_direction} superduper alerts")
            return None
        
        # Define bins (10% increments)
        # Start from 0.5 (50%) to 2.5 (250%) in 10% increments
        bin_edges = [i * 0.1 for i in range(5, 26)]  # 0.5, 0.6, 0.7, ..., 2.4, 2.5
        bin_labels = [f"{int(edge*100)}%-{int((edge+0.1)*100-1)}%" for edge in bin_edges[:-1]]
        
        # Add overflow bins for extreme values
        if min(ratios) < 0.5:
            bin_edges.insert(0, 0.0)
            bin_labels.insert(0, "<50%")
        if max(ratios) >= 2.5:
            bin_edges.append(float('inf'))
            bin_labels.append("≥250%")
        
        # Bin the ratios
        hist, _ = np.histogram(ratios, bins=bin_edges)
        
        # Filter out empty bins and their labels
        non_zero_indices = hist > 0
        filtered_counts = hist[non_zero_indices]
        filtered_labels = [bin_labels[i] for i in range(len(bin_labels)) if non_zero_indices[i]]
        
        if len(filtered_counts) == 0:
            if self.verbose:
                print(f"No data to plot for {alert_direction} superduper alerts ratio chart")
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar chart
        x_pos = range(len(filtered_labels))
        color = 'green' if alert_direction == 'bullish' else 'red'
        alpha = 0.7
        
        bars = ax.bar(x_pos, filtered_counts, color=color, alpha=alpha, 
                     label=f'{alert_direction.title()} Superduper Alerts')
        
        # Customize the chart
        ax.set_xlabel('Current Price / ORB High Ratio', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Superduper Alerts', fontsize=12, fontweight='bold')
        direction_title = alert_direction.title()
        ax.set_title(f'{direction_title} Superduper Alerts Distribution by Price/ORB-High Ratio\n{self.target_date}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(filtered_labels, rotation=45, ha='right')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, filtered_counts)):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add statistics text
        mean_ratio = np.mean(ratios)
        median_ratio = np.median(ratios)
        stats_text = f'Mean: {mean_ratio:.2f}x | Median: {median_ratio:.2f}x | Total: {valid_alerts} alerts'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save chart
        chart_filename = f"bar_chart_{alert_direction}_superduper_alerts_ratio_{date_str}.png"
        chart_filepath = summary_dir / chart_filename
        
        plt.savefig(chart_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Generated {alert_direction} superduper alerts ratio bar chart: {chart_filename}")
            print(f"  Ratios range: {min(ratios):.2f}x to {max(ratios):.2f}x")
            print(f"  Mean ratio: {mean_ratio:.2f}x, Median: {median_ratio:.2f}x")
        
        return str(chart_filepath)
    
    def _calculate_super_alert_symbol_statistics(self, super_alerts: List[Dict]) -> Dict:
        """Calculate symbol statistics for super alerts."""
        symbol_stats = defaultdict(lambda: {
            'total_alerts': 0,
            'alerts': []
        })
        
        for alert in super_alerts:
            symbol = alert.get('symbol', 'UNKNOWN')
            stats = symbol_stats[symbol]
            stats['total_alerts'] += 1
            stats['alerts'].append(alert)
        
        return dict(symbol_stats)
    
    def _generate_superduper_alerts_bar_chart(self, summary: Dict, summary_dir: Path, date_str: str) -> Optional[str]:
        """Generate a bar chart showing bullish (positive) and bearish (negative) superduper alerts by symbol."""
        bullish_stats = self._calculate_super_alert_symbol_statistics(
            summary['super_alerts']['bullish_alerts']
        )
        bearish_stats = self._calculate_super_alert_symbol_statistics(
            summary['super_alerts']['bearish_alerts']
        )
        
        # Combine all symbols from both bullish and bearish super alerts
        all_symbols = set()
        if bullish_stats:
            all_symbols.update(bullish_stats.keys())
        if bearish_stats:
            all_symbols.update(bearish_stats.keys())
        
        if not all_symbols:
            if self.verbose:
                print("No symbols found for superduper alerts bar chart")
            return None
        
        # Sort symbols alphabetically (ascending)
        sorted_symbols = sorted(all_symbols)
        
        # Prepare data for bar chart
        bullish_counts = []
        bearish_counts = []
        
        for symbol in sorted_symbols:
            # Get bullish count (positive)
            bullish_count = bullish_stats.get(symbol, {}).get('total_alerts', 0)
            bullish_counts.append(bullish_count)
            
            # Get bearish count (negative for chart display)
            bearish_count = bearish_stats.get(symbol, {}).get('total_alerts', 0)
            bearish_counts.append(-bearish_count)  # Negative for bearish
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar chart
        x_pos = range(len(sorted_symbols))
        
        # Plot bullish bars (positive, green)
        bullish_bars = ax.bar(x_pos, bullish_counts, color='green', alpha=0.7, label='Bullish Superduper Alerts')
        
        # Plot bearish bars (negative, red)
        bearish_bars = ax.bar(x_pos, bearish_counts, color='red', alpha=0.7, label='Bearish Superduper Alerts')
        
        # Customize the chart
        ax.set_xlabel('Symbols', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Superduper Alerts', fontsize=12, fontweight='bold')
        ax.set_title(f'Superduper Alert Distribution by Symbol\n{self.target_date}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_symbols, rotation=45, ha='right')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bull_count, bear_count) in enumerate(zip(bullish_counts, bearish_counts)):
            if bull_count > 0:
                ax.text(i, bull_count + 0.5, str(bull_count), ha='center', va='bottom', fontweight='bold', fontsize=9)
            if bear_count < 0:
                ax.text(i, bear_count - 0.5, str(-bear_count), ha='center', va='top', fontweight='bold', fontsize=9)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save chart
        chart_filename = f"bar_chart_superduper_alerts_{date_str}.png"
        chart_filepath = summary_dir / chart_filename
        
        plt.savefig(chart_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_filepath)
    
    def _create_alert_pie_chart(self, symbol_stats: Dict, title: str, alert_type: str, 
                               summary_dir: Path, date_str: str) -> Optional[str]:
        """Create a pie chart for alert distribution by symbol."""
        if not symbol_stats:
            if self.verbose:
                print(f"No data for {alert_type} pie chart")
            return None
        
        # Extract symbol names and alert counts
        symbols = []
        counts = []
        for symbol, stats in symbol_stats.items():
            symbols.append(symbol)
            counts.append(stats['total_alerts'])
        
        if not counts or sum(counts) == 0:
            if self.verbose:
                print(f"No alerts found for {alert_type} pie chart")
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart
        colors = plt.cm.Set3(range(len(symbols)))
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=symbols, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        # Customize appearance
        ax.set_title(f'{title}\n{self.target_date}', fontsize=14, fontweight='bold', pad=20)
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Add legend with counts
        legend_labels = [f'{symbol}: {count} alerts' for symbol, count in zip(symbols, counts)]
        ax.legend(wedges, legend_labels, title="Alert Counts", loc="center left", 
                 bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Save chart
        chart_filename = f"pie_chart_{alert_type}_{date_str}.png"
        chart_filepath = summary_dir / chart_filename
        
        plt.tight_layout()
        plt.savefig(chart_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_filepath)
    
    def _generate_bar_chart(self, summary: Dict, summary_dir: Path, date_str: str) -> Optional[str]:
        """Generate a bar chart showing bullish (positive) and bearish (negative) alerts by symbol."""
        bullish_stats = summary['bullish_analysis']['symbol_statistics']
        bearish_stats = summary['bearish_analysis']['symbol_statistics']
        
        # Combine all symbols from both bullish and bearish alerts
        all_symbols = set()
        if bullish_stats:
            all_symbols.update(bullish_stats.keys())
        if bearish_stats:
            all_symbols.update(bearish_stats.keys())
        
        if not all_symbols:
            if self.verbose:
                print("No symbols found for bar chart")
            return None
        
        # Sort symbols alphabetically (ascending)
        sorted_symbols = sorted(all_symbols)
        
        # Prepare data for bar chart
        bullish_counts = []
        bearish_counts = []
        
        for symbol in sorted_symbols:
            # Get bullish count (positive)
            bullish_count = bullish_stats.get(symbol, {}).get('total_alerts', 0)
            bullish_counts.append(bullish_count)
            
            # Get bearish count (negative for chart display)
            bearish_count = bearish_stats.get(symbol, {}).get('total_alerts', 0)
            bearish_counts.append(-bearish_count)  # Negative for bearish
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar chart
        x_pos = range(len(sorted_symbols))
        
        # Plot bullish bars (positive, green)
        bullish_bars = ax.bar(x_pos, bullish_counts, color='green', alpha=0.7, label='Bullish Alerts')
        
        # Plot bearish bars (negative, red)
        bearish_bars = ax.bar(x_pos, bearish_counts, color='red', alpha=0.7, label='Bearish Alerts')
        
        # Customize the chart
        ax.set_xlabel('Symbols', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Alerts', fontsize=12, fontweight='bold')
        ax.set_title(f'Alert Distribution by Symbol\n{self.target_date}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_symbols, rotation=45, ha='right')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bull_count, bear_count) in enumerate(zip(bullish_counts, bearish_counts)):
            if bull_count > 0:
                ax.text(i, bull_count + 0.5, str(bull_count), ha='center', va='bottom', fontweight='bold', fontsize=9)
            if bear_count < 0:
                ax.text(i, bear_count - 0.5, str(-bear_count), ha='center', va='top', fontweight='bold', fontsize=9)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save chart
        chart_filename = f"bar_chart_alerts_{date_str}.png"
        chart_filepath = summary_dir / chart_filename
        
        plt.savefig(chart_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_filepath)
    
    def print_summary_stats(self, summary: Dict) -> None:
        """Print key summary statistics."""
        metadata = summary['metadata']
        
        print(f"\n📊 ALERTS SUMMARY - {metadata['date']}")
        print("=" * 60)
        print(f"Total Alerts: {metadata['total_alerts']}")
        print(f"  📈 Bullish: {metadata['bullish_alerts']}")
        print(f"  📉 Bearish: {metadata['bearish_alerts']}")
        print(f"  🚀 Superduper: {metadata['superduper_alerts']}")
        print(f"Unique Symbols: {metadata['unique_symbols']}")
        
        # Most active symbols
        most_active = summary['overall_statistics']['most_active_symbols']
        if most_active:
            print(f"\n🏆 Most Active Symbols:")
            for symbol, count in most_active[:5]:
                print(f"  {symbol}: {count} alerts")
        
        # Highest confidence
        highest_conf = summary['overall_statistics']['highest_confidence_alerts']
        if highest_conf:
            print(f"\n⭐ Highest Confidence Alerts:")
            for alert in highest_conf[:3]:
                symbol = alert.get('symbol', 'N/A')
                confidence = alert.get('confidence_score', 0)
                price = alert.get('current_price', 0)
                breakout = alert.get('breakout_percentage', 0)
                print(f"  {symbol}: {confidence:.3f} confidence, ${price:.2f} (+{breakout:.1f}%)")
        
        # Time distribution
        time_dist = summary['overall_statistics']['time_distribution']
        if time_dist.get('peak_hour'):
            peak_hour, peak_count = time_dist['peak_hour']
            print(f"\n⏰ Peak Activity: {peak_hour:02d}:xx ({peak_count} alerts)")


def list_available_dates() -> None:
    """List all available dates."""
    generator = AlertsSummaryGenerator()
    print("\n📅 Available dates (10 most recent):")
    for i, date in enumerate(generator.available_dates, 1):
        marker = " (default)" if i == 1 else ""
        print(f"  {i}. {date}{marker}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORB Alerts Summary Generator")
    
    parser.add_argument(
        "--date",
        type=str,
        help="Date in YYYY-MM-DD format (default: most recent)"
    )
    
    parser.add_argument(
        "--list-dates",
        action="store_true",
        help="List available dates and exit"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        if args.list_dates:
            list_available_dates()
            return
        
        # Create generator and generate summary
        generator = AlertsSummaryGenerator(date=args.date, verbose=args.verbose)
        summary = generator.generate_summary()
        
        # Save summary
        json_file, csv_file, chart_files = generator.save_summary(summary)
        
        # Print key statistics
        generator.print_summary_stats(summary)
        
        print(f"\n✅ Complete summary saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📊 CSV: {csv_file}")
        if chart_files:
            print(f"   📈 Charts: {len(chart_files)} charts generated")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()