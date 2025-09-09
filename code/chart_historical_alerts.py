#!/usr/bin/env python3
"""
chart_historical_alerts.py

Search historical_data for sent superduper alerts and generate charts for each stock/date combination.
Calls alpaca.py --plot --symbol <symbol> --date <YYYY-MM-DD> and copies charts to /tmp.
"""

import os
import re
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class HistoricalAlertCharter:
    """
    Finds historical superduper alerts and generates charts for each stock/date combination.
    """
    
    def __init__(self, base_dir="historical_data", tmp_dir="./tmp/alpaca_charts"):
        self.base_dir = Path(base_dir)
        self.tmp_dir = Path(tmp_dir)
        self.alpaca_script = Path("code/alpaca.py")
        
        # Create tmp directory if it doesn't exist
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created tmp directory: {self.tmp_dir}")
    
    def find_alert_files(self):
        """Find all superduper alert JSON files in historical_data directory."""
        pattern = self.base_dir / "*/superduper_alerts_sent/**/*.json"
        alert_files = list(self.base_dir.rglob("superduper_alerts_sent/**/*.json"))
        
        print(f"🔍 Found {len(alert_files)} alert files")
        return alert_files
    
    def extract_symbol_and_date(self, alert_file):
        """
        Extract symbol and date from alert file path and filename.
        
        Expected format: historical_data/YYYY-MM-DD/superduper_alerts_sent/.../superduper_alert_SYMBOL_YYYYMMDD_HHMMSS.json
        """
        # Extract date from directory path
        path_parts = alert_file.parts
        date_match = None
        for part in path_parts:
            if re.match(r'\d{4}-\d{2}-\d{2}', part):
                date_match = part
                break
        
        if not date_match:
            print(f"⚠️  No date found in path: {alert_file}")
            return None, None
        
        # Extract symbol from filename
        filename = alert_file.name
        # Pattern: superduper_alert_SYMBOL_YYYYMMDD_HHMMSS.json
        symbol_match = re.search(r'superduper_alert_([A-Z]+)_\d{8}_\d{6}\.json', filename)
        
        if not symbol_match:
            print(f"⚠️  No symbol found in filename: {filename}")
            return None, None
        
        symbol = symbol_match.group(1)
        
        # Skip test symbols
        if symbol in ['TEST', 'BAD']:
            print(f"⏭️  Skipping test symbol: {symbol}")
            return None, None
        
        return symbol, date_match
    
    def get_unique_symbol_date_combinations(self):
        """Get unique symbol/date combinations from all alert files."""
        alert_files = self.find_alert_files()
        combinations = defaultdict(set)  # date -> set of symbols
        
        for alert_file in alert_files:
            symbol, date = self.extract_symbol_and_date(alert_file)
            if symbol and date:
                combinations[date].add(symbol)
        
        # Convert to sorted list of tuples
        sorted_combinations = []
        for date in sorted(combinations.keys()):
            for symbol in sorted(combinations[date]):
                sorted_combinations.append((symbol, date))
        
        print(f"📊 Found {len(sorted_combinations)} unique symbol/date combinations:")
        for symbol, date in sorted_combinations:
            print(f"  • {symbol} on {date}")
        
        return sorted_combinations, combinations
    
    def generate_chart(self, symbol, date):
        """Generate chart using alpaca.py --plot."""
        print(f"\n📈 Generating chart for {symbol} on {date}")
        
        # Build command - expand home directory
        python_path = os.path.expanduser("~/miniconda3/envs/alpaca/bin/python")
        cmd = [
            python_path,
            str(self.alpaca_script),
            "--plot",
            "--symbol", symbol,
            "--date", date
        ]
        
        try:
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully generated chart for {symbol} on {date}")
                return True
            else:
                print(f"❌ Error generating chart for {symbol} on {date}:")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Exception generating chart for {symbol} on {date}: {e}")
            return False
    
    def copy_chart_to_tmp(self, symbol, date):
        """Copy generated chart to /tmp directory."""
        # Chart should be in plots/YYYYMMDD/SYMBOL_chart.png
        date_formatted = date.replace('-', '')
        chart_path = Path(f"plots/{date_formatted}/{symbol}_chart.png")
        
        if chart_path.exists():
            # Copy to tmp with descriptive name
            tmp_filename = f"{symbol}_{date}_chart.png"
            tmp_path = self.tmp_dir / tmp_filename
            
            shutil.copy2(chart_path, tmp_path)
            print(f"📋 Copied chart to: {tmp_path}")
            return True
        else:
            print(f"⚠️  Chart not found at: {chart_path}")
            return False
    
    def create_sent_alerts_log(self, date, symbols_with_charts):
        """Create sent_superduper_alerts.log file for a specific date."""
        date_formatted = date.replace('-', '')
        plots_dir = Path(f"plots/{date_formatted}")
        
        # Create plots directory if it doesn't exist
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = plots_dir / "sent_superduper_alerts.log"
        
        # Write log file with symbols that have charts
        with open(log_path, 'w') as f:
            f.write(f"Sent Superduper Alerts for {date}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if symbols_with_charts:
                f.write(f"Charts generated for {len(symbols_with_charts)} symbols:\n")
                for symbol in sorted(symbols_with_charts):
                    chart_filename = f"{symbol}_chart.png"
                    f.write(f"  • {chart_filename}\n")
            else:
                f.write("No charts generated for this date.\n")
        
        print(f"📝 Created log file: {log_path}")
        return log_path
    
    def run_all_charts(self):
        """Generate charts for all unique symbol/date combinations."""
        combinations, date_symbols = self.get_unique_symbol_date_combinations()
        
        if not combinations:
            print("❌ No valid symbol/date combinations found")
            return
        
        print(f"\n🚀 Starting chart generation for {len(combinations)} combinations...")
        
        successful = 0
        failed = 0
        date_chart_tracking = defaultdict(set)  # date -> set of symbols with successful charts
        
        for symbol, date in combinations:
            print(f"\n{'='*60}")
            print(f"Processing: {symbol} on {date}")
            print(f"{'='*60}")
            
            # Generate chart
            if self.generate_chart(symbol, date):
                # Copy to tmp
                if self.copy_chart_to_tmp(symbol, date):
                    successful += 1
                    date_chart_tracking[date].add(symbol)
                else:
                    failed += 1
            else:
                failed += 1
        
        # Create log files for each date
        print(f"\n📝 Creating sent_superduper_alerts.log files...")
        for date in date_chart_tracking:
            symbols_with_charts = date_chart_tracking[date]
            self.create_sent_alerts_log(date, symbols_with_charts)
        
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY:")
        print(f"  ✅ Successful: {successful}")
        print(f"  ❌ Failed: {failed}")
        print(f"  📁 Charts saved to: {self.tmp_dir}")
        print(f"  📝 Log files created for {len(date_chart_tracking)} dates")
        print(f"{'='*60}")


def main():
    """Main entry point."""
    print("🚀 Historical Alert Charter")
    print("=" * 50)
    
    charter = HistoricalAlertCharter()
    charter.run_all_charts()


if __name__ == "__main__":
    main()