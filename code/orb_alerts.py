"""
ORB Trading Alerts System - Main Entry Point

This is the main entry point for the ORB (Opening Range Breakout) trading alerts system.
Based on PCA analysis showing 82.31% variance explained by ORB patterns.

Usage:
    python3 code/orb_alerts.py                    # Start monitoring all symbols
    python3 code/orb_alerts.py --symbols AAPL,TSLA # Monitor specific symbols
    python3 code/orb_alerts.py --test             # Run in test mode
"""

import asyncio
import argparse
import logging
import sys
import json
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.config.alert_config import config
from molecules.orb_alert_engine import ORBAlertEngine
from atoms.alerts.alert_formatter import ORBAlert


class ORBAlertSystem:
    """Main ORB Alert System orchestrator."""
    
    def __init__(self, symbols_file: Optional[str] = None, test_mode: bool = False):
        """
        Initialize ORB Alert System.
        
        Args:
            symbols_file: Path to symbols CSV file
            test_mode: Run in test mode (no actual alerts)
        """
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize alert engine
        self.alert_engine = ORBAlertEngine(symbols_file) if symbols_file is not None else ORBAlertEngine()
        self.test_mode = test_mode
        
        # Historical data storage setup
        self.historical_data_dir = Path("historical_data")
        self.historical_data_dir.mkdir(exist_ok=True)
        self._setup_data_storage()
        
        # Add alert callback
        self.alert_engine.add_alert_callback(self._handle_alert)
        
        # Statistics
        self.start_time = None
        self.last_data_save = None
        self.data_save_interval = timedelta(minutes=1)  # Save every 1 minute
        
        self.logger.info(f"ORB Alert System initialized in {'TEST' if test_mode else 'LIVE'} mode")
        self.logger.info(f"Historical data will be saved to: {self.historical_data_dir.absolute()}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _setup_data_storage(self) -> None:
        """Setup historical data storage directories."""
        # Create subdirectories for organized storage
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_data_dir = self.historical_data_dir / today
        self.daily_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        (self.daily_data_dir / "market_data").mkdir(exist_ok=True)
        (self.daily_data_dir / "alerts").mkdir(exist_ok=True)
        (self.daily_data_dir / "summary").mkdir(exist_ok=True)
        
        self.logger.info(f"Daily data directory: {self.daily_data_dir}")
    
    def _save_historical_data(self) -> None:
        """Save current market data to historical files."""
        try:
            current_time = datetime.now()
            symbols = self.alert_engine.get_monitored_symbols()
            
            for symbol in symbols:
                # Get symbol data from the data buffer
                symbol_data = self.alert_engine.data_buffer.get_symbol_data(symbol)
                
                if symbol_data is not None and not symbol_data.empty:
                    # Save to CSV format for easy analysis
                    filename = f"{symbol}_{current_time.strftime('%Y%m%d_%H%M%S')}.csv"
                    filepath = self.daily_data_dir / "market_data" / filename
                    
                    symbol_data.to_csv(filepath, index=False)
                    self.logger.debug(f"Saved {len(symbol_data)} records for {symbol} to {filename}")
            
            # Save metadata about the data save
            metadata = {
                "timestamp": current_time.isoformat(),
                "symbols_count": len(symbols),
                "symbols": symbols,
                "save_interval_minutes": self.data_save_interval.total_seconds() / 60,
                "format": "CSV",
                "total_records_saved": sum(
                    len(data) if (data := self.alert_engine.data_buffer.get_symbol_data(symbol)) is not None else 0
                    for symbol in symbols
                )
            }
            
            metadata_file = self.daily_data_dir / "summary" / f"save_metadata_{current_time.strftime('%H%M%S')}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.last_data_save = current_time
            self.logger.info(f"Historical data saved for {len(symbols)} symbols at {current_time.strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Error saving historical data: {e}")
    
    def _should_save_data(self) -> bool:
        """Check if it's time to save historical data."""
        if self.last_data_save is None:
            return True
        
        return datetime.now() - self.last_data_save >= self.data_save_interval
    
    def _handle_alert(self, alert: ORBAlert) -> None:
        """
        Handle generated ORB alert.
        
        Args:
            alert: Generated ORB alert
        """
        # Save alert to historical data
        self._save_alert_data(alert)
        
        if self.test_mode:
            print(f"[TEST MODE] {alert.alert_message}")
            self.logger.info(f"Test alert: {alert.symbol} - {alert.priority.value}")
        else:
            # Alert is already printed by the engine
            pass
    
    def _save_alert_data(self, alert: ORBAlert) -> None:
        """Save alert data to historical files."""
        try:
            # Save alert as JSON
            alert_filename = f"alert_{alert.symbol}_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            alert_filepath = self.daily_data_dir / "alerts" / alert_filename
            
            # Convert alert to dictionary for JSON serialization
            alert_data = {
                "symbol": alert.symbol,
                "timestamp": alert.timestamp.isoformat(),
                "current_price": alert.current_price,
                "orb_high": alert.orb_high,
                "orb_low": alert.orb_low,
                "orb_range": alert.orb_range,
                "orb_midpoint": alert.orb_midpoint,
                "breakout_type": alert.breakout_type.value,
                "breakout_percentage": alert.breakout_percentage,
                "volume_ratio": alert.volume_ratio,
                "confidence_score": alert.confidence_score,
                "priority": alert.priority.value,
                "confidence_level": alert.confidence_level,
                "recommended_stop_loss": alert.recommended_stop_loss,
                "recommended_take_profit": alert.recommended_take_profit,
                "alert_message": alert.alert_message
            }
            
            with open(alert_filepath, 'w') as f:
                json.dump(alert_data, f, indent=2)
                
            self.logger.debug(f"Saved alert data for {alert.symbol} to {alert_filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving alert data: {e}")
    
    async def _periodic_data_save(self) -> None:
        """Background task to periodically save historical data."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self._should_save_data():
                    self._save_historical_data()
                    
            except asyncio.CancelledError:
                self.logger.info("Periodic data save task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in periodic data save: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def start(self) -> None:
        """Start the ORB Alert System."""
        self.logger.info("Starting ORB Alert System...")
        self.start_time = datetime.now()
        
        # Start periodic data saving task
        data_save_task = None
        
        try:
            # Validate configuration
            config_errors = config.validate()
            if config_errors:
                self.logger.error(f"Configuration errors: {config_errors}")
                return
            
            # Start alert engine
            symbols = self.alert_engine.get_monitored_symbols()
            self.logger.info(f"Monitoring {len(symbols)} symbols: {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                self.logger.info(f"... and {len(symbols) - 10} more symbols")
            
            # Start periodic data saving in background
            data_save_task = asyncio.create_task(self._periodic_data_save())
            self.logger.info("Started periodic data saving task (every 1 minute)")
            
            # Start the alert engine
            await self.alert_engine.start()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in alert engine: {e}")
        finally:
            # Cancel data saving task
            if data_save_task:
                data_save_task.cancel()
                try:
                    await data_save_task
                except asyncio.CancelledError:
                    pass
                    
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the ORB Alert System."""
        self.logger.info("Stopping ORB Alert System...")
        
        try:
            await self.alert_engine.stop()
            self.logger.info("ORB Alert System stopped")
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    
    def get_statistics(self) -> dict:
        """Get system statistics."""
        engine_stats = self.alert_engine.get_stats()
        return {
            'alerts_generated': engine_stats.total_alerts_generated,
            'symbols_count': engine_stats.symbols_monitored,
            'start_time': self.start_time,
            'engine_stats': engine_stats
        }
    
    def print_daily_summary(self) -> None:
        """Print daily summary statistics."""
        summary = self.alert_engine.get_daily_summary()
        
        print("\n" + "="*60)
        print(f"ORB Alert System - Daily Summary")
        print(f"Date: {summary.get('date', 'N/A')}")
        print(f"Total Alerts: {summary.get('total_alerts', 0)}")
        print(f"Average Confidence: {summary.get('avg_confidence', 0):.3f}")
        print(f"Max Confidence: {summary.get('max_confidence', 0):.3f}")
        
        priority_breakdown = summary.get('priority_breakdown', {})
        print(f"Priority Breakdown:")
        for priority, count in priority_breakdown.items():
            print(f"  {priority}: {count}")
        
        print("="*60 + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORB Trading Alerts System")
    
    parser.add_argument(
        "--symbols-file",
        type=str,
        help="Path to symbols CSV file (default: data/symbols.csv)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (dry run)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show daily summary and exit"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start system
    try:
        system = ORBAlertSystem(
            symbols_file=args.symbols_file,
            test_mode=args.test
        )
        
        if args.summary:
            # Show summary and exit
            system.print_daily_summary()
            return
        
        if args.test:
            print("Running in test mode - alerts will be marked as [TEST MODE]")
        
        await system.start()
        
    except Exception as e:
        logging.error(f"Failed to start ORB Alert System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())