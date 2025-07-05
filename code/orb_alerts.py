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
from typing import List
from datetime import datetime

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.config.alert_config import config
from molecules.orb_alert_engine import ORBAlertEngine
from atoms.alerts.alert_formatter import ORBAlert


class ORBAlertSystem:
    """Main ORB Alert System orchestrator."""
    
    def __init__(self, symbols_file: str = None, test_mode: bool = False):
        """
        Initialize ORB Alert System.
        
        Args:
            symbols_file: Path to symbols CSV file
            test_mode: Run in test mode (no actual alerts)
        """
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize alert engine
        self.alert_engine = ORBAlertEngine(symbols_file)
        self.test_mode = test_mode
        
        # Add alert callback
        self.alert_engine.add_alert_callback(self._handle_alert)
        
        # Statistics
        self.start_time = None
        
        self.logger.info(f"ORB Alert System initialized in {'TEST' if test_mode else 'LIVE'} mode")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _handle_alert(self, alert: ORBAlert) -> None:
        """
        Handle generated ORB alert.
        
        Args:
            alert: Generated ORB alert
        """
        if self.test_mode:
            print(f"[TEST MODE] {alert.alert_message}")
            self.logger.info(f"Test alert: {alert.symbol} - {alert.priority.value}")
        else:
            # Alert is already printed by the engine
            pass
    
    async def start(self) -> None:
        """Start the ORB Alert System."""
        self.logger.info("Starting ORB Alert System...")
        self.start_time = datetime.now()
        
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
            
            # Start the alert engine
            await self.alert_engine.start()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in alert engine: {e}")
        finally:
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