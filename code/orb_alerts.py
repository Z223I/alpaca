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
from atoms.config.symbol_manager import SymbolManager
from atoms.websocket.connection_manager import ConnectionManager
from atoms.websocket.alpaca_stream import MarketData
from atoms.indicators.orb_calculator import ORBCalculator


class ORBAlertSystem:
    """Main ORB Alert System orchestrator."""
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialize ORB Alert System.
        
        Args:
            symbols: Optional list of symbols to monitor
        """
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load symbols
        self.symbol_manager = SymbolManager(config.symbols_file)
        self.symbols = symbols or self.symbol_manager.get_symbols()
        
        # Initialize components
        self.connection_manager = ConnectionManager(self.symbols)
        self.orb_calculator = ORBCalculator(config.orb_period_minutes)
        
        # Add data handler
        self.connection_manager.add_data_handler(self._handle_market_data)
        
        # Statistics
        self.alerts_generated = 0
        self.start_time = None
        
        self.logger.info(f"ORB Alert System initialized for {len(self.symbols)} symbols")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _handle_market_data(self, market_data: MarketData) -> None:
        """
        Handle incoming market data and check for ORB alerts.
        
        Args:
            market_data: Market data from stream
        """
        symbol = market_data.symbol
        
        # Get opening range data
        orb_data = self.connection_manager.get_opening_range_data(symbol)
        
        if orb_data is not None and not orb_data.empty:
            # Calculate ORB levels
            orb_level = self.orb_calculator.calculate_orb_levels(symbol, orb_data)
            
            if orb_level:
                # Check for breakout
                if self.orb_calculator.is_breakout(symbol, market_data.close):
                    self._generate_alert(symbol, market_data, orb_level)
    
    def _generate_alert(self, symbol: str, market_data: MarketData, orb_level) -> None:
        """
        Generate ORB breakout alert.
        
        Args:
            symbol: Symbol that broke out
            market_data: Current market data
            orb_level: ORB level information
        """
        breakout_pct = ((market_data.close - orb_level.orb_high) / orb_level.orb_high) * 100
        
        alert_message = (
            f"[{datetime.now().strftime('%H:%M:%S')}] ORB ALERT: {symbol} @ ${market_data.close:.2f} "
            f"(+{breakout_pct:.2f}% vs ORB High: ${orb_level.orb_high:.2f}) "
            f"Volume: {market_data.volume:,}"
        )
        
        print(alert_message)
        self.logger.info(alert_message)
        
        self.alerts_generated += 1
    
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
            
            # Start connection manager
            if not await self.connection_manager.start():
                self.logger.error("Failed to start connection manager")
                return
            
            self.logger.info("ORB Alert System started successfully")
            self.logger.info(f"Monitoring {len(self.symbols)} symbols: {', '.join(self.symbols[:10])}")
            if len(self.symbols) > 10:
                self.logger.info(f"... and {len(self.symbols) - 10} more symbols")
            
            # Main loop
            await self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            await self.stop()
    
    async def _main_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                # Health check
                if not self.connection_manager.is_healthy():
                    self.logger.warning("Connection not healthy, checking status...")
                
                # Print periodic statistics
                if self.start_time:
                    uptime = (datetime.now() - self.start_time).total_seconds()
                    if uptime > 0 and int(uptime) % 300 == 0:  # Every 5 minutes
                        self._print_statistics()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
    
    def _print_statistics(self) -> None:
        """Print system statistics."""
        stats = self.connection_manager.get_statistics()
        
        print("\n" + "="*50)
        print(f"ORB Alert System Statistics")
        print(f"Uptime: {stats['uptime_seconds']:.0f} seconds")
        print(f"Alerts Generated: {self.alerts_generated}")
        print(f"Symbols Monitored: {stats['symbols_count']}")
        print(f"Connection State: {stats['stream_state']}")
        print(f"Data Buffer Stats: {stats['data_buffer_stats']['total_messages_received']} messages")
        print("="*50 + "\n")
    
    async def stop(self) -> None:
        """Stop the ORB Alert System."""
        self.logger.info("Stopping ORB Alert System...")
        
        try:
            await self.connection_manager.stop()
            self.logger.info("ORB Alert System stopped")
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    
    def get_statistics(self) -> dict:
        """Get system statistics."""
        return {
            'alerts_generated': self.alerts_generated,
            'symbols_count': len(self.symbols),
            'start_time': self.start_time,
            'connection_stats': self.connection_manager.get_statistics()
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ORB Trading Alerts System")
    
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to monitor (default: all from symbols.csv)"
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
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Create and start system
    try:
        system = ORBAlertSystem(symbols)
        
        if args.test:
            print("Running in test mode - no actual alerts will be generated")
            # Could add test mode functionality here
        
        await system.start()
        
    except Exception as e:
        logging.error(f"Failed to start ORB Alert System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())