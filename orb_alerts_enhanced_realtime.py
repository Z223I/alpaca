#!/usr/bin/env python3
"""
Enhanced ORB Trading Alerts System - Real-Time Monitoring

This is the enhanced version of the ORB (Opening Range Breakout) trading alerts system
that monitors data continuously throughout the trading day, similar to orb_alerts.py.

Based on PCA analysis showing 82.31% variance explained by ORB patterns, with enhanced
filtering using:
- Volume Ratio Filter: orb_volume_ratio > 2.5x (improved profitability)
- Duration Filter: orb_duration_minutes > 10 (more realistic)
- Momentum Filter: orb_momentum > -0.01 (allows some negative momentum)
- Range Percentage Filter: 5-35% range optimization

Usage:
    python3 orb_alerts_enhanced_realtime.py                    # Start monitoring all symbols
    python3 orb_alerts_enhanced_realtime.py --symbols AAPL,TSLA # Monitor specific symbols
    python3 orb_alerts_enhanced_realtime.py --test             # Run in test mode
"""

import asyncio
import argparse
import logging
import sys
import json
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta, time
from pathlib import Path
import pandas as pd
import numpy as np
import pytz

# Add project root to path
sys.path.append('/home/wilsonb/dl/github.com/Z223I/alpaca')

from atoms.config.alert_config import config
from molecules.orb_alert_engine import ORBAlertEngine
from atoms.alerts.alert_formatter import ORBAlert
from atoms.websocket.alpaca_stream import MarketData

# Alpaca API imports
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    # Fallback to older alpaca-trade-api if available
    try:
        from alpaca_trade_api.rest import REST
        ALPACA_AVAILABLE = "legacy"
    except ImportError:
        ALPACA_AVAILABLE = False


class EnhancedORBAlert:
    """Enhanced ORB Alert with PCA-derived features."""
    
    def __init__(self, symbol: str, timestamp: datetime, alert_type: str, 
                 confidence: float, entry_price: float, stop_loss: float, 
                 target: float, volume_ratio: float, momentum: float, 
                 range_pct: float, expected_return: float, reasoning: str,
                 orb_features: Dict[str, Any]):
        self.symbol = symbol
        self.timestamp = timestamp
        self.alert_type = alert_type
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.target = target
        self.volume_ratio = volume_ratio
        self.momentum = momentum
        self.range_pct = range_pct
        self.expected_return = expected_return
        self.reasoning = reasoning
        self.orb_features = orb_features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "volume_ratio": self.volume_ratio,
            "momentum": self.momentum,
            "range_pct": self.range_pct,
            "expected_return": self.expected_return,
            "reasoning": self.reasoning,
            "orb_features": self.orb_features
        }
    
    @property
    def alert_message(self) -> str:
        """Generate alert message."""
        direction = "🚀 BULLISH" if "BULLISH" in self.alert_type else "🔻 BEARISH"
        return (f"{direction} {self.symbol} | Conf: {self.confidence:.0%} | "
                f"Entry: ${self.entry_price:.3f} | Vol: {self.volume_ratio:.1f}x | "
                f"Range: {self.range_pct:.1f}% | {self.reasoning}")


class EnhancedORBAlertSystem:
    """Enhanced ORB Alert System with real-time monitoring and PCA-derived filters."""
    
    def __init__(self, symbols_file: Optional[str] = None, test_mode: bool = False):
        """
        Initialize Enhanced ORB Alert System.
        
        Args:
            symbols_file: Path to symbols CSV file
            test_mode: Run in test mode (no actual alerts)
        """
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize the underlying ORB alert engine
        self.alert_engine = ORBAlertEngine(symbols_file) if symbols_file is not None else ORBAlertEngine()
        self.test_mode = test_mode
        
        # PCA-derived thresholds (adjusted for real-world application)
        self.volume_ratio_threshold = 2.5   # Reduced from 3.03 
        self.duration_threshold = 10        # Reduced from 195 (more realistic)
        self.momentum_threshold = -0.01     # Allow negative momentum too
        self.range_pct_min = 5.0           # More lenient minimum range
        self.range_pct_max = 35.0          # More lenient maximum range
        
        # Enhanced filtering parameters
        self.min_orb_samples = 3           # Minimum samples for ORB calculation
        self.orb_period_minutes = 15       # Standard ORB period
        self.market_open = time(9, 30)     # 9:30 AM ET
        
        # Real-time monitoring state
        self.orb_data_cache = {}           # Cache ORB data for each symbol
        self.alerts_generated = {}         # Track generated alerts to avoid duplicates
        self.enhanced_alerts = []          # Store enhanced alerts
        
        # Initialize historical data client
        self.historical_client = None
        if ALPACA_AVAILABLE == True:
            try:
                self.historical_client = StockHistoricalDataClient(
                    api_key=config.api_key,
                    secret_key=config.secret_key
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize historical data client: {e}")
        elif ALPACA_AVAILABLE == "legacy":
            try:
                self.historical_client = REST(
                    key_id=config.api_key,
                    secret_key=config.secret_key,
                    base_url=config.base_url
                )
                self.logger.info("Using legacy alpaca-trade-api for historical data")
            except Exception as e:
                self.logger.warning(f"Could not initialize legacy historical data client: {e}")
        
        # Historical data storage setup
        self.historical_data_dir = Path("historical_data")
        self.historical_data_dir.mkdir(exist_ok=True)
        self._setup_data_storage()
        
        # Add callback for market data to monitor in real-time
        self.alert_engine.data_buffer.add_callback(self._process_market_data)
        
        # Statistics
        self.start_time = None
        self.last_data_save = None
        self.data_save_interval = timedelta(minutes=config.data_save_interval_minutes)
        
        self.logger.info(f"Enhanced ORB Alert System initialized in {'TEST' if test_mode else 'LIVE'} mode")
        self.logger.info(f"PCA Filters: Vol>{self.volume_ratio_threshold}x, Duration>{self.duration_threshold}min, "
                        f"Momentum>{self.momentum_threshold}, Range:{self.range_pct_min}-{self.range_pct_max}%")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration with Eastern Time."""
        # Create custom formatter for Eastern Time
        class EasternFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                et_tz = pytz.timezone('US/Eastern')
                et_time = datetime.fromtimestamp(record.created, et_tz)
                if datefmt:
                    return et_time.strftime(datefmt)
                return et_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3] + ' ET'
        
        # Configure logging with Eastern Time
        logger = logging.getLogger(__name__)
        if not logger.handlers:  # Only configure if not already configured
            handler = logging.StreamHandler()
            formatter = EasternFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _setup_data_storage(self) -> None:
        """Setup historical data storage directories."""
        # Create subdirectories for organized storage
        et_tz = pytz.timezone('US/Eastern')
        today = datetime.now(et_tz).strftime("%Y-%m-%d")
        self.daily_data_dir = self.historical_data_dir / today
        self.daily_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        (self.daily_data_dir / "market_data").mkdir(exist_ok=True)
        (self.daily_data_dir / "alerts").mkdir(exist_ok=True)
        (self.daily_data_dir / "enhanced_alerts").mkdir(exist_ok=True)
        (self.daily_data_dir / "alerts" / "bullish").mkdir(exist_ok=True)
        (self.daily_data_dir / "alerts" / "bearish").mkdir(exist_ok=True)
        (self.daily_data_dir / "summary").mkdir(exist_ok=True)
        
        self.logger.info(f"Daily data directory: {self.daily_data_dir}")
    
    def _process_market_data(self, market_data: MarketData) -> None:
        """
        Process incoming market data for enhanced ORB analysis.
        
        Args:
            market_data: Real-time market data
        """
        try:
            # Check if we're in the ORB period or post-ORB
            et_tz = pytz.timezone('US/Eastern')
            current_time = datetime.now(et_tz)
            
            # Parse market open time for today
            market_open_hour, market_open_minute = map(int, config.market_open_time.split(':'))
            market_open_today = current_time.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
            orb_end_time = market_open_today + timedelta(minutes=self.orb_period_minutes)
            
            symbol = market_data.symbol
            
            # During ORB period: collect data
            if market_open_today <= current_time <= orb_end_time:
                self._collect_orb_data(symbol, market_data)
            
            # After ORB period: check for breakouts
            elif current_time > orb_end_time:
                self._check_enhanced_breakout(symbol, market_data, current_time)
                
        except Exception as e:
            self.logger.error(f"Error processing market data for {market_data.symbol}: {e}")
    
    def _collect_orb_data(self, symbol: str, market_data: MarketData) -> None:
        """Collect data during the ORB period."""
        if symbol not in self.orb_data_cache:
            self.orb_data_cache[symbol] = []
        
        self.orb_data_cache[symbol].append({
            'timestamp': market_data.timestamp,
            'high': market_data.high,
            'low': market_data.low,
            'close': market_data.close,
            'volume': market_data.volume,
            'trade_count': market_data.trade_count,
            'vwap': market_data.vwap
        })
    
    def _check_enhanced_breakout(self, symbol: str, market_data: MarketData, current_time: datetime) -> None:
        """Check for enhanced ORB breakouts using PCA-derived filters."""
        
        # Skip if we've already generated an alert for this symbol
        if symbol in self.alerts_generated:
            return
        
        # Skip if no ORB data collected
        if symbol not in self.orb_data_cache or len(self.orb_data_cache[symbol]) < self.min_orb_samples:
            return
        
        try:
            # Calculate enhanced ORB features
            orb_features = self._calculate_enhanced_orb_features(symbol)
            if not orb_features:
                return
            
            # Apply PCA-derived filters
            if not self._passes_pca_filters(orb_features):
                return
            
            # Check for breakout
            current_price = market_data.close
            orb_high = orb_features['orb_high']
            orb_low = orb_features['orb_low']
            
            alert = None
            
            # Bullish breakout above ORB high
            if current_price > orb_high:
                alert = self._create_enhanced_bullish_alert(symbol, current_time, current_price, orb_features)
            
            # Bearish breakdown below ORB low
            elif current_price < orb_low:
                alert = self._create_enhanced_bearish_alert(symbol, current_time, current_price, orb_features)
            
            if alert:
                self._handle_enhanced_alert(alert)
                self.alerts_generated[symbol] = current_time
                
        except Exception as e:
            self.logger.error(f"Error checking enhanced breakout for {symbol}: {e}")
    
    def _calculate_enhanced_orb_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Calculate enhanced ORB features with PCA insights."""
        
        orb_data = self.orb_data_cache[symbol]
        if not orb_data:
            return None
        
        try:
            df = pd.DataFrame(orb_data)
            
            # Basic ORB calculations
            orb_high = df['high'].max()
            orb_low = df['low'].min()
            orb_range = orb_high - orb_low
            orb_midpoint = (orb_high + orb_low) / 2
            
            # Enhanced features based on PCA analysis
            orb_volume = df['volume'].sum()
            orb_avg_volume = df['volume'].mean()
            orb_volume_ratio = orb_volume / orb_avg_volume if orb_avg_volume > 0 else 1.0
            
            # Price change and momentum
            orb_open = df.iloc[0]['close']
            orb_close = df.iloc[-1]['close']
            orb_price_change = orb_close - orb_open
            orb_price_change_pct = (orb_price_change / orb_open) * 100 if orb_open > 0 else 0
            
            # Duration and momentum
            orb_duration_minutes = len(df)
            orb_momentum = orb_price_change / orb_duration_minutes if orb_duration_minutes > 0 else 0
            
            # Range percentage
            orb_range_pct = (orb_range / orb_open) * 100 if orb_open > 0 else 0
            
            # Volatility
            orb_volatility = df['high'].std() if len(df) > 1 else 0
            
            # VWAP
            total_vwap_volume = (df['vwap'] * df['volume']).sum()
            total_volume = df['volume'].sum()
            orb_vwap = total_vwap_volume / total_volume if total_volume > 0 else orb_close
            
            return {
                'orb_high': orb_high,
                'orb_low': orb_low,
                'orb_range': orb_range,
                'orb_range_pct': orb_range_pct,
                'orb_midpoint': orb_midpoint,
                'orb_volume': orb_volume,
                'orb_avg_volume': orb_avg_volume,
                'orb_volume_ratio': orb_volume_ratio,
                'orb_price_change': orb_price_change,
                'orb_price_change_pct': orb_price_change_pct,
                'orb_volatility': orb_volatility,
                'orb_duration_minutes': orb_duration_minutes,
                'orb_momentum': orb_momentum,
                'orb_vwap': orb_vwap,
                'orb_open': orb_open,
                'orb_close': orb_close
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ORB features for {symbol}: {e}")
            return None
    
    def _passes_pca_filters(self, orb_features: Dict[str, Any]) -> bool:
        """Apply PCA-derived filters to determine if setup is worth alerting."""
        
        try:
            # Volume ratio filter
            if orb_features['orb_volume_ratio'] < self.volume_ratio_threshold:
                return False
            
            # Duration filter
            if orb_features['orb_duration_minutes'] < self.duration_threshold:
                return False
            
            # Momentum filter
            if orb_features['orb_momentum'] < self.momentum_threshold:
                return False
            
            # Range percentage filter
            range_pct = orb_features['orb_range_pct']
            if range_pct < self.range_pct_min or range_pct > self.range_pct_max:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying PCA filters: {e}")
            return False
    
    def _create_enhanced_bullish_alert(self, symbol: str, timestamp: datetime, 
                                     current_price: float, orb_features: Dict[str, Any]) -> EnhancedORBAlert:
        """Create enhanced bullish breakout alert."""
        
        entry_price = orb_features['orb_high'] * 1.002  # Entry slightly above ORB high
        stop_loss = orb_features['orb_low'] * 0.995     # Stop below ORB low
        target_range = orb_features['orb_range']
        target = entry_price + (target_range * 1.5)    # Target based on range projection
        
        # Calculate confidence based on PCA factors
        confidence = self._calculate_confidence(orb_features, 'bullish')
        
        # Expected return (simplified for real-time)
        expected_return = ((target - entry_price) / entry_price) * 100
        
        reasoning = (f"PCA-Enhanced: Vol Ratio {orb_features['orb_volume_ratio']:.1f}x, "
                    f"Momentum {orb_features['orb_momentum']:.3f}, "
                    f"Range {orb_features['orb_range_pct']:.1f}%")
        
        return EnhancedORBAlert(
            symbol=symbol,
            timestamp=timestamp,
            alert_type="ENHANCED_BULLISH_BREAKOUT",
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            volume_ratio=orb_features['orb_volume_ratio'],
            momentum=orb_features['orb_momentum'],
            range_pct=orb_features['orb_range_pct'],
            expected_return=expected_return,
            reasoning=reasoning,
            orb_features=orb_features
        )
    
    def _create_enhanced_bearish_alert(self, symbol: str, timestamp: datetime, 
                                     current_price: float, orb_features: Dict[str, Any]) -> EnhancedORBAlert:
        """Create enhanced bearish breakdown alert."""
        
        entry_price = orb_features['orb_low'] * 0.998   # Entry slightly below ORB low
        stop_loss = orb_features['orb_high'] * 1.005    # Stop above ORB high
        target_range = orb_features['orb_range']
        target = entry_price - (target_range * 1.5)    # Target based on range projection
        
        # Calculate confidence based on PCA factors
        confidence = self._calculate_confidence(orb_features, 'bearish')
        
        # Expected return (simplified for real-time)
        expected_return = ((entry_price - target) / entry_price) * 100
        
        reasoning = (f"PCA-Enhanced: Vol Ratio {orb_features['orb_volume_ratio']:.1f}x, "
                    f"Momentum {orb_features['orb_momentum']:.3f}, "
                    f"Range {orb_features['orb_range_pct']:.1f}%")
        
        return EnhancedORBAlert(
            symbol=symbol,
            timestamp=timestamp,
            alert_type="ENHANCED_BEARISH_BREAKDOWN",
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            volume_ratio=orb_features['orb_volume_ratio'],
            momentum=orb_features['orb_momentum'],
            range_pct=orb_features['orb_range_pct'],
            expected_return=expected_return,
            reasoning=reasoning,
            orb_features=orb_features
        )
    
    def _calculate_confidence(self, orb_features: Dict[str, Any], direction: str) -> float:
        """Calculate confidence score based on PCA-derived factors."""
        
        try:
            # Base confidence
            confidence = 0.5
            
            # Volume ratio contribution (higher volume = higher confidence)
            vol_ratio = orb_features['orb_volume_ratio']
            if vol_ratio > 5.0:
                confidence += 0.3
            elif vol_ratio > 3.0:
                confidence += 0.2
            elif vol_ratio > 2.5:
                confidence += 0.1
            
            # Range contribution (optimal range = higher confidence)
            range_pct = orb_features['orb_range_pct']
            if 15.0 <= range_pct <= 25.0:  # Optimal range based on PCA
                confidence += 0.15
            elif 10.0 <= range_pct <= 30.0:
                confidence += 0.1
            
            # Momentum contribution
            momentum = orb_features['orb_momentum']
            if direction == 'bullish' and momentum > 0:
                confidence += 0.1
            elif direction == 'bearish' and momentum < 0:
                confidence += 0.1
            
            # Duration contribution (longer ORB formation = more reliable)
            duration = orb_features['orb_duration_minutes']
            if duration >= 15:
                confidence += 0.05
            
            # Cap confidence at 100%
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.6  # Default confidence
    
    def _handle_enhanced_alert(self, alert: EnhancedORBAlert) -> None:
        """Handle generated enhanced ORB alert."""
        
        # Save alert
        self._save_enhanced_alert(alert)
        self.enhanced_alerts.append(alert)
        
        # Print alert
        if self.test_mode:
            print(f"[TEST MODE] {alert.alert_message}")
            self.logger.info(f"Enhanced test alert: {alert.symbol} - {alert.alert_type}")
        else:
            print(f"🎯 {alert.alert_message}")
            self.logger.info(f"Enhanced alert generated: {alert.symbol} - {alert.alert_type} - Confidence: {alert.confidence:.0%}")
    
    def _save_enhanced_alert(self, alert: EnhancedORBAlert) -> None:
        """Save enhanced alert data to historical files."""
        try:
            # Determine subdirectory based on alert type
            if "BULLISH" in alert.alert_type:
                subdir = "bullish"
            elif "BEARISH" in alert.alert_type:
                subdir = "bearish"
            else:
                subdir = ""
            
            # Save alert as JSON
            alert_filename = f"enhanced_alert_{alert.symbol}_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            
            if subdir:
                alert_filepath = self.daily_data_dir / "enhanced_alerts" / subdir / alert_filename
                # Ensure subdirectory exists
                alert_filepath.parent.mkdir(parents=True, exist_ok=True)
            else:
                alert_filepath = self.daily_data_dir / "enhanced_alerts" / alert_filename
            
            # Save alert data
            with open(alert_filepath, 'w') as f:
                json.dump(alert.to_dict(), f, indent=2)
                
            self.logger.debug(f"Saved enhanced alert data for {alert.symbol} to {alert_filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving enhanced alert data: {e}")
    
    async def start(self) -> None:
        """Start the Enhanced ORB Alert System."""
        self.logger.info("Starting Enhanced ORB Alert System...")
        
        try:
            # Validate configuration
            config_errors = config.validate()
            if config_errors:
                self.logger.error(f"Configuration errors: {config_errors}")
                return
            
            # Set start time
            et_tz = pytz.timezone('US/Eastern')
            self.start_time = datetime.now(et_tz)
            self.logger.info(f"Starting enhanced monitoring at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Get symbols
            symbols = self.alert_engine.get_monitored_symbols()
            self.logger.info(f"Enhanced monitoring {len(symbols)} symbols with PCA filters")
            self.logger.info(f"Symbols: {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                self.logger.info(f"... and {len(symbols) - 10} more symbols")
            
            # Start the underlying alert engine (this handles websocket connections)
            await self.alert_engine.start()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down enhanced system...")
        except Exception as e:
            self.logger.error(f"Error in enhanced alert engine: {e}")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the Enhanced ORB Alert System."""
        self.logger.info("Stopping Enhanced ORB Alert System...")
        
        try:
            await self.alert_engine.stop()
            
            # Save final summary
            self._save_daily_summary()
            
            self.logger.info("Enhanced ORB Alert System stopped")
        except Exception as e:
            self.logger.error(f"Error stopping enhanced system: {e}")
    
    def _save_daily_summary(self) -> None:
        """Save daily summary of enhanced alerts."""
        try:
            et_tz = pytz.timezone('US/Eastern')
            current_time = datetime.now(et_tz)
            
            summary = {
                "date": current_time.strftime("%Y-%m-%d"),
                "total_enhanced_alerts": len(self.enhanced_alerts),
                "alerts_by_type": {},
                "average_confidence": 0,
                "pca_filters_applied": {
                    "volume_ratio_threshold": self.volume_ratio_threshold,
                    "duration_threshold": self.duration_threshold,
                    "momentum_threshold": self.momentum_threshold,
                    "range_pct_min": self.range_pct_min,
                    "range_pct_max": self.range_pct_max
                },
                "alerts": [alert.to_dict() for alert in self.enhanced_alerts]
            }
            
            # Calculate statistics
            if self.enhanced_alerts:
                bullish_count = sum(1 for alert in self.enhanced_alerts if "BULLISH" in alert.alert_type)
                bearish_count = sum(1 for alert in self.enhanced_alerts if "BEARISH" in alert.alert_type)
                
                summary["alerts_by_type"]["bullish"] = bullish_count
                summary["alerts_by_type"]["bearish"] = bearish_count
                summary["average_confidence"] = sum(alert.confidence for alert in self.enhanced_alerts) / len(self.enhanced_alerts)
            
            # Save summary
            summary_file = self.daily_data_dir / "summary" / f"enhanced_alerts_summary_{current_time.strftime('%Y%m%d')}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Saved enhanced alerts daily summary: {len(self.enhanced_alerts)} alerts generated")
            
        except Exception as e:
            self.logger.error(f"Error saving daily summary: {e}")
    
    def get_statistics(self) -> dict:
        """Get enhanced system statistics."""
        engine_stats = self.alert_engine.get_stats()
        return {
            'enhanced_alerts_generated': len(self.enhanced_alerts),
            'symbols_monitored': len(self.alert_engine.get_monitored_symbols()),
            'start_time': self.start_time,
            'pca_filters_applied': {
                'volume_ratio_threshold': self.volume_ratio_threshold,
                'duration_threshold': self.duration_threshold,
                'momentum_threshold': self.momentum_threshold,
                'range_pct_min': self.range_pct_min,
                'range_pct_max': self.range_pct_max
            },
            'engine_stats': engine_stats
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced ORB Trading Alerts System with PCA Filters")
    
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
        system = EnhancedORBAlertSystem(
            symbols_file=args.symbols_file,
            test_mode=args.test
        )
        
        if args.summary:
            # Show summary and exit
            stats = system.get_statistics()
            print(f"\nEnhanced ORB Alert System Summary")
            print(f"Enhanced Alerts Generated: {stats['enhanced_alerts_generated']}")
            print(f"Symbols Monitored: {stats['symbols_monitored']}")
            print(f"PCA Filters: {stats['pca_filters_applied']}")
            return
        
        if args.test:
            print("Running Enhanced ORB Alert System in test mode - alerts will be marked as [TEST MODE]")
        
        print("🎯 Enhanced ORB Alert System with PCA-derived filters starting...")
        print(f"📊 Filters: Vol>{system.volume_ratio_threshold}x, Duration>{system.duration_threshold}min, "
              f"Momentum>{system.momentum_threshold}, Range:{system.range_pct_min}-{system.range_pct_max}%")
        
        await system.start()
        
    except Exception as e:
        logging.error(f"Failed to start Enhanced ORB Alert System: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())