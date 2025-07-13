"""
PyTest unit tests for ORB alerts historical data storage functionality.

This test suite validates the historical data storage system's ability to:
- Create proper directory structures for daily data storage
- Save market data in CSV format with correct columns and data
- Save alert data in JSON format with complete metadata
- Handle periodic data saving mechanisms
- Manage data storage errors gracefully
- Organize data by date and category (market_data, alerts, summary)
"""

import pytest
import pandas as pd
import json
import os
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

# Add project root to path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the ORB alerts module
import importlib.util
spec = importlib.util.spec_from_file_location("orb_alerts", os.path.join(project_root, "code", "orb_alerts.py"))
orb_alerts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orb_alerts_module)
ORBAlertSystem = orb_alerts_module.ORBAlertSystem
from atoms.alerts.alert_formatter import ORBAlert, AlertPriority
from atoms.alerts.breakout_detector import BreakoutType
from atoms.websocket.data_buffer import DataBuffer
from atoms.websocket.alpaca_stream import MarketData
from molecules.orb_alert_engine import ORBAlertEngine


class TestORBAlertsHistoricalData:
    """Test historical data storage functionality in ORB alerts system."""
    
    @pytest.fixture
    def temp_historical_dir(self):
        """Create temporary directory for historical data testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_alert_engine(self):
        """Create mocked ORB alert engine."""
        engine = Mock(spec=ORBAlertEngine)
        engine.get_monitored_symbols = Mock(return_value=['AAPL', 'TSLA', 'BMNR'])
        engine.data_buffer = Mock(spec=DataBuffer)
        engine.add_alert_callback = Mock()
        engine.start = AsyncMock()
        engine.stop = AsyncMock()
        engine.get_stats = Mock(return_value=Mock(
            total_alerts_generated=10,
            symbols_monitored=3
        ))
        engine.get_daily_summary = Mock(return_value={
            'date': '2025-06-30',
            'total_alerts': 5,
            'avg_confidence': 0.75,
            'max_confidence': 0.95,
            'priority_breakdown': {'HIGH': 2, 'MEDIUM': 3}
        })
        return engine
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return [
            MarketData(
                symbol='AAPL',
                timestamp=datetime(2025, 6, 30, 10, 0, 0),
                price=150.00,
                volume=1000000,
                high=150.50,
                low=149.25,
                close=150.00,
                trade_count=500,
                vwap=149.75
            ),
            MarketData(
                symbol='AAPL',
                timestamp=datetime(2025, 6, 30, 10, 1, 0),
                price=150.25,
                volume=1500000,
                high=150.75,
                low=149.50,
                close=150.25,
                trade_count=750,
                vwap=150.00
            ),
            MarketData(
                symbol='TSLA',
                timestamp=datetime(2025, 6, 30, 10, 0, 0),
                price=249.00,
                volume=2000000,
                high=250.50,
                low=248.25,
                close=249.00,
                trade_count=1000,
                vwap=249.25
            )
        ]
    
    @pytest.fixture
    def sample_orb_alert(self):
        """Create sample ORB alert for testing."""
        return ORBAlert(
            symbol='AAPL',
            timestamp=datetime(2025, 6, 30, 10, 30, 0),
            current_price=151.50,
            orb_high=150.00,
            orb_low=148.00,
            orb_range=2.00,
            orb_midpoint=149.00,
            breakout_type=BreakoutType.BULLISH_BREAKOUT,
            breakout_percentage=1.0,
            volume_ratio=2.5,
            confidence_score=0.85,
            priority=AlertPriority.HIGH,
            confidence_level='VERY_HIGH',
            recommended_stop_loss=140.13,  # 7.5% below current price
            recommended_take_profit=157.56,  # 4% above current price
            alert_message='Test alert message'
        )
    
    @pytest.fixture
    def orb_alert_system(self, temp_historical_dir, mock_alert_engine):
        """Create ORB alert system with mocked components and temp directory."""
        with patch('molecules.orb_alert_engine.ORBAlertEngine', return_value=mock_alert_engine):
            with patch.object(ORBAlertSystem, '__init__', lambda self, **kwargs: None):
                system = ORBAlertSystem()
                system.alert_engine = mock_alert_engine
                system.test_mode = True
                system.historical_data_dir = Path(temp_historical_dir)
                system.logger = Mock()
                system.start_time = datetime.now()
                system.last_data_save = None
                system.data_save_interval = timedelta(minutes=5)
                
                # Setup data storage manually
                system._setup_data_storage()
                
                return system
    
    def test_historical_data_directory_creation(self, orb_alert_system):
        """Test creation of historical data directory structure."""
        # Verify main historical data directory exists
        assert orb_alert_system.historical_data_dir.exists()
        
        # Verify daily directory exists
        today = datetime.now().strftime("%Y-%m-%d")
        daily_dir = orb_alert_system.historical_data_dir / today
        assert daily_dir.exists()
        
        # Verify subdirectories exist
        assert (daily_dir / "market_data").exists()
        assert (daily_dir / "alerts").exists()
        assert (daily_dir / "summary").exists()
    
    def test_market_data_csv_saving(self, orb_alert_system, sample_market_data):
        """Test saving market data to CSV format."""
        # Mock data buffer to return sample data
        symbol_data = pd.DataFrame([
            {
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'price': data.price,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume,
                'vwap': data.vwap,
                'trade_count': data.trade_count
            }
            for data in sample_market_data if data.symbol == 'AAPL'
        ])
        
        orb_alert_system.alert_engine.data_buffer.get_symbol_data = Mock(return_value=symbol_data)
        
        # Save historical data
        orb_alert_system._save_historical_data()
        
        # Verify CSV files were created
        market_data_dir = orb_alert_system.daily_data_dir / "market_data"
        csv_files = list(market_data_dir.glob("AAPL_*.csv"))
        assert len(csv_files) > 0
        
        # Verify CSV content
        csv_file = csv_files[0]
        saved_data = pd.read_csv(csv_file)
        
        # Check columns
        expected_columns = ['timestamp', 'symbol', 'price', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
        assert all(col in saved_data.columns for col in expected_columns)
        
        # Check data content
        assert len(saved_data) == 2  # Two AAPL records
        assert saved_data['symbol'].iloc[0] == 'AAPL'
        assert saved_data['high'].iloc[0] == 150.50
        assert saved_data['volume'].iloc[0] == 1000000
    
    def test_alert_json_saving(self, orb_alert_system, sample_orb_alert):
        """Test saving alert data to JSON format."""
        # Save alert data
        orb_alert_system._save_alert_data(sample_orb_alert)
        
        # Verify JSON file was created - bullish breakouts go to bullish subdirectory
        alerts_dir = orb_alert_system.daily_data_dir / "alerts"
        bullish_dir = alerts_dir / "bullish"
        json_files = list(bullish_dir.glob("alert_AAPL_*.json"))
        assert len(json_files) > 0
        
        # Verify JSON content
        json_file = json_files[0]
        with open(json_file, 'r') as f:
            saved_alert = json.load(f)
        
        # Check key fields
        assert saved_alert['symbol'] == 'AAPL'
        assert saved_alert['current_price'] == 151.50
        assert saved_alert['breakout_type'] == 'bullish_breakout'
        assert saved_alert['priority'] == 'HIGH'
        assert saved_alert['confidence_score'] == 0.85
        assert abs(saved_alert['recommended_stop_loss'] - 140.13) < 0.01
        assert abs(saved_alert['recommended_take_profit'] - 157.56) < 0.01
        assert 'timestamp' in saved_alert
        assert 'alert_message' in saved_alert
    
    def test_metadata_summary_saving(self, orb_alert_system, sample_market_data):
        """Test saving metadata summary files."""
        # Mock data buffer for multiple symbols
        orb_alert_system.alert_engine.data_buffer.get_symbol_data = Mock(
            side_effect=lambda symbol: pd.DataFrame([
                {
                    'timestamp': data.timestamp,
                    'symbol': data.symbol,
                    'price': data.price,
                    'high': data.high,
                    'low': data.low,
                    'close': data.close,
                    'volume': data.volume,
                    'vwap': data.vwap,
                    'trade_count': data.trade_count
                }
                for data in sample_market_data if data.symbol == symbol
            ]) if symbol in ['AAPL', 'TSLA'] else pd.DataFrame()
        )
        
        # Save historical data
        orb_alert_system._save_historical_data()
        
        # Verify metadata file was created
        summary_dir = orb_alert_system.daily_data_dir / "summary"
        metadata_files = list(summary_dir.glob("save_metadata_*.json"))
        assert len(metadata_files) > 0
        
        # Verify metadata content
        metadata_file = metadata_files[0]
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check metadata fields
        assert 'timestamp' in metadata
        assert metadata['symbols_count'] == 3  # AAPL, TSLA, BMNR
        assert metadata['symbols'] == ['AAPL', 'TSLA', 'BMNR']
        assert metadata['save_interval_minutes'] == 5.0
        assert metadata['format'] == 'CSV'
        assert 'total_records_saved' in metadata
    
    def test_periodic_data_save_timing(self, orb_alert_system):
        """Test periodic data save timing logic."""
        import pytz
        
        # Test initial state - should save data
        assert orb_alert_system._should_save_data() == True
        
        # Use Eastern Time for consistency
        et_tz = pytz.timezone('US/Eastern')
        current_time = datetime.now(et_tz)
        
        # Set last save time to recent (2 minutes ago)
        orb_alert_system.last_data_save = current_time - timedelta(minutes=2)
        assert orb_alert_system._should_save_data() == False
        
        # Set last save time to old enough (6 minutes ago)
        orb_alert_system.last_data_save = current_time - timedelta(minutes=6)
        assert orb_alert_system._should_save_data() == True
    
    def test_data_save_interval_configuration(self, orb_alert_system):
        """Test data save interval configuration."""
        # Verify default interval is 5 minutes
        assert orb_alert_system.data_save_interval == timedelta(minutes=5)
        
        # Test interval can be modified
        orb_alert_system.data_save_interval = timedelta(minutes=10)
        assert orb_alert_system.data_save_interval.total_seconds() == 600
    
    def test_error_handling_in_data_saving(self, orb_alert_system, sample_orb_alert):
        """Test error handling during data saving operations."""
        # Test market data saving with exception
        orb_alert_system.alert_engine.data_buffer.get_symbol_data = Mock(
            side_effect=Exception("Database connection failed")
        )
        
        # Should not raise exception, but log error
        orb_alert_system._save_historical_data()
        orb_alert_system.logger.error.assert_called()
        
        # Test alert data saving with invalid directory
        invalid_system = orb_alert_system
        invalid_system.daily_data_dir = Path("/invalid/path/that/does/not/exist")
        
        # Should not raise exception, but log error
        invalid_system._save_alert_data(sample_orb_alert)
        invalid_system.logger.error.assert_called()
    
    def test_empty_data_handling(self, orb_alert_system):
        """Test handling of empty data scenarios."""
        # Mock empty data buffer
        orb_alert_system.alert_engine.data_buffer.get_symbol_data = Mock(
            return_value=pd.DataFrame()  # Empty DataFrame
        )
        
        # Should handle empty data gracefully
        orb_alert_system._save_historical_data()
        
        # Verify no CSV files were created for empty data
        market_data_dir = orb_alert_system.daily_data_dir / "market_data"
        csv_files = list(market_data_dir.glob("*.csv"))
        assert len(csv_files) == 0
        
        # But metadata should still be saved
        summary_dir = orb_alert_system.daily_data_dir / "summary"
        metadata_files = list(summary_dir.glob("save_metadata_*.json"))
        assert len(metadata_files) > 0
    
    def test_multiple_symbols_data_saving(self, orb_alert_system, sample_market_data):
        """Test saving data for multiple symbols simultaneously."""
        # Create data for multiple symbols
        def mock_get_symbol_data(symbol):
            symbol_data = [data for data in sample_market_data if data.symbol == symbol]
            if symbol_data:
                return pd.DataFrame([
                    {
                        'timestamp': data.timestamp,
                        'symbol': data.symbol,
                        'high': data.high,
                        'low': data.low,
                        'close': data.close,
                        'volume': data.volume,
                        'vwap': data.vwap,
                        'trade_count': data.trade_count
                    }
                    for data in symbol_data
                ])
            return pd.DataFrame()
        
        orb_alert_system.alert_engine.data_buffer.get_symbol_data = Mock(
            side_effect=mock_get_symbol_data
        )
        
        # Save data for all symbols
        orb_alert_system._save_historical_data()
        
        # Verify separate CSV files for each symbol
        market_data_dir = orb_alert_system.daily_data_dir / "market_data"
        aapl_files = list(market_data_dir.glob("AAPL_*.csv"))
        tsla_files = list(market_data_dir.glob("TSLA_*.csv"))
        
        assert len(aapl_files) > 0
        assert len(tsla_files) > 0
        
        # Verify AAPL file contains only AAPL data
        aapl_data = pd.read_csv(aapl_files[0])
        assert all(aapl_data['symbol'] == 'AAPL')
        
        # Verify TSLA file contains only TSLA data
        tsla_data = pd.read_csv(tsla_files[0])
        assert all(tsla_data['symbol'] == 'TSLA')
    
    def test_daily_directory_date_handling(self, temp_historical_dir, mock_alert_engine):
        """Test proper date handling for daily directories."""
        # Test that the current date is used for directory creation
        with patch('molecules.orb_alert_engine.ORBAlertEngine', return_value=mock_alert_engine):
            with patch.object(ORBAlertSystem, '__init__', lambda self, **kwargs: None):
                system = ORBAlertSystem()
                system.alert_engine = mock_alert_engine
                system.historical_data_dir = Path(temp_historical_dir)
                system.logger = Mock()
                
                # Setup data storage
                system._setup_data_storage()
                
                # Verify that today's date directory was created
                today = datetime.now().strftime("%Y-%m-%d")
                expected_date_dir = Path(temp_historical_dir) / today
                assert expected_date_dir.exists()
                assert system.daily_data_dir == expected_date_dir
                
                # Verify subdirectories exist
                assert (expected_date_dir / "market_data").exists()
                assert (expected_date_dir / "alerts").exists()
                assert (expected_date_dir / "summary").exists()
    
    @pytest.mark.asyncio
    async def test_periodic_data_save_task_integration(self, orb_alert_system):
        """Test the periodic data save task runs correctly."""
        # Mock the should_save_data method to trigger saving
        orb_alert_system._should_save_data = Mock(return_value=True)
        orb_alert_system._save_historical_data = Mock()
        
        # Mock asyncio.sleep to avoid waiting
        original_sleep = asyncio.sleep
        sleep_call_count = 0
        
        async def mock_sleep(seconds):
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count >= 2:  # After 2 sleep calls, we've completed one cycle
                raise asyncio.CancelledError()
            await original_sleep(0.01)  # Very short actual sleep
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            task = asyncio.create_task(orb_alert_system._periodic_data_save())
            
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Verify save method was called when conditions were met
        assert orb_alert_system._save_historical_data.call_count >= 1
    
    def test_alert_callback_integration(self, orb_alert_system, sample_orb_alert):
        """Test alert callback properly triggers data saving."""
        # Test the alert callback handler
        orb_alert_system._handle_alert(sample_orb_alert)
        
        # Verify alert was saved to historical data - bullish breakouts go to bullish subdirectory
        alerts_dir = orb_alert_system.daily_data_dir / "alerts"
        bullish_dir = alerts_dir / "bullish"
        json_files = list(bullish_dir.glob("alert_AAPL_*.json"))
        assert len(json_files) > 0
    
    def test_data_storage_file_naming_convention(self, orb_alert_system, sample_market_data, sample_orb_alert):
        """Test proper file naming conventions for saved data."""
        # Test market data file naming
        symbol_data = pd.DataFrame([
            {
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'price': data.price,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume,
                'vwap': data.vwap,
                'trade_count': data.trade_count
            }
            for data in sample_market_data if data.symbol == 'AAPL'
        ])
        
        orb_alert_system.alert_engine.data_buffer.get_symbol_data = Mock(return_value=symbol_data)
        
        # Save data and check file naming
        orb_alert_system._save_historical_data()
        
        market_data_dir = orb_alert_system.daily_data_dir / "market_data"
        csv_files = list(market_data_dir.glob("AAPL_*.csv"))
        
        # Verify file name format: SYMBOL_YYYYMMDD_HHMMSS.csv
        assert len(csv_files) > 0
        filename = csv_files[0].name
        assert filename.startswith("AAPL_")
        assert filename.endswith(".csv")
        assert len(filename.split("_")[1]) == 8  # YYYYMMDD
        assert len(filename.split("_")[2].split(".")[0]) == 6  # HHMMSS
        
        # Test alert file naming
        orb_alert_system._save_alert_data(sample_orb_alert)
        
        alerts_dir = orb_alert_system.daily_data_dir / "alerts"
        bullish_dir = alerts_dir / "bullish"
        json_files = list(bullish_dir.glob("alert_AAPL_*.json"))
        
        # Verify alert file name format: alert_SYMBOL_YYYYMMDD_HHMMSS.json
        assert len(json_files) > 0
        alert_filename = json_files[0].name
        assert alert_filename.startswith("alert_AAPL_")
        assert alert_filename.endswith(".json")