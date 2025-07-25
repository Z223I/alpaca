"""
PyTest unit tests for the ORB class data filtering and data prep methods.
"""

import pytest
import pandas as pd
import json
import os
import time as time_module
from datetime import datetime, time, timedelta, date
from unittest.mock import Mock, patch, MagicMock
import pytz

# Add project root to path for imports
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import ORB class by importing the module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "orb", os.path.join(project_root, "code", "orb.py"))
orb_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orb_module)
ORB = orb_module.ORB


class TestORB:
    """Test class for the ORB class methods."""

    @pytest.fixture
    def sample_stock_data(self):
        """Fixture that provides sample stock data for testing."""
        # Load data from stock_data/20250630.json
        stock_data_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'stock_data', '20250630.json'
        )

        with open(stock_data_file, 'r') as f:
            data = json.load(f)

        # Convert to DataFrame format expected by ORB methods
        all_data = []
        for symbol, bars in data.items():
            for bar in bars:
                bar_data = bar.copy()
                bar_data['symbol'] = symbol
                all_data.append(bar_data)

        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    @pytest.fixture
    def orb_instance(self):
        """Fixture that provides a mocked ORB instance."""
        with patch.object(orb_module, 'init_alpaca_client') as mock_client:
            mock_client.return_value = Mock()
            orb = ORB()
            return orb

    @pytest.fixture
    def sample_symbol_data(self):
        """Fixture that provides sample data for a single symbol."""
        # Create 90 minutes of data from 9:30 to 11:00 ET
        et_tz = pytz.timezone('America/New_York')
        base_time = et_tz.localize(datetime(2025, 6, 30, 9, 30))

        data = []
        for i in range(90):
            # Use timedelta to add minutes properly
            timestamp = base_time + timedelta(minutes=i)
            data.append({
                'timestamp': timestamp,
                'symbol': 'TEST',
                'open': 100.0 + i * 0.1,
                'high': 100.5 + i * 0.1,
                'low': 99.5 + i * 0.1,
                'close': 100.2 + i * 0.1,
                'volume': 1000 + i * 10
            })

        return pd.DataFrame(data)

    def test_filter_stock_data_by_time_success(self, orb_instance,
                                               sample_symbol_data):
        """Test successful filtering of stock data by time range."""
        start_time = time(9, 30)
        end_time = time(10, 15)  # This includes data from 9:30 to 10:15 (inclusive)

        result = orb_instance._filter_stock_data_by_time(
            sample_symbol_data, start_time, end_time
        )

        assert result is not None
        assert len(result) == 46  # 9:30 to 10:15 inclusive = 46 minutes
        assert 'symbol' in result.columns
        assert 'timestamp' in result.columns
        # Ensure temporary columns are cleaned up
        assert 'timestamp_et' not in result.columns
        assert 'time_only' not in result.columns

    def test_filter_stock_data_by_time_empty_data(self, orb_instance):
        """Test filtering with empty DataFrame."""
        empty_df = pd.DataFrame()
        start_time = time(9, 30)
        end_time = time(10, 15)

        result = orb_instance._filter_stock_data_by_time(
            empty_df, start_time, end_time
        )

        assert result is None

    def test_filter_stock_data_by_time_none_input(self, orb_instance):
        """Test filtering with None input."""
        start_time = time(9, 30)
        end_time = time(10, 15)

        result = orb_instance._filter_stock_data_by_time(
            None, start_time, end_time
        )

        assert result is None

    def test_filter_stock_data_by_time_no_matches(self, orb_instance):
        """Test filtering when no data matches the time range."""
        # Create data outside the time range
        et_tz = pytz.timezone('America/New_York')
        base_time = et_tz.localize(datetime(2025, 6, 30, 15, 0))  # 3:00 PM

        data = []
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i)
            data.append({
                'timestamp': timestamp,
                'symbol': 'TEST',
                'open': 100.0,
                'high': 100.5,
                'low': 99.5,
                'close': 100.2,
                'volume': 1000
            })

        df = pd.DataFrame(data)
        start_time = time(9, 30)
        end_time = time(10, 15)

        result = orb_instance._filter_stock_data_by_time(
            df, start_time, end_time)

        assert result is None

    @patch.object(orb_module, 'extract_symbol_data')
    @patch('atoms.utils.calculate_orb_levels.calculate_orb_levels')
    @patch.object(orb_module, 'calculate_ema')
    @patch.object(orb_module, 'calculate_vwap_typical')
    @patch.object(orb_module, 'calculate_vector_angle')
    def test_pca_data_prep_success(self, mock_vector_angle, mock_vwap,
                                   mock_ema,
                                   mock_orb_levels, mock_extract_symbol,
                                   orb_instance, sample_symbol_data):
        """Test successful PCA data preparation."""
        # Setup mocks
        mock_extract_symbol.return_value = sample_symbol_data
        mock_orb_levels.return_value = (101.0, 99.0)
        mock_ema.return_value = (True, [100.1] * 90)
        mock_vwap.return_value = (True, [100.15] * 90)
        mock_vector_angle.return_value = 5.5

        # The method now uses head(data_samples) directly, no filtering needed
        result = orb_instance._pca_data_prep(sample_symbol_data, 'TEST')

        assert result is True
        assert orb_instance.pca_data is not None
        assert len(orb_instance.pca_data) == 90

        # Check that all expected independent feature columns are present
        expected_columns = [
            'symbol', 'timestamp', 'return_open_close', 'return_prev_close',
            'intraday_range', 'upper_wick', 'lower_wick', 'volume_ratio',
            'momentum_5', 'close_vs_orb_high', 'close_vs_orb_low',
            'high_vs_orb_high', 'low_vs_orb_low', 'close_vs_ema', 'close_vs_vwap',
            'vector_angle', 'session_progress'
        ]
        for col in expected_columns:
            assert col in orb_instance.pca_data.columns

        # Check that vector angle is repeated for all rows
        assert all(orb_instance.pca_data['vector_angle'] == 5.5)
        
        # Check that features are independent (not raw prices)
        assert 'open' not in orb_instance.pca_data.columns
        assert 'high' not in orb_instance.pca_data.columns
        assert 'low' not in orb_instance.pca_data.columns
        assert 'close' not in orb_instance.pca_data.columns

    @patch.object(orb_module, 'extract_symbol_data')
    def test_pca_data_prep_no_symbol_data(self, mock_extract_symbol,
                                          orb_instance):
        """Test PCA data prep when no symbol data is found."""
        mock_extract_symbol.return_value = None

        result = orb_instance._pca_data_prep(pd.DataFrame(), 'NONEXISTENT')

        assert result is False

    @patch.object(orb_module, 'extract_symbol_data')
    def test_pca_data_prep_wrong_data_length(self, mock_extract_symbol,
                                             orb_instance, sample_symbol_data):
        """Test PCA data prep when filtered data doesn't have 45 rows."""
        # Create data with wrong length
        short_data = sample_symbol_data.head(30)  # Only 30 rows instead of 45
        mock_extract_symbol.return_value = short_data

        # The method now uses head(data_samples) directly
        result = orb_instance._pca_data_prep(pd.DataFrame(), 'TEST')

        assert result is False

    @patch.object(orb_module, 'extract_symbol_data')
    def test_pca_data_prep_no_filtered_data(self, mock_extract_symbol,
                                            orb_instance):
        """Test PCA data prep when extract_symbol_data returns empty data."""
        # Return empty DataFrame to simulate no data scenario
        empty_df = pd.DataFrame()
        mock_extract_symbol.return_value = empty_df

        result = orb_instance._pca_data_prep(pd.DataFrame(), 'TEST')

        assert result is False

    def test_pca_data_accumulation(self, orb_instance, sample_symbol_data):
        """Test that PCA data accumulates across multiple calls."""
        # Mock all dependencies
        with patch.object(orb_module, 'extract_symbol_data') as mock_extract, \
             patch('atoms.utils.calculate_orb_levels.calculate_orb_levels') as mock_orb, \
             patch.object(orb_module, 'calculate_ema') as mock_ema, \
             patch.object(orb_module, 'calculate_vwap_typical') as mock_vwap, \
             patch.object(orb_module, 'calculate_vector_angle') as mock_vector:

            # Setup mocks
            mock_extract.return_value = sample_symbol_data
            mock_orb.return_value = (101.0, 99.0)
            mock_ema.return_value = (True, [100.1] * 90)
            mock_vwap.return_value = (True, [100.15] * 90)
            mock_vector.return_value = 5.5

            # First call
            result1 = orb_instance._pca_data_prep(sample_symbol_data, 'TEST1')
            assert result1 is True
            assert len(orb_instance.pca_data) == 90

            # Second call with different symbol
            result2 = orb_instance._pca_data_prep(sample_symbol_data, 'TEST2')
            assert result2 is True
            assert len(orb_instance.pca_data) == 180  # Should accumulate (90+90)

    def test_integration_with_real_data(self, orb_instance, sample_stock_data):
        """Integration test using real data from stock_data/20250630.json."""
        # Get a symbol that exists in the real data
        symbols = sample_stock_data['symbol'].unique()
        test_symbol = symbols[0] if len(symbols) > 0 else 'BMNR'

        # Mock the external dependencies that require API access
        with patch('atoms.utils.calculate_orb_levels.calculate_orb_levels') as mock_orb, \
             patch.object(orb_module, 'calculate_ema') as mock_ema, \
             patch.object(orb_module, 'calculate_vwap_typical') as mock_vwap, \
             patch.object(orb_module, 'calculate_vector_angle') as mock_vector:

            mock_orb.return_value = (101.0, 99.0)
            mock_ema.return_value = (True, [100.1] * 10)  # Fewer values
            mock_vwap.return_value = (True, [100.15] * 10)
            mock_vector.return_value = 5.5

            # This will likely fail the 45-row requirement, but tests the
            # integration
            result = orb_instance._pca_data_prep(sample_stock_data,
                                                 test_symbol)

            # The result depends on whether the real data has exactly 45 rows
            # in the 9:30-10:15 timeframe, which it likely doesn't
            assert isinstance(result, bool)

    # File Operations Tests
    def test_get_most_recent_csv_success(self, orb_instance, tmp_path):
        """Test successful retrieval of most recent CSV file."""
        # Create temporary CSV files with different modification times
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create CSV files with different timestamps
        old_file = data_dir / "20250625.csv"
        new_file = data_dir / "20250630.csv"
        newest_file = data_dir / "20250701.csv"
        
        old_file.write_text("symbol,data\nAAPL,test")
        new_file.write_text("symbol,data\nTSLA,test")
        newest_file.write_text("symbol,data\nMSFT,test")
        
        # Modify file times to ensure newest_file is most recent
        os.utime(old_file, (time_module.time() - 200, time_module.time() - 200))
        os.utime(new_file, (time_module.time() - 100, time_module.time() - 100))
        os.utime(newest_file, (time_module.time(), time_module.time()))
        
        # Update the data directory path
        orb_instance.data_directory = str(data_dir)
        
        result = orb_instance._get_most_recent_csv()
        
        assert result == str(newest_file)

    def test_get_most_recent_csv_no_files(self, orb_instance, tmp_path):
        """Test when no CSV files exist in directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Update the data directory path
        orb_instance.data_directory = str(data_dir)
        
        result = orb_instance._get_most_recent_csv()
        
        assert result is None

    def test_get_most_recent_csv_directory_not_exist(self, orb_instance, tmp_path):
        """Test when data directory doesn't exist."""
        # Set non-existent directory
        orb_instance.data_directory = str(tmp_path / "nonexistent")
        
        result = orb_instance._get_most_recent_csv()
        
        assert result is None

    def test_prompt_user_for_file_yes(self, orb_instance, monkeypatch):
        """Test user confirmation with 'yes' response."""
        monkeypatch.setattr('builtins.input', lambda _: 'yes')
        
        result = orb_instance._prompt_user_for_file("test.csv")
        
        assert result is True

    def test_prompt_user_for_file_y(self, orb_instance, monkeypatch):
        """Test user confirmation with 'y' response."""
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        
        result = orb_instance._prompt_user_for_file("test.csv")
        
        assert result is True

    def test_prompt_user_for_file_empty(self, orb_instance, monkeypatch):
        """Test user confirmation with empty response (default yes)."""
        monkeypatch.setattr('builtins.input', lambda _: '')
        
        result = orb_instance._prompt_user_for_file("test.csv")
        
        assert result is True

    def test_prompt_user_for_file_no(self, orb_instance, monkeypatch):
        """Test user confirmation with 'no' response."""
        monkeypatch.setattr('builtins.input', lambda _: 'no')
        
        result = orb_instance._prompt_user_for_file("test.csv")
        
        assert result is False

    def test_prompt_user_for_file_eof_error(self, orb_instance, monkeypatch):
        """Test user prompt with EOF error (Ctrl+D)."""
        def mock_input(_):
            raise EOFError()
        
        monkeypatch.setattr('builtins.input', mock_input)
        
        result = orb_instance._prompt_user_for_file("test.csv")
        
        assert result is False

    def test_prompt_user_for_file_keyboard_interrupt(self, orb_instance, monkeypatch):
        """Test user prompt with keyboard interrupt (Ctrl+C)."""
        def mock_input(_):
            raise KeyboardInterrupt()
        
        monkeypatch.setattr('builtins.input', mock_input)
        
        result = orb_instance._prompt_user_for_file("test.csv")
        
        assert result is False

    def test_save_market_data_success(self, orb_instance, tmp_path):
        """Test successful saving of market data to JSON."""
        # Setup mock market data
        mock_bar = Mock()
        mock_bar.t = datetime(2025, 6, 30, 10, 0, tzinfo=pytz.UTC)
        mock_bar.o = 150.0
        mock_bar.h = 155.0
        mock_bar.l = 149.0
        mock_bar.c = 152.0
        mock_bar.v = 1000
        
        orb_instance.market_data = {
            'AAPL': [mock_bar],
            'TSLA': [mock_bar]
        }
        orb_instance.current_file = "test_20250630.csv"
        
        # Change working directory to tmp_path
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = orb_instance._save_market_data()
            
            assert result is True
            
            # Check that JSON file was created
            json_file = tmp_path / "stock_data" / "test_20250630.json"
            assert json_file.exists()
            
            # Verify JSON content
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            assert 'AAPL' in data
            assert 'TSLA' in data
            assert len(data['AAPL']) == 1
            assert data['AAPL'][0]['open'] == 150.0
            assert data['AAPL'][0]['close'] == 152.0
            
        finally:
            os.chdir(original_cwd)

    def test_save_market_data_no_data(self, orb_instance):
        """Test saving when no market data exists."""
        orb_instance.market_data = None
        orb_instance.current_file = "test.csv"
        
        result = orb_instance._save_market_data()
        
        assert result is False

    def test_save_market_data_no_current_file(self, orb_instance):
        """Test saving when no current file is set."""
        orb_instance.market_data = {'AAPL': []}
        orb_instance.current_file = None
        
        result = orb_instance._save_market_data()
        
        assert result is False

    def test_load_market_dataframe_success(self, orb_instance, tmp_path):
        """Test successful loading of market data from JSON."""
        # Create test JSON file
        stock_data_dir = tmp_path / "stock_data"
        stock_data_dir.mkdir()
        
        test_data = {
            'AAPL': [
                {
                    'timestamp': '2025-06-30T10:00:00+00:00',
                    'open': 150.0,
                    'high': 155.0,
                    'low': 149.0,
                    'close': 152.0,
                    'volume': 1000
                }
            ],
            'TSLA': [
                {
                    'timestamp': '2025-06-30T10:00:00+00:00',
                    'open': 250.0,
                    'high': 255.0,
                    'low': 249.0,
                    'close': 252.0,
                    'volume': 2000
                }
            ]
        }
        
        json_file = stock_data_dir / "test_20250630.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        # Setup instance
        orb_instance.current_file = "test_20250630.csv"
        
        # Change working directory to tmp_path
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = orb_instance._load_market_dataframe()
            
            assert result is True
            assert hasattr(orb_instance, 'market_df')
            assert orb_instance.market_df is not None
            assert len(orb_instance.market_df) == 2
            assert 'symbol' in orb_instance.market_df.columns
            assert 'AAPL' in orb_instance.market_df['symbol'].values
            assert 'TSLA' in orb_instance.market_df['symbol'].values
            
        finally:
            os.chdir(original_cwd)

    def test_load_market_dataframe_no_current_file(self, orb_instance):
        """Test loading when no current file is set."""
        orb_instance.current_file = None
        
        result = orb_instance._load_market_dataframe()
        
        assert result is False

    def test_load_market_dataframe_file_not_found(self, orb_instance, tmp_path):
        """Test loading when JSON file doesn't exist."""
        orb_instance.current_file = "nonexistent_20250630.csv"
        
        # Change working directory to tmp_path
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = orb_instance._load_market_dataframe()
            
            assert result is False
            
        finally:
            os.chdir(original_cwd)

    def test_load_market_dataframe_empty_data(self, orb_instance, tmp_path):
        """Test loading when JSON file contains no market data."""
        # Create empty JSON file
        stock_data_dir = tmp_path / "stock_data"
        stock_data_dir.mkdir()
        
        json_file = stock_data_dir / "test_20250630.json"
        with open(json_file, 'w') as f:
            json.dump({}, f)
        
        orb_instance.current_file = "test_20250630.csv"
        
        # Change working directory to tmp_path
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = orb_instance._load_market_dataframe()
            
            assert result is False
            
        finally:
            os.chdir(original_cwd)

    @patch.object(orb_module, 'read_csv')
    def test_load_and_process_csv_data_success(self, mock_read_csv, orb_instance, tmp_path, monkeypatch):
        """Test successful loading and processing of CSV data."""
        # Setup mock data
        mock_read_csv.return_value = [
            {'symbol': 'AAPL', 'data': 'test1'},
            {'symbol': 'TSLA', 'data': 'test2'}
        ]
        
        # Create test CSV file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "20250630.csv"
        test_file.write_text("symbol,data\nAAPL,test1\nTSLA,test2")
        
        # Setup instance
        orb_instance.data_directory = str(data_dir)
        
        # Mock user input to confirm file selection
        monkeypatch.setattr('builtins.input', lambda _: 'yes')
        
        result = orb_instance._load_and_process_csv_data()
        
        assert result is True
        assert orb_instance.csv_data is not None
        assert len(orb_instance.csv_data) == 2
        assert orb_instance.current_file == str(test_file)
        assert orb_instance.csv_date == date(2025, 6, 30)

    def test_load_and_process_csv_data_no_files(self, orb_instance, tmp_path):
        """Test when no CSV files are found."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        orb_instance.data_directory = str(data_dir)
        
        result = orb_instance._load_and_process_csv_data()
        
        assert result is False

    def test_load_and_process_csv_data_user_cancels(self, orb_instance, tmp_path, monkeypatch):
        """Test when user cancels file selection."""
        # Create test CSV file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "20250630.csv"
        test_file.write_text("symbol,data\nAAPL,test")
        
        orb_instance.data_directory = str(data_dir)
        
        # Mock user input to cancel file selection
        monkeypatch.setattr('builtins.input', lambda _: 'no')
        
        result = orb_instance._load_and_process_csv_data()
        
        assert result is False

    @patch.object(orb_module, 'read_csv')
    def test_load_and_process_csv_data_empty_file(self, mock_read_csv, orb_instance, tmp_path, monkeypatch):
        """Test when CSV file is empty."""
        mock_read_csv.return_value = []
        
        # Create test CSV file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "20250630.csv"
        test_file.write_text("symbol,data")
        
        orb_instance.data_directory = str(data_dir)
        monkeypatch.setattr('builtins.input', lambda _: 'yes')
        
        result = orb_instance._load_and_process_csv_data()
        
        assert result is False

    @patch.object(orb_module, 'read_csv')
    def test_load_and_process_csv_data_invalid_filename(self, mock_read_csv, orb_instance, tmp_path, monkeypatch):
        """Test with invalid filename format that can't be parsed for date."""
        mock_read_csv.return_value = [{'symbol': 'AAPL', 'data': 'test'}]
        
        # Create test CSV file with invalid date format
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "invalid_filename.csv"
        test_file.write_text("symbol,data\nAAPL,test")
        
        orb_instance.data_directory = str(data_dir)
        monkeypatch.setattr('builtins.input', lambda _: 'yes')
        
        result = orb_instance._load_and_process_csv_data()
        
        assert result is True
        assert orb_instance.csv_date is None  # Date parsing should fail gracefully

    def test_standardize_pca_data_success(self, orb_instance):
        """Test successful PCA data standardization."""
        # Create mock PCA data with independent features
        mock_pca_data = pd.DataFrame({
            'symbol': ['TEST'] * 5,
            'timestamp': pd.date_range('2025-01-01', periods=5, freq='min'),
            'return_open_close': [0.001, 0.002, 0.001, 0.003, 0.0],
            'return_prev_close': [0.0, 0.001, 0.002, 0.001, 0.002],
            'intraday_range': [0.01, 0.012, 0.008, 0.015, 0.011],
            'upper_wick': [0.002, 0.003, 0.001, 0.004, 0.002],
            'lower_wick': [0.003, 0.002, 0.004, 0.001, 0.003],
            'volume_ratio': [1.0, 1.1, 0.9, 1.2, 1.05],
            'momentum_5': [0.0, 0.0, 0.0, 0.0, 0.005],
            'close_vs_orb_high': [-0.5, 0.0, 0.2, 0.8, 1.2],
            'close_vs_orb_low': [0.5, 1.0, 1.2, 1.8, 2.2],
            'vector_angle': [5.5] * 5,
            'session_progress': [0.0, 0.25, 0.5, 0.75, 1.0]
        })
        
        orb_instance.pca_data = mock_pca_data
        
        # Mock sklearn import and StandardScaler
        import numpy as np
        mock_scaler_class = Mock()
        mock_scaler_instance = Mock()
        
        # Return 5 rows x 11 columns for independent features (excluding symbol/timestamp)
        n_features = 11
        standardized_features = np.array([
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        ])
        
        mock_scaler_instance.fit_transform.return_value = standardized_features
        mock_scaler_class.return_value = mock_scaler_instance
        
        # Mock the sklearn module structure
        sklearn_mock = Mock()
        sklearn_mock.preprocessing.StandardScaler = mock_scaler_class
        
        with patch.dict('sys.modules', {'sklearn': sklearn_mock, 'sklearn.preprocessing': sklearn_mock.preprocessing}):
            result = orb_instance._standardize_pca_data()
            
        assert result is True
        assert hasattr(orb_instance, 'pca_data_standardized')
        assert orb_instance.pca_data_standardized is not None
        
        # Check that symbol and timestamp columns are removed
        assert 'symbol' not in orb_instance.pca_data_standardized.columns
        assert 'timestamp' not in orb_instance.pca_data_standardized.columns
        
        # Check that independent features are present
        assert 'return_open_close' in orb_instance.pca_data_standardized.columns
        assert 'intraday_range' in orb_instance.pca_data_standardized.columns
        assert 'volume_ratio' in orb_instance.pca_data_standardized.columns
        
        # Verify shape
        assert orb_instance.pca_data_standardized.shape == (5, n_features)

    def test_standardize_pca_data_no_data(self, orb_instance):
        """Test standardization when no PCA data exists."""
        orb_instance.pca_data = None
        
        result = orb_instance._standardize_pca_data()
        
        assert result is False

    def test_standardize_pca_data_missing_sklearn(self, orb_instance):
        """Test standardization when scikit-learn is not available."""
        # Create mock PCA data
        mock_pca_data = pd.DataFrame({
            'symbol': ['TEST'],
            'open': [100.0],
            'volume': [1000]
        })
        orb_instance.pca_data = mock_pca_data
        
        # Mock the import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == 'sklearn.preprocessing':
                raise ImportError("No module named 'sklearn'")
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = orb_instance._standardize_pca_data()
            
        assert result is False

    def test_perform_pca_computation_success(self, orb_instance):
        """Test successful PCA computation."""
        import numpy as np
        
        # Create mock standardized PCA data
        mock_standardized_data = pd.DataFrame({
            'open': [0.0, 0.1, 0.2, 0.3, 0.4],
            'high': [0.1, 0.2, 0.3, 0.4, 0.5],
            'low': [-0.1, 0.0, 0.1, 0.2, 0.3],
            'close': [0.05, 0.15, 0.25, 0.35, 0.45],
            'volume_standardized': [-1.0, -0.5, 0.0, 0.5, 1.0],
            'vector_angle_standardized': [0.2, 0.3, 0.4, 0.5, 0.6]
        })
        
        orb_instance.pca_data_standardized = mock_standardized_data
        
        # Mock sklearn PCA
        mock_pca_class = Mock()
        mock_pca_instance = Mock()
        
        # Configure PCA mock
        mock_pca_instance.components_ = np.array([
            [0.5, 0.3, 0.2, 0.4, 0.1, 0.3],
            [0.2, 0.4, 0.3, 0.1, 0.5, 0.2]
        ])
        mock_pca_instance.explained_variance_ = np.array([2.5, 1.8])
        mock_pca_instance.explained_variance_ratio_ = np.array([0.6, 0.4])
        mock_pca_instance.fit_transform.return_value = np.array([
            [1.2, 0.8], [1.0, 0.6], [0.8, 0.4], [0.6, 0.2], [0.4, 0.0]
        ])
        
        mock_pca_class.return_value = mock_pca_instance
        
        # Mock the sklearn module structure
        sklearn_mock = Mock()
        sklearn_mock.decomposition.PCA = mock_pca_class
        
        with patch.dict('sys.modules', {
            'sklearn': sklearn_mock, 
            'sklearn.decomposition': sklearn_mock.decomposition
        }):
            result = orb_instance._perform_pca_computation()
            
        assert result is True
        assert hasattr(orb_instance, 'pca_components')
        assert hasattr(orb_instance, 'pca_explained_variance')
        assert hasattr(orb_instance, 'pca_explained_variance_ratio')
        assert hasattr(orb_instance, 'pca_transformed_data')
        assert hasattr(orb_instance, 'pca_feature_names')
        
        # Check that PCA results are stored correctly
        assert orb_instance.pca_components.shape == (2, 6)
        assert len(orb_instance.pca_explained_variance) == 2
        assert len(orb_instance.pca_explained_variance_ratio) == 2
        assert orb_instance.pca_transformed_data.shape == (5, 2)

    def test_perform_pca_computation_no_data(self, orb_instance):
        """Test PCA computation when no standardized data exists."""
        orb_instance.pca_data_standardized = None
        
        result = orb_instance._perform_pca_computation()
        
        assert result is False

    def test_perform_pca_computation_empty_data(self, orb_instance):
        """Test PCA computation when standardized data is empty after NaN removal."""
        import numpy as np
        # Create DataFrame with all NaN values
        mock_standardized_data = pd.DataFrame({
            'open': [np.nan, np.nan, np.nan],
            'high': [np.nan, np.nan, np.nan],
            'volume_standardized': [np.nan, np.nan, np.nan]
        })
        
        orb_instance.pca_data_standardized = mock_standardized_data
        
        result = orb_instance._perform_pca_computation()
        
        assert result is False

    def test_perform_pca_computation_missing_sklearn(self, orb_instance):
        """Test PCA computation when scikit-learn is not available."""
        # Create mock standardized data
        mock_standardized_data = pd.DataFrame({
            'open': [0.0, 0.1, 0.2],
            'high': [0.1, 0.2, 0.3]
        })
        orb_instance.pca_data_standardized = mock_standardized_data
        
        # Mock the import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == 'sklearn.decomposition':
                raise ImportError("No module named 'sklearn'")
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = orb_instance._perform_pca_computation()
            
        assert result is False

    def test_perform_pca_computation_with_nan_values(self, orb_instance):
        """Test PCA computation with some NaN values that get removed."""
        import numpy as np
        
        # Create mock standardized data with some NaN values
        mock_standardized_data = pd.DataFrame({
            'open': [0.0, 0.1, np.nan, 0.3, 0.4],
            'high': [0.1, 0.2, 0.3, np.nan, 0.5],
            'low': [-0.1, 0.0, 0.1, 0.2, 0.3],
            'volume_standardized': [-1.0, -0.5, 0.0, 0.5, 1.0]
        })
        
        orb_instance.pca_data_standardized = mock_standardized_data
        
        # Mock sklearn PCA
        mock_pca_class = Mock()
        mock_pca_instance = Mock()
        
        # Configure PCA mock for remaining 3 valid rows
        mock_pca_instance.components_ = np.array([[0.5, 0.3, 0.2, 0.4]])
        mock_pca_instance.explained_variance_ = np.array([1.5])
        mock_pca_instance.explained_variance_ratio_ = np.array([0.8])
        mock_pca_instance.fit_transform.return_value = np.array([[1.0], [0.5], [0.2]])
        
        mock_pca_class.return_value = mock_pca_instance
        
        # Mock the sklearn module structure
        sklearn_mock = Mock()
        sklearn_mock.decomposition.PCA = mock_pca_class
        
        with patch.dict('sys.modules', {
            'sklearn': sklearn_mock, 
            'sklearn.decomposition': sklearn_mock.decomposition
        }):
            result = orb_instance._perform_pca_computation()
            
        assert result is True
        # Should have 3 rows after NaN removal
        assert orb_instance.pca_transformed_data.shape == (3, 1)
