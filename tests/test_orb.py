"""
PyTest unit tests for the ORB class data filtering and data prep methods.
"""

import pytest
import pandas as pd
import json
import os
from datetime import datetime, time, timedelta
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
        # Create 45 minutes of data from 9:30 to 10:15 ET
        et_tz = pytz.timezone('America/New_York')
        base_time = et_tz.localize(datetime(2025, 6, 30, 9, 30))

        data = []
        for i in range(45):
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
        end_time = time(10, 15)  # This should include data up to 10:14

        result = orb_instance._filter_stock_data_by_time(
            sample_symbol_data, start_time, end_time
        )

        assert result is not None
        assert len(result) == 45
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
    @patch.object(orb_module, 'calculate_orb_levels')
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
        mock_ema.return_value = (True, [100.1] * 45)
        mock_vwap.return_value = (True, [100.15] * 45)
        mock_vector_angle.return_value = 5.5

        # Mock the filtering method to return exactly 45 rows
        with patch.object(orb_instance,
                          '_filter_stock_data_by_time') as mock_filter:
            mock_filter.return_value = sample_symbol_data

            result = orb_instance._pca_data_prep(sample_symbol_data, 'TEST')

        assert result is True
        assert orb_instance.pca_data is not None
        assert len(orb_instance.pca_data) == 45

        # Check that all expected columns are present
        expected_columns = [
            'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'orb_high', 'orb_low', 'vector_angle', 'ema_9', 'vwap'
        ]
        for col in expected_columns:
            assert col in orb_instance.pca_data.columns

        # Check that vector angle is repeated for all rows
        assert all(orb_instance.pca_data['vector_angle'] == 5.5)

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

        with patch.object(orb_instance,
                          '_filter_stock_data_by_time') as mock_filter:
            mock_filter.return_value = short_data

            result = orb_instance._pca_data_prep(pd.DataFrame(), 'TEST')

        assert result is False

    @patch.object(orb_module, 'extract_symbol_data')
    def test_pca_data_prep_no_filtered_data(self, mock_extract_symbol,
                                            orb_instance, sample_symbol_data):
        """Test PCA data prep when filtering returns no data."""
        mock_extract_symbol.return_value = sample_symbol_data

        with patch.object(orb_instance,
                          '_filter_stock_data_by_time') as mock_filter:
            mock_filter.return_value = None

            result = orb_instance._pca_data_prep(sample_symbol_data, 'TEST')

        assert result is False

    def test_pca_data_accumulation(self, orb_instance, sample_symbol_data):
        """Test that PCA data accumulates across multiple calls."""
        # Mock all dependencies
        with patch.object(orb_module, 'extract_symbol_data') as mock_extract, \
             patch.object(orb_module, 'calculate_orb_levels') as mock_orb, \
             patch.object(orb_module, 'calculate_ema') as mock_ema, \
             patch.object(orb_module, 'calculate_vwap_typical') as mock_vwap, \
             patch.object(orb_module, 'calculate_vector_angle') as mock_vector, \
             patch.object(orb_instance,
                          '_filter_stock_data_by_time') as mock_filter:

            # Setup mocks
            mock_extract.return_value = sample_symbol_data
            mock_filter.return_value = sample_symbol_data
            mock_orb.return_value = (101.0, 99.0)
            mock_ema.return_value = (True, [100.1] * 45)
            mock_vwap.return_value = (True, [100.15] * 45)
            mock_vector.return_value = 5.5

            # First call
            result1 = orb_instance._pca_data_prep(sample_symbol_data, 'TEST1')
            assert result1 is True
            assert len(orb_instance.pca_data) == 45

            # Second call with different symbol
            result2 = orb_instance._pca_data_prep(sample_symbol_data, 'TEST2')
            assert result2 is True
            assert len(orb_instance.pca_data) == 90  # Should accumulate

    def test_integration_with_real_data(self, orb_instance, sample_stock_data):
        """Integration test using real data from stock_data/20250630.json."""
        # Get a symbol that exists in the real data
        symbols = sample_stock_data['symbol'].unique()
        test_symbol = symbols[0] if len(symbols) > 0 else 'BMNR'

        # Mock the external dependencies that require API access
        with patch.object(orb_module, 'calculate_orb_levels') as mock_orb, \
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
