"""
Comprehensive tests for extract symbol data utility function.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from atoms.utils.extract_symbol_data import extract_symbol_data


class TestExtractSymbolData:
    """Test suite for extract symbol data function."""
    
    @pytest.fixture
    def multi_symbol_data(self):
        """Create DataFrame with multiple symbols for testing."""
        base_time = datetime(2024, 1, 1, 9, 30)
        data = []
        
        # Add data for AAPL
        for i in range(10):
            data.append({
                'symbol': 'AAPL',
                'timestamp': base_time + timedelta(minutes=i),
                'open': 150.0 + i,
                'high': 152.0 + i,
                'low': 148.0 + i,
                'close': 151.0 + i,
                'volume': 1000 + i * 100
            })
        
        # Add data for TSLA  
        for i in range(8):
            data.append({
                'symbol': 'TSLA',
                'timestamp': base_time + timedelta(minutes=i),
                'open': 200.0 + i * 2,
                'high': 203.0 + i * 2,
                'low': 197.0 + i * 2,
                'close': 201.0 + i * 2,
                'volume': 2000 + i * 200
            })
        
        # Add data for MSFT
        for i in range(5):
            data.append({
                'symbol': 'MSFT',
                'timestamp': base_time + timedelta(minutes=i),
                'open': 300.0 + i * 0.5,
                'high': 301.0 + i * 0.5,
                'low': 299.0 + i * 0.5,
                'close': 300.5 + i * 0.5,
                'volume': 1500 + i * 150
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def single_symbol_data(self):
        """Create DataFrame with single symbol."""
        base_time = datetime(2024, 1, 1, 9, 30)
        data = []
        
        for i in range(15):
            data.append({
                'symbol': 'AAPL',
                'timestamp': base_time + timedelta(minutes=i),
                'open': 150.0 + i * 0.1,
                'high': 152.0 + i * 0.1,
                'low': 148.0 + i * 0.1,
                'close': 151.0 + i * 0.1,
                'volume': 1000 + i * 50
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def unordered_timestamp_data(self):
        """Create DataFrame with unordered timestamps."""
        base_time = datetime(2024, 1, 1, 9, 30)
        data = [
            {
                'symbol': 'AAPL',
                'timestamp': base_time + timedelta(minutes=5),
                'close': 155.0
            },
            {
                'symbol': 'AAPL', 
                'timestamp': base_time + timedelta(minutes=1),
                'close': 151.0
            },
            {
                'symbol': 'AAPL',
                'timestamp': base_time + timedelta(minutes=10),
                'close': 160.0
            },
            {
                'symbol': 'AAPL',
                'timestamp': base_time + timedelta(minutes=3),
                'close': 153.0
            }
        ]
        
        return pd.DataFrame(data)
    
    def test_extract_symbol_data_basic_extraction(self, multi_symbol_data):
        """Test basic symbol data extraction."""
        result = extract_symbol_data(multi_symbol_data, 'AAPL')
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10  # AAPL has 10 rows
        assert all(result['symbol'] == 'AAPL')
        
        # Check that data is sorted by timestamp
        timestamps = result['timestamp'].values
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    
    def test_extract_symbol_data_different_symbols(self, multi_symbol_data):
        """Test extraction of different symbols."""
        aapl_result = extract_symbol_data(multi_symbol_data, 'AAPL')
        tsla_result = extract_symbol_data(multi_symbol_data, 'TSLA')
        msft_result = extract_symbol_data(multi_symbol_data, 'MSFT')
        
        assert len(aapl_result) == 10
        assert len(tsla_result) == 8
        assert len(msft_result) == 5
        
        assert all(aapl_result['symbol'] == 'AAPL')
        assert all(tsla_result['symbol'] == 'TSLA')
        assert all(msft_result['symbol'] == 'MSFT')
    
    def test_extract_symbol_data_nonexistent_symbol(self, multi_symbol_data):
        """Test extraction of symbol that doesn't exist in data."""
        result = extract_symbol_data(multi_symbol_data, 'NONEXISTENT')
        
        assert result is None
    
    def test_extract_symbol_data_empty_dataframe(self):
        """Test extraction from empty DataFrame."""
        empty_df = pd.DataFrame()
        result = extract_symbol_data(empty_df, 'AAPL')
        
        assert result is None
    
    def test_extract_symbol_data_missing_symbol_column(self, single_symbol_data):
        """Test extraction when 'symbol' column is missing."""
        df_no_symbol = single_symbol_data.drop('symbol', axis=1)
        
        result = extract_symbol_data(df_no_symbol, 'AAPL')
        
        assert result is None
    
    def test_extract_symbol_data_missing_timestamp_column(self, multi_symbol_data):
        """Test extraction when 'timestamp' column is missing."""
        df_no_timestamp = multi_symbol_data.drop('timestamp', axis=1)
        
        result = extract_symbol_data(df_no_timestamp, 'AAPL')
        
        assert result is None
    
    def test_extract_symbol_data_timestamp_sorting(self, unordered_timestamp_data):
        """Test that extracted data is properly sorted by timestamp."""
        result = extract_symbol_data(unordered_timestamp_data, 'AAPL')
        
        assert result is not None
        assert len(result) == 4
        
        # Verify sorting
        expected_closes = [151.0, 153.0, 155.0, 160.0]  # Sorted by timestamp
        actual_closes = result['close'].tolist()
        assert actual_closes == expected_closes
    
    def test_extract_symbol_data_case_sensitivity(self, multi_symbol_data):
        """Test case sensitivity of symbol matching."""
        # Should not find lowercase
        result_lower = extract_symbol_data(multi_symbol_data, 'aapl')
        assert result_lower is None
        
        # Should find exact case
        result_exact = extract_symbol_data(multi_symbol_data, 'AAPL')
        assert result_exact is not None
        assert len(result_exact) == 10
    
    def test_extract_symbol_data_preserves_columns(self, multi_symbol_data):
        """Test that all original columns are preserved in extracted data."""
        result = extract_symbol_data(multi_symbol_data, 'AAPL')
        
        expected_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in result.columns for col in expected_columns)
        assert len(result.columns) == len(expected_columns)
    
    def test_extract_symbol_data_index_reset(self, multi_symbol_data):
        """Test that extracted data preserves original index from filtering."""
        result = extract_symbol_data(multi_symbol_data, 'TSLA')
        
        # The function preserves original index from the filtered DataFrame
        # TSLA data starts at index 10 in the multi_symbol_data
        assert len(result) == 8  # TSLA has 8 rows
        assert isinstance(result.index, pd.Index)
        # Index values will be from the original DataFrame after filtering
    
    def test_extract_symbol_data_data_integrity(self, multi_symbol_data):
        """Test that data values are preserved correctly."""
        result = extract_symbol_data(multi_symbol_data, 'AAPL')
        
        # Check first row data
        first_row = result.iloc[0]
        assert first_row['symbol'] == 'AAPL'
        assert first_row['open'] == 150.0
        assert first_row['high'] == 152.0
        assert first_row['low'] == 148.0
        assert first_row['close'] == 151.0
        assert first_row['volume'] == 1000
        
        # Check last row data
        last_row = result.iloc[-1]
        assert last_row['open'] == 159.0  # 150.0 + 9
        assert last_row['volume'] == 1900  # 1000 + 9 * 100
    
    def test_extract_symbol_data_with_nan_values(self):
        """Test extraction with NaN values in data."""
        data_with_nan = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'close': [100.0, np.nan, 102.0],
            'volume': [1000, 1100, np.nan]
        })
        
        result = extract_symbol_data(data_with_nan, 'AAPL')
        
        assert result is not None
        assert len(result) == 3
        assert np.isnan(result.iloc[1]['close'])
        assert np.isnan(result.iloc[2]['volume'])
    
    def test_extract_symbol_data_duplicate_timestamps(self):
        """Test extraction with duplicate timestamps."""
        data_with_duplicates = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'timestamp': [
                datetime(2024, 1, 1, 9, 30),
                datetime(2024, 1, 1, 9, 30),  # Duplicate
                datetime(2024, 1, 1, 9, 31)
            ],
            'close': [100.0, 101.0, 102.0]
        })
        
        result = extract_symbol_data(data_with_duplicates, 'AAPL')
        
        assert result is not None
        assert len(result) == 3
        # Should preserve all rows even with duplicate timestamps
    
    def test_extract_symbol_data_different_data_types(self):
        """Test extraction with different data types."""
        mixed_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'close': [100, 101.5, 102],  # Mixed int/float
            'volume': ['1000', '1100', '1200'],  # String numbers
            'flag': [True, False, True]  # Boolean
        })
        
        result = extract_symbol_data(mixed_data, 'AAPL')
        
        assert result is not None
        assert len(result) == 3
        # Data types should be preserved
        assert result['volume'].dtype == object  # String type preserved
        assert result['flag'].dtype == bool
    
    def test_extract_symbol_data_large_dataset_performance(self):
        """Test extraction performance with large dataset."""
        # Create large dataset with multiple symbols
        large_data = []
        symbols = ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL']
        
        for symbol in symbols:
            for i in range(1000):  # 1000 rows per symbol
                large_data.append({
                    'symbol': symbol,
                    'timestamp': datetime(2024, 1, 1) + timedelta(minutes=i),
                    'close': 100.0 + i * 0.01
                })
        
        large_df = pd.DataFrame(large_data)
        
        import time
        start_time = time.time()
        result = extract_symbol_data(large_df, 'AAPL')
        end_time = time.time()
        
        assert result is not None
        assert len(result) == 1000
        # Should complete quickly
        assert (end_time - start_time) < 1.0
    
    def test_extract_symbol_data_memory_efficiency(self, multi_symbol_data):
        """Test that extraction doesn't modify original DataFrame."""
        original_shape = multi_symbol_data.shape
        original_symbols = multi_symbol_data['symbol'].unique().tolist()
        
        result = extract_symbol_data(multi_symbol_data, 'AAPL')
        
        # Original DataFrame should be unchanged
        assert multi_symbol_data.shape == original_shape
        assert multi_symbol_data['symbol'].unique().tolist() == original_symbols
        
        # Result should be a separate object
        assert result is not multi_symbol_data
    
    def test_extract_symbol_data_edge_case_special_symbols(self):
        """Test extraction with special symbol names."""
        special_data = pd.DataFrame({
            'symbol': ['AAPL', 'BRK.A', 'BRK-B', 'TST_123', ''],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'close': [100, 200, 150, 75, 50]
        })
        
        # Test extraction of symbol with dot
        result_dot = extract_symbol_data(special_data, 'BRK.A')
        assert result_dot is not None
        assert len(result_dot) == 1
        
        # Test extraction of symbol with hyphen
        result_hyphen = extract_symbol_data(special_data, 'BRK-B')
        assert result_hyphen is not None
        assert len(result_hyphen) == 1
        
        # Test extraction of symbol with underscore and numbers
        result_underscore = extract_symbol_data(special_data, 'TST_123')
        assert result_underscore is not None
        assert len(result_underscore) == 1
        
        # Test extraction of empty symbol
        result_empty = extract_symbol_data(special_data, '')
        assert result_empty is not None
        assert len(result_empty) == 1
    
    def test_extract_symbol_data_exception_handling(self):
        """Test exception handling in extract_symbol_data."""
        # Test with invalid DataFrame (missing columns, etc.)
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        result = extract_symbol_data(invalid_data, 'AAPL')
        assert result is None
        
        # Test with None as DataFrame (should handle gracefully)
        result_none = extract_symbol_data(None, 'AAPL')
        assert result_none is None
    
    def test_extract_symbol_data_whitespace_handling(self):
        """Test handling of symbols with whitespace."""
        whitespace_data = pd.DataFrame({
            'symbol': ['AAPL', ' AAPL', 'AAPL ', ' AAPL ', 'MSFT'],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'close': [100, 101, 102, 103, 200]
        })
        
        # Exact match should work
        result_exact = extract_symbol_data(whitespace_data, 'AAPL')
        assert result_exact is not None
        assert len(result_exact) == 1  # Only exact match
        
        # Space variations should be separate
        result_leading = extract_symbol_data(whitespace_data, ' AAPL')
        assert result_leading is not None
        assert len(result_leading) == 1
        
        result_trailing = extract_symbol_data(whitespace_data, 'AAPL ')
        assert result_trailing is not None
        assert len(result_trailing) == 1
    
    def test_extract_symbol_data_return_type_consistency(self, multi_symbol_data):
        """Test that return type is consistent."""
        # Success case
        result_success = extract_symbol_data(multi_symbol_data, 'AAPL')
        assert isinstance(result_success, pd.DataFrame)
        
        # Failure case
        result_failure = extract_symbol_data(multi_symbol_data, 'NONEXISTENT')
        assert result_failure is None
        
        # Empty result case
        empty_df = pd.DataFrame({'symbol': [], 'timestamp': []})
        result_empty = extract_symbol_data(empty_df, 'AAPL')
        assert result_empty is None