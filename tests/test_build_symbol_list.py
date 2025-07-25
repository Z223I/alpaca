"""
Comprehensive tests for build_symbol_list API atom.
"""

import pytest
import tempfile
import os
import csv
from datetime import datetime
from unittest.mock import patch, MagicMock
from atoms.api.build_symbol_list import build_symbol_list, build_daily_accumulated_list, _write_accumulated_csv


class TestBuildSymbolList:
    """Test suite for build_symbol_list function."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        for file in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    
    @pytest.fixture
    def sample_csv_data_recent(self):
        """Sample CSV data for the most recent file (should be preserved)."""
        return """Symbol,Signal,Long Delta,Resistance,Max,Last,%Chg,Volume,Float,Mkt Cap
AAPL,150.00,5.25,155.25,160.00,158.75,8.5%,45.2M,15.3B,2.8T
GOOGL,2800.00,45.50,2845.50,2900.00,2875.25,6.2%,12.1M,6.8B,1.9T
MSFT,420.00,15.75,435.75,440.00,438.50,4.3%,28.3M,7.4B,3.2T"""
    
    @pytest.fixture
    def sample_csv_data_older(self):
        """Sample CSV data for older files (should be zeroed except Symbol)."""
        return """Symbol,Signal,Long Delta,Resistance,Max,Last,%Chg,Volume,Float,Mkt Cap
TSLA,800.00,25.50,825.50,850.00,845.25,12.1%,35.7M,3.2B,850B
META,380.00,18.25,398.25,405.00,402.75,9.8%,22.4M,2.6B,1.1T
NVDA,900.00,35.00,935.00,945.00,942.50,7.5%,18.9M,2.5B,2.3T"""
    
    @pytest.fixture
    def sample_csv_data_overlap(self):
        """Sample CSV data with overlapping symbols (should be overridden by most recent)."""
        return """Symbol,Signal,Long Delta,Resistance,Max,Last,%Chg,Volume,Float,Mkt Cap
AAPL,140.00,3.25,143.25,148.00,146.75,5.2%,32.1M,15.3B,2.7T
AMZN,3200.00,55.75,3255.75,3300.00,3285.50,8.9%,15.6M,5.1B,1.6T"""
    
    def create_test_csv_file(self, temp_dir: str, filename: str, content: str):
        """Helper to create test CSV files."""
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath
    
    def test_build_symbol_list_basic_functionality(self, temp_data_dir, sample_csv_data_recent, sample_csv_data_older):
        """Test basic symbol list building functionality."""
        # Create test files with dates
        self.create_test_csv_file(temp_data_dir, "20250725.csv", sample_csv_data_recent)
        self.create_test_csv_file(temp_data_dir, "20250724.csv", sample_csv_data_older)
        
        result = build_symbol_list(temp_data_dir)
        
        assert isinstance(result, list)
        # Should have 6 unique symbols total
        assert len(result) == 6
        
        # Create symbol lookup for easier testing
        symbol_dict = {row['Symbol']: row for row in result}
        
        # Check most recent data is preserved (AAPL, GOOGL, MSFT)
        assert symbol_dict['AAPL']['Signal'] == 150.00
        assert symbol_dict['AAPL']['Last'] == 158.75
        assert symbol_dict['GOOGL']['Signal'] == 2800.00
        assert symbol_dict['MSFT']['Signal'] == 420.00
        
        # Check older data is zeroed (TSLA, META, NVDA)
        assert symbol_dict['TSLA']['Signal'] == 0
        assert symbol_dict['TSLA']['Last'] == 0
        assert symbol_dict['META']['Signal'] == 0
        assert symbol_dict['NVDA']['Signal'] == 0
        
        # Symbol column should always be preserved
        assert symbol_dict['TSLA']['Symbol'] == 'TSLA'
        assert symbol_dict['META']['Symbol'] == 'META'
    
    def test_build_symbol_list_duplicate_symbols(self, temp_data_dir, sample_csv_data_recent, sample_csv_data_overlap):
        """Test handling of duplicate symbols across files."""
        # Create files where AAPL appears in both (most recent should win)
        self.create_test_csv_file(temp_data_dir, "20250725.csv", sample_csv_data_recent)
        self.create_test_csv_file(temp_data_dir, "20250723.csv", sample_csv_data_overlap)
        
        result = build_symbol_list(temp_data_dir)
        
        symbol_dict = {row['Symbol']: row for row in result}
        
        # AAPL should have data from most recent file (20250725.csv)
        assert symbol_dict['AAPL']['Signal'] == 150.00  # From recent, not 140.00 from older
        assert symbol_dict['AAPL']['Last'] == 158.75
        
        # AMZN should have zeroed data (from older file)
        assert symbol_dict['AMZN']['Signal'] == 0
        assert symbol_dict['AMZN']['Last'] == 0
        assert symbol_dict['AMZN']['Symbol'] == 'AMZN'
    
    def test_build_symbol_list_multiple_files_date_sorting(self, temp_data_dir):
        """Test that files are properly sorted by date."""
        # Create files in non-chronological order
        content1 = "Symbol,Signal\nTEST1,100.00"
        content2 = "Symbol,Signal\nTEST2,200.00"
        content3 = "Symbol,Signal\nTEST3,300.00"
        
        self.create_test_csv_file(temp_data_dir, "20250723.csv", content1)  # Older
        self.create_test_csv_file(temp_data_dir, "20250725.csv", content3)  # Most recent
        self.create_test_csv_file(temp_data_dir, "20250724.csv", content2)  # Middle
        
        result = build_symbol_list(temp_data_dir)
        
        symbol_dict = {row['Symbol']: row for row in result}
        
        # TEST3 should have preserved data (from most recent 20250725.csv)
        assert symbol_dict['TEST3']['Signal'] == 300.00
        
        # TEST1 and TEST2 should have zeroed data
        assert symbol_dict['TEST1']['Signal'] == 0
        assert symbol_dict['TEST2']['Signal'] == 0
    
    def test_build_symbol_list_invalid_date_files_ignored(self, temp_data_dir):
        """Test that files with invalid date formats are ignored."""
        valid_content = "Symbol,Signal\nVALID,100.00"
        invalid_content = "Symbol,Signal\nINVALID,200.00"
        
        self.create_test_csv_file(temp_data_dir, "20250725.csv", valid_content)
        self.create_test_csv_file(temp_data_dir, "invalid_date.csv", invalid_content)
        self.create_test_csv_file(temp_data_dir, "20250230.csv", invalid_content)  # Invalid date
        self.create_test_csv_file(temp_data_dir, "not_date.csv", invalid_content)
        
        result = build_symbol_list(temp_data_dir)
        
        # Should only process the valid date file
        assert len(result) == 1
        assert result[0]['Symbol'] == 'VALID'
        assert result[0]['Signal'] == 100.00
    
    def test_build_symbol_list_with_output_file(self, temp_data_dir, sample_csv_data_recent):
        """Test writing output to file."""
        self.create_test_csv_file(temp_data_dir, "20250725.csv", sample_csv_data_recent)
        output_file = os.path.join(temp_data_dir, "accumulated_symbols.csv")
        
        result = build_symbol_list(temp_data_dir, output_file)
        
        # Check that output file was created
        assert os.path.exists(output_file)
        
        # Read the output file and verify content
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            output_data = list(reader)
        
        assert len(output_data) == len(result)
        assert output_data[0]['Symbol'] == result[0]['Symbol']
        assert float(output_data[0]['Signal']) == result[0]['Signal']
    
    def test_build_symbol_list_empty_directory(self, temp_data_dir):
        """Test handling of directory with no CSV files."""
        with pytest.raises(ValueError, match="No valid YYYYMMDD.csv files found"):
            build_symbol_list(temp_data_dir)
    
    def test_build_symbol_list_nonexistent_directory(self):
        """Test handling of non-existent directory."""
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            build_symbol_list("/nonexistent/directory")
    
    def test_build_symbol_list_empty_most_recent_file(self, temp_data_dir):
        """Test handling of empty most recent file."""
        self.create_test_csv_file(temp_data_dir, "20250725.csv", "Symbol,Signal\n")  # Headers only
        
        with pytest.raises(ValueError, match="Most recent file .* is empty"):
            build_symbol_list(temp_data_dir)
    
    def test_build_symbol_list_file_read_error(self, temp_data_dir, sample_csv_data_recent):
        """Test handling of file read errors in older files."""
        # Create valid most recent file
        self.create_test_csv_file(temp_data_dir, "20250725.csv", sample_csv_data_recent)
        
        # Create older file with invalid content (will cause read error)
        problematic_file = os.path.join(temp_data_dir, "20250724.csv")
        with open(problematic_file, 'wb') as f:
            f.write(b'\xff\xfe\x00\x00')  # Invalid UTF-8
        
        # Should continue processing despite the problematic file
        result = build_symbol_list(temp_data_dir)
        
        # Should have symbols from the valid file only
        assert len(result) == 3
        symbol_dict = {row['Symbol']: row for row in result}
        assert 'AAPL' in symbol_dict
        assert symbol_dict['AAPL']['Signal'] == 150.00
    
    def test_build_symbol_list_all_columns_preserved(self, temp_data_dir):
        """Test that all columns are preserved in the output."""
        content = """Symbol,Signal,Long Delta,Resistance,Max,Last,%Chg,Volume,Float,Mkt Cap,Custom1,Custom2
AAPL,150.00,5.25,155.25,160.00,158.75,8.5%,45.2M,15.3B,2.8T,Extra1,Extra2"""
        
        self.create_test_csv_file(temp_data_dir, "20250725.csv", content)
        
        result = build_symbol_list(temp_data_dir)
        
        assert len(result) == 1
        row = result[0]
        
        # Check all columns are present
        expected_columns = ['Symbol', 'Signal', 'Long Delta', 'Resistance', 'Max', 'Last', 
                          '%Chg', 'Volume', 'Float', 'Mkt Cap', 'Custom1', 'Custom2']
        for col in expected_columns:
            assert col in row
        
        # Check values are preserved
        assert row['Custom1'] == 'Extra1'
        assert row['Custom2'] == 'Extra2'
    
    def test_build_symbol_list_column_mismatch_handling(self, temp_data_dir):
        """Test handling of files with different column structures."""
        recent_content = "Symbol,Signal,Volume\nAAPL,150.00,45.2M"
        older_content = "Symbol,Signal,Price,Volume\nTSLA,800.00,845.25,35.7M"
        
        self.create_test_csv_file(temp_data_dir, "20250725.csv", recent_content)
        self.create_test_csv_file(temp_data_dir, "20250724.csv", older_content)
        
        result = build_symbol_list(temp_data_dir)
        
        symbol_dict = {row['Symbol']: row for row in result}
        
        # AAPL should have all data from recent file
        assert symbol_dict['AAPL']['Signal'] == 150.00
        assert symbol_dict['AAPL']['Volume'] == '45.2M'
        
        # TSLA should have zeroed fields (missing columns get 0)
        assert symbol_dict['TSLA']['Signal'] == 0
        assert symbol_dict['TSLA']['Volume'] == 0
    
    def test_build_symbol_list_sorting_consistency(self, temp_data_dir, sample_csv_data_recent):
        """Test that output is consistently sorted by symbol."""
        self.create_test_csv_file(temp_data_dir, "20250725.csv", sample_csv_data_recent)
        
        # Run multiple times to ensure consistent sorting
        result1 = build_symbol_list(temp_data_dir)
        result2 = build_symbol_list(temp_data_dir)
        
        symbols1 = [row['Symbol'] for row in result1]
        symbols2 = [row['Symbol'] for row in result2]
        
        assert symbols1 == symbols2
        assert symbols1 == sorted(symbols1)  # Should be sorted alphabetically


class TestBuildDailyAccumulatedList:
    """Test suite for build_daily_accumulated_list function."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        for file in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    
    def create_test_csv_file(self, temp_dir: str, filename: str, content: str):
        """Helper to create test CSV files."""
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath
    
    def test_build_daily_accumulated_list_creates_file(self, temp_data_dir):
        """Test that daily accumulated list creates output file."""
        content = "Symbol,Signal\nAAPL,150.00"
        self.create_test_csv_file(temp_data_dir, "20250725.csv", content)
        
        accumulated_file = os.path.join(temp_data_dir, "accumulated.csv")
        result = build_daily_accumulated_list(temp_data_dir, accumulated_file)
        
        # Check file was created
        assert os.path.exists(accumulated_file)
        
        # Check result
        assert len(result) == 1
        assert result[0]['Symbol'] == 'AAPL'
    
    def test_build_daily_accumulated_list_empty_data(self, temp_data_dir):
        """Test handling of empty data."""
        # Create empty directory
        accumulated_file = os.path.join(temp_data_dir, "accumulated.csv")
        
        with pytest.raises(ValueError):
            build_daily_accumulated_list(temp_data_dir, accumulated_file)
        
        # File should not be created if there's no data
        assert not os.path.exists(accumulated_file)


class TestWriteAccumulatedCSV:
    """Test suite for _write_accumulated_csv helper function."""
    
    def test_write_accumulated_csv_basic(self):
        """Test basic CSV writing functionality."""
        data = [
            {'Symbol': 'AAPL', 'Signal': 150.00, 'Volume': '45.2M'},
            {'Symbol': 'GOOGL', 'Signal': 2800.00, 'Volume': '12.1M'}
        ]
        columns = ['Symbol', 'Signal', 'Volume']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            _write_accumulated_csv(data, output_file, columns)
            
            # Read back and verify
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                result = list(reader)
            
            assert len(result) == 2
            assert result[0]['Symbol'] == 'AAPL'
            assert float(result[0]['Signal']) == 150.00
            
        finally:
            os.unlink(output_file)
    
    def test_write_accumulated_csv_write_error(self):
        """Test handling of write errors."""
        data = [{'Symbol': 'AAPL', 'Signal': 150.00}]
        columns = ['Symbol', 'Signal']
        invalid_path = "/invalid/path/file.csv"
        
        with pytest.raises(IOError, match="Error writing to"):
            _write_accumulated_csv(data, invalid_path, columns)
    
    def test_write_accumulated_csv_empty_data(self):
        """Test writing empty data."""
        data = []
        columns = ['Symbol', 'Signal']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            _write_accumulated_csv(data, output_file, columns)
            
            # Should create file with headers only
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert 'Symbol,Signal' in content
                
        finally:
            os.unlink(output_file)


class TestIntegrationWithRealData:
    """Integration tests using real CSV fixture data."""
    
    @pytest.fixture
    def real_data_dir(self):
        """Use the real test fixtures directory."""
        return "/home/wilsonb/dl/github.com/z223i/alpaca/tests/fixtures/build_symbol_list"
    
    def test_with_real_fixture_data(self, real_data_dir):
        """Test with real CSV fixture data."""
        if not os.path.exists(real_data_dir):
            pytest.skip("Real fixture data not available")
        
        # Check if fixture files exist
        csv_files = [f for f in os.listdir(real_data_dir) if f.endswith('.csv')]
        if not csv_files:
            pytest.skip("No CSV fixture files found")
        
        result = build_symbol_list(real_data_dir)
        
        # Basic validation with real data
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Check that all results have Symbol field
        for row in result:
            assert 'Symbol' in row
            assert row['Symbol'] is not None
            assert row['Symbol'] != ''
        
        # Check that symbols are unique
        symbols = [row['Symbol'] for row in result]
        assert len(symbols) == len(set(symbols))
    
    def test_performance_with_real_data(self, real_data_dir):
        """Test performance with real data."""
        if not os.path.exists(real_data_dir):
            pytest.skip("Real fixture data not available")
        
        csv_files = [f for f in os.listdir(real_data_dir) if f.endswith('.csv')]
        if not csv_files:
            pytest.skip("No CSV fixture files found")
        
        import time
        start_time = time.time()
        result = build_symbol_list(real_data_dir)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 5.0
        assert len(result) > 0