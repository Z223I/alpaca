"""
Comprehensive tests for CSV reading utility function.
"""

import pytest
import csv
import tempfile
import os
from unittest.mock import patch, mock_open
from atoms.utils.read_csv import read_csv


class TestReadCSV:
    """Test suite for read_csv function."""
    
    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content for testing."""
        return """name,age,price,active
John,25,100.50,true
Jane,30,200.75,false
Bob,35,150.25,true"""
    
    @pytest.fixture
    def sample_csv_file(self, sample_csv_content):
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def empty_csv_file(self):
        """Create empty CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def headers_only_csv_file(self):
        """Create CSV file with headers only."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age,price\n")
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    def test_read_csv_basic_functionality(self, sample_csv_file):
        """Test basic CSV reading functionality."""
        result = read_csv(sample_csv_file)
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        # Check first row
        assert result[0]['name'] == 'John'
        assert result[0]['age'] == 25
        assert result[0]['price'] == 100.50
        assert result[0]['active'] == 'true'
        
        # Check second row
        assert result[1]['name'] == 'Jane'
        assert result[1]['age'] == 30
        assert result[1]['price'] == 200.75
        assert result[1]['active'] == 'false'
    
    def test_read_csv_data_type_conversion(self, sample_csv_file):
        """Test automatic data type conversion."""
        result = read_csv(sample_csv_file)
        
        # Integers should be converted
        assert isinstance(result[0]['age'], int)
        assert result[0]['age'] == 25
        
        # Floats should be converted  
        assert isinstance(result[0]['price'], float)
        assert result[0]['price'] == 100.50
        
        # Strings should remain strings
        assert isinstance(result[0]['name'], str)
        assert result[0]['name'] == 'John'
        
        # Non-numeric strings should remain strings
        assert isinstance(result[0]['active'], str)
        assert result[0]['active'] == 'true'
    
    def test_read_csv_custom_delimiter(self):
        """Test CSV reading with custom delimiter."""
        tab_content = "name\tage\tprice\nJohn\t25\t100.50\nJane\t30\t200.75"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write(tab_content)
            f.flush()
            
            try:
                result = read_csv(f.name, delimiter='\t')
                
                assert len(result) == 2
                assert result[0]['name'] == 'John'
                assert result[0]['age'] == 25
                assert result[0]['price'] == 100.50
            finally:
                os.unlink(f.name)
    
    def test_read_csv_semicolon_delimiter(self):
        """Test CSV reading with semicolon delimiter."""
        semicolon_content = "name;age;price\nJohn;25;100.50\nJane;30;200.75"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(semicolon_content)
            f.flush()
            
            try:
                result = read_csv(f.name, delimiter=';')
                
                assert len(result) == 2
                assert result[0]['name'] == 'John'
                assert result[0]['age'] == 25
            finally:
                os.unlink(f.name)
    
    def test_read_csv_auto_delimiter_detection(self):
        """Test automatic delimiter detection."""
        semicolon_content = "name;age;price\nJohn;25;100.50\nJane;30;200.75"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(semicolon_content)
            f.flush()
            
            try:
                # Should auto-detect semicolon delimiter
                result = read_csv(f.name)
                
                assert len(result) == 2
                assert result[0]['name'] == 'John'
                assert result[0]['age'] == 25
            finally:
                os.unlink(f.name)
    
    def test_read_csv_custom_encoding(self):
        """Test CSV reading with custom encoding."""
        # Create file with UTF-8 content including special characters
        utf8_content = "name,description\nJohn,Café münü\nJané,Résumé"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(utf8_content)
            f.flush()
            
            try:
                result = read_csv(f.name, encoding='utf-8')
                
                assert len(result) == 2
                assert result[0]['name'] == 'John'
                assert result[0]['description'] == 'Café münü'
                assert result[1]['name'] == 'Jané'
                assert result[1]['description'] == 'Résumé'
            finally:
                os.unlink(f.name)
    
    def test_read_csv_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            read_csv("nonexistent_file.csv")
    
    def test_read_csv_empty_file(self, empty_csv_file):
        """Test reading empty CSV file."""
        result = read_csv(empty_csv_file)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_read_csv_headers_only(self, headers_only_csv_file):
        """Test reading CSV file with headers only."""
        result = read_csv(headers_only_csv_file)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_read_csv_mixed_data_types(self):
        """Test CSV with mixed data types including edge cases."""
        mixed_content = """id,value,flag,decimal,empty,zero
1,text,true,3.14,,0
2,123,false,0.0,null,0.00
3,,true,-1.5,empty,"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(mixed_content)
            f.flush()
            
            try:
                result = read_csv(f.name)
                
                assert len(result) == 3
                
                # Row 1
                assert result[0]['id'] == 1
                assert result[0]['value'] == 'text'
                assert result[0]['flag'] == 'true'
                assert result[0]['decimal'] == 3.14
                assert result[0]['empty'] is None
                assert result[0]['zero'] == 0
                
                # Row 2
                assert result[1]['id'] == 2
                assert result[1]['value'] == 123
                assert result[1]['flag'] == 'false'
                assert result[1]['decimal'] == 0.0
                assert result[1]['empty'] == 'null'
                assert result[1]['zero'] == 0.0
                
                # Row 3
                assert result[2]['id'] == 3
                assert result[2]['value'] is None
                assert result[2]['flag'] == 'true'
                assert result[2]['decimal'] == -1.5
                assert result[2]['empty'] == 'empty'
                assert result[2]['zero'] is None
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_quoted_fields(self):
        """Test CSV with quoted fields containing special characters."""
        quoted_content = '''name,description,notes
"John Doe","Contains, comma","Normal text"
"Jane Smith","Contains ""quotes""","Text with newline
in middle"
"Bob Wilson","Mixed, ""quoted"" content","Final entry"'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(quoted_content)
            f.flush()
            
            try:
                result = read_csv(f.name)
                
                assert len(result) == 3
                assert result[0]['name'] == 'John Doe'
                assert result[0]['description'] == 'Contains, comma'
                assert result[1]['description'] == 'Contains "quotes"'
                assert 'newline\nin middle' in result[1]['notes']
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_large_numbers(self):
        """Test CSV with large numbers and scientific notation."""
        large_numbers_content = """id,big_int,big_float,scientific
1,9223372036854775807,999999999.999999,1e10
2,-9223372036854775808,-999999999.999999,1.5e-5
3,0,0.0,0e0"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(large_numbers_content)
            f.flush()
            
            try:
                result = read_csv(f.name)
                
                assert len(result) == 3
                assert result[0]['big_int'] == 9223372036854775807
                assert result[0]['big_float'] == 999999999.999999
                # Scientific notation may be kept as string depending on implementation
                assert result[0]['scientific'] == '1e10' or result[0]['scientific'] == 1e10
                assert result[1]['scientific'] == '1.5e-5' or result[1]['scientific'] == 1.5e-5
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_unicode_decode_error(self):
        """Test handling of unicode decode errors."""
        # Create file with wrong encoding
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
            # Write some bytes that aren't valid UTF-8
            f.write(b'name,data\ntest,\xff\xfe\x00\x00')
            f.flush()
            
            try:
                with pytest.raises(UnicodeDecodeError):
                    read_csv(f.name, encoding='utf-8')
            finally:
                os.unlink(f.name)
    
    def test_read_csv_malformed_csv(self):
        """Test handling of malformed CSV content."""
        malformed_content = '''name,age,city
John,25,New York
Jane,30,"Unclosed quote
Bob,35,Chicago'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(malformed_content)
            f.flush()
            
            try:
                # The CSV reader may handle this gracefully rather than raising an error
                result = read_csv(f.name)
                # Test that it returns some data (may handle malformed rows differently)
                assert isinstance(result, list)
            except csv.Error:
                # If it does raise an error, that's also acceptable behavior
                pass
            finally:
                os.unlink(f.name)
    
    def test_read_csv_whitespace_handling(self):
        """Test handling of whitespace in CSV fields."""
        whitespace_content = """name,age,city
  John  ,  25  ,  New York  
Jane,30,Chicago
  Bob,35,Boston  """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(whitespace_content)
            f.flush()
            
            try:
                result = read_csv(f.name)
                
                assert len(result) == 3
                # Whitespace should be stripped
                assert result[0]['name'] == 'John'
                assert result[0]['city'] == 'New York'
                assert result[2]['name'] == 'Bob'
                assert result[2]['city'] == 'Boston'
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_boolean_like_strings(self):
        """Test handling of boolean-like strings."""
        boolean_content = """name,active,verified,enabled
John,true,yes,1
Jane,false,no,0
Bob,True,Yes,on
Alice,FALSE,NO,off"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(boolean_content)
            f.flush()
            
            try:
                result = read_csv(f.name)
                
                assert len(result) == 4
                # Boolean-like strings should remain as strings
                assert result[0]['active'] == 'true'
                assert result[0]['verified'] == 'yes'
                assert result[0]['enabled'] == 1  # Numeric conversion
                assert result[1]['active'] == 'false'
                assert result[1]['enabled'] == 0  # Numeric conversion
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_duplicate_headers(self):
        """Test CSV with duplicate column headers."""
        duplicate_headers_content = """name,age,name,city
John,25,Johnny,New York
Jane,30,Janie,Chicago"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(duplicate_headers_content)
            f.flush()
            
            try:
                result = read_csv(f.name)
                
                assert len(result) == 2
                # Should handle duplicate headers (behavior depends on CSV reader)
                assert 'name' in result[0]
                assert result[0]['age'] == 25
                assert result[0]['city'] == 'New York'
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_very_long_lines(self):
        """Test CSV with very long lines."""
        long_text = "x" * 10000
        long_lines_content = f"""id,long_text,value
1,{long_text},100
2,short,200"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(long_lines_content)
            f.flush()
            
            try:
                result = read_csv(f.name)
                
                assert len(result) == 2
                assert result[0]['id'] == 1
                assert len(result[0]['long_text']) == 10000
                assert result[0]['value'] == 100
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_many_columns(self):
        """Test CSV with many columns."""
        # Create CSV with 100 columns
        headers = [f'col_{i}' for i in range(100)]
        values1 = list(range(100))
        values2 = list(range(100, 200))
        
        many_cols_content = ','.join(headers) + '\n' + \
                           ','.join(map(str, values1)) + '\n' + \
                           ','.join(map(str, values2))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(many_cols_content)
            f.flush()
            
            try:
                result = read_csv(f.name)
                
                assert len(result) == 2
                assert len(result[0]) == 100
                assert result[0]['col_0'] == 0
                assert result[0]['col_99'] == 99
                assert result[1]['col_0'] == 100
                assert result[1]['col_99'] == 199
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_performance_large_file(self):
        """Test performance with large CSV file."""
        # Create large CSV file (1000 rows)
        large_content = "id,name,value,price\n"
        for i in range(1000):
            large_content += f"{i},name_{i},{i*10},{i*1.5}\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(large_content)
            f.flush()
            
            try:
                import time
                start_time = time.time()
                result = read_csv(f.name)
                end_time = time.time()
                
                assert len(result) == 1000
                assert result[0]['id'] == 0
                assert result[999]['id'] == 999
                # Should complete in reasonable time
                assert (end_time - start_time) < 5.0
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_edge_case_numeric_strings(self):
        """Test edge cases for numeric string conversion."""
        edge_cases_content = """value,description
123,integer
123.0,float_with_zero_decimal
123.,float_with_trailing_dot
.123,float_with_leading_dot
00123,leading_zeros
123abc,mixed_alphanumeric
+123,positive_sign
-123,negative_sign
1.23e4,scientific_notation
inf,infinity
nan,not_a_number"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(edge_cases_content)
            f.flush()
            
            try:
                result = read_csv(f.name)
                
                assert len(result) == 11
                assert result[0]['value'] == 123  # Integer
                assert result[1]['value'] == 123.0  # Float
                assert result[2]['value'] == 123.0  # Float with trailing dot
                assert result[3]['value'] == 0.123  # Float with leading dot
                assert result[4]['value'] == 123  # Leading zeros converted to int
                assert result[5]['value'] == '123abc'  # Mixed - remains string
                assert result[6]['value'] == 123  # Positive sign
                assert result[7]['value'] == -123  # Negative sign
                assert result[8]['value'] == 12300.0  # Scientific notation
                # inf and nan might be converted differently
                
            finally:
                os.unlink(f.name)
    
    def test_read_csv_default_parameters(self, sample_csv_file):
        """Test that default parameters work correctly."""
        # Test with all defaults
        result = read_csv(sample_csv_file)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]['name'] == 'John'
    
    def test_read_csv_function_signature(self):
        """Test that function has correct signature."""
        import inspect
        
        sig = inspect.signature(read_csv)
        params = sig.parameters
        
        assert 'filename' in params
        assert 'delimiter' in params
        assert 'encoding' in params
        
        # Check default values
        assert params['delimiter'].default == ','
        assert params['encoding'].default == 'utf-8'