"""
Unit tests for ORB Alert Configuration System

Tests the configuration management for ORB trading alerts.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

from atoms.config.alert_config import ORBAlertConfig, config
from atoms.config.symbol_manager import SymbolManager


class TestORBAlertConfig:
    """Test cases for ORB Alert Configuration."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        test_config = ORBAlertConfig()
        
        assert test_config.orb_period_minutes == 15
        assert test_config.breakout_threshold == 0.002
        assert test_config.volume_multiplier == 1.5
        assert test_config.pc1_weight == 0.8231
        assert test_config.pc2_weight == 0.0854
        assert test_config.pc3_weight == 0.0378
        assert test_config.min_price == 0.01
        assert test_config.max_price == 10.00
        assert test_config.min_volume == 1000000
        assert test_config.alert_window_start == "09:45"
        assert test_config.alert_window_end == "15:30"
    
    @patch.dict(os.environ, {
        'ALPACA_API_KEY': 'test_api_key',
        'ALPACA_SECRET_KEY': 'test_secret_key',
        'ALPACA_BASE_URL': 'https://test.alpaca.markets'
    })
    def test_config_from_environment(self):
        """Test configuration loading from environment variables."""
        test_config = ORBAlertConfig()
        
        assert test_config.api_key == 'test_api_key'
        assert test_config.secret_key == 'test_secret_key'
        assert test_config.base_url == 'https://test.alpaca.markets'
    
    @patch.dict(os.environ, {}, clear=True)
    def test_config_missing_credentials(self):
        """Test configuration with missing credentials."""
        with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"):
            ORBAlertConfig()
    
    def test_config_validation_success(self):
        """Test successful configuration validation."""
        test_config = ORBAlertConfig()
        test_config.api_key = "test_key"
        test_config.secret_key = "test_secret"
        
        errors = test_config.validate()
        assert errors == []
    
    def test_config_validation_errors(self):
        """Test configuration validation with errors."""
        test_config = ORBAlertConfig()
        test_config.api_key = "test_key"
        test_config.secret_key = "test_secret"
        test_config.orb_period_minutes = -1
        test_config.breakout_threshold = -0.1
        test_config.volume_multiplier = 0
        test_config.min_price = 10.0
        test_config.max_price = 5.0
        test_config.min_volume = -100
        test_config.min_confidence_score = 1.5
        
        errors = test_config.validate()
        
        assert "orb_period_minutes must be positive" in errors
        assert "breakout_threshold must be positive" in errors
        assert "volume_multiplier must be positive" in errors
        assert "max_price must be greater than min_price" in errors
        assert "min_volume must be positive" in errors
        assert "min_confidence_score must be between 0 and 1" in errors
    
    def test_pca_weights_sum(self):
        """Test that PCA weights are reasonable."""
        test_config = ORBAlertConfig()
        
        total_weight = test_config.pc1_weight + test_config.pc2_weight + test_config.pc3_weight
        
        # Should be close to 1.0 (allowing for rounding)
        assert abs(total_weight - 1.0) < 0.1
        
        # PC1 should be dominant (from PCA analysis)
        assert test_config.pc1_weight > 0.8
        assert test_config.pc2_weight < 0.1
        assert test_config.pc3_weight < 0.1


class TestSymbolManager:
    """Test cases for Symbol Manager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.symbols_file = os.path.join(self.temp_dir, "test_symbols.csv")
        
        # Create test symbols file
        with open(self.symbols_file, 'w') as f:
            f.write("AAPL\n")
            f.write("TSLA\n")
            f.write("MSFT\n")
            f.write("GOOGL\n")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_symbols_success(self):
        """Test successful symbol loading."""
        manager = SymbolManager(self.symbols_file)
        
        symbols = manager.get_symbols()
        assert len(symbols) == 4
        assert "AAPL" in symbols
        assert "TSLA" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" in symbols
    
    def test_load_symbols_file_not_found(self):
        """Test loading symbols from non-existent file."""
        with pytest.raises(FileNotFoundError):
            SymbolManager("nonexistent_file.csv")
    
    def test_load_symbols_empty_file(self):
        """Test loading symbols from empty file."""
        empty_file = os.path.join(self.temp_dir, "empty.csv")
        with open(empty_file, 'w') as f:
            f.write("")
        
        with pytest.raises(ValueError, match="No symbols found"):
            SymbolManager(empty_file)
    
    def test_add_symbol(self):
        """Test adding a symbol."""
        manager = SymbolManager(self.symbols_file)
        
        initial_count = len(manager.get_symbols())
        manager.add_symbol("NVDA")
        
        assert len(manager.get_symbols()) == initial_count + 1
        assert "NVDA" in manager.get_symbols()
    
    def test_add_symbol_lowercase(self):
        """Test adding a symbol with lowercase conversion."""
        manager = SymbolManager(self.symbols_file)
        
        manager.add_symbol("nvda")
        assert "NVDA" in manager.get_symbols()
    
    def test_remove_symbol(self):
        """Test removing a symbol."""
        manager = SymbolManager(self.symbols_file)
        
        initial_count = len(manager.get_symbols())
        manager.remove_symbol("AAPL")
        
        assert len(manager.get_symbols()) == initial_count - 1
        assert "AAPL" not in manager.get_symbols()
    
    def test_is_symbol_tracked(self):
        """Test checking if symbol is tracked."""
        manager = SymbolManager(self.symbols_file)
        
        assert manager.is_symbol_tracked("AAPL")
        assert manager.is_symbol_tracked("aapl")  # Case insensitive
        assert not manager.is_symbol_tracked("NVDA")
    
    def test_get_symbol_count(self):
        """Test getting symbol count."""
        manager = SymbolManager(self.symbols_file)
        
        assert manager.get_symbol_count() == 4
        
        manager.add_symbol("NVDA")
        assert manager.get_symbol_count() == 5
    
    def test_save_symbols(self):
        """Test saving symbols to file."""
        manager = SymbolManager(self.symbols_file)
        
        manager.add_symbol("NVDA")
        manager.add_symbol("AMD")
        
        new_file = os.path.join(self.temp_dir, "new_symbols.csv")
        manager.save_symbols(new_file)
        
        # Load with new manager to verify
        new_manager = SymbolManager(new_file)
        symbols = new_manager.get_symbols()
        
        assert len(symbols) == 6
        assert "NVDA" in symbols
        assert "AMD" in symbols
    
    def test_reload_symbols(self):
        """Test reloading symbols from file."""
        manager = SymbolManager(self.symbols_file)
        
        # Add symbol in memory
        manager.add_symbol("NVDA")
        assert "NVDA" in manager.get_symbols()
        
        # Reload from file (should remove in-memory additions)
        manager.reload_symbols()
        assert "NVDA" not in manager.get_symbols()
        assert len(manager.get_symbols()) == 4
    
    def test_iteration(self):
        """Test symbol manager iteration."""
        manager = SymbolManager(self.symbols_file)
        
        symbols_list = list(manager)
        assert len(symbols_list) == 4
        assert symbols_list == sorted(symbols_list)  # Should be sorted
    
    def test_contains(self):
        """Test symbol manager contains operation."""
        manager = SymbolManager(self.symbols_file)
        
        assert "AAPL" in manager
        assert "aapl" in manager  # Case insensitive
        assert "NVDA" not in manager
    
    def test_len(self):
        """Test symbol manager length."""
        manager = SymbolManager(self.symbols_file)
        
        assert len(manager) == 4
        
        manager.add_symbol("NVDA")
        assert len(manager) == 5