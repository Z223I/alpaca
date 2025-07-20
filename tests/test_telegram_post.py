import unittest
import os
import tempfile
import csv
from unittest.mock import patch, MagicMock, mock_open
from atoms.telegram.telegram_post import TelegramPoster, send_message, send_alert
from atoms.telegram.user_manager import UserManager
from atoms.telegram.config import TelegramConfig

class TestTelegramPost(unittest.TestCase):
    """Test cases for Telegram posting functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, '.telegram_users.csv')
        
        # Create test CSV with sample users
        test_users = [
            ['chat_id', 'username', 'enabled', 'created_date', 'notes'],
            ['123456789', 'test_user1', 'true', '2024-01-15', 'Test user 1'],
            ['987654321', 'test_user2', 'true', '2024-01-15', 'Test user 2'],
            ['555666777', 'disabled_user', 'false', '2024-01-15', 'Disabled user']
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(test_users)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {
        'BOT_TOKEN': '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11',
        'TELEGRAM_ENABLED': 'true'
    })
    @patch('atoms.telegram.telegram_post.requests.post')
    def test_send_message_success(self, mock_post):
        """Test successful message sending to valid users."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'ok': True, 'result': {'message_id': 123}}
        mock_post.return_value = mock_response
        
        # Create poster with test CSV
        poster = TelegramPoster()
        poster.user_manager.csv_path = self.csv_path
        
        result = poster.send_message("Test message")
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertEqual(result['sent_count'], 2)  # Only enabled users
        self.assertEqual(result['failed_count'], 0)
        self.assertEqual(len(result['errors']), 0)
        
        # Verify API was called for each enabled user
        self.assertEqual(mock_post.call_count, 2)
    
    @patch.dict(os.environ, {
        'BOT_TOKEN': '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11',
        'TELEGRAM_ENABLED': 'true'
    })
    @patch('atoms.telegram.telegram_post.requests.post')
    def test_send_message_with_failures(self, mock_post):
        """Test handling of partial failures."""
        # Mock mixed responses - first succeeds, second fails
        responses = [
            MagicMock(status_code=200, json=lambda: {'ok': True}),
            MagicMock(status_code=400, json=lambda: {'ok': False, 'description': 'Bad Request'})
        ]
        mock_post.side_effect = responses
        
        poster = TelegramPoster()
        poster.user_manager.csv_path = self.csv_path
        
        result = poster.send_message("Test message")
        
        # Verify partial success
        self.assertTrue(result['success'])  # At least one succeeded
        self.assertEqual(result['sent_count'], 1)
        self.assertEqual(result['failed_count'], 1)
        self.assertEqual(len(result['errors']), 1)
    
    def test_csv_file_creation(self):
        """Test automatic CSV creation when missing."""
        non_existent_path = os.path.join(self.temp_dir, 'new_users.csv')
        
        user_manager = UserManager(non_existent_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(non_existent_path))
        
        # Verify it has the correct structure
        with open(non_existent_path, 'r', encoding='utf-8') as file:
            content = file.read()
            self.assertIn('chat_id,username,enabled,created_date,notes', content)
            self.assertIn('# Add your Telegram chat IDs here', content)
    
    def test_user_management(self):
        """Test add/disable/enable user functions."""
        user_manager = UserManager(self.csv_path)
        
        # Test getting active users
        active_users = user_manager.get_active_users()
        self.assertEqual(len(active_users), 2)  # Only enabled users
        
        # Test adding new user
        success = user_manager.add_user('111222333', 'new_user', True, 'Test addition')
        self.assertTrue(success)
        
        # Verify user was added
        active_users = user_manager.get_active_users()
        self.assertEqual(len(active_users), 3)
        
        # Test disabling user
        success = user_manager.disable_user('111222333')
        self.assertTrue(success)
        
        # Verify user was disabled
        active_users = user_manager.get_active_users()
        self.assertEqual(len(active_users), 2)
        
        # Test enabling user
        success = user_manager.enable_user('111222333')
        self.assertTrue(success)
        
        # Verify user was enabled
        active_users = user_manager.get_active_users()
        self.assertEqual(len(active_users), 3)
    
    def test_invalid_credentials(self):
        """Test handling of invalid bot token."""
        with patch.dict(os.environ, {'BOT_TOKEN': ''}):
            poster = TelegramPoster()
            result = poster.send_message("Test message")
            
            self.assertFalse(result['success'])
            self.assertEqual(result['sent_count'], 0)
            self.assertTrue(any('Configuration error' in error for error in result['errors']))
    
    @patch.dict(os.environ, {'BOT_TOKEN': '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11'})
    def test_telegram_disabled(self):
        """Test behavior when Telegram is disabled."""
        with patch.dict(os.environ, {'TELEGRAM_ENABLED': 'false'}):
            poster = TelegramPoster()
            result = poster.send_message("Test message")
            
            self.assertTrue(result['success'])  # Should succeed but do nothing
            self.assertEqual(result['sent_count'], 0)
            self.assertTrue(any('disabled' in error for error in result['errors']))
    
    @patch.dict(os.environ, {
        'BOT_TOKEN': '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11',
        'TELEGRAM_ENABLED': 'true'
    })
    @patch('atoms.telegram.telegram_post.requests.post')
    def test_rate_limiting(self, mock_post):
        """Test handling of rate limiting."""
        # Mock rate limit response followed by success
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {'Retry-After': '1'}
        
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {'ok': True}
        
        mock_post.side_effect = [rate_limit_response, success_response] * 2
        
        poster = TelegramPoster()
        poster.user_manager.csv_path = self.csv_path
        
        with patch('time.sleep'):  # Speed up test
            result = poster.send_message("Test message")
        
        # Should eventually succeed after retry
        self.assertTrue(result['success'])
    
    @patch.dict(os.environ, {
        'BOT_TOKEN': '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11',
        'TELEGRAM_ENABLED': 'true'
    })
    @patch('atoms.telegram.telegram_post.requests.post')
    def test_send_alert_formatting(self, mock_post):
        """Test alert message formatting."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'ok': True}
        mock_post.return_value = mock_response
        
        poster = TelegramPoster()
        poster.user_manager.csv_path = self.csv_path
        
        # Test different alert levels
        test_cases = [
            ('info', 'ℹ️'),
            ('warning', '⚠️'),
            ('error', '❌'),
            ('success', '✅')
        ]
        
        for level, expected_icon in test_cases:
            with patch('atoms.telegram.telegram_post.TelegramPoster.send_message') as mock_send:
                mock_send.return_value = {'success': True}
                poster.send_alert("Test alert", level)
                
                # Verify the message was formatted with correct icon
                called_message = mock_send.call_args[0][0]
                self.assertTrue(called_message.startswith(expected_icon))

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    @patch('atoms.telegram.telegram_post.TelegramPoster')
    def test_send_message_function(self, mock_poster_class):
        """Test send_message convenience function."""
        mock_poster = MagicMock()
        mock_poster.send_message.return_value = {'success': True}
        mock_poster_class.return_value = mock_poster
        
        result = send_message("Test message")
        
        self.assertEqual(result, {'success': True})
        mock_poster.send_message.assert_called_once_with("Test message", False)
    
    @patch('atoms.telegram.telegram_post.TelegramPoster')
    def test_send_alert_function(self, mock_poster_class):
        """Test send_alert convenience function."""
        mock_poster = MagicMock()
        mock_poster.send_alert.return_value = True
        mock_poster_class.return_value = mock_poster
        
        result = send_alert("Test alert", "error")
        
        self.assertTrue(result)
        mock_poster.send_alert.assert_called_once_with("Test alert", "error", None)

if __name__ == '__main__':
    unittest.main()