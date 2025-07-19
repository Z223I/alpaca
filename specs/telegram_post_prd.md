# Telegram Post Atom - Product Requirements Document (PRD)

## 1. Problem Statement

The alpaca trading system needs a simple, reliable way to send notifications and alerts to multiple Telegram users. Currently, there's no integrated messaging capability, limiting the ability to:
- Notify users of trade executions
- Send portfolio updates
- Alert on system events or errors
- Broadcast important trading signals

## 2. Solution Overview

Create a lightweight Telegram posting atom that accepts string messages and broadcasts them to a configured list of Telegram users. The solution will use a CSV-based user management system and secure credential handling via environment variables.

## 3. Functional Requirements

### 3.1 Core Functionality
- **FR-001**: Accept a string message as input parameter
- **FR-002**: Send message to all users listed in CSV file
- **FR-003**: Load Telegram bot credentials from environment variables
- **FR-004**: Read user list from CSV file in root directory
- **FR-005**: Handle messaging errors gracefully without crashing

### 3.2 User Management
- **FR-006**: CSV file must start with "." (hidden file)
- **FR-007**: CSV file must be located in project root directory
- **FR-008**: CSV file must be excluded from git tracking
- **FR-009**: Support user enable/disable functionality
- **FR-010**: Support user-specific message preferences (optional)

### 3.3 Security & Privacy
- **FR-011**: Never log or expose Telegram bot token
- **FR-012**: Validate user permissions before sending messages
- **FR-013**: Handle unauthorized access attempts gracefully

## 4. Technical Requirements

### 4.1 Dependencies
```
python-telegram-bot>=20.0
python-dotenv>=1.0.0
pandas>=1.5.0  # For CSV handling
requests>=2.28.0
```

### 4.2 Environment Configuration
```env
# Required in .env file
BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11

# Optional
TELEGRAM_TIMEOUT=30
TELEGRAM_PARSE_MODE=HTML
TELEGRAM_DISABLE_NOTIFICATION=false
```

### 4.3 CSV File Structure
**Filename**: `.telegram_users.csv`
**Location**: Project root directory
**Format**:
```csv
chat_id,username,enabled,created_date,notes
-1001234567890,trading_group,true,2024-01-15,Main trading alerts
123456789,john_doe,true,2024-01-15,Portfolio notifications
987654321,jane_smith,false,2024-01-16,Disabled temporarily
```

### 4.4 File System Requirements
- CSV file must be added to `.gitignore`
- Atom must handle missing CSV file gracefully
- Support CSV file creation with default structure

## 5. User Stories

### 5.1 Bot Setup Process
1. **Message @BotFather** on Telegram to create a new bot
2. **Send `/newbot`** command and follow prompts
3. **Copy the bot token** (format: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)
4. **Add token to .env** file as `BOT_TOKEN=your_token_here`
5. **Configure CSV file** with user chat IDs

### 5.2 Developer Stories
- **As a developer**, I want to send a simple string message so that I can notify users of system events
- **As a developer**, I want to add/remove users via CSV so that I can manage the notification list easily
- **As a developer**, I want error handling so that failed notifications don't break my application

### 5.3 End User Stories
- **As a trader**, I want to receive notifications on Telegram so that I can stay informed of my trades
- **As a trader**, I want to opt-out of notifications so that I can control my message frequency
- **As a system admin**, I want to manage user access so that I can control who receives sensitive trading information

## 6. API Design

### 6.1 Primary Function
```python
def send_message(message: str, urgent: bool = False) -> dict:
    """
    Send message to all enabled users in CSV file.
    
    Args:
        message (str): Message content to send
        urgent (bool): If True, ignore user notification preferences
        
    Returns:
        dict: {
            'success': bool,
            'sent_count': int,
            'failed_count': int,
            'errors': list
        }
    """
```

### 6.2 User Management Functions
```python
def add_user(chat_id: str, username: str = "", enabled: bool = True) -> bool:
    """Add new user to CSV file."""

def disable_user(chat_id: str) -> bool:
    """Disable user without removing from CSV."""

def get_active_users() -> list:
    """Return list of enabled users."""
```

## 7. Implementation Details

### 7.1 File Structure
```
atoms/
└── telegram/
    ├── __init__.py
    ├── telegram_post.py      # Main implementation
    ├── user_manager.py       # CSV user management
    └── config.py             # Configuration handling

tests/
└── test_telegram_post.py    # Unit tests

.telegram_users.csv           # User database (root dir)
```

### 7.2 Error Handling Strategy
- **Network Errors**: Retry up to 3 times with exponential backoff
- **Invalid Users**: Log warning, continue with valid users
- **Missing CSV**: Create default CSV with instructions
- **Invalid Credentials**: Raise configuration error immediately

### 7.3 CSV File Management
```python
# Default CSV creation if file doesn't exist
DEFAULT_CSV_CONTENT = """chat_id,username,enabled,created_date,notes
# Add your Telegram chat IDs here
# To get your chat ID, message @userinfobot on Telegram
# Group chat IDs are negative numbers (start with -)
# Example: -1001234567890,my_group,true,2024-01-15,Trading alerts
"""
```

## 8. Security Considerations

### 8.1 Credential Protection
- Use `python-dotenv` for environment variable loading
- Validate bot token format before API calls
- Never include credentials in logs or error messages

### 8.2 User Validation
- Validate chat_id format (numeric, correct length)
- Handle rate limiting from Telegram API
- Implement user permission checks

### 8.3 Data Protection
- CSV file excluded from version control
- No PII stored beyond necessary chat identifiers
- Option to encrypt CSV file (future enhancement)

## 9. Testing Requirements

### 9.1 Unit Tests
```python
class TestTelegramPost(unittest.TestCase):
    def test_send_message_success(self):
        """Test successful message sending to valid users."""
        
    def test_send_message_with_failures(self):
        """Test handling of partial failures."""
        
    def test_csv_file_creation(self):
        """Test automatic CSV creation when missing."""
        
    def test_user_management(self):
        """Test add/disable/enable user functions."""
        
    def test_invalid_credentials(self):
        """Test handling of invalid bot token."""
```

### 9.2 Integration Tests
- Test with real Telegram bot (test environment)
- Verify CSV file reading/writing
- Test rate limiting behavior
- Test network failure scenarios

## 10. Configuration Files

### 10.1 .gitignore Addition
```
# Telegram user database
.telegram_users.csv
```

### 10.2 Sample .env
```env
# Telegram Configuration
BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_TIMEOUT=30
TELEGRAM_PARSE_MODE=HTML
TELEGRAM_DISABLE_NOTIFICATION=false
```

## 11. Usage Examples

### 11.1 Basic Usage
```python
from atoms.telegram.telegram_post import send_message

# Send simple notification
result = send_message("Trade executed: AAPL +10 shares @ $150.25")
print(f"Sent to {result['sent_count']} users")

# Send urgent alert
result = send_message("SYSTEM ERROR: Trading halted", urgent=True)
```

### 11.2 Integration with Existing Code
```python
# In alpaca.py _buy method
if submit and order_filled:
    from atoms.telegram.telegram_post import send_message
    message = f"✅ BUY order filled: {symbol} {quantity} shares @ ${price:.2f}"
    send_message(message)
```

## 12. Success Metrics

### 12.1 Technical Metrics
- **Message Delivery Rate**: >95% successful delivery
- **Response Time**: <2 seconds per message batch
- **Error Rate**: <1% system errors
- **Uptime**: 99.9% availability

### 12.2 User Experience Metrics
- **Setup Time**: <5 minutes to configure and send first message
- **User Management**: <30 seconds to add/remove users
- **Error Recovery**: Automatic retry without user intervention

## 13. Future Enhancements

### 13.1 Phase 2 Features
- **Message Templates**: Pre-defined message formats
- **Scheduling**: Delayed message sending
- **Rich Media**: Support for images and charts
- **Two-way Communication**: Handle user responses

### 13.2 Phase 3 Features
- **Database Migration**: Move from CSV to SQLite
- **Web Interface**: GUI for user management
- **Analytics**: Message delivery statistics
- **Multi-language**: Internationalization support

## 14. Risk Assessment

### 14.1 Technical Risks
- **Telegram API Changes**: Monitor API updates, implement versioning
- **Rate Limiting**: Implement queue system for high-volume usage
- **Security Breach**: Encrypt CSV file, implement access logging

### 14.2 Mitigation Strategies
- Use official Telegram libraries with active maintenance
- Implement comprehensive error handling and logging
- Regular security audits of credential handling
- Backup and recovery procedures for user data

## 15. Definition of Done

### 15.1 Implementation Complete
- [ ] Core send_message function implemented and tested
- [ ] CSV user management working
- [ ] Environment variable configuration functional
- [ ] Error handling implemented
- [ ] Unit tests passing with >90% coverage

### 15.2 Documentation Complete
- [ ] Code comments and docstrings
- [ ] Usage examples in README
- [ ] Configuration guide
- [ ] Troubleshooting guide

### 15.3 Security Validated
- [ ] Credentials properly secured
- [ ] CSV file excluded from git
- [ ] Error messages don't expose sensitive data
- [ ] Rate limiting tested and functional