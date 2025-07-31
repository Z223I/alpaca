# Telegram Alpaca Integration PRD

## Product Overview

This PRD defines the integration between the existing Telegram polling service (`molecules/telegram_polling.py`) and the Alpaca trading CLI (`code/alpaca.py`) to enable trading commands through Telegram messages.

## Scope

Integration of Alpaca trading functionality into the Telegram bot with restricted access to a single authorized user ("Bruce") using a specific trigger word system.

## Requirements Analysis

### 1. Current Alpaca CLI Analysis

**File**: `code/alpaca.py` with argument parsing in `atoms/api/parse_args.py`

**Available Commands for Integration**:

#### Display Commands (Read-Only)
- `--positions`: Display current positions only
- `--cash`: Display cash balance only  
- `--active-order`: Display active orders only

#### Trading Commands
- `--buy`: Execute a buy order
  - Required: `--symbol` and either `--take-profit` or `--calc-take-profit`
  - Optional: `--stop-loss`, `--amount`, `--after-hours`, `--custom-limit-price`, `--submit`

**Command Structure**:
```python
# Display examples
python3 code/alpaca.py --positions
python3 code/alpaca.py --cash
python3 code/alpaca.py --active-order

# Buy examples  
python3 code/alpaca.py --buy --symbol AAPL --take-profit 155.0 --submit
python3 code/alpaca.py --buy --symbol AAPL --stop-loss 145.0 --calc-take-profit --amount 1000 --submit
```

### 2. Current Telegram Infrastructure Analysis

**File**: `molecules/telegram_polling.py`

**Key Components**:
- `TelegramPollingService` class with polling loop
- Command handlers in `command_handlers` dict
- User identification via `chat_id`, `username`, `first_name`
- Response system via `_send_response()`
- Integration with `UserManager` for access control

**Current Message Processing Flow**:
1. Poll updates from Telegram API
2. Process each message in `_process_update()`
3. Route commands to specific handlers
4. Send responses back to users

## Integration Requirements

### 1. Trigger Word System
- **Trigger Word**: `57chevy` (case insensitive)
- **Position**: Must be the first word of the message
- **Format**: `57chevy [alpaca_args]`

### 2. User Authorization
- **Authorized User**: Only user "Bruce" 
- **Access Control**: Messages only processed for Bruce
- **Response Restriction**: Only Bruce receives Alpaca output

### 3. Command Processing
- **Argument Source**: Only use arguments from Telegram message
- **Validation**: All argument validation performed by `alpaca.py`
- **Execution**: Implied - no need to specify execution command

### 4. Available Commands

#### Display Commands
```
57chevy --positions
57chevy --cash  
57chevy --active-order
```

#### Buy Commands
```
57chevy --buy --symbol AAPL --take-profit 155.0 --submit
57chevy --buy --symbol AAPL --stop-loss 145.0 --calc-take-profit --amount 1000
57chevy --buy --symbol AAPL --take-profit 155.0 --after-hours --custom-limit-price 150.5 --submit
```

## Technical Design

### 1. Message Processing Enhancement

**Add to `TelegramPollingService.__init__()`**:
```python
# Add alpaca trigger handler
self.command_handlers['57chevy'] = self._handle_alpaca_command
self.authorized_users = ['Bruce']  # Configurable authorized users
```

**New Handler Method**:
```python
def _handle_alpaca_command(self, chat_id: str, username: str, first_name: str, last_name: str, text: str):
    """Handle 57chevy trigger for Alpaca trading commands."""
    
    # Check user authorization
    if not self._is_authorized_user(username, first_name):
        return  # Silently ignore unauthorized users
    
    # Parse alpaca arguments from message
    args = self._parse_alpaca_args(text)
    
    # Execute alpaca command
    result = self._execute_alpaca_command(args)
    
    # Send result back to authorized user
    self._send_response(chat_id, result)
```

### 2. User Authorization System

```python
def _is_authorized_user(self, username: str, first_name: str) -> bool:
    """Check if user is authorized for Alpaca commands."""
    return (username and username.lower() == 'bruce') or \
           (first_name and first_name.lower() == 'bruce')
```

### 3. Argument Parsing

```python
def _parse_alpaca_args(self, text: str) -> List[str]:
    """Parse Alpaca arguments from Telegram message."""
    # Remove trigger word and extract arguments
    parts = text.strip().split()[1:]  # Skip '57chevy'
    return parts
```

### 4. Command Execution Integration

```python
def _execute_alpaca_command(self, args: List[str]) -> str:
    """Execute Alpaca command and return output."""
    try:
        # Import alpaca module
        from code.alpaca import execMain
        
        # Capture stdout
        import io
        import sys
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Execute alpaca command
        exit_code = execMain(args)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Get output
        output = captured_output.getvalue()
        
        if exit_code == 0:
            return f"✅ Command executed successfully:\n```\n{output}\n```"
        else:
            return f"❌ Command failed (exit code: {exit_code}):\n```\n{output}\n```"
            
    except Exception as e:
        return f"❌ Error executing command: {str(e)}"
```

### 5. Message Processing Logic Update

**Update `_process_update()` method**:
```python
def _process_update(self, update: Dict):
    """Process a single update from Telegram."""
    try:
        message = update.get('message', {})
        if not message:
            return
        
        # ... existing code ...
        
        # Process commands and trigger words
        if text.startswith('/'):
            command = text.split()[0].lower()
            if command in self.command_handlers:
                self.command_handlers[command](chat_id, username, first_name, last_name, text)
            else:
                self._handle_unknown_command(chat_id, text)
        elif text.lower().startswith('57chevy'):
            # Handle alpaca trigger word
            self._handle_alpaca_command(chat_id, username, first_name, last_name, text)
        else:
            # Handle non-command messages
            self._handle_regular_message(chat_id, username, first_name, text)
            
    except Exception as e:
        self._log(f"Error processing update: {e}", "ERROR")
```

## Implementation Details

### 1. File Modifications Required

**Primary File**: `molecules/telegram_polling.py`

**Additions**:
1. New command handler for `57chevy` trigger
2. User authorization checking
3. Alpaca argument parsing
4. Command execution with output capture
5. Error handling and response formatting

### 2. Dependencies

**Import Requirements**:
```python
import sys
import io
import subprocess
from typing import List
```

**Module Integration**:
```python
# Add to sys.path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.alpaca import execMain
```

### 3. Error Handling

**Validation Errors**: Handled by `alpaca.py` argument parser
**Execution Errors**: Captured and returned to user
**Authorization Errors**: Silently ignored (no response sent)
**System Errors**: Logged and error message sent to user

### 4. Security Considerations

**User Restriction**: Hard-coded authorization for "Bruce" only
**Command Limitation**: Only specified commands available
**Output Sanitization**: Raw alpaca output returned (no modification)
**Silent Rejection**: Unauthorized users receive no response

## Testing Strategy

### 1. Unit Tests

**Test Cases**:
- Trigger word recognition (case insensitive)
- User authorization (Bruce only)
- Argument parsing from message
- Command execution and output capture
- Error handling scenarios

### 2. Integration Tests

**Test Scenarios**:
- Display commands execution
- Buy command with various argument combinations
- Unauthorized user message handling
- Invalid argument handling
- Error condition responses

### 3. Manual Testing

**Test Messages**:
```
57chevy --positions
57CHEVY --cash
57Chevy --active-order
57chevy --buy --symbol AAPL --take-profit 155.0 --submit
```

## Configuration

### 1. Authorized Users

**Current**: Hard-coded "Bruce"
**Future**: Configurable via environment variable or config file

### 2. Alpaca Environment

**Inherits**: All existing Alpaca configuration from `.env`
**Integration**: Uses existing `alpaca.py` with all its settings

## Deployment

### 1. Code Changes

**Single File**: `molecules/telegram_polling.py`
**Backwards Compatible**: No breaking changes to existing functionality

### 2. Testing Requirements

**Environment**: Use paper trading environment for testing
**Verification**: Ensure existing Telegram functionality remains intact

## Success Criteria

### 1. Functional Requirements
- ✅ Trigger word `57chevy` recognized (case insensitive)
- ✅ Only user "Bruce" receives responses
- ✅ Display commands return current account information
- ✅ Buy commands execute with proper validation
- ✅ All argument validation handled by alpaca.py
- ✅ Raw alpaca output returned to user

### 2. Security Requirements
- ✅ Unauthorized users silently ignored
- ✅ No information leakage to unauthorized users
- ✅ Command execution restricted to specified operations

### 3. Integration Requirements
- ✅ Existing Telegram functionality preserved
- ✅ Alpaca CLI functionality unchanged
- ✅ Error handling maintains system stability

## Future Enhancements

### 1. Multi-User Support
- Configurable authorized user list
- Role-based access control
- User-specific command restrictions

### 2. Enhanced Security
- Token-based authentication
- Command logging and audit trail
- Rate limiting per user

### 3. Extended Command Set
- Additional Alpaca CLI commands
- Custom command shortcuts
- Batch command execution

## Implementation Timeline

**Phase 1**: Core integration (1-2 days)
- Trigger word processing
- User authorization
- Basic command execution

**Phase 2**: Testing and refinement (1 day)
- Unit and integration tests
- Error handling validation
- Security verification

**Phase 3**: Documentation and deployment (0.5 day)
- Implementation documentation
- Deployment procedures
- User testing guidelines