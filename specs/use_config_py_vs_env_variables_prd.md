# Product Requirements Document: Config File Migration

## Overview

This PRD outlines the migration from environment variable-based configuration to a Python dataclass-based configuration system for the Alpaca trading API wrapper.

## Background

The current system uses environment variables loaded from `.env` files to configure API credentials and trading parameters. This approach has limitations in terms of multi-provider support, environment management, and configuration validation.

## Current State Analysis

### Environment Variables Currently Used

1. **API Credentials** (in `atoms/api/init_alpaca_client.py:12-15`):
   - `ALPACA_API_KEY`: API key for authentication
   - `ALPACA_SECRET_KEY`: Secret key for authentication  
   - `ALPACA_BASE_URL`: Base URL for API endpoint (paper vs live)

2. **Trading Parameters** (in `code/alpaca.py:61`):
   - `PORTFOLIO_RISK`: Portfolio risk percentage (default 0.10)

3. **Environment Loading** (in `code/alpaca.py:30-36`):
   - Uses `python-dotenv` to load from `.env` file
   - Falls back to system environment variables

## Requirements

### Functional Requirements

#### FR1: Configuration File Structure
- **Requirement**: Create `./code/.alpaca_config.py` with dataclass-based configuration
- **Structure**: Support hierarchical configuration with Provider > Account > Environment structure
- **Environments**: Support `paper`, `live`, and `cash` environments
- **Provider Support**: Extensible to support multiple trading providers

#### FR2: Configuration Schema
The configuration must follow this JSON structure:
```json
{
  "Provider": {
    "<provider_name>": {
      "<account_name>": {
        "paper": {
          "APP Key": "<APP Key>",
          "APP Secret": "<APP Secret>",
          "URL": "<URL>"
        },
        "live": {
          "APP Key": "<APP Key>",
          "APP Secret": "<APP Secret>",
          "URL": "<URL>"
        },
        "cash": {
          "APP Key": "<APP Key>",
          "APP Secret": "<APP Secret>",
          "URL": "<URL>"
        }
      }
    }
  }
}
```

#### FR3: Code Migration
- **Requirement**: Update `code/alpaca.py` to use config file instead of environment variables
- **Requirement**: Update `atoms/api/init_alpaca_client.py` to use config file
- **Requirement**: Maintain backward compatibility during transition period

#### FR4: Security
- **Requirement**: Add `./code/.alpaca_config.py` to `.gitignore` to prevent credential exposure
- **Requirement**: Implement proper error handling for missing/invalid configuration

### Non-Functional Requirements

#### NFR1: Performance
- Configuration loading must not significantly impact application startup time
- Configuration should be loaded once and cached for the session

#### NFR2: Maintainability
- Configuration structure must be easily extensible for future providers
- Code changes must follow existing patterns and conventions
- Clear error messages for configuration issues

#### NFR3: Standards Compliance
- Must pass existing linting rules (`flake8`)
- Must maintain VS Code integration compatibility
- Must pass all existing tests
- Must maintain existing CLI interface

## Implementation Plan

### Phase 1: Infrastructure Setup
1. **Create Configuration Dataclass**
   - Design dataclass structure in `./code/.alpaca_config.py`
   - Implement configuration loading and validation
   - Add error handling for missing/malformed config

2. **Update .gitignore**
   - Add `./code/.alpaca_config.py` to prevent credential commits

### Phase 2: Code Migration
1. **Update API Client Initialization**
   - Modify `atoms/api/init_alpaca_client.py` to use config file
   - Maintain same function signature for compatibility

2. **Update Main Application**
   - Modify `code/alpaca.py` to use config file
   - Remove environment variable dependencies
   - Update portfolio risk configuration

### Phase 3: Testing and Validation
1. **Unit Testing**
   - Run existing test suite: `~/miniconda3/envs/alpaca/bin/python -m pytest tests/ -v`
   - Ensure all 26 after-hours tests pass
   - Verify ORB functionality tests

2. **Linting Compliance**
   - Run `flake8 code/ atoms/` to ensure compliance
   - Fix any style issues

3. **Integration Testing**
   - Verify CLI interface remains unchanged
   - Test both dry-run and live order execution paths

## Technical Specifications

### Configuration Dataclass Design

```python
@dataclass
class EnvironmentConfig:
    app_key: str
    app_secret: str
    url: str

@dataclass
class AccountConfig:
    paper: EnvironmentConfig
    live: EnvironmentConfig
    cash: EnvironmentConfig

@dataclass
class ProviderConfig:
    accounts: Dict[str, AccountConfig]

@dataclass
class AlpacaConfig:
    providers: Dict[str, ProviderConfig]
    portfolio_risk: float = 0.10
```

### Migration Strategy
- Implement config loading with fallback to environment variables
- Add configuration validation and helpful error messages
- Provide example configuration template
- Document migration process for users

## Risk Assessment

### High Risk
- **Credential Exposure**: Config file contains sensitive data
  - *Mitigation*: Proper .gitignore configuration, user documentation

### Medium Risk
- **Breaking Changes**: API changes could break existing workflows
  - *Mitigation*: Maintain backward compatibility, thorough testing

### Low Risk
- **Performance Impact**: Config file loading overhead
  - *Mitigation*: Efficient loading, caching strategy

## Acceptance Criteria

### AC1: Configuration System
- [ ] `./code/.alpaca_config.py` is created with proper dataclass structure
- [ ] Configuration supports multiple providers and environments
- [ ] Configuration file is added to `.gitignore`

### AC2: Code Migration
- [ ] `code/alpaca.py` uses config file instead of environment variables
- [ ] `atoms/api/init_alpaca_client.py` uses config file
- [ ] All existing CLI functionality works unchanged

### AC3: Quality Assurance
- [ ] All existing tests pass (`~/miniconda3/envs/alpaca/bin/python -m pytest tests/ -v`)
- [ ] Linting passes (`flake8 code/ atoms/`)
- [ ] VS Code integration works without errors
- [ ] No credentials committed to git

### AC4: Documentation
- [ ] Configuration structure is documented
- [ ] Migration process is documented
- [ ] Error handling provides clear guidance

## Success Metrics
- Zero test failures after migration
- Zero linting violations
- No credential exposure in git history
- Successful execution of dry-run and live trading operations
- Improved configuration management flexibility

## Dependencies
- `python-dotenv` may become optional after migration
- Existing test suite must continue to pass
- Alpaca API client initialization must remain functional