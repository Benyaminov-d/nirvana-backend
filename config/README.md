# Configuration Management - Centralized System

## Overview

This directory contains the new centralized configuration system for Nirvana App, replacing scattered configuration logic throughout the codebase.

## Structure

```
backend/config/
├── __init__.py           # Main entry point and configuration factory
├── base.py               # Base configuration class with all settings
├── development.py        # Development environment configuration
├── production.py         # Production environment configuration  
├── testing.py           # Testing environment configuration
├── schema.py            # Configuration validation and schema
└── README.md            # This file
```

## Usage

### Basic Usage

```python
from backend.config import get_config

# Auto-detect environment from NIRVANA_ENV
config = get_config()

# Explicit environment
config = get_config('production')

# Access configuration sections
print(config.database.host)
print(config.cvars.min_years)
print(config.compass.lambda_param)
```

### Configuration Sections

**Database Configuration** (`config.database`)
- Connection settings (host, port, credentials)
- Environment-specific database names

**CVaR Configuration** (`config.cvars`)
- Calculation parameters (min_years, trading_days, sims)
- Service mode (local/remote) and URLs
- Data validation settings

**Compass Configuration** (`config.compass`)
- Scoring parameters (lambda, anchors, thresholds)
- Calibration settings

**Authentication** (`config.auth`)
- Basic auth credentials
- CORS settings
- Session configuration

**External Services** (`config.external_services`)
- EODHD API configuration
- OpenAI Assistant settings
- Email/SMTP configuration
- Azure Service Bus settings

**Workers** (`config.workers`)
- Concurrency settings
- Batch processing parameters

## Environment-Specific Configurations

### Development
- Debug mode enabled
- Relaxed validation
- Extensive CORS origins
- Most startup tasks enabled
- Fast iteration settings

### Production
- Debug mode disabled
- Strict validation and security
- Limited CORS origins
- Optimized worker counts
- All production features enabled

### Testing
- Minimal configuration for speed
- Most startup tasks disabled
- Short timeouts
- Fast simulations (100 instead of 10,000)

## Environment Variables

Set `NIRVANA_ENV` to control which configuration is loaded:
- `development` or `dev` → DevelopmentConfig
- `production` or `prod` → ProductionConfig
- `testing` or `test` → TestingConfig

## Migration from Old System

**Before:**
```python
from backend.settings import get_settings
from backend.shared.constants import EnvKeys

settings = get_settings()
api_key = os.getenv(EnvKeys.EODHD_API_KEY)
```

**After:**
```python
from backend.config import get_config

config = get_config()
api_key = config.external_services.eodhd_api_key
```

## Validation

The configuration system includes built-in validation:

```python
config = get_config('production')
errors = config.validate()

if errors:
    for error in errors:
        print(f"Configuration error: {error}")
```

## Schema and Documentation

Generate configuration schema:

```python
from backend.config.schema import generate_config_schema

schema = generate_config_schema()
```

Generate .env templates:

```python
from backend.config.schema import generate_env_file_template

dev_template = generate_env_file_template('development')
prod_template = generate_env_file_template('production')
```

## Benefits of Centralized Configuration

1. **Single Source of Truth**: All configuration in one place
2. **Environment-Specific**: Different settings per environment
3. **Type Safety**: Strongly typed configuration classes
4. **Validation**: Built-in validation with helpful error messages
5. **Documentation**: Self-documenting with schema generation
6. **IDE Support**: Full IntelliSense support for configuration
7. **Testing**: Easy to override configuration for tests

## Migration Status

✅ **Consolidated Settings From:**
- `backend/settings.py` (Settings dataclass)
- `backend/shared/constants.py` (EnvKeys class)
- `backend/core/compass_config.py` (Compass configuration)
- Scattered `os.getenv()` calls throughout codebase
- Various config files and environment variable reads

✅ **Features:**
- Environment-specific configurations
- Comprehensive validation
- Schema documentation
- .env file template generation
- Backward compatibility support

**Status**: ✅ Completed - Ready for integration

