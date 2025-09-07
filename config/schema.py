"""Configuration schema validation and documentation."""

from typing import Dict, Any, List
from dataclasses import fields


def generate_config_schema() -> Dict[str, Any]:
    """
    Generate a schema dictionary describing all configuration options.
    
    Returns:
        Dictionary with configuration schema for documentation
    """
    from config.base import BaseConfig, DatabaseConfig, CVaRConfig, CompassConfig
    from config.base import AuthConfig, ExternalServiceConfig, WorkerConfig
    
    schema = {
        "meta": {
            "title": "Nirvana App Configuration Schema",
            "description": "Complete configuration schema for environment variables and settings",
            "version": "1.0.0"
        },
        "environments": {
            "development": "Development environment with relaxed validation",
            "production": "Production environment with strict validation",
            "testing": "Testing environment with minimal setup"
        },
        "sections": {}
    }
    
    # Generate schema for each configuration section
    config_classes = [
        ("database", DatabaseConfig),
        ("cvars", CVaRConfig), 
        ("compass", CompassConfig),
        ("auth", AuthConfig),
        ("external_services", ExternalServiceConfig),
        ("workers", WorkerConfig)
    ]
    
    for section_name, config_class in config_classes:
        section_schema = {
            "description": config_class.__doc__ or f"{section_name.title()} configuration",
            "fields": {}
        }
        
        # Get field information from dataclass
        for field in fields(config_class):
            field_schema = {
                "type": field.type.__name__ if hasattr(field.type, '__name__') else str(field.type),
                "default": field.default if field.default is not field.default_factory else None,
                "description": f"{field.name} configuration parameter"
            }
            
            # Add environment variable mapping if it exists
            env_var = _get_env_var_for_field(section_name, field.name)
            if env_var:
                field_schema["env_var"] = env_var
                
            section_schema["fields"][field.name] = field_schema
            
        schema["sections"][section_name] = section_schema
    
    return schema


def _get_env_var_for_field(section: str, field: str) -> str:
    """Map configuration field to environment variable name."""
    # This mapping could be more sophisticated, but for now use simple rules
    env_mappings = {
        # Database fields
        "url": "DATABASE_URL",
        "host": "DB_HOST",
        "port": "DB_PORT",
        "name": "DB_NAME",
        "user": "DB_USER",
        "password": "DB_PASSWORD",
        
        # CVaR fields
        "min_years": "NVAR_MIN_YEARS",
        "enforce_min_years": "NVAR_ENFORCE_MIN_YEARS",
        "eq_lookback_days": "NVAR_EQ_LOOKBACK_DAYS",
        "price_field": "NVAR_PRICE_FIELD",
        "allow_close_fallback": "NVAR_ALLOW_CLOSE_FALLBACK",
        "zero_return_eps": "NVAR_ZERO_RETURN_EPS",
        "last252_max_zero_share": "NVAR_LAST252_MAX_ZERO_SHARE",
        "max_drop_last252": "NVAR_MAX_DROP_LAST252",
        "service_mode": "NVAR_CVAR_SERVICE",
        "func_url": "NVAR_FUNC_URL",
        "func_timeout": "NVAR_FUNC_TIMEOUT",
        
        # Auth fields
        "basic_auth_user": "BASIC_AUTH_USER",
        "basic_auth_pass": "BASIC_AUTH_PASS",
        "public_token_secret": "NIR_PUBLIC_TOKEN_SECRET",
        "allowed_origins": "NIR_ALLOWED_ORIGINS",
        
        # External services
        "eodhd_api_key": "EODHD_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "openai_assistant_id": "OPENAI_ASSISTANT_ID",
        "openai_model": "OPENAI_MODEL",
        "smtp_server": "SMTP_SERVER",
        "smtp_port": "SMTP_PORT",
        
        # Workers
        "reprocess_workers": "EXP_REPROCESS_WORKERS",
        "validate_workers": "EXP_VALIDATE_WORKERS",
        "validate_years": "EXP_VALIDATE_YEARS"
    }
    
    return env_mappings.get(field, "")


def validate_config_completeness(config) -> List[str]:
    """
    Validate that all required configuration is present.
    
    Args:
        config: Configuration instance to validate
        
    Returns:
        List of validation warnings and errors
    """
    issues = []
    
    # Run configuration's own validation
    config_errors = config.validate()
    issues.extend(config_errors)
    
    # Additional cross-section validations
    if config.cvars.service_mode == "remote" and not config.cvars.func_url:
        issues.append("NVAR_FUNC_URL is required when CVaR service mode is 'remote'")
    
    if config.external_services.openai_assistant_id and not config.external_services.openai_api_key:
        issues.append("OPENAI_API_KEY is required when using OpenAI Assistant")
        
    if config.compass.lambda_param <= 0:
        issues.append("COMPASS_LAMBDA must be positive")
        
    if config.compass.mu_low >= config.compass.mu_high:
        issues.append("COMPASS_MU_LOW must be less than COMPASS_MU_HIGH")
    
    return issues


def generate_env_file_template(environment: str = "development") -> str:
    """
    Generate a .env file template for the specified environment.
    
    Args:
        environment: Target environment (development, production, testing)
        
    Returns:
        .env file content as string
    """
    templates = {
        "development": _dev_env_template(),
        "production": _prod_env_template(),
        "testing": _test_env_template()
    }
    
    return templates.get(environment, templates["development"])


def _dev_env_template() -> str:
    """Development environment template."""
    return '''# Nirvana App - Development Environment
# Copy this file to .env and adjust values as needed

# Environment
NIRVANA_ENV=development
DEBUG=true

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nirvana_dev
DB_USER=postgres
DB_PASSWORD=

# EODHD API (optional for development)
EODHD_API_KEY=

# Authentication (optional for development)
BASIC_AUTH_USER=
BASIC_AUTH_PASS=
NIR_PUBLIC_TOKEN_SECRET=

# CVaR Configuration
NVAR_MIN_YEARS=5.0
NVAR_ENFORCE_MIN_YEARS=false
NVAR_CVAR_SERVICE=local

# OpenAI Assistant (optional)
OPENAI_API_KEY=
OPENAI_ASSISTANT_ID=

# Logging
LOG_LEVEL=DEBUG
'''


def _prod_env_template() -> str:
    """Production environment template."""
    return '''# Nirvana App - Production Environment
# SECURITY: Keep this file secure and never commit to version control

# Environment
NIRVANA_ENV=production
DEBUG=false

# Database (REQUIRED)
DATABASE_URL=postgresql://user:password@host:port/dbname
# OR individual components:
# DB_HOST=
# DB_PORT=5432
# DB_NAME=nirvana
# DB_USER=
# DB_PASSWORD=

# EODHD API (REQUIRED)
EODHD_API_KEY=your_eodhd_api_key_here

# Authentication (REQUIRED)
BASIC_AUTH_USER=your_admin_username
BASIC_AUTH_PASS=your_secure_password
NIR_PUBLIC_TOKEN_SECRET=your_jwt_secret_key_here

# CORS (REQUIRED)
NIR_ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# CVaR Configuration
NVAR_MIN_YEARS=10.0
NVAR_ENFORCE_MIN_YEARS=true
NVAR_CVAR_SERVICE=auto
NVAR_FUNC_URL=https://your-azure-function.azurewebsites.net

# Azure Service Bus (optional)
AZURE_SERVICE_BUS_CONNECTION_STRING=
AZURE_SERVICE_BUS_QUEUE_NAME=

# OpenAI Assistant (optional)
OPENAI_API_KEY=
OPENAI_ASSISTANT_ID=

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Logging
LOG_LEVEL=INFO
'''


def _test_env_template() -> str:
    """Testing environment template."""
    return '''# Nirvana App - Testing Environment
# Minimal configuration for running tests

# Environment
NIRVANA_ENV=testing
DEBUG=true

# Test Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nirvana_test
DB_USER=postgres
DB_PASSWORD=

# Minimal CVaR Configuration
NVAR_MIN_YEARS=1.0
NVAR_ENFORCE_MIN_YEARS=false
NVAR_CVAR_SERVICE=local

# Disable startup tasks for faster tests
RAG_ENABLED=false

# Logging
LOG_LEVEL=DEBUG
'''

