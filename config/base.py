"""
Base configuration class with all application settings.

This module centralizes all configuration logic previously scattered across:
- settings.py
- shared/constants.py (EnvKeys)
- core/compass_config.py
- Various environment variable reads throughout the codebase
"""

import os
from dataclasses import dataclass
from typing import Optional, List


def _get_bool(name: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    v = val.strip().lower()
    return v in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    """Parse integer environment variable."""
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _get_float(name: str, default: float) -> float:
    """Parse float environment variable."""
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _get_list(name: str, default: List[str] = None, separator: str = ",") -> List[str]:
    """Parse comma-separated list environment variable."""
    if default is None:
        default = []
    
    val = os.getenv(name, "")
    if not val.strip():
        return default
    
    return [item.strip() for item in val.split(separator) if item.strip()]


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    url: str = ""
    host: str = "localhost"
    port: int = 5432
    name: str = "nirvana"
    user: str = "postgres"
    password: str = ""
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        return cls(
            url=os.getenv("DATABASE_URL", ""),
            host=os.getenv("DB_HOST", "localhost"),
            port=_get_int("DB_PORT", 5432),
            name=os.getenv("DB_NAME", "nirvana"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )


@dataclass
class CVaRConfig:
    """CVaR calculation configuration."""
    # Data validation
    min_years: float = 10.0
    enforce_min_years: bool = True
    eq_lookback_days: int = 0
    price_field: str = "adjusted_close"
    allow_close_fallback: bool = False
    
    # Liquidity diagnostics
    zero_return_eps: float = 1e-8
    last252_max_zero_share: float = 0.2
    max_drop_last252: int = 10
    
    # Computation parameters
    sims: int = 10000
    trading_days: int = 252
    log_phase: bool = False
    
    # Service configuration
    service_mode: str = "local"  # local, remote, auto
    func_url: str = ""
    func_timeout: int = 120
    func_connect_timeout: int = 10
    
    @classmethod
    def from_env(cls) -> 'CVaRConfig':
        return cls(
            min_years=_get_float("NVAR_MIN_YEARS", 10.0),
            enforce_min_years=_get_bool("NVAR_ENFORCE_MIN_YEARS", True),
            eq_lookback_days=_get_int("NVAR_EQ_LOOKBACK_DAYS", 0),
            price_field=os.getenv("NVAR_PRICE_FIELD", "adjusted_close").strip(),
            allow_close_fallback=_get_bool("NVAR_ALLOW_CLOSE_FALLBACK", False),
            zero_return_eps=_get_float("NVAR_ZERO_RETURN_EPS", 1e-8),
            last252_max_zero_share=_get_float("NVAR_LAST252_MAX_ZERO_SHARE", 0.2),
            max_drop_last252=_get_int("NVAR_MAX_DROP_LAST252", 10),
            sims=_get_int("NVAR_SIMS", 10000),
            trading_days=_get_int("NVAR_TRADING_DAYS", 252),
            log_phase=_get_bool("NVAR_LOG_PHASE", False),
            service_mode=os.getenv("NVAR_CVAR_SERVICE", "local").strip().lower(),
            func_url=os.getenv("NVAR_FUNC_URL", "").rstrip("/"),
            func_timeout=_get_int("NVAR_FUNC_TIMEOUT", 120),
            func_connect_timeout=_get_int("NVAR_FUNC_CONNECT_TIMEOUT", 10),
        )


@dataclass
class CompassConfig:
    """Compass scoring configuration."""
    # Core parameters
    lambda_param: float = 2.25  # Loss-aversion parameter
    mu_low: float = 0.02  # Lower anchor (2% - conservative)
    mu_high: float = 0.18  # Upper anchor (18% - excellent)
    default_loss_tolerance: float = 0.25  # Default loss tolerance (25%)
    
    # Thresholds and limits
    min_score_threshold: int = 3000
    max_results: int = 10
    min_sample_size: int = 10
    default_median_mu: float = 0.08  # Default median return (8%)
    
    # Calibration parameters
    winsor_p_low: float = 0.01  # Lower winsorization percentile
    winsor_p_high: float = 0.99  # Upper winsorization percentile
    anchor_hd_low: float = 0.05  # Lower Harrell-Davis percentile
    anchor_hd_high: float = 0.95  # Upper Harrell-Davis percentile
    anchor_min_spread: float = 0.02  # Minimum anchor spread (2%)
    anchor_max_spread: float = 0.60  # Maximum anchor spread (60%)
    
    @classmethod
    def from_env(cls) -> 'CompassConfig':
        return cls(
            lambda_param=_get_float("COMPASS_LAMBDA", 2.25),
            mu_low=_get_float("COMPASS_MU_LOW", 0.02),
            mu_high=_get_float("COMPASS_MU_HIGH", 0.18),
            default_loss_tolerance=_get_float("COMPASS_DEFAULT_LOSS_TOLERANCE", 0.25),
            min_score_threshold=_get_int("COMPASS_MIN_SCORE_THRESHOLD", 3000),
            max_results=_get_int("COMPASS_MAX_RESULTS", 10),
            min_sample_size=_get_int("COMPASS_MIN_SAMPLE_SIZE", 10),
            default_median_mu=_get_float("COMPASS_DEFAULT_MEDIAN_MU", 0.08),
            winsor_p_low=_get_float("COMPASS_WINSOR_P_LOW", 0.01),
            winsor_p_high=_get_float("COMPASS_WINSOR_P_HIGH", 0.99),
            anchor_hd_low=_get_float("COMPASS_ANCHOR_HD_LOW", 0.05),
            anchor_hd_high=_get_float("COMPASS_ANCHOR_HD_HIGH", 0.95),
            anchor_min_spread=_get_float("COMPASS_ANCHOR_MIN_SPREAD", 0.02),
            anchor_max_spread=_get_float("COMPASS_ANCHOR_MAX_SPREAD", 0.60),
        )


@dataclass
class AuthConfig:
    """Authentication configuration."""
    basic_auth_user: str = ""
    basic_auth_pass: str = ""
    public_token_secret: str = ""
    allowed_origins: List[str] = None
    session_timeout: int = 3600  # 1 hour
    
    @classmethod
    def from_env(cls) -> 'AuthConfig':
        return cls(
            basic_auth_user=os.getenv("BASIC_AUTH_USER", ""),
            basic_auth_pass=os.getenv("BASIC_AUTH_PASS", ""),
            public_token_secret=os.getenv("NIR_PUBLIC_TOKEN_SECRET", ""),
            allowed_origins=_get_list("NIR_ALLOWED_ORIGINS", ["*"]),
            session_timeout=_get_int("SESSION_TIMEOUT", 3600),
        )


@dataclass
class RedisConfig:
    """Redis configuration for caching and task queues."""
    url: str = "redis://localhost:6379/0"
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    
    # Caching settings
    cache_ttl: int = 3600  # Default TTL for cached items (1 hour)
    eodhd_cache_ttl: int = 86400  # EODHD API cache TTL (24 hours)
    max_connections: int = 10
    
    # Task queue settings  
    queue_name: str = "eodhd_queue"
    worker_timeout: int = 600  # 10 minutes
    job_timeout: int = 300  # 5 minutes
    
    # Rate limiting
    rate_limit_requests: int = 100  # Max requests per minute to EODHD
    rate_limit_window: int = 60  # Rate limit window in seconds
    
    @classmethod
    def from_env(cls) -> 'RedisConfig':
        return cls(
            url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            host=os.getenv("REDIS_HOST", "localhost"),
            port=_get_int("REDIS_PORT", 6379),
            db=_get_int("REDIS_DB", 0),
            password=os.getenv("REDIS_PASSWORD", ""),
            cache_ttl=_get_int("REDIS_CACHE_TTL", 3600),
            eodhd_cache_ttl=_get_int("REDIS_EODHD_CACHE_TTL", 86400),
            max_connections=_get_int("REDIS_MAX_CONNECTIONS", 10),
            queue_name=os.getenv("REDIS_QUEUE_NAME", "eodhd_queue"),
            worker_timeout=_get_int("REDIS_WORKER_TIMEOUT", 600),
            job_timeout=_get_int("REDIS_JOB_TIMEOUT", 300),
            rate_limit_requests=_get_int("REDIS_RATE_LIMIT_REQUESTS", 100),
            rate_limit_window=_get_int("REDIS_RATE_LIMIT_WINDOW", 60),
        )


@dataclass
class ExternalServiceConfig:
    """External service configuration."""
    # EODHD API
    eodhd_api_key: str = ""
    eodhd_base_url: str = "https://eodhd.com"
    eodhd_timeout: int = 60
    eodhd_connect_timeout: int = 30
    eodhd_retries: int = 3
    
    # OpenAI Assistant
    openai_api_key: str = ""
    openai_assistant_id: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_run_timeout: int = 30
    assistant_system_prompt: str = ""
    
    # Email configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    
    # Service Bus
    sb_connection_string: str = ""
    sb_queue_name: str = ""
    
    @classmethod
    def from_env(cls) -> 'ExternalServiceConfig':
        return cls(
            eodhd_api_key=os.getenv("EODHD_API_KEY", ""),
            eodhd_base_url=os.getenv("EODHD_BASE_URL", "https://eodhd.com"),
            eodhd_timeout=_get_int("EODHD_TIMEOUT", 60),
            eodhd_connect_timeout=_get_int("EODHD_CONNECT_TIMEOUT", 30),
            eodhd_retries=_get_int("EODHD_RETRIES", 3),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_assistant_id=os.getenv("OPENAI_ASSISTANT_ID", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_run_timeout=_get_int("OPENAI_RUN_TIMEOUT_SEC", 30),
            assistant_system_prompt=os.getenv("ASSISTANT_SYSTEM_PROMPT", ""),
            smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=_get_int("SMTP_PORT", 587),
            sb_connection_string=os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING", ""),
            sb_queue_name=os.getenv("AZURE_SERVICE_BUS_QUEUE_NAME", ""),
        )


@dataclass
class WorkerConfig:
    """Worker and concurrency configuration."""
    # Experiment workers
    reprocess_workers: int = 8
    validate_workers: int = 8
    validate_years: int = 25
    warm_workers: int = 4
    
    # Batch processing
    batch_size: int = 100
    max_concurrent_requests: int = 50
    request_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'WorkerConfig':
        return cls(
            reprocess_workers=_get_int("EXP_REPROCESS_WORKERS", 8),
            validate_workers=_get_int("EXP_VALIDATE_WORKERS", 8),
            validate_years=_get_int("EXP_VALIDATE_YEARS", 25),
            warm_workers=_get_int("NVAR_WARM_WORKERS", 4),
            batch_size=_get_int("BATCH_SIZE", 100),
            max_concurrent_requests=_get_int("MAX_CONCURRENT_REQUESTS", 50),
            request_timeout=_get_int("REQUEST_TIMEOUT", 30),
        )


class BaseConfig:
    """
    Base configuration class that consolidates all application settings.
    
    This replaces the scattered configuration throughout the codebase and provides
    a single source of truth for all application settings.
    """
    
    def __init__(self):
        # Core app configuration
        self.app_name: str = "Nirvana App"
        self.app_version: str = "1.0.0"
        self.debug: bool = _get_bool("DEBUG", False)
        self.environment: str = os.getenv("NIRVANA_ENV", "development")
        self.license_mode: str = os.getenv("NVAR_LICENSE", "DEV")
        
        # Configuration groups
        self.database = DatabaseConfig.from_env()
        self.redis = RedisConfig.from_env()
        self.cvars = CVaRConfig.from_env()
        self.compass = CompassConfig.from_env()
        self.auth = AuthConfig.from_env()
        self.external_services = ExternalServiceConfig.from_env()
        self.workers = WorkerConfig.from_env()
        
        # Initialize environment-specific settings
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup environment-specific configuration. Override in subclasses."""
        pass
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ("development", "dev")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ("production", "prod")
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment.lower() in ("testing", "test")
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of validation errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate required EODHD API key for production
        if self.is_production and not self.external_services.eodhd_api_key:
            errors.append("EODHD_API_KEY is required in production")
        
        # Validate database configuration
        if self.is_production and not (self.database.url or (self.database.host and self.database.name)):
            errors.append("Database configuration is incomplete for production")
        
        # Validate authentication in production
        if self.is_production and not (self.auth.basic_auth_user and self.auth.basic_auth_pass):
            if not self.auth.public_token_secret:
                errors.append("Authentication must be configured in production")
        
        return errors
