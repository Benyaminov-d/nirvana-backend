"""Production environment configuration."""

import os
from config.base import BaseConfig


class ProductionConfig(BaseConfig):
    """Configuration for production environment."""
    
    def _setup_environment(self):
        """Setup production-specific configuration."""
        # Disable debug mode
        self.debug = False
        
        # SAFETY CHECK: Create new CVaRConfig (never modify frozen dataclass directly!)
        # This replaces the old problematic: self.cvars.enforce_min_years = True  
        from config.base import CVaRConfig
        self.cvars = CVaRConfig(
            min_years=self.cvars.min_years,
            enforce_min_years=True,  # Strict validation in production
            eq_lookback_days=self.cvars.eq_lookback_days,
            price_field=self.cvars.price_field,
            allow_close_fallback=self.cvars.allow_close_fallback,
            zero_return_eps=self.cvars.zero_return_eps,
            last252_max_zero_share=self.cvars.last252_max_zero_share,
            max_drop_last252=self.cvars.max_drop_last252,
            sims=self.cvars.sims,
            trading_days=self.cvars.trading_days,
            log_phase=self.cvars.log_phase,
            service_mode=self.cvars.service_mode,
            func_url=self.cvars.func_url,
            func_timeout=self.cvars.func_timeout,
            func_connect_timeout=self.cvars.func_connect_timeout
        )
        
        # Production database from environment
        if not self.database.url:
            from config.base import DatabaseConfig
            self.database = DatabaseConfig(
                url=os.getenv("DATABASE_URL", ""),
                host=os.getenv("DB_HOST", ""),
                port=int(os.getenv("DB_PORT", "5432")),
                name=os.getenv("DB_NAME", "nirvana"),
                user=os.getenv("DB_USER", ""),
                password=os.getenv("DB_PASSWORD", "")
            )
        
        # Strict CORS in production
        allowed_origins = os.getenv("NIR_ALLOWED_ORIGINS", "").strip()
        if allowed_origins:
            self.auth.allowed_origins = [
                origin.strip() for origin in allowed_origins.split(",")
                if origin.strip()
            ]
        else:
            # Default secure origins for production
            self.auth.allowed_origins = []
        
        # Production logging
        os.environ.setdefault("LOG_LEVEL", "INFO")
        
        # Production worker configuration
        self._setup_production_workers()
        
        # Production startup configuration
        self._setup_production_startup()
    
    def _setup_production_workers(self):
        """Configure workers for production scale."""
        # Higher worker counts for production
        self.workers.reprocess_workers = max(self.workers.reprocess_workers, 16)
        self.workers.validate_workers = max(self.workers.validate_workers, 16)
        self.workers.warm_workers = max(self.workers.warm_workers, 8)
        
        # Production batch sizes
        self.workers.batch_size = max(self.workers.batch_size, 200)
        self.workers.max_concurrent_requests = max(self.workers.max_concurrent_requests, 100)
        
        # Longer timeouts for production workloads
        self.workers.request_timeout = max(self.workers.request_timeout, 60)
        self.cvars.func_timeout = max(self.cvars.func_timeout, 300)
    
    def _setup_production_startup(self):
        """Configure startup tasks for production."""
        # Conservative startup flags for production
        startup_flags = {
            "STARTUP_EXCHANGES_BOOTSTRAP": "1",
            "STARTUP_SYMBOLS_BOOTSTRAP": "1",
            "STARTUP_LOCAL_SYMBOLS_IMPORT": "1", 
            "STARTUP_CORE_SYMBOLS_IMPORT": "1",
            "EODHD_BOOTSTRAP_SYMBOLS": "1",
            "STARTUP_FIVE_STARS_PROCESSING": "1",
            "STARTUP_NORMALIZE_TYPES": "1",
            "STARTUP_NORMALIZE_COUNTRIES": "1",
            "STARTUP_COMPASS_ANCHORS": "1",
            "STARTUP_FIVE_STARS_USA_RECONCILE": "1",
            "STARTUP_FIVE_STARS_CANADA_IMPORT": "1", 
            "STARTUP_CVAR_BOOTSTRAP": "1",  # Enable for production
            "STARTUP_CACHE_WARMING": "1",
            "NIR_RECONCILE_SNAPSHOTS": "0",  # Manual trigger only
            "COMPASS_EXPERIMENT_ANCHORS": "0",  # Manual trigger only
            "STARTUP_COMPASS_PARAMETERS": "1",  # Enable for production
        }
        
        # Set defaults only if not explicitly configured
        for key, value in startup_flags.items():
            os.environ.setdefault(key, value)
        
        # Production-specific environment settings
        os.environ.setdefault("NVAR_LICENSE", "PROD")
        os.environ.setdefault("NIR_LAMBERT_STRICT", "1")
    
    def validate(self):
        """Production-specific validation (strict)."""
        errors = super().validate()
        
        # Additional production validations
        if not self.external_services.eodhd_api_key:
            errors.append("EODHD_API_KEY is required in production")
            
        if not (self.auth.basic_auth_user and self.auth.basic_auth_pass) and not self.auth.public_token_secret:
            errors.append("Authentication credentials are required in production")
            
        if not self.database.url and not (self.database.host and self.database.name and self.database.user):
            errors.append("Complete database configuration is required in production")
            
        # Validate external services
        if self.external_services.sb_connection_string and not self.external_services.sb_queue_name:
            errors.append("Service Bus queue name is required when connection string is provided")
            
        # Security validations
        if "*" in (self.auth.allowed_origins or []):
            errors.append("Wildcard CORS origins are not allowed in production")
            
        return errors
