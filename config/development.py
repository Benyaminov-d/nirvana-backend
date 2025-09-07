"""Development environment configuration."""

import os
from config.base import BaseConfig


class DevelopmentConfig(BaseConfig):
    """Configuration for development environment."""
    
    def _setup_environment(self):
        """Setup development-specific configuration."""
        # Enable debug mode
        self.debug = True
        
        # SAFETY CHECK: Create new CVaR config (never modify frozen dataclass directly!)
        # This replaces the old problematic: self.cvars.enforce_min_years = False
        from config.base import CVaRConfig
        self.cvars = CVaRConfig(
            min_years=1.0,  # Minimal years for development
            enforce_min_years=False,  # Relaxed validation in development
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
        
        # Development-specific database defaults
        if not self.database.url:
            from config.base import DatabaseConfig
            self.database = DatabaseConfig(
                host="localhost",
                port=5432,
                name="nirvana_dev", 
                user="postgres",
                password=os.getenv("DB_PASSWORD", "")
            )
        
        # Allow broader CORS in development
        if not self.auth.allowed_origins or self.auth.allowed_origins == ["*"]:
            self.auth.allowed_origins = [
                "http://localhost:3000",
                "http://localhost:8000", 
                "http://localhost:5173",  # Vite dev server
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
                "http://127.0.0.1:5173"
            ]
        
        # Development logging
        os.environ.setdefault("LOG_LEVEL", "DEBUG")
        
        # Startup configuration for development
        self._setup_development_startup()
    
    def _setup_development_startup(self):
        """Configure startup tasks for development."""
        # Enable most bootstrap tasks for development convenience
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
            "STARTUP_CVAR_BOOTSTRAP": "0",  # Don't auto-enqueue in dev
            "STARTUP_CACHE_WARMING": "1",
            "NIR_RECONCILE_SNAPSHOTS": "0",  # Don't reconcile by default
            "COMPASS_EXPERIMENT_ANCHORS": "0",  # Expensive
            "STARTUP_COMPASS_PARAMETERS": "0"   # Expensive
        }
        
        # Set defaults only if not already set
        for key, value in startup_flags.items():
            os.environ.setdefault(key, value)
    
    def validate(self):
        """Development-specific validation (more permissive)."""
        errors = super().validate()
        
        # Remove production-only validations
        errors = [e for e in errors if "production" not in e.lower()]
        
        # Development-specific warnings (non-blocking)
        warnings = []
        if not self.external_services.eodhd_api_key:
            warnings.append("EODHD_API_KEY not set - some features will use cached data")
        
        # Log warnings but don't block startup
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(f"Development config warning: {warning}")
        
        return errors
