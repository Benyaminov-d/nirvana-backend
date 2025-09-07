"""Testing environment configuration."""

import os
from config.base import BaseConfig


class TestingConfig(BaseConfig):
    """Configuration for testing environment."""
    
    def _setup_environment(self):
        """Setup testing-specific configuration."""
        # Enable debug mode for tests
        self.debug = True
        
        # SAFETY CHECK: Create new CVaRConfig (never modify frozen dataclass directly!)
        # This replaces the old problematic: self.cvars.enforce_min_years = False
        from config.base import CVaRConfig
        self.cvars = CVaRConfig(
            min_years=1.0,  # Minimal history for tests
            enforce_min_years=False,  # No validation for tests
            eq_lookback_days=self.cvars.eq_lookback_days,
            price_field=self.cvars.price_field,
            allow_close_fallback=True,  # Allow fallback in tests
            zero_return_eps=self.cvars.zero_return_eps,
            last252_max_zero_share=1.0,  # Allow any zero returns in tests
            max_drop_last252=252,  # Allow all drops in tests
            sims=1000,  # Fewer simulations for tests
            trading_days=self.cvars.trading_days,
            log_phase=True,  # Enable logging in tests
            service_mode="local",  # Always local for tests
            func_url="",
            func_timeout=30,  # Short timeout for tests
            func_connect_timeout=5
        )
        
        # Test database configuration
        from config.base import DatabaseConfig
        self.database = DatabaseConfig(
            host="localhost",
            port=5432,
            name="nirvana_test",
            user="postgres", 
            password=os.getenv("DB_PASSWORD", "")
        )
        
        # Permissive auth for tests
        self.auth.allowed_origins = ["*"]
        self.auth.session_timeout = 60  # Short sessions for tests
        
        # Fast processing for tests
        self.workers.reprocess_workers = 2
        self.workers.validate_workers = 2
        self.workers.warm_workers = 1
        self.workers.batch_size = 10
        self.workers.max_concurrent_requests = 5
        self.workers.request_timeout = 10
        
        # Short timeouts for tests
        self.cvars.func_timeout = 10
        self.cvars.func_connect_timeout = 5
        self.external_services.openai_run_timeout = 5
        
        # Test-specific CVaR configuration
        self.cvars.sims = 100  # Faster simulations
        self.cvars.log_phase = True  # Enable logging for test debugging
        
        # Setup testing startup configuration
        self._setup_testing_startup()
    
    def _setup_testing_startup(self):
        """Configure startup tasks for testing (minimal)."""
        # Disable most bootstrap tasks for faster test startup
        startup_flags = {
            "STARTUP_EXCHANGES_BOOTSTRAP": "0",
            "STARTUP_SYMBOLS_BOOTSTRAP": "0",
            "STARTUP_LOCAL_SYMBOLS_IMPORT": "0",
            "STARTUP_CORE_SYMBOLS_IMPORT": "0", 
            "EODHD_BOOTSTRAP_SYMBOLS": "0",
            "STARTUP_FIVE_STARS_PROCESSING": "0",
            "STARTUP_NORMALIZE_TYPES": "0",
            "STARTUP_NORMALIZE_COUNTRIES": "0",
            "STARTUP_COMPASS_ANCHORS": "0",
            "STARTUP_FIVE_STARS_USA_RECONCILE": "0",
            "STARTUP_FIVE_STARS_CANADA_IMPORT": "0",
            "STARTUP_CVAR_BOOTSTRAP": "0",
            "STARTUP_CACHE_WARMING": "0",
            "NIR_RECONCILE_SNAPSHOTS": "0",
            "COMPASS_EXPERIMENT_ANCHORS": "0",
            "STARTUP_COMPASS_PARAMETERS": "0",
            "RAG_ENABLED": "false",  # Disable RAG for tests
        }
        
        # Set defaults for testing
        for key, value in startup_flags.items():
            os.environ.setdefault(key, value)
        
        # Test-specific environment settings
        os.environ.setdefault("NVAR_LICENSE", "TEST")
        os.environ.setdefault("LOG_LEVEL", "DEBUG")
    
    def validate(self):
        """Testing-specific validation (very permissive)."""
        # Don't enforce any strict validations in testing
        errors = []
        
        # Only validate critical configuration that could break tests
        if not self.database.name:
            errors.append("Test database name is required")
        
        return errors
