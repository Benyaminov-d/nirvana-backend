"""
Harvard Universe Management Commands

CLI commands for managing Harvard release universe.
These commands can be used for automated maintenance and deployment.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

import click
import logging

from services.universe_manager import get_harvard_universe_manager
from core.universe_config import UniverseFeatureFlags, ProductCategory


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_LOG = logging.getLogger(__name__)


@click.group()
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def harvard(dry_run, verbose):
    """Harvard Universe Management Commands."""
    if dry_run:
        os.environ["HARVARD_DRY_RUN"] = "true"
        click.echo("🔍 DRY RUN MODE - No changes will be made")
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@harvard.command()
@click.option('--country', help='Filter by country (US, UK, CA)')
def stats(country):
    """Show Harvard universe statistics."""
    try:
        manager = get_harvard_universe_manager()
        stats = manager.get_universe_stats()
        
        click.echo(f"\n📊 Harvard Universe Statistics")
        click.echo(f"Last Updated: {stats.last_updated}")
        click.echo(f"Total Products: {stats.total_products}")
        
        click.echo(f"\n🌍 By Country:")
        for country_code, count in stats.by_country.items():
            click.echo(f"  {country_code}: {count}")
        
        click.echo(f"\n📂 By Category:")
        for category, count in stats.by_category.items():
            click.echo(f"  {category}: {count}")
        
        click.echo(f"\n⚠️  Missing Data:")
        click.echo(f"  Missing μ: {stats.missing_mu}")
        click.echo(f"  Missing CVaR: {stats.missing_cvar}")
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@harvard.command()
@click.option('--country', help='Filter by country (US, UK, CA)')
@click.option('--limit', default=50, help='Maximum products to show')
def products(country, limit):
    """List products in Harvard universe."""
    try:
        manager = get_harvard_universe_manager()
        products = manager.get_universe_products(country)
        
        if limit > 0:
            products = products[:limit]
        
        click.echo(f"\n📈 Harvard Universe Products")
        if country:
            click.echo(f"Country: {country}")
        click.echo(f"Showing: {len(products)} products")
        
        for product in products:
            status_indicators = []
            if not product.has_mu:
                status_indicators.append("❌μ")
            if not product.has_cvar:
                status_indicators.append("❌CVaR")
            if product.five_stars:
                status_indicators.append("⭐")
            
            status = " ".join(status_indicators) if status_indicators else "✅"
            
            click.echo(f"  {product.symbol:<12} {product.country} {product.instrument_type:<15} {status}")
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@harvard.command()
@click.option('--country', help='Process specific country only')
@click.option('--force', is_flag=True, help='Force complete refresh')
def ensure_completeness(country, force):
    """Ensure universe has all required data and anchors."""
    try:
        if force:
            os.environ["HARVARD_FORCE_REFRESH"] = "true"
            click.echo("🔄 Force refresh enabled")
        
        manager = get_harvard_universe_manager()
        
        click.echo("🔧 Ensuring Harvard universe completeness...")
        if country:
            click.echo(f"Processing country: {country}")
        
        report = manager.ensure_universe_completeness(country)
        
        # Display report
        click.echo(f"\n📋 Completeness Report")
        click.echo(f"Timestamp: {report['timestamp']}")
        click.echo(f"Products Processed: {report['products_processed']}")
        
        if report.get('dry_run'):
            click.echo("🔍 DRY RUN - No changes made")
        
        actions = report['actions']
        click.echo(f"\n✅ Actions Taken:")
        click.echo(f"  μ computed: {actions['mu_computed']}")
        click.echo(f"  CVaR computed: {actions['cvar_computed']}")
        
        if actions['anchors_recalibrated']:
            click.echo(f"  Anchors recalibrated:")
            for category in actions['anchors_recalibrated']:
                click.echo(f"    - {category}")
        
        if report['errors']:
            click.echo(f"\n❌ Errors:")
            for error in report['errors']:
                click.echo(f"  - {error}")
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@harvard.command()
def validate():
    """Validate Harvard universe integrity."""
    try:
        manager = get_harvard_universe_manager()
        validation_report = manager.validate_universe_integrity()
        
        click.echo(f"\n🔍 Harvard Universe Validation")
        click.echo(f"Timestamp: {validation_report['timestamp']}")
        
        stats = validation_report['universe_stats']
        click.echo(f"Total Products: {stats.total_products}")
        
        if validation_report['healthy']:
            click.echo("✅ Universe is healthy")
        else:
            click.echo("⚠️  Issues found:")
            for issue in validation_report['issues']:
                click.echo(f"  - {issue}")
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@harvard.command()
def config():
    """Show current Harvard universe configuration."""
    try:
        from core.universe_config import HarvardUniverseConfig
        
        config = HarvardUniverseConfig()
        enabled_countries = config.get_enabled_countries()
        
        click.echo(f"\n⚙️  Harvard Universe Configuration")
        
        for country_code, country_config in enabled_countries.items():
            click.echo(f"\n🌍 {country_code} - {country_config.country_name}")
            click.echo(f"  Compass Category: {country_config.compass_category}")
            click.echo(f"  Categories: {[cat.value for cat in country_config.categories]}")
            click.echo(f"  Min Market Cap: ${country_config.min_market_cap:,}" if country_config.min_market_cap else "  Min Market Cap: None")
            click.echo(f"  Min History Days: {country_config.min_history_days}")
            if country_config.special_lists:
                click.echo(f"  Special Lists: {country_config.special_lists}")
        
        click.echo(f"\n🚩 Feature Flags:")
        flags = UniverseFeatureFlags()
        click.echo(f"  Auto compute μ: {flags.auto_compute_missing_mu()}")
        click.echo(f"  Auto compute CVaR: {flags.auto_compute_missing_cvar()}")
        click.echo(f"  Auto recalibrate anchors: {flags.auto_recalibrate_anchors()}")
        click.echo(f"  Dry run mode: {flags.dry_run_mode()}")
        click.echo(f"  Max workers: {flags.max_parallel_workers()}")
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@harvard.command()
@click.option('--country', help='Recalibrate specific country only')
@click.confirmation_option(prompt='Are you sure you want to recalibrate anchors?')
def recalibrate_anchors(country):
    """Recalibrate anchors for Harvard universe."""
    try:
        from services.compass_anchors import auto_calibrate_from_db
        from core.universe_config import HarvardUniverseConfig
        
        config = HarvardUniverseConfig()
        compass_categories = config.get_compass_categories()
        
        if country:
            country_config = config.get_country_config(country.upper())
            if not country_config:
                click.echo(f"❌ Invalid country: {country}", err=True)
                sys.exit(1)
            compass_categories = [country_config.compass_category]
        
        click.echo("🔧 Recalibrating anchors...")
        
        for compass_category in compass_categories:
            click.echo(f"  Processing {compass_category}...")
            if not UniverseFeatureFlags.dry_run_mode():
                success = auto_calibrate_from_db(compass_category)
                if success:
                    click.echo(f"  ✅ {compass_category} recalibrated")
                else:
                    click.echo(f"  ❌ Failed to recalibrate {compass_category}")
            else:
                click.echo(f"  🔍 DRY RUN - Would recalibrate {compass_category}")
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@harvard.command()
@click.option('--env-file', default='.env', help='Environment file to update')
def set_env_defaults(env_file):
    """Set default environment variables for Harvard universe."""
    env_path = Path(env_file)
    
    defaults = {
        "HARVARD_AUTO_COMPUTE_MU": "true",
        "HARVARD_AUTO_COMPUTE_CVAR": "false",
        "HARVARD_AUTO_RECALIBRATE": "true", 
        "HARVARD_FORCE_REFRESH": "false",
        "HARVARD_DRY_RUN": "false",
        "HARVARD_MAX_WORKERS": "16",
    }
    
    click.echo(f"📝 Setting Harvard environment defaults in {env_file}")
    
    # Read existing env file if it exists
    existing_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    existing_vars[key.strip()] = value.strip()
    
    # Add new defaults (don't override existing)
    updated = False
    for key, default_value in defaults.items():
        if key not in existing_vars:
            existing_vars[key] = default_value
            updated = True
            click.echo(f"  + {key}={default_value}")
    
    if updated:
        # Write back to file
        with open(env_path, 'w') as f:
            for key, value in sorted(existing_vars.items()):
                f.write(f"{key}={value}\n")
        click.echo(f"✅ Updated {env_file}")
    else:
        click.echo("✅ All defaults already set")


if __name__ == '__main__':
    harvard()
