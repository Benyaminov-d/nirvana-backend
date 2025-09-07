#!/usr/bin/env python3
"""
Symbols Management Commands

CLI commands for managing symbols loading, flags, and validation.
Supports the new symbols structure with core flags and by_country loading.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

import click
import logging

from startup.symbols_core_loader import load_core_symbols
from startup.symbols_from_files import import_local_symbol_catalogs
from core.db import get_db_session
from core.models import Symbols


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_LOG = logging.getLogger(__name__)


@click.group()
@click.option('--dry-run', is_flag=True, help='Show what would be done without executing')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def symbols(dry_run, verbose):
    """Symbols Management Commands."""
    if dry_run:
        click.echo("🔍 DRY RUN MODE - No changes will be made")
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@symbols.command()
@click.option('--country', help='Filter by country (US, UK, CA, etc.)')
@click.option('--with-flags', is_flag=True, help='Show active flags for each symbol')
def stats(country, with_flags):
    """Show symbols database statistics."""
    try:
        session = get_db_session()
        if not session:
            click.echo("❌ Cannot connect to database", err=True)
            return
        
        query = session.query(Symbols)
        if country:
            query = query.filter(Symbols.country == country.upper())
        
        total = query.count()
        
        click.echo(f"\n📊 Symbols Statistics")
        if country:
            click.echo(f"Country: {country.upper()}")
        click.echo(f"Total Symbols: {total:,}")
        
        # Stats by country
        if not country:
            click.echo(f"\n🌍 By Country:")
            country_stats = (
                session.query(Symbols.country, session.query(Symbols.id).filter(Symbols.country == Symbols.country).count())
                .group_by(Symbols.country)
                .all()
            )
            for country_code, count in country_stats:
                click.echo(f"  {country_code or 'NULL'}: {count:,}")
        
        # Stats by instrument type
        click.echo(f"\n📂 By Instrument Type:")
        type_stats = (
            session.query(Symbols.instrument_type, session.query(Symbols.id).filter(Symbols.instrument_type == Symbols.instrument_type).count())
            .group_by(Symbols.instrument_type)
            .all()
        )
        for inst_type, count in type_stats[:10]:  # Top 10
            click.echo(f"  {inst_type or 'NULL'}: {count:,}")
        
        # Flag statistics
        if with_flags:
            click.echo(f"\n🚩 Active Flags:")
            flag_columns = Symbols.get_flag_columns()
            for flag in flag_columns:
                count = query.filter(getattr(Symbols, flag) == 1).count()
                if count > 0:
                    click.echo(f"  {flag}: {count:,}")
        
        session.close()
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@symbols.command()
@click.option('--country', help='Filter by country')
@click.option('--limit', default=50, help='Maximum symbols to show')
@click.option('--flags-only', is_flag=True, help='Show only symbols with active flags')
def list(country, limit, flags_only):
    """List symbols in database."""
    try:
        session = get_db_session()
        if not session:
            click.echo("❌ Cannot connect to database", err=True)
            return
        
        query = session.query(Symbols)
        
        if country:
            query = query.filter(Symbols.country == country.upper())
        
        if flags_only:
            # Show only symbols with at least one active flag
            flag_conditions = []
            for flag in Symbols.get_flag_columns():
                flag_conditions.append(getattr(Symbols, flag) == 1)
            from sqlalchemy import or_
            query = query.filter(or_(*flag_conditions))
        
        symbols = query.order_by(Symbols.symbol).limit(limit).all()
        
        click.echo(f"\n📈 Symbols ({len(symbols)} shown)")
        if country:
            click.echo(f"Country: {country.upper()}")
        
        for symbol in symbols:
            active_flags = symbol.get_active_flags()
            flags_display = f" [{', '.join(active_flags)}]" if active_flags else ""
            
            click.echo(f"  {symbol.symbol:<12} {symbol.country:<4} {symbol.instrument_type or 'Unknown':<15}{flags_display}")
        
        session.close()
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@symbols.command()
@click.option('--dry-run', is_flag=True, help='Show what would be loaded without executing')
def load_core(dry_run):
    """Load symbols from symbols/core/ structure with flags."""
    try:
        click.echo("🔧 Loading symbols from core structure...")
        
        stats = load_core_symbols(dry_run=dry_run, db_ready=True)
        
        if 'error' in stats:
            click.echo(f"❌ Loading failed: {stats['error']}", err=True)
            sys.exit(1)
        
        click.echo(f"\n📋 Core Loading Results:")
        click.echo(f"Files processed: {stats.get('files_processed', 0)}")
        click.echo(f"Symbols loaded: {stats.get('symbols_loaded', 0)}")
        click.echo(f"Flags applied: {stats.get('flags_applied', 0)}")
        click.echo(f"Errors: {stats.get('errors', 0)}")
        
        if dry_run:
            click.echo("🔍 DRY RUN - No changes were made")
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@symbols.command()
@click.option('--dry-run', is_flag=True, help='Show what would be loaded without executing')
def load_by_country():
    """Load symbols from symbols/by_country/ CSV files."""
    try:
        click.echo("🔧 Loading symbols from by_country structure...")
        
        if dry_run:
            click.echo("🔍 DRY RUN - Would load from symbols/by_country/")
            # Show what files would be processed
            from pathlib import Path
            by_country_path = Path(__file__).parents[1] / "symbols" / "by_country"
            if by_country_path.exists():
                csv_files = list(by_country_path.glob("*.csv"))
                click.echo(f"Files found: {len(csv_files)}")
                for csv_file in csv_files:
                    click.echo(f"  - {csv_file.name}")
            return
        
        count = import_local_symbol_catalogs(db_ready=True)
        
        click.echo(f"\n📋 By Country Loading Results:")
        click.echo(f"Total rows processed: {count}")
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@symbols.command()
@click.confirmation_option(prompt='This will reload all symbols. Are you sure?')
def reload_all():
    """Reload all symbols from both core and by_country structures."""
    try:
        click.echo("🔄 Reloading all symbols...")
        
        # Load by_country first (base symbols)
        click.echo("Step 1: Loading by_country symbols...")
        by_country_count = import_local_symbol_catalogs(db_ready=True)
        click.echo(f"✅ Loaded {by_country_count} rows from by_country")
        
        # Load core with flags
        click.echo("Step 2: Loading core symbols with flags...")
        core_stats = load_core_symbols(db_ready=True)
        
        if 'error' in core_stats:
            click.echo(f"❌ Core loading failed: {core_stats['error']}", err=True)
        else:
            click.echo(f"✅ Core loading completed:")
            click.echo(f"  Files: {core_stats.get('files_processed', 0)}")
            click.echo(f"  Symbols: {core_stats.get('symbols_loaded', 0)}")
            click.echo(f"  Flags: {core_stats.get('flags_applied', 0)}")
        
        click.echo("🎉 Symbol reload completed!")
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@symbols.command()
@click.argument('symbol')
@click.option('--country', help='Symbol country (if ambiguous)')
def info(symbol, country):
    """Show detailed information about a specific symbol."""
    try:
        session = get_db_session()
        if not session:
            click.echo("❌ Cannot connect to database", err=True)
            return
        
        query = session.query(Symbols).filter(Symbols.symbol == symbol.upper())
        if country:
            query = query.filter(Symbols.country == country.upper())
        
        symbols = query.all()
        
        if not symbols:
            click.echo(f"❌ Symbol '{symbol}' not found", err=True)
            return
        
        for sym in symbols:
            click.echo(f"\n📈 Symbol Information: {sym.symbol}")
            click.echo(f"Name: {sym.name}")
            click.echo(f"Country: {sym.country}")
            click.echo(f"Exchange: {sym.exchange}")
            click.echo(f"Currency: {sym.currency}")
            click.echo(f"Type: {sym.instrument_type}")
            click.echo(f"ISIN: {sym.isin}")
            click.echo(f"Created: {sym.created_at}")
            click.echo(f"Updated: {sym.updated_at}")
            
            # Show active flags
            active_flags = sym.get_active_flags()
            if active_flags:
                click.echo(f"Active Flags: {', '.join(active_flags)}")
            else:
                click.echo("Active Flags: None")
            
            # Show data quality flags
            click.echo(f"Valid: {sym.valid}")
            click.echo(f"Sufficient History: {'No' if sym.insufficient_history == 1 else 'Yes' if sym.insufficient_history == 0 else 'Unknown'}")
            
            if len(symbols) > 1:
                click.echo("-" * 50)
        
        session.close()
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


@symbols.command()
@click.argument('flag_name')
@click.option('--country', help='Filter by country')
@click.option('--limit', default=20, help='Maximum symbols to show')
def by_flag(flag_name, country, limit):
    """Show symbols that have a specific flag active."""
    try:
        session = get_db_session()
        if not session:
            click.echo("❌ Cannot connect to database", err=True)
            return
        
        # Validate flag name
        if flag_name not in Symbols.get_flag_columns():
            click.echo(f"❌ Unknown flag: {flag_name}", err=True)
            click.echo(f"Available flags: {', '.join(Symbols.get_flag_columns())}")
            return
        
        query = (
            session.query(Symbols)
            .filter(getattr(Symbols, flag_name) == 1)
        )
        
        if country:
            query = query.filter(Symbols.country == country.upper())
        
        symbols = query.order_by(Symbols.symbol).limit(limit).all()
        
        click.echo(f"\n🚩 Symbols with flag '{flag_name}' ({len(symbols)} shown)")
        if country:
            click.echo(f"Country: {country.upper()}")
        
        for symbol in symbols:
            click.echo(f"  {symbol.symbol:<12} {symbol.country:<4} {symbol.name or 'N/A'}")
        
        session.close()
        
    except Exception as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    symbols()
