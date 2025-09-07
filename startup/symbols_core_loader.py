"""
Core Symbols Loader - New symbol loading system

Loads symbols from symbols/core/ structure with categorical flags.
Supports the new folder-based flag system where folder names map to flags.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from core.db import get_db_session
from core.models import Symbols
from core.persistence import upsert_price_series_item


_LOG = logging.getLogger(__name__)


class SymbolsCoreLoader:
    """
    Loads symbols from symbols/core/ directory structure.
    
    Structure: symbols/core/{country}/{flag_name}/symbols.csv
    
    Each folder name becomes a flag, and symbols can have multiple flags
    if they appear in multiple folders.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path(__file__).parents[1] / "symbols" / "core"
        self.base_path = base_path
        self.session = get_db_session()
        self.processed_symbols: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # (symbol, country) -> flags
        
    def discover_flag_structure(self) -> Dict[str, Dict[str, List[Path]]]:
        """
        Discover the folder structure and CSV files.
        
        Returns:
            Dict of {country: {flag_name: [csv_paths]}}
        """
        structure = defaultdict(lambda: defaultdict(list))
        
        if not self.base_path.exists():
            _LOG.warning("Core symbols path does not exist: %s", self.base_path)
            return {}
        
        _LOG.info("Discovering symbols core structure in: %s", self.base_path)
        
        # Scan country directories
        for country_dir in self.base_path.iterdir():
            if not country_dir.is_dir():
                continue
                
            country_name = country_dir.name.upper()
            _LOG.debug("Found country directory: %s", country_name)
            
            # Scan flag directories within each country
            for flag_dir in country_dir.iterdir():
                if not flag_dir.is_dir():
                    continue
                    
                flag_name = flag_dir.name.lower()
                _LOG.debug("Found flag directory: %s/%s", country_name, flag_name)
                
                # Find CSV files in flag directory
                csv_files = list(flag_dir.glob("*.csv"))
                if csv_files:
                    structure[country_name][flag_name].extend(csv_files)
                    _LOG.debug("Found %d CSV files in %s/%s", len(csv_files), country_name, flag_name)
        
        _LOG.info("Discovered structure: %d countries, %d total flags", 
                 len(structure), sum(len(flags) for flags in structure.values()))
        
        return dict(structure)
    
    def normalize_flag_name(self, flag_name: str) -> str:
        """
        Normalize folder name to database column name.
        
        Examples:
        - "five_stars" -> "five_stars"
        - "sp_500" -> "sp_500" 
        - "dow_jones_industrial_average" -> "dow_jones_industrial_average"
        """
        return flag_name.lower().replace("-", "_").replace(" ", "_")
    
    def load_csv_file(self, csv_path: Path) -> List[Dict[str, str]]:
        """Load and parse a CSV file."""
        symbols = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Try to detect if first row is header
                first_line = f.readline().strip()
                f.seek(0)  # Reset to beginning
                
                # Check if first line looks like a header
                has_header = 'Code' in first_line or 'Symbol' in first_line or 'Name' in first_line
                
                reader = csv.DictReader(f) if has_header else csv.reader(f)
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        if isinstance(row, dict):
                            # DictReader (with header)
                            symbol_data = self._parse_dict_row(row)
                        else:
                            # Regular reader (no header)
                            symbol_data = self._parse_list_row(row)
                        
                        if symbol_data and symbol_data.get('code'):
                            symbols.append(symbol_data)
                            
                    except Exception as exc:
                        _LOG.warning("Failed to parse row %d in %s: %s", row_num, csv_path.name, exc)
                        continue
                        
        except Exception as exc:
            _LOG.error("Failed to read CSV file %s: %s", csv_path, exc)
            return []
        
        _LOG.debug("Loaded %d symbols from %s", len(symbols), csv_path.name)
        return symbols
    
    def _parse_dict_row(self, row: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Parse a dictionary row (from CSV with header)."""
        # Flexible column mapping
        def _get_value(preferred_names: List[str]) -> str:
            for name in preferred_names:
                if name in row and row[name]:
                    return str(row[name]).strip()
            return ""
        
        code = _get_value(['Code', 'Symbol', 'symbol', 'code', 'SYMBOL'])
        if not code:
            return None
            
        return {
            'code': code,
            'name': _get_value(['Name', 'name', 'NAME', 'Company', 'company']),
            'country': _get_value(['Country', 'country', 'COUNTRY']),
            'exchange': _get_value(['Exchange', 'exchange', 'EXCHANGE']),
            'currency': _get_value(['Currency', 'currency', 'CURRENCY']),
            'instrument_type': _get_value(['Type', 'type', 'TYPE', 'InstrumentType']),
            'isin': _get_value(['Isin', 'ISIN', 'isin']),
        }
    
    def _parse_list_row(self, row: List[str]) -> Optional[Dict[str, str]]:
        """Parse a list row (from CSV without header)."""
        if not row or len(row) == 0:
            return None
            
        # Assume standard format: Code,Name,Country,Exchange,Currency,Type,Isin
        code = str(row[0]).strip() if len(row) > 0 else ""
        if not code:
            return None
            
        return {
            'code': code,
            'name': str(row[1]).strip() if len(row) > 1 else "",
            'country': str(row[2]).strip() if len(row) > 2 else "",
            'exchange': str(row[3]).strip() if len(row) > 3 else "",
            'currency': str(row[4]).strip() if len(row) > 4 else "",
            'instrument_type': str(row[5]).strip() if len(row) > 5 else "",
            'isin': str(row[6]).strip() if len(row) > 6 else "",
        }
    
    def load_symbols_with_flags(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Load all symbols from core structure and apply flags.
        
        Args:
            dry_run: If True, only log what would be done without database changes
            
        Returns:
            Statistics about the loading process
        """
        stats = {
            'files_processed': 0,
            'symbols_loaded': 0,
            'symbols_updated': 0,
            'flags_applied': 0,
            'errors': 0,
        }
        
        if not self.session and not dry_run:
            _LOG.error("No database session available")
            return stats
        
        # Discover structure
        structure = self.discover_flag_structure()
        if not structure:
            _LOG.warning("No symbols core structure found")
            return stats
        
        # Load symbols and collect flags
        for country, flags_dict in structure.items():
            _LOG.info("Processing country: %s with %d flags", country, len(flags_dict))
            
            for flag_name, csv_paths in flags_dict.items():
                normalized_flag = self.normalize_flag_name(flag_name)
                _LOG.info("Processing flag: %s -> %s", flag_name, normalized_flag)
                
                for csv_path in csv_paths:
                    try:
                        symbols = self.load_csv_file(csv_path)
                        stats['files_processed'] += 1
                        
                        for symbol_data in symbols:
                            symbol_key = (symbol_data['code'], country)
                            self.processed_symbols[symbol_key].add(normalized_flag)
                            
                            if dry_run:
                                _LOG.debug("DRY RUN: Would load %s with flag %s", 
                                         symbol_data['code'], normalized_flag)
                            else:
                                # Upsert symbol data
                                self._upsert_symbol_with_flag(symbol_data, country, normalized_flag)
                                stats['symbols_loaded'] += 1
                                stats['flags_applied'] += 1
                        
                    except Exception as exc:
                        _LOG.error("Failed to process %s: %s", csv_path, exc)
                        stats['errors'] += 1
        
        # Apply accumulated flags
        if not dry_run:
            self._apply_accumulated_flags()
        
        _LOG.info("Symbols core loading completed: %s", stats)
        return stats
    
    def _upsert_symbol_with_flag(self, symbol_data: Dict[str, str], country: str, flag_name: str) -> None:
        """Upsert a symbol and prepare to set its flag."""
        try:
            # First upsert the basic symbol data
            upsert_price_series_item(
                code=symbol_data['code'],
                name=symbol_data['name'] or None,
                country=country,
                exchange=symbol_data['exchange'] or None,
                currency=symbol_data['currency'] or None,
                instrument_type=symbol_data['instrument_type'] or None,
                isin=symbol_data['isin'] or None,
            )
            
        except Exception as exc:
            _LOG.error("Failed to upsert symbol %s: %s", symbol_data['code'], exc)
            raise
    
    def _apply_accumulated_flags(self) -> None:
        """Apply all accumulated flags to symbols in the database."""
        if not self.processed_symbols:
            return
            
        _LOG.info("Applying flags to %d symbols", len(self.processed_symbols))
        
        try:
            for (symbol, country), flags in self.processed_symbols.items():
                # Normalize country name for database lookup
                normalized_country = self._normalize_country_name(country)
                
                # Find the symbol in database with normalized country name
                symbol_obj = (
                    self.session.query(Symbols)
                    .filter(Symbols.symbol == symbol, Symbols.country == normalized_country)
                    .first()
                )
                
                if symbol_obj:
                    # Apply each flag
                    for flag_name in flags:
                        if symbol_obj.set_flag(flag_name, 1):
                            _LOG.debug("Set flag %s=1 for %s", flag_name, symbol)
                        else:
                            _LOG.warning("Unknown flag %s for symbol %s", flag_name, symbol)
                    
                    # Auto-detect instrument type flags
                    self._auto_detect_type_flags(symbol_obj)
                else:
                    _LOG.warning("Symbol %s (%s) not found in database after upsert", symbol, country)
            
            # Commit all changes
            self.session.commit()
            _LOG.info("Successfully applied flags to symbols")
            
        except Exception as exc:
            _LOG.error("Failed to apply flags: %s", exc)
            self.session.rollback()
            raise
        finally:
            self.session.close()
    
    def _auto_detect_type_flags(self, symbol_obj: Symbols) -> None:
        """Auto-detect and set type flags based on instrument_type."""
        if not symbol_obj.instrument_type:
            return
            
        instrument_type = symbol_obj.instrument_type.lower()
        
        # Map instrument types to flags
        if 'etf' in instrument_type or 'exchange traded fund' in instrument_type:
            symbol_obj.etf_flag = 1
        elif 'mutual fund' in instrument_type or 'fund' in instrument_type:
            symbol_obj.mutual_fund_flag = 1
        elif 'stock' in instrument_type or 'equity' in instrument_type or 'common' in instrument_type:
            symbol_obj.common_stock_flag = 1
    
    def _normalize_country_name(self, country: str) -> str:
        """Normalize country name to match database format."""
        if not country:
            return country
            
        # Map folder names to database country names
        country_mapping = {
            'CANADA': 'Canada',
            'US': 'US',  # US symbols typically have country = 'US' in DB
            'UK': 'UK'   # UK symbols typically have country = 'UK' in DB
        }
        
        normalized = country_mapping.get(country.upper(), country)
        _LOG.debug("Normalized country %s -> %s", country, normalized)
        return normalized


def load_core_symbols(dry_run: bool = False, db_ready: bool = True) -> Dict[str, int]:
    """
    Main entry point for loading symbols from core structure.
    
    Args:
        dry_run: If True, only show what would be done
        db_ready: Database availability flag
        
    Returns:
        Loading statistics
    """
    if not db_ready and not dry_run:
        _LOG.warning("Database not ready, skipping core symbols loading")
        return {'error': 'database_not_ready'}
    
    try:
        loader = SymbolsCoreLoader()
        return loader.load_symbols_with_flags(dry_run=dry_run)
        
    except Exception as exc:
        _LOG.error("Core symbols loading failed: %s", exc)
        return {'error': str(exc)}
