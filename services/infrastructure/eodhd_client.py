"""
EODHD API Client - Infrastructure service for external market data.

This service encapsulates all interactions with the EODHD (End Of Day Historical Data) API,
providing clean abstractions for price data retrieval and symbol resolution.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import logging
import requests
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PriceDataPoint:
    """Single price data point from EODHD API."""
    date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int


@dataclass
class EODHDSymbolInfo:
    """Symbol information from EODHD API."""
    symbol: str
    name: str
    exchange: str
    country: str
    currency: str
    instrument_type: str


class EODHDClient:
    """
    Infrastructure service for EODHD API integration.
    
    Handles:
    - API authentication and rate limiting
    - Price data retrieval with proper error handling
    - Symbol resolution and metadata lookup
    - Response parsing and data validation
    - Caching and request optimization
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://eodhistoricaldata.com/api"):
        self.api_key = api_key or os.getenv("EODHD_API_KEY")
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": "Nirvana-App/1.0",
            "Accept": "application/json"
        })
        
        if not self.api_key:
            logger.warning("EODHD API key not configured - some features will be unavailable")
    
    def get_historical_prices(
        self,
        symbol: str,
        exchange: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        period: str = "d"  # d, w, m for daily, weekly, monthly
    ) -> List[PriceDataPoint]:
        """
        Retrieve historical price data for a symbol.
        
        Args:
            symbol: Ticker symbol
            exchange: Exchange code (e.g., 'US', 'LSE')
            from_date: Start date for data retrieval
            to_date: End date for data retrieval
            period: Data frequency ('d', 'w', 'm')
            
        Returns:
            List of price data points
            
        Raises:
            Exception: On API errors or invalid responses
        """
        
        if not self.api_key:
            raise Exception("EODHD API key required for price data retrieval")
        
        # Construct API endpoint
        # For Canadian stocks (TO exchange), make sure to include the exchange suffix
        if exchange.lower() in ('to', 'tsx', 'ca'):
            endpoint = f"{self.base_url}/eod/{symbol}.TO"
            logger.debug(f"Using Toronto exchange suffix for {symbol}: {endpoint}")
        else:
            endpoint = f"{self.base_url}/eod/{symbol}.{exchange}"
        
        # Prepare parameters
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "period": period
        }
        
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%d")
        
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%d")
        
        try:
            logger.debug(f"Requesting EODHD data for {symbol}.{exchange}")
            
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response format
            if not isinstance(data, list):
                raise Exception(f"Unexpected response format from EODHD API: {type(data)}")
            
            # Parse price data
            price_points = []
            for item in data:
                try:
                    price_point = PriceDataPoint(
                        date=datetime.strptime(item["date"], "%Y-%m-%d").date(),
                        open=float(item["open"]),
                        high=float(item["high"]),
                        low=float(item["low"]),
                        close=float(item["close"]),
                        adjusted_close=float(item["adjusted_close"]),
                        volume=int(item.get("volume", 0))
                    )
                    price_points.append(price_point)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid price data point for {symbol}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(price_points)} price points for {symbol}.{exchange}")
            return price_points
            
        except requests.RequestException as e:
            logger.error(f"EODHD API request failed for {symbol}.{exchange}: {e}")
            raise Exception(f"Failed to retrieve price data: {str(e)}")
        
        except Exception as e:
            logger.error(f"EODHD data processing failed for {symbol}.{exchange}: {e}")
            raise
    
    def get_symbol_info(self, symbol: str, exchange: str) -> Optional[EODHDSymbolInfo]:
        """
        Get detailed symbol information from EODHD.
        
        Args:
            symbol: Ticker symbol
            exchange: Exchange code
            
        Returns:
            Symbol information or None if not found
        """
        
        if not self.api_key:
            logger.warning("EODHD API key required for symbol info retrieval")
            return None
        
        endpoint = f"{self.base_url}/fundamentals/{symbol}.{exchange}"
        
        params = {
            "api_token": self.api_key,
            "fmt": "json"
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=15)
            
            if response.status_code == 404:
                logger.debug(f"Symbol {symbol}.{exchange} not found in EODHD")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Extract general information
            general = data.get("General", {})
            
            return EODHDSymbolInfo(
                symbol=symbol,
                name=general.get("Name", ""),
                exchange=exchange,
                country=general.get("CountryName", ""),
                currency=general.get("CurrencyCode", ""),
                instrument_type=general.get("Type", "")
            )
            
        except requests.RequestException as e:
            logger.warning(f"EODHD symbol info request failed for {symbol}.{exchange}: {e}")
            return None
        
        except Exception as e:
            logger.warning(f"EODHD symbol info processing failed for {symbol}.{exchange}: {e}")
            return None
    
    def search_symbols(self, query: str, limit: int = 50) -> List[EODHDSymbolInfo]:
        """
        Search for symbols matching a query.
        
        Args:
            query: Search term
            limit: Maximum results to return
            
        Returns:
            List of matching symbols
        """
        
        if not self.api_key:
            logger.warning("EODHD API key required for symbol search")
            return []
        
        endpoint = f"{self.base_url}/search/{query}"
        
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "limit": limit
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list):
                return []
            
            symbols = []
            for item in data:
                try:
                    symbol_info = EODHDSymbolInfo(
                        symbol=item["Code"],
                        name=item.get("Name", ""),
                        exchange=item["Exchange"],
                        country=item.get("Country", ""),
                        currency=item.get("Currency", ""),
                        instrument_type=item.get("Type", "")
                    )
                    symbols.append(symbol_info)
                except (KeyError, TypeError):
                    continue
            
            return symbols
            
        except Exception as e:
            logger.warning(f"EODHD symbol search failed for '{query}': {e}")
            return []
    
    def get_exchanges(self) -> List[Dict[str, str]]:
        """
        Get list of available exchanges from EODHD.
        
        Returns:
            List of exchange information dictionaries
        """
        
        if not self.api_key:
            logger.warning("EODHD API key required for exchange listing")
            return []
        
        endpoint = f"{self.base_url}/exchanges-list/"
        
        params = {
            "api_token": self.api_key,
            "fmt": "json"
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list):
                return data
            else:
                logger.warning("Unexpected response format for exchanges list")
                return []
                
        except Exception as e:
            logger.warning(f"EODHD exchanges list request failed: {e}")
            return []
    
    def validate_symbol(self, symbol: str, exchange: str) -> bool:
        """
        Validate if a symbol exists and has data available.
        
        Args:
            symbol: Ticker symbol
            exchange: Exchange code
            
        Returns:
            True if symbol exists and has data
        """
        
        try:
            # Try to get at least one data point
            prices = self.get_historical_prices(
                symbol=symbol,
                exchange=exchange,
                period="d"
            )
            
            return len(prices) > 0
            
        except Exception:
            return False
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Get API status and usage information.
        
        Returns:
            API status information
        """
        
        if not self.api_key:
            return {
                "status": "not_configured",
                "api_key_available": False,
                "message": "EODHD API key not configured"
            }
        
        # Simple test request to check API availability
        try:
            endpoint = f"{self.base_url}/exchanges-list/"
            params = {
                "api_token": self.api_key,
                "fmt": "json"
            }
            
            response = self.session.get(endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                return {
                    "status": "operational",
                    "api_key_available": True,
                    "message": "EODHD API is accessible"
                }
            elif response.status_code == 403:
                return {
                    "status": "access_denied", 
                    "api_key_available": False,
                    "message": "Invalid or expired API key"
                }
            else:
                return {
                    "status": "error",
                    "api_key_available": True,
                    "message": f"API returned status {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "unreachable",
                "api_key_available": bool(self.api_key),
                "message": f"API unreachable: {str(e)}"
            }
