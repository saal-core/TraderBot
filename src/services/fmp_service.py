# src/services/fmp_service.py
"""Financial Modeling Prep API service with improved error handling and fallbacks."""
import os
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


class FMPService:
    """Service for fetching financial data from Financial Modeling Prep API."""

    BASE_URL = "https://financialmodelingprep.com/api"

    def __init__(self):
        """Initialize FMP service with API key."""
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY not found in environment variables")

    def _make_request(self, endpoint: str, params: Dict = None, version: str = "v3") -> Optional[Any]:
        """
        Make a request to FMP API with error handling.

        Args:
            endpoint: API endpoint (e.g., "/quote/AAPL")
            params: Additional query parameters
            version: API version (v3 or v4)

        Returns:
            JSON response or None if failed
        """
        if params is None:
            params = {}
        params["apikey"] = self.api_key

        # Handle version in endpoint
        if endpoint.startswith("/v3") or endpoint.startswith("/v4"):
            url = f"{self.BASE_URL}{endpoint}"
        else:
            url = f"{self.BASE_URL}/{version}{endpoint}"

        try:
            print(f"  → FMP Request: {url}")
            response = requests.get(url, params=params, timeout=30)
            
            # Check for API errors
            if response.status_code == 401:
                print("  → FMP Error: Invalid API key")
                return None
            elif response.status_code == 403:
                print("  → FMP Error: API limit reached or endpoint not available in plan")
                return None
            elif response.status_code != 200:
                print(f"  → FMP Error: HTTP {response.status_code}")
                return None
                
            data = response.json()
            
            # Check for error messages in response
            if isinstance(data, dict) and "Error Message" in data:
                print(f"  → FMP Error: {data['Error Message']}")
                return None
                
            return data
            
        except requests.exceptions.Timeout:
            print("  → FMP Error: Request timeout")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  → FMP Error: {e}")
            return None
        except Exception as e:
            print(f"  → FMP Error: {e}")
            return None

    # ==================== PRICE DATA ====================

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current quote for a symbol."""
        data = self._make_request(f"/quote/{symbol.upper()}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_multiple_quotes(self, symbols: List[str]) -> List[Dict]:
        """Get quotes for multiple symbols."""
        symbols_str = ",".join([s.upper() for s in symbols])
        data = self._make_request(f"/quote/{symbols_str}")
        return data if data and isinstance(data, list) else []

    def get_historical_price(
        self,
        symbol: str,
        from_date: str = None,
        to_date: str = None
    ) -> Optional[List[Dict]]:
        """Get historical daily prices."""
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = self._make_request(f"/historical-price-full/{symbol.upper()}", params)
        if data and isinstance(data, dict) and "historical" in data:
            return data["historical"]
        return None

    def get_price_on_date(self, symbol: str, target_date: str) -> Optional[Dict]:
        """
        Get price for a specific date with fallback for weekends/holidays.
        
        Args:
            symbol: Stock symbol
            target_date: Date in YYYY-MM-DD format
            
        Returns:
            Price data for that date or nearest trading day
        """
        # Parse target date
        try:
            target = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            print(f"  → Invalid date format: {target_date}")
            return None

        # Expand search range to handle weekends/holidays (up to 10 days before and after)
        from_date = (target - timedelta(days=10)).strftime("%Y-%m-%d")
        to_date = (target + timedelta(days=10)).strftime("%Y-%m-%d")

        print(f"  → Fetching historical data for {symbol} from {from_date} to {to_date}")
        
        historical = self.get_historical_price(symbol, from_date=from_date, to_date=to_date)
        
        if not historical:
            print(f"  → No historical data returned for {symbol}")
            return None

        # Sort by date descending (most recent first)
        historical_sorted = sorted(historical, key=lambda x: x.get("date", ""), reverse=True)
        
        # Find the closest date <= target_date (last trading day before or on target)
        target_str = target_date
        best_match = None
        
        for entry in historical_sorted:
            entry_date = entry.get("date", "")
            if entry_date <= target_str:
                best_match = entry
                break
        
        # If no date before target, get the first available after
        if not best_match and historical_sorted:
            # Get the oldest date in our range (closest after target)
            best_match = historical_sorted[-1]
        
        if best_match:
            print(f"  → Found price data for {symbol} on {best_match.get('date')} (requested: {target_date})")
            return best_match
            
        return None

    # ==================== MARKET MOVERS ====================

    def get_gainers(self) -> List[Dict]:
        """Get top gaining stocks."""
        # Try primary endpoint
        data = self._make_request("/stock_market/gainers")
        if data and isinstance(data, list) and len(data) > 0:
            return data
            
        # Fallback: try alternative endpoint
        print("  → Primary gainers endpoint failed, trying alternative...")
        data = self._make_request("/gainers")
        if data and isinstance(data, list):
            return data
            
        # Second fallback: use actives and filter
        print("  → Trying actives endpoint as fallback...")
        actives = self._make_request("/stock_market/actives")
        if actives and isinstance(actives, list):
            # Filter for positive change
            gainers = [s for s in actives if s.get("changesPercentage", 0) > 0]
            gainers.sort(key=lambda x: x.get("changesPercentage", 0), reverse=True)
            return gainers[:20]
            
        return []

    def get_losers(self) -> List[Dict]:
        """Get top losing stocks."""
        data = self._make_request("/stock_market/losers")
        if data and isinstance(data, list) and len(data) > 0:
            return data
            
        # Fallback
        print("  → Primary losers endpoint failed, trying alternative...")
        data = self._make_request("/losers")
        if data and isinstance(data, list):
            return data
            
        # Second fallback: use actives and filter
        print("  → Trying actives endpoint as fallback...")
        actives = self._make_request("/stock_market/actives")
        if actives and isinstance(actives, list):
            losers = [s for s in actives if s.get("changesPercentage", 0) < 0]
            losers.sort(key=lambda x: x.get("changesPercentage", 0))
            return losers[:20]
            
        return []

    def get_most_active(self) -> List[Dict]:
        """Get most actively traded stocks."""
        data = self._make_request("/stock_market/actives")
        if data and isinstance(data, list):
            return data
            
        # Fallback
        data = self._make_request("/actives")
        return data if data and isinstance(data, list) else []

    def get_sector_performance(self) -> List[Dict]:
        """Get sector performance data."""
        data = self._make_request("/sector-performance")
        # Also try sectors-performance
        if not data:
            data = self._make_request("/sectors-performance")
        return data if data and isinstance(data, list) else []

    # ==================== NEWS ====================

    def get_stock_news(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get stock news."""
        params = {"limit": limit}
        if symbol:
            params["tickers"] = symbol.upper()

        data = self._make_request("/stock_news", params)
        return data if data and isinstance(data, list) else []

    def get_general_news(self, limit: int = 10) -> List[Dict]:
        """Get general market news."""
        params = {"limit": limit}
        
        # Try v4 endpoint first
        data = self._make_request("/general_news", params, version="v4")
        if data and isinstance(data, list):
            return data
            
        # Fallback to stock_news without ticker
        data = self._make_request("/stock_news", {"limit": limit})
        return data if data and isinstance(data, list) else []

    def search_news(self, query: str, limit: int = 10) -> List[Dict]:
        """Search news by keyword."""
        # Get more news and filter
        all_news = self.get_general_news(limit=50)
        if not all_news:
            all_news = self.get_stock_news(limit=50)
        
        if not all_news:
            return []
            
        query_lower = query.lower()
        query_words = query_lower.split()

        filtered = []
        for article in all_news:
            title = article.get("title", "").lower()
            text = article.get("text", "").lower()
            
            # Check if any query word is in title or text
            if any(word in title or word in text for word in query_words):
                filtered.append(article)
                if len(filtered) >= limit:
                    break

        return filtered

    # ==================== FOREX ====================

    def get_forex_rate(self, from_currency: str, to_currency: str) -> Optional[Dict]:
        """Get forex exchange rate."""
        pair = f"{from_currency.upper()}{to_currency.upper()}"
        data = self._make_request(f"/fx/{pair}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        
        # Try quote endpoint
        data = self._make_request(f"/quote/{pair}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
            
        return None

    def get_forex_quotes(self) -> List[Dict]:
        """Get all forex quotes."""
        data = self._make_request("/fx")
        return data if data and isinstance(data, list) else []

    # ==================== INDICES ====================

    def get_index_quote(self, symbol: str) -> Optional[Dict]:
        """Get index quote (S&P 500, NASDAQ, etc.)"""
        index_map = {
            "S&P500": "^GSPC",
            "S&P 500": "^GSPC",
            "SP500": "^GSPC",
            "GSPC": "^GSPC",
            "NASDAQ": "^IXIC",
            "IXIC": "^IXIC",
            "DOW": "^DJI",
            "DJI": "^DJI",
            "DOW JONES": "^DJI",
            "RUSSELL": "^RUT",
            "RUSSELL 2000": "^RUT",
            "RUT": "^RUT",
        }

        mapped_symbol = index_map.get(symbol.upper(), symbol)
        
        # Try direct quote
        quote = self.get_quote(mapped_symbol)
        if quote:
            return quote
            
        # Try without caret
        if mapped_symbol.startswith("^"):
            quote = self.get_quote(mapped_symbol[1:])
            if quote:
                return quote
                
        return None

    def get_market_hours(self) -> Optional[Dict]:
        """Get market hours status."""
        data = self._make_request("/is-the-market-open")
        return data if data else None

    # ==================== CRYPTO ====================

    def get_crypto_quote(self, symbol: str) -> Optional[Dict]:
        """Get cryptocurrency quote."""
        if not symbol.upper().endswith("USD"):
            symbol = f"{symbol}USD"
        data = self._make_request(f"/quote/{symbol.upper()}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    # ==================== COMMODITIES ====================

    def get_commodity_quote(self, commodity: str) -> Optional[Dict]:
        """Get commodity quote."""
        commodity_map = {
            "OIL": "CLUSD",
            "CRUDE": "CLUSD",
            "CRUDE OIL": "CLUSD",
            "BRENT": "BZUSD",
            "GOLD": "GCUSD",
            "SILVER": "SIUSD",
            "NATURAL GAS": "NGUSD",
            "GAS": "NGUSD",
            "COPPER": "HGUSD",
            "PLATINUM": "PLUSD",
        }

        mapped_symbol = commodity_map.get(commodity.upper(), commodity)
        data = self._make_request(f"/quote/{mapped_symbol}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    # ==================== COMPANY INFO ====================

    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """Get company profile information."""
        data = self._make_request(f"/profile/{symbol.upper()}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def search_symbol(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for stock symbols."""
        params = {"query": query, "limit": limit}
        data = self._make_request("/search", params)
        return data if data and isinstance(data, list) else []

    # ==================== SCREENER (Alternative for Market Movers) ====================
    
    def get_stock_screener(self, market_cap_min: int = None, volume_min: int = None, 
                           limit: int = 20, is_etf: bool = False) -> List[Dict]:
        """
        Use stock screener as alternative for market data.
        
        This can be used as a fallback when gainers/losers endpoints fail.
        """
        params = {"limit": limit}
        if market_cap_min:
            params["marketCapMoreThan"] = market_cap_min
        if volume_min:
            params["volumeMoreThan"] = volume_min
        if is_etf:
            params["isEtf"] = "true"
            
        data = self._make_request("/stock-screener", params)
        return data if data and isinstance(data, list) else []