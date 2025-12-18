"""FMP (Financial Modeling Prep) API service for real-time financial data."""
import logging
import os
from typing import Dict, List, Optional, Any
import requests
from dotenv import load_dotenv

load_dotenv()


class FMPService:
    """Service for fetching financial data from FMP API."""

    def __init__(self):
        """Initialize FMP service."""
        self.api_key = os.getenv("FMP_API_KEY")
        self.base_url = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/api/v3")
        
        if not self.api_key:
            logging.warning("FMP_API_KEY not found in environment variables")

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        """
        Make a request to FMP API.
        
        Args:
            endpoint: API endpoint path
            params: Additional query parameters
            
        Returns:
            JSON response or None on error
        """
        if not self.api_key:
            logging.error("FMP API key not configured")
            return None
            
        url = f"{self.base_url}/{endpoint}"
        request_params = {"apikey": self.api_key}
        
        if params:
            request_params.update(params)
            
        try:
            response = requests.get(url, params=request_params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"FMP API request failed: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time stock quote.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            Quote data dictionary or None on error
        """
        data = self._make_request(f"quote/{symbol.upper()}")
        
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_historical_prices(
        self, 
        symbol: str, 
        from_date: str = None, 
        to_date: str = None
    ) -> Optional[List[Dict]]:
        """
        Get historical price data for a stock.
        
        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of historical price data or None on error
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        data = self._make_request(f"historical-price-full/{symbol.upper()}", params)
        
        if data and "historical" in data:
            return data["historical"][:30]  # Return last 30 days by default
        return None

    def get_stock_news(self, symbol: str = None, limit: int = 10) -> Optional[List[Dict]]:
        """
        Get latest stock news.
        
        Args:
            symbol: Optional stock symbol to filter news
            limit: Maximum number of news items
            
        Returns:
            List of news articles or None on error
        """
        params = {"limit": limit}
        if symbol:
            params["tickers"] = symbol.upper()
            
        data = self._make_request("stock_news", params)
        
        if data and isinstance(data, list):
            return data
        return None

    def get_market_index(self, index_symbol: str = "^GSPC") -> Optional[Dict]:
        """
        Get market index data (S&P 500, NASDAQ, etc.).
        
        Args:
            index_symbol: Index symbol (^GSPC for S&P 500, ^IXIC for NASDAQ, ^DJI for Dow Jones)
            
        Returns:
            Index quote data or None on error
        """
        # Map common names to symbols
        index_mapping = {
            "sp500": "^GSPC",
            "s&p500": "^GSPC",
            "s&p 500": "^GSPC",
            "nasdaq": "^IXIC",
            "dow": "^DJI",
            "dow jones": "^DJI",
        }
        
        # Normalize the symbol
        normalized = index_symbol.lower().strip()
        symbol = index_mapping.get(normalized, index_symbol)
        
        # URL encode the ^ character
        encoded_symbol = symbol.replace("^", "%5E")
        data = self._make_request(f"quote/{encoded_symbol}")
        
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def detect_query_type(self, query: str) -> str:
        """
        Detect the type of financial query.
        
        Args:
            query: User's query string
            
        Returns:
            Query type: 'quote', 'historical', 'news', 'market_index', or 'unknown'
        """
        query_lower = query.lower()
        
        # Market index indicators
        index_keywords = [
            "s&p", "sp500", "s&p 500", "nasdaq", "dow jones", "dow", 
            "market index", "market performance", "index"
        ]
        if any(kw in query_lower for kw in index_keywords):
            return "market_index"
        
        # News indicators
        news_keywords = ["news", "latest", "headlines", "article", "report"]
        if any(kw in query_lower for kw in news_keywords):
            return "news"
        
        # Historical price indicators
        historical_keywords = [
            "history", "historical", "past", "trend", "over time",
            "last week", "last month", "last year", "performance"
        ]
        if any(kw in query_lower for kw in historical_keywords):
            return "historical"
        
        # Quote indicators (current price)
        quote_keywords = [
            "price", "current", "now", "today", "trading at",
            "worth", "value", "quote"
        ]
        if any(kw in query_lower for kw in quote_keywords):
            return "quote"
        
        return "unknown"

    def extract_symbol(self, query: str) -> Optional[str]:
        """
        Extract stock symbol from query.
        
        Args:
            query: User's query string
            
        Returns:
            Extracted symbol or None
        """
        # Common stock name to symbol mapping
        stock_mapping = {
            "apple": "AAPL",
            "tesla": "TSLA",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "meta": "META",
            "facebook": "META",
            "nvidia": "NVDA",
            "netflix": "NFLX",
            "bitcoin": "BTC-USD",
            "ethereum": "ETH-USD",
        }
        
        query_lower = query.lower()
        
        # Check for common stock names
        for name, symbol in stock_mapping.items():
            if name in query_lower:
                return symbol
        
        # Look for uppercase symbols (2-5 letters)
        import re
        symbols = re.findall(r'\b[A-Z]{2,5}\b', query)
        if symbols:
            return symbols[0]
        
        return None

    def fetch_data(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Fetch appropriate data based on query type.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary with 'type', 'data', and 'symbol' keys, or None
        """
        query_type = self.detect_query_type(query)
        
        if query_type == "market_index":
            # Extract index name or default to S&P 500
            query_lower = query.lower()
            if "nasdaq" in query_lower:
                index = "^IXIC"
                index_name = "NASDAQ"
            elif "dow" in query_lower:
                index = "^DJI"
                index_name = "Dow Jones"
            else:
                index = "^GSPC"
                index_name = "S&P 500"
                
            data = self.get_market_index(index)
            if data:
                return {
                    "type": "market_index",
                    "data": data,
                    "index_name": index_name
                }
        
        elif query_type == "news":
            symbol = self.extract_symbol(query)
            data = self.get_stock_news(symbol, limit=5)
            if data:
                return {
                    "type": "news",
                    "data": data,
                    "symbol": symbol
                }
        
        elif query_type == "historical":
            symbol = self.extract_symbol(query)
            if symbol:
                data = self.get_historical_prices(symbol)
                if data:
                    return {
                        "type": "historical",
                        "data": data,
                        "symbol": symbol
                    }
        
        elif query_type == "quote":
            symbol = self.extract_symbol(query)
            if symbol:
                data = self.get_quote(symbol)
                if data:
                    return {
                        "type": "quote",
                        "data": data,
                        "symbol": symbol
                    }
        
        return None
