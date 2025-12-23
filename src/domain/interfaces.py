"""Abstract interfaces (Protocols) for the domain layer.

These protocols define the contracts that infrastructure implementations must fulfill.
This allows the domain and application layers to be independent of external services.
"""
from typing import Protocol, Optional, Any, runtime_checkable
import pandas as pd

from .result import Result


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM (Large Language Model) providers."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        ...
    
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Generate a response from a conversation."""
        ...
    
    def test_connection(self) -> bool:
        """Test if the LLM service is available."""
        ...


@runtime_checkable
class MarketDataProvider(Protocol):
    """Interface for market data providers (e.g., FMP API)."""
    
    def get_quote(self, symbol: str) -> Optional[dict]:
        """Get current quote for a symbol."""
        ...
    
    def get_historical(
        self,
        symbol: str,
        from_date: str,
        to_date: str
    ) -> list[dict]:
        """Get historical price data for a symbol."""
        ...
    
    def search_symbol(self, query: str) -> list[dict]:
        """Search for symbols matching a query."""
        ...


@runtime_checkable
class PortfolioRepository(Protocol):
    """Interface for portfolio data access."""
    
    def connect(self) -> bool:
        """Establish database connection."""
        ...
    
    def get_portfolios(self) -> list[str]:
        """Get list of portfolio names."""
        ...
    
    def execute_query(self, sql: str) -> Result[pd.DataFrame]:
        """Execute a SQL query and return results."""
        ...
    
    def generate_sql(self, question: str) -> Optional[str]:
        """Generate SQL from natural language question."""
        ...


@runtime_checkable
class CacheProvider(Protocol):
    """Interface for caching providers."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        ...
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        ...
    
    def clear(self) -> None:
        """Clear all cached values."""
        ...


@runtime_checkable
class TranslationProvider(Protocol):
    """Interface for translation services."""
    
    def translate(self, text: str, target_language: str) -> str:
        """Translate text to target language."""
        ...
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        ...


@runtime_checkable
class WebSearchProvider(Protocol):
    """Interface for web search providers (e.g., Perplexity)."""
    
    def query(self, question: str, chat_history: list[dict]) -> str:
        """Search the web and return a response."""
        ...
