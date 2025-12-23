"""Domain entities for the TraderBot application.

These are the core data structures used throughout the application,
independent of any external framework or library.
"""
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime


@dataclass
class QueryContext:
    """Context passed to all query handlers."""
    
    query: str
    session_id: str
    chat_history: list[dict]
    language: str = "English"
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_arabic(self) -> bool:
        """Check if the query language is Arabic."""
        return self.language == "Arabic"
    
    @property
    def recent_history(self) -> list[dict]:
        """Get last 8 messages from chat history."""
        return self.chat_history[-8:] if self.chat_history else []


@dataclass
class QueryResult:
    """Standardized result from query handlers."""
    
    content: str
    sql_query: Optional[str] = None
    data_frame: Any = None  # pandas DataFrame, but we avoid tight coupling
    sources: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def has_data(self) -> bool:
        """Check if result contains data."""
        if self.data_frame is None:
            return False
        # Check for pandas DataFrame emptiness
        if hasattr(self.data_frame, 'empty'):
            return not self.data_frame.empty
        return True


@dataclass
class MarketQuote:
    """Market quote data for a stock symbol."""
    
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_positive(self) -> bool:
        """Check if the change is positive."""
        return self.change >= 0


@dataclass
class PortfolioSummary:
    """Summary of a portfolio's holdings."""
    
    name: str
    total_value: float
    currency: str = "AED"
    holdings_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
