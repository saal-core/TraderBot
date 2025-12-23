"""Base query handler interface using Strategy pattern.

All query handlers must implement this interface, allowing the dispatcher
to route queries to the appropriate handler based on classification.
"""
from abc import ABC, abstractmethod
from typing import Set

from domain.entities import QueryContext, QueryResult


class QueryHandler(ABC):
    """
    Abstract base class for query handlers.
    
    Implements the Strategy pattern - each handler knows:
    1. What types of queries it can handle
    2. How to process those queries
    
    Usage:
        class MyHandler(QueryHandler):
            def can_handle(self, classification: str) -> bool:
                return classification == "my_type"
            
            def handle(self, context: QueryContext) -> QueryResult:
                # Process the query
                return QueryResult(content="Response")
    """
    
    @property
    @abstractmethod
    def supported_classifications(self) -> Set[str]:
        """
        Return the set of classifications this handler supports.
        
        Returns:
            Set of classification strings (e.g., {"portfolio", "database"})
        """
        ...
    
    def can_handle(self, classification: str) -> bool:
        """
        Check if this handler can process the given classification.
        
        Args:
            classification: Query classification string
            
        Returns:
            True if this handler can process the classification
        """
        return classification.lower() in self.supported_classifications
    
    @abstractmethod
    def handle(self, context: QueryContext) -> QueryResult:
        """
        Process the query and return a result.
        
        Args:
            context: Query context with query, history, language, etc.
            
        Returns:
            QueryResult with response content and metadata
        """
        ...
    
    @property
    def name(self) -> str:
        """Handler name for logging and debugging."""
        return self.__class__.__name__
