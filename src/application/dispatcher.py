"""Query dispatcher for routing queries to appropriate handlers.

Routes queries based on classification to registered handlers using the 
Strategy pattern. Supports fallback handlers and classification caching.
"""
from typing import List, Optional, Callable
import re

from domain.entities import QueryContext, QueryResult
from domain.interfaces import LLMProvider, CacheProvider
from infrastructure.observability import get_logger, get_metrics
from infrastructure.cache import MemoryCache
from .handlers.query_handler import QueryHandler


logger = get_logger(__name__)
metrics = get_metrics()


class QueryClassifier:
    """
    Classifies queries into categories using a 3-tier approach:
    1. Regex patterns (fastest)
    2. Keyword matching
    3. LLM fallback (slowest, cached)
    """
    
    def __init__(
        self,
        llm: LLMProvider,
        cache: Optional[CacheProvider] = None
    ):
        self._llm = llm
        self._cache = cache or MemoryCache(max_size=500, default_ttl=300)
        
        # Tier 1: Pre-compiled regex patterns
        self._greeting_patterns = [
            re.compile(r"^(hi|hello|hey|good\s*(morning|afternoon|evening))[\s!?.]*$", re.I),
            re.compile(r"^how\s+are\s+you", re.I),
            re.compile(r"^what's\s+up", re.I),
        ]
        
        self._portfolio_patterns = [
            re.compile(r"\b(my|our)\s+(portfolio|holdings|investments|stocks|dividends)", re.I),
            re.compile(r"\bshow\s+(me\s+)?(my|the)\s+portfolio", re.I),
            re.compile(r"\b(list|display)\s+(all\s+)?(my\s+)?portfolios", re.I),
            re.compile(r"\bwhat\s+(are|is)\s+my\s+(portfolio|holdings)", re.I),
            re.compile(r"\b(total|current)\s+(portfolio\s+)?value", re.I),
        ]
        
        # Tier 2: Keywords for classification
        self._portfolio_keywords = {
            "portfolio", "holdings", "invested", "dividends", "my stocks",
            "my shares", "allocation", "sector", "benchmark"
        }
        
        self._general_keywords = {
            "current price", "stock price", "market", "news", "weather",
            "what is", "who is", "how to", "explain", "cagr", "calculate"
        }
    
    def classify(self, query: str) -> str:
        """
        Classify a query using the 3-tier approach.
        
        Args:
            query: User query string
            
        Returns:
            Classification: "portfolio", "general", or "other"
        """
        query_lower = query.lower().strip()
        
        # Tier 1: Regex patterns (~10ms)
        for pattern in self._greeting_patterns:
            if pattern.match(query_lower):
                logger.debug("Classified via regex as greeting", query=query[:30])
                return "other"
        
        for pattern in self._portfolio_patterns:
            if pattern.search(query_lower):
                logger.debug("Classified via regex as portfolio", query=query[:30])
                return "portfolio"
        
        # Tier 2: Keyword matching (~50ms)
        for keyword in self._portfolio_keywords:
            if keyword in query_lower:
                logger.debug("Classified via keyword as portfolio", query=query[:30])
                return "portfolio"
        
        for keyword in self._general_keywords:
            if keyword in query_lower:
                logger.debug("Classified via keyword as general", query=query[:30])
                return "general"
        
        # Tier 3: LLM fallback with caching
        cache_key = f"classify:{hash(query_lower)}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Classification from cache", query=query[:30], classification=cached)
            return cached
        
        classification = self._classify_with_llm(query)
        self._cache.set(cache_key, classification, ttl=300)
        return classification
    
    def _classify_with_llm(self, query: str) -> str:
        """Use LLM for classification when patterns fail."""
        prompt = f"""Classify this query into one category: portfolio, general, or other.

Rules:
- portfolio: Questions about user's personal portfolio data, holdings, investments
- general: Questions requiring web search, current prices, market news
- other: Greetings, small talk, casual conversation

Query: "{query}"
Category:"""
        
        try:
            response = self._llm.generate(prompt, temperature=0.1)
            response = response.strip().lower()
            
            if "portfolio" in response:
                return "portfolio"
            if "general" in response:
                return "general"
            return "other"
            
        except Exception as e:
            logger.error("LLM classification failed", exception=e)
            return "other"


class QueryDispatcher:
    """
    Dispatches queries to registered handlers based on classification.
    
    Usage:
        dispatcher = QueryDispatcher(classifier)
        dispatcher.register(DatabaseQueryHandler(...))
        dispatcher.register(InternetQueryHandler(...))
        dispatcher.register(GreetingHandler(...))
        
        result = dispatcher.dispatch(context)
    """
    
    def __init__(
        self,
        classifier: QueryClassifier,
        default_handler: Optional[QueryHandler] = None
    ):
        """
        Initialize dispatcher.
        
        Args:
            classifier: Query classifier for determining handler
            default_handler: Fallback handler if no match found
        """
        self._classifier = classifier
        self._handlers: List[QueryHandler] = []
        self._default_handler = default_handler
    
    def register(self, handler: QueryHandler) -> None:
        """Register a handler with the dispatcher."""
        self._handlers.append(handler)
        logger.info(f"Registered handler: {handler.name}")
    
    def get_handler(self, classification: str) -> Optional[QueryHandler]:
        """Get the handler for a classification."""
        for handler in self._handlers:
            if handler.can_handle(classification):
                return handler
        return self._default_handler
    
    def dispatch(self, context: QueryContext) -> QueryResult:
        """
        Classify and dispatch a query to the appropriate handler.
        
        Args:
            context: Query context
            
        Returns:
            QueryResult from the handler
        """
        # Classify the query
        with metrics.timer("query_classification"):
            classification = self._classifier.classify(context.query)
        
        logger.info(
            "Query classified",
            query=context.query[:50],
            classification=classification
        )
        metrics.increment("queries_classified", tags={"type": classification})
        
        # Get handler
        handler = self.get_handler(classification)
        if handler is None:
            logger.warning("No handler for classification", classification=classification)
            return QueryResult(
                content="I'm not sure how to help with that. Please try rephrasing your question.",
                sources=[]
            )
        
        # Dispatch to handler
        logger.debug(f"Dispatching to {handler.name}")
        with metrics.timer("query_handling", tags={"handler": handler.name}):
            return handler.handle(context)
