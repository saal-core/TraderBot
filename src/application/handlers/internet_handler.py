"""Handler for internet/web search queries.

Processes queries that require real-time data from the web,
such as current stock prices, market news, etc.
"""
from typing import Set
import logging

from domain.entities import QueryContext, QueryResult
from domain.interfaces import LLMProvider, WebSearchProvider
from infrastructure.observability import get_logger, get_metrics
from .query_handler import QueryHandler


logger = get_logger(__name__)
metrics = get_metrics()


class InternetQueryHandler(QueryHandler):
    """
    Handler for web search queries.
    
    Uses a web search provider (e.g., Perplexity) to fetch real-time information.
    """
    
    def __init__(
        self,
        llm: LLMProvider,
        web_search: WebSearchProvider
    ):
        """
        Initialize internet handler.
        
        Args:
            llm: LLM provider for processing
            web_search: Web search provider (e.g., Perplexity)
        """
        self._llm = llm
        self._web_search = web_search
    
    @property
    def supported_classifications(self) -> Set[str]:
        return {"general", "internet", "web", "market"}
    
    def handle(self, context: QueryContext) -> QueryResult:
        """Process an internet/web search query."""
        logger.info("Processing internet query", query=context.query[:50])
        
        with metrics.timer("internet_query"):
            try:
                # Query web search
                result = self._web_search.query(
                    context.query,
                    context.chat_history
                )
                
                metrics.increment("internet_queries", tags={"status": "success"})
                return QueryResult(
                    content=f"### 🌐 Answer (Internet)\n{result}",
                    sources=["web_search"],
                    metadata={"provider": "perplexity"}
                )
                
            except Exception as e:
                logger.error("Internet query failed", exception=e)
                metrics.increment("internet_queries", tags={"status": "error"})
                return QueryResult(
                    content="Sorry, I couldn't retrieve that information from the web. Please try again.",
                    sources=["web_search"]
                )
