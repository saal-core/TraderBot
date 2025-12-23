"""Handler for greetings and conversational queries.

Processes casual conversation, small talk, and general questions
that don't require database access or web searches.
"""
from typing import Set

from domain.entities import QueryContext, QueryResult
from domain.interfaces import LLMProvider
from infrastructure.observability import get_logger, get_metrics
from .query_handler import QueryHandler


logger = get_logger(__name__)
metrics = get_metrics()


class GreetingHandler(QueryHandler):
    """
    Handler for greetings and conversational queries.
    
    Provides friendly responses to casual conversation and general questions.
    """
    
    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str = ""
    ):
        """
        Initialize greeting handler.
        
        Args:
            llm: LLM provider for responses
            system_prompt: Optional custom system prompt
        """
        self._llm = llm
        self._system_prompt = system_prompt or self._default_prompt()
    
    @property
    def supported_classifications(self) -> Set[str]:
        return {"other", "greeting", "conversational", "chat"}
    
    def handle(self, context: QueryContext) -> QueryResult:
        """Process a greeting or conversational query."""
        logger.debug("Processing conversational query", query=context.query[:50])
        
        with metrics.timer("greeting_query"):
            try:
                # Build messages with context
                messages = [
                    {"role": "system", "content": self._system_prompt}
                ]
                
                # Add recent chat history
                for msg in context.recent_history:
                    if msg["role"] in ["user", "assistant"]:
                        messages.append(msg)
                
                # Add current query
                messages.append({"role": "user", "content": context.query})
                
                # Generate response
                response = self._llm.chat(messages, temperature=0.7)
                
                metrics.increment("greeting_queries", tags={"status": "success"})
                return QueryResult(
                    content=f"### 🤖 Answer (Conversational)\n{response}",
                    sources=[]
                )
                
            except Exception as e:
                logger.error("Greeting handler failed", exception=e)
                metrics.increment("greeting_queries", tags={"status": "error"})
                return QueryResult(
                    content="Hello! How can I help you with your financial questions today?",
                    sources=[]
                )
    
    def _default_prompt(self) -> str:
        """Return default system prompt."""
        return (
            "You are a helpful and professional AI financial assistant. "
            "Answer only finance related questions, Small Talk and Casual conversation questions. "
            "The response should be helpful and clear for the user to understand and not too long. "
            "If users greet you like 'hi' or 'hey' respond in a friendly manner. "
            "Don't use technical jargon unless specifically asked. "
            "Ensure your response is ALWAYS plain ASCII text. "
            "Ensure that you respond in the same language as the user's question."
        )
