"""Internet comparison handler using Perplexity API for web-based financial queries."""
from typing import List, Dict
from src.services.perplexity_service import PerplexityService
from src.services.chat_memory import ChatMemory


class InternetComparisonHandler:
    """Handles internet-based financial queries and comparisons using Perplexity API."""

    def __init__(self, memory_max_pairs: int = 5):
        """
        Initialize the internet comparison handler.

        Args:
            memory_max_pairs: Maximum number of Q&A pairs to remember for context (default: 5)
        """
        self.perplexity_service = PerplexityService()
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

    def handle_comparison(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Handle internet comparison queries using Perplexity API.

        Args:
            user_query: User's question requiring web search
            chat_history: Previous conversation history for context

        Returns:
            Response from Perplexity API with sources
        """
        if chat_history is None:
            chat_history = []

        try:
            # Use Perplexity service to get web-based answer
            response = self.perplexity_service.query(user_query, chat_history)
            return response

        except Exception as e:
            error_msg = f"Error processing internet comparison: {str(e)}"
            print(f"❌ {error_msg}")
            return f"I encountered an error while searching for that information: {str(e)}"
