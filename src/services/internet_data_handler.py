"""Internet data fetching handler using Perplexity API for real-time financial information."""
from typing import List, Dict
from src.services.perplexity_service import PerplexityService
from src.services.chat_memory import ChatMemory


class InternetDataHandler:
    """Handles internet-based financial data fetching using Perplexity API."""

    def __init__(self, memory_max_pairs: int = 5):
        """
        Initialize the internet data handler.

        Args:
            memory_max_pairs: Maximum number of Q&A pairs to remember for context (default: 5)
        """
        self.perplexity_service = PerplexityService()
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

    def fetch_data(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Fetch real-time financial data from the internet using Perplexity API.

        Handles queries such as:
        - Current stock prices (e.g., "What's the current price of Tesla?")
        - Latest news (e.g., "Give me the latest news on Apple")
        - Market performance (e.g., "What's the S&P 500 performance today?")
        - Cryptocurrency prices, commodities, forex rates, etc.

        Args:
            user_query: User's question requiring web search for financial data
            chat_history: Previous conversation history for context

        Returns:
            Response from Perplexity API with sources and real-time data
        """
        if chat_history is None:
            chat_history = []

        try:
            # Clean chat history - remove DataFrames and keep only role/content
            clean_history = []
            for msg in chat_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    clean_msg = {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                    clean_history.append(clean_msg)

            # Use Perplexity service to get web-based real-time data
            response = self.perplexity_service.query(user_query, clean_history)
            return response

        except Exception as e:
            error_msg = f"Error fetching internet data: {str(e)}"
            print(f"❌ {error_msg}")
            return f"I encountered an error while fetching that information: {str(e)}"
