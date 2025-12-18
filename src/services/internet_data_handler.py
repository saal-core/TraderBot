"""Internet data fetching handler using FMP API with Ollama formatting and Perplexity fallback."""
import logging
from typing import List, Dict, Any
from src.services.fmp_service import FMPService
from src.services.perplexity_service import PerplexityService
from src.services.ollama_service import OllamaService
from src.services.chat_memory import ChatMemory


# System prompt for formatting FMP data
FMP_FORMATTER_PROMPT = """You are a helpful financial data assistant. Your task is to take raw financial data and present it in a clear, conversational manner.

Format the data in a user-friendly way:
- Use bullet points for multiple data points
- Highlight important numbers (prices, changes, percentages)
- Add brief context when helpful
- Keep the response concise but informative
- For news, summarize the key points

Do NOT add information that isn't in the provided data. Only format and present what you're given."""


class InternetDataHandler:
    """Handles internet-based financial data fetching using FMP API with Perplexity fallback."""

    def __init__(self, memory_max_pairs: int = 5):
        """
        Initialize the internet data handler.

        Args:
            memory_max_pairs: Maximum number of Q&A pairs to remember for context (default: 5)
        """
        self.fmp_service = FMPService()
        self.perplexity_service = PerplexityService()
        self.ollama_service = OllamaService()
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

    def _format_quote_data(self, data: Dict) -> str:
        """Format stock quote data for Ollama."""
        return f"""Stock Quote Data:
- Symbol: {data.get('symbol', 'N/A')}
- Name: {data.get('name', 'N/A')}
- Current Price: ${data.get('price', 'N/A')}
- Change: {data.get('change', 'N/A')} ({data.get('changesPercentage', 'N/A')}%)
- Day High: ${data.get('dayHigh', 'N/A')}
- Day Low: ${data.get('dayLow', 'N/A')}
- Open: ${data.get('open', 'N/A')}
- Previous Close: ${data.get('previousClose', 'N/A')}
- Volume: {data.get('volume', 'N/A'):,}
- Market Cap: ${data.get('marketCap', 'N/A'):,}
- 52 Week High: ${data.get('yearHigh', 'N/A')}
- 52 Week Low: ${data.get('yearLow', 'N/A')}"""

    def _format_historical_data(self, data: List[Dict], symbol: str) -> str:
        """Format historical price data for Ollama."""
        if not data:
            return "No historical data available."
        
        lines = [f"Historical Price Data for {symbol} (last {len(data)} trading days):"]
        for i, day in enumerate(data[:10]):  # Show last 10 days
            lines.append(
                f"- {day.get('date', 'N/A')}: Open ${day.get('open', 'N/A')}, "
                f"Close ${day.get('close', 'N/A')}, "
                f"High ${day.get('high', 'N/A')}, "
                f"Low ${day.get('low', 'N/A')}"
            )
        
        if len(data) > 10:
            lines.append(f"... and {len(data) - 10} more days of data")
            
        return "\n".join(lines)

    def _format_news_data(self, data: List[Dict], symbol: str = None) -> str:
        """Format news data for Ollama."""
        if not data:
            return "No news articles available."
        
        header = f"Latest News" + (f" for {symbol}" if symbol else "") + ":"
        lines = [header]
        
        for article in data[:5]:
            lines.append(f"""
- Title: {article.get('title', 'N/A')}
  Source: {article.get('site', 'N/A')}
  Published: {article.get('publishedDate', 'N/A')}
  Summary: {article.get('text', 'N/A')[:200]}...""")
        
        return "\n".join(lines)

    def _format_market_index_data(self, data: Dict, index_name: str) -> str:
        """Format market index data for Ollama."""
        return f"""{index_name} Market Index Data:
- Current Value: {data.get('price', 'N/A'):,.2f}
- Change: {data.get('change', 'N/A'):,.2f} ({data.get('changesPercentage', 'N/A'):.2f}%)
- Day High: {data.get('dayHigh', 'N/A'):,.2f}
- Day Low: {data.get('dayLow', 'N/A'):,.2f}
- Open: {data.get('open', 'N/A'):,.2f}
- Previous Close: {data.get('previousClose', 'N/A'):,.2f}
- Year High: {data.get('yearHigh', 'N/A'):,.2f}
- Year Low: {data.get('yearLow', 'N/A'):,.2f}"""

    def _format_with_ollama(self, user_query: str, raw_data: str) -> str:
        """
        Use Ollama to format raw FMP data into a natural language response.
        
        Args:
            user_query: Original user question
            raw_data: Formatted raw data string
            
        Returns:
            Natural language response
        """
        messages = [
            {"role": "system", "content": FMP_FORMATTER_PROMPT},
            {"role": "user", "content": f"""User asked: "{user_query}"

Here is the raw financial data:

{raw_data}

Please format this data as a helpful response to the user's question."""}
        ]
        
        try:
            response = self.ollama_service.generate_response(messages, temperature=0.3)
            return response
        except Exception as e:
            logging.error(f"Ollama formatting failed: {e}")
            # Return raw data if Ollama fails
            return raw_data

    def fetch_data(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Fetch real-time financial data using FMP API, with Perplexity as fallback.

        Handles queries such as:
        - Current stock prices (e.g., "What's the current price of Tesla?")
        - Latest news (e.g., "Give me the latest news on Apple")
        - Market performance (e.g., "What's the S&P 500 performance today?")
        - Historical prices (e.g., "Show me Tesla's price history")

        Args:
            user_query: User's question requiring web search for financial data
            chat_history: Previous conversation history for context

        Returns:
            Formatted response with financial data
        """
        if chat_history is None:
            chat_history = []

        try:
            # Try FMP first
            fmp_result = self.fmp_service.fetch_data(user_query)
            
            if fmp_result:
                data_type = fmp_result.get("type")
                data = fmp_result.get("data")
                
                # Format the raw data based on type
                if data_type == "quote":
                    raw_formatted = self._format_quote_data(data)
                elif data_type == "historical":
                    raw_formatted = self._format_historical_data(data, fmp_result.get("symbol"))
                elif data_type == "news":
                    raw_formatted = self._format_news_data(data, fmp_result.get("symbol"))
                elif data_type == "market_index":
                    raw_formatted = self._format_market_index_data(data, fmp_result.get("index_name"))
                else:
                    raw_formatted = str(data)
                
                # Use Ollama to create a natural language response
                response = self._format_with_ollama(user_query, raw_formatted)
                logging.info(f"FMP data fetched and formatted successfully for query type: {data_type}")
                return response
            
            # Fall back to Perplexity if FMP couldn't handle the query
            logging.info("FMP couldn't handle query, falling back to Perplexity")
            return self._fallback_to_perplexity(user_query, chat_history)

        except Exception as e:
            error_msg = f"Error fetching internet data: {str(e)}"
            logging.error(f"❌ {error_msg}")
            
            # Try Perplexity as fallback on error
            try:
                logging.info("FMP failed, attempting Perplexity fallback")
                return self._fallback_to_perplexity(user_query, chat_history)
            except Exception as fallback_error:
                return f"I encountered an error while fetching that information: {str(e)}"

    def _fallback_to_perplexity(self, user_query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Fall back to Perplexity API for queries FMP can't handle.
        
        Args:
            user_query: User's question
            chat_history: Previous conversation history
            
        Returns:
            Response from Perplexity API
        """
        # Clean chat history - remove DataFrames and keep only role/content
        clean_history = []
        for msg in chat_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                clean_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                clean_history.append(clean_msg)

        # Use Perplexity service as fallback
        response = self.perplexity_service.query(user_query, clean_history)
        return response
