# src/services/internet_data_handler.py
"""Enhanced Internet data handler using FMP API for comprehensive financial data."""
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import re
from src.services.fmp_service import FMPService
from src.services.perplexity_service import PerplexityService
from src.services.chat_memory import ChatMemory
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config
from src.config.prompts import INTERNET_DATA_EXPLANATION_PROMPT
import time
import os
from dotenv import load_dotenv

load_dotenv()


class InternetDataHandler:
    """Handles internet-based financial data fetching using FMP API with intelligent query processing.
    
    Uses Qwen3-30B-3B on H100 for explanation generation (optimized for speed and quality).
    """

    def __init__(self, memory_max_pairs: int = 5):
        """
        Initialize the internet data handler.

        Args:
            memory_max_pairs: Maximum number of Q&A pairs to remember
            
        Note:
            Internet data explanations always use Qwen H100 for optimal performance.
            If Qwen is unavailable, falls back to Ollama (gpt-oss:20b).
        """
        self.fmp = FMPService()
        self.perplexity = PerplexityService()
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

        # Initialize Ollama for query analysis and response generation
        ollama_config = get_ollama_config()
        self.llm = Ollama(
            model=ollama_config["model_name"],
            base_url=ollama_config["base_url"],
            temperature=0.2
        )

        # Initialize explanation LLM: Always use Qwen H100 for faster explanations
        self.explanation_llm = ChatOpenAI(
            model="Qwen3-30B-A3B",
            base_url=os.getenv("QWEN_BASE_URL", "http://192.168.71.72:8080/v1"),
            api_key=os.getenv("QWEN_API_KEY", "123"),
            temperature=0.3
        )
        self.llm_type = "Qwen H100"

        # Symbol name mapping for common companies
        self.company_to_symbol = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "meta": "META",
            "facebook": "META",
            "netflix": "NFLX",
            "disney": "DIS",
            "nike": "NKE",
            "coca-cola": "KO",
            "pepsi": "PEP",
            "walmart": "WMT",
            "boeing": "BA",
            "intel": "INTC",
            "amd": "AMD",
            "spotify": "SPOT",
            "uber": "UBER",
            "airbnb": "ABNB",
            "zoom": "ZM",
            "paypal": "PYPL",
            "visa": "V",
            "mastercard": "MA",
            "jpmorgan": "JPM",
            "goldman": "GS",
            "berkshire": "BRK-B",
        }

    def _resolve_symbol(self, query: str) -> Optional[str]:
        """
        Extract and resolve stock symbol from query.

        Args:
            query: User query containing company name or symbol

        Returns:
            Resolved stock symbol or None
        """
        query_lower = query.lower()

        # Check direct symbol mapping
        for company, symbol in self.company_to_symbol.items():
            if company in query_lower:
                return symbol

        # Try to find ticker pattern (1-5 uppercase letters)
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
        if ticker_match:
            potential_symbol = ticker_match.group(1)
            # Verify it's a valid symbol by checking with FMP
            quote = self.fmp.get_quote(potential_symbol)
            if quote:
                return potential_symbol

        # Use FMP search as fallback
        # Extract potential company names
        words = query.split()
        for word in words:
            if len(word) > 2 and word[0].isupper():
                results = self.fmp.search_symbol(word, limit=1)
                if results:
                    return results[0].get("symbol")

        return None

    def _parse_date_from_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse date references from query.

        Args:
            query: User query with date references

        Returns:
            Tuple of (from_date, to_date) in YYYY-MM-DD format
        """
        query_lower = query.lower()
        today = datetime.now()
        current_year = today.year

        # Handle "in January", "in February", etc.
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }

        for month_name, month_num in months.items():
            if month_name in query_lower:
                # Assume current year if month is in the past, else last year
                year = current_year if month_num <= today.month else current_year - 1
                from_date = f"{year}-{month_num:02d}-01"
                # Get last day of month
                if month_num == 12:
                    to_date = f"{year}-12-31"
                else:
                    next_month = datetime(year, month_num + 1, 1) - timedelta(days=1)
                    to_date = next_month.strftime("%Y-%m-%d")
                return from_date, to_date

        # Handle "this week", "last week", etc.
        if "this week" in query_lower:
            start = today - timedelta(days=today.weekday())
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

        if "last week" in query_lower:
            start = today - timedelta(days=today.weekday() + 7)
            end = start + timedelta(days=6)
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        if "this month" in query_lower:
            start = today.replace(day=1)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

        if "last month" in query_lower:
            first_of_this_month = today.replace(day=1)
            last_month_end = first_of_this_month - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            return last_month_start.strftime("%Y-%m-%d"), last_month_end.strftime("%Y-%m-%d")

        return None, None

    def _parse_investment_amount(self, query: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse investment amount and currency from query.

        Args:
            query: User query with investment amount

        Returns:
            Tuple of (amount, currency)
        """
        # Pattern: AED 10,000 or $10,000 or 10000 USD
        patterns = [
            r'AED\s*([\d,]+(?:\.\d+)?)',
            r'USD\s*([\d,]+(?:\.\d+)?)',
            r'\$\s*([\d,]+(?:\.\d+)?)',
            r'([\d,]+(?:\.\d+)?)\s*AED',
            r'([\d,]+(?:\.\d+)?)\s*USD',
            r'([\d,]+(?:\.\d+)?)\s*dollars?',
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                amount = float(match.group(1).replace(",", ""))
                currency = "AED" if "AED" in pattern.upper() or "aed" in query.lower() else "USD"
                return amount, currency

        return None, None

    def _classify_query(self, query: str) -> str:
        """
        Classify the type of internet query.

        Returns one of:
        - HYPOTHETICAL_INVESTMENT
        - MARKET_MOVERS
        - NEWS
        - CURRENT_PRICE
        - COMMODITY_PRICE
        - CRYPTO_PRICE
        - INDEX_PERFORMANCE
        - FOREX
        - GENERAL
        """
        query_lower = query.lower()

        # Hypothetical investment
        if any(phrase in query_lower for phrase in [
            "if i had invested", "had i invested", "if i invested",
            "would it be worth", "would be worth", "how much would"
        ]):
            return "HYPOTHETICAL_INVESTMENT"

        # Market movers
        if any(phrase in query_lower for phrase in [
            "top gainers", "top losers", "biggest gainers", "biggest losers",
            "most active", "trending stocks", "market movers"
        ]):
            return "MARKET_MOVERS"

        # News
        if any(phrase in query_lower for phrase in [
            "news", "latest on", "what's happening", "any updates"
        ]):
            return "NEWS"

        # Commodities
        if any(phrase in query_lower for phrase in [
            "oil price", "gold price", "silver price", "commodity",
            "crude oil", "natural gas"
        ]):
            return "COMMODITY_PRICE"

        # Crypto
        if any(phrase in query_lower for phrase in [
            "bitcoin", "ethereum", "crypto", "btc", "eth"
        ]):
            return "CRYPTO_PRICE"

        # Index
        if any(phrase in query_lower for phrase in [
            "s&p 500", "sp500", "nasdaq", "dow jones", "dow",
            "russell", "market index"
        ]):
            return "INDEX_PERFORMANCE"

        # Forex
        if any(phrase in query_lower for phrase in [
            "exchange rate", "forex", "currency", "aed to usd", "usd to"
        ]):
            return "FOREX"

        # Current price (default for stock queries)
        if any(phrase in query_lower for phrase in [
            "price of", "current price", "stock price", "how much is"
        ]):
            return "CURRENT_PRICE"

        return "GENERAL"
# src/services/internet_data_handler.py (partial update - key methods)

    def _handle_hypothetical_investment(self, query: str) -> str:
        """Handle 'If I had invested X in Y' queries with improved date handling."""
        print("  → Processing hypothetical investment query...")

        # Extract components
        symbol = self._resolve_symbol(query)
        amount, currency = self._parse_investment_amount(query)
        from_date, _ = self._parse_date_from_query(query)

        if not symbol:
            return "I couldn't identify the stock you're asking about. Please specify a valid stock symbol or company name (e.g., 'NVIDIA', 'Apple', 'TSLA')."

        if not amount:
            return "I couldn't determine the investment amount. Please specify an amount like 'AED 10,000' or '$10,000'."

        if not from_date:
            # Default to beginning of current year
            current_year = datetime.now().year
            from_date = f"{current_year}-01-02"  # Use Jan 2 to avoid New Year's Day

        print(f"  → Symbol: {symbol}, Amount: {amount} {currency}, From: {from_date}")

        # Get historical price with fallback for holidays/weekends
        historical_price = self.fmp.get_price_on_date(symbol, from_date)
        if not historical_price:
            # Try alternative: get the first available price of the month
            try:
                date_obj = datetime.strptime(from_date, "%Y-%m-%d")
                # Try getting a range for the month
                month_start = date_obj.replace(day=1).strftime("%Y-%m-%d")
                month_end = (date_obj.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                month_end_str = month_end.strftime("%Y-%m-%d")
                
                historical_data = self.fmp.get_historical_price(symbol, from_date=month_start, to_date=month_end_str)
                if historical_data and len(historical_data) > 0:
                    # Get the earliest date in the range
                    historical_data.sort(key=lambda x: x.get("date", ""))
                    historical_price = historical_data[0]
                    print(f"  → Used fallback: found price on {historical_price.get('date')}")
            except Exception as e:
                print(f"  → Fallback failed: {e}")
            
            if not historical_price:
                return f"I couldn't find historical price data for {symbol} around {from_date}. The market may have been closed on that date. Please try a different date."

        # Get current price
        current_quote = self.fmp.get_quote(symbol)
        if not current_quote:
            return f"I couldn't fetch the current price for {symbol}. Please try again later."

        # Extract prices
        historical_close = historical_price.get("close") or historical_price.get("adjClose")
        if not historical_close:
            return f"Historical price data for {symbol} is incomplete. Please try a different date."
            
        current_price = current_quote.get("price")
        if not current_price:
            return f"Current price data for {symbol} is unavailable. Please try again later."
            
        company_name = current_quote.get("name", symbol)
        actual_date = historical_price.get("date", from_date)

        # Handle currency conversion if needed
        investment_usd = amount
        aed_to_usd_rate = 0.2723  # Default rate
        
        if currency == "AED":
            forex = self.fmp.get_forex_rate("AED", "USD")
            if forex:
                rate = forex.get("ask") or forex.get("price") or forex.get("bid") or aed_to_usd_rate
                aed_to_usd_rate = rate
            investment_usd = amount * aed_to_usd_rate
            print(f"  → Converted {amount} AED to {investment_usd:.2f} USD (rate: {aed_to_usd_rate})")

        # Calculate returns
        shares_bought = investment_usd / historical_close
        current_value_usd = shares_bought * current_price
        profit_usd = current_value_usd - investment_usd
        return_pct = ((current_value_usd - investment_usd) / investment_usd) * 100

        # Convert back to original currency for display
        if currency == "AED":
            current_value_display = current_value_usd / aed_to_usd_rate
            profit_display = profit_usd / aed_to_usd_rate
            currency_symbol = "AED "
        else:
            current_value_display = current_value_usd
            profit_display = profit_usd
            currency_symbol = "$"

        # Format response
        profit_loss_word = "Profit" if profit_display >= 0 else "Loss"
        
        response = f"""**Hypothetical Investment Analysis for {company_name} ({symbol})**

**Investment Details:**
- Initial Investment: {currency_symbol}{amount:,.2f}
- Investment Date: {actual_date}
- Price on {actual_date}: ${historical_close:.2f}
- Shares Purchased: {shares_bought:,.4f}

**Current Status (as of today):**
- Current Price: ${current_price:.2f}
- Current Value: {currency_symbol}{current_value_display:,.2f}
- Total Return: {return_pct:+.2f}%
- {profit_loss_word}: {currency_symbol}{abs(profit_display):,.2f}

**Summary:**
If you had invested {currency_symbol}{amount:,.0f} in {symbol} on {actual_date}, your investment would be worth approximately **{currency_symbol}{current_value_display:,.2f}** today — a **{return_pct:+.1f}%** return."""

        return response

    def _handle_market_movers(self, query: str) -> str:
        """Handle top gainers/losers queries with better error handling."""
        print("  → Fetching market movers...")

        query_lower = query.lower()

        # Determine which data to fetch
        if "loser" in query_lower:
            data = self.fmp.get_losers()
            title = "Top Market Losers"
            empty_message = "losers"
        elif "active" in query_lower:
            data = self.fmp.get_most_active()
            title = "Most Active Stocks"
            empty_message = "most active stocks"
        else:
            data = self.fmp.get_gainers()
            title = "Top Market Gainers"
            empty_message = "gainers"

        if not data:
            # Provide helpful error message
            return f"""I couldn't fetch {empty_message} data at the moment. This could be because:

1. **Market is closed** - Market movers data is most accurate during trading hours (9:30 AM - 4:00 PM ET)
2. **API limitation** - The data endpoint may be temporarily unavailable

**Alternative:** You can ask about:
- Current price of a specific stock (e.g., "What's Tesla's price?")
- Sector performance
- Latest market news

Please try again later or ask a different question."""

        # Limit to top 10
        data = data[:10]

        # Check if asking about specific market (NASDAQ, NYSE, etc.)
        market_filter = None
        if "nasdaq" in query_lower:
            market_filter = "NASDAQ"
            title = f"Top NASDAQ Gainers"
        elif "nyse" in query_lower:
            market_filter = "NYSE"
            title = f"Top NYSE Gainers"

        # Filter by exchange if specified
        if market_filter:
            data = [s for s in data if s.get("exchange", "").upper() == market_filter][:10]
            if not data:
                return f"I couldn't find {empty_message} specifically for {market_filter}. Here are the overall market {empty_message} instead."

        response = f"**{title} (Today)**\n\n"
        
        for i, stock in enumerate(data, 1):
            symbol = stock.get("symbol", "N/A")
            name = stock.get("name", symbol)
            price = stock.get("price", 0)
            change = stock.get("change", 0)
            change_pct = stock.get("changesPercentage", 0)

            # Truncate long names
            if len(name) > 30:
                name = name[:27] + "..."

            response += f"**{i}. {symbol}** - {name}\n"
            response += f"   💰 ${price:.2f} | Change: {change:+.2f} ({change_pct:+.2f}%)\n\n"

        response += "\n*Data is real-time during market hours.*"
        return response

    def _handle_news(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Handle news queries with FMP + Perplexity hybrid approach for enhanced coverage."""
        print("  → Fetching news (FMP + Perplexity hybrid)...")

        query_lower = query.lower()
        symbol = self._resolve_symbol(query)
        topic = symbol.upper() if symbol else None
        
        # Detect specific topics for context
        if "oil" in query_lower:
            topic = "Oil"
        elif "gold" in query_lower:
            topic = "Gold"
        elif "crypto" in query_lower or "bitcoin" in query_lower:
            topic = "Cryptocurrency"
        elif "market" in query_lower:
            topic = "Market"

        # Step 1: Try FMP for structured stock-specific news
        fmp_news = []
        if symbol:
            fmp_news = self.fmp.get_stock_news(symbol, limit=5)
        
        if not fmp_news and topic:
            # Try searching for topic-related news
            fmp_news = self.fmp.search_news(topic, limit=5)
        
        if not fmp_news:
            fmp_news = self.fmp.get_general_news(limit=5)

        # Step 2: Enhance with Perplexity for AI-analyzed news
        perplexity_response = ""
        try:
            perplexity_response = self._fetch_perplexity_news(query, chat_history)
        except Exception as e:
            print(f"  → Perplexity enhancement failed: {e}")

        # Step 3: Combine responses
        response = self._combine_news_sources(query, topic, fmp_news, perplexity_response)

        return response

    def _fetch_perplexity_news(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Fetch news using Perplexity for AI-enhanced financial news with sources.
        
        Args:
            query: User's news query
            chat_history: Conversation history for context
            
        Returns:
            Formatted news response from Perplexity
        """
        # Create a finance-focused news prompt
        news_prompt = (
            f"Get the latest financial news and market analysis about: {query}. "
            "Focus on stock performance, market impact, and analyst opinions. "
            "Include specific data points, price movements, and key developments. "
            "Be concise but comprehensive."
        )
        
        history = chat_history or []
        return self.perplexity.query(news_prompt, history)

    def _combine_news_sources(self, query: str, topic: str, fmp_news: List[Dict], perplexity_response: str) -> str:
        """
        Combine FMP structured news with Perplexity AI analysis.
        
        Args:
            query: Original user query
            topic: Detected topic/symbol
            fmp_news: List of news articles from FMP
            perplexity_response: AI-analyzed news from Perplexity
            
        Returns:
            Combined formatted news response
        """
        
        # If we have Perplexity response and it's substantial, use it as primary
        if perplexity_response and len(perplexity_response) > 100 and not perplexity_response.startswith("Error"):
            response = f"**Latest News & Analysis{' — ' + topic if topic else ''}**\n\n"
            response += perplexity_response
            
            # Append FMP headlines as additional references if available
            if fmp_news:
                response += "\n\n---\n**📰 Additional Headlines:**\n"
                for i, article in enumerate(fmp_news[:3], 1):
                    title = article.get("title", "No title")
                    url = article.get("url", "")
                    source = article.get("site", article.get("source", "Unknown"))
                    if len(title) > 80:
                        title = title[:77] + "..."
                    if url:
                        response += f"{i}. [{title}]({url}) — *{source}*\n"
                    else:
                        response += f"{i}. {title} — *{source}*\n"
            
            return response
        
        # Fallback to FMP-only if Perplexity fails or returns empty
        if fmp_news:
            topic_display = topic if topic else "Market"
            response = f"**Latest News about {topic_display}**\n\n"
            
            for i, article in enumerate(fmp_news[:5], 1):
                title = article.get("title", "No title")
                source = article.get("site", article.get("source", "Unknown"))
                date = article.get("publishedDate", "")[:10] if article.get("publishedDate") else ""
                url = article.get("url", "")
                
                if len(title) > 100:
                    title = title[:97] + "..."
                
                response += f"**{i}. {title}**\n"
                response += f"   📰 {source}"
                if date:
                    response += f" | 📅 {date}"
                response += "\n"
                if url:
                    response += f"   🔗 [Read more]({url})\n"
                response += "\n"
            
            return response
        
        # No news found from either source
        return f"""I couldn't find news about {topic if topic else 'that topic'} at the moment.

**Suggestions:**
- Try asking about a specific company (e.g., "News about Apple")
- Ask for general market news
- Check back later for updated news

You can also ask about current prices, market performance, or other financial data."""

    def _handle_market_movers(self, query: str) -> str:
        """Handle top gainers/losers queries."""
        print("  → Fetching market movers...")

        query_lower = query.lower()

        if "loser" in query_lower:
            data = self.fmp.get_losers()
            title = "Top Market Losers"
        elif "active" in query_lower:
            data = self.fmp.get_most_active()
            title = "Most Active Stocks"
        else:
            data = self.fmp.get_gainers()
            title = "Top Market Gainers"

        if not data:
            return "I couldn't fetch market mover data at the moment. Please try again later."

        # Limit to top 10
        data = data[:10]

        response = f"**{title} (Today)**\n\n"
        for i, stock in enumerate(data, 1):
            symbol = stock.get("symbol", "N/A")
            name = stock.get("name", symbol)
            price = stock.get("price", 0)
            change = stock.get("change", 0)
            change_pct = stock.get("changesPercentage", 0)

            response += f"{i}. **{symbol}** ({name})\n"
            response += f"   Price: ${price:.2f} | Change: {change:+.2f} ({change_pct:+.2f}%)\n\n"

        return response

    def _handle_current_price(self, query: str) -> str:
        """Handle current stock price queries."""
        symbol = self._resolve_symbol(query)

        if not symbol:
            return "I couldn't identify the stock you're asking about. Please specify a valid stock symbol or company name."

        quote = self.fmp.get_quote(symbol)

        if not quote:
            return f"I couldn't fetch the current price for {symbol}. Please verify the symbol and try again."

        name = quote.get("name", symbol)
        price = quote.get("price", 0)
        change = quote.get("change", 0)
        change_pct = quote.get("changesPercentage", 0)
        day_high = quote.get("dayHigh", 0)
        day_low = quote.get("dayLow", 0)
        year_high = quote.get("yearHigh", 0)
        year_low = quote.get("yearLow", 0)
        volume = quote.get("volume", 0)
        market_cap = quote.get("marketCap", 0)

        response = f"""**{name} ({symbol})**

**Current Price:** ${price:.2f}
**Daily Change:** {change:+.2f} ({change_pct:+.2f}%)

**Today's Range:** ${day_low:.2f} - ${day_high:.2f}
**52-Week Range:** ${year_low:.2f} - ${year_high:.2f}

**Volume:** {volume:,}
**Market Cap:** ${market_cap/1e9:.2f}B"""

        return response

    def _handle_commodity(self, query: str) -> str:
        """Handle commodity price queries."""
        query_lower = query.lower()

        if "oil" in query_lower or "crude" in query_lower:
            commodity = "OIL"
            name = "Crude Oil"
        elif "gold" in query_lower:
            commodity = "GOLD"
            name = "Gold"
        elif "silver" in query_lower:
            commodity = "SILVER"
            name = "Silver"
        elif "gas" in query_lower:
            commodity = "NATURAL GAS"
            name = "Natural Gas"
        else:
            return "Please specify a commodity (oil, gold, silver, natural gas)."

        quote = self.fmp.get_commodity_quote(commodity)

        if not quote:
            return f"I couldn't fetch the current price for {name}. Please try again later."

        price = quote.get("price", 0)
        change = quote.get("change", 0)
        change_pct = quote.get("changesPercentage", 0)

        return f"""**{name} Price**

**Current Price:** ${price:.2f}
**Change:** {change:+.2f} ({change_pct:+.2f}%)"""

    def _handle_crypto(self, query: str) -> str:
        """Handle cryptocurrency price queries."""
        query_lower = query.lower()

        if "bitcoin" in query_lower or "btc" in query_lower:
            symbol = "BTCUSD"
            name = "Bitcoin"
        elif "ethereum" in query_lower or "eth" in query_lower:
            symbol = "ETHUSD"
            name = "Ethereum"
        else:
            return "Please specify a cryptocurrency (Bitcoin, Ethereum, etc.)."

        quote = self.fmp.get_crypto_quote(symbol)

        if not quote:
            return f"I couldn't fetch the current price for {name}. Please try again later."

        price = quote.get("price", 0)
        change = quote.get("change", 0)
        change_pct = quote.get("changesPercentage", 0)

        return f"""**{name} ({symbol})**

**Current Price:** ${price:,.2f}
**Change:** {change:+,.2f} ({change_pct:+.2f}%)"""

    def _handle_index(self, query: str) -> str:
        """Handle market index queries."""
        query_lower = query.lower()

        indices_to_fetch = []

        if "s&p" in query_lower or "sp500" in query_lower:
            indices_to_fetch.append(("^GSPC", "S&P 500"))
        if "nasdaq" in query_lower:
            indices_to_fetch.append(("^IXIC", "NASDAQ"))
        if "dow" in query_lower:
            indices_to_fetch.append(("^DJI", "Dow Jones"))

        if not indices_to_fetch:
            # Default to S&P 500 (SPX) as the primary benchmark
            indices_to_fetch = [
                ("SPX", "S&P 500"),
            ]

        response = "**Market Indices**\n\n"

        for symbol, name in indices_to_fetch:
            quote = self.fmp.get_quote(symbol)
            if quote:
                price = quote.get("price", 0)
                change = quote.get("change", 0)
                change_pct = quote.get("changesPercentage", 0)
                response += f"**{name}:** {price:,.2f} ({change:+.2f}, {change_pct:+.2f}%)\n"

        return response

    def _handle_forex(self, query: str) -> str:
        """Handle forex/currency queries."""
        # Try to extract currency pairs
        query_upper = query.upper()

        pairs = [
            ("USD", "AED"),
            ("EUR", "USD"),
            ("GBP", "USD"),
        ]

        if "AED" in query_upper and "USD" in query_upper:
            pairs = [("USD", "AED"), ("AED", "USD")]
        elif "EUR" in query_upper:
            pairs = [("EUR", "USD")]
        elif "GBP" in query_upper:
            pairs = [("GBP", "USD")]

        response = "**Exchange Rates**\n\n"

        for from_curr, to_curr in pairs:
            rate = self.fmp.get_forex_rate(from_curr, to_curr)
            if rate:
                ask = rate.get("ask", 0)
                response += f"**{from_curr}/{to_curr}:** {ask:.4f}\n"

        return response

    def fetch_raw_data(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Fetch real-time financial data WITHOUT explanation.
        Use this when you want to combine the data with other sources before explaining.
        (e.g., ComparisonHandler uses this to avoid double-explanation)

        Args:
            user_query: User's question
            chat_history: Previous conversation for context

        Returns:
            Raw formatted response with financial data (no LLM explanation)
        """
        if chat_history is None:
            chat_history = []

        try:
            start_time = time.time()
            print(f"⏱️  Starting: Internet Data Fetch (raw)...")

            # Classify the query type
            query_type = self._classify_query(user_query)
            print(f"  → Query classified as: {query_type}")

            # Route to appropriate handler
            if query_type == "HYPOTHETICAL_INVESTMENT":
                response = self._handle_hypothetical_investment(user_query)
            elif query_type == "MARKET_MOVERS":
                response = self._handle_market_movers(user_query)
            elif query_type == "NEWS":
                response = self._handle_news(user_query, chat_history)
            elif query_type == "COMMODITY_PRICE":
                response = self._handle_commodity(user_query)
            elif query_type == "CRYPTO_PRICE":
                response = self._handle_crypto(user_query)
            elif query_type == "INDEX_PERFORMANCE":
                response = self._handle_index(user_query)
            elif query_type == "FOREX":
                response = self._handle_forex(user_query)
            elif query_type == "CURRENT_PRICE":
                response = self._handle_current_price(user_query)
            else:
                # General query - try to determine best approach
                symbol = self._resolve_symbol(user_query)
                if symbol:
                    response = self._handle_current_price(user_query)
                else:
                    response = self._handle_news(user_query)

            elapsed = time.time() - start_time
            print(f"✅ Completed: Internet Data Fetch (raw) in {elapsed:.2f}s")

            # Return raw data WITHOUT explanation
            return response

        except Exception as e:
            print(f"❌ Error fetching internet data: {e}")
            import traceback
            traceback.print_exc()
            return f"Error fetching internet data: {str(e)}"

    def fetch_data(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Fetch real-time financial data based on user query.

        This is the main entry point that routes to specific handlers.
        Data is explained by LLM before returning.

        Args:
            user_query: User's question
            chat_history: Previous conversation for context

        Returns:
            Explained response with financial data
        """
        # Get raw data first
        raw_response = self.fetch_raw_data(user_query, chat_history)
        
        if raw_response.startswith("Error"):
            return raw_response

        # Pass the raw response through the explainer for better interpretation
        explained_response = self.explain_internet_data(user_query, raw_response)
        return explained_response

    def explain_internet_data(self, query: str, raw_data: str) -> str:
        """
        Generate a natural language explanation of internet data from the user's perspective.

        Args:
            query: Original user question
            raw_data: The raw data/response fetched from internet sources

        Returns:
            Natural language explanation interpreted for the user
        """
        explain_prompt = PromptTemplate(
            input_variables=["query", "data", "today_date"],
            template=INTERNET_DATA_EXPLANATION_PROMPT
        )

        explain_chain = explain_prompt | self.explanation_llm | StrOutputParser()

        try:
            start_time = time.time()
            print(f"⏱️  [{self.llm_type}] Starting: Internet Data Explanation...")

            # Get today's date for context
            today_date = datetime.now().strftime("%A, %B %d, %Y")

            explanation = explain_chain.invoke({
                "query": query,
                "data": raw_data,
                "today_date": today_date
            })

            elapsed = time.time() - start_time
            print(f"✅ [{self.llm_type}] Completed: Internet Data Explanation in {elapsed:.2f}s")

            return explanation.strip()
        except Exception as e:
            print(f"❌ Error explaining internet data: {e}")
            # Fallback to returning the raw data if explanation fails
            return raw_data

    def stream_explain_internet_data(self, query: str, raw_data: str):
        """
        Stream a natural language explanation of internet data.

        Args:
            query: Original user question
            raw_data: The raw data/response fetched from internet sources

        Yields:
            String chunks of the explanation
        """
        explain_prompt = PromptTemplate(
            input_variables=["query", "data", "today_date"],
            template=INTERNET_DATA_EXPLANATION_PROMPT
        )

        explain_chain = explain_prompt | self.explanation_llm | StrOutputParser()

        try:
            print(f"⏱️  [{self.llm_type}] Starting: Streaming Internet Data Explanation...")

            # Get today's date for context
            today_date = datetime.now().strftime("%A, %B %d, %Y")

            for chunk in explain_chain.stream({
                "query": query,
                "data": raw_data,
                "today_date": today_date
            }):
                yield chunk

            print(f"✅ [{self.llm_type}] Completed: Streaming Internet Data Explanation")
        except Exception as e:
            print(f"❌ Error streaming internet data explanation: {e}")
            # Fallback to returning the raw data if streaming fails
            yield raw_data