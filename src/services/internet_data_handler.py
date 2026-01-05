"""
Internet Data Handler - Fetches real-time financial data and explains using QWEN streaming
"""
from typing import List, Dict, Optional, Tuple, Any, Generator
from datetime import datetime, timedelta
import re
import time
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.services.fmp_service import FMPService
from src.services.perplexity_service import PerplexityService
from src.services.chat_memory import ChatMemory
from src.config.settings import get_qwen_config
from src.config.prompts import (
    INTERNET_DATA_EXPLANATION_PROMPT,
    detect_language,
    ARABIC_FINANCIAL_GLOSSARY
)

from dotenv import load_dotenv
load_dotenv()


class InternetDataHandler:
    """
    Handles internet-based financial data fetching using FMP API.
    Uses QWEN H100 for streaming explanations.
    """

    def __init__(self, memory_max_pairs: int = 5):
        """
        Initialize the internet data handler.

        Args:
            memory_max_pairs: Maximum number of Q&A pairs to remember
        """
        self.fmp = FMPService()
        self.perplexity = PerplexityService()
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

        # Use QWEN H100 for explanations (streaming enabled)
        qwen_config = get_qwen_config()
        self.explanation_llm = ChatOpenAI(
            model=qwen_config.get("model_name", "Qwen3-30B-A3B"),
            base_url=qwen_config["base_url"],
            api_key=qwen_config["api_key"],
            temperature=qwen_config.get("temperature", 0.3),
            streaming=True
        )
        self.llm_type = "QWEN H100"  # For logging purposes

        # Explanation prompt with language and glossary placeholders
        self.explain_prompt = PromptTemplate(
            input_variables=["query", "data", "today_date", "language", "arabic_glossary"],
            template=INTERNET_DATA_EXPLANATION_PROMPT
        )

        # Company to symbol mapping
        self.company_to_symbol = {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "alphabet": "GOOGL", "amazon": "AMZN", "tesla": "TSLA",
            "nvidia": "NVDA", "meta": "META", "facebook": "META",
            "netflix": "NFLX", "disney": "DIS", "nike": "NKE",
            "coca-cola": "KO", "pepsi": "PEP", "walmart": "WMT",
            "boeing": "BA", "intel": "INTC", "amd": "AMD",
            "spotify": "SPOT", "uber": "UBER", "airbnb": "ABNB",
            "zoom": "ZM", "paypal": "PYPL", "visa": "V",
            "mastercard": "MA", "jpmorgan": "JPM", "goldman": "GS",
            "berkshire": "BRK-B",
        }

    def _resolve_symbol(self, query: str) -> Optional[str]:
        """Extract and resolve stock symbol from query."""
        query_lower = query.lower()

        for company, symbol in self.company_to_symbol.items():
            if company in query_lower:
                return symbol

        ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
        if ticker_match:
            potential_symbol = ticker_match.group(1)
            quote = self.fmp.get_quote(potential_symbol)
            if quote:
                return potential_symbol

        words = query.split()
        for word in words:
            if len(word) > 2 and word[0].isupper():
                results = self.fmp.search_symbol(word, limit=1)
                if results:
                    return results[0].get("symbol")

        return None

    def _parse_date_from_query(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse date references from query."""
        query_lower = query.lower()
        today = datetime.now()
        current_year = today.year

        # Parse "X months/years/weeks/days ago" patterns
        ago_patterns = [
            (r'(\d+)\s*months?\s+ago', 'months'),
            (r'(\d+)\s*years?\s+ago', 'years'),
            (r'(\d+)\s*weeks?\s+ago', 'weeks'),
            (r'(\d+)\s*days?\s+ago', 'days'),
        ]
        
        for pattern, unit in ago_patterns:
            match = re.search(pattern, query_lower)
            if match:
                num = int(match.group(1))
                if unit == 'months':
                    # Calculate date X months ago
                    target_month = today.month - num
                    target_year = today.year
                    while target_month <= 0:
                        target_month += 12
                        target_year -= 1
                    # Handle day overflow (e.g., Jan 31 -> Feb doesn't have 31)
                    target_day = min(today.day, 28)  # Safe day
                    target_date = datetime(target_year, target_month, target_day)
                elif unit == 'years':
                    target_date = today.replace(year=today.year - num)
                elif unit == 'weeks':
                    target_date = today - timedelta(weeks=num)
                elif unit == 'days':
                    target_date = today - timedelta(days=num)
                
                from_date = target_date.strftime("%Y-%m-%d")
                return from_date, today.strftime("%Y-%m-%d")

        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }

        # Parse "Month Year" patterns like "July 2024", "in July 2024", "January 2023"
        for month_name, month_num in months.items():
            # Pattern: "July 2024" or "in July 2024"
            month_year_pattern = rf'{month_name}\s+(\d{{4}})'
            match = re.search(month_year_pattern, query_lower)
            if match:
                year = int(match.group(1))
                from_date = f"{year}-{month_num:02d}-01"
                if month_num == 12:
                    to_date = f"{year}-12-31"
                else:
                    next_month = datetime(year, month_num + 1, 1) - timedelta(days=1)
                    to_date = next_month.strftime("%Y-%m-%d")
                return from_date, to_date

        # Parse standalone year like "in 2024", "2024"
        year_pattern = r'(?:in\s+)?(\d{4})(?!\d)'
        year_match = re.search(year_pattern, query_lower)
        if year_match:
            year = int(year_match.group(1))
            # Only accept reasonable years (1990-current year)
            if 1990 <= year <= current_year:
                # Check if there's also a month mentioned
                found_month = None
                for month_name, month_num in months.items():
                    if month_name in query_lower:
                        found_month = month_num
                        break
                
                if found_month:
                    # Month + Year already handled above, but fallback here
                    from_date = f"{year}-{found_month:02d}-01"
                else:
                    # Just year - use January 1st
                    from_date = f"{year}-01-02"
                return from_date, today.strftime("%Y-%m-%d")

        # Parse standalone month names (assumes current or previous year)
        for month_name, month_num in months.items():
            if month_name in query_lower:
                year = current_year if month_num <= today.month else current_year - 1
                from_date = f"{year}-{month_num:02d}-01"
                if month_num == 12:
                    to_date = f"{year}-12-31"
                else:
                    next_month = datetime(year, month_num + 1, 1) - timedelta(days=1)
                    to_date = next_month.strftime("%Y-%m-%d")
                return from_date, to_date

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
        """Parse investment amount and currency from query."""
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
        """Classify the type of internet query."""
        query_lower = query.lower()

        if any(phrase in query_lower for phrase in [
            "if i had invested", "had i invested", "if i invested",
            "would it be worth", "would be worth", "how much would"
        ]):
            return "HYPOTHETICAL_INVESTMENT"

        if any(phrase in query_lower for phrase in [
            "top gainers", "top losers", "biggest gainers", "biggest losers",
            "most active", "trending stocks", "market movers"
        ]):
            return "MARKET_MOVERS"

        if any(phrase in query_lower for phrase in [
            "news", "latest on", "what's happening", "any updates"
        ]):
            return "NEWS"

        if any(phrase in query_lower for phrase in [
            "oil price", "gold price", "silver price", "commodity",
            "crude oil", "natural gas"
        ]):
            return "COMMODITY_PRICE"

        if any(phrase in query_lower for phrase in [
            "bitcoin", "ethereum", "crypto", "btc", "eth"
        ]):
            return "CRYPTO_PRICE"

        if any(phrase in query_lower for phrase in [
            "s&p 500", "sp500", "nasdaq", "dow jones", "dow",
            "russell", "market index"
        ]):
            return "INDEX_PERFORMANCE"

        if any(phrase in query_lower for phrase in [
            "exchange rate", "forex", "currency", "aed to usd", "usd to"
        ]):
            return "FOREX"

        if any(phrase in query_lower for phrase in [
            "price of", "current price", "stock price", "how much is"
        ]):
            return "CURRENT_PRICE"

        return "GENERAL"

    def _handle_hypothetical_investment(self, query: str) -> str:
        """Handle 'If I had invested X in Y' queries."""
        print("  → Processing hypothetical investment query...")

        symbol = self._resolve_symbol(query)
        amount, currency = self._parse_investment_amount(query)
        from_date, _ = self._parse_date_from_query(query)

        if not symbol:
            return "I couldn't identify the stock. Please specify a valid stock symbol or company name."

        if not amount:
            return "I couldn't determine the investment amount. Please specify like 'AED 10,000' or '$10,000'."

        if not from_date:
            current_year = datetime.now().year
            from_date = f"{current_year}-01-02"

        print(f"  → Symbol: {symbol}, Amount: {amount} {currency}, From: {from_date}")

        historical_price = self.fmp.get_price_on_date(symbol, from_date)
        if not historical_price:
            try:
                date_obj = datetime.strptime(from_date, "%Y-%m-%d")
                month_start = date_obj.replace(day=1).strftime("%Y-%m-%d")
                month_end = (date_obj.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                month_end_str = month_end.strftime("%Y-%m-%d")

                historical_data = self.fmp.get_historical_price(symbol, from_date=month_start, to_date=month_end_str)
                if historical_data and len(historical_data) > 0:
                    historical_data.sort(key=lambda x: x.get("date", ""))
                    historical_price = historical_data[0]
            except Exception as e:
                print(f"  → Fallback failed: {e}")

            if not historical_price:
                return f"I couldn't find historical price data for {symbol} around {from_date}."

        current_quote = self.fmp.get_quote(symbol)
        if not current_quote:
            return f"I couldn't fetch the current price for {symbol}."

        historical_close = historical_price.get("close") or historical_price.get("adjClose")
        if not historical_close:
            return f"Historical price data for {symbol} is incomplete."

        current_price = current_quote.get("price")
        if not current_price:
            return f"Current price data for {symbol} is unavailable."

        company_name = current_quote.get("name", symbol)
        actual_date = historical_price.get("date", from_date)

        investment_usd = amount
        aed_to_usd_rate = 0.2723

        if currency == "AED":
            forex = self.fmp.get_forex_rate("AED", "USD")
            if forex:
                rate = forex.get("ask") or forex.get("price") or forex.get("bid") or aed_to_usd_rate
                aed_to_usd_rate = rate
            investment_usd = amount * aed_to_usd_rate

        shares_bought = investment_usd / historical_close
        current_value_usd = shares_bought * current_price
        profit_usd = current_value_usd - investment_usd
        return_pct = ((current_value_usd - investment_usd) / investment_usd) * 100

        if currency == "AED":
            current_value_display = current_value_usd / aed_to_usd_rate
            profit_display = profit_usd / aed_to_usd_rate
            currency_symbol = "AED "
        else:
            current_value_display = current_value_usd
            profit_display = profit_usd
            currency_symbol = "$"

        profit_loss_word = "Profit" if profit_display >= 0 else "Loss"

        response = f"""**Hypothetical Investment Analysis for {company_name} ({symbol})**

**Investment Details:**
- Initial Investment: {currency_symbol}{amount:,.2f}
- Investment Date: {actual_date}
- Price on {actual_date}: ${historical_close:.2f}
- Shares Purchased: {shares_bought:,.4f}

**Current Status:**
- Current Price: ${current_price:.2f}
- Current Value: {currency_symbol}{current_value_display:,.2f}
- Total Return: {return_pct:+.2f}%
- {profit_loss_word}: {currency_symbol}{abs(profit_display):,.2f}

**Summary:**
If you had invested {currency_symbol}{amount:,.0f} in {symbol} on {actual_date}, it would be worth **{currency_symbol}{current_value_display:,.2f}** today — a **{return_pct:+.1f}%** return."""

        return response

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
            return f"I couldn't fetch market movers data. The market may be closed."

        data = data[:10]

        response = f"**{title} (Today)**\n\n"
        for i, stock in enumerate(data, 1):
            symbol = stock.get("symbol", "N/A")
            name = stock.get("name", symbol)
            price = stock.get("price", 0)
            change = stock.get("change", 0)
            change_pct = stock.get("changesPercentage", 0)

            if len(name) > 30:
                name = name[:27] + "..."

            response += f"**{i}. {symbol}** - {name}\n"
            response += f"   💰 ${price:.2f} | Change: {change:+.2f} ({change_pct:+.2f}%)\n\n"

        return response

    def _handle_news(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Handle news queries."""
        print("  → Fetching news...")
        query_lower = query.lower()
        symbol = self._resolve_symbol(query)
        topic = symbol.upper() if symbol else None

        if "oil" in query_lower:
            topic = "Oil"
        elif "gold" in query_lower:
            topic = "Gold"
        elif "crypto" in query_lower or "bitcoin" in query_lower:
            topic = "Cryptocurrency"
        elif "market" in query_lower:
            topic = "Market"

        fmp_news = []
        if symbol:
            fmp_news = self.fmp.get_stock_news(symbol, limit=5)

        if not fmp_news and topic:
            fmp_news = self.fmp.search_news(topic, limit=5)

        if not fmp_news:
            fmp_news = self.fmp.get_general_news(limit=5)

        perplexity_response = ""
        try:
            news_prompt = f"Get the latest financial news about: {query}. Focus on key developments."
            history = chat_history or []
            perplexity_response = self.perplexity.query(news_prompt, history)
        except Exception as e:
            print(f"  → Perplexity failed: {e}")

        if perplexity_response and len(perplexity_response) > 100:
            response = f"**Latest News & Analysis{' — ' + topic if topic else ''}**\n\n"
            response += perplexity_response

            if fmp_news:
                response += "\n\n---\n**📰 Additional Headlines:**\n"
                for i, article in enumerate(fmp_news[:3], 1):
                    title = article.get("title", "No title")
                    source = article.get("site", "Unknown")
                    if len(title) > 80:
                        title = title[:77] + "..."
                    response += f"{i}. {title} — *{source}*\n"

            return response

        if fmp_news:
            topic_display = topic if topic else "Market"
            response = f"**Latest News about {topic_display}**\n\n"

            for i, article in enumerate(fmp_news[:5], 1):
                title = article.get("title", "No title")
                source = article.get("site", "Unknown")
                date = article.get("publishedDate", "")[:10] if article.get("publishedDate") else ""

                if len(title) > 100:
                    title = title[:97] + "..."

                response += f"**{i}. {title}**\n"
                response += f"   📰 {source}"
                if date:
                    response += f" | 📅 {date}"
                response += "\n\n"

            return response

        return f"I couldn't find news about {topic if topic else 'that topic'}."

    def _handle_current_price(self, query: str) -> str:
        """Handle current stock price queries."""
        symbol = self._resolve_symbol(query)

        if not symbol:
            return "I couldn't identify the stock. Please specify a valid symbol or company name."

        quote = self.fmp.get_quote(symbol)

        if not quote:
            return f"I couldn't fetch the price for {symbol}."

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

        return f"""**{name} ({symbol})**

**Current Price:** ${price:.2f}
**Daily Change:** {change:+.2f} ({change_pct:+.2f}%)

**Today's Range:** ${day_low:.2f} - ${day_high:.2f}
**52-Week Range:** ${year_low:.2f} - ${year_high:.2f}

**Volume:** {volume:,}
**Market Cap:** ${market_cap/1e9:.2f}B"""

    def _handle_commodity(self, query: str) -> str:
        """Handle commodity price queries."""
        query_lower = query.lower()

        if "oil" in query_lower or "crude" in query_lower:
            commodity, name = "OIL", "Crude Oil"
        elif "gold" in query_lower:
            commodity, name = "GOLD", "Gold"
        elif "silver" in query_lower:
            commodity, name = "SILVER", "Silver"
        elif "gas" in query_lower:
            commodity, name = "NATURAL GAS", "Natural Gas"
        else:
            return "Please specify a commodity (oil, gold, silver, natural gas)."

        quote = self.fmp.get_commodity_quote(commodity)

        if not quote:
            return f"I couldn't fetch the price for {name}."

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
            symbol, name = "BTCUSD", "Bitcoin"
        elif "ethereum" in query_lower or "eth" in query_lower:
            symbol, name = "ETHUSD", "Ethereum"
        else:
            return "Please specify a cryptocurrency (Bitcoin, Ethereum, etc.)."

        quote = self.fmp.get_crypto_quote(symbol)

        if not quote:
            return f"I couldn't fetch the price for {name}."

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
            indices_to_fetch = [("SPX", "S&P 500")]

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
        query_upper = query.upper()

        pairs = [("USD", "AED"), ("EUR", "USD"), ("GBP", "USD")]

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
        Returns raw formatted data for subsequent explanation.

        Args:
            user_query: User's question
            chat_history: Previous conversation for context

        Returns:
            Raw formatted response with financial data
        """
        if chat_history is None:
            chat_history = []

        try:
            start_time = time.time()
            print(f"⏱️  Starting: Internet Data Fetch...")

            query_type = self._classify_query(user_query)
            print(f"  → Query classified as: {query_type}")

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
                symbol = self._resolve_symbol(user_query)
                if symbol:
                    response = self._handle_current_price(user_query)
                else:
                    response = self._handle_news(user_query)

            elapsed = time.time() - start_time
            print(f"✅ Completed: Internet Data Fetch in {elapsed:.2f}s")

            return response

        except Exception as e:
            print(f"❌ Error fetching internet data: {e}")
            import traceback
            traceback.print_exc()
            return f"Error fetching data: {str(e)}"

    def fetch_data(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Fetch data and explain using QWEN (non-streaming).

        Args:
            user_query: User's question
            chat_history: Previous conversation for context

        Returns:
            Explained response with financial data
        """
        raw_response = self.fetch_raw_data(user_query, chat_history)

        if raw_response.startswith("Error"):
            return raw_response

        explained_response = self.explain_internet_data(user_query, raw_response)
        return explained_response

    def explain_internet_data(self, query: str, raw_data: str) -> str:
        """
        Generate explanation using QWEN (non-streaming).

        Args:
            query: Original user question
            raw_data: Raw data fetched from internet sources

        Returns:
            Natural language explanation
        """
        try:
            start_time = time.time()
            print(f"⏱️  [QWEN H100] Starting: Internet Data Explanation...")

            today_date = datetime.now().strftime("%A, %B %d, %Y")
            
            # Detect language and prepare glossary
            language = detect_language(query)
            arabic_glossary = ARABIC_FINANCIAL_GLOSSARY if language == "Arabic" else "N/A"

            formatted_prompt = self.explain_prompt.format(
                query=query,
                data=raw_data,
                today_date=today_date,
                language=language,
                arabic_glossary=arabic_glossary
            )

            explanation = self.explanation_llm.invoke(formatted_prompt)

            elapsed = time.time() - start_time
            print(f"✅ [QWEN H100] Completed in {elapsed:.2f}s")

            if hasattr(explanation, 'content'):
                return explanation.content.strip()
            return str(explanation).strip()

        except Exception as e:
            print(f"❌ Error explaining internet data: {e}")
            return raw_data

    def explain_internet_data_streaming(self, query: str, raw_data: str) -> Generator[Dict, None, None]:
        """
        Generate explanation using QWEN with streaming.

        Args:
            query: Original user question
            raw_data: Raw data fetched from internet sources

        Yields:
            Dictionary containing:
            {
                "type": "chunk" | "metadata" | "error",
                "content": str,
                "elapsed_time": float (only for metadata)
            }
        """
        try:
            start_time = time.time()
            print(f"⏱️  [QWEN H100] Starting: Internet Data Explanation (Streaming)...")

            today_date = datetime.now().strftime("%A, %B %d, %Y")
            
            # Detect language and prepare glossary
            language = detect_language(query)
            arabic_glossary = ARABIC_FINANCIAL_GLOSSARY if language == "Arabic" else "N/A"

            formatted_prompt = self.explain_prompt.format(
                query=query,
                data=raw_data,
                today_date=today_date,
                language=language,
                arabic_glossary=arabic_glossary
            )

            # Stream from QWEN
            for chunk in self.explanation_llm.stream(formatted_prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield {
                        "type": "chunk",
                        "content": chunk.content
                    }
                elif isinstance(chunk, str) and chunk:
                    yield {
                        "type": "chunk",
                        "content": chunk
                    }

            elapsed = time.time() - start_time
            print(f"✅ [QWEN H100] Completed: Streaming in {elapsed:.2f}s")

            yield {
                "type": "metadata",
                "content": "",
                "elapsed_time": elapsed
            }

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
        # Detect language and prepare glossary
        language = detect_language(query)
        arabic_glossary = ARABIC_FINANCIAL_GLOSSARY if language == "Arabic" else "N/A"
        
        explain_prompt = PromptTemplate(
            input_variables=["query", "data", "today_date", "language", "arabic_glossary"],
            template=INTERNET_DATA_EXPLANATION_PROMPT
        )

        explain_chain = explain_prompt | self.explanation_llm | StrOutputParser()

        try:
            print(f"⏱️  [{self.llm_type}] Starting: Streaming Internet Data Explanation ({language})...")

            # Get today's date for context
            today_date = datetime.now().strftime("%A, %B %d, %Y")

            for chunk in explain_chain.stream({
                "query": query,
                "data": raw_data,
                "today_date": today_date,
                "language": language,
                "arabic_glossary": arabic_glossary
            }):
                yield chunk

            print(f"✅ [{self.llm_type}] Completed: Streaming Internet Data Explanation")
        except Exception as e:
            print(f"❌ Error streaming internet data explanation: {e}")
            # Fallback to returning the raw data if streaming fails
            yield raw_data
