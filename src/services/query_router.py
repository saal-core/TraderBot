# src/services/query_router.py
"""
Query Router - Routes queries to appropriate handlers based on intent classification.
Uses priority-based pattern matching before falling back to LLM classification.
"""
from typing import Dict, List, Tuple, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config
import os
import re
import time
from rapidfuzz import fuzz, process

from dotenv import load_dotenv
load_dotenv()


class QueryRouter:
    """Routes queries to appropriate handlers based on intent classification"""

    def __init__(self, model_name: str = None, sql_executor=None):
        """
        Initialize the query router

        Args:
            model_name: Name of the Ollama model to use (defaults to config)
            sql_executor: Optional SQL executor for checking database contents
        """
        self.sql_executor = sql_executor
        self._symbols_cache = None
        self._portfolios_cache = None

        # Always use Ollama for routing
        ollama_config = get_ollama_config()
        self.model_name = model_name or ollama_config["model_name"]
        self.base_url = ollama_config["base_url"]
        self.temperature = ollama_config["temperature_routing"]

        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature
        )

        # Define routing prompt template (used as fallback)
        self.routing_prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are a query classifier for a financial assistant. Classify the query into ONE category.

Categories:
1. "database" - Questions about LOCAL portfolio/holdings data stored in our database:
   - Portfolio holdings, positions, quantities, lots
   - Stock performance within portfolios, profit/loss, unrealized gains
   - Portfolio statistics, returns, values, metrics
   - Rankings/comparisons of LOCAL data (e.g., "my top stocks", "which of my holdings")
   - Questions with "my", "our" referring to portfolio data
   Examples: "show me all portfolios", "what are my holdings", "which of my stocks has the highest profit"

2. "greeting" - Greetings, chitchat, hello, how are you, goodbye, thank you, etc.

3. "internet_data" - Questions requiring EXTERNAL real-time data:
   - Current/live stock prices from the market
   - Latest financial news
   - Market indices (S&P 500, NASDAQ, Dow Jones performance)
   - Market movers: top gainers, top losers, trending stocks
   - Cryptocurrency prices
   - Commodity prices (oil, gold, etc.)
   - Forex/exchange rates
   - Hypothetical investment calculations
   - Any question about stocks/companies NOT in our local database
   Examples: "What's Tesla's current price?", "Top gainers today", "If I had invested in NVIDIA..."

Rules:
- Return ONLY: database, greeting, or internet_data
- Questions about external market data = "internet_data"
- Questions about user's own portfolio/holdings = "database"
- When uncertain about stock queries, prefer "internet_data"

User Query: {query}

Category:"""
        )

        self.routing_chain = self.routing_prompt | self.llm | StrOutputParser()

    def _get_all_symbols(self) -> List[str]:
        """Fetch all stock symbols from database with caching"""
        if self._symbols_cache is not None:
            return self._symbols_cache

        if not self.sql_executor:
            return []

        try:
            query = """
            SELECT DISTINCT symbol
            FROM ai_trading.portfolio_holdings
            WHERE symbol IS NOT NULL
            ORDER BY symbol
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                symbols = df['symbol'].tolist()
                self._symbols_cache = symbols
                return symbols
            return []
        except Exception as e:
            print(f"Error fetching symbols for routing: {e}")
            return []

    def _get_all_portfolios(self) -> List[str]:
        """Fetch all portfolio names from database with caching"""
        if self._portfolios_cache is not None:
            return self._portfolios_cache

        if not self.sql_executor:
            return []

        try:
            query = """
            SELECT DISTINCT portfolio_name
            FROM ai_trading.portfolio_summary
            WHERE portfolio_name IS NOT NULL
            ORDER BY portfolio_name
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                portfolios = df['portfolio_name'].tolist()
                self._portfolios_cache = portfolios
                return portfolios
            return []
        except Exception as e:
            print(f"Error fetching portfolios for routing: {e}")
            return []

    def _check_symbols_in_database(self, mentioned_terms: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Check which mentioned terms exist in our database

        Returns:
            Tuple of (terms_found_in_db, terms_not_found, portfolios_found)
        """
        if not mentioned_terms:
            return [], [], []

        all_symbols = self._get_all_symbols()
        all_portfolios = self._get_all_portfolios()
        all_data_sources = all_symbols + all_portfolios

        if not all_data_sources:
            return [], mentioned_terms, []

        found = []
        not_found = []
        portfolios_found = []

        all_symbols_upper = [s.upper() for s in all_symbols]
        all_portfolios_upper = [p.upper() for p in all_portfolios]
        all_data_upper = [s.upper() for s in all_data_sources]

        for term in mentioned_terms:
            term_upper = term.upper()
            
            if term_upper in all_portfolios_upper:
                found.append(term)
                portfolios_found.append(term)
            elif term_upper in all_symbols_upper:
                found.append(term)
            else:
                # Fuzzy match
                best_match = process.extractOne(
                    term_upper,
                    all_data_upper,
                    scorer=fuzz.ratio,
                    score_cutoff=80
                )
                if best_match:
                    matched_value = all_data_sources[all_data_upper.index(best_match[0])]
                    print(f"  → Fuzzy matched '{term}' to '{matched_value}' (score: {best_match[1]})")
                    found.append(term)
                    if matched_value.upper() in all_portfolios_upper:
                        portfolios_found.append(term)
                else:
                    not_found.append(term)

        return found, not_found, portfolios_found

    def _extract_potential_symbols(self, query: str) -> List[str]:
        """
        Extract potential stock symbols/company names from query using regex.
        Fast extraction without LLM call.
        """
        extracted = []
        
        # Common company name to symbol mapping
        company_map = {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA",
            "meta": "META", "facebook": "META", "netflix": "NFLX",
            "disney": "DIS", "boeing": "BA", "intel": "INTC",
            "amd": "AMD", "paypal": "PYPL", "visa": "V",
            "mastercard": "MA", "jpmorgan": "JPM", "walmart": "WMT",
            "coca-cola": "KO", "pepsi": "PEP", "nike": "NKE",
        }
        
        query_lower = query.lower()
        
        # Check for company names
        for company, symbol in company_map.items():
            if company in query_lower:
                extracted.append(symbol)
        
        # Extract uppercase ticker patterns (1-5 letters)
        tickers = re.findall(r'\b([A-Z]{1,5})\b', query)
        for ticker in tickers:
            if ticker not in ['I', 'A', 'THE', 'AND', 'OR', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'IS', 'IT', 'BE', 'AS', 'BY', 'IF', 'SO', 'NO', 'UP', 'DO', 'GO', 'MY', 'WE', 'US', 'AN', 'AM', 'PM', 'VS', 'UK', 'EU', 'USD', 'AED', 'EUR', 'GBP', 'YTD', 'MTD', 'QTD', 'P&L', 'PNL']:
                extracted.append(ticker)
        
        return list(set(extracted))

    def classify_query(self, query: str) -> str:
        """
        Classify a user query with priority-based pattern matching.
        
        Priority Order:
        1. Internet-first patterns (highest priority) - ALWAYS route to internet
        2. Greeting patterns
        3. Strong database patterns (possessive + holdings context)
        4. Symbol-based routing with context awareness
        5. LLM fallback
        """
        try:
            query_lower = query.lower()
            
            # ================================================================
            # STEP 1: INTERNET-FIRST PATTERNS (Highest Priority)
            # These patterns ALWAYS indicate internet data is needed
            # ================================================================
            
            # 1a. Hypothetical investment calculations
            hypothetical_patterns = [
                r'if i had invested',
                r'had i invested',
                r'if i invested',
                r'would it be worth',
                r'would be worth now',
                r'how much would .* be worth',
                r'what would .* be worth',
                r'invested .* in (january|february|march|april|may|june|july|august|september|october|november|december)',
                r'if .* bought .* (shares?|stock)',
                r'what if .* invested',
            ]
            for pattern in hypothetical_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Hypothetical investment pattern detected")
                    return "internet_data"

            # 1b. Market movers / rankings (real-time market data)
            market_mover_patterns = [
                r'top\s+(gainers?|losers?|movers?)',
                r'biggest\s+(gainers?|losers?|movers?)',
                r'(gainers?|losers?|movers?)\s+(in|on|for)\s+.*(nasdaq|nyse|s&p|dow|market)',
                r'(nasdaq|nyse|s&p|dow|russell).*this\s+(week|month|day)',
                r'this\s+(week|month).*(?:nasdaq|nyse|s&p|dow)',
                r'trending\s+stocks?',
                r'most\s+active\s+stocks?',
                r'market\s+(leaders?|laggards?)',
                r'best\s+performing\s+stocks?\s+(today|this week|this month)',
                r'worst\s+performing\s+stocks?\s+(today|this week|this month)',
            ]
            for pattern in market_mover_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Market movers pattern detected")
                    return "internet_data"

            # 1c. News queries
            news_patterns = [
                r'(any\s+|latest\s+|recent\s+)?news\s+(about|on|regarding|affecting|for)',
                r'news\s+impact',
                r'what.*(happening|going on).*(with|at|in)',
                r'(oil|commodity|commodities)\s+prices?\s+affecting',
                r'latest\s+(on|about|news)',
                r'breaking\s+news',
                r'market\s+news',
                r'financial\s+news',
                r'stock\s+news',
            ]
            for pattern in news_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] News pattern detected")
                    return "internet_data"

            # 1d. Current/live price requests
            price_patterns = [
                r'(current|live|latest|real-?time)\s+price',
                r"what'?s\s+the\s+price\s+of",
                r'price\s+of\s+.+\s+(today|now|currently)',
                r'how\s+much\s+(is|does)\s+.+\s+(cost|trading|worth)\s*(now|today)?',
                r'(stock|share)\s+price\s+(of|for)',
                r'trading\s+at',
                r"what'?s\s+.+\s+trading\s+at",
            ]
            for pattern in price_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Current price pattern detected")
                    return "internet_data"

            # 1e. Cryptocurrency queries
            crypto_patterns = [
                r'\b(bitcoin|btc|ethereum|eth|crypto|cryptocurrency)\b',
                r'\b(dogecoin|doge|solana|sol|cardano|ada)\b',
                r'crypto\s+price',
            ]
            for pattern in crypto_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Cryptocurrency pattern detected")
                    return "internet_data"

            # 1f. Commodity prices
            commodity_patterns = [
                r'(oil|crude|gold|silver|platinum)\s+price',
                r'price\s+of\s+(oil|crude|gold|silver|platinum)',
                r'(natural\s+gas|copper|wheat|corn)\s+price',
                r'commodity\s+prices?',
            ]
            for pattern in commodity_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Commodity pattern detected")
                    return "internet_data"

            # 1g. Market index performance (when asking about market, not portfolio comparison)
            index_patterns = [
                r"(how|what).*(s&p|sp500|nasdaq|dow\s*jones?|dow|russell).*(doing|performing|performance)",
                r"(s&p|sp500|nasdaq|dow\s*jones?|dow|russell)\s+(performance|today|this week)",
                r"(s&p|sp500|nasdaq|dow)\s+500?\s*(index)?\s*(today|now|performance)",
                r"market\s+(performance|today|update)",
                r"how\s+(is|are)\s+the\s+markets?\s+(doing|today)",
            ]
            for pattern in index_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Market index pattern detected")
                    return "internet_data"

            # 1h. Forex / Exchange rates
            forex_patterns = [
                r'exchange\s+rate',
                r'(usd|eur|gbp|aed|jpy)\s+(to|\/)\s+(usd|eur|gbp|aed|jpy)',
                r'(dollar|euro|pound|dirham)\s+(to|vs|versus)',
                r'forex',
                r'currency\s+(rate|exchange|conversion)',
            ]
            for pattern in forex_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Forex pattern detected")
                    return "internet_data"

            # 1i. Time-specific external market queries
            time_market_patterns = [
                r'(today|yesterday|this week|last week|this month)\s*.*(market|stocks?|nasdaq|dow|s&p)',
                r'(market|stocks?)\s*.*(today|yesterday|this week|last week|this month)',
                r"what\s+happened\s+(to|in|with)\s+.*(market|stocks?)",
                r'market\s+update',
                r'market\s+close',
                r'market\s+open',
            ]
            for pattern in time_market_patterns:
                if re.search(pattern, query_lower):
                    # But NOT if it's about "my portfolio today"
                    if not re.search(r'(my|our)\s+(portfolio|holdings?|stocks?)', query_lower):
                        print(f"  → [INTERNET] Time-specific market pattern detected")
                        return "internet_data"

            # ================================================================
            # STEP 2: GREETING PATTERNS
            # ================================================================
            greeting_patterns = [
                r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))[\s!.,]*$',
                r'^how\s+are\s+you',
                r'^thank(s|\s+you)',
                r'^(bye|goodbye|see\s+you|take\s+care)',
                r'^(what\s+can\s+you\s+do|help\s+me|who\s+are\s+you)',
                r"^what'?s\s+up",
            ]
            for pattern in greeting_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [GREETING] Greeting pattern detected")
                    return "greeting"

            # ================================================================
            # STEP 3: STRONG DATABASE PATTERNS
            # These indicate the user is asking about their own data
            # ================================================================
            database_patterns = [
                r'(my|our)\s+(portfolio|holdings?|positions?|stocks?|investments?)',
                r'(show|list|display|get)\s+(my|our|all)\s+(portfolio|holdings?|positions?)',
                r'(my|our)\s+(profit|loss|returns?|gains?|pnl|p&l)',
                r'(unrealized|realized)\s+(profit|loss|gains?|pnl)',
                r'(portfolio|holdings?)\s+(summary|overview|breakdown|performance)',
                r'how\s+(is|are)\s+(my|our)\s+(portfolio|holdings?|stocks?)\s+(doing|performing)',
                r'(total|current)\s+(value|worth)\s+of\s+(my|our)',
                r'what\s+(is|are)\s+(my|our)\s+(portfolio|holdings?)',
                r'(my|our)\s+top\s+\d*\s*(stocks?|holdings?|positions?)',
                r'(my|our)\s+(best|worst)\s+(performing)?\s*(stocks?|holdings?)',
                r'which\s+of\s+(my|our)\s+(stocks?|holdings?)',
                r'(ytd|mtd|qtd)\s+(profit|return|performance)\s+(of|for)?\s*(my|our)?',
                r'(my|our)\s+(ytd|mtd|qtd)',
                r'all\s+(my|our)?\s*portfolios',
                r'list\s+portfolios',
                r'show\s+portfolios',
                r'portfolio\s+names?',
                r'(how\s+many|number\s+of)\s+portfolios?',
            ]
            for pattern in database_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [DATABASE] Strong database pattern detected")
                    return "database"

            # ================================================================
            # STEP 4: SYMBOL-BASED ROUTING WITH CONTEXT
            # ================================================================
            mentioned_symbols = self._extract_potential_symbols(query)
            
            if mentioned_symbols and self.sql_executor:
                found_in_db, not_found, portfolios_found = self._check_symbols_in_database(mentioned_symbols)
                
                print(f"  → Symbols extracted: {mentioned_symbols}")
                print(f"  → Found in DB: {found_in_db}")
                print(f"  → Not in DB: {not_found}")
                print(f"  → Portfolios: {portfolios_found}")

                # Check for holdings context (indicates database query)
                holdings_context_words = [
                    'holding', 'position', 'quantity', 'shares', 'lots',
                    'bought', 'sold', 'cost basis', 'unrealized', 'realized',
                    'my', 'our', 'portfolio', 'invested'
                ]
                has_holdings_context = any(word in query_lower for word in holdings_context_words)

                if portfolios_found:
                    # Portfolio mentioned - always database
                    print(f"  → [DATABASE] Portfolio name found in DB")
                    return "database"
                
                if found_in_db and has_holdings_context:
                    # Symbol in DB + holdings context = database
                    print(f"  → [DATABASE] Symbol in DB with holdings context")
                    return "database"
                
                if found_in_db and not has_holdings_context and not not_found:
                    # Symbol in DB but no holdings context - might be asking for current price
                    # Check for price indicators
                    price_indicators = ['price', 'trading', 'worth', 'cost', 'value']
                    if any(ind in query_lower for ind in price_indicators):
                        print(f"  → [INTERNET] Symbol in DB but asking for market price")
                        return "internet_data"
                    # Default to database for ambiguous cases about known symbols
                    print(f"  → [DATABASE] Symbol in DB, no clear internet intent")
                    return "database"
                
                if not_found and not found_in_db:
                    # All symbols NOT in database - need internet
                    print(f"  → [INTERNET] Symbols not found in database")
                    return "internet_data"
                
                if not_found and found_in_db:
                    # Mixed - some in DB, some not
                    # If asking about comparison or external data, use internet
                    if any(word in query_lower for word in ['compare', 'vs', 'versus', 'against', 'benchmark']):
                        print(f"  → [DATABASE] Comparison query with mixed symbols")
                        return "database"  # Database handler will handle comparison
                    # Default to internet for external symbols
                    print(f"  → [INTERNET] Mixed symbols, external data likely needed")
                    return "internet_data"

            # ================================================================
            # STEP 5: ADDITIONAL DATABASE INDICATORS
            # ================================================================
            database_keywords = [
                'holding', 'holdings', 'position', 'positions',
                'lot', 'lots', 'quantity', 'shares',
                'unrealized', 'realized', 'pnl', 'p&l',
                'cost basis', 'average cost',
            ]
            if any(keyword in query_lower for keyword in database_keywords):
                print(f"  → [DATABASE] Database keyword detected")
                return "database"

            # ================================================================
            # STEP 6: LLM FALLBACK
            # ================================================================
            start_time = time.time()
            print(f"⏱️  Starting: Query Classification (LLM fallback)...")

            try:
                response = self.routing_chain.invoke({"query": query})
                elapsed = time.time() - start_time
                print(f"✅ LLM Classification completed in {elapsed:.2f}s")

                category = response.strip().lower()

                if "database" in category:
                    return "database"
                elif "greeting" in category or "chitchat" in category:
                    return "greeting"
                elif "internet" in category:
                    return "internet_data"
                else:
                    # Default to internet_data for unknown (safer for market queries)
                    print(f"  → [INTERNET] LLM unclear, defaulting to internet_data")
                    return "internet_data"

            except Exception as e:
                print(f"❌ LLM classification failed: {e}")
                # Default to internet_data as safer fallback
                return "internet_data"

        except Exception as e:
            print(f"❌ Error in query classification: {e}")
            import traceback
            traceback.print_exc()
            return "database"  # Fallback

    def clear_cache(self):
        """Clear the symbols and portfolios cache"""
        self._symbols_cache = None
        self._portfolios_cache = None
        print("  → Router cache cleared")
