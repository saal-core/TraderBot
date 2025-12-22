"""
Query Router - Routes queries to appropriate handlers based on intent classification.
Includes detection for comparison queries that require both local and external data.
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
        Initialize the query router (always uses Ollama)

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

        # Define routing prompt template with comparison category
        self.routing_prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are a query classifier. Classify the following user query into ONE of these categories:

Categories:
1. "database" - Questions about data in a local database, including:
   - Portfolio holdings, positions, quantities, lots
   - Stock performance, profit/loss, unrealized gains
   - Portfolio statistics, returns, values, metrics
   - Rankings, comparisons, aggregations of local data (e.g., "highest profit", "best performing", "top 10 stocks")
   - Current state of holdings (e.g., "which stock has the highest value right now")
   Examples: "show me all portfolios", "which stock has the highest profit", "what are my holdings", "top 5 stocks by return"

2. "greeting" - Greetings, chitchat, hello, how are you, goodbye, thank you, etc.

3. "internet_data" - Questions that require fetching REAL-TIME data from the internet ONLY:
   - Current stock prices, cryptocurrency prices, forex rates
   - Latest financial news about companies or markets
   - Market indices performance (S&P 500, NASDAQ, Dow Jones, etc.)
   - Market movers: top gainers, top losers, trending stocks
   - Real-time market data not in our database
   Examples: "What's the current price of Tesla?", "Give me the latest news on Apple", "What's Bitcoin's price?"

4. "comparison" - Questions that COMPARE local portfolio/holdings data WITH external market data:
   - Portfolio performance vs market indices (S&P 500, NASDAQ, etc.)
   - Portfolio returns vs benchmark returns
   - Local holdings vs current market prices
   - Any comparison between "my/our portfolio/holdings" and external data
   Examples: 
   - "Compare my portfolio to S&P 500"
   - "How does my portfolio perform against the market?"
   - "Is my portfolio outperforming NASDAQ?"
   - "Compare A-Balanced portfolio returns vs S&P 500 YTD"
   - "How do my holdings compare to current market prices?"
   - "What's the difference between my returns and the benchmark?"

Rules:
- Return ONLY the category name: database, greeting, internet_data, or comparison
- Do not include any explanation or additional text
- If query mentions BOTH local data (portfolio, holdings, my stocks) AND external benchmarks/indices → "comparison"
- Questions about local data analysis only = "database"
- Questions requiring ONLY real-time internet data = "internet_data"
- If query uses words like "vs", "against", "compared to", "benchmark", "outperform" with portfolio data → "comparison"
- If unsure between database and comparison, prefer "comparison" if external reference is mentioned

User Query: {query}

Category:"""
        )

        self.routing_chain = self.routing_prompt | self.llm | StrOutputParser()

    def _get_all_symbols(self) -> List[str]:
        """
        Fetch all stock symbols from database with caching

        Returns:
            List of all available stock symbols
        """
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
        """
        Fetch all portfolio names from database with caching

        Returns:
            List of all available portfolio names
        """
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

    def _extract_stock_symbols(self, query: str) -> List[str]:
        """
        Extract stock symbols/portfolio names from query by fuzzy matching against database.
        Fast approach using rapidfuzz against cached database symbols/portfolios.

        Args:
            query: User's question

        Returns:
            List of matched stock symbols/portfolio names
        """
        # Get all symbols and portfolios from database (cached)
        all_symbols = self._get_all_symbols()
        all_portfolios = self._get_all_portfolios()
        all_data_sources = all_symbols + all_portfolios
        
        if not all_data_sources:
            return []
        
        # Tokenize the query into words
        words = re.findall(r'\b\w+\b', query)
        
        extracted = []
        all_data_upper = [s.upper() for s in all_data_sources]
        
        for word in words:
            word_upper = word.upper()
            
            # Skip very short words (less than 2 chars)
            if len(word) < 2:
                continue
            
            # Direct match first
            if word_upper in all_data_upper:
                matched_value = all_data_sources[all_data_upper.index(word_upper)]
                if matched_value not in extracted:
                    extracted.append(matched_value)
                continue
            
            # Fuzzy match using rapidfuzz
            best_match = process.extractOne(
                word_upper,
                all_data_upper,
                scorer=fuzz.ratio,
                score_cutoff=80
            )
            if best_match:
                matched_value = all_data_sources[all_data_upper.index(best_match[0])]
                if matched_value not in extracted:
                    extracted.append(matched_value)
        
        return extracted

    def _check_symbols_in_database(self, mentioned_terms: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Check which mentioned terms exist in our database (stocks or portfolios)

        Args:
            mentioned_terms: List of stock symbols/portfolio names from query

        Returns:
            Tuple of (terms_found_in_db, terms_not_found, portfolios_found)
        """
        if not mentioned_terms:
            return [], [], []

        all_symbols = self._get_all_symbols()
        all_portfolios = self._get_all_portfolios()

        # Combine both for comprehensive checking
        all_data_sources = all_symbols + all_portfolios

        if not all_data_sources:
            return [], mentioned_terms, []

        found = []
        not_found = []
        portfolios_found = []

        # Convert all to uppercase for comparison
        all_symbols_upper = [s.upper() for s in all_symbols]
        all_portfolios_upper = [p.upper() for p in all_portfolios]
        all_data_upper = [s.upper() for s in all_data_sources]

        for term in mentioned_terms:
            term_upper = term.upper()
            # Check if it's a portfolio first
            if term_upper in all_portfolios_upper:
                found.append(term)
                portfolios_found.append(term)
            # Direct match in symbols
            elif term_upper in all_symbols_upper:
                found.append(term)
            else:
                # Fuzzy match using rapidfuzz
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
                    # Check if matched value is a portfolio
                    if matched_value.upper() in all_portfolios_upper:
                        portfolios_found.append(term)
                else:
                    not_found.append(term)

        return found, not_found, portfolios_found

    def _is_comparison_query(self, query: str) -> bool:
        """
        Quick heuristic check if the query is asking for a comparison
        between local data and external data.

        Args:
            query: User's input query

        Returns:
            True if query appears to be a comparison between local and external data
        """
        query_lower = query.lower()

        # Keywords that indicate comparison
        comparison_keywords = [
            'compare', 'vs', 'versus', 'against', 'benchmark',
            'outperform', 'underperform', 'beat', 'relative to',
            'difference between', 'better than', 'worse than',
            'compared to', 'stack up', 'measure against'
        ]

        # External/market reference keywords
        external_keywords = [
            's&p', 'sp500', 's&p 500', 'nasdaq', 'dow jones', 'dow',
            'market', 'index', 'benchmark', 'russell', 'nyse',
            'market average', 'market return', 'market performance'
        ]

        # Local data reference keywords
        local_keywords = [
            'my portfolio', 'our portfolio', 'my holdings', 'our holdings',
            'my stocks', 'our stocks', 'my returns', 'our returns',
            'portfolio', 'holdings'
        ]

        # Check for comparison keyword
        has_comparison = any(kw in query_lower for kw in comparison_keywords)

        # Check for external reference
        has_external = any(kw in query_lower for kw in external_keywords)

        # Check for local reference
        has_local = any(kw in query_lower for kw in local_keywords)

        # It's a comparison if it has comparison keyword AND mentions both local and external
        # OR if it has comparison patterns like "portfolio vs S&P"
        if has_comparison and has_external and has_local:
            return True

        # Check for specific comparison patterns
        comparison_patterns = [
            r'portfolio.*(?:vs|versus|against|to).*(?:s&p|nasdaq|dow|index|market|benchmark)',
            r'(?:my|our).*(?:vs|versus|against|compared).*(?:s&p|nasdaq|dow|index|market)',
            r'(?:compare|comparing).*portfolio.*(?:to|with|against)',
            r'(?:how|what).*portfolio.*(?:compare|perform).*(?:against|vs|versus)',
            r'(?:outperform|underperform|beat).*(?:market|index|benchmark)',
            r'(?:returns?|performance).*(?:vs|versus|against).*(?:s&p|nasdaq|market)',
        ]

        for pattern in comparison_patterns:
            if re.search(pattern, query_lower):
                return True

        return False
        
    def classify_query(self, query: str) -> str:
        """
        Classify a user query with corrected priority ordering
        """
        query_lower = query.lower()

        # ============================================================
        # STEP 1: INTERNET-FIRST PATTERNS (Highest Priority)
        # These patterns ALWAYS indicate internet data is needed,
        # regardless of whether stock symbols exist in our database
        # ============================================================
        
        # Hypothetical investment calculations - ALWAYS internet
        hypothetical_patterns = [
            r'if i had invested',
            r'had i invested', 
            r'if i invested',
            r'would it be worth',
            r'would be worth now',
            r'how much would .* be worth',
            r'what would .* be worth',
            r'invested .* in january|february|march|april|may|june|july|august|september|october|november|december',
        ]
        for pattern in hypothetical_patterns:
            if re.search(pattern, query_lower):
                print(f"  → Hypothetical investment query detected, routing to 'internet_data'")
                return "internet_data"

        # Market movers / rankings - ALWAYS internet (real-time data)
        market_mover_patterns = [
            r'top (gainers?|losers?|movers?)',
            r'biggest (gainers?|losers?|movers?)',
            r'(gainers?|losers?|movers?) (in|on|for) .*(nasdaq|nyse|s&p|dow|market)',
            r'(nasdaq|nyse|s&p|dow|russell).*this (week|month|day)',
            r'this (week|month).*(?:nasdaq|nyse|s&p|dow)',
            r'trending stocks',
            r'most active stocks',
        ]
        for pattern in market_mover_patterns:
            if re.search(pattern, query_lower):
                print(f"  → Market movers/rankings query detected, routing to 'internet_data'")
                return "internet_data"

        # News queries - ALWAYS internet
        news_patterns = [
            r'(any |latest |recent )?news (about|on|regarding|affecting)',
            r'news impact',
            r'what.*(happening|going on).*(with|at|in)',
            r'(oil|commodity|commodities) prices? affecting',
        ]
        for pattern in news_patterns:
            if re.search(pattern, query_lower):
                print(f"  → News query detected, routing to 'internet_data'")
                return "internet_data"

        # Current/live price requests - ALWAYS internet
        price_patterns = [
            r'(current|live|latest|real-?time) price',
            r"what'?s the price of",
            r'price of .* (today|now|currently)',
            r'how much (is|does) .* (cost|trading|worth) (now|today)',
        ]
        for pattern in price_patterns:
            if re.search(pattern, query_lower):
                print(f"  → Current price query detected, routing to 'internet_data'")
                return "internet_data"

        # ============================================================
        # STEP 2: COMPARISON QUERIES
        # ============================================================
        if self._is_comparison_query(query):
            print(f"  → Detected comparison query pattern, routing to 'comparison'")
            return "comparison"

        # ============================================================
        # STEP 3: GREETING CHECK
        # ============================================================
        greeting_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)[\s!.,]*$',
            r'^how are you',
            r'^thank(s| you)',
            r'^bye|goodbye|see you',
        ]
        for pattern in greeting_patterns:
            if re.search(pattern, query_lower):
                return "greeting"

        # ============================================================
        # STEP 4: DATABASE-SPECIFIC PATTERNS
        # Only route to database if query is clearly about LOCAL data
        # ============================================================
        
        # Strong database indicators (possessive + data terms)
        database_patterns = [
            r'(my|our) (portfolio|holdings?|positions?|stocks?|investments?)',
            r'(show|list|display) (my|our|all) (portfolio|holdings?|positions?)',
            r'(my|our) (profit|loss|returns?|gains?)',
            r'(unrealized|realized) (profit|loss|gains?|pnl)',
            r'(portfolio|holdings?) (summary|overview|breakdown)',
            r'how (is|are) (my|our) (portfolio|holdings?|stocks?) (doing|performing)',
            r'(total|current) (value|worth) of (my|our)',
        ]
        for pattern in database_patterns:
            if re.search(pattern, query_lower):
                print(f"  → Local portfolio query detected, routing to 'database'")
                return "database"

        # ============================================================
        # STEP 5: SYMBOL-BASED ROUTING (with context awareness)
        # ============================================================
        mentioned_terms = self._extract_stock_symbols(query)
        
        if mentioned_terms and self.sql_executor:
            found_in_db, not_found, portfolios_found = self._check_symbols_in_database(mentioned_terms)
            
            # Only route to database if:
            # 1. Symbols are found AND
            # 2. Query is asking about holdings/positions (not general info)
            holdings_context_words = [
                'holding', 'position', 'quantity', 'shares', 'lots',
                'bought', 'sold', 'cost basis', 'unrealized', 'realized',
                'my', 'our', 'portfolio'
            ]
            
            has_holdings_context = any(word in query_lower for word in holdings_context_words)
            
            if found_in_db and has_holdings_context:
                print(f"  → Symbols {found_in_db} found in DB with holdings context, routing to 'database'")
                return "database"
            elif found_in_db and not has_holdings_context:
                # Symbol exists in DB but query might be asking for external info
                # e.g., "What's the news on NVIDIA?" - NVIDIA in DB but asking for news
                print(f"  → Symbols {found_in_db} found but no holdings context, routing to 'internet_data'")
                return "internet_data"
            elif not_found:
                print(f"  → Symbols {not_found} not in DB, routing to 'internet_data'")
                return "internet_data"

        # ============================================================
        # STEP 6: LLM FALLBACK
        # ============================================================
        try:
            response = self.routing_chain.invoke({"query": query})
            category = response.strip().lower()
            
            if "comparison" in category:
                return "comparison"
            elif "database" in category:
                return "database"
            elif "greeting" in category:
                return "greeting"
            else:
                return "internet_data"
        except Exception as e:
            print(f"❌ LLM classification failed: {e}")
            return "internet_data"  # Safer default for unknown queries