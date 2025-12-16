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
        Extract potential stock symbols/names from query using LLM

        Args:
            query: User's question

        Returns:
            List of potential stock symbols/names mentioned
        """
        extraction_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Extract any stock names, company names, or stock symbols mentioned in this question.
Return only the extracted terms, separated by commas. If none found, return "NONE".

Examples:
- "Compare ABalanced performance against QQQ" -> ABalanced, QQQ
- "What is the price of AAPL stock?" -> AAPL
- "Show me MSFT and GOOGL performance" -> MSFT, GOOGL
- "How is Tesla doing?" -> Tesla
- "What are my portfolios?" -> NONE

Question: {query}

Extracted terms (comma-separated):"""
        )

        extraction_chain = extraction_prompt | self.llm | StrOutputParser()

        try:
            start_time = time.time()
            print(f"⏱️  Starting: Stock Symbol Extraction...")

            result = extraction_chain.invoke({"query": query})

            elapsed = time.time() - start_time
            print(f"✅ Completed: Stock Symbol Extraction in {elapsed:.2f}s")

            result = result.strip()

            if result.upper() == "NONE" or not result:
                return []

            # Split by comma and clean
            terms = [term.strip() for term in result.split(',') if term.strip()]
            return terms
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Failed: Stock Symbol Extraction after {elapsed:.2f}s - Error: {e}")
            return []

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
        Classify a user query into one of the routing categories with database-aware logic

        Args:
            query: User's input query

        Returns:
            Category string: "database", "greeting", "internet_data", or "comparison"
        """
        try:
            query_lower = query.lower()

            # STEP 1: Quick check for comparison queries (highest priority for mixed queries)
            if self._is_comparison_query(query):
                print(f"  → Detected comparison query pattern, routing to 'comparison'")
                return "comparison"

            # STEP 2: Check for database-related keywords that indicate local data queries
            database_keywords = [
                'holding', 'holdings', 'portfolio', 'position', 'positions',
                'unrealized', 'profit', 'loss', 'gain', 'return',
                'my stock', 'my stocks', 'our stock', 'our stocks',
                'quantity', 'lot', 'lots', 'shares', 'invested',
                'total value', 'market value', 'cost basis'
            ]

            has_database_keywords = any(keyword in query_lower for keyword in database_keywords)

            # STEP 3: Check for real-time internet data keywords
            internet_data_keywords = [
                'current price', 'latest news', 'news on', 'price of',
                'performance today', 'today\'s performance', 'real-time',
                'live price', 'market news', 'stock news',
                'bitcoin', 'crypto', 'cryptocurrency', 'btc', 'eth',
                's&p 500 performance', 'nasdaq performance', 'dow jones performance',
                'market update', 'latest on', 'what\'s the price',
                'top gainers', 'top losers', 'biggest movers', 'most active',
                'trending stocks', 'market leaders', 'market laggards',
                'gainers in', 'losers in', 'movers in',
                'this week', 'this month', 'today', 'yesterday',
                'last week', 'last month', 'recent',
                'nasdaq', 'dow jones', 's&p 500', 'sp500', 's&p',
                'russell', 'market index', 'sector performance'
            ]
            has_internet_data_keywords = any(keyword in query_lower for keyword in internet_data_keywords)

            # STEP 4: If query has BOTH local and internet keywords, it might be comparison
            if has_database_keywords and has_internet_data_keywords:
                # Check if there's any comparison intent
                comparison_hints = ['vs', 'versus', 'against', 'compare', 'compared', 'benchmark', 'outperform']
                if any(hint in query_lower for hint in comparison_hints):
                    print(f"  → Query has both local and external keywords with comparison intent, routing to 'comparison'")
                    return "comparison"

            # STEP 5: If query explicitly asks for real-time/current data only, route to internet_data
            if has_internet_data_keywords and not has_database_keywords:
                print(f"  → Query requires real-time internet data, routing to 'internet_data'")
                return "internet_data"

            # STEP 6: If it has database keywords and NO internet data keywords, route to database
            if has_database_keywords and not has_internet_data_keywords:
                print(f"  → Query contains database keywords, routing to 'database'")
                return "database"

            # STEP 7: Check for stock symbol mentions
            mentioned_terms = self._extract_stock_symbols(query)

            if mentioned_terms and self.sql_executor:
                found_in_db, not_found, portfolios_found = self._check_symbols_in_database(mentioned_terms)

                print(f"  → Stock symbols extracted: {mentioned_terms}")
                print(f"  → Found in database: {found_in_db}")
                print(f"  → Portfolios found: {portfolios_found}")
                print(f"  → Not found in database: {not_found}")

                # Smart routing based on database availability and query type
                if found_in_db and not not_found:
                    # All mentioned stocks/portfolios are in database
                    print(f"  → All symbols found in database, routing to 'database'")
                    return "database"
                elif portfolios_found and not_found:
                    # Portfolio + external reference - likely a comparison
                    # Check for comparison keywords
                    if any(kw in query_lower for kw in ['vs', 'versus', 'against', 'compare', 'benchmark']):
                        print(f"  → Portfolio with external reference detected, routing to 'comparison'")
                        return "comparison"
                    print(f"  → Portfolio comparison detected, routing to 'database'")
                    return "database"
                elif not_found and not found_in_db:
                    # All mentioned stocks not in database
                    print(f"  → No stocks found in database, routing to 'internet_data'")
                    return "internet_data"

            # STEP 8: Fall back to LLM classification
            start_time = time.time()
            print(f"⏱️  Starting: Query Classification (LLM)...")

            response = self.routing_chain.invoke({"query": query})

            elapsed = time.time() - start_time
            print(f"✅ Completed: Query Classification in {elapsed:.2f}s")

            category = response.strip().lower()

            # Normalize the response
            if "comparison" in category:
                return "comparison"
            elif "database" in category:
                return "database"
            elif "greeting" in category or "chitchat" in category:
                return "greeting"
            elif "internet_data" in category or "internet" in category:
                return "internet_data"
            else:
                # Default to database for data-related queries
                return "database"

        except Exception as e:
            print(f"❌ Error in query classification: {e}")
            return "database"  # Default fallback