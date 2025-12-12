from typing import Dict, List, Tuple, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config
import os
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

        # Define routing prompt template
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

3. "internet_comparison" - Questions that EXPLICITLY require fetching EXTERNAL internet data and comparing with local data:
   - Must mention external entities NOT in our database (like market indices: SPY, S&P 500, NASDAQ,)
   - Must involve comparison between our data and external/market data
   Examples: "compare our portfolio with QQQ", "how do we perform vs the market", "benchmark against S&P 500"

Rules:
- Return ONLY the category name: database, greeting, or internet_comparison
- Do not include any explanation or additional text
- Questions about local data analysis, rankings, or comparisons within our database = "database"
- Only use "internet_comparison" when external data is EXPLICITLY needed
- If unsure, default to "database" for data-related queries

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
        if hasattr(self, '_portfolios_cache') and self._portfolios_cache is not None:
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
                    score_cutoff=80  # Slightly lower threshold to catch "ABalanced" -> "A-Balanced"
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
        Check if the query is asking for a comparison

        Args:
            query: User's input query

        Returns:
            True if query contains comparison keywords
        """
        comparison_keywords = ['compare', 'vs', 'versus', 'against', 'difference between',
                              'better than', 'worse than', 'performance of', 'benchmark']
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in comparison_keywords)

    def classify_query(self, query: str) -> str:
        """
        Classify a user query into one of the routing categories with database-aware logic

        Args:
            query: User's input query

        Returns:
            Category string: "database", "greeting", or "internet_comparison"
        """
        try:
            # Check for database-related keywords that indicate local data queries
            database_keywords = [
                'holding', 'holdings', 'portfolio', 'position', 'positions',
                'unrealized', 'profit', 'loss', 'gain', 'return',
                'my stock', 'my stocks', 'our stock', 'our stocks',
                'quantity', 'lot', 'lots', 'shares', 'invested',
                'total value', 'market value', 'cost basis'
            ]

            query_lower = query.lower()
            has_database_keywords = any(keyword in query_lower for keyword in database_keywords)

            # Check for external market indicators
            external_market_keywords = ['qqq', 'spy', 's&p 500', 's&p500', 'market index',
                                       'nasdaq', 'dow jones', 'benchmark against']
            has_external_keywords = any(keyword in query_lower for keyword in external_market_keywords)

            # If it has database keywords and NO external market keywords, route to database
            if has_database_keywords and not has_external_keywords:
                print(f"  → Query contains database keywords, routing to 'database'")
                return "database"

            # First, check if query mentions any stocks
            mentioned_terms = self._extract_stock_symbols(query)

            if mentioned_terms and self.sql_executor:
                # Check which symbols are in our database
                found_in_db, not_found, portfolios_found = self._check_symbols_in_database(mentioned_terms)

                print(f"  → Stock symbols extracted: {mentioned_terms}")
                print(f"  → Found in database: {found_in_db}")
                print(f"  → Portfolios found: {portfolios_found}")
                print(f"  → Not found in database: {not_found}")

                # Smart routing based on database availability and query type
                if found_in_db and not not_found:
                    # All mentioned stocks/portfolios are in database -> route to database
                    print(f"  → All symbols found in database, routing to 'database'")
                    return "database"
                elif portfolios_found and not_found:
                    # Portfolio comparison with external stock (e.g., "Compare ABalanced vs QQQ")
                    # Route to database - the database handler can handle portfolio data
                    # and the user can fetch external stock data separately if needed
                    print(f"  → Portfolio comparison detected (portfolio in DB, external stock mentioned)")
                    print(f"  → Routing to 'database' - handler will work with available portfolio data")
                    return "database"
                elif found_in_db and not_found and not portfolios_found:
                    # Mixed stocks: some in DB, some not, no portfolios -> needs internet comparison
                    print(f"  → Mixed stocks (some in DB, some not), routing to 'internet_comparison'")
                    return "internet_comparison"
                elif not_found and not found_in_db:
                    # All mentioned stocks not in database -> needs internet
                    print(f"  → No stocks found in database, routing to 'internet_comparison'")
                    return "internet_comparison"

            # Fall back to LLM classification for non-stock queries
            start_time = time.time()
            print(f"⏱️  Starting: Query Classification (LLM)...")

            response = self.routing_chain.invoke({"query": query})

            elapsed = time.time() - start_time
            print(f"✅ Completed: Query Classification in {elapsed:.2f}s")

            category = response.strip().lower()

            # Normalize the response
            if "database" in category:
                return "database"
            elif "greeting" in category or "chitchat" in category:
                return "greeting"
            elif "internet" in category or "comparison" in category:
                return "internet_comparison"
            else:
                # Default to database for data-related queries
                return "database"

        except Exception as e:
            print(f"❌ Error in query classification: {e}")
            return "database"  # Default fallback