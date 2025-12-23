# src/services/query_router.py
"""
Query Router - Routes queries to appropriate handlers based on intent classification.
Uses priority-based pattern matching with database-aware entity detection.
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
        self._group_names_cache = None

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
            input_variables=["query", "db_entities"],
            template="""You are a query classifier for a financial assistant. Classify the query into ONE category.

**IMPORTANT: Database Entities**
The following entities exist in our LOCAL database:
{db_entities}

Categories:
1. "database" - Questions about LOCAL portfolio/holdings data:
   - Any question mentioning entities from our database (portfolios, groups, account IDs)
   - Portfolio holdings, positions, quantities
   - Portfolio statistics, returns, values, metrics, descriptions
   - Questions with "my", "our" referring to portfolio data
   
2. "greeting" - Greetings, chitchat, hello, how are you, goodbye, etc.

3. "internet_data" - Questions requiring EXTERNAL real-time data:
   - Current stock prices from the market (NOT in our database)
   - Latest financial news
   - Market indices performance
   - Market movers: top gainers/losers
   - Cryptocurrency, commodity prices
   - Hypothetical investment calculations

**CRITICAL RULE:** If ANY entity from our database is mentioned, classify as "database".

User Query: {query}

Category:"""
        )

        self.routing_chain = self.routing_prompt | self.llm | StrOutputParser()

    # ==================== DATABASE ENTITY CACHING ====================

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

    def _get_all_group_names(self) -> List[str]:
        """Fetch all group names from database with caching"""
        if self._group_names_cache is not None:
            return self._group_names_cache

        if not self.sql_executor:
            return []

        try:
            query = """
            SELECT DISTINCT group_name
            FROM ai_trading.portfolio_summary
            WHERE group_name IS NOT NULL
            ORDER BY group_name
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                groups = df['group_name'].tolist()
                self._group_names_cache = groups
                return groups
            return []
        except Exception as e:
            print(f"Error fetching group names for routing: {e}")
            return []

    def _get_all_db_entities(self) -> Dict[str, List[str]]:
        """Get all database entities for matching"""
        return {
            "portfolios": self._get_all_portfolios(),
            "groups": self._get_all_group_names(),
            "symbols": self._get_all_symbols(),
        }

    # ==================== ENTITY MATCHING ====================

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching (lowercase, remove special chars)"""
        # Remove quotes and extra whitespace
        text = text.replace('"', '').replace("'", "").replace('\\', '')
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def _check_entity_in_query(self, query: str, entities: List[str], threshold: int = 80) -> Tuple[bool, List[str]]:
        """
        Check if any database entity is mentioned in the query.
        
        Args:
            query: User query
            entities: List of entity names to check
            threshold: Fuzzy match threshold (0-100)
            
        Returns:
            Tuple of (found, list of matched entities)
        """
        if not entities:
            return False, []

        query_normalized = self._normalize_text(query)
        matched = []

        # Convert entities to lowercase for comparison
        entities_lower = {e.lower(): e for e in entities}

        for entity_lower, entity_original in entities_lower.items():
            # Exact substring match
            if entity_lower in query_normalized:
                matched.append(entity_original)
                continue

            # Fuzzy match for potential typos or variations
            # Extract potential entity mentions (2+ word phrases or single words > 3 chars)
            words = query_normalized.split()
            
            # Check single words
            for word in words:
                if len(word) > 3:
                    score = fuzz.ratio(word, entity_lower)
                    if score >= threshold:
                        matched.append(entity_original)
                        break

            # Check 2-word phrases
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                score = fuzz.ratio(phrase, entity_lower)
                if score >= threshold:
                    matched.append(entity_original)
                    break

            # Check 3-word phrases
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                score = fuzz.ratio(phrase, entity_lower)
                if score >= threshold:
                    matched.append(entity_original)
                    break

        return len(matched) > 0, list(set(matched))

    def _find_db_entities_in_query(self, query: str) -> Dict[str, List[str]]:
        """
        Find all database entities mentioned in the query.
        
        Returns:
            Dict with matched portfolios, groups, and symbols
        """
        all_entities = self._get_all_db_entities()
        
        results = {
            "portfolios": [],
            "groups": [],
            "symbols": [],
        }

        # Check portfolios (most important - use lower threshold for names)
        found, matched = self._check_entity_in_query(query, all_entities["portfolios"], threshold=75)
        if found:
            results["portfolios"] = matched
            print(f"  → Found portfolio(s) in query: {matched}")

        # Check group names
        found, matched = self._check_entity_in_query(query, all_entities["groups"], threshold=75)
        if found:
            results["groups"] = matched
            print(f"  → Found group(s) in query: {matched}")

        # Check symbols (use higher threshold - tickers are exact)
        found, matched = self._check_entity_in_query(query, all_entities["symbols"], threshold=90)
        if found:
            results["symbols"] = matched
            print(f"  → Found symbol(s) in query: {matched}")

        return results

    def _has_any_db_entity(self, query: str) -> Tuple[bool, str]:
        """
        Check if query contains any database entity.
        
        Returns:
            Tuple of (has_entity, entity_type)
        """
        matches = self._find_db_entities_in_query(query)
        
        if matches["portfolios"]:
            return True, "portfolio"
        if matches["groups"]:
            return True, "group"
        if matches["symbols"]:
            return True, "symbol"
            
        return False, ""

    # ==================== PATTERN EXTRACTION ====================

    def _extract_potential_symbols(self, query: str) -> List[str]:
        """Extract potential stock symbols/company names from query using regex."""
        extracted = []
        
        # Common company name to symbol mapping
        company_map = {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA",
            "meta": "META", "facebook": "META", "netflix": "NFLX",
            "disney": "DIS", "boeing": "BA", "intel": "INTC",
            "amd": "AMD", "paypal": "PYPL", "visa": "V",
            "mastercard": "MA", "jpmorgan": "JPM", "walmart": "WMT",
        }
        
        query_lower = query.lower()
        
        # Check for company names
        for company, symbol in company_map.items():
            if company in query_lower:
                extracted.append(symbol)
        
        # Extract uppercase ticker patterns (1-5 letters)
        tickers = re.findall(r'\b([A-Z]{1,5})\b', query)
        excluded = {'I', 'A', 'THE', 'AND', 'OR', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 
                   'IS', 'IT', 'BE', 'AS', 'BY', 'IF', 'SO', 'NO', 'UP', 'DO', 'GO', 
                   'MY', 'WE', 'US', 'AN', 'AM', 'PM', 'VS', 'UK', 'EU', 'USD', 'AED', 
                   'EUR', 'GBP', 'YTD', 'MTD', 'QTD', 'PNL', 'ETF', 'IPO', 'CEO', 'CFO'}
        
        for ticker in tickers:
            if ticker not in excluded:
                extracted.append(ticker)
        
        return list(set(extracted))

    # ==================== MAIN CLASSIFICATION ====================

    def classify_query(self, query: str) -> str:
        """
        Classify a user query with priority-based pattern matching.
        
        Priority Order:
        1. Database entity check (HIGHEST - if portfolio/group name found, it's database)
        2. Internet-first patterns (hypothetical, market movers, news, prices)
        3. Greeting patterns
        4. Strong database patterns (possessive + holdings context)
        5. Symbol-based routing with context awareness
        6. LLM fallback
        """
        try:
            query_lower = query.lower()
            query_normalized = self._normalize_text(query)
            
            # ================================================================
            # STEP 0: CHECK FOR DATABASE ENTITIES FIRST (Highest Priority)
            # If query mentions a portfolio name, group name, or symbol from our DB,
            # it's almost certainly a database query
            # ================================================================
            
            has_db_entity, entity_type = self._has_any_db_entity(query)
            
            if has_db_entity:
                # Check if it's NOT asking for internet data about this entity
                internet_intent_patterns = [
                    r'(current|live|real-?time)\s+price',
                    r'news\s+(about|on|for)',
                    r'latest\s+news',
                    r'if\s+i\s+had\s+invested',
                    r'how\s+much\s+would',
                ]
                
                has_internet_intent = any(re.search(p, query_lower) for p in internet_intent_patterns)
                
                if not has_internet_intent:
                    print(f"  → [DATABASE] Found {entity_type} from database in query")
                    return "database"
                else:
                    print(f"  → Found {entity_type} but query has internet intent")

            # ================================================================
            # STEP 1: INTERNET-FIRST PATTERNS (High Priority)
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
            ]
            for pattern in market_mover_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Market movers pattern detected")
                    return "internet_data"

            # 1c. News queries (but NOT about our database entities)
            news_patterns = [
                r'(any\s+|latest\s+|recent\s+)?news\s+(about|on|regarding|affecting|for)',
                r'news\s+impact',
                r'what.*(happening|going on).*(with|at|in)',
                r'(oil|commodity|commodities)\s+prices?\s+affecting',
                r'latest\s+(on|about|news)',
                r'breaking\s+news',
                r'market\s+news',
            ]
            # Only route to internet if no DB entity was found
            if not has_db_entity:
                for pattern in news_patterns:
                    if re.search(pattern, query_lower):
                        print(f"  → [INTERNET] News pattern detected")
                        return "internet_data"

            # 1d. Current/live price requests (external market prices)
            price_patterns = [
                r'(current|live|latest|real-?time)\s+price',
                r"what'?s\s+the\s+price\s+of",
                r'price\s+of\s+.+\s+(today|now|currently)',
                r'how\s+much\s+(is|does)\s+.+\s+(cost|trading|worth)\s*(now|today)?',
                r'(stock|share)\s+price\s+(of|for)',
                r'trading\s+at',
            ]
            # Only if no DB entity
            if not has_db_entity:
                for pattern in price_patterns:
                    if re.search(pattern, query_lower):
                        print(f"  → [INTERNET] Current price pattern detected")
                        return "internet_data"

            # 1e. Cryptocurrency queries
            crypto_patterns = [
                r'\b(bitcoin|btc|ethereum|eth|crypto|cryptocurrency)\b',
                r'\b(dogecoin|doge|solana|sol|cardano|ada)\b',
            ]
            for pattern in crypto_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Cryptocurrency pattern detected")
                    return "internet_data"

            # 1f. Commodity prices
            commodity_patterns = [
                r'(oil|crude|gold|silver|platinum)\s+price',
                r'price\s+of\s+(oil|crude|gold|silver|platinum)',
                r'commodity\s+prices?',
            ]
            for pattern in commodity_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Commodity pattern detected")
                    return "internet_data"

            # 1g. Market index performance
            index_patterns = [
                r"(how|what).*(s&p|sp500|nasdaq|dow\s*jones?|dow|russell).*(doing|performing|performance)",
                r"(s&p|sp500|nasdaq|dow\s*jones?|dow|russell)\s+(performance|today|this week)",
                r"market\s+(performance|today|update)",
            ]
            for pattern in index_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Market index pattern detected")
                    return "internet_data"

            # 1h. Forex / Exchange rates
            forex_patterns = [
                r'exchange\s+rate',
                r'(usd|eur|gbp|aed|jpy)\s+(to|\/)\s+(usd|eur|gbp|aed|jpy)',
                r'forex',
                r'currency\s+(rate|exchange|conversion)',
            ]
            for pattern in forex_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [INTERNET] Forex pattern detected")
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
            ]
            for pattern in greeting_patterns:
                if re.search(pattern, query_lower):
                    print(f"  → [GREETING] Greeting pattern detected")
                    return "greeting"

            # ================================================================
            # STEP 3: STRONG DATABASE PATTERNS
            # ================================================================
            database_patterns = [
                r'(my|our)\s+(portfolio|holdings?|positions?|stocks?|investments?)',
                r'(show|list|display|get)\s+(my|our|all)\s+(portfolio|holdings?|positions?)',
                r'(my|our)\s+(profit|loss|returns?|gains?|pnl|p&l)',
                r'(unrealized|realized)\s+(profit|loss|gains?|pnl)',
                r'(portfolio|holdings?)\s+(summary|overview|breakdown|performance|description)',
                r'how\s+(is|are)\s+(my|our)\s+(portfolio|holdings?|stocks?)\s+(doing|performing)',
                r'(total|current)\s+(value|worth)\s+of\s+(my|our)',
                r'what\s+(is|are)\s+(my|our)\s+(portfolio|holdings?)',
                r'all\s+(my|our)?\s*portfolios',
                r'list\s+portfolios',
                r'show\s+portfolios',
                r'portfolio\s+names?',
                r'(how\s+many|number\s+of)\s+portfolios?',
                r'description\s+(for|of)',  # "description for X"
                r'what\s+is\s+the\s+description',
                r'(default\s+)?index\s+(for|of)',
                r'(start|inception)\s+date',
                r'cost\s+model',
                r'benchmark',
                r'group\s+name',
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
                # Check which symbols are in our database
                db_symbols = self._get_all_symbols()
                db_symbols_upper = [s.upper() for s in db_symbols]
                
                found_in_db = [s for s in mentioned_symbols if s.upper() in db_symbols_upper]
                not_in_db = [s for s in mentioned_symbols if s.upper() not in db_symbols_upper]
                
                print(f"  → Symbols extracted: {mentioned_symbols}")
                print(f"  → Found in DB: {found_in_db}")
                print(f"  → Not in DB: {not_in_db}")

                # Check for holdings context
                holdings_context = ['holding', 'position', 'quantity', 'shares', 'lots',
                                   'bought', 'sold', 'cost basis', 'unrealized', 'realized',
                                   'my', 'our', 'portfolio']
                has_holdings_context = any(word in query_lower for word in holdings_context)

                if found_in_db and has_holdings_context:
                    print(f"  → [DATABASE] Symbol in DB with holdings context")
                    return "database"
                
                if not_in_db and not found_in_db:
                    print(f"  → [INTERNET] Symbols not found in database")
                    return "internet_data"

            # ================================================================
            # STEP 5: ADDITIONAL DATABASE KEYWORDS
            # ================================================================
            database_keywords = [
                'holding', 'holdings', 'position', 'positions',
                'lot', 'lots', 'quantity', 'shares',
                'unrealized', 'realized', 'pnl', 'p&l',
                'cost basis', 'average cost', 'allocated', 'utilized',
            ]
            if any(keyword in query_lower for keyword in database_keywords):
                print(f"  → [DATABASE] Database keyword detected")
                return "database"

            # ================================================================
            # STEP 6: LLM FALLBACK WITH DATABASE CONTEXT
            # ================================================================
            start_time = time.time()
            print(f"⏱️  Starting: Query Classification (LLM fallback)...")

            try:
                # Provide database entities to LLM for context
                all_entities = self._get_all_db_entities()
                db_entities_str = f"""
Portfolios: {', '.join(all_entities['portfolios'][:20])}{'...' if len(all_entities['portfolios']) > 20 else ''}
Groups: {', '.join(all_entities['groups'][:10])}
"""
                response = self.routing_chain.invoke({
                    "query": query,
                    "db_entities": db_entities_str
                })
                
                elapsed = time.time() - start_time
                print(f"✅ LLM Classification completed in {elapsed:.2f}s")

                category = response.strip().lower()

                if "database" in category:
                    return "database"
                elif "greeting" in category:
                    return "greeting"
                elif "internet" in category:
                    return "internet_data"
                else:
                    # Default to internet_data for unknown
                    return "internet_data"

            except Exception as e:
                print(f"❌ LLM classification failed: {e}")
                return "internet_data"

        except Exception as e:
            print(f"❌ Error in query classification: {e}")
            import traceback
            traceback.print_exc()
            return "database"  # Safe fallback

    def clear_cache(self):
        """Clear all caches"""
        self._symbols_cache = None
        self._portfolios_cache = None
        self._group_names_cache = None
        print("  → Router cache cleared")