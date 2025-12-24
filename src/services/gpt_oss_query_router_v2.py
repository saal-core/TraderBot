# src/services/optimized_query_router.py
"""
Optimized Query Router - Fast classification with minimal LLM calls.
Target: <500ms for most queries, <2s for LLM fallback.
"""
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import re
import time
import hashlib

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config

from dotenv import load_dotenv
load_dotenv()


class OptimizedQueryRouter:
    """
    High-performance query router using a tiered classification approach:
    
    Tier 1: Regex pattern matching (instant, <10ms)
    Tier 2: Keyword + entity matching (fast, <50ms)
    Tier 3: LLM classification (slow, fallback only)
    
    Goal: Route 90%+ of queries without LLM.
    """

    def __init__(self, model_name: str = None, sql_executor=None, enable_cache: bool = True):
        """
        Initialize the optimized router.

        Args:
            model_name: Ollama model name (use smaller model for speed)
            sql_executor: SQL executor for database entity lookup
            enable_cache: Whether to cache LLM responses
        """
        self.sql_executor = sql_executor
        self.enable_cache = enable_cache
        
        # Caches
        self._portfolios_cache = None
        self._groups_cache = None
        self._symbols_cache = None
        self._llm_response_cache = {}  # Query hash -> category
        
        # Initialize LLM (only used as fallback)
        ollama_config = get_ollama_config()
        self.model_name = model_name or ollama_config["model_name"]
        self.base_url = ollama_config["base_url"]
        
        self._llm = None  # Lazy initialization
        
        # Pre-compile regex patterns for speed
        self._compile_patterns()

    def _is_comparison_query(self, query: str) -> bool:
        """
        Check if query requires comparing local portfolio data with external market data.
        Only returns True for queries that EXPLICITLY compare portfolio vs market/benchmark.
        """
        query_lower = query.lower()
    
        # Must have BOTH portfolio reference AND external benchmark reference
        portfolio_indicators = [
            'my portfolio', 'our portfolio', 'portfolio', 
            'my returns', 'our returns', 'my performance'
        ]
        
        external_indicators = [
            's&p', 'sp500', 's&p 500', 'nasdaq', 'dow', 'dow jones',
            'market', 'benchmark', 'index', 'russell'
        ]
        
        comparison_verbs = [
            'compare', 'vs', 'versus', 'against', 'relative to',
            'outperform', 'underperform', 'beat', 'how does', 'how do'
        ]
        
        has_portfolio = any(ind in query_lower for ind in portfolio_indicators)
        has_external = any(ind in query_lower for ind in external_indicators)
        has_comparison = any(verb in query_lower for verb in comparison_verbs)
        
        # Need all three for a true comparison query
        return has_portfolio and has_external and has_comparison

    def _compile_patterns(self):
        """Pre-compile all regex patterns for faster matching."""
        
        # ============================================================
        # TIER 1: GREETING PATTERNS (Highest priority, instant match)
        # ============================================================
        self._greeting_patterns = [
            re.compile(r'^(hi|hello|hey|yo)[\s!.,]*$', re.IGNORECASE),
            re.compile(r'^good\s+(morning|afternoon|evening|day)[\s!.,]*$', re.IGNORECASE),
            re.compile(r'^(how\s+are\s+you|how\'s\s+it\s+going|what\'s\s+up)[\s?!.,]*$', re.IGNORECASE),
            re.compile(r'^thank(s|\s+you)[\s!.,]*$', re.IGNORECASE),
            re.compile(r'^(bye|goodbye|see\s+you|take\s+care)[\s!.,]*$', re.IGNORECASE),
            re.compile(r'^(who\s+are\s+you|what\s+can\s+you\s+do)[\s?]*$', re.IGNORECASE),
        ]
        
        # ============================================================
        # TIER 1: INTERNET PATTERNS (High confidence, no DB needed)
        # ============================================================
        self._internet_patterns = [
            # Hypothetical investments
            re.compile(r'if\s+i\s+(had\s+)?invested', re.IGNORECASE),
            re.compile(r'would\s+(it\s+)?be\s+worth', re.IGNORECASE),
            re.compile(r'how\s+much\s+would.*worth', re.IGNORECASE),
            
            # Market movers with exchange
            re.compile(r'(top|biggest)\s+(gainers?|losers?|movers?)\s+(in|on)\s+(nasdaq|nyse|the\s+market)', re.IGNORECASE),
            re.compile(r'(nasdaq|nyse|dow|s&p)\s+(gainers?|losers?|movers?)', re.IGNORECASE),
            re.compile(r'(gainers?|losers?)\s+(in|on)\s+(nasdaq|nyse)', re.IGNORECASE),
            re.compile(r'trending\s+stocks?\s+(in|on)\s+(the\s+)?(market|nasdaq|nyse)', re.IGNORECASE),
            re.compile(r'market\s+movers?\s+(today|this\s+week)', re.IGNORECASE),
            
            # Current prices (external)
            re.compile(r'(current|live|real-?time)\s+price', re.IGNORECASE),
            re.compile(r'what\'?s\s+(the\s+)?price\s+of', re.IGNORECASE),
            re.compile(r'how\s+much\s+(is|does)\s+\w+\s+(cost|trading)', re.IGNORECASE),
            
            # News
            re.compile(r'(latest|recent|any)\s+news\s+(about|on|for)', re.IGNORECASE),
            re.compile(r'news\s+(about|on|affecting)', re.IGNORECASE),
            
            # Crypto
            re.compile(r'\b(bitcoin|btc|ethereum|eth|crypto)\s*(price)?', re.IGNORECASE),
            
            # Commodities
            re.compile(r'(oil|gold|silver|crude)\s+price', re.IGNORECASE),
            re.compile(r'price\s+of\s+(oil|gold|silver|crude)', re.IGNORECASE),
            
            # Forex
            re.compile(r'(usd|eur|gbp|aed)\s*(to|\/)\s*(usd|eur|gbp|aed)', re.IGNORECASE),
            re.compile(r'exchange\s+rate', re.IGNORECASE),
            
            # Market indices performance
            re.compile(r'(s&p|nasdaq|dow)\s*(500)?\s*(performance|today|this\s+week)', re.IGNORECASE),
            re.compile(r'how\s+(is|are)\s+(the\s+)?(market|markets)\s+(doing|today)', re.IGNORECASE),
        ]
        
        # ============================================================
        # TIER 1: DATABASE PATTERNS (High confidence)
        # ============================================================
        self._database_patterns = [
            # Possessive + portfolio terms
            re.compile(r'(my|our)\s+(portfolio|holdings?|positions?|stocks?|investments?)', re.IGNORECASE),
            re.compile(r'(my|our)\s+(profit|loss|returns?|gains?|pnl)', re.IGNORECASE),
            
            # Portfolio operations
            re.compile(r'(show|list|display|get)\s+(all\s+)?(my\s+)?(portfolio|holdings?|positions?)', re.IGNORECASE),
            re.compile(r'(all|list)\s+(my\s+)?portfolios?', re.IGNORECASE),
            re.compile(r'how\s+many\s+portfolios?', re.IGNORECASE),
            
            # Portfolio attributes
            re.compile(r'(description|benchmark|default\s+index|cost\s+model|inception)\s+(for|of)', re.IGNORECASE),
            re.compile(r'what\s+is\s+the\s+(description|benchmark|default\s+index)', re.IGNORECASE),
            
            # Unrealized/realized
            re.compile(r'(unrealized|realized)\s+(profit|loss|gains?|pnl)', re.IGNORECASE),
            
            # Portfolio performance (local)
            re.compile(r'(ytd|mtd|qtd)\s+(return|profit|performance)', re.IGNORECASE),
            re.compile(r'(return|profit|performance)\s+(ytd|mtd|qtd)', re.IGNORECASE),
            
            # Portfolio rankings (NOT market movers)
            re.compile(r'^(top|bottom)\s+\d+\s*(profit|loss|profitable|losing)?\s*stocks?$', re.IGNORECASE),
            re.compile(r'^(best|worst)\s+(performing\s+)?(stocks?|investments?|holdings?)', re.IGNORECASE),
            re.compile(r'(most|least)\s+profitable\s+(stocks?|investments?)', re.IGNORECASE),
            re.compile(r'(highest|lowest)\s+(profit|return|gain|loss)', re.IGNORECASE),
            re.compile(r'which\s+(stock|investment)\s+has\s+(the\s+)?(highest|best|lowest|worst)', re.IGNORECASE),
            re.compile(r'what\s+investment\s+has\s+(the\s+)?best', re.IGNORECASE),
            re.compile(r'list\s+(the\s+)?top\s+\d+\s+(most\s+)?profitable', re.IGNORECASE),
            re.compile(r'list\s+(the\s+)?bottom\s+\d+', re.IGNORECASE),
        ]
        
        # ============================================================
        # TIER 2: KEYWORD SETS (Fast lookup)
        # ============================================================
        self._database_keywords = {
            'holding', 'holdings', 'portfolio', 'portfolios', 'position', 'positions',
            'unrealized', 'realized', 'pnl', 'allocated', 'utilized',
            'lot', 'lots', 'quantity', 'shares', 'cost basis',
        }
        
        self._internet_keywords = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'forex', 'exchange rate',
            'oil price', 'gold price', 'commodity',
            'market news', 'latest news', 'breaking',
        }

    # ==================== DATABASE ENTITY CACHING ====================

    def _get_portfolios_set(self) -> set:
        """Get portfolio names as a lowercase set for fast lookup."""
        if self._portfolios_cache is not None:
            return self._portfolios_cache

        if not self.sql_executor:
            return set()

        try:
            query = """
            SELECT DISTINCT LOWER(portfolio_name) as name
            FROM ai_trading.portfolio_summary
            WHERE portfolio_name IS NOT NULL
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                self._portfolios_cache = set(df['name'].tolist())
                return self._portfolios_cache
            return set()
        except Exception as e:
            print(f"Error fetching portfolios: {e}")
            return set()

    def _get_groups_set(self) -> set:
        """Get group names as a lowercase set for fast lookup."""
        if self._groups_cache is not None:
            return self._groups_cache

        if not self.sql_executor:
            return set()

        try:
            query = """
            SELECT DISTINCT LOWER(group_name) as name
            FROM ai_trading.portfolio_summary
            WHERE group_name IS NOT NULL
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                self._groups_cache = set(df['name'].tolist())
                return self._groups_cache
            return set()
        except Exception as e:
            print(f"Error fetching groups: {e}")
            return set()

    def _get_symbols_set(self) -> set:
        """Get symbols as a lowercase set for fast lookup."""
        if self._symbols_cache is not None:
            return self._symbols_cache

        if not self.sql_executor:
            return set()

        try:
            query = """
            SELECT DISTINCT LOWER(symbol) as symbol
            FROM ai_trading.portfolio_holdings
            WHERE symbol IS NOT NULL
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                self._symbols_cache = set(df['symbol'].tolist())
                return self._symbols_cache
            return set()
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return set()

    # ==================== TIER 1: PATTERN MATCHING ====================

    def _tier1_pattern_match(self, query: str) -> Optional[str]:
        """
        Tier 1: Fast regex pattern matching.
        Returns category if matched, None otherwise.
        
        Target: <10ms
        """
        # Check greetings first (fastest to match)
        for pattern in self._greeting_patterns:
            if pattern.match(query):
                return "greeting"
        
        # Check internet patterns
        for pattern in self._internet_patterns:
            if pattern.search(query):
                return "internet_data"
        
        # Check database patterns
        for pattern in self._database_patterns:
            if pattern.search(query):
                return "database"
        
        return None

    # ==================== TIER 2: KEYWORD + ENTITY MATCHING ====================

    def _tier2_keyword_entity_match(self, query: str) -> Optional[str]:
        """
        Tier 2: Keyword and database entity matching.
        Returns category if matched, None otherwise.
        
        Target: <50ms
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Check for database keywords
        if query_words & self._database_keywords:
            return "database"
        
        # Check for internet keywords
        for keyword in self._internet_keywords:
            if keyword in query_lower:
                return "internet_data"
        
        # Check if query mentions a database entity
        portfolios = self._get_portfolios_set()
        groups = self._get_groups_set()
        symbols = self._get_symbols_set()
        
        # Check for portfolio names in query
        for portfolio in portfolios:
            if portfolio in query_lower:
                return "database"
        
        # Check for group names
        for group in groups:
            if group in query_lower:
                return "database"
        
        # Check for symbols (exact word match)
        for word in query_words:
            if word in symbols:
                # Symbol found, but is it asking for internet data about it?
                internet_indicators = ['price', 'news', 'current', 'live', 'today']
                if any(ind in query_lower for ind in internet_indicators):
                    # Check if it's about holdings or external price
                    holdings_indicators = ['my', 'our', 'holding', 'position', 'portfolio']
                    if any(ind in query_lower for ind in holdings_indicators):
                        return "database"
                    return "internet_data"
                return "database"
        
        return None

    # ==================== TIER 3: LLM FALLBACK ====================

    def _get_llm(self) -> Ollama:
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.1,
                num_predict=10,  # We only need one word response
            )
        return self._llm

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _tier3_llm_classify(self, query: str) -> str:
        """
        Tier 3: LLM classification (fallback).
        Uses a minimal prompt for speed.
        
        Target: <3s
        """
        # Check cache first
        if self.enable_cache:
            cache_key = self._get_cache_key(query)
            if cache_key in self._llm_response_cache:
                print(f"  → [CACHE HIT] Using cached classification")
                return self._llm_response_cache[cache_key]
        
        # Minimal prompt for speed
        prompt = f"""Classify this query into ONE category: database, internet_data, or greeting.

- database: Questions about user's portfolio, holdings, positions, local data
- internet_data: Questions needing real-time market data, news, prices
- greeting: Hello, hi, thanks, chitchat

Query: {query}

Category:"""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            category = response.strip().lower()
            
            # Parse response
            if "database" in category:
                result = "database"
            elif "internet" in category:
                result = "internet_data"
            elif "greeting" in category:
                result = "greeting"
            else:
                result = "database"  # Safe default
            
            # Cache result
            if self.enable_cache:
                self._llm_response_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"  → LLM error: {e}")
            return "database"

    # ==================== MAIN CLASSIFICATION ====================

    def classify_query(self, query: str) -> str:
        """
        Classify query using tiered approach.
        Now includes comparison detection.
        """
        start_time = time.time()
        
        # NEW: Check for comparison FIRST (before other tiers)
        if self._is_comparison_query(query):
            elapsed = (time.time() - start_time) * 1000
            print(f"  → [COMPARISON] Detected comparison query ({elapsed:.1f}ms)")
            return "comparison"
        
        # Tier 1: Pattern matching
        result = self._tier1_pattern_match(query)
        if result:
            elapsed = (time.time() - start_time) * 1000
            print(f"  → [TIER 1] Pattern match: {result} ({elapsed:.1f}ms)")
            return result
        
        # Tier 2: Keyword + entity matching
        result = self._tier2_keyword_entity_match(query)
        if result:
            elapsed = (time.time() - start_time) * 1000
            print(f"  → [TIER 2] Keyword/entity match: {result} ({elapsed:.1f}ms)")
            return result
        
        # Tier 3: LLM fallback
        print(f"  → [TIER 3] Using LLM fallback...")
        result = self._tier3_llm_classify(query)
        elapsed = (time.time() - start_time) * 1000
        print(f"  → [TIER 3] LLM classification: {result} ({elapsed:.1f}ms)")
        return result

    def clear_cache(self):
        """Clear all caches."""
        self._portfolios_cache = None
        self._groups_cache = None
        self._symbols_cache = None
        self._llm_response_cache = {}
        print("  → All caches cleared")

    def get_stats(self) -> Dict:
        """Get router statistics."""
        return {
            "portfolios_cached": len(self._portfolios_cache) if self._portfolios_cache else 0,
            "groups_cached": len(self._groups_cache) if self._groups_cache else 0,
            "symbols_cached": len(self._symbols_cache) if self._symbols_cache else 0,
            "llm_cache_size": len(self._llm_response_cache),
        }