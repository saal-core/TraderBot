# src/services/gpt_oss_query_router.py
"""
GPT-OSS Query Router - Uses gpt-oss model via Ollama for LLM-only query classification.
Unlike the pattern-matching QueryRouter, this relies purely on LLM reasoning.
"""
from typing import Dict, List, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config
import time


class GPTOSSQueryRouter:


    def __init__(self, model_name: str = None, sql_executor=None):
        """
        Initialize the LLM query router.

        Args:
            model_name: Name of the Ollama model to use
            sql_executor: SQL executor for fetching database entities
        """
        self.sql_executor = sql_executor
        self._portfolios_cache = None
        self._groups_cache = None
        self._symbols_cache = None

        # Initialize Ollama
        ollama_config = get_ollama_config()
        self.model_name = model_name or ollama_config["model_name"]
        self.base_url = ollama_config["base_url"]
        self.temperature = ollama_config.get("temperature_routing", 0.1)

        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature
        )

        # Main classification prompt with improved disambiguation
        self.classification_prompt = PromptTemplate(
            input_variables=["query", "portfolios", "groups", "symbols_sample"],
            template="""You are an expert query classifier for a financial portfolio management system.

## YOUR TASK
Classify the user's query into exactly ONE of these categories:
- **database**: Query is about LOCAL portfolio data stored in our system
- **internet_data**: Query requires EXTERNAL real-time data from the internet
- **greeting**: Query is a greeting, chitchat, or general conversation

## DATABASE ENTITIES (Local Data)
These entities exist in our LOCAL database:

**Portfolios:** {portfolios}
**Groups:** {groups}
**Sample Symbols in DB:** {symbols_sample}

## CLASSIFICATION RULES

### Route to "database" if:
1. Query mentions ANY portfolio name, group name, or symbol from our database
2. Query asks about: holdings, positions, lots, quantities, allocations
3. Query asks about: portfolio performance, returns, profit/loss, YTD, MTD, QTD
4. Query asks about: portfolio descriptions, default index, benchmark, cost model, inception date
5. Query uses possessive words: "my portfolio", "our holdings", "my stocks"
6. Query asks to list/show portfolios, holdings, or positions
7. Query asks about comparisons BETWEEN portfolios in our database
8. Query asks about unrealized/realized gains within portfolios

### ⚠️ CRITICAL: "Top/Bottom/Best/Worst Stocks" DISAMBIGUATION

**Route to "database" if asking about stocks WITHIN the user's portfolio:**
- "Top 10 profit stocks" → database (ranking stocks in portfolio by profit)
- "Bottom 10 stocks" → database (worst performing in portfolio)
- "Best performing stocks" → database (user's holdings performance)
- "Which stocks have the highest profit?" → database (portfolio analysis)
- "List the top 10 most profitable stocks overall" → database (portfolio stocks)
- "What investment has the best performance?" → database (portfolio investment)
- "Top 5 stocks by return" → database (user's stocks ranked)
- "Worst stocks in my portfolio" → database
- "Best investment in terms of YTD profit" → database

**Route to "internet_data" ONLY if explicitly asking about MARKET-WIDE movers:**
- "Top gainers in NASDAQ this week" → internet_data (market-wide, exchange-specific)
- "Top gainers in NYSE today" → internet_data (market-wide, exchange-specific)
- "Market's biggest losers today" → internet_data (market-wide)
- "Trending stocks in the market" → internet_data (market-wide)
- "Most active stocks on the exchange" → internet_data (market-wide)

**Key Distinction:**
- NO exchange/market mentioned + asking about profits/performance → "database" (user's portfolio)
- Exchange/market mentioned (NASDAQ, NYSE, S&P, market) + today/this week → "internet_data"

### Route to "internet_data" if:
1. Query asks for CURRENT/LIVE market prices (not about holdings)
2. Query asks for latest NEWS about companies or markets
3. Query explicitly mentions market indices: "NASDAQ gainers", "NYSE losers", "S&P movers"
4. Query asks about market performance with time reference: "market today", "this week's gainers"
5. Query asks about cryptocurrency prices: Bitcoin, Ethereum, etc.
6. Query asks about commodity prices: oil, gold, silver
7. Query asks about forex/exchange rates
8. Query is a HYPOTHETICAL investment calculation: "If I had invested X in Y..."

### Route to "greeting" if:
1. Query is a greeting: hi, hello, hey, good morning
2. Query is chitchat: how are you, what's up
3. Query asks about the bot's identity or capabilities
4. Query is a thank you or goodbye

## EXAMPLES FOR DISAMBIGUATION

| Query | Category | Reason |
|-------|----------|--------|
| "Top 10 profit stocks" | database | Ranking portfolio stocks by profit |
| "Bottom 10 stocks" | database | Worst performing portfolio stocks |
| "Best performing stocks" | database | User's holdings performance |
| "What investment has the best YTD profit?" | database | Portfolio investment analysis |
| "List the top 10 most profitable stocks overall" | database | Portfolio profit ranking |
| "Which stock has highest return?" | database | Portfolio analysis |
| "Top gainers in Nasdaq this week" | internet_data | Market-wide, exchange-specific |
| "NYSE biggest losers today" | internet_data | Market-wide, exchange-specific |
| "Market movers today" | internet_data | External market data |
| "Trending stocks on the market" | internet_data | Market-wide trending |

## OUTPUT FORMAT
Respond with ONLY the category name. No explanation, no punctuation, no extra text.
Valid responses: database, internet_data, greeting

## USER QUERY
{query}

## CATEGORY:"""
        )

        self.classification_chain = self.classification_prompt | self.llm | StrOutputParser()

    # ==================== DATABASE ENTITY FETCHING ====================

    def _get_portfolios(self) -> List[str]:
        """Fetch portfolio names from database with caching."""
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
                self._portfolios_cache = df['portfolio_name'].tolist()
                return self._portfolios_cache
            return []
        except Exception as e:
            print(f"Error fetching portfolios: {e}")
            return []

    def _get_groups(self) -> List[str]:
        """Fetch group names from database with caching."""
        if self._groups_cache is not None:
            return self._groups_cache

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
                self._groups_cache = df['group_name'].tolist()
                return self._groups_cache
            return []
        except Exception as e:
            print(f"Error fetching groups: {e}")
            return []

    def _get_symbols(self) -> List[str]:
        """Fetch stock symbols from database with caching."""
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
            LIMIT 100
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                self._symbols_cache = df['symbol'].tolist()
                return self._symbols_cache
            return []
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []

    def _format_entity_list(self, entities: List[str], max_items: int = 30) -> str:
        """Format entity list for prompt, truncating if necessary."""
        if not entities:
            return "None"
        
        if len(entities) <= max_items:
            return ", ".join(entities)
        else:
            return ", ".join(entities[:max_items]) + f"... (+{len(entities) - max_items} more)"

    # ==================== PRE-CLASSIFICATION RULES ====================

    def _apply_pre_classification_rules(self, query: str) -> Optional[str]:
        """
        Apply deterministic rules before LLM classification.
        Returns category if rule matches, None otherwise.
        
        This handles edge cases that the LLM might get wrong.
        """
        query_lower = query.lower()

        # ============================================================
        # RULE 1: Portfolio stock rankings (NOT market movers)
        # These ask about user's portfolio performance, not market-wide
        # ============================================================
        portfolio_ranking_patterns = [
            # Profit/loss rankings
            r'^top\s+\d*\s*(profit|profitable|performing|gaining)',
            r'^bottom\s+\d*\s*(stock|loss|losing|performing)',
            r'^best\s+(performing|profit|investment|stock)',
            r'^worst\s+(performing|profit|investment|stock)',
            r'(top|bottom|best|worst)\s+\d+\s+stocks?$',
            r'(top|bottom|best|worst)\s+\d+\s+(profit|loss)\s+stocks?',
            r'(most|least)\s+profitable\s+stocks?',
            r'highest\s+(profit|return|gain)',
            r'lowest\s+(profit|return|loss)',
            r'(best|worst)\s+performance.*ytd',
            r'ytd\s+(profit|return).*best',
            r'what\s+investment.*best\s+performance',
            r'which\s+(stock|investment).*highest\s+(profit|return)',
            r'list.*top\s+\d+.*profitable',
            r'list.*bottom\s+\d+',
        ]
        
        import re
        for pattern in portfolio_ranking_patterns:
            if re.search(pattern, query_lower):
                # Make sure it's NOT asking about a specific exchange/market
                market_indicators = ['nasdaq', 'nyse', 's&p', 'dow', 'market', 'exchange', 'this week', 'today', 'this month']
                if not any(indicator in query_lower for indicator in market_indicators):
                    print(f"  → [PRE-RULE] Portfolio ranking pattern detected: {pattern}")
                    return "database"

        # ============================================================
        # RULE 2: Market-wide movers (requires exchange/market mention)
        # ============================================================
        market_mover_patterns = [
            r'(top|biggest)\s+(gainers?|losers?|movers?)\s+(in|on)\s+(nasdaq|nyse|the\s+market)',
            r'(nasdaq|nyse|s&p|dow|market)\s+(gainers?|losers?|movers?)',
            r'(gainers?|losers?|movers?)\s+(in|on)\s+(nasdaq|nyse)',
            r'(trending|active)\s+stocks?\s+(on|in)\s+(the\s+)?(market|nasdaq|nyse)',
            r'market\s+movers?\s+(today|this\s+week)',
        ]
        
        for pattern in market_mover_patterns:
            if re.search(pattern, query_lower):
                print(f"  → [PRE-RULE] Market mover pattern detected: {pattern}")
                return "internet_data"

        # ============================================================
        # RULE 3: Hypothetical investments
        # ============================================================
        hypothetical_patterns = [
            r'if\s+i\s+had\s+invested',
            r'had\s+i\s+invested',
            r'what\s+would.*be\s+worth',
            r'how\s+much\s+would.*worth',
        ]
        
        for pattern in hypothetical_patterns:
            if re.search(pattern, query_lower):
                print(f"  → [PRE-RULE] Hypothetical investment pattern detected")
                return "internet_data"

        # ============================================================
        # RULE 4: Greetings
        # ============================================================
        greeting_patterns = [
            r'^(hi|hello|hey|greetings)[\s!.,]*$',
            r'^good\s+(morning|afternoon|evening)',
            r'^how\s+are\s+you',
            r'^thank(s|\s+you)',
            r'^(bye|goodbye)',
        ]
        
        for pattern in greeting_patterns:
            if re.search(pattern, query_lower):
                print(f"  → [PRE-RULE] Greeting pattern detected")
                return "greeting"

        return None  # No pre-rule matched, use LLM

    # ==================== CLASSIFICATION ====================

    def classify_query(self, query: str) -> str:
        """
        Classify a user query using pre-rules and LLM.

        Args:
            query: User's input query

        Returns:
            Category string: "database", "internet_data", or "greeting"
        """
        try:
            start_time = time.time()

            # Step 1: Apply pre-classification rules
            pre_result = self._apply_pre_classification_rules(query)
            if pre_result:
                elapsed = time.time() - start_time
                print(f"  → Classification (pre-rule): {pre_result} in {elapsed:.3f}s")
                return pre_result

            # Step 2: Fetch database entities for context
            portfolios = self._get_portfolios()
            groups = self._get_groups()
            symbols = self._get_symbols()

            # Format for prompt
            portfolios_str = self._format_entity_list(portfolios, max_items=30)
            groups_str = self._format_entity_list(groups, max_items=15)
            symbols_str = self._format_entity_list(symbols, max_items=20)

            # Step 3: Call LLM
            response = self.classification_chain.invoke({
                "query": query,
                "portfolios": portfolios_str,
                "groups": groups_str,
                "symbols_sample": symbols_str
            })

            elapsed = time.time() - start_time
            print(f"  → Classification (LLM): completed in {elapsed:.2f}s")

            # Parse response
            category = response.strip().lower()

            # Normalize response
            if "database" in category:
                return "database"
            elif "internet" in category:
                return "internet_data"
            elif "greeting" in category:
                return "greeting"
            else:
                print(f"  → Warning: Unexpected LLM response '{response}', defaulting to 'database'")
                return "database"

        except Exception as e:
            print(f"  → Error in LLM classification: {e}")
            return "database"  # Safe fallback

    def clear_cache(self):
        """Clear all caches."""
        self._portfolios_cache = None
        self._groups_cache = None
        self._symbols_cache = None 