"""
Portfolio Alias Resolver Service

Handles fuzzy matching of user portfolio/account references to actual database values,
creating placeholders for privacy when sending queries to external LLMs (OpenAI).

This service uses:
1. Local Ollama LLM for semantic entity extraction
2. RapidFuzz for typo-tolerant matching
3. Placeholder substitution for privacy
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from rapidfuzz import fuzz, process
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config


@dataclass
class EntityResolution:
    """Result of entity resolution containing placeholders and mappings."""
    rewritten_query: str
    placeholder_map: Dict[str, str] = field(default_factory=dict)
    extracted_portfolios: List[str] = field(default_factory=list)
    extracted_accounts: List[str] = field(default_factory=list)
    placeholder_info: str = ""  # Human-readable placeholder list for prompt


class PortfolioAliasResolver:
    """
    Resolves fuzzy user references to actual portfolio names and account IDs.
    
    Uses a two-phase approach:
    1. Extract entity mentions from user query using local Ollama
    2. Fuzzy match to actual database values using RapidFuzz + semantic matching
    
    Returns placeholder mappings to hide actual values from external LLMs.
    """
    
    def __init__(self, sql_executor=None):
        """
        Initialize the alias resolver.
        
        Args:
            sql_executor: PostgreSQLExecutor instance for fetching portfolio/account data
        """
        self.sql_executor = sql_executor
        self._portfolio_cache: Optional[List[str]] = None
        self._account_cache: Optional[List[str]] = None
        
        # Initialize local Ollama for entity extraction
        ollama_config = get_ollama_config()
        self.llm = Ollama(
            model=ollama_config["model_name"],
            base_url=ollama_config["base_url"],
            temperature=0.1  # Low temperature for consistent extraction
        )
        
        # Entity extraction prompt
        self.extract_prompt = PromptTemplate(
            input_variables=["query", "portfolio_names", "account_ids"],
            template="""You are an entity extraction assistant. Given a user question about portfolios and accounts, 
identify which portfolio names or account IDs the user is referring to.

AVAILABLE PORTFOLIO NAMES:
{portfolio_names}

AVAILABLE ACCOUNT IDS:
{account_ids}

USER QUESTION: {query}

INSTRUCTIONS:
1. Look for any reference to portfolios or accounts in the question
2. Match fuzzy references like "balanced fund" to the closest portfolio name like "A-Balanced"
3. Handle variations like "ABalanced", "A Balanced", "balanced portfolio" -> "A-Balanced"
4. Handle partial matches like "growth" -> "A-Growth" (if that exists)
5. If user says "all portfolios" or doesn't mention a specific one, return NONE

OUTPUT FORMAT (one per line, exact format required):
PORTFOLIO: [matched_portfolio_name] (user said: "[original_reference]")
ACCOUNT: [matched_account_id] (user said: "[original_reference]")

If no portfolios/accounts are mentioned, respond with:
NONE

Examples:
- User: "show my balanced fund" -> PORTFOLIO: A-Balanced (user said: "balanced fund")
- User: "what is ACC123 performance" -> ACCOUNT: ACC-123 (user said: "ACC123")
- User: "show all portfolios" -> NONE
- User: "compare growth and balanced" -> PORTFOLIO: A-Growth (user said: "growth")
                                         PORTFOLIO: A-Balanced (user said: "balanced")

Your response:"""
        )
        
        self.extract_chain = self.extract_prompt | self.llm | StrOutputParser()
    
    def _fetch_portfolio_names(self) -> List[str]:
        """Fetch distinct portfolio names from database (cached)."""
        if self._portfolio_cache is not None:
            return self._portfolio_cache
        
        if not self.sql_executor:
            return []
        
        try:
            query = """
            SELECT DISTINCT portfolio_name
            FROM ai_trading.portfolio_summary
            ORDER BY portfolio_name
            LIMIT 100
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                self._portfolio_cache = df['portfolio_name'].tolist()
                logging.info(f"Cached {len(self._portfolio_cache)} portfolio names")
                return self._portfolio_cache
            return []
        except Exception as e:
            logging.error(f"Error fetching portfolio names: {e}")
            return []
    
    def _fetch_account_ids(self) -> List[str]:
        """Fetch distinct account IDs from database (cached)."""
        if self._account_cache is not None:
            return self._account_cache
        
        if not self.sql_executor:
            return []
        
        try:
            query = """
            SELECT DISTINCT account_id
            FROM ai_trading.portfolio_summary
            ORDER BY account_id
            LIMIT 200
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                self._account_cache = [str(aid) for aid in df['account_id'].tolist()]
                logging.info(f"Cached {len(self._account_cache)} account IDs")
                return self._account_cache
            return []
        except Exception as e:
            logging.error(f"Error fetching account IDs: {e}")
            return []
    
    def _fuzzy_match_portfolio(self, user_reference: str, portfolio_names: List[str]) -> Optional[str]:
        """
        Fuzzy match a user reference to actual portfolio name.
        
        Args:
            user_reference: What the user said (e.g., "balanced fund")
            portfolio_names: List of actual portfolio names
            
        Returns:
            Best matching portfolio name or None
        """
        if not portfolio_names or not user_reference:
            return None
        
        # Normalize the reference
        normalized = user_reference.lower().strip()
        
        # Try exact match first (case-insensitive)
        for name in portfolio_names:
            if name.lower() == normalized:
                return name
        
        # Try fuzzy matching with rapidfuzz
        # Use multiple scoring methods for better results
        
        # Method 1: Token set ratio (handles word order differences)
        result = process.extractOne(
            normalized,
            portfolio_names,
            scorer=fuzz.token_set_ratio,
            score_cutoff=60
        )
        
        if result:
            return result[0]
        
        # Method 2: Partial ratio (handles substrings)
        result = process.extractOne(
            normalized,
            portfolio_names,
            scorer=fuzz.partial_ratio,
            score_cutoff=70
        )
        
        if result:
            return result[0]
        
        # Method 3: Try matching against lowercase versions
        lowercase_map = {name.lower().replace("-", ""): name for name in portfolio_names}
        normalized_no_dash = normalized.replace("-", "").replace(" ", "")
        
        if normalized_no_dash in lowercase_map:
            return lowercase_map[normalized_no_dash]
        
        # Method 4: Check if reference is contained in any portfolio name
        for name in portfolio_names:
            name_normalized = name.lower().replace("-", "")
            if normalized_no_dash in name_normalized or name_normalized in normalized_no_dash:
                return name
        
        return None
    
    def _fuzzy_match_account(self, user_reference: str, account_ids: List[str]) -> Optional[str]:
        """
        Fuzzy match a user reference to actual account ID.
        
        Args:
            user_reference: What the user said (e.g., "ACC123")
            account_ids: List of actual account IDs
            
        Returns:
            Best matching account ID or None
        """
        if not account_ids or not user_reference:
            return None
        
        normalized = user_reference.upper().strip()
        
        # Try exact match first
        for aid in account_ids:
            if str(aid).upper() == normalized:
                return aid
        
        # Try fuzzy matching
        result = process.extractOne(
            normalized,
            account_ids,
            scorer=fuzz.ratio,
            score_cutoff=70
        )
        
        if result:
            return result[0]
        
        # Try without dashes/spaces
        normalized_clean = re.sub(r'[\s\-_]', '', normalized)
        for aid in account_ids:
            aid_clean = re.sub(r'[\s\-_]', '', str(aid).upper())
            if normalized_clean == aid_clean:
                return aid
        
        return None
    
    def _parse_extraction_result(
        self, 
        result: str, 
        portfolio_names: List[str], 
        account_ids: List[str]
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Parse the LLM extraction result and validate against actual data.
        
        Returns:
            Tuple of (portfolio_matches, account_matches)
            Each match is (actual_value, user_reference)
        """
        portfolio_matches = []
        account_matches = []
        
        if "NONE" in result.upper():
            return portfolio_matches, account_matches
        
        for line in result.strip().split('\n'):
            line = line.strip()
            
            # Parse PORTFOLIO lines
            if line.startswith("PORTFOLIO:"):
                match = re.match(r'PORTFOLIO:\s*([^\(]+)\s*\(user said:\s*"([^"]+)"\)', line)
                if match:
                    extracted_name = match.group(1).strip()
                    user_said = match.group(2).strip()
                    
                    # Validate against actual portfolio names
                    if extracted_name in portfolio_names:
                        portfolio_matches.append((extracted_name, user_said))
                    else:
                        # Try fuzzy match as fallback
                        fuzzy_match = self._fuzzy_match_portfolio(extracted_name, portfolio_names)
                        if fuzzy_match:
                            portfolio_matches.append((fuzzy_match, user_said))
            
            # Parse ACCOUNT lines
            elif line.startswith("ACCOUNT:"):
                match = re.match(r'ACCOUNT:\s*([^\(]+)\s*\(user said:\s*"([^"]+)"\)', line)
                if match:
                    extracted_id = match.group(1).strip()
                    user_said = match.group(2).strip()
                    
                    # Validate against actual account IDs
                    if extracted_id in account_ids:
                        account_matches.append((extracted_id, user_said))
                    else:
                        # Try fuzzy match as fallback
                        fuzzy_match = self._fuzzy_match_account(extracted_id, account_ids)
                        if fuzzy_match:
                            account_matches.append((fuzzy_match, user_said))
        
        return portfolio_matches, account_matches
    
    def resolve_entities(self, query: str) -> EntityResolution:
        """
        Resolve portfolio and account references in a user query.
        
        Args:
            query: User's natural language question
            
        Returns:
            EntityResolution with rewritten query and placeholder mappings
        """
        result = EntityResolution(rewritten_query=query)
        
        # Fetch available entities
        portfolio_names = self._fetch_portfolio_names()
        account_ids = self._fetch_account_ids()
        
        if not portfolio_names and not account_ids:
            logging.warning("No portfolio names or account IDs available for resolution")
            return result
        
        try:
            # Extract entities using local LLM
            print("  → Extracting portfolio/account references...")
            extraction_result = self.extract_chain.invoke({
                "query": query,
                "portfolio_names": ", ".join(portfolio_names) if portfolio_names else "N/A",
                "account_ids": ", ".join(account_ids) if account_ids else "N/A"
            })
            
            print(f"  → Extraction result: {extraction_result.strip()}")
            
            # Parse and validate extraction
            portfolio_matches, account_matches = self._parse_extraction_result(
                extraction_result, portfolio_names, account_ids
            )
            
            # Build placeholder map and rewrite query
            placeholder_map = {}
            rewritten_query = query
            placeholder_parts = []
            
            # Handle portfolio matches
            for i, (actual_name, user_reference) in enumerate(portfolio_matches, 1):
                placeholder = f":PORTFOLIO_{i}"
                placeholder_map[placeholder] = actual_name
                result.extracted_portfolios.append(user_reference)
                
                # Replace user reference with placeholder in query
                # Use case-insensitive replacement
                pattern = re.compile(re.escape(user_reference), re.IGNORECASE)
                rewritten_query = pattern.sub(placeholder, rewritten_query)
                
                placeholder_parts.append(f"{placeholder} = '{actual_name}'")
                print(f"  → Matched: '{user_reference}' → '{actual_name}' → {placeholder}")
            
            # Handle account matches
            for i, (actual_id, user_reference) in enumerate(account_matches, 1):
                placeholder = f":ACCOUNT_{i}"
                placeholder_map[placeholder] = actual_id
                result.extracted_accounts.append(user_reference)
                
                # Replace user reference with placeholder in query
                pattern = re.compile(re.escape(user_reference), re.IGNORECASE)
                rewritten_query = pattern.sub(placeholder, rewritten_query)
                
                placeholder_parts.append(f"{placeholder} = '{actual_id}'")
                print(f"  → Matched: '{user_reference}' → '{actual_id}' → {placeholder}")
            
            result.rewritten_query = rewritten_query
            result.placeholder_map = placeholder_map
            result.placeholder_info = "\n".join(placeholder_parts) if placeholder_parts else "No specific portfolio or account mentioned"
            
            if placeholder_map:
                print(f"  → Final placeholder map: {placeholder_map}")
            
        except Exception as e:
            logging.error(f"Error in entity resolution: {e}")
            # Return original query on error
            result.rewritten_query = query
        
        return result
    
    def substitute_placeholders(self, sql: str, placeholder_map: Dict[str, str]) -> str:
        """
        Substitute placeholders in SQL with actual values.
        
        Args:
            sql: SQL query with placeholders (e.g., :PORTFOLIO_1)
            placeholder_map: Mapping of placeholders to actual values
            
        Returns:
            SQL with actual values substituted
        """
        if not placeholder_map:
            return sql
        
        result_sql = sql
        for placeholder, actual_value in placeholder_map.items():
            # Handle both :PLACEHOLDER and ':PLACEHOLDER' formats
            # Replace :PORTFOLIO_1 with 'actual_value' (with quotes for string)
            result_sql = result_sql.replace(placeholder, f"'{actual_value}'")
        
        print(f"  → SQL after placeholder substitution: {result_sql}")
        return result_sql
    
    def get_portfolio_count(self) -> int:
        """Get the count of available portfolios."""
        return len(self._fetch_portfolio_names())
    
    def get_account_count(self) -> int:
        """Get the count of available accounts."""
        return len(self._fetch_account_ids())
    
    def clear_cache(self):
        """Clear the cached portfolio and account data."""
        self._portfolio_cache = None
        self._account_cache = None
        logging.info("Alias resolver cache cleared")
