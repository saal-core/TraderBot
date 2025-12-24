
from typing import Dict, List, Tuple, Optional
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config, get_openai_config, get_qwen_config
from src.config.prompts import (
    DATABASE_EXPLANATION_PROMPT,
    STOCK_EXTRACTION_PROMPT,
    SYMBOL_MATCHING_PROMPT
)
from src.services.chat_memory import ChatMemory
from src.services.portfolio_alias_resolver import PortfolioAliasResolver
import os
import time
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
load_dotenv()

# TOON formatter for token-efficient data formatting
from src.utils.toon_formatter import format_query_results, format_symbol_list

class DatabaseQueryHandler:
    """Handles database-related queries by generating SQL using custom prompt"""

    def __init__(self, model_name: str = None, use_openai: bool = True, sql_executor=None, memory_max_pairs: int = 5):
        """
        Initialize the database query handler

        Args:
            model_name: Name of the model to use (defaults to config)
            use_openai: Whether to use OpenAI for SQL generation (True) or Ollama (False)
            sql_executor: Optional SQL executor for fetching dynamic data
            memory_max_pairs: Maximum number of Q&A pairs to remember for follow-up questions (default: 3)
        """
        self.sql_executor = sql_executor
        self._symbols_cache = None
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

        # Initialize SQL generation model (OpenAI or Ollama)
        if use_openai:
            openai_config = get_openai_config()
            self.model_name = model_name or openai_config["model_name"]
            self.temperature = openai_config["temperature_sql"]

            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=openai_config["api_key"]
            )
        else:
            ollama_config = get_ollama_config()
            self.model_name = model_name or ollama_config["model_name"]
            self.base_url = ollama_config["base_url"]
            self.temperature = ollama_config["temperature_sql"]

            self.llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature
            )

        # Use QWEN (H100) for explanations - faster than local Ollama
        qwen_config = get_qwen_config()
        self.explanation_llm = ChatOpenAI(
            model="Qwen3-30B-A3B",
            base_url=qwen_config["base_url"],
            api_key=qwen_config["api_key"],
            temperature=0.3,
            max_retries=2
        )

        # Initialize alias resolver for privacy-preserving placeholder generation
        self.alias_resolver = PortfolioAliasResolver(sql_executor=sql_executor)

        # Load custom prompt template
        self.custom_prompt_template = self._load_custom_prompt()

        # Create prompt template with dynamic data including conversation history
        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "entity_placeholders", "query", "matched_symbols", "conversation_history"],
            template=self.custom_prompt_template
        )

        self.sql_chain = self.sql_prompt | self.llm | StrOutputParser()

    def _load_custom_prompt(self) -> str:
        """Load custom prompt from test2sql_prompt.md"""
        prompt_file = "/home/dev/Hussein/trader_bot_enhance/test2sql_prompt.md"

        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
                print("="*20 +"\nLoaded custom SQL prompt template."+"\n"+"="*20)
        else:
            # Fallback to basic prompt if file not found
            prompt_content = """You are a PostgreSQL SQL expert. Generate a SQL query based on the user's question and database schema."""

        # Add the dynamic sections and output instructions
        full_template = f"""{prompt_content}

---

### **4. Conversation History (for follow-up questions)**

{{conversation_history}}

**Note:** If the current question refers to previous context (e.g., "show more", "what about top 10", "give me details"),
use the conversation history to understand the full intent. Otherwise, treat as a standalone question.

---

### **5. Database Schema**

{{schema}}

---

### **6. Entity Placeholders (Privacy-Preserving)**

**IMPORTANT**: The user's query has been preprocessed. Portfolio names and account IDs are replaced with placeholders.
Use these placeholders EXACTLY as shown in your SQL WHERE clauses.

**Resolved Entities:**
{{entity_placeholders}}

**Matched Stock Symbols (if any):**
{{matched_symbols}}

**PLACEHOLDER USAGE RULES:**
- Use placeholders directly in WHERE clauses: `WHERE portfolio_name = :PORTFOLIO_1`
- Do NOT wrap placeholders in quotes: Use `:PORTFOLIO_1` not `':PORTFOLIO_1'`
- If no placeholder is provided, the query is about all portfolios/accounts
- Placeholders will be substituted with actual values after SQL generation

---

### **7. Output Instructions**

- Use the placeholders provided above when filtering by portfolio or account
- Generate ONLY a SELECT query
- Do not use INSERT, UPDATE, DELETE, or any data modification statements
- Return only the SQL query without any explanation or markdown
- Use proper PostgreSQL syntax
- Always use schema prefix: `ai_trading.table_name`
- For "current" or "latest" data, filter by: `WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.table_name)`
- Pay attention to SQL query syntax and correctness
- Take care of comparisons between portfolio names and stocks or group name or account ids
---

### **8. Comparison Guidelines**:
### **Strategic Logic: Portfolio vs. Stock Comparison**

**The Core Problem:**

  * **Portfolios** (in `portfolio_summary`) have explicit **Percentage Return** metrics (e.g., `ytd_return`).
  * **Stocks** (in `portfolio_holdings`) **DO NOT** have a pre-calculated percentage return column. They only have **Dollar Value** and **Dollar Profit** (`ytd_total_pnl`).

**The Rule:**
**"Compare Apples to Apples (Dollars to Dollars)."**
Since you cannot safely calculate the stock's percentage return (due to buy/sell timing affecting the cost basis), you must compare them based on **Total Profit (PnL)** or **Market Value**.

-----

### **Step-by-Step Execution Plan**

1.  **Identify the Entities:**

      * **Entity A (Portfolio):** "A-Balanced" $\rightarrow$ Source: `ai_trading.portfolio_summary`
      * **Entity B (Stock):** "QQQ" $\rightarrow$ Source: `ai_trading.portfolio_holdings_realized_pnl` (Use this table because it captures Total PnL = Realized + Unrealized).

2.  **Select the Common Metric:**

      * **Metric:** **Year-to-Date Profit (YTD PnL)**.
      * *Portfolio Mapping:* `ytd_profit`
      * *Stock Mapping:* `ytd_total_pnl` (This is the equivalent of `ytd_profit` for a specific stock).

3.  **Construct the Query:**

      * Use a **Common Table Expression (CTE)** for each entity to ensure you get the `MAX(datetime)` for both independently.
      * Join them on `1=1` (Cross Join) since they are unrelated entities, just to display them in one row.

-----

### **User Question:**

{{query}}

### **SQL Query:**"""

        return full_template

    def _format_conversation_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Format conversation history for SQL generation context

        Args:
            chat_history: List of previous messages

        Returns:
            Formatted conversation history string
        """
        if not chat_history:
            return "No previous conversation (this is the first question)."

        # Get last 3 Q&A pairs using chat memory
        recent_messages = self.chat_memory.get_context_messages(chat_history)

        if not recent_messages:
            return "No previous conversation (this is the first question)."

        # Format as Q&A pairs for better context
        formatted_lines = []
        for msg in recent_messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")

            # Truncate very long responses to keep context manageable
            if len(content) > 200:
                content = content[:200] + "..."

            formatted_lines.append(f"{role}: {content}")

        return "\n".join(formatted_lines)

    def _get_portfolio_names(self) -> str:
        """Fetch distinct portfolio names from database"""
        if not self.sql_executor:
            return "N/A"


        query = """
        SELECT DISTINCT portfolio_name
        FROM ai_trading.portfolio_summary
        ORDER BY portfolio_name
        LIMIT 50
        """
        success, df, _ = self.sql_executor.execute_query(query)
        print(f"*************Portfolio names query success*****************: {success}")
        if success and df is not None and not df.empty:
            names = df['portfolio_name'].tolist()
            return ", ".join([f"'{name}'" for name in names])
        return "N/A"


    def _get_account_ids(self) -> str:
        """Fetch distinct account IDs from database"""
        if not self.sql_executor:
            return "N/A"

        try:
            query = """
            SELECT DISTINCT account_id
            FROM ai_trading.portfolio_summary
            ORDER BY account_id
            LIMIT 100
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                account_ids = df['account_id'].tolist()
                return ", ".join([f"'{aid}'" for aid in account_ids])
            return "N/A"
        except:
            return "N/A"

    def _get_all_symbols_dict(self) -> Dict[str, str]:
        """
        Fetch all symbols from database and return as dictionary
        Uses cache to avoid repeated queries

        Returns:
            Dictionary with id as key and symbol as value
        """
        if self._symbols_cache is not None:
            return self._symbols_cache

        if not self.sql_executor:
            return {}

        try:
            query = """
            SELECT DISTINCT symbol
            FROM ai_trading.portfolio_holdings
            WHERE symbol IS NOT NULL
            ORDER BY symbol
            """
            success, df, _ = self.sql_executor.execute_query(query)
            if success and df is not None and not df.empty:
                # Create dictionary with index as id
                symbols_dict = {str(i): symbol for i, symbol in enumerate(df['symbol'].tolist())}
                self._symbols_cache = symbols_dict
                return symbols_dict
            return {}
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return {}

    def _extract_stock_mentions(self, query: str) -> List[str]:
        """
        Extract potential stock mentions from query using LLM

        Args:
            query: User's question

        Returns:
            List of potential stock names/symbols mentioned
        """
        extraction_prompt = PromptTemplate(
            input_variables=["query"],
            template=STOCK_EXTRACTION_PROMPT
        )

        extraction_chain = extraction_prompt | self.llm | StrOutputParser()

        try:
            start_time = time.time()
            print(f"⏱️  Starting: Stock Mention Extraction...")

            result = extraction_chain.invoke({"query": query})

            elapsed = time.time() - start_time
            print(f"✅ Completed: Stock Mention Extraction in {elapsed:.2f}s")

            result = result.strip()

            if result.upper() == "NONE" or not result:
                return []

            # Split by comma and clean
            terms = [term.strip() for term in result.split(',') if term.strip()]
            return terms
        except Exception as e:
            print(f"❌ Error extracting stock mentions: {e}")
            return []

    def _match_symbol_from_list(self, extracted_term: str, all_symbols_list: List[str]) -> Optional[str]:
        """
        Use LLM to find the best matching symbol from the list for the extracted term.
        
        Args:
            extracted_term: The term extracted from user query (e.g. "Apple")
            all_symbols_list: List of all available symbols in database
            
        Returns:
            The matching symbol or None if no match found
        """
        if not extracted_term or not all_symbols_list:
            return None
            
        # Create a TOON formatted symbol list for token efficiency
        # TOON format reduces tokens by ~40% for large lists
        symbols_str = format_symbol_list(all_symbols_list)
        
        match_prompt = PromptTemplate(
            input_variables=["term", "symbols"],
            template=SYMBOL_MATCHING_PROMPT
        )
        
        match_chain = match_prompt | self.llm | StrOutputParser()

        try:
            start_time = time.time()
            print(f"⏱️  Starting: Symbol Matching for '{extracted_term}' (TOON format)...")

            # Using TOON format for token efficiency
            # We skip fuzzy pre-filtering because it misses cases like "Apple" -> "AAPL" (score ~22)

            result = match_chain.invoke({
                "term": extracted_term,
                "symbols": symbols_str
            })

            elapsed = time.time() - start_time
            print(f"✅ Completed: Symbol Matching in {elapsed:.2f}s")

            result = result.strip()
            if result == "NONE" or result not in all_symbols_list:
                return None

            return result

        except Exception as e:
            print(f"❌ Error matching symbol: {e}")
            return None


    def _fuzzy_match_symbols(self, mentioned_terms: List[str], symbols_dict: Dict[str, str], threshold: int = 70) -> str:
        """
        Match mentioned terms to actual symbols using LLM selection from list
        
        Args:
            mentioned_terms: List of terms extracted from query
            symbols_dict: Dictionary of all available symbols
            threshold: Unused in new logic but kept for signature compatibility
            
        Returns:
            String with matched symbols information
        """
        if not mentioned_terms or not symbols_dict:
            return "No stock symbols detected in the question"

        symbols_list = list(symbols_dict.values())
        matched_info = []

        for term in mentioned_terms:
            # Use new LLM matching logic
            matched_symbol = self._match_symbol_from_list(term, symbols_list)
            
            if matched_symbol:
                matched_info.append(f"'{term}' -> '{matched_symbol}'")
            else:
                # Fallback to simple fuzzy if LLM fails or returns nothing? 
                # Or just report no match. Let's report no match to avoid bad data.
                pass

        if matched_info:
            return "Detected stock mentions:\n" + "\n".join(matched_info)
        else:
            return f"No close matches found for: {', '.join(mentioned_terms)}"

    def generate_sql(self, query: str, schema: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Generate SQL query from natural language question with fuzzy stock matching and conversation context.
        
        Uses privacy-preserving placeholder approach:
        1. Resolve portfolio/account references to placeholders using local LLM
        2. Send placeholders (not actual names) to OpenAI for SQL generation
        3. Substitute actual values into SQL after generation

        Args:
            query: User's natural language question
            schema: Database schema information
            chat_history: Previous conversation history for follow-up questions

        Returns:
            Generated SQL query string with actual values substituted
        """
        if chat_history is None:
            chat_history = []

        # Phase 1: Resolve entities using local LLM (privacy-preserving)
        print("🔒 Phase 1: Resolving portfolio/account references locally...")
        entity_resolution = self.alias_resolver.resolve_entities(query)
        
        # Use rewritten query with placeholders
        rewritten_query = entity_resolution.rewritten_query
        placeholder_map = entity_resolution.placeholder_map
        entity_placeholders = entity_resolution.placeholder_info
        
        if placeholder_map:
            print(f"  → Placeholder map: {placeholder_map}")
            print(f"  → Rewritten query: {rewritten_query}")
        else:
            print("  → No specific portfolio/account mentioned")

        # Format conversation history for context
        conversation_text = self._format_conversation_history(chat_history)

        # Check for stock mentions and fuzzy match
        print("  → Checking for stock mentions...")
        mentioned_terms = self._extract_stock_mentions(query)

        matched_symbols = "N/A"
        if mentioned_terms:
            print(f"  → Found potential stock mentions: {mentioned_terms}")
            symbols_dict = self._get_all_symbols_dict()
            matched_symbols = self._fuzzy_match_symbols(mentioned_terms, symbols_dict)
            print(f"  → {matched_symbols}")

        # Phase 2: Generate SQL with placeholders (OpenAI - no actual names exposed)
        start_time = time.time()
        print(f"⏱️  Phase 2: SQL Query Generation (with placeholders)...")

        sql = self.sql_chain.invoke({
            "schema": schema,
            "entity_placeholders": entity_placeholders,
            "matched_symbols": matched_symbols,
            "conversation_history": conversation_text,
            "query": rewritten_query  # Use rewritten query with placeholders
        })

        elapsed = time.time() - start_time
        print(f"✅ Completed: SQL Query Generation in {elapsed:.2f}s")

        # Clean up the response
        sql = sql.strip()

        # Remove markdown code blocks if present
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join([line for line in lines if not line.startswith("```") and "sql" not in line.lower()])

        sql = sql.strip()
        print(f"  → Generated SQL (with placeholders): {sql}")

        # Phase 3: Substitute placeholders with actual values
        if placeholder_map:
            print(f"🔓 Phase 3: Substituting placeholders with actual values...")
            sql = self.alias_resolver.substitute_placeholders(sql, placeholder_map)
            print(f"  → Final SQL: {sql}")

        return sql

    def explain_results(self, query: str, results_df, sql_query: str) -> str:
        """
        Generate a natural language explanation of query results.
        Uses QWEN (H100) for faster explanations.

        Args:
            query: Original user question
            results_df: Pandas DataFrame with results
            sql_query: The SQL query that was executed

        Returns:
            Natural language explanation
        """
        # Use TOON format for token-efficient results representation
        results_text = format_query_results(results_df) if results_df is not None and not results_df.empty else "No results found"

        explain_prompt = PromptTemplate(
            input_variables=["query", "results", "sql_query"],
            template=DATABASE_EXPLANATION_PROMPT
        )

        explain_chain = explain_prompt | self.explanation_llm | StrOutputParser()

        try:
            start_time = time.time()
            print(f"⏱️  [QWEN H100] Starting: Results Explanation Generation...")

            explanation = explain_chain.invoke({
                "query": query,
                "results": results_text,
                "sql_query": sql_query
            })

            elapsed = time.time() - start_time
            print(f"✅ Completed: Results Explanation in {elapsed:.2f}s")

            return explanation.strip()
        except Exception as e:
            print(f"❌ Error explaining results: {e}")
            return "I found the results but had trouble explaining them."

