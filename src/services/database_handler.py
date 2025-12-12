
from typing import Dict, List, Tuple, Optional
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config, get_openai_config
from src.services.chat_memory import ChatMemory
import os
import time
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
load_dotenv()

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

        # Always use Ollama for explanations
        ollama_config = get_ollama_config()
        self.explanation_llm = Ollama(
            model=ollama_config["model_name"],
            base_url=ollama_config["base_url"],
            temperature=ollama_config["temperature_greeting"]  # Use moderate temperature for explanations
        )

        # Load custom prompt template
        self.custom_prompt_template = self._load_custom_prompt()

        # Create prompt template with dynamic data including conversation history
        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "portfolio_names", "account_ids", "query", "matched_symbols", "conversation_history"],
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

### **6. Available Data (Current State)**

**Portfolio Names:**
{{portfolio_names}}

**Account IDs:**
{{account_ids}}

**Matched Stock Symbols (if any):**
{{matched_symbols}}

---

### **7. Output Instructions**

- You have only the provided portfolio names and account IDs to reference
- Match user questions to these names/IDs only even if he asks in a different way
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

### **The Gold Standard SQL Query**

```sql
WITH portfolio_stats AS (
    SELECT 
        portfolio_name, 
        ytd_profit AS portfolio_pnl, 
        ytd_return AS portfolio_return_pct -- Include for context, even if comparing PnL
    FROM ai_trading.portfolio_summary
    WHERE portfolio_name = 'A-Balanced'
      AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE portfolio_name = 'A-Balanced')
),
stock_stats AS (
    SELECT 
        symbol, 
        ytd_total_pnl AS stock_pnl, 
        market_value AS stock_current_value
    FROM ai_trading.portfolio_holdings_realized_pnl
    WHERE symbol = 'QQQ'
      -- Important: Sum across all portfolios if the user owns QQQ in multiple places
      -- Or filter by specific portfolio if implied. Here we assume aggregate holding.
      AND datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_holdings_realized_pnl)
)
SELECT 
    p.portfolio_name,
    p.portfolio_pnl,
    s.symbol,
    s.stock_pnl,
    (p.portfolio_pnl - s.stock_pnl) AS pnl_difference
FROM portfolio_stats p
CROSS JOIN stock_stats s;
```

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
            template="""Extract any stock names, company names, or stock symbols mentioned in this question.
Return only the extracted terms, separated by commas. If none found, return "NONE".

Examples:
- "What is the price of Apple stock?" -> Apple
- "Show me MSFT and GOOGL performance" -> MSFT, GOOGL
- "How is Tesla doing?" -> Tesla
- "What are my portfolios?" -> NONE

Question: {query}

Extracted terms:"""
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
            
        # Create a string representation of the symbols list
        # If list is too long, we might need to truncate or use a retrieval approach
        # For ~800 symbols, it might fit in context but let's be careful
        symbols_str = ", ".join(all_symbols_list)
        
        match_prompt = PromptTemplate(
            input_variables=["term", "symbols"],
            template="""You are an expert financial data assistant.
Your task is to identify the correct stock symbol from the provided list that corresponds to the company or term mentioned by the user.

User Term: "{term}"

Available Symbols List:
{symbols}

Instructions:
1. Find the symbol in the list that best matches the User Term.
2. Example: If User Term is "Apple", and list has "AAPL", return "AAPL".
3. Example: If User Term is "National Bank", and list has "NBK", return "NBK".
4. If the exact symbol is in the list, return it.
5. If a very strong match is found (e.g. company name to ticker), return the ticker.
6. If NO match is found, return "NONE".
7. Return ONLY the symbol name (or "NONE"). Do not add any explanation.

Matching Symbol:"""
        )
        
        match_chain = match_prompt | self.llm | StrOutputParser()

        try:
            start_time = time.time()
            print(f"⏱️  Starting: Symbol Matching for '{extracted_term}'...")

            # For ~800 symbols, it fits in context.
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
        Generate SQL query from natural language question with fuzzy stock matching and conversation context

        Args:
            query: User's natural language question
            schema: Database schema information
            chat_history: Previous conversation history for follow-up questions

        Returns:
            Generated SQL query string
        """
        if chat_history is None:
            chat_history = []

        # Get dynamic data from database
        portfolio_names = self._get_portfolio_names()
        account_ids = self._get_account_ids()

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

        # Invoke the chain with all parameters including conversation history
        start_time = time.time()
        print(f"⏱️  Starting: SQL Query Generation...")

        sql = self.sql_chain.invoke({
            "schema": schema,
            "portfolio_names": portfolio_names,
            "account_ids": account_ids,
            "matched_symbols": matched_symbols,
            "conversation_history": conversation_text,
            "query": query
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
        print(f"  → Generated SQL: {sql}")
        return sql

        # except Exception as e:
        #     print(f"Error generating SQL: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return "ERROR: Failed to generate SQL query"

    def explain_results(self, query: str, results_df, sql_query: str) -> str:
        """
        Generate a natural language explanation of query results
        Uses Ollama for explanations regardless of SQL generation model

        Args:
            query: Original user question
            results_df: Pandas DataFrame with results
            sql_query: The SQL query that was executed

        Returns:
            Natural language explanation
        """
        results_text = results_df.to_string(index=False) if results_df is not None and not results_df.empty else "No results"

        explain_prompt = PromptTemplate(
            input_variables=["query", "sql", "results"],
            template="""You are a helpful assistant explaining database query results for financial trading data.

User Question: {query}

SQL Query: {sql}

Results:
{results}

Provide a clear, concise explanation of the results in natural language.
Focus on answering the user's question directly with specific numbers and insights."""
        )

        explain_chain = explain_prompt | self.explanation_llm | StrOutputParser()

        try:
            start_time = time.time()
            print(f"⏱️  Starting: Results Explanation Generation...")

            explanation = explain_chain.invoke({
                "query": query,
                "sql": sql_query,
                "results": results_text
            })

            elapsed = time.time() - start_time
            print(f"✅ Completed: Results Explanation in {elapsed:.2f}s")

            return explanation.strip()
        except Exception as e:
            print(f"❌ Error explaining results: {e}")
            return "I found the results but had trouble explaining them."

