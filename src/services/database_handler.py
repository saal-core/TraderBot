
from langchain_community.llms import Ollama
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config
import os
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
from src.services.stock_matcher import StockMatcher
load_dotenv()

class DatabaseQueryHandler:
    """Handles database-related queries by generating SQL using custom prompt"""

    def __init__(self, model_name: str = None, ollama_base_url: str = None, sql_executor=None, llm_type: str = "gpt-oss"):
        """
        Initialize the database query handler

        Args:
            model_name: Name of the Ollama model to use (defaults to config)
            ollama_base_url: Base URL for Ollama API (defaults to config)
            sql_executor: Optional SQL executor for fetching dynamic data
            llm_type: "gpt-oss", "claude", or "gpt-4.1"
        """
        ollama_config = get_ollama_config()

        self.model_name = model_name or ollama_config["model_name"]
        self.base_url = ollama_base_url or ollama_config["base_url"]
        self.temperature = ollama_config["temperature_sql"]
        self.sql_executor = sql_executor
        self.llm_type = llm_type

        # Initialize StockMatcher
        # We pass None for LLM initially, it will use default Ollama.
        # Ideally StockMatcher should also support Claude if we want full "Claude without Vanna" experience,
        # but for now let's focus on SQL generation.
        self.stock_matcher = StockMatcher(llm=None, sql_executor=self.sql_executor)

        if self.llm_type == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            self.llm = ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
                api_key=api_key,
                temperature=0.0
            )
        elif self.llm_type == "gpt-4.1":
            api_key = os.getenv("OPENAI_API_KEY")
            self.llm = ChatOpenAI(
                model="gpt-4.1",
                api_key=api_key,
                temperature=0.0
            )
        elif self.llm_type == "gpt-5.1":
            api_key = os.getenv("OPENAI_API_KEY")
            self.llm = ChatOpenAI(
                model="gpt-5.1",
                api_key=api_key,
                temperature=0.0
            )
        else:
            self.llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature
            )


        # Load custom prompt template
        self.custom_prompt_template = self._load_custom_prompt()

        # Create prompt template with dynamic data
        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "portfolio_names", "account_ids", "query", "matched_symbols"],
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

### **4. Database Schema**

{{schema}}

---

### **5. Available Data (Current State)**

**Portfolio Names:**
{{portfolio_names}}

**Account IDs:**
{{account_ids}}

**Matched Stock Symbols (if any):**
{{matched_symbols}}

---

### **6. Output Instructions**

- You have only the provided portfolio names and account IDs to reference
- Match user questions to these names/IDs only even if he asks in a different way
- Generate ONLY a SELECT query
- Do not use INSERT, UPDATE, DELETE, or any data modification statements
- Return only the SQL query without any explanation or markdown
- Use proper PostgreSQL syntax
- Always use schema prefix: `ai_trading.table_name`
- For "current" or "latest" data, filter by: `WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.table_name)`
- Pay attention to SQL query syntax and correctness
- If the query cannot be answered with the given schema, return "ERROR: Cannot generate query"

---

### **User Question:**

{{query}}

### **SQL Query:**"""

        return full_template

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

    # Stock matching methods moved to StockMatcher class


    def generate_sql(self, query: str, schema: str) -> str:
        """
        Generate SQL query from natural language question with fuzzy stock matching

        Args:
            query: User's natural language question
            schema: Database schema information

        Returns:
            Generated SQL query string
        """
        try:
            # Get dynamic data from database
            portfolio_names = self._get_portfolio_names()
            account_ids = self._get_account_ids()

            # Check for stock mentions and fuzzy match
            print("  → Checking for stock mentions...")
            mentioned_terms = self.stock_matcher.extract_stock_mentions(query)

            matched_symbols = "N/A"
            if mentioned_terms:
                print(f"  → Found potential stock mentions: {mentioned_terms}")
                matched_symbols = self.stock_matcher.fuzzy_match_symbols(mentioned_terms)
                print(f"  → {matched_symbols}")

            # Invoke the chain with all parameters
            sql = self.sql_chain.invoke({
                "schema": schema,
                "portfolio_names": portfolio_names,
                "account_ids": account_ids,
                "matched_symbols": matched_symbols,
                "query": query
            })

            # Clean up the response
            sql = sql.strip()

            # Remove markdown code blocks if present
            if sql.startswith("```"):
                lines = sql.split("\n")
                sql = "\n".join([line for line in lines if not line.startswith("```") and "sql" not in line.lower()])

            sql = sql.strip()

            return sql

        except Exception as e:
            print(f"Error generating SQL: {e}")
            import traceback
            traceback.print_exc()
            return "ERROR: Failed to generate SQL query"

    def explain_results(self, query: str, results_df, sql_query: str) -> str:
        """
        Generate a natural language explanation of query results

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

        explain_chain = explain_prompt | self.llm | StrOutputParser()

        try:
            explanation = explain_chain.invoke({
                "query": query,
                "sql": sql_query,
                "results": results_text
            })
            return explanation.strip()
        except Exception as e:
            print(f"Error explaining results: {e}")
            return "I found the results but had trouble explaining them."

