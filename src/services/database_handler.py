"""
Database Query Handler - Generates SQL and explains results using configurable LLM provider
"""
from typing import Dict, List, Tuple, Optional, Generator
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.llm_provider import get_llm, get_streaming_llm, get_active_provider, get_provider_config
from src.config.prompts import (
    STOCK_EXTRACTION_PROMPT,
    SYMBOL_MATCHING_PROMPT,
)
from src.services.chat_memory import ChatMemory
import os
import time
from datetime import datetime
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
load_dotenv()

# TOON formatter for token-efficient data formatting
from src.utils.toon_formatter import format_query_results, format_symbol_list


class DatabaseQueryHandler:
    """Handles database-related queries by generating SQL using custom prompt"""

    def __init__(self, model_name: str = None, sql_executor=None, memory_max_pairs: int = 5):
        """
        Initialize the database query handler

        Args:
            model_name: Name of the model to use (defaults to provider config)
            sql_executor: Optional SQL executor for fetching dynamic data
            memory_max_pairs: Maximum number of Q&A pairs to remember for follow-up questions
        """
        self.sql_executor = sql_executor
        self._symbols_cache = None
        self._portfolio_context_cache = None  # Cache for portfolio context
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

        # Get LLM from provider configuration  
        provider = get_active_provider()
        config = get_provider_config()
        self.model_name = model_name or config["model_name"]
        
        print(f"🚀 SQL Generation using {provider.upper()}: {self.model_name}")
        
        # LLM for SQL generation (low temperature for precision)
        self.llm = get_llm(temperature=0.1)
        
        # LLM for explanations (streaming, higher temperature)
        self.explanation_llm = get_streaming_llm(temperature=0.7)

        # Load custom prompt template
        self.custom_prompt_template = self._load_custom_prompt()

        # Create prompt template with dynamic data including conversation history and portfolio context
        self.sql_prompt = PromptTemplate(
            input_variables=["query", "matched_symbols", "conversation_history", "portfolio_context"],
            template=self.custom_prompt_template
        )

        self.sql_chain = self.sql_prompt | self.llm | StrOutputParser()

    def _load_custom_prompt(self) -> str:
        """Load custom prompt from test2sql_prompt.md"""
        prompt_file = "src/services/test2sql_prompt.md"

        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
                print("=" * 20 + "\nLoaded custom SQL prompt template.\n" + "=" * 20)
        else:
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        return prompt_content

    def _format_conversation_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format conversation history for SQL generation context.
        
        Includes SQL queries and results summaries from previous messages to give 
        the LLM context about what data was previously queried and returned.
        """
        if not chat_history:
            print("📝 Conversation history: Empty (first question)")
            return "No previous conversation (this is the first question)."

        recent_messages = self.chat_memory.get_context_messages(chat_history)

        if not recent_messages:
            print("📝 Conversation history: Empty after filtering")
            return "No previous conversation (this is the first question)."

        print(f"📝 Conversation history: {len(recent_messages)} messages found")

        formatted_lines = []
        for msg in recent_messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            sql_query = msg.get("sql_query", "")
            results = msg.get("results")

            if len(content) > 200:
                content = content[:200] + "..."

            formatted_lines.append(f"{role}: {content}")
            
            # Include SQL query if present (critical for follow-up questions!)
            if sql_query and role.lower() == "assistant":
                # Truncate very long SQL queries
                if len(sql_query) > 500:
                    sql_query = sql_query[:500] + "..."
                formatted_lines.append(f"[Previous SQL Query: {sql_query}]")
            
            # Include results summary if present (critical for "those", "them" references!)
            if results and role.lower() == "assistant":
                results_summary = self._summarize_results(results)
                if results_summary:
                    formatted_lines.append(f"[Query Results: {results_summary}]")

        return "\n".join(formatted_lines)

    def _summarize_results(self, results) -> str:
        """Create a compact summary of query results for context.
        
        Extracts key identifiers (symbols, portfolio names, groups) that can be referenced
        in follow-up questions. Provides structured metadata for better LLM reasoning.
        """
        try:
            if results is None:
                return ""
            
            # Handle list of dicts (common format from API)
            if isinstance(results, list) and len(results) > 0:
                # Extract key columns for context
                sample = results[0] if results else {}
                keys = list(sample.keys()) if isinstance(sample, dict) else []
                
                summary_parts = []
                summary_parts.append(f"{len(results)} rows")
                
                # Extract distinct values for key identifier columns
                portfolios = set()
                symbols = set()
                groups = set()
                account_ids = set()
                
                for row in results:
                    if isinstance(row, dict):
                        # Collect distinct values for each key column
                        if 'portfolio_name' in row and row['portfolio_name']:
                            portfolios.add(str(row['portfolio_name']))
                        if 'symbol' in row and row['symbol']:
                            symbols.add(str(row['symbol']))
                        if 'group_name' in row and row['group_name']:
                            groups.add(str(row['group_name']))
                        if 'account_id' in row and row['account_id']:
                            account_ids.add(str(row['account_id']))
                
                # Build structured summary with distinct values
                if portfolios:
                    portfolio_list = sorted(list(portfolios))[:8]  # First 8 unique
                    summary_parts.append(f"portfolios: {', '.join(portfolio_list)}")
                
                if groups:
                    group_list = sorted(list(groups))[:5]  # First 5 unique
                    summary_parts.append(f"groups: {', '.join(group_list)}")
                
                if symbols:
                    symbol_list = sorted(list(symbols))[:10]  # First 10 unique
                    summary_parts.append(f"symbols: {', '.join(symbol_list)}")
                
                if account_ids:
                    account_list = sorted(list(account_ids))[:5]  # First 5 unique
                    summary_parts.append(f"accounts: {', '.join(account_list)}")
                
                # Use pipe separator for better parsing
                return " | ".join(summary_parts)
            
            return ""
        except Exception as e:
            print(f"⚠️ Error summarizing results: {e}")
            return ""

    def _get_all_symbols_dict(self) -> Dict[str, str]:
        """Fetch all symbols from database and return as dictionary"""
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
                symbols_dict = {str(i): symbol for i, symbol in enumerate(df['symbol'].tolist())}
                self._symbols_cache = symbols_dict
                return symbols_dict
            return {}
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return {}

    def _get_portfolio_context(self) -> str:
        """
        Fetch portfolio context (names, account IDs, indices) from database for SQL generation.
        Returns formatted string with this context information.
        """
        if self._portfolio_context_cache is not None:
            return self._portfolio_context_cache

        if not self.sql_executor:
            return "No portfolio context available."

        try:
            query = """
            SELECT DISTINCT 
                portfolio_name, 
                account_id, 
                default_index
            FROM ai_trading.portfolio_summary
            WHERE datetime = (SELECT MAX(datetime) FROM ai_trading.portfolio_summary WHERE is_active = 1)
              AND is_active = 1
            ORDER BY portfolio_name
            """
            success, df, _ = self.sql_executor.execute_query(query)
            
            if success and df is not None and not df.empty:
                # Format portfolio names
                portfolio_names = df['portfolio_name'].dropna().unique().tolist()
                account_ids = df['account_id'].dropna().unique().tolist()
                
                # Format as a context string
                context_parts = []
                context_parts.append(f"**Available Portfolio Names:** {', '.join(portfolio_names)}")
                context_parts.append(f"**Available Account IDs:** {', '.join(account_ids)}")
                
                # Add portfolio-index mapping
                index_mapping = []
                for _, row in df.iterrows():
                    if row['portfolio_name'] and row['default_index']:
                        index_mapping.append(f"  - {row['portfolio_name']}: {row['default_index']}")
                
                if index_mapping:
                    context_parts.append("**Portfolio Default Indices:**\n" + "\n".join(index_mapping))
                
                context_str = "\n".join(context_parts)
                self._portfolio_context_cache = context_str
                print(f"📋 Loaded portfolio context: {len(portfolio_names)} portfolios, {len(account_ids)} accounts")
                return context_str
            
            return "No portfolio context available."
            
        except Exception as e:
            print(f"Error fetching portfolio context: {e}")
            return "No portfolio context available."

    def _extract_stock_mentions(self, query: str) -> List[str]:
        """Extract potential stock mentions from query using LLM"""
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

            terms = [term.strip() for term in result.split(',') if term.strip()]
            return terms
        except Exception as e:
            print(f"❌ Error extracting stock mentions: {e}")
            return []

    def _match_symbol_from_list(self, extracted_term: str, all_symbols_list: List[str]) -> Optional[str]:
        """Use LLM to find the best matching symbol from the list"""
        if not extracted_term or not all_symbols_list:
            return None

        symbols_str = format_symbol_list(all_symbols_list)

        match_prompt = PromptTemplate(
            input_variables=["term", "symbols"],
            template=SYMBOL_MATCHING_PROMPT
        )

        match_chain = match_prompt | self.llm | StrOutputParser()

        try:
            start_time = time.time()
            print(f"⏱️  Starting: Symbol Matching for '{extracted_term}'...")

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
        """Match mentioned terms to actual symbols using LLM selection"""
        if not mentioned_terms or not symbols_dict:
            return "No stock symbols detected in the question"

        symbols_list = list(symbols_dict.values())
        matched_info = []

        for term in mentioned_terms:
            matched_symbol = self._match_symbol_from_list(term, symbols_list)

            if matched_symbol:
                matched_info.append(f"'{term}' -> '{matched_symbol}'")

        if matched_info:
            return "Detected stock mentions:\n" + "\n".join(matched_info)
        else:
            return f"No close matches found for: {', '.join(mentioned_terms)}"

    def generate_sql(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Generate SQL query from natural language question with fuzzy stock matching.

        Args:
            query: User's natural language question
            chat_history: Previous conversation history for follow-up questions

        Returns:
            Generated SQL query string
        """
        if chat_history is None:
            chat_history = []

        conversation_text = self._format_conversation_history(chat_history)

        # Check for stock mentions
        print("📊 Checking for stock mentions...")
        mentioned_terms = self._extract_stock_mentions(query)

        matched_symbols = "N/A"
        if mentioned_terms:
            print(f"  → Found potential stock mentions: {mentioned_terms}")
            symbols_dict = self._get_all_symbols_dict()
            matched_symbols = self._fuzzy_match_symbols(mentioned_terms, symbols_dict)
            print(f"  → {matched_symbols}")

        # Get portfolio context for better SQL generation
        portfolio_context = self._get_portfolio_context()

        # Generate SQL
        start_time = time.time()
        print(f"⏱️  SQL Query Generation...")

        sql = self.sql_chain.invoke({
            "matched_symbols": matched_symbols,
            "conversation_history": conversation_text,
            "portfolio_context": portfolio_context,
            "query": query
        })

        elapsed = time.time() - start_time
        print(f"✅ Completed: SQL Query Generation in {elapsed:.2f}s")

        # Clean up the response
        sql = sql.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join([line for line in lines if not line.startswith("```") and "sql" not in line.lower()])
        sql = sql.strip()

        print(f"  → Generated SQL: {sql}")

        return sql

    def explain_results(self, query: str, results_df, sql_query: str) -> str:
        """
        Generate a natural language explanation of query results (non-streaming).
        Uses UnifiedResponseGenerator for consistent formatting.

        Args:
            query: Original user question
            results_df: Pandas DataFrame with results
            sql_query: The SQL query that was executed

        Returns:
            Natural language explanation
        """
        from src.services.unified_response_generator import get_response_generator
        
        results_text = format_query_results(results_df) if results_df is not None and not results_df.empty else "No results found"

        try:
            start_time = time.time()
            print(f"⏱️  Starting: Results Explanation Generation...")
            
            generator = get_response_generator()
            data = {
                "sql_query": sql_query,
                "results": results_text
            }
            
            explanation = generator.generate_response(
                query=query,
                context_type="database",
                data=data
            )

            elapsed = time.time() - start_time
            print(f"✅ Completed: Results Explanation in {elapsed:.2f}s")

            return explanation

        except Exception as e:
            print(f"❌ Error explaining results: {e}")
            return "I found the results but had trouble explaining them."

    def stream_explain_results(self, query: str, results_df, sql_query: str):
        """
        Stream a natural language explanation of query results.
        Uses UnifiedResponseGenerator for consistent formatting.

        Args:
            query: Original user question
            results_df: Pandas DataFrame with results
            sql_query: The SQL query that was executed

        Yields:
            String chunks of the explanation
        """
        from src.services.unified_response_generator import get_response_generator
        
        # Use TOON format for token-efficient results representation
        results_text = format_query_results(results_df) if results_df is not None and not results_df.empty else "No results found"

        try:
            print(f"⏱️  Starting: Streaming Results Explanation...")
            start_time = time.time()
            first_token_received = False
            
            generator = get_response_generator()
            data = {
                "sql_query": sql_query,
                "results": results_text
            }
            
            for chunk in generator.stream_response(
                query=query,
                context_type="database",
                data=data
            ):
                if not first_token_received:
                    elapsed = time.time() - start_time
                    print(f"⚡ First token received in {elapsed:.4f}s")
                    first_token_received = True
                yield chunk

            print(f"✅ Completed: Streaming Explanation")
        except Exception as e:
            print(f"❌ Error streaming explanation: {e}")
            yield "I found the results but had trouble explaining them."
