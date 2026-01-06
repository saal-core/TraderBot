"""
Database Query Handler - Generates SQL and explains results using QWEN streaming
"""
from typing import Dict, List, Tuple, Optional, Generator
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config, get_openai_config, get_qwen_config
from src.config.prompts import (
    DATABASE_EXPLANATION_PROMPT,
    STOCK_EXTRACTION_PROMPT,
    SYMBOL_MATCHING_PROMPT,
    detect_language,
    ARABIC_FINANCIAL_GLOSSARY
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
            model_name: Name of the model to use (defaults to config)
            use_openai: Whether to use OpenAI for SQL generation (True) or Ollama (False)
            sql_executor: Optional SQL executor for fetching dynamic data
            memory_max_pairs: Maximum number of Q&A pairs to remember for follow-up questions
        """
        self.sql_executor = sql_executor
        self._symbols_cache = None
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

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
            model=qwen_config.get("model_name", "Qwen3-30B-A3B"),
            base_url=qwen_config["base_url"],
            api_key=qwen_config["api_key"],
            temperature=qwen_config.get("temperature", 0.3),
            max_retries=2,
            streaming=True  # Enable streaming
        )

        # Load custom prompt template
        self.custom_prompt_template = self._load_custom_prompt()

        # Create prompt template with dynamic data including conversation history
        self.sql_prompt = PromptTemplate(
            input_variables=["query", "matched_symbols", "conversation_history"],
            template=self.custom_prompt_template
        )

        self.sql_chain = self.sql_prompt | self.llm | StrOutputParser()

        # Explanation prompt with language and glossary placeholders
        self.explain_prompt = PromptTemplate(
            input_variables=["query", "results", "sql_query", "today_date", "language", "arabic_glossary"],
            template=DATABASE_EXPLANATION_PROMPT
        )

    def _load_custom_prompt(self) -> str:
        """Load custom prompt from test2sql_prompt.md"""
        prompt_file = "/home/dev/Hussein/TraderBot/test2sql_prompt.md"

        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
                print("=" * 20 + "\nLoaded custom SQL prompt template.\n" + "=" * 20)
        else:
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        return prompt_content

    def _format_conversation_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format conversation history for SQL generation context"""
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

            if len(content) > 200:
                content = content[:200] + "..."

            formatted_lines.append(f"{role}: {content}")

        return "\n".join(formatted_lines)

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

        # Generate SQL
        start_time = time.time()
        print(f"⏱️  SQL Query Generation...")

        sql = self.sql_chain.invoke({
            "matched_symbols": matched_symbols,
            "conversation_history": conversation_text,
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

        Args:
            query: Original user question
            results_df: Pandas DataFrame with results
            sql_query: The SQL query that was executed

        Returns:
            Natural language explanation
        """
        results_text = format_query_results(results_df) if results_df is not None and not results_df.empty else "No results found"

        try:
            start_time = time.time()
            print(f"⏱️  [QWEN H100] Starting: Results Explanation Generation...")

            today_date = datetime.now().strftime("%A, %B %d, %Y")
            
            # Detect language and prepare glossary
            language = detect_language(query)
            arabic_glossary = ARABIC_FINANCIAL_GLOSSARY if language == "Arabic" else "N/A"

            formatted_prompt = self.explain_prompt.format(
                query=query,
                results=results_text,
                sql_query=sql_query,
                today_date=today_date,
                language=language,
                arabic_glossary=arabic_glossary
            )

            explanation = self.explanation_llm.invoke(formatted_prompt)

            elapsed = time.time() - start_time
            print(f"✅ Completed: Results Explanation in {elapsed:.2f}s")

            # Handle AIMessage response
            if hasattr(explanation, 'content'):
                return explanation.content.strip()
            return str(explanation).strip()

        except Exception as e:
            print(f"❌ Error explaining results: {e}")
            return "I found the results but had trouble explaining them."

    def stream_explain_results(self, query: str, results_df, sql_query: str):
        """
        Stream a natural language explanation of query results.
        Uses QWEN (H100) for faster explanations with streaming.

        Args:
            query: Original user question
            results_df: Pandas DataFrame with results
            sql_query: The SQL query that was executed

        Yields:
            String chunks of the explanation
        """
        # Use TOON format for token-efficient results representation
        results_text = format_query_results(results_df) if results_df is not None and not results_df.empty else "No results found"

        # Detect language and prepare glossary
        language = detect_language(query)
        arabic_glossary = ARABIC_FINANCIAL_GLOSSARY if language == "Arabic" else "N/A"

        explain_prompt = PromptTemplate(
            input_variables=["query", "results", "sql_query", "today_date", "language", "arabic_glossary"],
            template=DATABASE_EXPLANATION_PROMPT
        )

        explain_chain = explain_prompt | self.explanation_llm | StrOutputParser()

        try:
            print(f"⏱️  [QWEN H100] Starting: Streaming Results Explanation ({language})...")

            # Get today's date for context
            today_date = datetime.now().strftime("%A, %B %d, %Y")
            
            start_time = time.time()
            first_token_received = False

            for chunk in explain_chain.stream({
                "query": query,
                "results": results_text,
                "sql_query": sql_query,
                "today_date": today_date,
                "language": language,
                "arabic_glossary": arabic_glossary
            }):
                if not first_token_received:
                    elapsed = time.time() - start_time
                    print(f"⚡ First token received in {elapsed:.4f}s")
                    first_token_received = True
                yield chunk

            print(f"✅ [QWEN H100] Completed: Streaming Explanation")
        except Exception as e:
            print(f"❌ Error streaming explanation: {e}")
            yield "I found the results but had trouble explaining them."
