"""
QWEN-based Explanation Service

Uses QWEN3:30B-3B model on H100 for streaming result explanations.
"""
import time
from typing import Dict, Any, Generator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.prompts import DATABASE_EXPLANATION_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()

from src.utils.toon_formatter import format_query_results


def get_qwen_config() -> Dict[str, Any]:
    """Get QWEN configuration from environment variables."""
    return {
        "base_url": os.getenv("QWEN_BASE_URL", "http://192.168.71.72:8080/v1"),
        "api_key": os.getenv("QWEN_API_KEY", "123"),
        "model_name": os.getenv("QWEN_MODEL", "qwen3-30b-3b"),  # Adjust model name as needed
        "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.3")),
    }


class QwenExplanationService:
    """
    Explanation service using QWEN3:30B-3B model on H100.
    Supports both streaming and non-streaming explanations.
    """

    def __init__(self, model_name: str = None, temperature: float = None):
        """
        Initialize QWEN explanation service.

        Args:
            model_name: Override model name (optional)
            temperature: Override temperature (optional)
        """
        config = get_qwen_config()

        self.model_name = model_name or config["model_name"]
        self.temperature = temperature if temperature is not None else config["temperature"]
        self.base_url = config["base_url"]
        self.api_key = config["api_key"]

        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_retries=2,
            streaming=True
        )

        self.explain_prompt = PromptTemplate(
            input_variables=["query", "results", "sql_query", "today_date"],
            template=DATABASE_EXPLANATION_PROMPT
        )

    def explain_results(self, query: str, results_df, sql_query: str) -> tuple[str, float]:
        """
        Generate a natural language explanation of query results (non-streaming).

        Args:
            query: Original user question
            results_df: Pandas DataFrame with results
            sql_query: The SQL query that was executed

        Returns:
            Tuple of (explanation, elapsed_time_seconds)
        """
        results_text = format_query_results(results_df) if results_df is not None and not results_df.empty else "No results found"

        try:
            start_time = time.time()
            print(f"⏱️  [QWEN H100] Starting: Results Explanation...")

            from datetime import datetime
            today_date = datetime.now().strftime("%A, %B %d, %Y")

            formatted_prompt = self.explain_prompt.format(
                query=query,
                results=results_text,
                sql_query=sql_query,
                today_date=today_date
            )

            explanation = self.llm.invoke(formatted_prompt)

            elapsed = time.time() - start_time
            print(f"✅ [QWEN H100] Completed in {elapsed:.2f}s")

            if hasattr(explanation, 'content'):
                return explanation.content.strip(), elapsed
            return str(explanation).strip(), elapsed

        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ [QWEN H100] Error: {e}")
            return f"Error: {e}", elapsed
    
    def stream_explain_results(self, query: str, results_df, sql_query: str):
        """
        Stream a natural language explanation of query results.
        
        Args:
            query: Original user question
            results_df: Pandas DataFrame with results
            sql_query: The SQL query that was executed
            
        Yields:
            String chunks of the explanation
        """
        # Use TOON format for token-efficient results representation
        results_text = format_query_results(results_df) if results_df is not None and not results_df.empty else "No results found"
        
        try:
            print(f"⏱️  [QWEN H100] Starting: Streaming Results Explanation...")
            
            for chunk in self.explain_chain.stream({
                "query": query,
                "results": results_text,
                "sql_query": sql_query
            }):
                yield chunk
                
            print(f"✅ [QWEN H100] Completed: Streaming Explanation")
        except Exception as e:
            print(f"❌ [QWEN H100] Error streaming explanation: {e}")
            yield f"Error: {e}", elapsed
    
    def test_connection(self) -> tuple[bool, str, float]:
        """
        Test the QWEN API connection.

        Returns:
            Tuple of (success, message, response_time)
        """
        try:
            start_time = time.time()
            response = self.llm.invoke("Hello, respond with just 'OK'.")
            elapsed = time.time() - start_time
            return True, f"Connection successful ({elapsed:.2f}s)", elapsed
        except Exception as e:
            return False, f"Connection failed: {e}", 0.0


def create_qwen_explainer() -> QwenExplanationService:
    """Create and return a QWEN explanation service instance."""
    return QwenExplanationService()