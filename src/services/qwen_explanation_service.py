"""
QWEN-based Explanation Service

Uses QWEN3:30B-3B model on H100 (wrapped by OpenAI API) for result explanations.
This is an alternative to the Ollama-based gpt-oss:20b model for comparison testing.
"""

import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()


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
    
    Uses OpenAI-compatible API endpoint for model access.
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
        
        self.llm= ChatOpenAI(
                model="Qwen3-30B-A3B",
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=0,
                max_retries=2
            )
        # Explanation prompt (same as database_handler)
        self.explain_prompt = PromptTemplate(
            input_variables=["query", "results", "sql_query"],
            template="""You are a financial portfolio assistant interpreting data for users.

**User Question:** {query}

**Context (SQL Query Used):**
{sql_query}

**Retrieved Data:**
{results}

**Your Role:**
Interpret and explain the data **from the user's perspective**. Your job is to answer their question directly, not describe the data structure.

**Rules:**
1. **Answer the question directly** - Focus on what the user asked, not on how the data is structured
2. **Never mention data rows, columns, or table structures** - Speak as if you're a financial advisor explaining insights
3. **Use specific numbers and names** - Reference actual values from the data (portfolio names, amounts, percentages)
4. **Be conversational and helpful** - The user doesn't need to know about databases or queries
5. **If no results found** - Simply say the information wasn't found, don't suggest technical solutions
6. **Format nicely** - Use bullet points or brief paragraphs for clarity when appropriate
7. **No code or SQL** - Never include code, SQL, or technical syntax in your response

**Response:**"""
        )
        
        self.explain_chain = self.explain_prompt | self.llm | StrOutputParser()
    
    def explain_results(self, query: str, results_df, sql_query: str) -> tuple[str, float]:
        """
        Generate a natural language explanation of query results.
        
        Args:
            query: Original user question
            results_df: Pandas DataFrame with results
            sql_query: The SQL query that was executed
            
        Returns:
            Tuple of (explanation, elapsed_time_seconds)
        """
        results_text = results_df.to_string(index=False) if results_df is not None and not results_df.empty else "No results found"
        
        try:
            start_time = time.time()
            print(f"⏱️  [QWEN H100] Starting: Results Explanation Generation...")
            
            explanation = self.explain_chain.invoke({
                "query": query,
                "results": results_text,
                "sql_query": sql_query
            })
            
            elapsed = time.time() - start_time
            print(f"✅ [QWEN H100] Completed: Results Explanation in {elapsed:.2f}s")
            
            return explanation.strip(), elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ [QWEN H100] Error explaining results: {e}")
            return f"Error: {e}", elapsed
    
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
            return True, f"Connection successful (response time: {elapsed:.2f}s)", elapsed
        except Exception as e:
            return False, f"Connection failed: {e}", 0.0


# Convenience function
def create_qwen_explainer() -> QwenExplanationService:
    """Create and return a QWEN explanation service instance."""
    return QwenExplanationService()
