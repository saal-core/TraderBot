"""Vanna AI service for text-to-SQL portfolio queries."""
import logging
import shutil
from typing import Optional, Dict, Any
import pandas as pd
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

try:
    from vanna.anthropic import Anthropic
except (ImportError, ModuleNotFoundError):
    # Fallback if vanna.anthropic is not available or anthropic is not installed
    try:
        import anthropic
        class Anthropic:
            def __init__(self, config=None):
                if config is None:
                    config = {}
                self.api_key = config.get("api_key")
                self.model = config.get("model", "claude-3-opus-20240229")
                self.client = anthropic.Anthropic(api_key=self.api_key)

            def system_message(self, message: str) -> any:
                return {"role": "system", "content": message}

            def user_message(self, message: str) -> any:
                return {"role": "user", "content": message}

            def assistant_message(self, message: str) -> any:
                return {"role": "assistant", "content": message}

            def submit_prompt(self, prompt, **kwargs) -> str:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
    except ModuleNotFoundError:
        # If anthropic package is not installed, create a stub
        class Anthropic:
            def __init__(self, config=None):
                raise ImportError("anthropic package is not installed. Install it with: pip install anthropic")

from src.config.settings import get_settings
from src.config.prompts import VANNA_EXPLANATION_PROMPT
from src.services.ollama_service import OllamaService
import os


settings = get_settings()

class MyVanna(ChromaDB_VectorStore, Ollama):
    """Custom Vanna class combining ChromaDB vector store with Ollama LLM."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Vanna with ChromaDB and Ollama.

        Args:
            config: Configuration dict with model, ollama_api_url, and persist_directory
        """
        if config is None:
            config = {
                "model": settings.ollama_model,
                "ollama_api_url": settings.ollama_api_url,
                "persist_directory": settings.vanna_persist_dir,
            }
        
        persist_dir = config.get("persist_directory", settings.vanna_persist_dir)
        
        try:
            ChromaDB_VectorStore.__init__(self, config=config)
        except (KeyError, Exception) as e:
            logging.warning(f"ChromaDB initialization failed ({type(e).__name__}: {e}), clearing and reinitializing...")
            # Remove corrupted database
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
                logging.info(f"Cleared corrupted ChromaDB at {persist_dir}")
            # Retry initialization
            try:
                ChromaDB_VectorStore.__init__(self, config=config)
            except Exception as retry_error:
                logging.error(f"ChromaDB reinitialization failed: {retry_error}")
                raise
        
        Ollama.__init__(self, config=config)


class MyVannaClaude(ChromaDB_VectorStore, Anthropic):
    """Custom Vanna class combining ChromaDB vector store with Anthropic LLM."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Vanna with ChromaDB and Anthropic.
        """
        if config is None:
            config = {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model": "claude-sonnet-4-5-20250929", # Default to Sonnet 4.5
                "persist_directory": settings.vanna_persist_dir,
            }
        ChromaDB_VectorStore.__init__(self, config=config)
        Anthropic.__init__(self, config=config)
    
    # These abstract methods are inherited from Anthropic but may not be needed for Vanna
    def system_message(self, message: str):
        """Create a system message (inherited from Anthropic)."""
        return {"role": "system", "content": message}

    def user_message(self, message: str):
        """Create a user message (inherited from Anthropic)."""
        return {"role": "user", "content": message}

    def assistant_message(self, message: str):
        """Create an assistant message (inherited from Anthropic)."""
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        """Submit a prompt to Claude (inherited from Anthropic)."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=messages
        )
        return response.content[0].text



class VannaService:
    """Service for portfolio queries using Vanna AI text-to-SQL."""

    def __init__(self, llm_type: str = "gpt-oss"):
        """Initialize Vanna service."""
        self.llm_type = llm_type
        self.vn_ollama = MyVanna()
        self.vn_claude = None # Lazy init
        
        self.vn = self.vn_ollama # Default
        if llm_type == "claude":
             self.vn_claude = MyVannaClaude()
             self.vn = self.vn_claude

        self.ollama = OllamaService()
        self.connected = False

    def set_llm(self, llm_type: str):
        """Switch LLM backend."""
        self.llm_type = llm_type
        if llm_type == "claude":
            if self.vn_claude is None:
                self.vn_claude = MyVannaClaude()
            self.vn = self.vn_claude
        else:
            self.vn = self.vn_ollama


    def connect(self) -> bool:
        """
        Connect to PostgreSQL database.

        Returns:
            bool: True if connection successful
        """
        try:
            # Connect both instances if they exist
            self.vn_ollama.connect_to_postgres(
                host=settings.db_host,
                dbname=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                port=settings.db_port
            )
            
            if self.vn_claude:
                self.vn_claude.connect_to_postgres(
                    host=settings.db_host,
                    dbname=settings.db_name,
                    user=settings.db_user,
                    password=settings.db_password,
                    port=settings.db_port
                )
                
            self.connected = True
            logging.info(f"Connected to Vanna database (LLM: {self.llm_type}).")
            return True
        except Exception as e:
            logging.error(f"Vanna DB connection failed: {e}")
            self.connected = False
            return False

    def generate_sql(self, question: str) -> Optional[str]:
        """
        Generate SQL from natural language question.

        Args:
            question: Natural language question

        Returns:
            str: Generated SQL query, or None if failed
        """
        if not self.connected:
            raise ConnectionError("Database not connected. Call connect() first.")

        try:
            sql = self.vn.generate_sql(question)
            return sql
        except Exception as e:
            logging.error(f"Error generating SQL: {e}")
            return None

    def execute_sql(self, sql: str) -> Optional[pd.DataFrame]:
        """
        Execute SQL query and return results as DataFrame.

        Args:
            sql: SQL query to execute

        Returns:
            pd.DataFrame: Query results, or None if failed
        """
        if not self.connected:
            raise ConnectionError("Database not connected. Call connect() first.")

        try:
            result_df = self.vn.run_sql(sql)
            return result_df
        except Exception as e:
            logging.error(f"Error executing SQL: {e}")
            return None

    def explain_results(
        self,
        question: str,
        result_df: pd.DataFrame,
        chat_history: list
    ) -> str:
        """
        Generate natural language explanation of query results.

        Args:
            question: Original question
            result_df: Query results DataFrame
            chat_history: Previous conversation history

        Returns:
            str: Natural language explanation
        """
        if result_df is None or result_df.empty:
            return "I found no data in the database that matches your query."

        result_text = result_df.to_string(index=False)

        # Use recent dialogue history (last 8 messages)
        filtered_history = [msg for msg in chat_history if msg["role"] in ["user", "assistant"]]
        recent_dialogue = filtered_history[-8:]

        # Remove leading assistant message if present
        if recent_dialogue and recent_dialogue[0]["role"] == "assistant":
            recent_dialogue = recent_dialogue[1:]

        # Construct explanation prompt
        final_instruction = (
            f"My original question was: \"{question}\"\n\n"
            f"The database returned the following data:\n---\n{result_text}\n---\n\n"
            "Based on our conversation and the data above, please provide a clear explanation."
        )

        messages = [
            {"role": "system", "content": VANNA_EXPLANATION_PROMPT}
        ] + recent_dialogue + [
            {"role": "user", "content": final_instruction}
        ]

        explanation = self.ollama.generate_response(messages, temperature=0.3)
        return explanation

    def query(
        self,
        question: str,
        chat_history: list
    ) -> tuple[Optional[str], Optional[pd.DataFrame], Optional[str]]:
        """
        Execute complete query workflow: generate SQL, execute, and explain.

        Args:
            question: Natural language question
            chat_history: Previous conversation history

        Returns:
            tuple: (sql, result_df, explanation)
        """
        try:
            # Generate SQL
            sql = self.generate_sql(question)
            if not sql or "SELECT" not in sql.upper():
                raise ValueError("Failed to generate a valid SQL query.")

            # Execute SQL
            result_df = self.execute_sql(sql)

            # Generate explanation
            explanation = self.explain_results(question, result_df, chat_history)

            return sql, result_df, explanation

        except Exception as e:
            logging.error(f"Error in query workflow: {e}")
            return None, None, None
