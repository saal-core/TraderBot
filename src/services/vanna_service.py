"""Vanna AI service for text-to-SQL portfolio queries."""
import logging
from typing import Optional, Dict, Any
import pandas as pd
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from config.settings import get_settings
from config.prompts import VANNA_EXPLANATION_PROMPT
from services.ollama_service import OllamaService

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
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


class VannaService:
    """Service for portfolio queries using Vanna AI text-to-SQL."""

    def __init__(self):
        """Initialize Vanna service."""
        self.vn = MyVanna()
        self.ollama = OllamaService()
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to PostgreSQL database.

        Returns:
            bool: True if connection successful
        """
        try:
            self.vn.connect_to_postgres(
                host=settings.db_host,
                dbname=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                port=settings.db_port
            )
            self.connected = True
            logging.info("Connected to Vanna database.")
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
