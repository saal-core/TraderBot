import os
from typing import Optional, Dict, Any
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import os
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# Helper Functions
# ============================================================================

def get_postgres_config() -> Dict[str, Any]:
    """
    Get PostgreSQL configuration as a dictionary

    Returns:
        Dictionary containing database configuration
    """
    return {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "options": f"-c search_path={os.getenv('DB_SCHEMA', 'ai_trading')}",
    }


def get_ollama_config() -> Dict[str, Any]:
    """
    Get Ollama configuration as a dictionary

    Returns:
        Dictionary containing Ollama configuration
    """
    return {
        "model_name": os.getenv("OLLAMA_MODEL"),
        "base_url": os.getenv("OLLAMA_API_URL"),
        "temperature_routing": os.getenv("OLLAMA_TEMPERATURE_ROUTING", 0.1),
        "temperature_sql": os.getenv("OLLAMA_TEMPERATURE_SQL", 0.2),
        "temperature_greeting": os.getenv("OLLAMA_TEMPERATURE_GREETING", 0.3),
        "temperature_comparison": os.getenv("OLLAMA_TEMPERATURE_COMPARISON", 0.3),
    }


def get_app_config() -> Dict[str, Any]:
    """
    Get application configuration as a dictionary

    Returns:
        Dictionary containing app configuration
    """
    return {
        "title": os.getenv("APP_TITLE",),
        "icon": os.getenv("APP_ICON",),
        "max_history_display": os.getenv("MAX_HISTORY_DISPLAY", 10),
        "chat_context_size": os.getenv("CHAT_CONTEXT_SIZE", 5),
        "max_result_rows": os.getenv("MAX_RESULT_ROWS", 100),
        "query_timeout": os.getenv("QUERY_TIMEOUT", 30),
        "allow_only_select": os.getenv("ALLOW_ONLY_SELECT", True),
        "sql_injection_protection": os.getenv("SQL_INJECTION_PROTECTION", True),
    }


def get_vanna_config() -> Dict[str, Any]:
    return {
        "model": os.getenv("OLLAMA_MODEL"),
        "ollama_api_url": os.getenv("OLLAMA_API_URL"),
        "persist_directory": os.getenv("CHROMADB_PERSIST_DIRECTORY", "chroma_db"),
    }

class Settings:
    def __init__(self):
        self.ollama_model = os.getenv("OLLAMA_MODEL")
        self.ollama_api_url = os.getenv("OLLAMA_API_URL")
        self.vanna_persist_dir = os.getenv("CHROMADB_PERSIST_DIRECTORY", "chroma_db")
        self.db_host = os.getenv("DB_HOST")
        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_port = os.getenv("DB_PORT")
        self.temperature_routing = float(os.getenv("OLLAMA_TEMPERATURE_ROUTING", 0.1))
        self.temperature_sql = float(os.getenv("OLLAMA_TEMPERATURE_SQL", 0.2))

def get_settings() -> Settings:
    return Settings()

