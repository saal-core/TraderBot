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


def get_openai_config() -> Dict[str, Any]:
    """
    Get OpenAI configuration as a dictionary

    Returns:
        Dictionary containing OpenAI configuration
    """
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature_routing": float(os.getenv("OPENAI_TEMPERATURE_ROUTING", "0.1")),
        "temperature_sql": float(os.getenv("OPENAI_TEMPERATURE_SQL", "0.0")),
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

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = get_vanna_config()
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

def get_vanna() -> MyVanna:
    return MyVanna()

