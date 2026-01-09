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


def get_perplexity_config() -> Dict[str, Any]:
    """
    Get Perplexity API configuration as a dictionary

    Returns:
        Dictionary containing Perplexity configuration
    """
    return {
        "api_key": os.getenv("PERPLEXITY_API_KEY"),
        "api_url": os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/chat/completions"),
    }


def get_fmp_config() -> Dict[str, Any]:
    """
    Get FMP (Financial Modeling Prep) API configuration as a dictionary

    Returns:
        Dictionary containing FMP configuration
    """
    return {
        "api_key": os.getenv("FMP_API_KEY"),
        "base_url": os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/api/v3"),
    }


def get_qwen_config() -> Dict[str, Any]:
    """
    Get QWEN configuration as a dictionary (H100 via OpenAI-compatible API)
    
    Uses Qwen team recommended settings:
    - temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05

    Returns:
        Dictionary containing QWEN configuration
    """
    return {
        "base_url": os.getenv("QWEN_BASE_URL", "http://192.168.71.72:8080/v1"),
        "api_key": os.getenv("QWEN_API_KEY", "123"),
        "model_name": os.getenv("QWEN_MODEL", "qwen3-30b-3b"),
        # Qwen team recommended settings
        "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.7")),
        "top_p": float(os.getenv("QWEN_TOP_P", "0.8")),
        # Extra parameters for OpenAI-compatible servers (vLLM, TGI, etc.)
        "extra_body": {
            "top_k": int(os.getenv("QWEN_TOP_K", "20")),
            "repetition_penalty": float(os.getenv("QWEN_REPETITION_PENALTY", "1.05")),
        }
    }


def get_huggingface_config() -> Dict[str, Any]:
    """
    Get HuggingFace configuration as a dictionary (OpenAI-compatible API)
    
    Uses standard ChatOpenAI with HuggingFace's v1 router.
    
    Returns:
        Dictionary containing HuggingFace configuration
    """
    return {
        "api_key": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        "model_name": os.getenv("HUGGINGFACE_MODEL"),
        "base_url": "https://router.huggingface.co/v1/", #os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1"),
        "temperature": float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("HUGGINGFACE_MAX_TOKENS", "2048")),
        "type": os.getenv("HUGGINGFACE_TYPE", "api"), # api or local
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

