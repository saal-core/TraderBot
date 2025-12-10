import os
from typing import Optional, Dict, Any
from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
VANNA_PERSIST_DIR = os.getenv("VANNA_PERSIST_DIR", "./vanna_chromadb_store")

def get_vanna_config() -> Dict[str, Any]:
    return {
        "model": OLLAMA_MODEL,
        "ollama_api_url": OLLAMA_API_URL,
        "persist_directory": VANNA_PERSIST_DIR,
    }

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = get_vanna_config()
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

def get_vanna() -> MyVanna:
    return MyVanna()

def get_db_params():
    host = os.getenv("DB_HOST")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port_str = os.getenv("DB_PORT")
    port = int(port_str) if port_str else None
    return host, name, user, password, port
