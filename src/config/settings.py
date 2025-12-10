"""Application settings using Pydantic for environment variable management."""
import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    db_host: str
    db_name: str
    db_user: str
    db_password: str
    db_port: int = 5432

    # Ollama Configuration
    ollama_model: str = "gpt-oss:20b"
    ollama_api_url: str = "http://localhost:11434"

    # Perplexity API
    perplexity_api_key: str
    perplexity_api_url: str = "https://api.perplexity.ai/chat/completions"

    # OpenAI (Optional)
    openai_api_key: Optional[str] = None

    # Vanna Configuration
    vanna_persist_dir: str = "./vanna_chromadb_store"

    # Application Settings
    max_ui_messages_to_display: int = 20
    log_level: str = "INFO"
    cuda_visible_devices: str = "0"

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Singleton settings instance
    """
    return Settings()


def validate_settings() -> None:
    """
    Validate that all required settings are present.

    Raises:
        ValueError: If required settings are missing
    """
    try:
        settings = get_settings()

        # Check required database fields
        required_fields = ['db_host', 'db_name', 'db_user', 'db_password']
        missing = [f for f in required_fields if not getattr(settings, f, None)]

        if missing:
            raise ValueError(f"Missing required database configuration: {', '.join(missing)}")

        # Check API keys
        if not settings.perplexity_api_key:
            raise ValueError("Missing PERPLEXITY_API_KEY")

    except Exception as e:
        raise ValueError(f"Settings validation failed: {e}")
