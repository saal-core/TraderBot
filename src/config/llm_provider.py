"""
LLM Provider Configuration - Simple switch between Ollama (local) and Qwen H100 (remote)

Both providers use OpenAI-compatible API for consistent behavior (especially streaming).

Usage:
    Set LLM_PROVIDER=ollama or LLM_PROVIDER=qwen in your .env file
    
    from src.config.llm_provider import get_llm, get_streaming_llm
    
    llm = get_llm()  # Get configured LLM
    streaming_llm = get_streaming_llm()  # Get streaming-enabled LLM
"""

import os
import logging
from typing import Literal, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from src.config.settings import get_ollama_config, get_qwen_config

from dotenv import load_dotenv
load_dotenv()

# Set up logging for query classification
logger = logging.getLogger("llm_provider")

# Valid provider choices
LLMProvider = Literal["ollama", "qwen"]


def get_active_provider() -> LLMProvider:
    """
    Get the active LLM provider from environment.
    
    Set LLM_PROVIDER=ollama or LLM_PROVIDER=qwen in your .env file.
    Defaults to 'qwen' if not specified.
    
    Returns:
        'ollama' or 'qwen'
    """
    provider = os.getenv("LLM_PROVIDER", "qwen").lower().strip()
    
    if provider not in ("ollama", "qwen"):
        print(f"⚠️  Invalid LLM_PROVIDER '{provider}', defaulting to 'qwen'")
        provider = "qwen"
    
    return provider


def get_provider_config() -> Dict[str, Any]:
    """
    Get the configuration for the active provider.
    
    Returns:
        Dictionary with provider configuration
    """
    provider = get_active_provider()
    
    if provider == "ollama":
        config = get_ollama_config()
        base_url = config["base_url"]
        # Ensure /v1 suffix for OpenAI compatibility
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        return {
            "provider": "ollama",
            "model_name": config["model_name"],
            "base_url": base_url,
            "api_key": "ollama",  # Ollama doesn't need a real key
            "temperature": float(config.get("temperature_sql", 0.2)),
            "config": config
        }
    else:  # qwen
        config = get_qwen_config()
        return {
            "provider": "qwen",
            "model_name": config["model_name"],
            "base_url": config["base_url"],
            "api_key": config["api_key"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "extra_body": config.get("extra_body", {}),
            "config": config
        }


def get_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
) -> ChatOpenAI:
    """
    Get a configured LLM instance based on the active provider.
    
    Both Ollama and Qwen use ChatOpenAI (OpenAI-compatible API) for consistent
    behavior, especially for streaming support.
    
    Args:
        temperature: Override temperature (uses provider default if not specified)
        max_tokens: Max tokens to generate
        streaming: Enable streaming mode
        
    Returns:
        ChatOpenAI instance configured for the active provider
    """
    provider = get_active_provider()
    config = get_provider_config()
    
    # Common kwargs for both providers
    llm_kwargs = {
        "model": config["model_name"],
        "base_url": config["base_url"],
        "api_key": config["api_key"],
        "temperature": temperature if temperature is not None else config["temperature"],
        "streaming": streaming,
    }
    
    # Add max_tokens if specified
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens
    
    # Add extra body params for Qwen
    if provider == "qwen":
        llm_kwargs["top_p"] = config.get("top_p", 0.8)
        llm_kwargs["extra_body"] = config.get("extra_body", {})
    
    llm = ChatOpenAI(**llm_kwargs)
    
    # Log provider info
    emoji = "🦙" if provider == "ollama" else "🚀"
    stream_status = " (streaming)" if streaming else ""
    print(f"{emoji} Using {provider.upper()}: {config['model_name']} @ {config['base_url']}{stream_status}")
    
    return llm


def get_streaming_llm(temperature: Optional[float] = None) -> ChatOpenAI:
    """
    Get a streaming-enabled LLM instance.
    
    Args:
        temperature: Override temperature
        
    Returns:
        ChatOpenAI instance with streaming=True
    """
    return get_llm(temperature=temperature, streaming=True)


def get_sql_llm() -> ChatOpenAI:
    """Get LLM configured for SQL generation (low temperature for precision)."""
    return get_llm(temperature=0.1)


def get_explanation_llm() -> ChatOpenAI:
    """Get LLM configured for explanations (streaming, higher temperature)."""
    return get_streaming_llm(temperature=0.7)


def get_classification_llm() -> ChatOpenAI:
    """Get LLM configured for classification (low temp, short output)."""
    return get_llm(temperature=0.1, max_tokens=50)


def log_query_classification(query: str, classification: str, elapsed_ms: float):
    """Log query and its classification for monitoring."""
    print(f"📋 Query: \"{query[:80]}{'...' if len(query) > 80 else ''}\"")
    print(f"   → Classification: {classification} ({elapsed_ms:.1f}ms)")
    logger.info(f"Query classified: '{query}' -> {classification} ({elapsed_ms:.1f}ms)")


def print_provider_status():
    """Print current LLM provider configuration."""
    provider = get_active_provider()
    config = get_provider_config()
    
    print("\n" + "=" * 50)
    print("🔧 LLM Provider Configuration")
    print("=" * 50)
    print(f"  Provider: {provider.upper()}")
    print(f"  Model: {config['model_name']}")
    print(f"  Base URL: {config['base_url']}")
    print(f"  Temperature: {config['temperature']}")
    
    if provider == "ollama":
        print("  💡 Tip: Set LLM_PROVIDER=qwen to switch to H100")
    else:
        print("  💡 Tip: Set LLM_PROVIDER=ollama to switch to local Ollama")
    
    print("=" * 50 + "\n")


# Print status on import (helpful for debugging)
if __name__ == "__main__":
    print_provider_status()

