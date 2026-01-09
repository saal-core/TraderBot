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
import torch
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
from src.config.settings import get_ollama_config, get_qwen_config, get_huggingface_config

from dotenv import load_dotenv
load_dotenv()

# Set up logging for query classification
logger = logging.getLogger("llm_provider")

# Valid provider choices
LLMProvider = Literal["ollama", "qwen", "huggingface"]


def get_active_provider() -> LLMProvider:
    """
    Get the active LLM provider from environment.
    
    Set LLM_PROVIDER=ollama or LLM_PROVIDER=qwen in your .env file.
    Defaults to 'qwen' if not specified.
    
    Returns:
        'ollama' or 'qwen'
    """
    provider = os.getenv("LLM_PROVIDER", "qwen").lower().strip()
    
    if provider not in ("ollama", "qwen", "huggingface"):
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
    elif provider == "huggingface":
        config = get_huggingface_config()
        # Common config for both API and Local
        base_config = {
            "provider": "huggingface",
            "model_name": config["model_name"],
            "temperature": config["temperature"],
            "type": config["type"], # api or local
            "config": config
        }
        # Add API-specific fields only if type is api
        if config["type"] == "api":
            base_config["base_url"] = config["base_url"]
            base_config["api_key"] = config["api_key"]
        return base_config
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


def get_local_huggingface_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> HuggingFacePipeline:
    """
    Get a generic HuggingFace LLM loaded locally.
    
    Uses 4-bit quantization for efficiency.
    
    Args:
        temperature: Override temperature
        max_tokens: Max tokens to generate
        
    Returns:
        HuggingFacePipeline instance
    """
    config = get_huggingface_config()
    model_id = config["model_name"]
    
    print(f"🤗 Loading local model: {model_id} (this may take a while)...")
    
    # Quantization config (4-bit)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Pipeline arguments
    pipeline_kwargs = {
        "max_new_tokens": max_tokens if max_tokens is not None else config["max_tokens"],
        "temperature": temperature if temperature is not None else config["temperature"],
        "do_sample": True if (temperature if temperature is not None else config["temperature"]) > 0 else False,
        "return_full_text": False,
        "top_k": 10,
    }

    # Load model pipeline
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device_map="auto",  # Uses Accelerate to map to GPU
        pipeline_kwargs=pipeline_kwargs,
        model_kwargs={
            "quantization_config": quantization_config,
            "trust_remote_code": True,
        },
    )
    
    print(f"✅ Local model loaded: {model_id}")
    return llm


def get_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
) -> Any:  # Returns ChatOpenAI or HuggingFacePipeline
    """
    Get a configured LLM instance based on the active provider.
    
    All providers (Ollama, Qwen, HuggingFace API) use ChatOpenAI.
    Local HuggingFace uses HuggingFacePipeline.
    
    Args:
        temperature: Override temperature (uses provider default if not specified)
        max_tokens: Max tokens to generate
        streaming: Enable streaming mode
        
    Returns:
        ChatOpenAI or HuggingFacePipeline instance configured for the active provider
    """
    provider = get_active_provider()
    config = get_provider_config()
    
    # Handle Local HuggingFace
    if provider == "huggingface" and config.get("type") == "local":
        return get_local_huggingface_llm(temperature, max_tokens)
    
    # Common kwargs for ChatOpenAI providers
    
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
    emoji = "🦙" if provider == "ollama" else ("🤗" if provider == "huggingface" else "🚀")
    stream_status = " (streaming)" if streaming else ""
    
    base_url_info = config.get('base_url', 'N/A')
    print(f"{emoji} Using {provider.upper()}: {config['model_name']} @ {base_url_info}{stream_status}")
    
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
    
    print("\n" + "=" * 50)
    print("🔧 LLM Provider Configuration")
    print("=" * 50)
    print(f"  Provider: {provider.upper()}")
    
    if provider == "ollama":
        print("  💡 Tip: Set LLM_PROVIDER=qwen or huggingface to switch providers")
    elif provider == "huggingface":
        config = get_huggingface_config()
        print(f"  Model: {config['model_name']}")
        print(f"  Type: {config['type'].upper()}")
        if config['type'] == 'api':
             print(f"  Base URL: {config['base_url']}")
        print(f"  Temperature: {config['temperature']}")
        print("  💡 Tip: Set LLM_PROVIDER=ollama or qwen to switch providers")
    else:
        print("  💡 Tip: Set LLM_PROVIDER=ollama or huggingface to switch providers")
    
    print("=" * 50 + "\n")


# Print status on import (helpful for debugging)
if __name__ == "__main__":
    print_provider_status()

