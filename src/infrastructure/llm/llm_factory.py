"""Factory for creating LLM providers based on configuration."""
from typing import Optional

from domain.interfaces import LLMProvider
from .ollama_adapter import OllamaAdapter


def create_llm_provider(
    provider: str = "ollama",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Factory function to create LLM provider based on configuration.
    
    Args:
        provider: Provider type ("ollama" or "openai")
        model: Model name
        base_url: API base URL
        **kwargs: Additional provider-specific options
        
    Returns:
        LLMProvider implementation
        
    Raises:
        ValueError: If unknown provider is specified
    """
    if provider == "ollama":
        return OllamaAdapter(
            model=model or "gpt-oss:20b",
            base_url=base_url or "http://localhost:11434",
            **kwargs
        )
    elif provider == "openai":
        # OpenAI adapter can be added here when needed
        raise NotImplementedError("OpenAI adapter not yet implemented")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
