"""Ollama LLM adapter implementing the LLMProvider protocol.

Wraps the Ollama API with circuit breaker protection and structured logging.
"""
import logging
import traceback
from typing import List, Dict, Optional
import ollama

from domain.interfaces import LLMProvider
from infrastructure.resilience import CircuitBreaker, CircuitBreakerConfig
from infrastructure.observability import get_logger, get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


class OllamaAdapter:
    """
    LLM adapter for Ollama, implementing the LLMProvider protocol.
    
    Features:
    - Circuit breaker protection for resilience
    - Structured logging
    - Metrics collection
    """
    
    def __init__(
        self,
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """
        Initialize Ollama adapter.
        
        Args:
            model: Ollama model name
            base_url: Ollama API URL
            circuit_breaker: Optional circuit breaker for resilience
        """
        self.model = model
        self.base_url = base_url
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            "ollama",
            CircuitBreakerConfig(failure_threshold=3, timeout_seconds=60)
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Text prompt
            **kwargs: Additional options (temperature, max_tokens)
            
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 10000,
        **kwargs
    ) -> str:
        """
        Generate a response from a conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options
            
        Returns:
            Generated response text
        """
        def _call_ollama() -> str:
            logger.debug(
                "Calling Ollama",
                model=self.model,
                temperature=temperature,
                message_count=len(messages)
            )
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                stream=False
            )
            
            return response['message']['content'].strip()
        
        try:
            with metrics.timer("llm_call", tags={"provider": "ollama"}):
                result = self._circuit_breaker.call(_call_ollama)
            
            metrics.increment("llm_calls", tags={"provider": "ollama", "status": "success"})
            return result
            
        except Exception as e:
            logger.error("Ollama call failed", exception=e)
            metrics.increment("llm_calls", tags={"provider": "ollama", "status": "error"})
            return f"Error: Could not get response from Ollama. {e}"
    
    def test_connection(self) -> bool:
        """Test if Ollama service is available."""
        try:
            ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                stream=False
            )
            return True
        except Exception as e:
            logger.error("Ollama connection test failed", exception=e)
            return False
    
    def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = ollama.list()
            return [model['name'] for model in response.get('models', [])]
        except Exception:
            return []
