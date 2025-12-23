"""Dependency injection container for managing service dependencies.

Provides a central registry for creating and accessing service instances,
ensuring proper initialization order and singleton management.
"""
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache

from config.settings import get_settings
from domain.interfaces import LLMProvider, CacheProvider
from infrastructure.llm import create_llm_provider
from infrastructure.cache import MemoryCache


@dataclass
class Container:
    """
    Dependency injection container.
    
    Manages the lifecycle of all service dependencies, providing:
    - Lazy initialization of services
    - Singleton instances
    - Configurable implementations
    
    Usage:
        container = get_container()
        llm = container.llm_provider
        dispatcher = container.query_dispatcher
    """
    
    # Core providers
    llm_provider: LLMProvider = None
    cache_provider: CacheProvider = None
    
    # Lazy-loaded services
    _query_classifier: "QueryClassifier" = field(default=None, repr=False)
    _query_dispatcher: "QueryDispatcher" = field(default=None, repr=False)
    _vanna_service: "VannaService" = field(default=None, repr=False)
    _perplexity_service: "PerplexityService" = field(default=None, repr=False)
    _translation_service: "TranslationService" = field(default=None, repr=False)
    
    @property
    def query_classifier(self):
        """Get or create query classifier."""
        if self._query_classifier is None:
            from application.dispatcher import QueryClassifier
            self._query_classifier = QueryClassifier(
                llm=self.llm_provider,
                cache=self.cache_provider
            )
        return self._query_classifier
    
    @property
    def query_dispatcher(self):
        """Get or create query dispatcher with registered handlers."""
        if self._query_dispatcher is None:
            from application.dispatcher import QueryDispatcher
            from application.handlers import (
                DatabaseQueryHandler,
                InternetQueryHandler,
                GreetingHandler,
            )
            
            # Create dispatcher
            self._query_dispatcher = QueryDispatcher(
                classifier=self.query_classifier
            )
            
            # Register handlers
            self._query_dispatcher.register(
                DatabaseQueryHandler(
                    llm=self.llm_provider,
                    repository=self.vanna_service
                )
            )
            
            self._query_dispatcher.register(
                InternetQueryHandler(
                    llm=self.llm_provider,
                    web_search=self.perplexity_service
                )
            )
            
            self._query_dispatcher.register(
                GreetingHandler(llm=self.llm_provider)
            )
        
        return self._query_dispatcher
    
    @property
    def vanna_service(self):
        """Get or create Vanna service."""
        if self._vanna_service is None:
            from services.vanna_service import VannaService
            self._vanna_service = VannaService()
        return self._vanna_service
    
    @property
    def perplexity_service(self):
        """Get or create Perplexity service."""
        if self._perplexity_service is None:
            from services.perplexity_service import PerplexityService
            self._perplexity_service = PerplexityService()
        return self._perplexity_service
    
    @property
    def translation_service(self):
        """Get or create translation service."""
        if self._translation_service is None:
            from services.translation_service import TranslationService
            self._translation_service = TranslationService()
        return self._translation_service


# Singleton container instance
_container: Optional[Container] = None


@lru_cache()
def get_container() -> Container:
    """
    Get or create the dependency injection container.
    
    Returns:
        Container: Singleton container instance
    """
    global _container
    
    if _container is not None:
        return _container
    
    settings = get_settings()
    
    # Create core providers
    llm_provider = create_llm_provider(
        provider="ollama",
        model=settings.ollama_model,
        base_url=settings.ollama_api_url
    )
    
    cache_provider = MemoryCache(
        max_size=1000,
        default_ttl=3600
    )
    
    _container = Container(
        llm_provider=llm_provider,
        cache_provider=cache_provider
    )
    
    return _container


def reset_container() -> None:
    """Reset the container (useful for testing)."""
    global _container
    _container = None
    get_container.cache_clear()
