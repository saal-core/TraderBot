"""Ollama LLM service for chat completions."""
import logging
import traceback
from typing import List, Dict
import ollama
from config.settings import get_settings

settings = get_settings()


class OllamaService:
    """Service for interacting with Ollama LLM."""

    def __init__(self, model: str = None):
        """
        Initialize Ollama service.

        Args:
            model: Ollama model name (defaults to settings value)
        """
        self.model = model or settings.ollama_model

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 10000
    ) -> str:
        """
        Generate a chat completion response from Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated response text
        """
        logging.debug("\n--- Prompt to Ollama ---")
        logging.debug(f"Model: {self.model}, Temp: {temperature}, Max Tokens: {max_tokens}")
        for msg in messages:
            logging.debug(f"{msg['role'].upper()}: {msg['content']}")
        logging.debug("------------------------")

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
                stream=False
            )
            raw_output = response['message']['content'].strip()
            logging.debug(raw_output)
            return raw_output

        except Exception as e:
            logging.error(f"Error calling Ollama chat: {e}")
            traceback.print_exc()
            return f"Error: Could not get response from Ollama. {e}"

    def test_connection(self) -> bool:
        """
        Test connection to Ollama service.

        Returns:
            bool: True if connection successful
        """
        try:
            ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                stream=False
            )
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            return False
