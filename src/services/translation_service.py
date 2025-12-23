"""Translation service using Ollama LLM."""
import logging
from typing import List, Dict
from infrastructure.llm import OllamaAdapter
from config.prompts import TRANSLATION_SYSTEM_PROMPT


class TranslationService:
    """Service for translating text between languages using Ollama."""

    def __init__(self):
        """Initialize translation service."""
        self.ollama = OllamaAdapter()

    def translate(self, text: str, target_language: str) -> str:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            target_language: Target language (e.g., "English", "Arabic")

        Returns:
            str: Translated text (or original if translation fails)
        """
        if not text:
            return ""

        system_prompt = TRANSLATION_SYSTEM_PROMPT.format(target_language=target_language)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        try:
            # Use low temperature for deterministic translation
            return self.ollama.generate_response(messages, temperature=0.05)
        except Exception as e:
            logging.error(f"Error during translation: {e}")
            # Fallback to original text if translation fails
            return text
