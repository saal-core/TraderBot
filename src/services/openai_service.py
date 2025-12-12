"""OpenAI service wrapper for ChatGPT integration."""
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIService:
    """Service for ChatGPT integration."""
    
    def __init__(self, model: str = "gpt-4-turbo", temperature: float = 0.0, api_key: Optional[str] = None):
        """
        Initialize OpenAI service with ChatGPT.
        
        Args:
            model: OpenAI model to use (e.g., "gpt-4-turbo", "gpt-4.1", "gpt-3.5-turbo")
            temperature: Model temperature for response randomness
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.model = model
        self.temperature = temperature
        self.client = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            temperature=temperature
        )
    
    def get_client(self):
        """Get the ChatOpenAI client."""
        return self.client
    
    def set_model(self, model: str):
        """Change the model being used."""
        self.model = model
        self.client = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            temperature=self.temperature
        )
