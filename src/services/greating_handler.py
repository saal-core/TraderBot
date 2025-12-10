
from typing import Dict, List, Tuple, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config
import os
from rapidfuzz import fuzz, process
from .query_router import QueryRouter
from dotenv import load_dotenv
load_dotenv()

class GreetingHandler:
    """Handles greetings and chitchat"""

    def __init__(self, model_name: str = None, ollama_base_url: str = None):
        """
        Initialize the greeting handler

        Args:
            model_name: Name of the Ollama model to use (defaults to config)
            ollama_base_url: Base URL for Ollama API (defaults to config)
        """
        ollama_config = get_ollama_config()

        self.model_name = model_name or ollama_config["model_name"]
        self.base_url = ollama_base_url or ollama_config["base_url"]
        self.temperature = ollama_config["temperature_greeting"]

        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature
        )

        self.greeting_prompt = PromptTemplate(
            input_variables=["query", "chat_history"],
            template="""You are a friendly and helpful AI assistant for a text-to-SQL chatbot.

Chat History:
{chat_history}

User: {query}

Respond in a friendly and concise manner. Keep your response brief (1-3 sentences).
If asked what you can do, mention that you can help query databases and answer questions about data.Assistant:"""
        )

        self.greeting_chain = self.greeting_prompt | self.llm | StrOutputParser()

    def respond(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Generate a response to a greeting or chitchat

        Args:
            query: User's greeting or chitchat message
            chat_history: Optional chat history

        Returns:
            Friendly response string
        """
        if chat_history is None:
            chat_history = []

        try:
            context_size = int(os.getenv("CHAT_HISTORY_CONTEXT_SIZE", 5))
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-context_size:]])
            response = self.greeting_chain.invoke({
                "query": query,
                "chat_history": history_text
            })
            return response.strip()

        except Exception as e:
            print(f"Error generating greeting response: {e}")
            return "Hello! I'm here to help you query databases. How can I assist you?"

