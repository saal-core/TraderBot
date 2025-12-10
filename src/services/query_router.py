from typing import Dict, List, Tuple, Optional
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_ollama_config
import os
from rapidfuzz import fuzz, process

from dotenv import load_dotenv
load_dotenv()

class QueryRouter:
    """Routes queries to appropriate handlers based on intent classification"""

    def __init__(self, model_name: str = None, ollama_base_url: str = None):
        """
        Initialize the query router

        Args:
            model_name: Name of the Ollama model to use (defaults to config)
            ollama_base_url: Base URL for Ollama API (defaults to config)
        """
        ollama_config = get_ollama_config()

        self.model_name = model_name or ollama_config["model_name"]
        self.base_url = ollama_base_url or ollama_config["base_url"]
        self.temperature = ollama_config["temperature_routing"]

        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature
        )

        # Define routing prompt template
        self.routing_prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are a query classifier. Classify the following user query into ONE of these categories:

Categories:
1. "database" - Questions about data in a local database (e.g., "show me all customers", "what are the total sales")
2. "greeting" - Greetings, chitchat, hello, how are you, goodbye, thank you, etc.
3. "internet_comparison" - Questions that require fetching internet data and comparing with local data (e.g., "compare our prices with competitors", "how do our sales compare to market trends")

Rules:
- Return ONLY the category name: database, greeting, or internet_comparison
- Do not include any explanation or additional text
- If unsure, default to "database" for data-related queries

User Query: {query}

Category:"""
        )

        self.routing_chain = self.routing_prompt | self.llm | StrOutputParser()

    def classify_query(self, query: str) -> str:
        """
        Classify a user query into one of the routing categories

        Args:
            query: User's input query

        Returns:
            Category string: "database", "greeting", or "internet_comparison"
        """
        try:
            response = self.routing_chain.invoke({"query": query})
            category = response.strip().lower()

            # Normalize the response
            if "database" in category:
                return "database"
            elif "greeting" in category or "chitchat" in category:
                return "greeting"
            elif "internet" in category or "comparison" in category:
                return "internet_comparison"
            else:
                # Default to database for data-related queries
                return "database"

        except Exception as e:
            print(f"Error in query classification: {e}")
            return "database"  # Default fallback