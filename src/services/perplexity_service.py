"""Perplexity API service for web search and general financial queries."""
import logging
import traceback
from typing import List, Dict
import requests
from config.prompts import PERPLEXITY_SYSTEM_PROMPT
from helpers import clean_text

settings = get_settings()


class PerplexityService:
    """Service for querying Perplexity API with web search capabilities."""

    def __init__(self):
        """Initialize Perplexity service."""
        self.api_key = settings.perplexity_api_key
        self.api_url = settings.perplexity_api_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Send a query to Perplexity API with chat history.

        Args:
            query: User's question
            chat_history: Previous conversation messages

        Returns:
            str: Response with sources
        """
        # 1. Keep only last 15 messages to limit payload size
        recent_history = [
            msg for msg in chat_history
            if msg["role"] in ["user", "assistant"]
        ][-15:]

        # 2. Append new user query
        recent_history.append({"role": "user", "content": query})

        # 3. Enforce strict chronological alternation
        cleaned_dialogue = []
        last_role = "system"  # system role precedes user/assistant
        for msg in recent_history:
            if msg["role"] == last_role:
                # Skip to enforce alternation
                continue
            cleaned_dialogue.append(msg)
            last_role = msg["role"]

        # 4. Construct final messages payload
        perplexity_messages = [
            {"role": "system", "content": PERPLEXITY_SYSTEM_PROMPT}
        ] + cleaned_dialogue

        # 5. Final safety check to remove consecutive same-role messages
        final_messages = [perplexity_messages[0]]  # start with system message
        for msg in perplexity_messages[1:]:
            if msg["role"] != final_messages[-1]["role"]:
                final_messages.append(msg)

        # Debug logging
        logging.debug("\n--- Final Perplexity Messages ---")
        for msg in final_messages:
            logging.debug(f"- Role: {msg['role']}, Content preview: {msg['content'][:60]}")
        logging.debug("--------------------------------\n")

        payload = {
            "model": "sonar",
            "messages": final_messages,
            "temperature": 0.3,
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            resp_json = response.json()

            answer = resp_json['choices'][0]['message']['content']
            answer = clean_text(answer)  # Clean formatting issues

            # Extract and include sources
            source_links = self._extract_sources(resp_json)
            if source_links:
                answer += "\n\n**Sources:**\n" + "\n".join(f"- {link}" for link in source_links)

            return answer

        except requests.HTTPError as e:
            error_detail = e.response.json().get('error', {}).get('message', e.response.text)
            logging.error(f"HTTP error: {e} | Response: {error_detail}")
            return f"Error from Perplexity API: {error_detail}"
        except Exception as ex:
            logging.error(f"General error: {ex}")
            traceback.print_exc()
            return f"Error: {str(ex)}"

    def _extract_sources(self, resp_json: dict) -> List[str]:
        """
        Extract source URLs from Perplexity API response.

        Args:
            resp_json: Response JSON from API

        Returns:
            list: List of source URLs
        """
        source_links = []

        # Check 'sources' field
        if 'sources' in resp_json:
            source_links = [
                s.get('url')
                for s in resp_json['sources']
                if s.get('url')
            ]

        # Check alternative 'text_outputs' field
        elif 'text_outputs' in resp_json and 'url_citations' in resp_json['text_outputs']:
            source_links = [
                c['url']
                for c in resp_json['text_outputs']['url_citations']
            ]

        return source_links
