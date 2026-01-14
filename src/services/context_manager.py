# src/services/context_manager.py
import time
from typing import List, Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config.llm_provider import get_llm
from src.config.prompts import CONTEXT_RELEVANCE_PROMPT
from src.services.chat_memory import ChatMemory

class ContextAwareManager:
    """
    Manages conversation context and determines if the current query
    is relevant to the previous history or initiates a new topic.
    """

    def __init__(self):
        """Initialize the Context Manager with a fast LLM."""
        # Use a fast/cheap model if possible (low temperature for determinism)
        self.llm = get_llm(temperature=0.0, max_tokens=10)
        self.chat_memory = ChatMemory(max_pairs=3)  # Only need recent history for relevance

        self.prompt = PromptTemplate(
            input_variables=["history", "query"],
            template=CONTEXT_RELEVANCE_PROMPT
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        print("✅ ContextAwareManager initialized.")

    def check_relevance(self, query: str, chat_history: List[Dict[str, str]]) -> bool:
        """
        Check if the current query is relevant to the provided chat history.
        
        Args:
            query: The user's current query.
            chat_history: The conversation history.
            
        Returns:
            True if relevant (keep history), False if new topic (reset history).
        """
        if not chat_history:
            return True # No history to be irrelevant to

        start_time = time.time()
        
        # Format recent history for the prompt
        history_text = self._format_history(chat_history)
        if not history_text:
             return True

        try:
            response = self.chain.invoke({
                "history": history_text,
                "query": query
            })
            
            decision = response.strip().upper()
            elapsed = (time.time() - start_time) * 1000
            
            is_relevant = "RELEVANT" in decision
            
            log_msg = "KEEP Context" if is_relevant else "DROP Context"
            print(f"🧠 Context Check ({elapsed:.0f}ms): {query[:40]}... -> {decision} ({log_msg})")
            
            return is_relevant

        except Exception as e:
            print(f"⚠️ Context check failed: {e}. Defaulting to keeping history.")
            return True

    def _format_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format the last few messages for the prompt."""
        recent = self.chat_memory.get_context_messages(chat_history)
        if not recent:
            return ""
            
        lines = []
        for msg in recent:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            # Truncate content specifically for this check to save tokens
            if len(content) > 100:
                content = content[:100] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
