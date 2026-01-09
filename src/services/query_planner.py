"""
Query Planner Service
Generates execution plans for complex user queries.
"""
import json
import time
from typing import Dict, List, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config.llm_provider import get_llm, get_active_provider, get_provider_config
from src.config.prompts import QUERY_PLANNER_PROMPT
from src.services.chat_memory import ChatMemory

class QueryPlanner:
    """
    Generates multi-step execution plans for user queries.
    """

    def __init__(self, model_name: str = None, memory_max_pairs: int = 5):
        """
        Initialize the Query Planner.

        Args:
            model_name: Model name (defaults to provider config)
            memory_max_pairs: Max Q&A pairs to include in context
        """
        # Get LLM from provider configuration
        provider = get_active_provider()
        config = get_provider_config()
        
        self.model_name = model_name or config["model_name"]
        self.llm = get_llm(temperature=0.1, max_tokens=1000)
        
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)
        
        self.prompt = PromptTemplate(
            input_variables=["query", "conversation_history"],
            template=QUERY_PLANNER_PROMPT
        )
        
        # We want strict JSON output
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser
        
        print(f"✅ Query Planner initialized with {provider.upper()}: {self.model_name}")

    def _format_conversation_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format conversation history for the planner prompt."""
        if not chat_history:
            return "No previous conversation."
        
        recent_messages = self.chat_memory.get_context_messages(chat_history)
        
        if not recent_messages:
            return "No previous conversation."
        
        formatted_lines = []
        for msg in recent_messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            
            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."
            
            formatted_lines.append(f"{role}: {content}")
        
        return "\n".join(formatted_lines)

    def generate_plan(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generate an execution plan for the query.

        Args:
            query: User's query string
            chat_history: Optional list of previous chat messages

        Returns:
            Dict containing the 'query' and 'plan' (list of steps)
        """
        start_time = time.time()
        print(f"🧠 Query Planner analyzing: {query}")
        
        # Format conversation history for context
        conversation_text = self._format_conversation_history(chat_history or [])
        
        if chat_history:
            print(f"   → With {len(chat_history)} history messages")
        
        try:
            result = self.chain.invoke({
                "query": query,
                "conversation_history": conversation_text
            })
            elapsed = (time.time() - start_time) * 1000
            print(f"   → Plan generated ({elapsed:.1f}ms): {len(result.get('plan', []))} steps")
            return result
        except Exception as e:
            print(f"❌ Error generating plan: {e}")
            # Fallback for error cases - treat as simple database query
            return {
                "query": query,
                "plan": [
                    {
                        "step": 1,
                        "action": "fetch_database",
                        "description": "Execute as direct database query (Fallback)",
                        "query_hint": query
                    }
                ]
            }
