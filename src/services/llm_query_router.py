# src/services/llm_query_router.py
"""
LLM-Only Query Router - Uses configurable LLM provider for query classification.
"""
from typing import Dict, List, Optional
import time

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.llm_provider import get_llm, get_active_provider, get_provider_config
from src.services.chat_memory import ChatMemory

from dotenv import load_dotenv
load_dotenv()


# Classification prompt with conversation history support
LLM_ROUTER_PROMPT = """You are an expert query classifier for a financial portfolio application.

**Conversation History:**
{conversation_history}

**Current User Query:** {query}

**Classification Rules:**

1. **database** - Questions about the user's portfolio data (INCLUDING stored benchmark comparisons):
   - Portfolio holdings, positions, stocks, investments
   - Profit/loss, returns, PnL, gains
   - YTD/MTD/QTD performance
   - Portfolio attributes (description, benchmark, cost model, default index)
   - Top/bottom performing stocks in their portfolio
   - **Portfolio vs benchmark/index comparisons** (benchmark data is stored in the database)
   - "Which portfolio outperformed its YTD index return?"
   - "Compare portfolio returns with benchmarks" / "قارن عوائد المحفظة بالمؤشرات القياسية"
   - Any follow-up questions about previous database queries

2. **internet_data** - Questions requiring ONLY real-time external market data:
   - Current stock prices (not asking about portfolio holdings)
   - Market news, trends, live indices
   - Hypothetical investments ("if I had invested...")
   - Crypto, forex, commodity prices
   - Market movers on NASDAQ/NYSE
   - Top gainers/losers in the market TODAY

3. **hybrid** - Questions that need BOTH portfolio data AND LIVE market data:
   - "What are the top performing stocks today and how much do I have of each in my portfolio?"
   - "Show me CURRENT prices of all stocks in my portfolio"
   - "What's the CURRENT market cap of my top holdings?"
   - Queries asking for portfolio data enriched with LIVE market data
   - Must explicitly need REAL-TIME data from external sources

4. **greeting** - Chitchat, greetings, small talk:
   - "Hi", "Hello", "How are you?"
   - "Who are you?", "What can you do?"
   - "Thank you", "Goodbye"

**CRITICAL - Category Selection:**
- Portfolio vs benchmark/index comparisons using STORED data → **database** (NOT hybrid)
- If query needs portfolio data + LIVE/CURRENT market data → **hybrid**
- If query is about portfolio data (including stored benchmarks) → **database**
- If query is ONLY about real-time market data → **internet_data**

Respond with ONLY ONE of: database, internet_data, hybrid, greeting

Category:"""


class LLMQueryRouter:
    """
    LLM-only query router with conversation history support.
    Uses configurable LLM provider (Ollama or Qwen H100).
    """

    def __init__(self, model_name: str = None, memory_max_pairs: int = 5):
        """
        Initialize the LLM router.

        Args:
            model_name: Model name (defaults to provider config)
            memory_max_pairs: Max Q&A pairs to include in context
        """
        # Get LLM from provider configuration
        provider = get_active_provider()
        config = get_provider_config()
        
        self.model_name = model_name or config["model_name"]
        self.llm = get_llm(temperature=0.1, max_tokens=50)
        
        # Chat memory for formatting history
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)
        
        # Create classification chain
        self.prompt = PromptTemplate(
            input_variables=["conversation_history", "query"],
            template=LLM_ROUTER_PROMPT
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        print(f"✅ LLM Query Router initialized with {provider.upper()}: {self.model_name}")

    def _format_conversation_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
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
            if len(content) > 150:
                content = content[:150] + "..."
            
            formatted_lines.append(f"{role}: {content}")
        
        return "\n".join(formatted_lines)

    def classify_query(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Classify query using LLM with conversation context.
        
        Args:
            query: User's current query
            chat_history: Previous conversation history
            
        Returns:
            Category: database, internet_data, comparison, or greeting
        """
        start_time = time.time()
        
        # Format conversation history
        conversation_text = self._format_conversation_history(chat_history or [])
        
        print(f"🤖 LLM Router classifying query...")
        if chat_history:
            print(f"   → With {len(chat_history)} history messages")
        
        try:
            response = self.chain.invoke({
                "conversation_history": conversation_text,
                "query": query
            })
            
            # Parse response
            category = response.strip().lower()
            
            # Extract category from response
            if "hybrid" in category:
                result = "hybrid"
            elif "database" in category:
                result = "database"
            elif "internet" in category:
                result = "internet_data"
            elif "greeting" in category:
                result = "greeting"
            else:
                # Default to database for any portfolio-related keywords
                query_lower = query.lower()
                if any(kw in query_lower for kw in ['portfolio', 'holding', 'position', 'stock', 'profit', 'loss']):
                    result = "database"
                else:
                    result = "database"  # Safe default
            
            elapsed = (time.time() - start_time) * 1000
            
            # Log query and classification
            query_preview = f"{query[:60]}..." if len(query) > 60 else query
            print(f"📋 Query: \"{query_preview}\"")
            print(f"   → Classification: {result} ({elapsed:.1f}ms)")
            
            return result
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            print(f"   → LLM error ({elapsed:.1f}ms): {e}")
            return "database"  # Safe fallback
