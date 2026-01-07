# src/services/llm_query_router.py
"""
LLM-Only Query Router - Uses QWEN H100 for all query classification with conversation context.
"""
from typing import Dict, List, Optional
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import get_qwen_config
from src.services.chat_memory import ChatMemory

from dotenv import load_dotenv
load_dotenv()


# Classification prompt with conversation history support
LLM_ROUTER_PROMPT = """You are an expert query classifier for a financial portfolio application.

**Conversation History:**
{conversation_history}

**Current User Query:** {query}

**Classification Rules:**

1. **database** - Questions about the user's personal portfolio data ONLY:
   - Portfolio holdings, positions, stocks, investments
   - Profit/loss, returns, PnL, gains
   - YTD/MTD/QTD performance
   - Portfolio attributes (description, benchmark, cost model)
   - Top/bottom performing stocks in their portfolio
   - Any follow-up questions about previous database queries

2. **internet_data** - Questions requiring ONLY real-time external market data:
   - Current stock prices (not asking about portfolio holdings)
   - Market news, trends, indices
   - Hypothetical investments ("if I had invested...")
   - Crypto, forex, commodity prices
   - Market movers on NASDAQ/NYSE
   - Top gainers/losers in the market

3. **hybrid** - Questions that need BOTH portfolio data AND market data:
   - "What are the top performing stocks today and how much do I have of each in my portfolio?"
   - "Show me current prices of all stocks in my portfolio"
   - "What's the market cap of my top holdings?"
   - Queries asking for portfolio data enriched with live market data
   - Queries asking for market data filtered by portfolio holdings

4. **comparison** - Questions explicitly COMPARING portfolio performance WITH benchmarks:
   - "Compare my portfolio to S&P 500"
   - "How does my portfolio perform against the market?"
   - "Am I beating the market?"
   - "Is my portfolio outperforming NASDAQ?"
   - Must have explicit comparison intent (vs, against, outperform, beat)

5. **greeting** - Chitchat, greetings, small talk:
   - "Hi", "Hello", "How are you?"
   - "Who are you?", "What can you do?"
   - "Thank you", "Goodbye"

**IMPORTANT - Category Selection:**
- If query needs portfolio data + live market data WITHOUT comparison intent → **hybrid**
- If query explicitly compares performance → **comparison**
- If query is ONLY about portfolio → **database**
- If query is ONLY about market data → **internet_data**

Respond with ONLY ONE of: database, internet_data, hybrid, comparison, greeting

Category:"""


class LLMQueryRouter:
    """
    LLM-only query router with conversation history support.
    Uses QWEN H100 for all query classification.
    """

    def __init__(self, model_name: str = None, memory_max_pairs: int = 5):
        """
        Initialize the LLM router.

        Args:
            model_name: Model name (defaults to QWEN config)
            memory_max_pairs: Max Q&A pairs to include in context
        """
        # Initialize QWEN H100
        qwen_config = get_qwen_config()
        self.model_name = model_name or qwen_config["model_name"]
        self.base_url = qwen_config["base_url"]
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=qwen_config["api_key"],
            temperature=0.1,  # Low for classification accuracy
            top_p=qwen_config.get("top_p", 0.8),
            max_tokens=50,  # Small response expected
            max_retries=2,
            extra_body=qwen_config.get("extra_body", {})
        )
        
        # Chat memory for formatting history
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)
        
        # Create classification chain
        self.prompt = PromptTemplate(
            input_variables=["conversation_history", "query"],
            template=LLM_ROUTER_PROMPT
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        
        print(f"✅ LLM Query Router initialized with QWEN H100: {self.model_name}")

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
            elif "comparison" in category:
                result = "comparison"
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
            print(f"   → Classification: {result} ({elapsed:.1f}ms)")
            
            return result
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            print(f"   → LLM error ({elapsed:.1f}ms): {e}")
            return "database"  # Safe fallback
