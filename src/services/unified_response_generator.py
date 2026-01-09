"""
Unified Response Generator Service
Single point of response generation for all query types with streaming support.
"""
from typing import Dict, List, Any, Optional, Generator, Literal
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config.llm_provider import get_streaming_llm
from src.config.prompts import detect_language, ARABIC_FINANCIAL_GLOSSARY
from src.utils.response_cleaner import clean_llm_chunk
from src.config.prompts import UNIFIED_RESPONSE_PROMPT

# ============================================================================
# Singleton Instance
# ============================================================================

_generator_instance: Optional["UnifiedResponseGenerator"] = None


def get_response_generator() -> "UnifiedResponseGenerator":
    """Get the singleton instance of the UnifiedResponseGenerator."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = UnifiedResponseGenerator()
    return _generator_instance


# ============================================================================
# Unified Response Generator Class
# ============================================================================

class UnifiedResponseGenerator:
    """
    Single point of response generation for all query types.
    
    Provides consistent streaming responses with unified formatting
    for database, internet, greeting, and hybrid queries.
    """

    def __init__(self, temperature: float = 0.7):
        """
        Initialize the unified response generator.
        
        Args:
            temperature: LLM temperature for response generation (default: 0.7)
        """
        self.llm = get_streaming_llm(temperature=temperature)
        self.prompt_template = PromptTemplate(
            input_variables=["language", "today_date", "query", "context_type", "data_context", "arabic_glossary"],
            template=UNIFIED_RESPONSE_PROMPT
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        print("✅ UnifiedResponseGenerator initialized")

    def _format_data_context(
        self, 
        context_type: str, 
        data: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format the data context based on query type.
        
        Args:
            context_type: Type of query (database, internet, greeting, hybrid)
            data: Optional data dictionary containing query results
            chat_history: Optional conversation history
            
        Returns:
            Formatted string for the data context
        """
        if context_type == "greeting":
            # For greetings, include brief chat history context
            if chat_history and len(chat_history) > 0:
                recent = chat_history[-3:] if len(chat_history) > 3 else chat_history
                history_text = "\n".join([
                    f"- {msg.get('role', 'user').capitalize()}: {msg.get('content', '')[:100]}..."
                    for msg in recent
                ])
                return f"Recent conversation:\n{history_text}"
            return "This is the start of the conversation."
        
        if data is None:
            return "No data available."
        
        context_parts = []
        
        # SQL Query (for database queries)
        if "sql_query" in data:
            context_parts.append(f"**SQL Query Used:**\n```sql\n{data['sql_query']}\n```")
        
        # Results (for database queries)
        if "results" in data:
            context_parts.append(f"**Query Results:**\n{data['results']}")
        
        # Raw data (for internet queries)
        if "raw_data" in data:
            context_parts.append(f"**Retrieved Data:**\n{data['raw_data']}")
        
        # Context from multiple steps (for hybrid queries)
        if "context" in data:
            context_parts.append(f"**Combined Data:**\n{data['context']}")
        
        return "\n\n".join(context_parts) if context_parts else "No data available."

    def stream_response(
        self,
        query: str,
        context_type: Literal["database", "internet", "greeting", "hybrid"],
        data: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        """
        Stream a unified response for any query type.
        
        Args:
            query: The user's question
            context_type: Type of query (database, internet, greeting, hybrid)
            data: Optional data dictionary containing:
                - For database: {"sql_query": str, "results": str}
                - For internet: {"raw_data": str}
                - For hybrid: {"context": str}
            chat_history: Optional conversation history
            
        Yields:
            String chunks of the response
        """
        try:
            # Detect language from query
            language = detect_language(query)
            
            # Format data context based on query type
            data_context = self._format_data_context(context_type, data, chat_history)
            
            # Build prompt input
            prompt_input = {
                "language": language,
                "today_date": datetime.now().strftime("%Y-%m-%d"),
                "query": query,
                "context_type": context_type,
                "data_context": data_context,
                "arabic_glossary": ARABIC_FINANCIAL_GLOSSARY if language == "Arabic" else ""
            }
            
            # Stream response chunks
            for chunk in self.llm.stream(self.prompt_template.format(**prompt_input)):
                # Handle both string chunks and AIMessageChunk objects
                if hasattr(chunk, 'content'):
                    yield clean_llm_chunk(chunk.content)
                else:
                    yield clean_llm_chunk(str(chunk))
                    
        except Exception as e:
            error_msg = f"<p>Error generating response: {str(e)}</p>"
            print(f"❌ UnifiedResponseGenerator error: {e}")
            yield error_msg

    def generate_response(
        self,
        query: str,
        context_type: Literal["database", "internet", "greeting", "hybrid"],
        data: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a complete (non-streaming) response.
        
        Args:
            query: The user's question
            context_type: Type of query (database, internet, greeting, hybrid)
            data: Optional data dictionary
            chat_history: Optional conversation history
            
        Returns:
            Complete response string
        """
        chunks = list(self.stream_response(query, context_type, data, chat_history))
        return "".join(chunks)
