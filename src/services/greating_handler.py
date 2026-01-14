
from typing import Dict, List, Tuple, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.llm_provider import get_llm, get_active_provider, get_provider_config
from src.services.chat_memory import ChatMemory
from src.config.prompts import detect_language, ARABIC_FINANCIAL_GLOSSARY
import os
import time
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
load_dotenv()

class GreetingHandler:
    """Handles greetings and chitchat with conversation memory"""

    def __init__(self, model_name: str = None, base_url: str = None, memory_max_pairs: int = 5):
        """
        Initialize the greeting handler

        Args:
            model_name: Name of the model to use (defaults to provider config)
            base_url: Base URL for LLM API (defaults to config)
            memory_max_pairs: Maximum number of Q&A pairs to remember (default: 5)
        """
        # Get LLM from provider configuration
        provider = get_active_provider()
        config = get_provider_config()

        self.model_name = model_name or config["model_name"]
        self.chat_memory = ChatMemory(max_pairs=memory_max_pairs)

        self.llm = get_llm(temperature=0.3)
        
        print(f"✅ GreetingHandler initialized with {provider.upper()}: {self.model_name}")

        self.greeting_prompt = PromptTemplate(
            input_variables=["query", "chat_history", "language", "arabic_glossary"],
            template="""You are a friendly and helpful AI financial assistant.

**Response Language:** {language}

Chat History:
{chat_history}

User: {query}

Respond in a friendly and concise manner in {language}. Keep your response brief (1-3 sentences).
If asked what you can do, mention that you can help query portfolios, financial databases, and answer questions about data.

**HTML FORMATTING (CRITICAL - MUST FOLLOW):**
- Generate your response as HTML, NOT markdown
- Use <p> tags for paragraphs
- NEVER use markdown syntax (no **, *, #, -)
- **FOR ARABIC RESPONSES:** Wrap your ENTIRE response in: <div class="rtl-content">...</div>

**CRITICAL FOR ARABIC RESPONSES:**
- You MUST respond ENTIRELY in Arabic when the user's language is Arabic
- DO NOT mix English words in Arabic responses
- Wrap the entire response in: <div class="rtl-content">your content here</div>

{arabic_glossary}

Assistant (HTML only):"""
        )

        self.greeting_chain = self.greeting_prompt | self.llm | StrOutputParser()

    def respond(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Generate a response to a greeting or chitchat with memory context

        Args:
            query: User's greeting or chitchat message
            chat_history: Full chat history (will use last N pairs automatically)

        Returns:
            Friendly response string
        """
        if chat_history is None:
            chat_history = []

        try:
            # Detect language from user query
            language = detect_language(query)
            
            # Use ChatMemory to get relevant context
            context_messages = self.chat_memory.get_context_messages(chat_history)
            history_text = self.chat_memory.get_summary_text(chat_history)

            if not history_text or history_text == "No previous conversation.":
                history_text = "This is the start of the conversation."

            # Get Arabic glossary if responding in Arabic
            arabic_glossary = ARABIC_FINANCIAL_GLOSSARY if language == "Arabic" else ""

            start_time = time.time()
            print(f"⏱️  Starting: Greeting Response Generation (Language: {language})...")

            response = self.greeting_chain.invoke({
                "query": query,
                "chat_history": history_text,
                "language": language,
                "arabic_glossary": arabic_glossary
            })

            elapsed = time.time() - start_time
            print(f"✅ Completed: Greeting Response in {elapsed:.2f}s")

            return response.strip()

        except Exception as e:
            print(f"❌ Error generating greeting response: {e}")
            # Return Arabic fallback if the query was in Arabic
            if detect_language(query) == "Arabic":
                return '<div class="rtl-content"><p>مرحباً! أنا هنا لمساعدتك في استفسارات قاعدة البيانات. كيف يمكنني مساعدتك؟</p></div>'
            return "<p>Hello! I'm here to help you query databases. How can I assist you?</p>"
