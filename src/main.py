"""Main Streamlit application for AI Financial Assistant."""
import logging
import streamlit as st
from config.settings import get_settings, validate_settings
from config.prompts import CUSTOM_ERROR_MESSAGE, OTHER_HANDLER_PROMPT
from helpers import (
    detect_language,
    classify_query_type,
    replace_wallet_with_portfolio,
)
from services.ollama_service import OllamaService
from services.translation_service import TranslationService
from services.vanna_service import VannaService
from services.perplexity_service import PerplexityService

# Initialize settings
try:
    validate_settings()
    settings = get_settings()
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=settings.log_level)

# Initialize services
ollama_service = OllamaService()
translation_service = TranslationService()
vanna_service = VannaService()
perplexity_service = PerplexityService()

# Streamlit UI Configuration
st.set_page_config(page_title="AI Financial Assistant", layout="centered")
st.title("📈 AI Financial Assistant")
st.caption("Ask about your portfolio data or general financial topics.")

# Sidebar: Connection Status
st.sidebar.header("System Status")

# Test Ollama connection
if ollama_service.test_connection():
    st.sidebar.success(f"✓ Connected to Ollama ({settings.ollama_model})")
else:
    st.sidebar.error("✗ Failed to connect to Ollama")
    st.stop()

# Connect to Vanna database
if vanna_service.connect():
    st.sidebar.success("✓ Connected to Vanna database")
else:
    st.sidebar.warning("⚠ Database connection failed; portfolio queries disabled")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (last MAX_UI_MESSAGES_TO_DISPLAY messages)
for message in st.session_state.messages[-settings.max_ui_messages_to_display:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question_input = st.chat_input("Ask your question...")

if question_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question_input})
    with st.chat_message("user"):
        st.markdown(question_input)

    with st.spinner("Thinking..."):
        # Detect language
        language = detect_language(question_input)
        is_arabic = (language == "Arabic")

        # Classify query type
        query_type = classify_query_type(question_input)
        st.info(f"Query classified as: **{query_type.upper()}**")

        response_content = ""

        # Handle Portfolio Queries
        if query_type == "portfolio":
            if not vanna_service.connected:
                response_content = "I'm sorry, the database connection is not available."
            else:
                # Translate to English if needed
                processing_question = question_input
                if is_arabic:
                    processing_question = translation_service.translate(question_input, "English")
                    st.info(f"Translated for processing: '{processing_question}'")

                # Replace 'wallet' with 'portfolio'
                processing_question = replace_wallet_with_portfolio(processing_question)
                logging.debug(f"Processing portfolio question: {processing_question}")

                try:
                    # Execute Vanna query workflow
                    sql, result_df, explanation = vanna_service.query(
                        processing_question,
                        st.session_state.messages
                    )

                    if sql:
                        st.code(sql, language="sql")

                    if result_df is not None and not result_df.empty:
                        st.dataframe(result_df)
                        response_content = "### 🧠 Answer (From Database)\n" + explanation
                    else:
                        response_content = CUSTOM_ERROR_MESSAGE

                except Exception as e:
                    logging.error(f"Portfolio query error: {e}")
                    response_content = CUSTOM_ERROR_MESSAGE

        # Handle General Queries (Web Search)
        elif query_type == "general":
            try:
                px_result = perplexity_service.query(question_input, st.session_state.messages)
                response_content = "### 🌐 Answer (Internet)\n" + px_result
            except Exception as e:
                logging.error(f"Perplexity query error: {e}")
                response_content = CUSTOM_ERROR_MESSAGE

        # Handle Other Queries (Conversational)
        elif query_type == "other":
            recent_dialogue = st.session_state.messages[-8:]
            messages_for_other = [
                {"role": "system", "content": OTHER_HANDLER_PROMPT}
            ] + recent_dialogue

            try:
                other_response = ollama_service.generate_response(
                    messages_for_other,
                    temperature=0.7
                )
                response_content = "### 🤖 Answer (Conversational)\n" + other_response
            except Exception as e:
                logging.error(f"Conversational query error: {e}")
                response_content = CUSTOM_ERROR_MESSAGE

        # Translate response back to Arabic if needed
        if is_arabic and response_content:
            response_content = translation_service.translate(response_content, "Arabic")

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    with st.chat_message("assistant"):
        st.markdown(response_content)
