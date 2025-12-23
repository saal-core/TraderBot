"""Main Streamlit application for AI Financial Assistant.

Uses the new Clean Architecture with dependency injection and handler pattern.
"""
import logging
import streamlit as st

from config.settings import get_settings, validate_settings
from config.container import get_container
from config.prompts import CUSTOM_ERROR_MESSAGE
from domain.entities import QueryContext
from helpers import detect_language, replace_wallet_with_portfolio


# Initialize settings
try:
    validate_settings()
    settings = get_settings()
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=settings.log_level)

# Get dependency container
container = get_container()

# Streamlit UI Configuration
st.set_page_config(page_title="AI Financial Assistant", layout="centered")
st.title("📈 AI Financial Assistant")
st.caption("Ask about your portfolio data or general financial topics.")

# Sidebar: Connection Status
st.sidebar.header("System Status")

# Test Ollama connection
if container.llm_provider.test_connection():
    st.sidebar.success(f"✓ Connected to Ollama ({settings.ollama_model})")
else:
    st.sidebar.error("✗ Failed to connect to Ollama")
    st.stop()

# Connect to Vanna database
if container.vanna_service.connect():
    st.sidebar.success("✓ Connected to Vanna database")
else:
    st.sidebar.warning("⚠ Database connection failed; portfolio queries disabled")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

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

        # Preprocess query
        processing_question = question_input
        if is_arabic:
            processing_question = container.translation_service.translate(question_input, "English")
            st.info(f"Translated for processing: '{processing_question}'")
        
        # Replace 'wallet' with 'portfolio'
        processing_question = replace_wallet_with_portfolio(processing_question)

        # Create query context
        context = QueryContext(
            query=processing_question,
            session_id=st.session_state.session_id,
            chat_history=st.session_state.messages,
            language=language
        )

        # Dispatch query to appropriate handler
        try:
            result = container.query_dispatcher.dispatch(context)
            response_content = result.content
            
            # Show SQL if available
            if result.sql_query:
                st.code(result.sql_query, language="sql")
            
            # Show data if available
            if result.has_data:
                st.dataframe(result.data_frame)
                
        except Exception as e:
            logging.error(f"Query processing error: {e}")
            response_content = CUSTOM_ERROR_MESSAGE

        # Translate response back to Arabic if needed
        if is_arabic and response_content:
            response_content = container.translation_service.translate(response_content, "Arabic")

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    with st.chat_message("assistant"):
        st.markdown(response_content)
