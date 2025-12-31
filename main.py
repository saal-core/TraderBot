"""
TraderBot Streamlit UI
Chat interface for natural language database queries with streaming support.
Fixed markdown rendering for streaming responses.
"""
import streamlit as st
import requests
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configure the page
st.set_page_config(
    page_title="TraderBot - Financial Assistant",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #1e3a5f;
    }
    .assistant-message {
        background-color: #2d3748;
    }
    .status-indicator {
        color: #718096;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# API endpoint configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False





def initialize_api():
    """Initialize the API connection."""
    try:
        response = requests.post(f"{API_BASE_URL}/initialize", timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("success", False), data.get("message", "Unknown error")
        return False, f"HTTP {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"


def convert_messages_to_chat_history():
    """Convert session messages to API chat history format."""
    chat_history = []
    for msg in st.session_state.messages[-10:]:  # Last 10 messages
        chat_history.append({
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg.get("timestamp"),
            "sql_query": msg.get("sql_query"),
            "query_type": msg.get("query_type")
        })
    return chat_history


def process_streaming_query(question: str):
    """Process a query with streaming response with fixed markdown handling."""
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json"
    }
    
    chat_history = convert_messages_to_chat_history()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/query/stream",
            json={"query": question, "chat_history": chat_history},
            headers=headers,
            stream=True,
            timeout=120
        )
        
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        final_data = {"final_answer": "", "sql_query": None, "results": None, "query_type": None}
        

        full_response = ""
        
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            
            try:
                evt = json.loads(raw[5:].strip())
                event_type = evt.get("type")
                
                if event_type == "status":
                    data = evt.get("data", {})
                    status_placeholder.text(f"{data.get('message', '')}")
                    
                elif event_type == "content":
                    content = evt.get("content", "")
                    if content:
                        full_response += content
                        # Get safe-to-display text (with complete markdown)
                        # Display as plain text
                        response_placeholder.text(full_response + "▌")
                    status_placeholder.empty()
                    
                elif event_type == "assistant_message_complete":
                    final_data.update(evt.get("data", {}))
                    status_placeholder.empty()
                    # Final render - show complete response as plain text
                    response_placeholder.text(full_response)
                    
                elif event_type == "error":
                    data = evt.get("data", {})
                    error_msg = data.get("error", "Unknown error")
                    status_placeholder.empty()
                    st.error(f"❌ {error_msg}")
                    return None
                    
                elif event_type == "stream_end":
                    # Final render without cursor
                    response_placeholder.text(full_response)
                    break
                    
            except json.JSONDecodeError:
                continue
        
        # Ensure final display is clean
        response_placeholder.text(full_response)
        
        # Update final_data with the full streamed response
        if full_response:
            final_data["final_answer"] = full_response
        
        return final_data
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API: {str(e)}")
        return None


def process_non_streaming_query(question: str):
    """Process a query without streaming (fallback)."""
    headers = {"Content-Type": "application/json"}
    chat_history = convert_messages_to_chat_history()
    
    try:
        # First classify the query
        classify_response = requests.post(
            f"{API_BASE_URL}/query/classify",
            json={"query": question},
            headers=headers,
            timeout=30
        )
        
        if classify_response.status_code != 200:
            st.error(f"Classification error: {classify_response.text}")
            return None
        
        query_type = classify_response.json().get("query_type", "database")
        
        # Route to appropriate endpoint
        endpoint_map = {
            "database": "/query/database",
            "greeting": "/query/greeting",
            "internet_data": "/query/internet",
            "comparison": "/query/comparison"
        }
        
        endpoint = endpoint_map.get(query_type, "/query/database")
        
        with st.spinner(f"Processing {query_type} query..."):
            response = requests.post(
                f"{API_BASE_URL}{endpoint}",
                json={"query": question, "chat_history": chat_history},
                headers=headers,
                timeout=120
            )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "final_answer": data.get("content", "No answer available"),
                "sql_query": data.get("sql_query"),
                "results": data.get("results"),
                "query_type": data.get("query_type")
            }
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API: {str(e)}")
        return None


# Title and description
st.title("📊 TraderBot - Financial Assistant")
st.markdown("""
    Ask questions about your portfolio, compare with market data, or get real-time financial information.
""")

# Sidebar for settings and info
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Initialize button
    if st.button("🔄 Initialize API", type="primary"):
        with st.spinner("Initializing..."):
            success, message = initialize_api()
            if success:
                st.session_state.initialized = True
                st.success("✅ " + message)
            else:
                st.error("❌ " + message)
    
    # Status indicator
    if st.session_state.initialized:
        st.success("🟢 API Connected")
    else:
        st.warning("🟡 API Not Initialized")
    
    # Streaming toggle
    streaming_enabled = st.toggle("Enable Streaming", value=ENABLE_STREAMING)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Stats display
    if st.session_state.initialized:
        try:
            stats_response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                st.subheader("📈 Query Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Database", stats.get("database", 0))
                    st.metric("Internet", stats.get("internet_data", 0))
                with col2:
                    st.metric("Greeting", stats.get("greeting", 0))
                    st.metric("Comparison", stats.get("comparison", 0))
                st.metric("Total", stats.get("total", 0))
        except:
            pass

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            st.text(message["content"])
        
        # Show SQL query if available
        if message.get("sql_query"):
            with st.expander("🔍 SQL Query"):
                st.code(message["sql_query"], language="sql")
        
        # Show results table if available
        if message.get("results"):
            with st.expander("📋 Results Table"):
                st.dataframe(message["results"])

# Auto-initialize on first load
if not st.session_state.initialized:
    with st.spinner("Connecting to API..."):
        success, message = initialize_api()
        if success:
            st.session_state.initialized = True
            st.toast("✅ API Connected!")
        else:
            st.warning(f"⚠️ Could not auto-initialize: {message}. Click 'Initialize API' in the sidebar.")

# Chat input
if question := st.chat_input("Ask a question about your portfolio or financial data..."):
    if not st.session_state.initialized:
        st.warning("Please initialize the API first using the button in the sidebar.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Process query and display assistant response
        with st.chat_message("assistant"):
            if streaming_enabled:
                result = process_streaming_query(question)
            else:
                result = process_non_streaming_query(question)
            
            if result:
                # Store the response with metadata
                assistant_message = {
                    "role": "assistant",
                    "content": result.get("final_answer", ""),
                    "sql_query": result.get("sql_query"),
                    "results": result.get("results"),
                    "query_type": result.get("query_type")
                }
                st.session_state.messages.append(assistant_message)
                
                # Show SQL query expander if available
                if result.get("sql_query"):
                    with st.expander("🔍 SQL Query"):
                        st.code(result["sql_query"], language="sql")
                
                # Show results table if available
                if result.get("results"):
                    with st.expander("📋 Results Table"):
                        st.dataframe(result["results"])

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>TraderBot - Financial Assistant v1.0.1</p>
    </div>
""", unsafe_allow_html=True)
