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
    /* HTML Response Styling */
    .response-content p {
        margin-bottom: 0.8rem;
        line-height: 1.6;
    }
    .response-content ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .response-content li {
        margin-bottom: 0.4rem;
    }
    .currency {
        color: #48bb78;
        font-weight: 600;
    }
    .percent {
        font-weight: 600;
    }
    .highlight {
        color: #63b3ed;
        font-weight: 600;
    }
    .positive {
        color: #48bb78;
        font-weight: 600;
    }
    .negative {
        color: #fc8181;
        font-weight: 600;
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
        # Handle timestamp serialization
        timestamp = msg.get("timestamp")
        if hasattr(timestamp, 'isoformat'):
            timestamp = timestamp.isoformat()
        elif hasattr(timestamp, 'strftime'):  # Handle pandas Timestamp
            timestamp = str(timestamp)

        chat_history.append({
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": timestamp,
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
        
        # 1. Router Classification placeholder (shown FIRST, before response)
        type_placeholder = st.empty()
        
        # Status and response placeholders
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        # Foldable elements placeholders
        sql_placeholder = st.empty()
        results_placeholder = st.empty()
        
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
                    msg = data.get('message', '')
                    status_placeholder.text(f"{msg}")
                    
                    # Show router classification as soon as we get the query type status
                    if "Query type:" in msg:
                        query_type = msg.replace("Query type:", "").strip()
                        type_placeholder.info(f"🔍 **Router Classification:** {query_type}")
                        final_data["query_type"] = query_type
                    
                elif event_type == "content":
                    content = evt.get("content", "")
                    if content:
                        full_response += content
                        # Display as HTML with cursor
                        response_placeholder.markdown(f'<div class="response-content">{full_response}</div>▌', unsafe_allow_html=True)
                    status_placeholder.empty()
                    
                elif event_type == "assistant_message_complete":
                    data = evt.get("data", {})
                    final_data.update(data)
                    status_placeholder.empty()
                    # Final render - show complete response as HTML
                    response_placeholder.markdown(f'<div class="response-content">{full_response}</div>', unsafe_allow_html=True)
                    
                    # Update router classification if not already shown
                    if final_data.get("query_type") and type_placeholder:
                        type_placeholder.info(f"🔍 **Router Classification:** {final_data['query_type']}")
                    
                    # 2. Show SQL Query in foldable expander
                    if final_data.get("sql_query"):
                        with sql_placeholder.container():
                            with st.expander("📝 Generated SQL Query", expanded=False):
                                st.code(final_data["sql_query"], language="sql")
                    
                    # 3. Show Results Table in foldable expander
                    if final_data.get("results"):
                        with results_placeholder.container():
                            with st.expander("📋 Results Table", expanded=False):
                                st.dataframe(final_data["results"])
                    
                elif event_type == "error":
                    data = evt.get("data", {})
                    error_msg = data.get("error", "Unknown error")
                    status_placeholder.empty()
                    st.error(f"❌ {error_msg}")
                    return None
                    
                elif event_type == "stream_end":
                    # Final render without cursor
                    response_placeholder.markdown(f'<div class="response-content">{full_response}</div>', unsafe_allow_html=True)
                    break
                    
            except json.JSONDecodeError:
                continue
        
        # Ensure final display is clean
        response_placeholder.markdown(f'<div class="response-content">{full_response}</div>', unsafe_allow_html=True)
        
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
            # 1. Show Router Classification FIRST (before response)
            if message.get("query_type"):
                st.info(f"🔍 **Router Classification:** {message['query_type']}")
            
            # Show response content as HTML
            st.markdown(f'<div class="response-content">{message["content"]}</div>', unsafe_allow_html=True)
            
            # 2. Show SQL query in foldable expander
            if message.get("sql_query"):
                with st.expander("📝 Generated SQL Query", expanded=False):
                    st.code(message["sql_query"], language="sql")
            
            # 3. Show results table in foldable expander
            if message.get("results"):
                with st.expander("📋 Results Table", expanded=False):
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
                
                # Elements are already displayed during streaming
                # Only display if non-streaming mode was used
                if not streaming_enabled:
                    # 1. Show Router Classification FIRST
                    if result.get("query_type"):
                        st.info(f"🔍 **Router Classification:** {result['query_type']}")
                    
                    # 2. Show SQL Query in foldable expander
                    if result.get("sql_query"):
                        with st.expander("📝 Generated SQL Query", expanded=False):
                            st.code(result["sql_query"], language="sql")
                    
                    # 3. Show Results Table in foldable expander
                    if result.get("results"):
                        with st.expander("📋 Results Table", expanded=False):
                            st.dataframe(result["results"])

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>TraderBot - Financial Assistant v1.0.1</p>
    </div>
""", unsafe_allow_html=True)
