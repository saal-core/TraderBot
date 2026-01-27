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
    /* Arabic RTL Styling */
    .rtl-content {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Naskh Arabic', 'Segoe UI', Tahoma, sans-serif;
    }
    .rtl-content p, .rtl-content li, .rtl-content span {
        direction: rtl;
        text-align: right;
    }
    .rtl-content ul {
        padding-right: 1.5rem;
        padding-left: 0;
        list-style-position: inside;
    }
    </style>
    """, unsafe_allow_html=True)

# API endpoint configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False






def check_api_health():
    """Check if the API is healthy and initialized."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("initialized", False), data.get("status", "unknown")
        return False, f"HTTP {response.status_code}"
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
            "results": msg.get("results"),  # Include results data for context
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
        
        # Use iter_content for unbuffered streaming
        buffer = ""
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                buffer += chunk
            
            # Process complete SSE events (end with \n\n)
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                
                for raw in event_str.split("\n"):
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
                                # Helper to strip indentation for rendering
                                render_text = re.sub(r'^\s+', '', full_response, flags=re.MULTILINE)
                                # Display as HTML with cursor - use write for better HTML handling
                                html_content = f'<div class="response-content">{render_text}</div>▌'
                                response_placeholder.markdown(html_content, unsafe_allow_html=True)
                            status_placeholder.empty()
                            
                        elif event_type == "assistant_message_complete":
                            data = evt.get("data", {})
                            final_data.update(data)
                            status_placeholder.empty()
                            # Final render - show complete response as HTML
                            render_text = re.sub(r'^\s+', '', full_response, flags=re.MULTILINE)
                            html_content = f'<div class="response-content">{render_text}</div>'
                            response_placeholder.markdown(html_content, unsafe_allow_html=True)
                            
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
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as render_error:
                        # Catch any rendering/processing error and log it
                        print(f"⚠️ Rendering warning: {render_error}")
                        continue
        
        # Ensure final display is clean with error handling
        try:
            if full_response:
                render_text = re.sub(r'^\s+', '', full_response, flags=re.MULTILINE)
                html_content = f'<div class="response-content">{render_text}</div>'
                response_placeholder.markdown(html_content, unsafe_allow_html=True)
        except Exception as final_render_error:
            # Fallback: show plain text if HTML rendering fails
            print(f"❌ Final render error: {final_render_error}")
            try:
                # Try without HTML wrapper
                response_placeholder.markdown(full_response)
            except:
                response_placeholder.text(full_response or "Response could not be displayed")
        
        # Update final_data with the full streamed response
        if full_response:
            final_data["final_answer"] = full_response
        
        return final_data
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the API: {str(e)}")
        return None
    except Exception as general_error:
        st.error(f"An error occurred: {str(general_error)}")
        return None


# process_non_streaming_query removed - all queries now use streaming via /query/stream


# Title and description
st.title("📊 TraderBot - Financial Assistant")
st.markdown("""
    Ask questions about your portfolio, compare with market data, or get real-time financial information.
""")

# Function to update stats display
def update_stats_display(placeholder):
    """Fetch and display query statistics in the given placeholder."""
    if not st.session_state.initialized:
        return
        
    try:
        stats_response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            with placeholder.container():
                st.subheader("📈 Query Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Database", stats.get("database", 0))
                    st.metric("Internet", stats.get("internet_data", 0))
                with col2:
                    st.metric("Greeting", stats.get("greeting", 0))
                    st.metric("Hybrid", stats.get("hybrid", 0))
                st.metric("Total", stats.get("total", 0))
    except Exception:
        pass

# Sidebar for settings and info
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Status indicator - check API health
    if st.session_state.initialized:
        st.success("🟢 API Connected")
    else:
        st.warning("🟡 Connecting to API...")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Stats display placeholder
    stats_placeholder = st.empty()
    update_stats_display(stats_placeholder)

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
            html_content = f'<div class="response-content">{message["content"]}</div>'
            st.markdown(html_content, unsafe_allow_html=True)
            
            # 2. Show SQL query in foldable expander
            if message.get("sql_query"):
                with st.expander("📝 Generated SQL Query", expanded=False):
                    st.code(message["sql_query"], language="sql")
            
            # 3. Show results table in foldable expander
            if message.get("results"):
                with st.expander("📋 Results Table", expanded=False):
                    st.dataframe(message["results"])

# Check API health on first load
if not st.session_state.initialized:
    with st.spinner("Connecting to API..."):
        initialized, status = check_api_health()
        if initialized:
            st.session_state.initialized = True
            st.toast("✅ API Connected!")
        else:
            st.warning(f"⚠️ API not ready: {status}. Please wait for services to start.")

# Chat input
if question := st.chat_input("Ask a question about your portfolio or financial data..."):
    if not st.session_state.initialized:
        st.warning("Please wait for the API to initialize.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Process query and display assistant response
        with st.chat_message("assistant"):
            result = process_streaming_query(question)
            
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
                
                # Trigger a stats update now that the query is finished
                update_stats_display(stats_placeholder)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>TraderBot - Financial Assistant v1.0.1</p>
    </div>
""", unsafe_allow_html=True)
